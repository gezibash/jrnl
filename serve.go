package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// jrnl serve answers searches over a Unix socket from memory, the way
// Typesense does: the model and every entry's vector stay loaded, so a search
// costs one embedding of the question (about 8 ms), a vector scan and a
// keyword lookup, not a process start, a git call and a model load.
// Coding-agent hooks call it on every prompt.

const (
	serveIdle = 30 * time.Minute // exit after this long without a request
	// A journal unsearched this long is closed and its index freed (about
	// 1.2 KB an entry); the next search reloads it in about 8 ms per 1,000.
	projectIdle = 10 * time.Minute
)

func socketPath() string { return filepath.Join(journalHome(), ".run", "jrnl.sock") }

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	background := fs.Bool("background", false, "")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	sock := socketPath()
	dir := filepath.Dir(sock)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return err
	}
	lockPath := filepath.Join(dir, "jrnl.lock")

	if *background {
		// A server of another version (an upgrade, or a switch between
		// installs) is asked to quit, and its lock awaited, so the new
		// binary serves.
		if running, _, ok := runningServer(sock); ok {
			if running == version {
				fmt.Println("jrnl serve: already running at", sock)
				return nil
			}
			if err := quitServer(sock, lockPath); err != nil {
				return err
			}
		}
		exe, err := os.Executable()
		if err != nil {
			return err
		}
		cmd := exec.Command(exe, "serve")
		cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
		if err := cmd.Start(); err != nil {
			return err
		}
		fmt.Println("jrnl serve: started, pid", cmd.Process.Pid)
		return cmd.Process.Release()
	}

	// One server per journal home: the lock's holder alone loads the model
	// and owns the socket, held until the process exits. Without it, two
	// servers started together both ran, one of them unreachable.
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		if errors.Is(err, syscall.EWOULDBLOCK) {
			fmt.Println("jrnl serve: already running at", sock)
			return nil
		}
		return err
	}

	// The embedder that wrote the journals' vectors embeds the queries. The
	// local one is loaded, and downloaded on first use, before the socket
	// opens; searches meanwhile find no server and do without.
	ai, err := NewAI()
	if err != nil {
		return err
	}
	ai.download, ai.serving = true, true
	if ai.local {
		if _, err := local.load(true); err != nil {
			return fmt.Errorf("serve: %w", err)
		}
	}
	os.Remove(sock) // left by a server that was killed
	l, err := net.Listen("unix", sock)
	if err != nil {
		return err
	}
	defer os.Remove(sock)
	if err := os.Chmod(sock, 0o600); err != nil {
		return err
	}
	sv := newServer(ai)
	sv.reconfigured = func() bool {
		now, err := NewAI()
		return err == nil && now.embedID != ai.embedID
	}
	srv := &http.Server{Handler: sv}
	go func() {
		<-sv.quit
		srv.Close()
	}()
	if err := srv.Serve(l); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func socketClient(sock string) *http.Client {
	return &http.Client{Timeout: 2 * time.Second, Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", sock)
		}}}
}

// runningServer is the version and the embedder of the server answering
// at sock, if any.
func runningServer(sock string) (version, embedder string, ok bool) {
	r, err := socketClient(sock).Get("http://jrnl/health")
	if err != nil {
		return "", "", false
	}
	defer r.Body.Close()
	b, _ := io.ReadAll(r.Body)
	f := append(strings.Fields(string(b)), "", "") // "jrnl <version> <embedder>"
	return f[1], f[2], true
}

// quitServer stops the server at sock and waits until it has released
// the lock, so the next one can take it.
func quitServer(sock, lockPath string) error {
	if r, err := socketClient(sock).Post("http://jrnl/quit", "", nil); err == nil {
		r.Body.Close()
	}
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	for range 50 {
		if syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) == nil {
			return syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
		}
		time.Sleep(100 * time.Millisecond)
	}
	return errors.New("serve: the running server did not quit within 5 s")
}

type server struct {
	ai       *AI
	mu       sync.Mutex // one search at a time: each Store has one connection
	projects map[string]*served
	last     time.Time
	quit     chan struct{}
	quitOnce sync.Once
	mux      *http.ServeMux
	// reconfigured says the settings now name another embedder than the one
	// this server started with. It would go on answering with the old one
	// and embedding new entries for it: the server quits instead, and the
	// next one starts with the model the settings name.
	reconfigured func() bool
}

// served is one project's journal and its in-memory index, rebuilt when
// another process has written to the journal.
type served struct {
	s       *Store
	idx     *memIndex
	version int64
	used    time.Time
	// backfilling: the server is embedding this journal's waiting entries.
	backfilling bool
	retryAt     time.Time // after a failed backfill
}

func newServer(ai *AI) *server {
	sv := &server{ai: ai, projects: map[string]*served{}, last: time.Now(), quit: make(chan struct{})}
	sv.mux = http.NewServeMux()
	sv.mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "jrnl %s %s\n", version, ai.embedID)
	})
	sv.mux.HandleFunc("POST /quit", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "bye")
		sv.stop()
	})
	sv.mux.HandleFunc("GET /search", sv.handleSearch)
	sv.mux.HandleFunc("POST /embed", sv.handleEmbed)
	sv.mux.HandleFunc("POST /query", sv.handleQuery)
	go func() {
		for range time.Tick(time.Minute) {
			sv.mu.Lock()
			idle := time.Since(sv.last) > serveIdle
			sv.unload(time.Now().Add(-projectIdle))
			sv.mu.Unlock()
			if idle {
				sv.stop()
				return
			}
		}
	}()
	return sv
}

func (sv *server) ServeHTTP(w http.ResponseWriter, r *http.Request) { sv.mux.ServeHTTP(w, r) }

func (sv *server) stop() { sv.quitOnce.Do(func() { close(sv.quit) }) }

func (sv *server) handleSearch(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	q, project := r.URL.Query().Get("q"), r.URL.Query().Get("project")
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 {
		limit = 5
	}
	if q == "" || project == "" || project != slug(project) {
		http.Error(w, "give q and project", http.StatusBadRequest)
		return
	}
	sv.mu.Lock()
	defer sv.mu.Unlock()
	sv.last = time.Now()
	// exclude: entries to leave out, so the answer's room goes to others.
	var exclude []string
	if x := r.URL.Query().Get("exclude"); x != "" {
		exclude = strings.Split(x, ",")
	}
	// kind: only entries of that kind, from the best 20.
	kind := r.URL.Query().Get("kind")
	n := limit
	if kind != "" {
		n = max(limit, 20)
	}
	hits, err := sv.search(project, q, n, exclude...)
	if kind != "" && err == nil {
		hits = slices.DeleteFunc(hits, func(h Hit) bool { return h.Entry.Kind != kind })
		hits = hits[:min(limit, len(hits))]
	}
	if errors.Is(err, os.ErrNotExist) {
		http.Error(w, "no journal for "+project, http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var out strings.Builder
	printHits(&out, hits)
	w.Header().Set("X-Jrnl-Micros", strconv.FormatInt(time.Since(t0).Microseconds(), 10))
	io.WriteString(w, out.String())
}

// search runs one fast search of project. The caller holds sv.mu.
func (sv *server) search(project, q string, limit int, exclude ...string) ([]Hit, error) {
	p, err := sv.project(project)
	if err != nil {
		return nil, err
	}
	// Without a query vector (an embeddings API down) it answers by keyword.
	var qvec []float32
	if v, err := sv.ai.Embed([]Input{{Text: q, Query: true}}); err == nil {
		qvec = v[0]
	}
	cands := slices.DeleteFunc(p.idx.rank(q, qvec, nil, ""), func(c candidate) bool { return slices.Contains(exclude, c.id) })
	return p.s.hits(gateCut(cands, qvec != nil, gateFor(sv.ai.embedModel), limit), q)
}

// unload closes the journals last searched before cutoff. The caller holds sv.mu.
func (sv *server) unload(cutoff time.Time) {
	for name, p := range sv.projects {
		if p.used.Before(cutoff) {
			p.s.Close()
			delete(sv.projects, name)
		}
	}
}

// project opens a journal that exists (serving never creates one) and
// brings its index up to date. PRAGMA data_version changes when another
// connection has committed, so a check costs microseconds.
func (sv *server) project(name string) (*served, error) {
	p := sv.projects[name]
	if p == nil {
		if _, err := os.Stat(filepath.Join(journalHome(), name, "journal.db")); err != nil {
			return nil, err
		}
		s, err := Open(name)
		if err != nil {
			return nil, err
		}
		// Map the file and keep pages cached: without this every search
		// re-read the keyword index with pread (46 of 48 ms at 20k entries).
		for _, pragma := range []string{`PRAGMA mmap_size = 1073741824`, `PRAGMA cache_size = -131072`} {
			if _, err := s.db.Exec(pragma); err != nil {
				return nil, err
			}
		}
		p = &served{s: s, version: -1}
		sv.projects[name] = p
	}
	p.used = time.Now()
	var v int64
	if err := p.s.db.QueryRow(`PRAGMA data_version`).Scan(&v); err != nil {
		return nil, err
	}
	if v != p.version {
		// New entries are added; anything else (a deleted or re-captioned
		// entry) leaves a fingerprint that differs, and rebuilds.
		ok, err := false, error(nil)
		if p.idx != nil {
			if err := p.idx.addSince(p.s); err != nil {
				return nil, err
			}
			if ok, err = p.idx.current(p.s); err != nil {
				return nil, err
			}
			if ok {
				// Vectors written since: by jrnl add, or by the backfill.
				if err := p.idx.loadVecs(p.s); err != nil {
					return nil, err
				}
			}
		}
		if !ok {
			idx, err := buildMemIndex(p.s, sv.ai.embedID)
			if err != nil {
				return nil, err
			}
			p.idx = idx
		}
		p.version = v
	}
	if p.idx.missing() > 0 && !p.backfilling && time.Now().After(p.retryAt) {
		p.backfilling = true
		go sv.backfill(p)
	}
	return p, nil
}

// backfill embeds and stores the vectors a journal is waiting for, outside
// the search lock: entries saved by an older jrnl, by another embedder, or
// before the model was downloaded. Searches meanwhile use what is there.
// It waits a moment first, so jrnl add stores its own entry's vector.
func (sv *server) backfill(p *served) {
	time.Sleep(backfillDelay)
	stored := 0
	var ps []pending
	var err error
	if sv.reconfigured != nil && sv.reconfigured() {
		sv.stop()
	} else {
		ps, err = p.s.Pending("", sv.ai.embedID, "")
	}
	defer func() {
		sv.mu.Lock()
		p.backfilling = false
		if stored > 0 {
			// data_version moves only for other connections' commits, so
			// the index would not see what this one stored.
			p.version = -1
		}
		if len(ps) == 0 {
			// Missing vectors with nothing to embed (text that can never be
			// embedded): look again later, not on every search.
			p.retryAt = time.Now().Add(10 * time.Minute)
		}
		sv.mu.Unlock()
	}()
	for start := 0; err == nil && start < len(ps); start += 16 {
		batch := ps[start:min(start+16, len(ps))]
		ins := make([]Input, len(batch))
		for i, x := range batch {
			ins[i] = x.in
		}
		vecs, errs := sv.ai.EmbedEach(ins)
		for i, x := range batch {
			switch {
			case errs[i] == nil:
				err = p.s.SetVec(x.chunkID, vecs[i], sv.ai.embedID)
				stored++
			case isPermanent(errs[i]):
				err = p.s.SetError(x.chunkID, errs[i].Error())
			default:
				err = errs[i]
			}
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		sv.mu.Lock()
		p.retryAt = time.Now().Add(time.Minute)
		sv.mu.Unlock()
	}
	if stored > 0 && sv.ai.local {
		// A backfill leaves the model's buffer pool at its largest (1.15 GB
		// resident after re-embedding 197 chunks, against about 500 MB);
		// a server that idles for half an hour should give it back.
		local.reset()
		debug.FreeOSMemory()
	}
}

var backfillDelay = 2 * time.Second

var startOnce sync.Once

// startServer starts jrnl serve in the background, once per process, and
// does not wait for it: it downloads the embedding model if needed and
// embeds the entries that waited for it.
func startServer() {
	startOnce.Do(func() {
		exe, err := os.Executable()
		if err != nil {
			return
		}
		cmd := exec.Command(exe, "serve", "--background")
		cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
		if cmd.Start() == nil {
			go cmd.Wait()
		}
	})
}

// embedRequest is what commands send to POST /embed: texts already
// formatted for the model, and the embedder their vectors are stored under.
type embedRequest struct {
	Embedder string   `json:"embedder"`
	Inputs   []string `json:"inputs"`
}

// handleEmbed embeds texts for a command with the model this server holds.
// It refuses a request for another embedder (a server of another version,
// or another model), and the command embeds by itself.
func (sv *server) handleEmbed(w http.ResponseWriter, r *http.Request) {
	var req embedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Inputs) == 0 {
		http.Error(w, "give embedder and inputs", http.StatusBadRequest)
		return
	}
	if !sv.ai.local || req.Embedder != sv.ai.embedID {
		http.Error(w, "this server embeds with "+sv.ai.embedID, http.StatusConflict)
		return
	}
	sv.mu.Lock()
	sv.last = time.Now()
	sv.mu.Unlock()
	vecs, err := local.embed(req.Inputs, false)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(vecs)
}

// serverEmbed asks a running jrnl serve to embed inputs with the built-in
// model; ok is false when none answers for it.
func serverEmbed(inputs []string) ([][]float32, bool) {
	body, _ := json.Marshal(embedRequest{Embedder: localID, Inputs: inputs})
	c := socketClient(socketPath())
	c.Timeout = 2 * time.Minute // a reindex sends large batches
	r, err := c.Post("http://jrnl/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, false
	}
	defer r.Body.Close()
	var vecs [][]float32
	if r.StatusCode != http.StatusOK || json.NewDecoder(r.Body).Decode(&vecs) != nil || len(vecs) != len(inputs) {
		return nil, false
	}
	return vecs, true
}

// settings names what decides a search's results, keys aside: the
// embedder, the chat model and the reranker.
func (e *AI) settings() string {
	extra, _ := json.Marshal(e.chatExtra)
	return strings.Join([]string{e.embedID, e.chatURL, e.chatModel, string(extra), e.rerankURL, e.rerankModel}, "|")
}

// handleQuery runs a command's whole text search: ranking from memory under
// the lock, then the reranker outside it, over a connection kept warm
// between searches. It refuses a client of another version or settings
// (409), which then searches by itself; when its own settings file has
// moved on, the server also quits, so the next command starts a current one.
func (sv *server) handleQuery(w http.ResponseWriter, r *http.Request) {
	var req query
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Q == "" || req.Project != slug(req.Project) || req.Project == "" {
		http.Error(w, "give project and q", http.StatusBadRequest)
		return
	}
	if req.Version != version || req.Settings != sv.ai.settings() {
		if now, err := NewAI(); err == nil && now.settings() != sv.ai.settings() {
			sv.stop()
		}
		http.Error(w, "this server is "+version+" with other settings", http.StatusConflict)
		return
	}
	sv.mu.Lock()
	sv.last = time.Now()
	p, err := sv.project(req.Project)
	var hits []Hit
	var hasVec bool
	var warns []string
	if err == nil {
		hits, hasVec, warns, err = rankHits(p.s, sv.ai, p.idx, req)
	}
	sv.mu.Unlock()
	if errors.Is(err, os.ErrNotExist) {
		http.Error(w, "no journal for "+req.Project, http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	hits, more := finishHits(sv.ai, req, hits, hasVec)
	json.NewEncoder(w).Encode(queryResult{Hits: hits, Warn: append(warns, more...)})
}

// serverQuery asks a running jrnl serve to run a search; ok is false when
// none answers for this version and these settings.
func serverQuery(ai *AI, req query) (queryResult, bool) {
	req.Version, req.Settings = version, ai.settings()
	body, _ := json.Marshal(req)
	c := socketClient(socketPath())
	c.Timeout = embedDeadline + rerankDeadline + 2*time.Second
	var res queryResult
	r, err := c.Post("http://jrnl/query", "application/json", bytes.NewReader(body))
	if err != nil {
		return res, false
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK || json.NewDecoder(r.Body).Decode(&res) != nil {
		return res, false
	}
	return res, true
}
