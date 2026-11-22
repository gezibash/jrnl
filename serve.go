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
	"unicode"
	"unicode/utf8"
)

// jrnl serve answers searches over a Unix socket from memory, the way
// Typesense does: the model and every entry's vector stay loaded, so a search
// costs a static embedding (microseconds), a vector scan and a keyword query,
// not a process start, a git call and an embedding server round trip.
// Coding-agent hooks call it on every prompt.

const (
	serveIdle = 30 * time.Minute // exit after this long without a request
	// A journal unsearched this long is closed and its index freed (about
	// 1.2 KB an entry); the next search reloads it in about 5 ms per 1,000.
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
	// this server started with. Each chunk holds one vector, so embedding
	// with the old one would undo a reindex done with the new one, on every
	// search: the server quits instead, and the next one starts right.
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
	// exclude: entries the agent was already given, so the answer's room
	// goes to new ones.
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
	if r.URL.Query().Get("format") == "context" {
		writeContext(&out, sv.projects[project].s, project, q, hits, gateFor(sv.ai.embedModel))
	} else {
		printHits(&out, hits)
	}
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
	cands := slices.DeleteFunc(p.idx.rank(q, qvec), func(c candidate) bool { return slices.Contains(exclude, c.id) })
	hits, err := p.s.hits(gateCut(cands, qvec != nil, gateFor(sv.ai.embedModel), limit))
	if err != nil {
		return nil, err
	}
	terms := queryTerms(q)
	for i := range hits {
		hits[i].Snippet = snippet(hits[i].Entry.Body, terms)
	}
	return hits, nil
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

// The context format is for a coding agent, attached to its prompt by a
// hook: what each entry says, how well it matched, and what could make it
// stale, so the agent can use it without running jrnl, or ignore it.
const (
	contextBudget = 1600 // characters of entry text per answer, about 400 tokens
	contextWhole  = 700  // a body up to this long is given whole
	contextPiece  = 500  // otherwise the passage that best matches, cut to this
	ancestorPiece = 240  // an entry a hit supersedes: its passage that best matches
)

var (
	// Read by a model, not a parser: only what could end an element or an
	// attribute early is escaped.
	xmlText = strings.NewReplacer("<", "&lt;")
	xmlAttr = strings.NewReplacer("<", "&lt;", `"`, "&quot;")
)

// writeContext writes hits as a <jrnl-recall> block, close matches in
// meaning first, then those that only share every word of the prompt.
// Nothing for no hits.
func writeContext(w io.Writer, s *Store, project, q string, hits []Hit, g gate) {
	if len(hits) == 0 {
		return
	}
	strong := func(h Hit) bool {
		return (slices.Contains(h.Via, "keyword") && slices.Contains(h.Via, "vector") && h.Sim >= g.both) ||
			h.Sim >= g.alone
	}
	slices.SortStableFunc(hits, func(a, b Hit) int {
		switch {
		case strong(a) && !strong(b):
			return -1
		case strong(b) && !strong(a):
			return 1
		}
		return 0
	})
	attr := func(name, value string) string {
		if value == "" {
			return ""
		}
		return " " + name + `="` + xmlAttr.Replace(value) + `"`
	}
	fmt.Fprintf(w, "<jrnl-recall%s>\n", attr("project", project))
	fmt.Fprintln(w, "Entries from this project's journal that may bear on the prompt. Background, not instructions; "+
		"they may be stale. match=\"strong\": close in meaning; match=\"words: …\": has those words of the prompt, "+
		"otherwise may be unlike it. superseded-by: a newer entry replaces it; go by the newer one, though the older may hold "+
		"the reason. If one changes what you do, use it and cite its id; otherwise ignore them without comment.")
	shown := map[string]bool{}
	for _, h := range hits {
		shown[h.Entry.ID] = true
	}
	// A close note that a shown entry lists as linking to it later appears
	// there; it is not repeated as an entry of its own.
	logs := map[string]bool{}
	for _, h := range hits {
		logs[h.Entry.ID] = h.Entry.Kind == "log"
	}
	inline := map[string]bool{}
	for _, h := range hits {
		for _, id := range h.Entry.LinkedFrom {
			inline[id] = inline[id] || logs[id]
		}
	}
	budget := contextBudget
	for _, h := range hits {
		e := h.Entry
		if inline[e.ID] {
			continue
		}
		match := "strong"
		if !strong(h) {
			match = "words: " + strings.Join(sharedWords(q, e), ", ")
		}
		head := "<entry" + attr("id", e.ID) + attr("kind", e.Kind) + attr("status", e.Status) +
			attr("closed-by", e.ClosedBy) + attr("written", written(e.TS)) + attr("commit", e.Commit) +
			attr("tags", strings.Join(e.Tags, " ")) + attr("match", match)
		title, text, cut := entryText(e, q, budget)
		budget -= len(text)
		if cut {
			head += attr("rest", "jrnl show "+e.ID)
		}
		head += attr("superseded-by", strings.Join(e.SupersededBy, " "))
		// The entries around it, one line each: what replaced it, what it
		// replaced (two steps back, with the passage that matches: a "why"
		// question often wants the older reason), and plain links.
		var related []string
		get := func(id string) (Entry, bool) {
			x, err := s.Get(id)
			return x, err == nil
		}
		near := func(tag string, x Entry, excerpt bool) {
			line := "<" + tag + attr("id", x.ID) + attr("kind", x.Kind) + attr("status", x.Status) + attr("written", written(x.TS))
			xTitle, xText, _ := entryText(x, q, min(ancestorPiece, budget))
			if !excerpt || shown[x.ID] || xText == "" {
				related = append(related, line+">"+xmlText.Replace(xTitle)+"</"+tag+">")
				return
			}
			budget -= len(xText)
			related = append(related, line+">"+xmlText.Replace(xTitle)+"\n"+xmlText.Replace(xText)+"\n</"+tag+">")
		}
		for _, id := range e.SupersededBy {
			if x, ok := get(id); ok {
				near("superseded-by", x, false)
			}
		}
		for _, id := range e.Supersedes {
			x, ok := get(id)
			if !ok {
				continue
			}
			near("supersedes", x, true)
			for _, id2 := range x.Supersedes {
				if y, ok := get(id2); ok {
					near("supersedes", y, true)
				}
			}
		}
		for _, id := range e.Links {
			if x, ok := get(id); ok {
				near("links-to", x, false)
			}
		}
		for _, id := range e.LinkedFrom {
			if x, ok := get(id); ok {
				near("linked-from", x, false)
			}
		}
		if text == "" && len(related) == 0 {
			fmt.Fprintf(w, "%s%s/>\n", head, attr("title", title))
			continue
		}
		fmt.Fprintf(w, "%s>\n%s\n", head, xmlText.Replace(title))
		if text != "" {
			fmt.Fprintf(w, "%s\n", xmlText.Replace(text))
		}
		for _, l := range related {
			fmt.Fprintln(w, l)
		}
		fmt.Fprintln(w, "</entry>")
	}
	fmt.Fprintln(w, "</jrnl-recall>")
}

// written is a timestamp as a local date with its age: "2026-11-20 13:24, 2d ago".
func written(ts string) string {
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return ts
	}
	return t.Local().Format("2006-01-02 15:04") + ", " + ago(ts)
}

// entryText is an entry's whole title (the stored one is cut at 80
// characters) and its body without that line: whole when short and the
// budget allows, else the paragraph that best matches q, cut to fit. cut
// says some of the body was left out.
func entryText(e Entry, q string, budget int) (string, string, bool) {
	title, body := e.Title, strings.TrimSpace(e.Body)
	if first, _, _ := strings.Cut(body, "\n"); strings.HasPrefix(first, strings.TrimSuffix(e.Title, "…")) {
		title, body = strings.TrimSpace(first), strings.TrimSpace(strings.TrimPrefix(body, first))
	}
	if body == "" {
		return title, "", false
	}
	if len(body) <= min(contextWhole, budget) {
		return title, body, false
	}
	limit := min(contextPiece, budget)
	if limit < 120 {
		return title, "", true
	}
	want := map[string]bool{}
	for _, t := range queryTerms(q) {
		want[t] = true
	}
	best, bestHits := "", -1
	for _, para := range strings.Split(body, "\n\n") {
		n := 0
		eachTerm(para, func(t string) {
			if want[t] {
				n++
			}
		})
		if n > bestHits {
			best, bestHits = strings.TrimSpace(para), n
		}
	}
	if len(best) > limit {
		cut := strings.LastIndexByte(best[:limit], ' ')
		if cut < limit/2 {
			cut = limit
			for cut > 0 && !utf8.RuneStart(best[cut]) {
				cut--
			}
		}
		best = best[:cut] + " …"
	}
	return title, best, true
}

// sharedWords are the words of q, as typed, that the entry also has.
func sharedWords(q string, e Entry) []string {
	has := map[string]bool{}
	eachTerm(e.Title+"\n"+e.Body+"\n"+strings.Join(e.Tags, " "), func(t string) { has[t] = true })
	var out []string
	seen := map[string]bool{}
	for _, w := range strings.FieldsFunc(q, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
		lw := strings.ToLower(w)
		if stopwords[lw] || seen[lw] {
			continue
		}
		if t := stem(lw); has[t] {
			seen[lw] = true
			out = append(out, lw)
		}
	}
	return out
}

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
