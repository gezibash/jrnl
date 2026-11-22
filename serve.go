package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"

	potion "github.com/trengrj/go-potion"
)

// jrnl serve answers searches over a Unix socket from memory, the way
// Typesense does: the model and every entry's vector stay loaded, so a search
// costs a static embedding (microseconds), a vector scan and a keyword query,
// not a process start, a git call and an embedding server round trip.
// Coding-agent hooks call it on every prompt.

const (
	serveModel = potion.BASE32M
	serveIdle  = 30 * time.Minute // exit after this long without a request
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
	if c, err := net.DialTimeout("unix", sock, 200*time.Millisecond); err == nil {
		c.Close()
		fmt.Println("jrnl serve: already running at", sock)
		return nil
	}
	if *background {
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

	dir := filepath.Dir(sock)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return err
	}
	// Downloads the model on first use (131 MB, once per machine).
	enc, err := potion.New(context.Background(), serveModel)
	if err != nil {
		return fmt.Errorf("serve: load %s: %w", serveModel, err)
	}
	os.Remove(sock)
	l, err := net.Listen("unix", sock)
	if err != nil {
		return err
	}
	defer os.Remove(sock)
	if err := os.Chmod(sock, 0o600); err != nil {
		return err
	}
	sv := newServer(enc)
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

type server struct {
	enc      *potion.Potion
	mu       sync.Mutex // one search at a time: each Store has one connection
	projects map[string]*served
	last     time.Time
	quit     chan struct{}
	quitOnce sync.Once
	mux      *http.ServeMux
}

// served is one project's journal and its in-memory index, rebuilt when
// another process has written to the journal.
type served struct {
	s       *Store
	idx     *memIndex
	version int64
	used    time.Time
}

func newServer(enc *potion.Potion) *server {
	sv := &server{enc: enc, projects: map[string]*served{}, last: time.Now(), quit: make(chan struct{})}
	sv.mux = http.NewServeMux()
	sv.mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "jrnl %s %s\n", version, serveModel)
	})
	sv.mux.HandleFunc("POST /quit", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "bye")
		sv.stop()
	})
	sv.mux.HandleFunc("GET /search", sv.handleSearch)
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
	hits, err := sv.search(project, q, limit)
	if errors.Is(err, os.ErrNotExist) {
		http.Error(w, "no journal for "+project, http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("X-Jrnl-Micros", strconv.FormatInt(time.Since(t0).Microseconds(), 10))
	printHits(w, hits)
}

// search runs one fast search of project. The caller holds sv.mu.
func (sv *server) search(project, q string, limit int) ([]Hit, error) {
	p, err := sv.project(project)
	if err != nil {
		return nil, err
	}
	qvec, err := sv.enc.Encode(q)
	if err != nil {
		return nil, err
	}
	hits, err := p.s.hits(gateCut(p.idx.rank(q, qvec), true, potionGate, limit))
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
			if err := p.idx.addSince(p.s, sv.enc); err != nil {
				return nil, err
			}
			if ok, err = p.idx.current(p.s); err != nil {
				return nil, err
			}
		}
		if !ok {
			idx, err := buildMemIndex(p.s, sv.enc)
			if err != nil {
				return nil, err
			}
			p.idx = idx
		}
		p.version = v
	}
	return p, nil
}
