package main

// Benchmarks of jrnl serve's hot path on a synthetic 20,000-entry journal.
// go test -run '^$' -bench . -benchmem -count 6 | tee new.txt; benchstat old.txt new.txt

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const benchEntries = 20_000

var (
	benchOnce sync.Once
	benchSrv  *server
	benchErr  error
)

var benchWords = strings.Fields(`login token refresh session cookie redirect cache invalidation queue worker retry
backoff timeout deadline database migration index query planner vacuum replica primary failover lock deadlock
transaction isolation snapshot schema column table view trigger payment invoice refund webhook signature stripe
card expired billing plan trial upgrade downgrade tutorial onboarding player coder studio build room bay floor
stair parking car lot thumbnail render shader texture mesh animation tween button panel menu rail thumbstick
phone tablet desktop layout grid flex font color theme dark light contrast shadow border radius icon label
release binary tarball checksum tag version changelog rollback deploy canary staging production alert pager
incident outage latency throughput memory leak profile benchmark allocation garbage collector goroutine
channel mutex race condition flaky test fixture mock stub integration end to end coverage lint format
the a of to and in for on with after before when while because so but not only one two three few many`)

func benchText(r *rand.Rand, n int) string {
	w := make([]string, n)
	for i := range w {
		w[i] = benchWords[r.IntN(len(benchWords))]
	}
	return strings.Join(w, " ")
}

// benchServer builds the journal once per run, under a temporary home.
func benchServer(b *testing.B) *server {
	b.Helper()
	ai := localTestAI(b)
	benchOnce.Do(func() {
		// Kept between runs: embedding 20,000 entries takes minutes.
		home := filepath.Join(os.TempDir(), "jrnl-bench-"+localID[len(localModel)+7:])
		os.Setenv("JRNL_HOME", home)
		_, statErr := os.Stat(filepath.Join(home, "bench", "journal.db"))
		s, err := Open("bench")
		if err != nil {
			benchErr = err
			return
		}
		defer s.Close()
		if statErr == nil {
			benchSrv = newServer(ai)
			return
		}
		r := rand.New(rand.NewPCG(1, 2))
		ts := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
		for i := range benchEntries {
			title := benchText(r, 8)
			e := Entry{ID: newID(), TS: ts.Add(time.Duration(i) * time.Minute).Format(time.RFC3339), Project: "bench",
				Kind: "note", Title: title, Body: title + "\n\n" + benchText(r, 40+r.IntN(80))}
			if err := s.Insert(e, []Chunk{{Modality: "text"}}); err != nil {
				benchErr = err
				return
			}
		}
		if _, err := process(s, ai, "", true); err != nil {
			benchErr = err
			return
		}
		benchSrv = newServer(ai)
	})
	if benchErr != nil {
		b.Fatal(benchErr)
	}
	return benchSrv
}

var benchQueries = []string{
	"why did the login token refresh fail after the session expired",
	"billing broke because of an expired credit card",
	"flaky test in the integration suite",
	"thumbstick covers the menu on small phones",
	"how did we roll back the release",
}

// The whole request but the socket: data_version check, query embedding,
// keyword and vector ranking, the gate, loading the shown entries.
func BenchmarkServeSearch(b *testing.B) {
	sv := benchServer(b)
	if _, err := sv.search("bench", "warm up", 5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		if _, err := sv.search("bench", benchQueries[i%len(benchQueries)], 5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodeQuery(b *testing.B) {
	sv := benchServer(b)
	for i := 0; b.Loop(); i++ {
		sv.ai.Embed([]Input{{Text: benchQueries[i%len(benchQueries)], Query: true}})
	}
}

func BenchmarkRank(b *testing.B) {
	sv := benchServer(b)
	p, err := sv.project("bench")
	if err != nil {
		b.Fatal(err)
	}
	vecs := make([][]float32, len(benchQueries))
	for i, q := range benchQueries {
		v, err := sv.ai.Embed([]Input{{Text: q, Query: true}})
		if err != nil {
			b.Fatal(err)
		}
		vecs[i] = v[0]
	}
	for i := 0; b.Loop(); i++ {
		p.idx.rank(benchQueries[i%len(benchQueries)], vecs[i%len(vecs)])
	}
}

// After any write by another process the index is rebuilt; this is that cost.
func BenchmarkIndexBuild(b *testing.B) {
	sv := benchServer(b)
	p, err := sv.project("bench")
	if err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := buildMemIndex(p.s, sv.ai.embedID); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoadHits(b *testing.B) {
	sv := benchServer(b)
	p, err := sv.project("bench")
	if err != nil {
		b.Fatal(err)
	}
	cands := make([]candidate, 5)
	for i := range cands {
		cands[i].id = fmt.Sprint(p.idx.ids[i*1000])
	}
	for b.Loop() {
		if _, err := p.s.hits(cands); err != nil {
			b.Fatal(err)
		}
	}
}
