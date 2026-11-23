package main

// Measures search on real journals, with questions kept outside this
// repository because they describe private projects:
//
//	JRNL_EVAL=~/.journal/eval.json go test -run TestEval -v
//
// The search gate and the README's quality figures are fitted on such a
// file. It is a list of sets, each run against one journal in ~/.journal:
//
//	[{"name": "long questions", "project": "owner-repo", "questions": [
//	  {"q": "why did logins fail after the deploy", "want": ["01M7…", "01M8…"]},
//	  {"q": "kubernetes pod keeps restarting"}]}]
//
// want lists the entries that answer q, picked by reading and never from
// search output; a question without want is unrelated, and nothing should
// be shown for it. at (RFC 3339) is when a real prompt was typed: entries
// written later, often because of it, are left out of its search. Journals keep growing, so a question can gain an
// answer: the output gives each journal's entry count, to quote beside any
// figure. It needs the built-in embedding model, and skips without it.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type evalSet struct {
	Name, Project string
	Questions     []struct {
		Q    string
		Want []string
		At   string // when it was asked: only entries written before count
	}
}

func TestEval(t *testing.T) {
	path := os.Getenv("JRNL_EVAL")
	if path == "" {
		t.Skip("JRNL_EVAL not set")
	}
	if rest, ok := strings.CutPrefix(path, "~/"); ok {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, rest)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var sets []evalSet
	if err := json.Unmarshal(b, &sets); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	ai := localTestAI(t)
	for _, set := range sets {
		// A copy is measured, never the journal itself: opening one with
		// this code upgrades its schema, and an installed jrnl that is
		// older would then refuse it.
		home, _ := os.UserHomeDir()
		src := filepath.Join(home, ".journal", set.Project, "journal.db")
		if h := os.Getenv("JRNL_EVAL_HOME"); h != "" {
			src = filepath.Join(h, set.Project, "journal.db")
		}
		if _, err := os.Stat(src); err != nil {
			t.Skipf("no journal %s here", set.Project)
		}
		tmp := t.TempDir()
		os.MkdirAll(filepath.Join(tmp, set.Project), 0o755)
		ro, err := sql.Open("sqlite", "file:"+src+"?mode=ro")
		if err != nil {
			t.Fatal(err)
		}
		_, err = ro.Exec(`VACUUM INTO ?`, filepath.Join(tmp, set.Project, "journal.db"))
		ro.Close()
		if err != nil {
			t.Fatalf("copy %s: %v", src, err)
		}
		t.Setenv("JRNL_HOME", tmp)
		s, err := Open(set.Project)
		if err != nil {
			t.Fatal(err)
		}
		idx, err := buildMemIndex(s, ai.embedID)
		if err != nil {
			t.Fatal(err)
		}
		g := gateFor(ai.embedModel)
		// first, top5: the wanted entry's place in the ranking, ungated
		// (jrnl search before reranking). shown: through the gate, as
		// jrnl serve and search --fast answer. leaks: negatives shown a hit.
		// vectors: the wanted entry is the closest by meaning alone.
		// entries, wanted: all entries shown, and those of them wanted.
		var pos, neg, first, top5, shown, leaks, vectors, entries, wanted int
		for _, e := range set.Questions {
			v, err := ai.Embed([]Input{{Text: e.Q, Query: true}})
			if err != nil {
				t.Fatal(err)
			}
			var only map[string]bool
			if e.At != "" {
				only = map[string]bool{}
				for i, ts := range idx.ts {
					if ts < e.At {
						only[idx.ids[i]] = true
					}
				}
			}
			cands := idx.rank(e.Q, v[0], only, "")
			kept := gateCut(slices.Clone(cands), true, g, 5)
			entries += len(kept)
			for _, k := range kept {
				if slices.Contains(e.Want, k.id) {
					wanted++
				}
			}
			if len(e.Want) == 0 {
				neg++
				if len(kept) > 0 {
					leaks++
					t.Logf("leak: %q shows %s", e.Q, kept[0].id)
				}
				continue
			}
			pos++
			sims, _ := idx.similarities(v[0])
			for i := range sims {
				if only != nil && !only[idx.ids[i]] {
					sims[i] = 0
				}
			}
			if best := topK(sims, 1); len(best) == 1 && slices.Contains(e.Want, idx.ids[best[0]]) {
				vectors++
			}
			at := slices.IndexFunc(cands, func(c candidate) bool { return slices.Contains(e.Want, c.id) })
			if at == 0 {
				first++
			}
			if at >= 0 && at < 5 {
				top5++
			}
			if slices.ContainsFunc(kept, func(c candidate) bool { return slices.Contains(e.Want, c.id) }) {
				shown++
			} else {
				t.Logf("miss: %q (ranked %d)", e.Q, at+1)
			}
		}
		fmt.Printf("%-15s %s, %d entries: first %d/%d (by vectors alone %d), top five %d/%d | gated: shown %d/%d, unrelated shown %d/%d, %d of %d entries shown wanted\n",
			set.Name, set.Project, len(idx.ids), first, pos, vectors, top5, pos, shown, pos, leaks, neg, wanted, entries)
		s.Close()
	}
}
