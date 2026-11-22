package main

import (
	"slices"
	"sort"
	"strings"
)

const (
	rrfK = 60 // standard reciprocal-rank-fusion constant
	// Candidates handed to the reranker. 30 + 30 put the right entry first
	// for 21 of 30 reworded test questions. Widening to 50 + 100 dropped that
	// to 15, made Gemma return an empty list at times, and slowed searches
	// from about 1 s to 3-22 s.
	kwCandidates  = 30
	vecCandidates = 30
)

type Hit struct {
	Entry   Entry
	Score   float64
	Sim     float32  // meaning score: cosine of the best vector match, 0 for keyword only
	Via     []string // "keyword", "vector"
	Snippet string
	Match   string // which piece matched by vector: "text" or an attachment
}

// Words too common to mean anything. With OR between terms, "a beeping tone"
// made every entry containing "a" a keyword hit, and that outranked the right
// one.
var stopwords = map[string]bool{
	"a": true, "an": true, "the": true, "and": true, "or": true, "of": true, "to": true,
	"in": true, "on": true, "at": true, "for": true, "with": true, "by": true, "from": true,
	"is": true, "are": true, "was": true, "were": true, "be": true, "it": true, "this": true,
	"that": true, "what": true, "why": true, "how": true, "when": true, "where": true, "who": true,
	"do": true, "does": true, "did": true, "i": true, "we": true, "you": true, "my": true, "our": true,
}

// candidate is a ranked search result before its entry is loaded.
type candidate struct {
	id, match string
	score     float64
	sim       float32
	via       []string
	allTerms  bool // the entry has every query term
}

// vecSource calls yield with each vector the search compares the query to:
// an entry, the piece it came from ("text", or "image:<hash>"), the vector.
type vecSource func(yield func(id, piece string, vec []float32)) error

// chunkVecs reads the vectors stored by embedder for the given modalities.
func (s *Store) chunkVecs(embedder string, modalities []string, f Filter) vecSource {
	return func(yield func(id, piece string, vec []float32)) error {
		w, args, err := f.where()
		if err != nil {
			return err
		}
		marks := strings.TrimSuffix(strings.Repeat("?,", len(modalities)), ",")
		margs := []any{embedder}
		for _, m := range modalities {
			margs = append(margs, m)
		}
		rows, err := s.db.Query(`SELECT c.entry_id, c.modality, c.hash, c.vec FROM chunks c
			JOIN entries e ON e.id = c.entry_id
			WHERE c.vec IS NOT NULL AND c.embedder = ? AND c.modality IN (`+marks+`) AND `+w, append(margs, args...)...)
		if err != nil {
			return err
		}
		defer rows.Close()
		for rows.Next() {
			var id, modality, hash string
			var blob []byte
			if err := rows.Scan(&id, &modality, &hash, &blob); err != nil {
				return err
			}
			piece := modality
			if hash != "" {
				piece = modality + ":" + hash
			}
			yield(id, piece, decodeVec(blob))
		}
		return rows.Err()
	}
}

// nearest ranks entries by their closest vector to qvec alone: search
// --like, which has no words to match.
func nearest(qvec []float32, vecs vecSource) ([]candidate, error) {
	best := map[string]*candidate{}
	err := vecs(func(id, piece string, vec []float32) {
		sim := dot(qvec, vec)
		if c, ok := best[id]; !ok || sim > c.sim {
			best[id] = &candidate{id: id, match: piece, sim: sim, via: []string{"vector"}}
		}
	})
	if err != nil {
		return nil, err
	}
	cands := make([]candidate, 0, len(best))
	for _, c := range best {
		cands = append(cands, *c)
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].sim != cands[j].sim {
			return cands[i].sim > cands[j].sim
		}
		return cands[i].id > cands[j].id
	})
	return cands[:min(len(cands), vecCandidates)], nil
}

// allowed is the IDs of the entries that match f, or nil when f has no
// conditions: what a filtered search may return.
func (s *Store) allowed(f Filter) (map[string]bool, error) {
	w, args, err := f.where()
	if err != nil || w == "1=1" {
		return nil, err
	}
	rows, err := s.db.Query(`SELECT e.id FROM entries e WHERE `+w, args...)
	if err != nil {
		return nil, err
	}
	only := map[string]bool{}
	for _, id := range scanIDs(rows) {
		only[id] = true
	}
	return only, nil
}

// hits loads the entries of ranked candidates, each with the passage of its
// body that has a word of q.
func (s *Store) hits(cands []candidate, q string) ([]Hit, error) {
	terms := queryTerms(q)
	hits := make([]Hit, 0, len(cands))
	for _, c := range cands {
		e, err := s.Get(c.id)
		if err != nil {
			return nil, err
		}
		h := Hit{Entry: e, Score: c.score, Sim: c.sim, Via: c.via, Snippet: snippet(e.Body, terms)}
		// Name the attachment that matched, so the reader can open it.
		if m := c.match; strings.Contains(m, ":") {
			hash := m[strings.IndexByte(m, ':')+1:]
			for _, a := range e.Assets {
				if hash != "" && a.Hash == hash {
					h.Match = a.Modality() + " " + a.Path
				}
			}
		}
		hits = append(hits, h)
	}
	return hits, nil
}

// gate decides which fast-search candidates are shown: those with every
// query term, one of them rare (an identifier such as CarryPose, which
// an embedding model places poorly), those both signals found with a meaning score of at least
// both, or the vector alone with at least alone. Neither signal alone separates: on the test set below,
// unrelated questions scored up to 0.72 with EmbeddingGemma, and some
// answers as low as 0.71.
type gate struct{ both, alone float32 }

// Fitted on 27 reworded questions and 6 unrelated ones over a 100-entry
// journal (2026-11-22): each shows the answer for 23 of 27 and nothing for
// the 6. Few questions, so the margins are thin.
var (
	gemmaGate = gate{0.68, 0.72} // embeddinggemma-2
	// bge-small-en-v1.5 ([CLS] pooling) through jrnl serve's ranking, fitted
	// 2026-11-22 on the same 27 questions plus 18 unrelated ones: 25 of 27
	// shown, 2 of 18 unrelated show a hit (potion-base-32M: 24 and 2).
	localGate = gate{0.64, 0.64}
)

// gateFor is the gate fitted for an embedding model; models never fitted
// get EmbeddingGemma's.
func gateFor(model string) gate {
	if model == localModel {
		return localGate
	}
	return gemmaGate
}

func (g gate) keep(c candidate) bool {
	both := slices.Contains(c.via, "keyword") && slices.Contains(c.via, "vector")
	return c.allTerms || (both && c.sim >= g.both) || c.sim >= g.alone
}

// gateCut keeps the candidates that pass the gate (all of them without a
// query vector), at most limit.
func gateCut(cands []candidate, hasVec bool, g gate, limit int) []candidate {
	if hasVec {
		cands = slices.DeleteFunc(cands, func(c candidate) bool { return !g.keep(c) })
	}
	if len(cands) > limit {
		cands = cands[:limit]
	}
	return cands
}
