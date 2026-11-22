package main

import (
	"slices"
	"sort"
	"strings"
	"unicode"
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

// ftsQuery turns free text into an FTS5 query of quoted terms joined by op
// ("OR" or "AND"), so user punctuation can't break FTS syntax.
func ftsQuery(q, op string) string {
	var terms []string
	for _, t := range strings.FieldsFunc(q, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' }) {
		if !stopwords[strings.ToLower(t)] {
			terms = append(terms, `"`+t+`"`)
		}
	}
	return strings.Join(terms, " "+op+" ")
}

// Search fuses BM25 keyword ranking with vector ranking and returns every
// candidate, best first; the caller reranks and cuts. qvec may be nil, in
// which case only keywords are used.
//
// order "newest" or "oldest" is for "the latest on X" and "when did X first
// happen": keyword candidates are then taken in date order, with all terms
// required if any entry has them all. On the 20,000-entry test, relevance
// order put the newest of 351 "Redis cluster" entries nowhere in the top 5.
func (s *Store) Search(q string, qvec []float32, embedder string, modalities []string, f Filter, order string) ([]Hit, error) {
	cands, err := s.rank(q, qvec, s.chunkVecs(embedder, modalities, f), f, order)
	if err != nil {
		return nil, err
	}
	return s.hits(cands)
}

// candidate is a ranked search result before its entry is loaded.
type candidate struct {
	id, snippet, match string
	score              float64
	sim                float32
	via                []string
	allTerms           bool // the entry has every query term
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

// rank fuses keyword and vector ranks (RRF) without loading any entry.
func (s *Store) rank(q string, qvec []float32, vecs vecSource, f Filter, order string) ([]candidate, error) {
	w, args, err := f.where()
	if err != nil {
		return nil, err
	}
	byID := map[string]*candidate{}
	get := func(id string) *candidate {
		c := byID[id]
		if c == nil {
			c = &candidate{id: id}
			byID[id] = c
		}
		return c
	}

	sortBy := "bm25(entries_fts)"
	switch order {
	case "newest":
		sortBy = "e.ts DESC"
	case "oldest":
		sortBy = "e.ts ASC"
	}
	fq := ftsQuery(q, "OR")
	if order != "" && fq != "" {
		all := ftsQuery(q, "AND")
		var n int
		s.db.QueryRow(`SELECT count(*) FROM entries_fts JOIN entries e ON e.id = entries_fts.entry_id
			WHERE entries_fts MATCH ? AND `+w, append([]any{all}, args...)...).Scan(&n)
		if n > 0 {
			fq = all
		}
	}
	if fq != "" {
		rows, err := s.db.Query(`SELECT e.id, snippet(entries_fts, 2, '[', ']', '…', 16)
			FROM entries_fts JOIN entries e ON e.id = entries_fts.entry_id
			WHERE entries_fts MATCH ? AND `+w+` ORDER BY `+sortBy+` LIMIT ?`,
			append(append([]any{fq}, args...), kwCandidates)...)
		if err != nil {
			return nil, err
		}
		rank := 0
		for rows.Next() {
			var id, snip string
			if err := rows.Scan(&id, &snip); err != nil {
				rows.Close()
				return nil, err
			}
			rank++
			c := get(id)
			c.score += 1.0 / float64(rrfK+rank)
			c.via = append(c.via, "keyword")
			c.snippet = snip
		}
		rows.Close()
	}

	if all := ftsQuery(q, "AND"); all != "" && order == "" {
		rows, err := s.db.Query(`SELECT e.id FROM entries_fts JOIN entries e ON e.id = entries_fts.entry_id
			WHERE entries_fts MATCH ? AND `+w+` LIMIT ?`, append(append([]any{all}, args...), kwCandidates)...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				rows.Close()
				return nil, err
			}
			if c := byID[id]; c != nil {
				c.allTerms = true
			}
		}
		rows.Close()
	}

	if qvec != nil {
		type best struct {
			id, piece string
			sim       float32
		}
		bestBy := map[string]*best{}
		err := vecs(func(id, piece string, vec []float32) {
			sim := dot(qvec, vec)
			if b, ok := bestBy[id]; !ok || sim > b.sim {
				bestBy[id] = &best{id, piece, sim}
			}
		})
		if err != nil {
			return nil, err
		}
		ranked := make([]*best, 0, len(bestBy))
		for _, b := range bestBy {
			ranked = append(ranked, b)
		}
		sort.Slice(ranked, func(i, j int) bool { return ranked[i].sim > ranked[j].sim })
		if len(ranked) > vecCandidates {
			ranked = ranked[:vecCandidates]
		}
		for i, b := range ranked {
			c := get(b.id)
			c.score += 1.0 / float64(rrfK+i+1)
			c.via = append(c.via, "vector")
			c.match = b.piece
			c.sim = b.sim
		}
	}

	cands := make([]candidate, 0, len(byID))
	for _, c := range byID {
		cands = append(cands, *c)
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].score != cands[j].score {
			return cands[i].score > cands[j].score
		}
		return cands[i].id > cands[j].id // newer first on ties
	})
	return cands, nil
}

// hits loads the entries of ranked candidates.
func (s *Store) hits(cands []candidate) ([]Hit, error) {
	hits := make([]Hit, 0, len(cands))
	for _, c := range cands {
		e, err := s.Get(c.id)
		if err != nil {
			return nil, err
		}
		h := Hit{Entry: e, Score: c.score, Sim: c.sim, Via: c.via, Snippet: c.snippet}
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
// query term (an identifier such as CarryPose, which a static embedding
// cannot place), those both signals found with a meaning score of at least
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

// fastSearch ranks without a reranker, keeps what passes the gate and loads
// only the entries it returns.
func fastSearch(s *Store, q string, qvec []float32, vecs vecSource, f Filter, g gate, limit int) ([]Hit, error) {
	cands, err := s.rank(q, qvec, vecs, f, "")
	if err != nil {
		return nil, err
	}
	return s.hits(gateCut(cands, qvec != nil, g, limit))
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
