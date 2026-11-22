package main

import (
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
	w, args, err := f.where()
	if err != nil {
		return nil, err
	}
	scores := map[string]float64{}
	via := map[string][]string{}
	snippets := map[string]string{}
	match := map[string]string{}
	sims := map[string]float32{}

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
			scores[id] += 1.0 / float64(rrfK+rank)
			via[id] = append(via[id], "keyword")
			snippets[id] = snip
		}
		rows.Close()
	}

	if qvec != nil {
		marks := strings.TrimSuffix(strings.Repeat("?,", len(modalities)), ",")
		margs := []any{embedder}
		for _, m := range modalities {
			margs = append(margs, m)
		}
		rows, err := s.db.Query(`SELECT c.entry_id, c.modality, c.hash, c.vec FROM chunks c
			JOIN entries e ON e.id = c.entry_id
			WHERE c.vec IS NOT NULL AND c.embedder = ? AND c.modality IN (`+marks+`) AND `+w, append(margs, args...)...)
		if err != nil {
			return nil, err
		}
		type best struct {
			id, piece string
			sim       float32
		}
		bestBy := map[string]*best{}
		for rows.Next() {
			var id, modality, hash string
			var blob []byte
			if err := rows.Scan(&id, &modality, &hash, &blob); err != nil {
				rows.Close()
				return nil, err
			}
			sim := dot(qvec, decodeVec(blob))
			if b, ok := bestBy[id]; !ok || sim > b.sim {
				piece := modality
				if hash != "" {
					piece = modality + ":" + hash
				}
				bestBy[id] = &best{id, piece, sim}
			}
		}
		rows.Close()
		ranked := make([]*best, 0, len(bestBy))
		for _, b := range bestBy {
			ranked = append(ranked, b)
		}
		sort.Slice(ranked, func(i, j int) bool { return ranked[i].sim > ranked[j].sim })
		if len(ranked) > vecCandidates {
			ranked = ranked[:vecCandidates]
		}
		for i, b := range ranked {
			scores[b.id] += 1.0 / float64(rrfK+i+1)
			via[b.id] = append(via[b.id], "vector")
			match[b.id] = b.piece
			sims[b.id] = b.sim
		}
	}

	ids := make([]string, 0, len(scores))
	for id := range scores {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		if scores[ids[i]] != scores[ids[j]] {
			return scores[ids[i]] > scores[ids[j]]
		}
		return ids[i] > ids[j] // newer first on ties
	})
	hits := make([]Hit, 0, len(ids))
	for _, id := range ids {
		e, err := s.Get(id)
		if err != nil {
			return nil, err
		}
		h := Hit{Entry: e, Score: scores[id], Sim: sims[id], Via: via[id], Snippet: snippets[id]}
		// Name the attachment that matched, so the reader can open it.
		if m := match[id]; strings.Contains(m, ":") {
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
