package main

import (
	"container/heap"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// memIndex is everything jrnl serve searches, in memory, the way Typesense
// keeps its index: an inverted index of each entry's words for BM25, and the
// vectors stored for its text and attachment captions. SQLite's FTS5 took
// 20 ms to score a common query over 20,000 entries (42 ms with snippets);
// this takes about 1 ms.
type memIndex struct {
	ids      []string
	ts       []string // when each entry happened, for --newest and --oldest
	postings map[string][]posting
	docLen   []int32
	norm     []float32 // BM25's length factor per entry: k1*(1-b+b*len/avg)
	totalLen int64
	maxRowid int64 // the keyword table's last row, to add new entries alone
	textLen  int64 // characters indexed, as SQL length() counts them

	// Vectors made by embedder, several per entry (its text, each caption):
	// row i belongs to entry vecDoc[i] and is vecs[i*dim:(i+1)*dim] times
	// scale[i]. int8 keeps a quarter of float32's memory; cosine moves < 0.01.
	embedder string
	dim      int
	vecs     []int8
	scale    []float32
	vecDoc   []int32
	vecHash  []string // the attachment a caption's vector describes; "" for the entry's text
	has      []bool   // entry has a vector; one without waits for backfill
}

const bm25k1, bm25b = 1.2, 0.75

// rareShare: a term in at most this share of entries is rare enough that
// matching every query term counts on its own.
var rareShare = 0.03

type posting struct {
	doc int32
	tf  int32
}

// buildMemIndex indexes every entry's searchable text, what the keyword
// table holds (title, body, tags, attachment descriptions), and loads the
// vectors embedder stored for them.
func buildMemIndex(s *Store, embedder string) (*memIndex, error) {
	m := &memIndex{postings: map[string][]posting{}, embedder: embedder}
	return m, m.addSince(s)
}

// addSince adds the keyword rows written after the last one indexed.
func (m *memIndex) addSince(s *Store) error {
	rows, err := s.db.Query(`SELECT f.rowid, f.entry_id, f.title, f.body, f.tags, f.captions,
		length(f.entry_id)+length(f.title)+length(f.body)+length(f.tags)+length(f.captions), e.ts
		FROM entries_fts f JOIN entries e ON e.id = f.entry_id WHERE f.rowid > ? ORDER BY f.rowid`, m.maxRowid)
	if err != nil {
		return err
	}
	var ids, texts, tss []string
	for rows.Next() {
		var id, title, body, tags, captions, ts string
		var n int64
		if err := rows.Scan(&m.maxRowid, &id, &title, &body, &tags, &captions, &n, &ts); err != nil {
			rows.Close()
			return err
		}
		ids, tss = append(ids, id), append(tss, ts)
		texts = append(texts, strings.Join([]string{title, body, tags, captions}, "\n"))
		m.textLen += n
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	if len(texts) == 0 {
		return nil
	}
	docs := make([]docTerms, len(texts))
	parallel(len(texts), func(i int) { docs[i] = countTerms(texts[i]) })
	for i, d := range docs {
		doc := int32(len(m.ids))
		m.ids = append(m.ids, ids[i])
		m.ts = append(m.ts, tss[i])
		m.has = append(m.has, false)
		for j, t := range d.terms {
			m.postings[t] = append(m.postings[t], posting{doc, d.tf[j]})
		}
		m.docLen = append(m.docLen, d.n)
		m.totalLen += int64(d.n)
	}
	avg := float64(m.totalLen) / float64(len(m.ids))
	m.norm = make([]float32, len(m.ids))
	for i, n := range m.docLen {
		m.norm[i] = float32(bm25k1 * (1 - bm25b + bm25b*float64(n)/avg))
	}
	return m.loadVecs(s)
}

// loadVecs reads the stored vectors of the entries that have none yet.
func (m *memIndex) loadVecs(s *Store) error {
	want := map[string]int32{}
	for doc, ok := range m.has {
		if !ok {
			want[m.ids[doc]] = int32(doc)
		}
	}
	if len(want) == 0 {
		return nil
	}
	rows, err := s.db.Query(`SELECT c.entry_id, c.hash, v.vec FROM vectors v JOIN chunks c ON c.id = v.chunk_id
		WHERE v.embedder = ? AND c.modality IN ('text', 'caption') ORDER BY c.id`, m.embedder)
	if err != nil {
		return err
	}
	defer rows.Close()
	found := map[int32]bool{}
	for rows.Next() {
		var id, hash string
		var blob []byte
		if err := rows.Scan(&id, &hash, &blob); err != nil {
			return err
		}
		doc, ok := want[id]
		if !ok {
			continue
		}
		v := decodeVec(blob)
		if m.dim == 0 {
			m.dim = len(v)
		}
		if len(v) != m.dim {
			continue
		}
		m.vecs, m.scale = appendInt8(m.vecs, m.scale, v)
		m.vecDoc = append(m.vecDoc, doc)
		m.vecHash = append(m.vecHash, hash)
		found[doc] = true
	}
	for doc := range found {
		m.has[doc] = true
	}
	return rows.Err()
}

// missing counts the entries with no vector yet.
func (m *memIndex) missing() int {
	n := 0
	for _, ok := range m.has {
		if !ok {
			n++
		}
	}
	return n
}

// docTerms are an entry's distinct terms with their counts, and its length.
type docTerms struct {
	terms []string
	tf    []int32
	n     int32
}

// countTerms counts terms by sorting them: no map per entry.
func countTerms(text string) docTerms {
	var all []string
	eachTerm(text, func(t string) { all = append(all, t) })
	sort.Strings(all)
	d := docTerms{n: int32(len(all))}
	for i := 0; i < len(all); {
		j := i
		for j < len(all) && all[j] == all[i] {
			j++
		}
		d.terms = append(d.terms, all[i])
		d.tf = append(d.tf, int32(j-i))
		i = j
	}
	return d
}

// parallel calls fn(i) for i in [0, n), split across the cores.
func parallel(n int, fn func(int)) {
	parts := min(runtime.GOMAXPROCS(0), max(1, n/256))
	var wg sync.WaitGroup
	for p := range parts {
		lo, hi := p*n/parts, (p+1)*n/parts
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// current reports whether the index still matches the keyword table, by
// row count, total text length and the entry in the last row. SQLite reuses
// the number of a deleted last row, so new rows alone do not show a delete.
func (m *memIndex) current(s *Store) (bool, error) {
	var rows, maxRowid int64
	var textLen float64
	err := s.db.QueryRow(`SELECT count(*), coalesce(max(rowid), 0),
		total(length(entry_id)+length(title)+length(body)+length(tags)+length(captions)) FROM entries_fts`).Scan(&rows, &maxRowid, &textLen)
	if err != nil {
		return false, err
	}
	if rows != int64(len(m.ids)) || maxRowid != m.maxRowid || int64(textLen) != m.textLen {
		return false, nil
	}
	if rows == 0 {
		return true, nil
	}
	var last string
	if err := s.db.QueryRow(`SELECT entry_id FROM entries_fts WHERE rowid = ?`, maxRowid).Scan(&last); err != nil {
		return false, err
	}
	return last == m.ids[len(m.ids)-1], nil
}

// eachTerm calls fn with each indexed term of text: lowercased runs of
// letters and digits (as FTS5's unicode61 splits them), lightly stemmed.
func eachTerm(text string, fn func(string)) {
	start := -1
	for i, r := range text {
		word := unicode.IsLetter(r) || unicode.IsDigit(r)
		if word && start < 0 {
			start = i
		}
		if !word && start >= 0 {
			fn(stem(strings.ToLower(text[start:i])))
			start = -1
		}
	}
	if start >= 0 {
		fn(stem(strings.ToLower(text[start:])))
	}
}

// queryTerms are the distinct terms of q, without stopwords.
func queryTerms(q string) []string {
	var terms []string
	seen := map[string]bool{}
	start := -1
	add := func(w string) {
		if stopwords[strings.ToLower(w)] {
			return
		}
		t := stem(strings.ToLower(w))
		if !seen[t] {
			seen[t] = true
			terms = append(terms, t)
		}
	}
	for i, r := range q {
		word := unicode.IsLetter(r) || unicode.IsDigit(r)
		if word && start < 0 {
			start = i
		}
		if !word && start >= 0 {
			add(q[start:i])
			start = -1
		}
	}
	if start >= 0 {
		add(q[start:])
	}
	return terms
}

// rank fuses BM25 keyword ranks with vector ranks (reciprocal rank fusion)
// and returns every candidate, best first, without loading any entry.
// qvec may be nil: then only keywords are used. only, when not nil, holds
// the entries a filter allows.
//
// order "newest" or "oldest" is for "the latest on X" and "when did X first
// happen": keyword candidates are then taken in date order, with all terms
// required if any entry has them all. On the 20,000-entry test, relevance
// order put the newest of 351 "Redis cluster" entries nowhere in the top 5.
func (m *memIndex) rank(q string, qvec []float32, only map[string]bool, order string) []candidate {
	byDoc := map[int32]*candidate{}
	get := func(doc int32) *candidate {
		c := byDoc[doc]
		if c == nil {
			c = &candidate{id: m.ids[doc]}
			byDoc[doc] = c
		}
		return c
	}
	allowed := func(doc int) bool { return only == nil || only[m.ids[doc]] }

	terms := queryTerms(q)
	if len(terms) > 0 && len(m.ids) > 0 {
		n := float64(len(m.ids))
		scores := make([]float32, len(m.ids))
		matched := make([]uint8, len(m.ids))
		for _, t := range terms {
			ps := m.postings[t]
			if len(ps) == 0 {
				continue
			}
			idf := float32(math.Log((n-float64(len(ps))+0.5)/(float64(len(ps))+0.5) + 1))
			w := idf * (bm25k1 + 1)
			for _, p := range ps {
				if !allowed(int(p.doc)) {
					continue
				}
				tf := float32(p.tf)
				scores[p.doc] += w * tf / (tf + m.norm[p.doc])
				matched[p.doc]++
			}
		}
		var top []int32
		if order == "" {
			top = topK(scores, kwCandidates)
		} else {
			top = m.byDate(matched, len(terms), order == "oldest")
		}
		for rank, doc := range top {
			c := get(doc)
			c.score += 1.0 / float64(rrfK+rank+1)
			c.via = append(c.via, "keyword")
		}
		// Every term present, one of them rare: strong evidence alone (an
		// identifier). Common words together ("need", "restart") are not.
		rare := false
		for _, t := range terms {
			if df := len(m.postings[t]); df > 0 && float64(df) <= max(1, rareShare*n) {
				rare = true
			}
		}
		for _, doc := range top {
			if rare && order == "" && int(matched[doc]) == len(terms) {
				get(doc).allTerms = true
			}
		}
	}

	if qvec != nil && len(m.ids) > 0 && len(qvec) == m.dim {
		sims, rows := m.similarities(qvec)
		for doc := range sims {
			if !allowed(doc) {
				sims[doc] = 0
			}
		}
		for rank, doc := range topK(sims, vecCandidates) {
			c := get(doc)
			c.score += 1.0 / float64(rrfK+rank+1)
			c.via = append(c.via, "vector")
			c.match = "text"
			if h := m.vecHash[rows[doc]]; h != "" {
				c.match = "caption:" + h
			}
			c.sim = sims[doc]
		}
	}

	cands := make([]candidate, 0, len(byDoc))
	for _, c := range byDoc {
		cands = append(cands, *c)
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].score != cands[j].score {
			return cands[i].score > cands[j].score
		}
		return cands[i].id > cands[j].id // newer first on ties
	})
	return cands
}

// byDate is the keyword candidates in date order: the entries with every
// query term if any has them all, else those with any, newest first or
// oldest first.
func (m *memIndex) byDate(matched []uint8, terms int, oldest bool) []int32 {
	var all, some []int32
	for doc, n := range matched {
		if int(n) == terms {
			all = append(all, int32(doc))
		}
		if n > 0 {
			some = append(some, int32(doc))
		}
	}
	if len(all) > 0 {
		some = all
	}
	sort.Slice(some, func(i, j int) bool {
		a, b := some[i], some[j]
		if m.ts[a] != m.ts[b] {
			return (m.ts[a] < m.ts[b]) == oldest
		}
		return (m.ids[a] < m.ids[b]) == oldest
	})
	return some[:min(len(some), kwCandidates)]
}

// similarities is each entry's best cosine with q over its vectors, and
// the row of the vector that gave it. The dot products are split across
// the cores: 20,000 x 384 is 8 million multiplications.
func (m *memIndex) similarities(q []float32) ([]float32, []int32) {
	sims, rows := make([]float32, len(m.ids)), make([]int32, len(m.ids))
	if m.dim == 0 {
		return sims, rows
	}
	q8, qs := appendInt8(nil, nil, q)
	rowSims := make([]float32, len(m.vecDoc))
	parallel(len(m.vecDoc), func(i int) {
		rowSims[i] = qs[0] * m.scale[i] * float32(dot8(q8, m.vecs[i*m.dim:(i+1)*m.dim]))
	})
	for i, doc := range m.vecDoc {
		if rowSims[i] > sims[doc] {
			sims[doc], rows[doc] = rowSims[i], int32(i)
		}
	}
	return sims, rows
}

// appendInt8 appends v as int8 with its own scale: the largest component
// maps to ±127.
func appendInt8(dst []int8, scales []float32, v []float32) ([]int8, []float32) {
	var top float32
	for _, x := range v {
		top = max(top, x, -x)
	}
	if top == 0 {
		top = 1
	}
	k := 127 / top
	for _, x := range v {
		dst = append(dst, int8(math.Round(float64(x*k))))
	}
	return dst, append(scales, top/127)
}

// dot8 is a dot product of equal-length vectors with eight accumulators,
// which lets the CPU overlap the multiplications.
func dot8(a, b []int8) int32 {
	b = b[:len(a)]
	var s0, s1, s2, s3, s4, s5, s6, s7 int32
	i := 0
	for ; i+8 <= len(a); i += 8 {
		s0 += int32(a[i]) * int32(b[i])
		s1 += int32(a[i+1]) * int32(b[i+1])
		s2 += int32(a[i+2]) * int32(b[i+2])
		s3 += int32(a[i+3]) * int32(b[i+3])
		s4 += int32(a[i+4]) * int32(b[i+4])
		s5 += int32(a[i+5]) * int32(b[i+5])
		s6 += int32(a[i+6]) * int32(b[i+6])
		s7 += int32(a[i+7]) * int32(b[i+7])
	}
	for ; i < len(a); i++ {
		s0 += int32(a[i]) * int32(b[i])
	}
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
}

// topK returns the indexes of the k largest positive values, largest first.
func topK(vals []float32, k int) []int32 {
	h := make(minHeap, 0, k)
	for i, v := range vals {
		if v <= 0 {
			continue
		}
		if len(h) < k {
			heap.Push(&h, scored{int32(i), v})
		} else if v > h[0].v {
			h[0] = scored{int32(i), v}
			heap.Fix(&h, 0)
		}
	}
	out := make([]int32, len(h))
	for i := len(h) - 1; i >= 0; i-- {
		out[i] = heap.Pop(&h).(scored).i
	}
	return out
}

type scored struct {
	i int32
	v float32
}

type minHeap []scored

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].v < h[j].v }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(scored)) }
func (h *minHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// snippet marks the first query term in body, with a few words around it,
// as FTS5's snippet() does: "…the [token] refresh raced…".
func snippet(body string, terms []string) string {
	words := strings.Fields(body)
	want := map[string]bool{}
	for _, t := range terms {
		want[t] = true
	}
	for i, w := range words {
		var hit bool
		eachTerm(w, func(t string) { hit = hit || want[t] })
		if !hit {
			continue
		}
		from, to := max(0, i-6), min(len(words), i+10)
		out := make([]string, 0, to-from)
		for j := from; j < to; j++ {
			x := words[j]
			eachTerm(x, func(t string) {
				if want[t] && !strings.HasPrefix(x, "[") {
					x = "[" + x + "]"
				}
			})
			out = append(out, x)
		}
		s := strings.Join(out, " ")
		if from > 0 {
			s = "…" + s
		}
		if to < len(words) {
			s += "…"
		}
		return s
	}
	return ""
}
