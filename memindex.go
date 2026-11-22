package main

import (
	"container/heap"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
	"unicode"

	potion "github.com/trengrj/go-potion"
)

// memIndex is everything jrnl serve searches, in memory, the way Typesense
// keeps its index: one static (potion) vector per entry and an inverted
// index of its words for BM25. SQLite's FTS5 took 20 ms to score a common
// query over 20,000 entries (42 ms with snippets); this takes about 1 ms.
type memIndex struct {
	ids      []string
	dim      int
	vecs     []int8    // entry i's vector is vecs[i*dim : (i+1)*dim] times scale[i]
	scale    []float32 // int8 keeps a quarter of float32's memory; cosine moves < 0.01
	postings map[string][]posting
	docLen   []int32
	norm     []float32 // BM25's length factor per entry: k1*(1-b+b*len/avg)
	totalLen int64
	maxRowid int64 // the keyword table's last row, to add new entries alone
	textLen  int64 // characters indexed, as SQL length() counts them
}

const bm25k1, bm25b = 1.2, 0.75

// rareShare: a term in at most this share of entries is rare enough that
// matching every query term counts on its own.
var rareShare = 0.03

type posting struct {
	doc int32
	tf  int32
}

// buildMemIndex indexes every entry's searchable text: what the keyword
// table holds (title, body, tags, attachment descriptions).
func buildMemIndex(s *Store, enc *potion.Potion) (*memIndex, error) {
	m := &memIndex{postings: map[string][]posting{}}
	return m, m.addSince(s, enc)
}

// addSince adds the keyword rows written after the last one indexed.
func (m *memIndex) addSince(s *Store, enc *potion.Potion) error {
	rows, err := s.db.Query(`SELECT rowid, entry_id, title, body, tags, captions,
		length(entry_id)+length(title)+length(body)+length(tags)+length(captions)
		FROM entries_fts WHERE rowid > ? ORDER BY rowid`, m.maxRowid)
	if err != nil {
		return err
	}
	var ids, texts []string
	for rows.Next() {
		var id, title, body, tags, captions string
		var n int64
		if err := rows.Scan(&m.maxRowid, &id, &title, &body, &tags, &captions, &n); err != nil {
			rows.Close()
			return err
		}
		ids = append(ids, id)
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

	// Embedding and tokenizing are independent; both use every core.
	var vecs [][]float32
	var encErr error
	done := make(chan struct{})
	go func() {
		vecs, encErr = enc.EncodeMany(texts)
		close(done)
	}()
	docs := make([]docTerms, len(texts))
	parallel(len(texts), func(i int) { docs[i] = countTerms(texts[i]) })
	<-done
	if encErr != nil {
		return encErr
	}

	for i, d := range docs {
		doc := int32(len(m.ids))
		m.ids = append(m.ids, ids[i])
		for j, t := range d.terms {
			m.postings[t] = append(m.postings[t], posting{doc, d.tf[j]})
		}
		m.docLen = append(m.docLen, d.n)
		m.totalLen += int64(d.n)
		m.dim = len(vecs[i])
		m.vecs, m.scale = appendInt8(m.vecs, m.scale, vecs[i])
	}
	avg := float64(m.totalLen) / float64(len(m.ids))
	m.norm = make([]float32, len(m.ids))
	for i, n := range m.docLen {
		m.norm[i] = float32(bm25k1 * (1 - bm25b + bm25b*float64(n)/avg))
	}
	return nil
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

// stem strips common English suffixes, so "refreshed", "refreshes" and
// "refreshing" meet "refresh". Rougher than FTS5's Porter stemmer, and
// applied the same way to entries and queries.
func stem(w string) string {
	if len(w) <= 3 {
		return w
	}
	switch {
	case strings.HasSuffix(w, "ies") && len(w) > 4:
		return w[:len(w)-3] + "y"
	case strings.HasSuffix(w, "sses"):
		return w[:len(w)-2]
	case strings.HasSuffix(w, "ing") && len(w) > 5:
		return w[:len(w)-3]
	case strings.HasSuffix(w, "ed") && len(w) > 4:
		return w[:len(w)-2]
	case strings.HasSuffix(w, "es") && len(w) > 4 && !strings.HasSuffix(w, "ses"):
		return w[:len(w)-2]
	case strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss") && !strings.HasSuffix(w, "us"):
		return w[:len(w)-1]
	}
	return w
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

// rank fuses BM25 and vector ranks as Store.rank does, from memory.
func (m *memIndex) rank(q string, qvec []float32) []candidate {
	byDoc := map[int32]*candidate{}
	get := func(doc int32) *candidate {
		c := byDoc[doc]
		if c == nil {
			c = &candidate{id: m.ids[doc]}
			byDoc[doc] = c
		}
		return c
	}

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
				tf := float32(p.tf)
				scores[p.doc] += w * tf / (tf + m.norm[p.doc])
				matched[p.doc]++
			}
		}
		top := topK(scores, kwCandidates)
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
			if rare && int(matched[doc]) == len(terms) {
				get(doc).allTerms = true
			}
		}
	}

	if qvec != nil && len(m.ids) > 0 && len(qvec) == m.dim {
		sims := m.similarities(qvec)
		for rank, doc := range topK(sims, vecCandidates) {
			c := get(doc)
			c.score += 1.0 / float64(rrfK+rank+1)
			c.via = append(c.via, "vector")
			c.match = "text"
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
		return cands[i].id > cands[j].id
	})
	return cands
}

// similarities is the dot product of q with every entry's vector, split
// across the cores: 20,000 x 512 is 10 million multiplications.
func (m *memIndex) similarities(q []float32) []float32 {
	sims := make([]float32, len(m.ids))
	q8, qs := appendInt8(nil, nil, q)
	parallel(len(m.ids), func(i int) { sims[i] = qs[0] * m.scale[i] * float32(dot8(q8, m.vecs[i*m.dim:(i+1)*m.dim])) })
	return sims
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
