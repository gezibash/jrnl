package main

// stem is the Porter stemmer (M.F. Porter, "An algorithm for suffix
// stripping", 1980), as SQLite's FTS5 uses for its keyword table: "moved"
// and "move" meet at "move", "increases" and "increase" at "increas". It is
// applied the same way to entries and to queries. Words that are not
// lowercase ASCII letters are left as they are.
func stem(w string) string {
	if len(w) <= 2 {
		return w
	}
	for i := 0; i < len(w); i++ {
		if w[i] < 'a' || w[i] > 'z' {
			return w
		}
	}
	s := stemmer{b: []byte(w), k: len(w) - 1}
	s.step1()
	s.replace(1, [][2]string{{"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"}, {"anci", "ance"}, {"izer", "ize"},
		{"bli", "ble"}, {"alli", "al"}, {"entli", "ent"}, {"eli", "e"}, {"ousli", "ous"}, {"ization", "ize"},
		{"ation", "ate"}, {"ator", "ate"}, {"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
		{"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"}, {"biliti", "ble"}, {"logi", "log"}})
	s.replace(1, [][2]string{{"icate", "ic"}, {"ative", ""}, {"alize", "al"}, {"iciti", "ic"}, {"ical", "ic"},
		{"ful", ""}, {"ness", ""}})
	s.step4()
	s.step5()
	return string(s.b[:s.k+1])
}

// stemmer holds the word in b[0..k]; j is the end of the stem that the
// last suffix test left.
type stemmer struct {
	b    []byte
	j, k int
}

// cons: b[i] is a consonant. y is one at the start and after a vowel.
func (s *stemmer) cons(i int) bool {
	switch s.b[i] {
	case 'a', 'e', 'i', 'o', 'u':
		return false
	case 'y':
		return i == 0 || !s.cons(i-1)
	}
	return true
}

// m counts the vowel-consonant sequences in the stem b[0..j].
func (s *stemmer) m() int {
	n, i := 0, 0
	for i <= s.j && s.cons(i) {
		i++
	}
	for i <= s.j {
		for i <= s.j && !s.cons(i) {
			i++
		}
		if i > s.j {
			break
		}
		n++
		for i <= s.j && s.cons(i) {
			i++
		}
	}
	return n
}

func (s *stemmer) vowelInStem() bool {
	for i := 0; i <= s.j; i++ {
		if !s.cons(i) {
			return true
		}
	}
	return false
}

func (s *stemmer) doubleCons(i int) bool { return i >= 1 && s.b[i] == s.b[i-1] && s.cons(i) }

// cvc: b[i-2..i] is consonant, vowel, consonant, the last not w, x or y
// (hop, but not snow, box or tray): the form that gets its e back.
func (s *stemmer) cvc(i int) bool {
	if i < 2 || !s.cons(i) || s.cons(i-1) || !s.cons(i-2) {
		return false
	}
	return s.b[i] != 'w' && s.b[i] != 'x' && s.b[i] != 'y'
}

// ends reports whether the word ends in suf, and sets j before it.
func (s *stemmer) ends(suf string) bool {
	if len(suf) > s.k+1 || string(s.b[s.k+1-len(suf):s.k+1]) != suf {
		return false
	}
	s.j = s.k - len(suf)
	return true
}

func (s *stemmer) setTo(t string) {
	s.b = append(s.b[:s.j+1], t...)
	s.k = s.j + len(t)
}

// replace swaps the first suffix the word ends in, if the stem before it
// measures more than min.
func (s *stemmer) replace(min int, rules [][2]string) {
	for _, r := range rules {
		if s.ends(r[0]) {
			if s.m() >= min {
				s.setTo(r[1])
			}
			return
		}
	}
}

// step1 removes plurals, -ed and -ing, and turns a final y into i.
func (s *stemmer) step1() {
	if s.b[s.k] == 's' {
		switch {
		case s.ends("sses"):
			s.k -= 2
		case s.ends("ies"):
			s.setTo("i")
		case s.b[s.k-1] != 's':
			s.k--
		}
	}
	if s.ends("eed") {
		if s.m() > 0 {
			s.k--
		}
	} else if (s.ends("ed") || s.ends("ing")) && s.vowelInStem() {
		s.k = s.j
		switch {
		case s.ends("at"):
			s.setTo("ate")
		case s.ends("bl"):
			s.setTo("ble")
		case s.ends("iz"):
			s.setTo("ize")
		case s.doubleCons(s.k):
			if c := s.b[s.k]; c != 'l' && c != 's' && c != 'z' {
				s.k--
			}
		default:
			if s.j = s.k; s.m() == 1 && s.cvc(s.k) {
				s.setTo("e")
			}
		}
	}
	if s.ends("y") && s.vowelInStem() {
		s.b[s.k] = 'i'
	}
}

// step4 drops a last suffix from a stem that measures more than 1.
func (s *stemmer) step4() {
	for _, suf := range []string{"al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
		"ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"} {
		if !s.ends(suf) {
			continue
		}
		if suf == "ion" && (s.j < 0 || (s.b[s.j] != 's' && s.b[s.j] != 't')) {
			return
		}
		if s.m() > 1 {
			s.k = s.j
		}
		return
	}
}

// step5 drops a final e and one of a final double l.
func (s *stemmer) step5() {
	s.j = s.k
	if s.b[s.k] == 'e' {
		if a := s.m(); a > 1 || (a == 1 && !s.cvc(s.k-1)) {
			s.k--
		}
	}
	if s.b[s.k] == 'l' && s.doubleCons(s.k) && s.m() > 1 {
		s.k--
	}
}
