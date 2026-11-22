package main

import "testing"

// Expected stems are the ones Porter's paper and its published sample
// vocabulary give, not this code's output.
func TestStem(t *testing.T) {
	for w, want := range map[string]string{
		"caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
		"feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled", "motoring": "motor", "sing": "sing",
		"conflated": "conflat", "troubled": "troubl", "sized": "size", "hopping": "hop", "tanned": "tan",
		"falling": "fall", "hissing": "hiss", "fizzed": "fizz", "failing": "fail", "filing": "file",
		"happy": "happi", "sky": "sky",
		"relational": "relat", "conditional": "condit", "rational": "ration", "digitizer": "digit",
		"vietnamization": "vietnam", "predication": "predic", "operator": "oper", "feudalism": "feudal",
		"decisiveness": "decis", "hopefulness": "hope", "callousness": "callous", "formality": "formal",
		"sensitivity": "sensit", "sensibility": "sensibl",
		"triplicate": "triplic", "formative": "form", "formalize": "formal", "electricity": "electr",
		"electrical": "electr", "hopeful": "hope", "goodness": "good",
		"revival": "reviv", "allowance": "allow", "inference": "infer", "airliner": "airlin",
		"gyroscopic": "gyroscop", "adjustable": "adjust", "defensible": "defens", "irritant": "irrit",
		"replacement": "replac", "adjustment": "adjust", "dependent": "depend", "adoption": "adopt",
		"communism": "commun", "activate": "activ", "angularity": "angular", "homologous": "homolog",
		"effective": "effect", "bowdlerize": "bowdler",
		"probate": "probat", "rate": "rate", "cease": "ceas", "controlling": "control", "rolling": "roll",
		"generalizations": "gener", "oscillators": "oscil",
		// What the old suffix stripper got wrong: a search for one form missed the other.
		"moved": "move", "move": "move", "increases": "increas", "increase": "increas",
		// Left alone: short, or not lowercase ASCII letters.
		"as": "as", "sqlite3": "sqlite3", "größe": "größe",
	} {
		if got := stem(w); got != want {
			t.Errorf("stem(%q) = %q, want %q", w, got, want)
		}
	}
}
