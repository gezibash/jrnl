package main

import (
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// Entries are sent to Gemma in parts of at most this many characters;
	// the part summaries are then combined. Keeps each call fast and inside
	// the model's attention, whatever the period's length.
	summaryPartChars = 40000
	shortIDLen       = 14 // time + 4 random characters: unique, and jrnl show accepts it
)

// Entries are cited by number, and jrnl turns the numbers into entry IDs.
// Asked to copy 14-character IDs, Gemma listed 30-90 per bullet (about 8
// tokens each), ran out of its 4,000-token budget on a 1,512-entry week, and
// garbled some IDs.
const summaryRules = "Cover what was done, decisions and why, problems hit and how they were fixed, " +
	"and what is still open. An entry marked closed or replaced is settled: report it as past, with how " +
	"it ended, and never as the current state. Under what is still open, list only entries marked (open). " +
	"Group short markdown bullets under headings. End each bullet with the " +
	"numbers of up to 5 entries it comes from, like [12, 40]. Leave out nothing important and add " +
	"nothing that isn't in the entries. No preamble."

const maxCitesPerBullet = 5

func cmdSummarize(args []string) error {
	fs, project := newFlags("summarize")
	f, _ := filterFlags(fs)
	refresh := fs.Bool("refresh", false, "")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	if f.Since == "" && f.Until == "" && f.On == "" && f.Session == "" {
		return fmt.Errorf(`summarize: say which period, e.g. --on "last week", --since monday or --session last`)
	}
	s, g, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := resolveFilter(s, f); err != nil {
		return err
	}
	ai, err := NewAI()
	if err != nil {
		return err
	}
	sum, err := summarizeFilter(s, ai, g, *f, *refresh)
	if errors.Is(err, errNoChat) {
		return fmt.Errorf("summaries need a chat model, which is not set up (see `jrnl doctor`); `jrnl list` with the same filters shows the entries")
	}
	if err != nil {
		return err
	}
	if sum == nil {
		fmt.Println("No entries in that period.")
		return nil
	}
	printEntry(*sum)
	return nil
}

// summarizeFilter returns the saved summary of the entries matching f,
// writing it first if it is missing or an entry was added since. A period of
// several days that is too long for one call is built from day summaries,
// which are saved too and reused by later weeks and months.
func summarizeFilter(s *Store, ai *AI, g gitInfo, f Filter, refresh bool) (*Entry, error) {
	all, err := s.List(f, 1_000_000, true)
	if err != nil {
		return nil, err
	}
	var entries []Entry
	for _, e := range all {
		if e.Kind != "summary" {
			entries = append(entries, e)
		}
	}
	if len(entries) == 0 {
		return nil, nil
	}
	key, err := summaryKey(f)
	if err != nil {
		return nil, err
	}

	// Reuse the saved summary unless an entry was added after it, or one of
	// its entries was closed or replaced since: see coverage. IDs sort
	// by creation time, so a newer ID means a newer entry, even one logged
	// with --at for an earlier day.
	newest := ""
	for _, e := range entries {
		newest = max(newest, e.ID)
	}
	old, err := s.List(Filter{Kind: "summary"}, 1_000_000, false)
	if err != nil {
		return nil, err
	}
	for _, o := range old {
		if o.Title != key {
			continue
		}
		if o.ID > newest && !refresh && strings.HasSuffix(o.Body, coverage(entries)) {
			return &o, nil
		}
		if err := s.Delete(o.ID); err != nil {
			return nil, err
		}
	}

	size := 0
	var days []string
	for n, e := range entries {
		size += len(entryLine(s, n, e)) + 1
		if d := localTime(e.TS)[:10]; len(days) == 0 || days[len(days)-1] != d {
			days = append(days, d)
		}
	}
	var body string
	if len(days) > 1 && size > summaryPartChars {
		body, err = summarizeByDay(s, ai, g, f, refresh, entries, days)
	} else {
		body, err = summarizeEntries(s, ai, entries)
	}
	if err != nil {
		return nil, err
	}
	sum := Entry{
		ID: newID(), TS: utc(now()), Project: g.project, Session: os.Getenv("CLAUDE_CODE_SESSION_ID"),
		Author: defaultAuthor(), Kind: "summary", Branch: g.branch, Commit: g.commit, Tags: []string{"summary"},
		Title: key, Body: body + "\n\n" + coverage(entries),
	}
	if err := s.Insert(sum, []Chunk{{Modality: "text"}}); err != nil {
		return nil, err
	}
	if _, err := process(s, ai, sum.ID, true); err != nil {
		return nil, err
	}
	saved, err := s.Get(sum.ID)
	return &saved, err
}

// summarizeByDay gets each day's summary (saved or new, 4 days at a time)
// and writes a short overview citing them. Summarizing 1,512 entries in
// 40,000-character parts made Gemma list them nearly one by one and hit its
// 4,000-token limit in 2 of 5 parts; a day at a time keeps every call small.
func summarizeByDay(s *Store, ai *AI, g gitInfo, f Filter, refresh bool, entries []Entry, days []string) (string, error) {
	fmt.Fprintf(os.Stderr, "jrnl: summarizing %d entries day by day (%d days)\n", len(entries), len(days))
	sums, errs := make([]*Entry, len(days)), make([]error, len(days))
	sem := make(chan struct{}, 4)
	var wg sync.WaitGroup
	for i, d := range days {
		wg.Add(1)
		go func(i int, d string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			df := f
			df.Since, df.Until, df.On = "", "", d
			sums[i], errs[i] = summarizeFilter(s, ai, g, df, refresh)
		}(i, d)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return "", err
		}
	}
	// The period summary is a short overview that cites day summaries; the
	// day summaries cite entries. Asked to merge the days' entry-level detail
	// instead, Gemma listed all ~1,500 facts of a busy week and ran past
	// 4,000 and then 8,000 tokens.
	var parts, footer []string
	var dayIDs []string
	for i, sum := range sums {
		if sum == nil {
			continue
		}
		dayIDs = append(dayIDs, sum.ID[:shortIDLen])
		body, _, _ := strings.Cut(sum.Body, "\n\n_Covers ")
		body = reCiteGroup.ReplaceAllString(body, "") // entry citations stay in the day summary
		parts = append(parts, fmt.Sprintf("D%d (%s):\n%s", len(dayIDs), days[i], body))
		footer = append(footer, fmt.Sprintf("%s [%s]", days[i], sum.ID[:shortIDLen]))
	}
	prompt := "Write a short overview of this period for a teammate catching up, from the daily summaries below. " +
		"Use at most 15 bullets in total, under the headings Themes, Decisions, Problems and fixes, and Still open; " +
		"skip a heading with nothing under it. End each bullet with the days it draws on, like [D2, D5]. " +
		"Say nothing that isn't in the summaries. No preamble.\n\n" + strings.Join(parts, "\n\n---\n\n")
	out, err := ai.chat(prompt, 2000)
	if err != nil {
		return "", err
	}
	out, dropped := citeDays(out, dayIDs)
	if dropped > 0 {
		fmt.Fprintf(os.Stderr, "jrnl: removed %d citations that match no day\n", dropped)
	}
	return out + "\n\nDetails by day: " + strings.Join(footer, ", ") + ".", nil
}

var reDayList = regexp.MustCompile(`^\s*D\d+(\s*[,;]\s*D\d+)*\s*$`)

// citeDays turns [D2, D5] into the IDs of those days' summaries, at most 5
// per bullet, dropping days that don't exist.
func citeDays(text string, dayIDs []string) (string, int) {
	dropped := 0
	out := reCiteGroup.ReplaceAllStringFunc(text, func(group string) string {
		inner := group[1 : len(group)-1]
		if !reDayList.MatchString(inner) {
			return group
		}
		var keep []string
		seen := map[int]bool{}
		for _, f := range strings.FieldsFunc(inner, func(r rune) bool { return r == ',' || r == ';' || r == ' ' }) {
			n, err := strconv.Atoi(strings.TrimPrefix(f, "D"))
			if err != nil || n < 1 || n > len(dayIDs) {
				dropped++
				continue
			}
			if !seen[n] && len(keep) < maxCitesPerBullet {
				seen[n] = true
				keep = append(keep, dayIDs[n-1])
			}
		}
		if len(keep) == 0 {
			return ""
		}
		return "[" + strings.Join(keep, ", ") + "]"
	})
	return out, dropped
}

// summaryKey names a summary by the absolute period and filters it covers,
// so "last week" asked next week is a different summary.
func summaryKey(f Filter) (string, error) {
	ref := now()
	var parts []string
	day := func(t time.Time) string { return t.Format("2006-01-02") }
	span := func(flag, v string, useStart bool) error {
		if v == "" {
			return nil
		}
		sp, err := parseWhen(v, ref)
		if err != nil {
			return fmt.Errorf("%s: %w", flag, err)
		}
		switch {
		case flag == "on":
			parts = append(parts, day(sp.start)+" → "+day(sp.end.AddDate(0, 0, -1)))
		case sp.start.Equal(sp.end):
			parts = append(parts, flag+" "+sp.start.Format("2006-01-02 15:04"))
		case useStart:
			parts = append(parts, flag+" "+day(sp.start))
		default:
			parts = append(parts, flag+" "+day(sp.end.AddDate(0, 0, -1)))
		}
		return nil
	}
	if err := span("on", f.On, true); err != nil {
		return "", err
	}
	if err := span("since", f.Since, true); err != nil {
		return "", err
	}
	if err := span("until", f.Until, false); err != nil {
		return "", err
	}
	for _, kv := range [][2]string{{"session", f.Session}, {"kind", f.Kind}, {"status", f.Status}, {"author", f.Author}, {"file", f.File}} {
		if kv[1] != "" {
			parts = append(parts, kv[0]+" "+kv[1])
		}
	}
	for _, t := range f.Tags {
		parts = append(parts, "tag "+t)
	}
	return "Summary: " + strings.Join(parts, ", "), nil
}

// summarizeEntries sends the entries to Gemma, in parts if they are long,
// and combines the part summaries into one.
func summarizeEntries(s *Store, ai *AI, entries []Entry) (string, error) {
	var parts []string
	var cur strings.Builder
	for n, e := range entries {
		line := entryLine(s, n, e)
		if cur.Len()+len(line) > summaryPartChars && cur.Len() > 0 {
			parts = append(parts, cur.String())
			cur.Reset()
		}
		cur.WriteString(line + "\n")
	}
	parts = append(parts, cur.String())

	// Parts go 4 at a time: one after another, a 1,512-entry week took 253 s.
	sums, errs := make([]string, len(parts)), make([]error, len(parts))
	if len(parts) > 1 {
		fmt.Fprintf(os.Stderr, "jrnl: summarizing %d entries in %d parts\n", len(entries), len(parts))
	}
	sem := make(chan struct{}, 4)
	var wg sync.WaitGroup
	for i, p := range parts {
		wg.Add(1)
		go func(i int, p string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			prompt := "Summarize these journal entries from a software project for a teammate catching up. " + summaryRules + "\n\nEntries:\n" + p
			sums[i], errs[i] = ai.chat(prompt, 4000)
		}(i, p)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return "", err
		}
	}
	out := sums[0]
	if len(sums) > 1 {
		prompt := "Combine these summaries of consecutive parts of one period into one summary, keeping the [number] citations. " +
			summaryRules + "\n\n" + strings.Join(sums, "\n\n---\n\n")
		var err error
		if out, err = ai.chat(prompt, 4000); err != nil {
			return "", err
		}
	}
	out, dropped := citeIDs(out, entries)
	if dropped > 0 {
		fmt.Fprintf(os.Stderr, "jrnl: removed %d citations that match no entry in the period\n", dropped)
	}
	return out, nil
}

var (
	reCiteGroup = regexp.MustCompile(`\[([^\[\]]*)\]`)
	reNumList   = regexp.MustCompile(`^\s*\d+(\s*[,;]\s*\d+)*\s*$`)
)

// citeIDs turns [12, 40] citations into entry IDs that jrnl show accepts,
// keeps at most 5 per bullet, and drops numbers outside the entry list.
// Brackets that aren't number lists are left alone.
func citeIDs(text string, entries []Entry) (string, int) {
	dropped := 0
	out := reCiteGroup.ReplaceAllStringFunc(text, func(group string) string {
		inner := group[1 : len(group)-1]
		if !reNumList.MatchString(inner) {
			return group
		}
		var keep []string
		seen := map[int]bool{}
		for _, f := range strings.FieldsFunc(inner, func(r rune) bool { return r == ',' || r == ';' || r == ' ' }) {
			n, err := strconv.Atoi(f)
			if err != nil || n < 1 || n > len(entries) {
				dropped++
				continue
			}
			if !seen[n] && len(keep) < maxCitesPerBullet {
				seen[n] = true
				keep = append(keep, entries[n-1].ID[:shortIDLen])
			}
		}
		if len(keep) == 0 {
			return ""
		}
		return "[" + strings.Join(keep, ", ") + "]"
	})
	return out, dropped
}

// coverage is a summary's last line: what it covers, and how many of those
// entries are settled. A saved summary is reused only while this line still
// holds, so one written before an entry was closed, reopened or replaced is
// redone, whenever that happened.
func coverage(entries []Entry) string {
	var closed, replaced int
	for _, e := range entries {
		if e.Status == "closed" {
			closed++
		}
		if len(e.SupersededBy) > 0 {
			replaced++
		}
	}
	settled := ""
	if closed+replaced > 0 {
		settled = fmt.Sprintf(" (%d closed, %d replaced since)", closed, replaced)
	}
	return fmt.Sprintf("_Covers %d entries%s, %s → %s._", len(entries), settled, localTime(entries[0].TS), localTime(entries[len(entries)-1].TS))
}

// entryLine is how an entry is shown to the summarizer: its number, time,
// kind, status, text and attachment descriptions. A closed entry carries
// when and how it was closed, and a replaced one what replaced it, also
// when that happened after the period: the summary is read later.
func entryLine(s *Store, n int, e Entry) string {
	line := fmt.Sprintf("[%d] %s %s", n+1, localTime(e.TS), e.Kind)
	if e.Status == "closed" {
		line += " (closed"
		if e.ClosedAt != "" {
			line += " " + localTime(e.ClosedAt)[:10]
		}
		if by, err := s.Get(e.ClosedBy); err == nil && by.Body != "" {
			line += ": " + truncate(oneLine(by.Body), 200)
		}
		line += ")"
	} else if e.Status != "" {
		line += " (" + e.Status + ")"
	}
	for _, id := range e.SupersededBy {
		if by, err := s.Get(id); err == nil {
			line += fmt.Sprintf(" (replaced %s by: %s)", localTime(by.TS)[:10], by.Title)
		}
	}
	line += ": " + truncate(oneLine(e.Body), 600)
	for _, a := range e.Assets {
		if a.Description != "" {
			line += " | attachment: " + truncate(oneLine(a.Description), 200)
		}
	}
	return line
}
