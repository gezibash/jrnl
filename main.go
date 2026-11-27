// jrnl is a per-project journal for coding agents. Entries live in SQLite;
// markdown is only the input and output format.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"
)

const usage = `jrnl — per-project journal

  jrnl add [flags] [body…]     add an entry (body from args or stdin); prints its ID
      --title T                default: first line of body
      --kind K                 decision | bug | idea | todo | log | milestone | note (default note)
      --tag T                  repeatable
      --link ID                related entry (ID prefix is enough), repeatable
      --supersedes ID          an earlier entry this one replaces, such as an older decision; repeatable
      --file PATH              file the entry is about, repeatable
      --attach PATH[::CAPTION] image, video or audio, repeatable
      --at WHEN                when it happened (default now), e.g. yesterday
      --open                   track it as open until closed (todo entries start open)
  jrnl close ID… [--note TEXT] mark entries closed; the note is saved as an entry linked to them
  jrnl link FROM TO            link two entries (FROM relates to TO), or change or remove the link
      --supersedes             FROM replaces TO: TO is shown as superseded
      --rm                     remove the link between them
  jrnl reopen ID…              mark entries open again
  jrnl rm ID…                  delete entries for good, with their attachments and the summaries citing them
  jrnl show ID…                print entries as markdown (ID prefix is enough); also read, open
  jrnl list [filters]          newest first; --oldest for oldest first
  jrnl search [filters] QUERY  keyword + semantic search over entries and attachment descriptions
      --newest | --oldest      the latest / the first matches, instead of the best
      --fast                   no reranker, and only hits that clearly match (see jrnl serve)
      --min-score S            drop hits whose meaning score (cosine) is under S
  jrnl search --like FILE      find attachments whose description is like FILE's
  jrnl sessions [filters]      recent sessions: last active, how many entries, last title
  jrnl stats [filters]         counts by kind, tag and author
  jrnl summarize [filters]     summary of a period, citing entry IDs; saved and reused
      --refresh                write it again even if nothing changed
  jrnl reindex                 describe and embed whatever failed earlier
      --prune                  also delete the vectors of embedding models no longer set up
  jrnl backup                  consistent copy of the database
  jrnl where                   print this project's journal directory
  jrnl doctor                  what is set up, and how to turn on the rest
  jrnl serve [--background]    answer fast searches from memory over ~/.journal/.run/jrnl.sock (for agent hooks)
  jrnl version

filters: --since WHEN  --until WHEN  --on WHEN
         --kind K  --tag T (repeatable, all must match)  --file NAME
         --status open|closed  --session ID|last  --author A  --limit N
WHEN, in local time: today, yesterday, 3 days ago, monday, last monday,
         this week, last week, this month, last month, october, 2026-11-18,
         2026-11, 36h, 7d, 2w. --since takes its start, --until its end,
         --on the whole day, week or month.
--session last: the most recent session before this one.
every command: --project P (default: from git remote, else directory name)

jrnl works with no setup: entries, filters, keyword and meaning search. Optional parts,
each a role in ~/.config/jrnl/config.toml (or JRNL_CONFIG) that names a provider
and a model; see the README:
  [embed]             search by meaning, any OpenAI-compatible API
                      (default: bge-small-en-v1.5 inside jrnl, downloaded once)
  [chat]              reranking, summaries, captions (a model that reads images)
  [rerank]            faster reranking: a Cohere-style /rerank API
  [media_embed]       search --like by look and sound (Cloudflare Workers AI);
                      without it, --like compares descriptions
  audio transcripts   parakeet-mlx through mise or uvx (Apple silicon)
JRNL_<ROLE>_URL, _KEY and _MODEL (ROLE: EMBED, CHAT, RERANK) and JRNL_CHAT_EXTRA
override the file. Keys come from variables, which jrnl also reads from
JRNL_ENV_FILE (default ~/.config/jrnl/.env), such as a mounted 1Password Environment.
jrnl doctor shows which are on. jrnl reindex indexes what was saved before setup.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	cmds := map[string]func([]string) error{
		"add": cmdAdd, "show": cmdShow, "read": cmdShow, "open": cmdShow, "list": cmdList, "ls": cmdList, "search": cmdSearch,
		"reindex": cmdReindex, "backup": cmdBackup, "where": cmdWhere,
		"sessions": cmdSessions, "stats": cmdStats, "close": cmdClose, "reopen": cmdReopen, "link": cmdLink, "rm": cmdRm,
		"summarize": cmdSummarize, "summarise": cmdSummarize, "doctor": cmdDoctor, "serve": cmdServe,
	}
	name := os.Args[1]
	if name == "version" || name == "--version" {
		fmt.Println("jrnl", version)
		return
	}
	if name == "help" || name == "-h" || name == "--help" {
		fmt.Print(usage)
		return
	}
	cmd, ok := cmds[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "jrnl: unknown command %q\n\n%s", name, usage)
		os.Exit(2)
	}
	if err := cmd(os.Args[2:]); err != nil {
		fmt.Fprintln(os.Stderr, "jrnl:", err)
		os.Exit(1)
	}
}

// version is set at release build time: -ldflags "-X main.version=1.0.0".
var version = "dev"

// Media vectors serve only search --like. Embedding a 941 MB video peaked
// at 7.6 GB of memory before Workers AI rejected it with 413, while 52 MB
// went through. Larger files are matched by their description.
const maxRawBytes = 50 << 20

// Search deadlines. Past them, search falls back to keyword-only results
// (embedding) or keyword+vector order (rerank) instead of waiting.
const (
	embedDeadline  = 3 * time.Second
	rerankDeadline = 4 * time.Second
)

type multi []string

func (m *multi) String() string     { return strings.Join(*m, ",") }
func (m *multi) Set(v string) error { *m = append(*m, v); return nil }

// parse lets flags and positional arguments mix in any order.
func parse(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		if fs.NArg() == 0 {
			return pos, nil
		}
		pos = append(pos, fs.Arg(0))
		args = fs.Args()[1:]
	}
}

func newFlags(name string) (*flag.FlagSet, *string) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprint(os.Stderr, usage) }
	return fs, fs.String("project", "", "project name")
}

func filterFlags(fs *flag.FlagSet) (*Filter, *int) {
	f := &Filter{}
	fs.StringVar(&f.Kind, "kind", "", "")
	fs.StringVar(&f.Session, "session", "", "")
	fs.StringVar(&f.Author, "author", "", "")
	fs.StringVar(&f.Since, "since", "", "")
	fs.StringVar(&f.Until, "until", "", "")
	fs.StringVar(&f.On, "on", "", "")
	fs.StringVar(&f.File, "file", "", "")
	fs.StringVar(&f.Status, "status", "", "")
	fs.Var((*multi)(&f.Tags), "tag", "")
	return f, fs.Int("limit", 20, "")
}

// resolveFilter turns --session last into the previous session's ID.
// me is the caller's own session, which "last" skips.
func resolveFilter(s *Store, f *Filter, me string) error {
	if f.Session == "last" {
		id, err := s.LastSession(me)
		if err != nil {
			return err
		}
		f.Session = id
	}
	return nil
}

// noJournal: the project has no journal yet. Only jrnl add starts one;
// a command that reads must not leave an empty journal behind.
type noJournal struct{ project string }

func (n noJournal) Error() string {
	return "no journal for " + n.project + " yet (`jrnl add` starts one)"
}

// open opens the project's journal. create is for jrnl add alone.
func open(project string, create bool) (*Store, gitInfo, error) {
	g := detectGit()
	if project == "" {
		project = g.project
	}
	g.project = project
	if _, err := os.Stat(filepath.Join(journalHome(), project, "journal.db")); err != nil && !create {
		return nil, g, noJournal{project}
	}
	s, err := Open(project)
	return s, g, err
}

var kinds = []string{"decision", "bug", "idea", "todo", "log", "milestone", "note"}

func cmdAdd(args []string) error {
	fs, project := newFlags("add")
	title := fs.String("title", "", "")
	kind := fs.String("kind", "note", "")
	author := fs.String("author", defaultAuthor(), "")
	session := fs.String("session", os.Getenv("CLAUDE_CODE_SESSION_ID"), "")
	var tags, links, supersedes, files, attach multi
	fs.Var(&tags, "tag", "")
	fs.Var(&links, "link", "")
	fs.Var(&supersedes, "supersedes", "")
	fs.Var(&files, "file", "")
	fs.Var(&attach, "attach", "")
	at := fs.String("at", "", "")
	open_ := fs.Bool("open", false, "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}

	body := strings.Join(pos, " ")
	if body == "" && !isTerminal(os.Stdin) {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		body = string(b)
	}
	body = strings.TrimSpace(body)
	if body == "" && *title == "" && len(attach) == 0 {
		return fmt.Errorf("empty entry: give a body as arguments or on stdin")
	}
	if *title == "" {
		*title = firstLine(body)
	}
	// A typo here would otherwise become a kind or tag of its own.
	if !slices.Contains(kinds, *kind) {
		return fmt.Errorf("--kind is one of %s, not %q", strings.Join(kinds, ", "), *kind)
	}
	for _, t := range tags {
		if t == "" || strings.ContainsFunc(t, unicode.IsSpace) {
			return fmt.Errorf("--tag is one word, not %q", t)
		}
	}

	s, g, err := open(*project, true)
	if err != nil {
		return err
	}
	defer s.Close()

	ts := now()
	if *at != "" {
		sp, err := parseWhen(*at, ts)
		if err != nil {
			return fmt.Errorf("--at: %w", err)
		}
		ts = sp.start
	}
	for i, l := range links {
		if links[i], err = s.Resolve(l); err != nil {
			return fmt.Errorf("--link: %w", err)
		}
	}
	for i, l := range supersedes {
		if supersedes[i], err = s.Resolve(l); err != nil {
			return fmt.Errorf("--supersedes: %w", err)
		}
	}
	e := Entry{
		ID: newID(), TS: utc(ts), Project: g.project,
		Session: *session, Author: *author, Kind: *kind, Branch: g.branch, Commit: g.commit,
		Title: *title, Body: body, Tags: tags, Links: links, Supersedes: supersedes, Files: files,
	}
	if *open_ || *kind == "todo" {
		e.Status = "open"
	}
	for _, a := range attach {
		path, caption, _ := strings.Cut(a, "::")
		asset, err := s.Ingest(path, caption)
		if err != nil {
			return fmt.Errorf("attach %s: %w", path, err)
		}
		e.Assets = append(e.Assets, asset)
	}

	// The entry is saved first; describing and embedding happen after, outside
	// the write lock, and anything that fails stays queued for `jrnl reindex`.
	chunks := []Chunk{{Modality: "text"}}
	for _, a := range e.Assets {
		if m := a.Modality(); m != "" && a.Bytes <= maxRawBytes {
			chunks = append(chunks, Chunk{Hash: a.Hash, Modality: m})
		} else if m == "" {
			fmt.Fprintf(os.Stderr, "jrnl: %s stored but not searchable by content (%s is not image, video or audio)\n", a.Name, a.Mime)
		}
	}
	if err := s.Insert(e, chunks); err != nil {
		return err
	}
	// Print the ID as soon as the entry is saved, so a later failure can't
	// lead the caller to retry and create a duplicate.
	fmt.Println(e.ID)
	if e.Kind == "decision" && len(e.Supersedes) == 0 {
		suggestSupersedes(e)
	}
	if ai, err := NewAI(); err != nil {
		fmt.Fprintf(os.Stderr, "jrnl: saved without embedding (run `jrnl reindex` later): %v\n", err)
	} else if _, err := process(s, ai, e.ID, true); err != nil {
		return err
	}
	return nil
}

// suggestSupersedes lists earlier decisions like e, so the agent that wrote
// it can say which one it replaces while it still knows. It asks a running
// jrnl serve (a search there takes about a millisecond) and stays quiet
// without one; it never links anything itself, since a wrong supersedes
// link hides a decision that still holds.
func suggestSupersedes(e Entry) {
	q := e.Title + "\n" + e.Body
	if len(q) > 500 {
		q = q[:500]
	}
	v := url.Values{"project": {e.Project}, "q": {q}, "kind": {"decision"}, "limit": {"3"}, "exclude": {e.ID}}
	r, err := socketClient(socketPath()).Get("http://jrnl/search?" + v.Encode())
	if err != nil {
		return
	}
	defer r.Body.Close()
	b, _ := io.ReadAll(r.Body)
	if r.StatusCode != http.StatusOK || strings.TrimSpace(string(b)) == "" {
		return
	}
	fmt.Fprintf(os.Stderr, "jrnl: earlier decisions like this one. If it replaces one, run: jrnl link %s <ID> --supersedes\n%s",
		e.ID, b)
}

func cmdLink(args []string) error {
	fs, project := newFlags("link")
	supersedes := fs.Bool("supersedes", false, "")
	rm := fs.Bool("rm", false, "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 2 {
		return fmt.Errorf("link: give two entry IDs, FROM and TO")
	}
	s, _, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	from, err := s.Resolve(pos[0])
	if err != nil {
		return err
	}
	to, err := s.Resolve(pos[1])
	if err != nil {
		return err
	}
	if *rm {
		found, err := s.Unlink(from, to)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("link: %s and %s are not linked", from, to)
		}
		fmt.Printf("%s  unlinked  %s\n", from, to)
		return nil
	}
	typ := "relates"
	if *supersedes {
		typ = "supersedes"
	}
	if err := s.Link(from, to, typ); err != nil {
		return err
	}
	fmt.Printf("%s  %s  %s\n", from, typ, to)
	return nil
}

// cmdRm deletes entries for good: the way out when a secret, or something
// wrong, was written down. Backups made earlier keep their copies.
func cmdRm(args []string) error {
	fs, project := newFlags("rm")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(pos) == 0 {
		return fmt.Errorf("rm: give at least one entry ID")
	}
	s, _, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	var ids, titles []string
	for _, p := range pos {
		id, err := s.Resolve(p)
		if err != nil {
			return err
		}
		e, err := s.Get(id)
		if err != nil {
			return err
		}
		ids, titles = append(ids, id), append(titles, e.Title)
	}
	summaries, err := s.Remove(ids)
	if err != nil {
		return err
	}
	for i, id := range ids {
		fmt.Printf("%s  removed  %s\n", id, titles[i])
	}
	if summaries > 0 {
		fmt.Fprintf(os.Stderr, "jrnl: also removed %d saved summary(ies) that cited them; `jrnl summarize` writes them again\n", summaries)
	}
	if bs, _ := filepath.Glob(filepath.Join(s.dir, "backups", "*.db")); len(bs) > 0 {
		fmt.Fprintf(os.Stderr, "jrnl: %d backup(s) in %s may still hold them\n", len(bs), filepath.Join(s.dir, "backups"))
	}
	return nil
}

func cmdShow(args []string) error {
	fs, project := newFlags("show")
	ids, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(ids) == 0 {
		return fmt.Errorf("show: give at least one entry ID")
	}
	s, _, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	for i, p := range ids {
		id, err := s.Resolve(p)
		if err != nil {
			return err
		}
		e, err := s.Get(id)
		if err != nil {
			return err
		}
		if i > 0 {
			fmt.Println()
		}
		printEntry(e)
	}
	return nil
}

// printEntry writes the entry as markdown: its metadata in a YAML front
// matter block, then the body exactly as it was written.
func printEntry(e Entry) {
	var b strings.Builder
	field := func(k, v string) { fmt.Fprintf(&b, "%s: %s\n", k, v) }
	list := func(k string, vs []string) {
		if len(vs) > 0 {
			field(k, "["+strings.Join(vs, ", ")+"]")
		}
	}
	b.WriteString("---\n")
	field("id", e.ID)
	field("title", yamlString(e.Title))
	field("time", fmt.Sprintf("%s (%s)", localTime(e.TS), ago(e.TS)))
	field("kind", e.Kind)
	field("author", yamlString(e.Author))
	if e.Status != "" {
		field("status", e.Status)
	}
	if e.Status == "closed" {
		field("closed", fmt.Sprintf("%s (%s)", localTime(e.ClosedAt), ago(e.ClosedAt)))
		if e.ClosedBy != "" {
			field("closed_by", e.ClosedBy)
		}
	}
	list("tags", e.Tags)
	list("supersedes", e.Supersedes)
	list("superseded_by", e.SupersededBy)
	list("links", e.Links)
	list("linked_from", e.LinkedFrom)
	files := make([]string, len(e.Files))
	for i, f := range e.Files {
		files[i] = yamlString(f)
	}
	list("files", files)
	if e.Branch != "" {
		field("branch", yamlString(e.Branch))
	}
	if e.Commit != "" {
		field("commit", e.Commit)
	}
	if e.Session != "" {
		field("session", yamlString(e.Session))
	}
	if len(e.Assets) > 0 {
		b.WriteString("attachments:\n")
		for _, a := range e.Assets {
			fmt.Fprintf(&b, "  - path: %s\n    name: %s\n    type: %s\n", yamlString(a.Path), yamlString(a.Name), a.Mime)
			if a.Caption != "" {
				fmt.Fprintf(&b, "    caption: %s\n", yamlString(a.Caption))
			}
			if a.Description != "" {
				fmt.Fprintf(&b, "    description: %s\n", yamlString(oneLine(a.Description)))
			}
		}
	}
	b.WriteString("---\n")
	if body := strings.TrimSpace(e.Body); body != "" {
		b.WriteString("\n" + body + "\n")
	}
	fmt.Print(b.String())
}

// yamlString quotes free text for the front matter. A JSON string is valid
// YAML, and quoting every value keeps titles like "true", "null" or "2026"
// from reading as a bool, null or number.
func yamlString(s string) string {
	q, _ := json.Marshal(s)
	return string(q)
}

func summaryLine(e Entry) string {
	line := fmt.Sprintf("%s  %-8s  %-9s %s", e.ID, ago(e.TS), e.Kind, e.Title)
	if e.Status != "" {
		line += "  [" + e.Status + "]"
	}
	for _, t := range e.Tags {
		line += " #" + t
	}
	if n := len(e.Assets); n > 0 {
		line += fmt.Sprintf("  [%d attachment(s)]", n)
	}
	return line
}

func cmdList(args []string) error {
	fs, project := newFlags("list")
	f, limit := filterFlags(fs)
	oldest := fs.Bool("oldest", false, "")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := resolveFilter(s, f, os.Getenv("CLAUDE_CODE_SESSION_ID")); err != nil {
		return err
	}
	es, err := s.List(*f, *limit, *oldest)
	if err != nil {
		return err
	}
	for _, e := range es {
		fmt.Println(summaryLine(e))
	}
	return nil
}

func cmdSessions(args []string) error {
	fs, project := newFlags("sessions")
	f, limit := filterFlags(fs)
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	ss, err := s.Sessions(*f, *limit)
	if err != nil {
		return err
	}
	me := os.Getenv("CLAUDE_CODE_SESSION_ID")
	for _, si := range ss {
		mark := ""
		if si.ID == me {
			mark = "  (this session)"
		}
		fmt.Printf("%s  %-8s  %3d entries  last: %s%s\n", si.ID, ago(si.Last), si.Entries, si.LastTitle, mark)
	}
	return nil
}

func cmdStats(args []string) error {
	fs, project := newFlags("stats")
	f, _ := filterFlags(fs)
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := resolveFilter(s, f, os.Getenv("CLAUDE_CODE_SESSION_ID")); err != nil {
		return err
	}
	total, first, last, by, err := s.Stats(*f)
	if err != nil {
		return err
	}
	fmt.Printf("%d entries", total)
	if total > 0 {
		fmt.Printf(", %s → %s", localTime(first), localTime(last))
	}
	fmt.Println()
	for _, name := range []string{"kind", "tag", "author"} {
		var parts []string
		for _, c := range by[name] {
			parts = append(parts, fmt.Sprintf("%s %d", c.Key, c.N))
		}
		if len(parts) > 0 {
			fmt.Printf("by %s: %s\n", name, strings.Join(parts, ", "))
		}
	}
	return nil
}

func cmdSearch(args []string) error {
	fs, project := newFlags("search")
	f, limit := filterFlags(fs)
	like := fs.String("like", "", "")
	newest := fs.Bool("newest", false, "")
	oldest := fs.Bool("oldest", false, "")
	fast := fs.Bool("fast", false, "")
	minScore := fs.Float64("min-score", -1, "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	req := query{Q: strings.TrimSpace(strings.Join(pos, " ")), Filter: *f, Limit: *limit, Fast: *fast,
		MinScore: *minScore, Me: os.Getenv("CLAUDE_CODE_SESSION_ID")}
	if req.Q == "" && *like == "" {
		return fmt.Errorf("search: give a query, or --like FILE")
	}
	switch {
	case *newest && *oldest:
		return fmt.Errorf("search: pick one of --newest and --oldest")
	case *newest:
		req.Order = "newest"
	case *oldest:
		req.Order = "oldest"
	}
	ai, err := NewAI()
	if err != nil {
		return err
	}
	if *like != "" {
		return searchLike(ai, *project, *like, req)
	}
	// A running jrnl serve answers from memory, and keeps the reranker's
	// connection warm: the TLS handshake alone took 350-520 ms here.
	req.Project = *project
	if req.Project == "" {
		req.Project = detectGit().project
	}
	if _, err := os.Stat(filepath.Join(journalHome(), req.Project, "journal.db")); err != nil {
		return noJournal{req.Project}
	}
	if err := req.Filter.pinDates(); err != nil {
		return err
	}
	if res, ok, err := serverQuery(ai, req); err != nil {
		return err
	} else if ok {
		printWarnings(res.Warn)
		printHits(os.Stdout, res.Hits)
		return nil
	}
	// No server: the same ranking, from the journal read into memory
	// (about 8 ms per 1,000 entries).
	s, err := Open(req.Project)
	if err != nil {
		return err
	}
	defer s.Close()
	idx, err := buildMemIndex(s, ai.embedID)
	if err != nil {
		return err
	}
	hits, hasVec, warns, err := rankHits(s, ai, idx, req)
	if err != nil {
		return err
	}
	hits, w := finishHits(ai, req, hits, hasVec)
	printWarnings(append(warns, w...))
	printHits(os.Stdout, hits)
	return nil
}

// query is one text search, as a command sends it to jrnl serve.
type query struct {
	// The client's version and settings: a server answers only when they
	// match its own, so a stale server never ranks with other settings.
	Version, Settings string
	Project, Q        string
	Filter            Filter
	Limit             int
	Order             string // "", newest or oldest
	Fast              bool
	MinScore          float64
	Me                string // the caller's session, for --session last
}

type queryResult struct {
	Hits []Hit
	Warn []string
}

func printWarnings(warns []string) {
	for _, w := range warns {
		fmt.Fprintln(os.Stderr, "jrnl: "+w)
	}
}

// rankHits embeds the question and ranks the journal: everything that
// reads the store. The rest of a search (finishHits) needs no store, so jrnl
// serve runs it outside its lock.
func rankHits(s *Store, ai *AI, idx *memIndex, r query) (hits []Hit, hasVec bool, warns []string, err error) {
	f := r.Filter
	if err := resolveFilter(s, &f, r.Me); err != nil {
		return nil, false, nil, err
	}
	var qvec []float32
	vecs, aiErr := ai.Interactive(embedDeadline).Embed([]Input{{Text: r.Q, Query: true}})
	if aiErr == nil {
		qvec = vecs[0]
	} else if !awaitingSetup(s, aiErr) {
		// Without embeddings set up, keyword search is how jrnl works; hits
		// are marked (keyword). Say so only when it finds nothing.
		warns = append(warns, fmt.Sprintf("keyword search only: %v", aiErr))
	}
	only, err := s.allowed(f)
	if err != nil {
		return nil, false, nil, err
	}
	cands := idx.rank(r.Q, qvec, only, r.Order)
	fast := r.Fast && r.Order == ""
	if fast {
		g := gateFor(ai.embedModel)
		if r.MinScore >= 0 {
			g = gate{float32(r.MinScore), float32(r.MinScore)}
		}
		cands = gateCut(cands, qvec != nil, g, r.Limit)
	}
	if hits, err = s.hits(cands, r.Q); err != nil {
		return nil, false, nil, err
	}
	if fast {
		return hits, qvec != nil, warns, nil
	}
	if len(hits) == 0 && aiErr != nil && awaitingSetup(s, aiErr) {
		warns = append(warns, "no keyword match; search by meaning is off (see `jrnl doctor`)")
	}
	if qvec != nil && r.MinScore > 0 {
		hits = slices.DeleteFunc(hits, func(h Hit) bool { return float64(h.Sim) < r.MinScore })
	}
	return hits, qvec != nil, warns, nil
}

// finishHits reranks, orders and cuts ranked hits.
func finishHits(ai *AI, r query, hits []Hit, hasVec bool) ([]Hit, []string) {
	if r.Fast && r.Order == "" {
		return hits, nil
	}
	var warns []string
	if hasVec && r.Q != "" && len(hits) > 1 {
		k := min(r.Limit, 20)
		if r.Order != "" {
			k = 0 // every relevant entry; sorted by date below
		}
		var w string
		if hits, w = rerank(ai.Interactive(rerankDeadline), r.Q, hits, k); w != "" {
			warns = append(warns, w)
		}
	}
	if r.Order != "" {
		// The reranker's picks are the relevant ones; show them in date
		// order. Without a reranker, all candidates in date order.
		var picked []Hit
		for _, h := range hits {
			if slices.Contains(h.Via, "reranked") {
				picked = append(picked, h)
			}
		}
		if len(picked) > 0 {
			hits = picked
		}
		sort.SliceStable(hits, func(i, j int) bool {
			if r.Order == "oldest" {
				return hits[i].Entry.TS < hits[j].Entry.TS
			}
			return hits[i].Entry.TS > hits[j].Entry.TS
		})
	}
	if len(hits) > r.Limit {
		hits = hits[:r.Limit]
	}
	return hits, warns
}

// searchLike finds attachments like a file. --like ranks media of the same
// kind by their vectors when a media embedder is set up (vectors compare
// well within one modality, not across them). Without one, it describes the
// file the way attachments are described and ranks the descriptions most
// like it.
func searchLike(ai *AI, project, like string, r query) error {
	s, _, err := open(project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := resolveFilter(s, &r.Filter, r.Me); err != nil {
		return err
	}
	a := Asset{Path: like, Name: filepath.Base(like), Mime: mediaTypes[strings.ToLower(filepath.Ext(like))]}
	if a.Modality() == "" {
		return fmt.Errorf("--like needs an image, video or audio file")
	}
	in, modalities, embedder := Input{File: a.Path, Mime: a.Mime}, []string{a.Modality()}, ai.mediaID
	if ai.mediaErr != nil {
		desc, err := ai.Describe(a)
		if err != nil {
			return fmt.Errorf("--like: %w", err)
		}
		in, modalities, embedder = captionInput(a.Name, "", desc), []string{"caption"}, ai.embedID
	}
	vecs, err := ai.Embed([]Input{in})
	if err != nil {
		return err
	}
	cands, err := nearest(vecs[0], s.chunkVecs(embedder, modalities, r.Filter))
	if err != nil {
		return err
	}
	hits, err := s.hits(cands, "")
	if err != nil {
		return err
	}
	if r.MinScore > 0 {
		hits = slices.DeleteFunc(hits, func(h Hit) bool { return float64(h.Sim) < r.MinScore })
	}
	r.Q, r.Fast = "", false
	hits, _ = finishHits(ai, r, hits, true)
	printHits(os.Stdout, hits)
	return nil
}

func printHits(w io.Writer, hits []Hit) {
	for _, h := range hits {
		fmt.Fprintf(w, "%s  (%s)\n", summaryLine(h.Entry), strings.Join(h.Via, "+"))
		if h.Snippet != "" {
			fmt.Fprintf(w, "    %s\n", oneLine(h.Snippet))
		} else if h.Entry.Body != "" {
			fmt.Fprintf(w, "    %s\n", truncate(oneLine(h.Entry.Body), 160))
		}
		if h.Match != "" {
			fmt.Fprintf(w, "    matched %s\n", h.Match)
		}
	}
}

func cmdReindex(args []string) error {
	fs, project := newFlags("reindex")
	prune := fs.Bool("prune", false, "")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	ai, err := NewAI()
	if err != nil {
		return err
	}
	ai.download = true // reindex is how the embedding model gets downloaded
	r, err := process(s, ai, "", false)
	if err != nil {
		return err
	}
	fmt.Printf("described %d file(s), embedded %d pending chunk(s)", r.described, r.embedded)
	if r.failed > 0 {
		fmt.Printf(", %d can never be processed", r.failed)
	}
	if r.waiting > 0 {
		fmt.Printf(", %d still pending (%v)", r.waiting, r.lastErr)
	}
	fmt.Println()
	if *prune {
		unused, err := s.Unused(ai.embedID, ai.mediaID)
		if err != nil {
			return err
		}
		if err := s.Prune(ai.embedID, ai.mediaID); err != nil {
			return err
		}
		for _, u := range unused {
			fmt.Printf("removed %d vector(s) from %s\n", u.N, u.Key)
		}
	}
	return nil
}

func cmdBackup(args []string) error {
	fs, project := newFlags("backup")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	dst, err := s.Backup()
	if err != nil {
		return err
	}
	fmt.Println(dst)
	return nil
}

func cmdWhere(args []string) error {
	fs, project := newFlags("where")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	p := *project
	if p == "" {
		p = detectGit().project
	}
	fmt.Println(filepath.Join(journalHome(), p))
	return nil
}

type gitInfo struct{ project, branch, commit string }

func git(args ...string) string {
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// detectGit names the project after its origin remote, so every worktree and
// clone of one repository shares a journal. It reads .git itself: each git
// process costs 70-180 ms on a Mac with endpoint security, three of them per
// command. A .git file (a linked worktree or a submodule) points at the
// repository's git directory. Only what it can't read falls back to git.
func detectGit() gitInfo {
	wd, _ := os.Getwd()
	for dir := wd; ; dir = filepath.Dir(dir) {
		gd := filepath.Join(dir, ".git")
		if st, err := os.Stat(gd); err == nil {
			if st.IsDir() {
				return readGitDir(dir, gd, gd)
			}
			if gd, common, ok := linkedGitDir(dir, gd); ok {
				return readGitDir(dir, gd, common)
			}
			break
		}
		if filepath.Dir(dir) == dir {
			break
		}
	}
	g := gitInfo{branch: git("branch", "--show-current"), commit: git("rev-parse", "--short", "HEAD")}
	if r := git("remote", "get-url", "origin"); r != "" {
		g.project = slugFromRemote(r)
	} else if top := git("rev-parse", "--show-toplevel"); top != "" {
		g.project = slug(filepath.Base(top))
	} else {
		g.project = slug(filepath.Base(wd))
	}
	return g
}

// linkedGitDir follows a .git file ("gitdir: <path>") to the worktree's own
// git directory, and its commondir file to the directory shared by every
// worktree, which holds config and refs. A submodule has no commondir.
func linkedGitDir(top, file string) (gd, common string, ok bool) {
	b, err := os.ReadFile(file)
	if err != nil {
		return "", "", false
	}
	gd, ok = strings.CutPrefix(strings.TrimSpace(string(b)), "gitdir: ")
	if !ok {
		return "", "", false
	}
	if !filepath.IsAbs(gd) {
		gd = filepath.Join(top, gd)
	}
	common = gd
	if c, err := os.ReadFile(filepath.Join(gd, "commondir")); err == nil {
		common = strings.TrimSpace(string(c))
		if !filepath.IsAbs(common) {
			common = filepath.Join(gd, common)
		}
	}
	if _, err := os.Stat(filepath.Join(gd, "HEAD")); err != nil {
		return "", "", false
	}
	return gd, common, true
}

// readGitDir reads the origin URL, branch and commit of the checkout at top:
// HEAD from its git directory gd, config and refs from common.
func readGitDir(top, gd, common string) gitInfo {
	g := gitInfo{project: slug(filepath.Base(top))}
	if url := originURL(common); url != "" {
		g.project = slugFromRemote(url)
	}
	head, _ := os.ReadFile(filepath.Join(gd, "HEAD"))
	ref, isRef := strings.CutPrefix(strings.TrimSpace(string(head)), "ref: ")
	sha := ref
	if isRef {
		g.branch = strings.TrimPrefix(ref, "refs/heads/")
		if sha = resolveRef(gd, ref); sha == "" && common != gd {
			sha = resolveRef(common, ref)
		}
	}
	if len(sha) >= 7 {
		g.commit = sha[:7]
	}
	return g
}

// originURL is the url of [remote "origin"] in the repository's config.
func originURL(gd string) string {
	b, err := os.ReadFile(filepath.Join(gd, "config"))
	if err != nil {
		return ""
	}
	inOrigin := false
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") {
			inOrigin = line == `[remote "origin"]`
			continue
		}
		if k, v, ok := strings.Cut(line, "="); inOrigin && ok && strings.TrimSpace(k) == "url" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

// resolveRef finds the commit a ref names, loose or in packed-refs; "" for
// a branch with no commits yet.
func resolveRef(gd, ref string) string {
	if b, err := os.ReadFile(filepath.Join(gd, filepath.FromSlash(ref))); err == nil {
		return strings.TrimSpace(string(b))
	}
	b, _ := os.ReadFile(filepath.Join(gd, "packed-refs"))
	for _, line := range strings.Split(string(b), "\n") {
		if sha, name, ok := strings.Cut(strings.TrimSpace(line), " "); ok && name == ref {
			return sha
		}
	}
	return ""
}

func slugFromRemote(r string) string {
	r = strings.TrimSuffix(strings.TrimSuffix(r, "/"), ".git")
	parts := strings.FieldsFunc(r, func(c rune) bool { return c == '/' || c == ':' })
	if len(parts) >= 2 {
		return slug(parts[len(parts)-2] + "-" + parts[len(parts)-1])
	}
	return slug(r)
}

func slug(s string) string {
	var b strings.Builder
	dash := false
	for _, c := range strings.ToLower(s) {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			b.WriteRune(c)
			dash = false
		} else if !dash && b.Len() > 0 {
			b.WriteByte('-')
			dash = true
		}
	}
	return strings.TrimSuffix(b.String(), "-")
}

func defaultAuthor() string {
	if os.Getenv("CLAUDECODE") != "" {
		return "claude"
	}
	return "me"
}

func firstLine(body string) string {
	line, _, _ := strings.Cut(body, "\n")
	line = strings.TrimSpace(strings.TrimLeft(line, "# "))
	if line == "" {
		return "untitled"
	}
	return truncate(line, 80)
}

func oneLine(s string) string { return strings.Join(strings.Fields(s), " ") }

func localTime(ts string) string {
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return ts
	}
	return t.Local().Format("2006-01-02 15:04")
}

func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

// rerank reorders hits with Gemma; the ones it picks come first, marked
// "reranked", then the rest in fused order. If the call fails, the fused
// order stands.
// k is how many best entries to ask for; 0 asks for every relevant one.
func rerank(ai *AI, q string, hits []Hit, k int) ([]Hit, string) {
	// Entries with identical text are shown to the reranker once.
	var items []string
	var itemHit []int
	seen := map[string]bool{}
	for i, h := range hits {
		text := truncate(oneLine(h.Entry.Body), 400)
		for _, a := range h.Entry.Assets {
			if a.Description != "" {
				text += " | attachment: " + truncate(oneLine(a.Description), 200)
			}
		}
		if !seen[text] {
			seen[text] = true
			items = append(items, text)
			itemHit = append(itemHit, i)
		}
	}
	order, err := ai.Rerank(q, items, k)
	if errors.Is(err, errNoChat) {
		return hits, "" // not set up; `jrnl doctor` says so
	}
	if err != nil {
		return hits, fmt.Sprintf("not reranked: %v", err)
	}
	if len(order) == 0 {
		return hits, "no entry clearly answers this; showing the closest matches"
	}
	out := make([]Hit, 0, len(hits))
	picked := map[int]bool{}
	for _, it := range order {
		i := itemHit[it]
		h := hits[i]
		h.Via = append(h.Via, "reranked")
		out = append(out, h)
		picked[i] = true
	}
	for i, h := range hits {
		if !picked[i] {
			out = append(out, h)
		}
	}
	return out, ""
}

func cmdClose(args []string) error {
	fs, project := newFlags("close")
	note := fs.String("note", "", "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	return setStatus(*project, pos, "closed", *note)
}

func cmdReopen(args []string) error {
	fs, project := newFlags("reopen")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	return setStatus(*project, pos, "open", "")
}

// setStatus closes or reopens entries. A closing note becomes its own entry
// linked to them, so how something was resolved stays searchable.
func setStatus(project string, prefixes []string, status, note string) error {
	if len(prefixes) == 0 {
		return fmt.Errorf("give at least one entry ID")
	}
	s, g, err := open(project, false)
	if err != nil {
		return err
	}
	defer s.Close()
	var ids, titles []string
	for _, p := range prefixes {
		id, err := s.Resolve(p)
		if err != nil {
			return err
		}
		e, err := s.Get(id)
		if err != nil {
			return err
		}
		ids, titles = append(ids, id), append(titles, e.Title)
	}
	closedBy := ""
	if note != "" {
		ne := Entry{
			ID: newID(), TS: utc(now()), Project: g.project, Session: os.Getenv("CLAUDE_CODE_SESSION_ID"),
			Author: defaultAuthor(), Kind: "log", Branch: g.branch, Commit: g.commit,
			Title: "Closed: " + truncate(strings.Join(titles, "; "), 70), Body: note, Closes: ids,
		}
		if err := s.Insert(ne, []Chunk{{Modality: "text"}}); err != nil {
			return err
		}
		if ai, err := NewAI(); err == nil {
			if _, err := process(s, ai, ne.ID, true); err != nil {
				return err
			}
		}
		closedBy = ne.ID
	}
	if err := s.SetStatus(ids, status, closedBy); err != nil {
		return err
	}
	for i, id := range ids {
		fmt.Printf("%s  %s  %s\n", id, status, titles[i])
	}
	return nil
}

// cmdDoctor shows which optional parts of jrnl are on and how to turn on the
// rest. It never prints a credential, and it exits 0 when parts are off:
// a journal with none of them is still a working journal.
func cmdDoctor(args []string) error {
	fs, project := newFlags("doctor")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	ai, err := NewAI()
	if err != nil {
		return err
	}
	// Works before the first entry too: it is how setup is checked.
	s, g, err := open(*project, false)
	var none noJournal
	if err != nil && !errors.As(err, &none) {
		return err
	}
	total, waiting := 0, 0
	var unused []Count
	if s != nil {
		defer s.Close()
		if unused, err = s.Unused(ai.embedID, ai.mediaID); err != nil {
			return err
		}
		if total, _, _, _, err = s.Stats(Filter{}); err != nil {
			return err
		}
		if waiting, err = s.Waiting(ai.embedID, ai.mediaID); err != nil {
			return err
		}
	}
	row := func(name string, on bool, what, howTo string) {
		state := "on "
		if !on {
			state = "off"
		}
		fmt.Printf("%-10s %s  %s\n", name, state, what)
		if howTo != "" {
			label := "to turn on:"
			if on {
				label = "using:"
			}
			fmt.Printf("%-10s      %s %s\n", "", label, howTo)
		}
	}
	fmt.Printf("journal    %s: %d entries", filepath.Join(journalHome(), g.project), total)
	if s == nil {
		fmt.Print(" (`jrnl add` starts it)")
	}
	if waiting > 0 {
		fmt.Printf(", %d item(s) waiting for `jrnl reindex`", waiting)
	}
	fmt.Println()
	for _, u := range unused {
		fmt.Printf("%-10s %d vector(s) from %s, which is not set up now; `jrnl reindex --prune` deletes them\n", "", u.N, u.Key)
	}
	row("basics", true, "add, list, show, keyword search, stats, close", "")
	if p, err := exec.LookPath("jrnl"); err == nil {
		if isMiseShim(p) {
			fmt.Printf("%-10s %s is a mise shim: it runs mise before every call (100-300 ms).\n", "speed", p)
			fmt.Printf("%-10s      put the real binary on PATH: mise settings set activate_shims false,\n", "")
			fmt.Printf("%-10s      and mise activate in your shell (see the README)\n", "")
		}
	}

	// where says which setting chose a role, when it wasn't a default.
	where := func(role string) string {
		if f := ai.from[role]; f != "" {
			return " (" + f + ")"
		}
		return ""
	}
	if _, err := os.Stat(configFile()); err == nil {
		fmt.Printf("%-10s %s\n", "config", configFile())
	}
	embedOn, how := embedStatus(ai)
	if embedOn {
		how += where("embed")
	}
	row("embedding", embedOn, "search by meaning, not only exact words", how)
	how = "[chat] in " + configFile() + ", or JRNL_CHAT_URL, JRNL_CHAT_KEY, JRNL_CHAT_MODEL"
	if ai.chatErr == nil {
		how = ai.chatModel + " at " + ai.chatURL + where("chat")
	}
	row("chat", ai.chatErr == nil, "reranking, summaries, captions for images and video", how)
	if ai.rerankURL != "" {
		row("reranker", true, "ranks search results; chat still answers --newest and --oldest",
			ai.rerankModel+" at "+ai.rerankURL+where("rerank"))
	}
	how = "[media_embed] with a cloudflare provider in config.toml; until then --like compares descriptions"
	mediaOn := ai.mediaErr == nil
	if mediaOn {
		if _, run, err := ai.media.cloudflareURLs(); err != nil {
			mediaOn, how = false, err.Error()
		} else {
			how = ai.media.model + " at " + run + where("media_embed")
		}
	}
	row("--like", mediaOn, "find attachments that look or sound alike", how)
	_, ffErr := exec.LookPath("ffmpeg")
	how = ""
	if ffErr != nil {
		how = "install ffmpeg"
	}
	row("ffmpeg", ffErr == nil, "reading video and audio attachments", how)
	_, miseErr := exec.LookPath("mise")
	_, uvxErr := exec.LookPath("uvx")
	appleSilicon := runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
	parakeetOn := appleSilicon && (miseErr == nil || uvxErr == nil)
	how = ""
	if !parakeetOn {
		how = "install mise or uv (runs parakeet-mlx)"
		if !appleSilicon {
			how = "needs a Mac with Apple silicon"
		}
	}
	row("parakeet", parakeetOn, "transcripts of audio and video", how)
	running, emb, ok := runningServer(socketPath())
	how = "`jrnl serve --background`; jrnl add starts it when entries wait for the embedding model"
	if ok {
		how = "server " + running + " running at " + socketPath()
		if running == version && emb != ai.embedID {
			how += ", embedding with " + emb + " (started under other settings)"
		}
	}
	row("serve", ok, "searches from memory in milliseconds, for coding-agent hooks", how)
	if waiting > 0 {
		fmt.Println("\nAfter turning a part on, run `jrnl reindex` to index what was saved before.")
	}
	return nil
}

// embedStatus embeds one word, which works the same on every provider.
func embedStatus(ai *AI) (bool, string) {
	if ai.local && !localDownloaded() {
		return false, "run `jrnl reindex` once: it downloads " + localModel + " (" + localSize + ") and embeds what waited"
	}
	_, err := ai.Interactive(10 * time.Second).embedText([]string{"jrnl doctor"})
	if err == nil && ai.local {
		return true, localModel + " in this binary (" + localDir() + ")"
	}
	if err == nil {
		return true, ai.embedModel + " at " + ai.embedURL
	}
	var down embedDown
	if errors.As(err, &down) && !down.set {
		return false, "install Ollama (ollama.com), run `ollama serve`, then `ollama pull " + ollamaModel +
			"`; or set [embed] in config.toml (or JRNL_EMBED_URL, JRNL_EMBED_MODEL) for any OpenAI-compatible API"
	}
	return false, err.Error()
}

// isMiseShim says the jrnl at p runs mise first: a link to the mise binary,
// or a script that calls it.
func isMiseShim(p string) bool {
	real, err := filepath.EvalSymlinks(p)
	if err != nil {
		return false
	}
	if filepath.Base(real) == "mise" {
		return true
	}
	f, err := os.Open(real)
	if err != nil {
		return false
	}
	defer f.Close()
	head := make([]byte, 512)
	n, _ := io.ReadFull(f, head)
	return strings.HasPrefix(string(head[:n]), "#!") && miseCommand.Match(head[:n])
}

// miseCommand is a call to the mise binary: "mise x", "/path/to/mise exec",
// not a path through mise's install folder.
var miseCommand = regexp.MustCompile(`(^|[\s/"'])mise(\s|["']|$)`)
