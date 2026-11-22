package main

// Tests drive the real jrnl binary against a fake Workers AI server, the same
// boundary Claude uses.

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/knights-analytics/hugot/backends"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

var bin string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "jrnl-test-bin")
	if err != nil {
		panic(err)
	}
	bin = filepath.Join(dir, "jrnl")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		panic(string(out))
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// fakeAI stands in for the two Workers AI models jrnl calls.
//
// Embeddings are hand-built on concept axes, so a text and an image can be
// "about" the same thing without sharing a word:
//
//	axis 0: signing in   (words: sign, login, authentication, credentials)
//	axis 1: images       (words: screenshot, picture)
//	axis 9: anything else
//
// Queries point straight along their concept. Documents only half-match
// theirs (weight 0.5, the rest on a noise axis), and every raw image is a
// "hub" that sits 0.577 from every concept. That reproduces what Workers AI
// measured: a meaningless tone outscored real text matches for every query.
//
// Captions (Gemma) describe a fake image as "A screenshot showing <bytes after
// the PNG magic>".
type fakeAI struct {
	fail         atomic.Bool
	unauth       atomic.Bool  // answer 401, like a revoked token
	maxMedia     atomic.Int64 // largest raw media payload sent for embedding
	rerankFail   atomic.Bool  // reranking calls answer 500
	rerankDelay  atomic.Int64 // reranking calls stall this many seconds first
	summaryCalls atomic.Int64
	rerankCalls  atomic.Int64 // calls to the /rerank API
	mu           sync.Mutex
	texts        []string
	dims         []int
	embedModels  []string         // model named in each text embedding request
	chatReqs     []map[string]any // each chat request body, plus "auth"
	requests     atomic.Int64
}

func (f *fakeAI) vector(content any) []float32 {
	v := make([]float32, 256)
	if parts, ok := content.([]any); ok {
		// Raw media: the hub vector.
		b, _ := json.Marshal(parts)
		if n := int64(len(b)); n > f.maxMedia.Load() {
			f.maxMedia.Store(n)
		}
		v[0], v[1], v[9] = 0.577, 0.577, 0.577
		return v
	}
	text := strings.ToLower(content.(string))
	f.mu.Lock()
	f.texts = append(f.texts, content.(string))
	f.mu.Unlock()
	axis := 9
	for _, w := range []string{"sign", "login", "authentication", "credentials"} {
		if strings.Contains(text, w) {
			axis = 0
		}
	}
	for _, w := range []string{"screenshot", "picture"} {
		if strings.Contains(text, w) {
			axis = 1
		}
	}
	if strings.HasPrefix(text, "task: search result") {
		v[axis] = 1
		return v
	}
	v[axis], v[20+len(text)%200] = 0.5, 0.866
	return v
}

func (f *fakeAI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.requests.Add(1)
	// fail and unauth model cloud trouble; Ollama is local and unaffected.
	if r.URL.Path == "/v1/rerank" { // Cohere-style; "voyage" models answer like Voyage
		var req struct {
			Model     string   `json:"model"`
			Query     string   `json:"query"`
			Documents []string `json:"documents"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		f.rerankCalls.Add(1)
		var scores []map[string]any
		for i, d := range req.Documents {
			score := 0.1
			if strings.Contains(d, "rerank-me") {
				score = 0.9
			}
			scores = append(scores, map[string]any{"index": i, "relevance_score": score})
		}
		key := "results"
		if strings.HasPrefix(req.Model, "voyage") {
			key = "data"
		}
		json.NewEncoder(w).Encode(map[string]any{key: scores})
		return
	}
	if r.URL.Path == "/v1/embeddings" { // Ollama's OpenAI-compatible API
		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		f.mu.Lock()
		f.embedModels = append(f.embedModels, req.Model)
		f.mu.Unlock()
		var data []map[string]any
		for i, t := range req.Input {
			data = append(data, map[string]any{"index": i, "embedding": f.vector(t)})
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data})
		return
	}
	if f.fail.Load() {
		http.Error(w, "boom", 500)
		return
	}
	if f.unauth.Load() {
		http.Error(w, `{"success":false,"errors":[{"message":"Authentication error"}]}`, 401)
		return
	}
	reply := func(result any) {
		json.NewEncoder(w).Encode(map[string]any{"success": true, "result": result})
	}
	switch {
	case strings.HasSuffix(r.URL.Path, "/v1/chat/completions"):
		reply := func(result any) { json.NewEncoder(w).Encode(result) }
		body, _ := io.ReadAll(r.Body)
		var sent map[string]any
		json.Unmarshal(body, &sent)
		sent["auth"] = r.Header.Get("Authorization")
		f.mu.Lock()
		f.chatReqs = append(f.chatReqs, sent)
		f.mu.Unlock()
		var req struct {
			Messages []struct {
				Content any `json:"content"`
			} `json:"messages"`
		}
		json.Unmarshal(body, &req)
		if prompt, ok := req.Messages[0].Content.(string); ok && strings.HasPrefix(prompt, "Write a short overview") {
			// Overview of a long period: one bullet per day it was given,
			// plus a day that doesn't exist, which jrnl must drop.
			f.summaryCalls.Add(1)
			var lines []string
			for _, m := range regexp.MustCompile(`(?m)^D(\d+) \(`).FindAllStringSubmatch(prompt, -1) {
				lines = append(lines, "- theme [D"+m[1]+"]")
			}
			lines = append(lines, "- invented [D99]")
			reply(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": strings.Join(lines, "\n")}}}})
			return
		}
		if prompt, ok := req.Messages[0].Content.(string); ok && (strings.HasPrefix(prompt, "Summarize") || strings.HasPrefix(prompt, "Combine")) {
			// Summaries: one bullet citing every [ID] the prompt contains,
			// so a test can see whether any entry was dropped.
			f.summaryCalls.Add(1)
			// One bullet per entry number in the prompt (or per citation in
			// part summaries being combined), plus a number that matches no
			// entry, which jrnl must drop.
			var lines []string
			for _, m := range regexp.MustCompile(`(?m)^\[(\d+)\]|- item \[(\d+)\]`).FindAllStringSubmatch(prompt, -1) {
				lines = append(lines, "- item ["+m[1]+m[2]+"]")
			}
			lines = append(lines, "- invented [99999]")
			reply(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": strings.Join(lines, "\n")}}}})
			return
		}
		if prompt, ok := req.Messages[0].Content.(string); ok {
			// Reranking: entries containing "rerank-me" go first, the rest
			// keep their order.
			if f.rerankFail.Load() {
				http.Error(w, "boom", 500)
				return
			}
			time.Sleep(time.Duration(f.rerankDelay.Load()) * time.Second)
			var first, rest []int
			for _, line := range strings.Split(prompt, "\n") {
				var i int
				if _, err := fmt.Sscanf(line, "[%d]", &i); err == nil {
					if strings.Contains(strings.ToLower(line), "rerank-me") {
						first = append(first, i)
					} else {
						rest = append(rest, i)
					}
				}
			}
			order, _ := json.Marshal(append(first, rest...))
			reply(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": string(order)}}}})
			return
		}
		parts := req.Messages[0].Content.([]any)
		url := parts[0].(map[string]any)["image_url"].(map[string]any)["url"].(string)
		raw, _ := base64.StdEncoding.DecodeString(url[strings.IndexByte(url, ',')+1:])
		caption := "A screenshot showing " + strings.TrimPrefix(string(raw), "\x89PNG ")
		reply(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": caption}}}})
		return
	case r.URL.Path != "/run/@cf/google/embeddinggemma-2":
		http.Error(w, "unknown model "+r.URL.Path, 404)
		return
	}
	var req struct {
		Input      []string           `json:"input"`
		Messages   [][]map[string]any `json:"messages"`
		Dimensions int                `json:"dimensions"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	// Text sent through "messages" embeds differently from text sent through
	// "input" (cosine 0.74-0.81 on the real API), so jrnl must never do it.
	for _, m := range req.Messages {
		if _, isText := m[0]["content"].(string); isText {
			http.Error(w, `{"success":false,"errors":[{"message":"text sent on the messages path"}]}`, 400)
			return
		}
	}
	// Like the real API rejecting an input it can't decode: any request
	// carrying a video fails as a whole with a 400.
	for _, m := range req.Messages {
		if parts, ok := m[0]["content"].([]any); ok && parts[0].(map[string]any)["type"] == "video_url" {
			http.Error(w, `{"success":false,"errors":[{"message":"cannot decode video"}]}`, 400)
			return
		}
	}
	f.mu.Lock()
	f.dims = append(f.dims, req.Dimensions)
	f.mu.Unlock()
	type item struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	var data []item
	for i, t := range req.Input {
		data = append(data, item{i, f.vector(t)})
	}
	for i, m := range req.Messages {
		data = append(data, item{i, f.vector(m[0]["content"])})
	}
	reply(map[string]any{"data": data})
}

type env struct {
	t      *testing.T
	home   string
	ai     *fakeAI
	url    string
	ollama string   // OLLAMA_HOST; the fake server unless a test changes it
	noChat bool     // no chat model set up, like a fresh install
	config string   // config.toml; by default Cloudflare media vectors from the fake server
	extra  []string // more environment, such as JRNL_CHAT_URL=...
	path   string   // PATH for jrnl, when a test hides tools such as ffmpeg
	dir    string   // working directory; when set, jrnl names the project itself
}

func newEnv(t *testing.T) *env {
	ai := &fakeAI{}
	srv := httptest.NewServer(ai)
	t.Cleanup(srv.Close)
	return &env{t: t, home: t.TempDir(), ai: ai, url: srv.URL, ollama: srv.URL, config: `
[providers.cf]
type = "cloudflare"
url  = "` + srv.URL + `"
key  = "env:JRNL_TEST_CF_KEY"

[media_embed]
provider = "cf"
model    = "@cf/google/embeddinggemma-2"
`}
}

func (e *env) run(stdin string, args ...string) (string, string, error) {
	cmd := exec.Command(bin, append(args, "--project", "test")...)
	if e.dir != "" {
		cmd = exec.Command(bin, args...)
		cmd.Dir = e.dir
	}
	cmd.Env = append(os.Environ(), "JRNL_HOME="+e.home, "OLLAMA_HOST="+e.ollama,
		// A fixed clock and zone: Thursday 2026-11-19, 15:30 in Berlin (UTC+1).
		"TZ=Europe/Berlin", "JRNL_NOW=2026-11-19T15:30:00+01:00", "CLAUDECODE=", "CLAUDE_CODE_SESSION_ID=sess-1")
	// This machine's own settings must not leak in.
	cmd.Env = append(cmd.Env, "JRNL_CHAT_URL="+e.url+"/v1", "JRNL_CHAT_KEY=chat-test", "JRNL_CHAT_MODEL=test/vision",
		// Embeddings from the fake, set up as Ollama at OLLAMA_HOST would be;
		// the local model is out of reach unless a test opts in.
		"JRNL_CHAT_EXTRA=", "JRNL_EMBED_URL="+e.ollama+"/v1", "JRNL_EMBED_KEY=", "JRNL_EMBED_MODEL=embeddinggemma-2:270m",
		"JRNL_MODEL_DIR="+filepath.Join(e.home, "no-model"), "JRNL_MODELS_URL=http://127.0.0.1:1",
		"JRNL_RERANK_URL=", "JRNL_RERANK_KEY=", "JRNL_RERANK_MODEL=", "JRNL_ENV_FILE="+filepath.Join(e.home, "no-env-file"),
		"JRNL_TEST_CF_KEY=cf-test")
	conf := filepath.Join(e.home, "config.toml")
	os.Remove(conf)
	if e.config != "" {
		os.WriteFile(conf, []byte(e.config), 0o644)
	}
	cmd.Env = append(cmd.Env, "JRNL_CONFIG="+conf)
	if e.noChat {
		cmd.Env = append(cmd.Env, "JRNL_CHAT_URL=", "JRNL_CHAT_KEY=", "JRNL_CHAT_MODEL=")
	}
	cmd.Env = append(cmd.Env, e.extra...)
	if e.path != "" {
		cmd.Env = append(cmd.Env, "PATH="+e.path)
	}
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	var out, errb strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &errb
	err := cmd.Run()
	return out.String(), errb.String(), err
}

func (e *env) must(stdin string, args ...string) string {
	e.t.Helper()
	out, errOut, err := e.run(stdin, args...)
	if err != nil {
		e.t.Fatalf("jrnl %v: %v\n%s", args, err, errOut)
	}
	return out
}

func firstID(out string) string { return strings.Fields(out)[0] }

// Many sessions writing to one project at once must not lose or reject a
// single entry, including the very first runs that create the database.
func TestParallelSessionsLoseNothing(t *testing.T) {
	e := newEnv(t)
	const procs, each = 20, 25
	var wg sync.WaitGroup
	errs := make(chan string, procs*each)
	for p := 0; p < procs; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				if _, errOut, err := e.run("", "add", "--kind", "log", fmt.Sprintf("session %d entry %d", p, i)); err != nil {
					errs <- errOut
				}
			}
		}(p)
	}
	wg.Wait()
	close(errs)
	failed := 0
	for msg := range errs {
		if failed == 0 {
			t.Errorf("first failure: %s", msg)
		}
		failed++
	}
	if failed > 0 {
		t.Fatalf("%d of %d adds failed", failed, procs*each)
	}
	out := e.must("", "list", "--limit", "100000")
	if n := strings.Count(out, "\n"); n != procs*each {
		t.Fatalf("listed %d entries, want %d", n, procs*each)
	}
}

func TestSearchFindsByKeywordAndByMeaning(t *testing.T) {
	e := newEnv(t)
	// No word in common with the query below; only meaning links them.
	auth := firstID(e.must("Users could not sign in after the token refresh.", "add", "--kind", "bug", "--tag", "auth"))
	busy := firstID(e.must("Writers hit SQLITE_BUSY when two sessions started together.", "add", "--kind", "bug"))
	firstID(e.must("Chose Go over Bun for the CLI.", "add", "--kind", "decision"))

	out := e.must("", "search", "authentication failure")
	if !strings.HasPrefix(out, auth) {
		t.Errorf("semantic query: want %s first, got:\n%s", auth, out)
	}
	out = e.must("", "search", "SQLITE_BUSY")
	if !strings.HasPrefix(out, busy) || !strings.Contains(strings.SplitN(out, "\n", 2)[0], "keyword") {
		t.Errorf("keyword query: want %s first via keyword, got:\n%s", busy, out)
	}
	// Filters apply to both rankings.
	out = e.must("", "search", "--kind", "decision", "authentication failure")
	if strings.Contains(out, auth) {
		t.Errorf("--kind decision returned a bug entry:\n%s", out)
	}
	// Spec: documents carry the document prefix, queries the query prefix,
	// and both use the same dimension.
	for _, txt := range e.ai.texts {
		if !strings.HasPrefix(txt, "title: ") && !strings.HasPrefix(txt, "task: search result | query: ") {
			t.Errorf("text sent without a task prefix: %q", txt)
		}
	}
	// With no settings, text goes to EmbeddingGemma on Ollama.
	if len(e.ai.embedModels) == 0 {
		t.Error("no text was embedded through Ollama")
	}
	for _, m := range e.ai.embedModels {
		if m != "embeddinggemma-2:270m" {
			t.Errorf("embedding request for %q, want embeddinggemma-2:270m", m)
		}
	}
}

func TestTextQueryFindsImage(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "broken-header.png")
	os.WriteFile(img, []byte("\x89PNG fake bytes"), 0o644)
	id := firstID(e.must("Header overlaps the nav on mobile.", "add", "--attach", img+"::header overlapping nav"))
	e.must("Bumped the Go version.", "add")

	out := e.must("", "search", "picture")
	first := strings.SplitN(out, "\n", 2)[0]
	if !strings.HasPrefix(first, id) {
		t.Fatalf("want %s first, got:\n%s", id, out)
	}
	if !strings.Contains(out, "matched image ") || !strings.Contains(out, filepath.Join(e.home, "test", "assets")) {
		t.Errorf("search should name the matching attachment's stored path:\n%s", out)
	}
	show := e.must("", "show", id[:12])
	if !strings.Contains(show, "header overlapping nav") {
		t.Errorf("show is missing the caption:\n%s", show)
	}
	if !strings.Contains(show, `description: "A screenshot showing fake bytes"`) {
		t.Errorf("show is missing the generated description:\n%s", show)
	}
}

// A Workers AI outage must not hold back text: the entry is searchable by
// meaning at once (Ollama embedded it), and its attachment is described and
// made searchable by reindex once the cloud is back.
func TestCloudOutageDoesNotBlockText(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "shot.png")
	os.WriteFile(img, []byte("\x89PNG settings page"), 0o644)
	e.ai.fail.Store(true)
	out, errOut, err := e.run("Login form rejects valid credentials.", "add", "--attach", img)
	if err != nil {
		t.Fatalf("add failed during outage: %v\n%s", err, errOut)
	}
	id := firstID(out)
	if !strings.Contains(errOut, "reindex") {
		t.Errorf("add should say the attachment waits for reindex: %q", errOut)
	}
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Fatalf("text not searchable by meaning during a cloud outage:\n%s", out)
	}
	if out := e.must("", "search", "picture"); strings.Contains(out, "matched image") {
		t.Fatalf("image matched before it was described:\n%s", out)
	}

	e.ai.fail.Store(false)
	if out := e.must("", "reindex"); !strings.Contains(out, "described 1 file(s)") {
		t.Fatalf("reindex: %s", out)
	}
	if out := e.must("", "search", "picture"); !strings.Contains(out, "matched image") {
		t.Fatalf("image not searchable after reindex:\n%s", out)
	}
}

// One attachment the API rejects must not leave the rest of the entry
// unsearchable, and reindex must not retry it forever.
func TestRejectedAttachmentDoesNotBlockOthers(t *testing.T) {
	e := newEnv(t)
	dir := t.TempDir()
	img, vid, txt := filepath.Join(dir, "shot.png"), filepath.Join(dir, "clip.mp4"), filepath.Join(dir, "notes.txt")
	os.WriteFile(img, []byte("\x89PNG fake"), 0o644)
	os.WriteFile(vid, []byte("fake mp4"), 0o644)
	os.WriteFile(txt, []byte("plain notes"), 0o644)

	out, errOut, err := e.run("Login button does nothing.", "add", "--attach", img, "--attach", vid, "--attach", txt)
	if err != nil {
		t.Fatalf("add: %v\n%s", err, errOut)
	}
	id := firstID(out)
	if !strings.Contains(errOut, "clip.mp4 can't be embedded") || !strings.Contains(errOut, "clip.mp4 can't be described") {
		t.Errorf("stderr should name the rejected video: %q", errOut)
	}
	if !strings.Contains(errOut, "notes.txt stored but not searchable") {
		t.Errorf("stderr should warn about the unsupported type: %q", errOut)
	}
	// Text (axis 0, "login") and image (axis 1) were still embedded.
	if out := e.must("", "search", "authentication"); !strings.HasPrefix(out, id) {
		t.Errorf("text not searchable after a rejected sibling:\n%s", out)
	}
	if out := e.must("", "search", "picture"); !strings.Contains(out, "matched image") {
		t.Errorf("image not searchable after a rejected sibling:\n%s", out)
	}
	before := e.ai.requests.Load()
	if out := e.must("", "reindex"); strings.TrimSpace(out) != "described 0 file(s), embedded 0 pending chunk(s)" {
		t.Errorf("reindex should have nothing left to try, got %q", out)
	}
	if n := e.ai.requests.Load() - before; n != 0 {
		t.Errorf("reindex sent %d requests for a permanently rejected input", n)
	}
}

// Raw media vectors sit close to every query (the fake's hub, the real
// model's tone). Default search must rank on text and descriptions only, and
// --like must still find media by media.
func TestRawMediaDoesNotDrownText(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "chart.png")
	os.WriteFile(img, []byte("\x89PNG quarterly chart"), 0o644)
	withImage := firstID(e.must("Bumped the Go version.", "add", "--attach", img))
	login := firstID(e.must("Users could not sign in after deploy.", "add"))

	out := e.must("", "search", "authentication")
	if !strings.HasPrefix(out, login) {
		t.Fatalf("text match should rank first, not the attachment hub:\n%s", out)
	}
	chats := len(e.ai.chatReqs)
	out = e.must("", "search", "--like", img)
	if !strings.HasPrefix(out, withImage) || strings.Contains(out, login) {
		t.Fatalf("--like should return only the entry with a similar image:\n%s", out)
	}
	if len(e.ai.chatReqs) != chats {
		t.Errorf("--like with a media embedder should compare vectors, not describe the file")
	}
}

// Without a media embedder, --like describes the file and matches
// attachment descriptions only: an entry whose text has the same words is
// not an attachment like it.
func TestSearchLike(t *testing.T) {
	e := newEnv(t)
	e.config = ""
	dir := t.TempDir()
	img, query := filepath.Join(dir, "chart.png"), filepath.Join(dir, "query.png")
	os.WriteFile(img, []byte("\x89PNG quarterly chart"), 0o644)
	os.WriteFile(query, []byte("\x89PNG quarterly chart"), 0o644)
	withImage := firstID(e.must("Bumped the Go version.", "add", "--attach", img))
	sameWords := firstID(e.must("A screenshot showing quarterly chart", "add"))

	out := e.must("", "search", "--like", query)
	if !strings.HasPrefix(out, withImage) || strings.Contains(out, sameWords) {
		t.Fatalf("--like should return only the entry with a similar attachment:\n%s", out)
	}
	e.noChat = true
	if _, errOut, err := e.run("", "search", "--like", query); err == nil || !strings.Contains(errOut, "no chat model") {
		t.Errorf("--like without a chat model should say why it can't run: %v %q", err, errOut)
	}
}

// A revoked token is not the input's fault. Entries written while it was bad
// must be processed once the token works again.
func TestBadTokenDoesNotPoisonEntries(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "login.png")
	os.WriteFile(img, []byte("\x89PNG login dialog"), 0o644)
	e.ai.unauth.Store(true)
	id := firstID(e.must("Session expired banner shows twice.", "add", "--attach", img))
	e.ai.unauth.Store(false)
	out := e.must("", "reindex")
	if !strings.Contains(out, "described 1 file(s)") || strings.Contains(out, "never") {
		t.Fatalf("reindex after the token was fixed: %q", out)
	}
	if out := e.must("", "search", "picture"); !strings.HasPrefix(out, id) {
		t.Fatalf("entry not searchable after the token was fixed:\n%s", out)
	}
}

// A huge file must not be base64'd into one embedding request (a 941 MB
// video peaked at 7.6 GB of memory), but must stay findable by description.
func TestHugeAttachmentSkipsRawVector(t *testing.T) {
	e := newEnv(t)
	big := filepath.Join(t.TempDir(), "huge.png")
	f, _ := os.Create(big)
	f.WriteString("\x89PNG giant diagram")
	f.Truncate(51 << 20) // sparse: cheap to create
	f.Close()
	out, errOut, err := e.run("Architecture overview.", "add", "--attach", big)
	if err != nil {
		t.Fatalf("add: %v\n%s", err, errOut)
	}
	if n := e.ai.maxMedia.Load(); n > 0 {
		t.Errorf("sent %d bytes of raw media for embedding; want none over the cap", n)
	}
	if got := e.must("", "search", "picture"); !strings.HasPrefix(got, firstID(out)) {
		t.Errorf("huge attachment should still be found by its description:\n%s", got)
	}
}

// Common words must not count as keyword matches: with OR between terms,
// "a beeping tone" made every entry containing "a" a keyword hit.
func TestStopwordsAreNotKeywordHits(t *testing.T) {
	e := newEnv(t)
	e.must("Fixed a crash in the parser.", "add")
	e.must("Moved the docs to a new folder.", "add")
	out := e.must("", "search", "what is the a of zzzqx")
	if strings.Contains(out, "keyword") {
		t.Fatalf("stopwords alone produced keyword hits:\n%s", out)
	}
}

// Search must follow the reranker's order, and fall back to keyword+vector
// order when the reranker fails.
func TestSearchFollowsRerankerAndSurvivesItsFailure(t *testing.T) {
	e := newEnv(t)
	login := firstID(e.must("Login rejects valid credentials after deploy.", "add"))
	other := firstID(e.must("Moved the docs folder (rerank-me).", "add"))

	out := e.must("", "search", "login credentials")
	if !strings.HasPrefix(out, other) || !strings.Contains(strings.SplitN(out, "\n", 2)[0], "reranked") {
		t.Fatalf("reranker's pick should come first, marked reranked:\n%s", out)
	}

	e.ai.rerankFail.Store(true)
	out, errOut, err := e.run("", "search", "login credentials")
	if err != nil {
		t.Fatalf("search failed when the reranker did: %v\n%s", err, errOut)
	}
	if !strings.HasPrefix(out, login) || !strings.Contains(errOut, "not reranked") {
		t.Fatalf("want fused order with a warning; stderr %q, got:\n%s", errOut, out)
	}
}

// A stalled or failing reranker must not hold a search hostage: no retries
// with backoff, and a deadline after which the keyword+vector order is used.
func TestSearchDoesNotWaitOnSlowReranker(t *testing.T) {
	e := newEnv(t)
	login := firstID(e.must("Login rejects valid credentials after deploy.", "add"))
	e.must("Moved the docs folder (rerank-me).", "add")

	e.ai.rerankDelay.Store(10)
	start := time.Now()
	out, errOut, err := e.run("", "search", "login credentials")
	took := time.Since(start)
	if err != nil {
		t.Fatalf("search: %v\n%s", err, errOut)
	}
	if took > 8*time.Second {
		t.Errorf("search waited %s on a stalled reranker", took.Round(time.Second))
	}
	if !strings.HasPrefix(out, login) || !strings.Contains(errOut, "not reranked") {
		t.Errorf("want fused order with a warning; stderr %q, got:\n%s", errOut, out)
	}

	e.ai.rerankDelay.Store(0)
	e.ai.rerankFail.Store(true)
	start = time.Now()
	e.must("", "search", "login credentials")
	if took := time.Since(start); took > 3*time.Second {
		t.Errorf("search retried a failing reranker for %s", took.Round(time.Second))
	}
}

// With the embeddings API down, entries are still saved and search answers
// by keyword, saying so; once it is back, reindex makes them findable by
// meaning. There is no silent fallback to another embedder.
func TestEmbedderDown(t *testing.T) {
	e := newEnv(t)
	e.ollama = "http://127.0.0.1:1" // nothing listens at the embeddings URL
	out, errOut, err := e.run("Login form rejects valid credentials.", "add")
	if err != nil {
		t.Fatalf("add failed with the embedder down: %v\n%s", err, errOut)
	}
	id := firstID(out)
	if !strings.Contains(errOut, "reindex") {
		t.Errorf("add should say to reindex later: %q", errOut)
	}
	out, errOut, _ = e.run("", "search", "Login")
	if !strings.HasPrefix(out, id) || !strings.Contains(errOut, "keyword search only") {
		t.Errorf("search should answer by keyword and say so; stderr %q:\n%s", errOut, out)
	}

	e.ollama = e.url
	if out := e.must("", "reindex"); !strings.Contains(out, "embedded 1 ") {
		t.Fatalf("reindex after the embedder came back: %q", out)
	}
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Fatalf("not findable by meaning after reindex:\n%s", out)
	}
}

// A fresh install has the built-in embedding model not yet downloaded (and
// here it cannot be: no network), and no chat model. jrnl must work without
// a word of setup noise, say how to get more only when it matters, and index
// everything saved meanwhile once the parts are turned on.
func TestFreshInstall(t *testing.T) {
	e := newEnv(t)
	media := e.config
	e.noChat, e.config = true, ""
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL="} // the built-in model, not downloaded
	e.path = t.TempDir()                                       // a machine with none of the optional tools
	img := filepath.Join(t.TempDir(), "shot.png")
	os.WriteFile(img, []byte("\x89PNG settings page"), 0o644)

	start := time.Now()
	out, errOut, err := e.run("Login form rejects valid credentials.", "add")
	if err != nil || errOut != "" {
		t.Fatalf("add should work quietly: %v, stderr %q", err, errOut)
	}
	// add never waits for the download.
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("add without the model took %s", d)
	}
	id := firstID(out)
	out, errOut, err = e.run("Settings page layout.", "add", "--attach", img)
	if err != nil || strings.Count(errOut, "\n") != 1 || !strings.Contains(errOut, "jrnl doctor") {
		t.Fatalf("attaching should say once that captions need setup: %v, stderr %q", err, errOut)
	}
	imgID := firstID(out)

	if out, errOut, _ := e.run("", "search", "Login"); !strings.HasPrefix(out, id) || errOut != "" {
		t.Errorf("keyword search should work quietly; stderr %q:\n%s", errOut, out)
	}
	if out, errOut, _ := e.run("", "search", "authentication problem"); out != "" || !strings.Contains(errOut, "jrnl doctor") {
		t.Errorf("a search with no keyword match should point to jrnl doctor; stderr %q:\n%s", errOut, out)
	}
	_, errOut, err = e.run("", "summarize", "--on", "today")
	if err == nil || !strings.Contains(errOut, "jrnl list") || !strings.Contains(errOut, "jrnl doctor") {
		t.Errorf("summarize should explain and point to jrnl list: %v, stderr %q", err, errOut)
	}
	doc := e.must("", "doctor")
	// Waiting: the text of both entries, and the image's description.
	for _, want := range []string{"2 entries", "3 item(s) waiting", "embedding  off", "jrnl reindex", "chat       off", "JRNL_CHAT_URL"} {
		if !strings.Contains(doc, want) {
			t.Errorf("doctor is missing %q:\n%s", want, doc)
		}
	}

	// Audio saved before ffmpeg is installed waits for it, instead of being
	// marked as a file that can never be read.
	memo := filepath.Join(t.TempDir(), "memo.m4a")
	os.WriteFile(memo, []byte("not really audio"), 0o644)
	e.must("Voice memo.", "add", "--attach", memo)
	e.must("", "reindex")
	// 3 from before, plus the memo's text and its transcript.
	if doc := e.must("", "doctor"); !strings.Contains(doc, "5 item(s) waiting") {
		t.Errorf("audio without ffmpeg should stay waiting:\n%s", doc)
	}

	// Offline, reindex says why the model is missing and saves nothing wrong.
	if _, errOut, _ := e.run("", "reindex"); !strings.Contains(errOut, "download") {
		t.Errorf("reindex should say the download failed: %q", errOut)
	}

	// An embeddings API set up: search by meaning, and no complaint about reranking.
	e.extra = nil
	e.must("", "reindex")
	if out, errOut, _ := e.run("", "search", "authentication problem"); !strings.HasPrefix(out, id) || errOut != "" {
		t.Errorf("with embeddings on, search by meaning should work quietly; stderr %q:\n%s", errOut, out)
	}

	// A chat model and a media embedder too: the attachment saved before
	// setup gets described and its own vector.
	e.noChat, e.config = false, media
	if out := e.must("", "reindex"); !strings.Contains(out, "described 1 file(s)") {
		t.Fatalf("reindex after chat setup: %s", out)
	}
	if out := e.must("", "search", "picture"); !strings.HasPrefix(out, imgID) {
		t.Errorf("attachment not searchable after setup:\n%s", out)
	}
	chats := len(e.ai.chatReqs)
	if out := e.must("", "search", "--like", img); !strings.HasPrefix(out, imgID) || len(e.ai.chatReqs) != chats {
		t.Errorf("--like should find the attachment by its media vector after setup:\n%s", out)
	}
}

// Vectors made by another embedder must never be compared with the query.
func TestVectorsFromAnotherEmbedderAreIgnored(t *testing.T) {
	e := newEnv(t)
	id := firstID(e.must("Login form rejects valid credentials.", "add"))
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Fatalf("setup: not found by meaning:\n%s", out)
	}
	db, err := sql.Open("sqlite", filepath.Join(e.home, "test", "journal.db"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE chunks SET embedder = 'some-other/model/768'`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	if out := e.must("", "search", "authentication problem"); strings.Contains(out, id) {
		t.Fatalf("matched a vector from another embedder:\n%s", out)
	}
}

func ids(out string) []string {
	var got []string
	for _, l := range strings.Split(out, "\n") {
		if l != "" && !strings.HasPrefix(l, " ") {
			got = append(got, strings.Fields(l)[0])
		}
	}
	return got
}

func sameSet(t *testing.T, what string, got []string, want ...string) {
	t.Helper()
	g, w := strings.Join(slices.Sorted(slices.Values(got)), " "), strings.Join(slices.Sorted(slices.Values(want)), " ")
	if g != w {
		t.Errorf("%s:\n got  %s\n want %s", what, g, w)
	}
}

// "Yesterday", "last week" and friends are local calendar days. The clock is
// Thursday 2026-11-19 in Berlin; 00:30 local on the 18th is 23:30 UTC on the
// 17th, which a UTC day boundary gets wrong.
func TestTimeFilters(t *testing.T) {
	e := newEnv(t)
	at := func(when, body string) string { return firstID(e.must(body, "add", "--at", when)) }
	sun8 := at("2026-11-08T12:00:00+01:00", "sunday before last")
	mon9 := at("2026-11-09T10:00:00+01:00", "monday last week")
	sun15 := at("2026-11-15T22:00:00+01:00", "sunday night last week")
	mon16 := at("2026-11-16T08:00:00+01:00", "monday this week")
	tue17 := at("2026-11-17T23:30:00+01:00", "tuesday late")
	wedEarly := at("2026-11-18T00:30:00+01:00", "wednesday just after midnight")
	wedLate := at("2026-11-18T23:30:00+01:00", "wednesday late")
	thu := at("2026-11-19T00:10:00+01:00", "thursday just after midnight")
	oct := at("2026-10-31T23:59:00+01:00", "last day of october")

	list := func(args ...string) []string {
		return ids(e.must("", append([]string{"list", "--limit", "100"}, args...)...))
	}
	sameSet(t, "--on yesterday", list("--on", "yesterday"), wedEarly, wedLate)
	sameSet(t, "--since yesterday --until yesterday", list("--since", "yesterday", "--until", "yesterday"), wedEarly, wedLate)
	sameSet(t, "--on today", list("--on", "today"), thu)
	sameSet(t, "--on '2 days ago'", list("--on", "2 days ago"), tue17)
	sameSet(t, "--since monday", list("--since", "monday"), mon16, tue17, wedEarly, wedLate, thu)
	sameSet(t, "--on 'last week'", list("--on", "last week"), mon9, sun15)
	sameSet(t, "--until 'last week'", list("--until", "last week"), sun8, mon9, sun15, oct)
	sameSet(t, "--on october", list("--on", "october"), oct)

	got := list("--oldest", "--limit", "2")
	if len(got) != 2 || got[0] != oct || got[1] != sun8 {
		t.Errorf("--oldest: got %v, want [%s %s]", got, oct, sun8)
	}
	if _, errOut, err := e.run("", "list", "--on", "36h"); err == nil || !strings.Contains(errOut, "--on needs a day") {
		t.Errorf("--on with a point in time should be refused, stderr %q", errOut)
	}
}

// "What's the latest on X" and "when did X first happen" must find the newest
// and oldest matches, even when older entries match the words better.
func TestSearchNewestAndOldest(t *testing.T) {
	e := newEnv(t)
	first := firstID(e.must("Redis cluster went read-only after a long afternoon of unrelated chores on the reporting dashboards, the billing exports and the onboarding emails.", "add", "--at", "2026-08-01"))
	for i := 0; i < 35; i++ {
		e.must("Redis cluster: Redis cluster.", "add", "--at", fmt.Sprintf("2026-09-%02d", i%28+1))
	}
	latest := firstID(e.must("Redis cluster upgraded after a long afternoon of unrelated chores on the reporting dashboards, the billing exports and the onboarding emails.", "add", "--at", "2026-11-19T09:00:00+01:00"))

	if got := ids(e.must("", "search", "--newest", "--limit", "5", "Redis cluster")); len(got) == 0 || got[0] != latest {
		t.Errorf("--newest: got %v, want %s first", got, latest)
	}
	if got := ids(e.must("", "search", "--oldest", "--limit", "5", "Redis cluster")); len(got) == 0 || got[0] != first {
		t.Errorf("--oldest: got %v, want %s first", got, first)
	}
}

// "What did the last session do" means the session before the current one
// (the test runs as sess-1).
func TestLastSession(t *testing.T) {
	e := newEnv(t)
	a1 := firstID(e.must("first session, step one", "add", "--session", "sess-0", "--at", "2026-11-18T10:00:00+01:00"))
	a2 := firstID(e.must("first session, step two", "add", "--session", "sess-0", "--at", "2026-11-18T11:00:00+01:00"))
	e.must("an older session", "add", "--session", "sess-old", "--at", "2026-11-10")
	e.must("this session", "add") // sess-1, now
	sameSet(t, "--session last", ids(e.must("", "list", "--session", "last")), a1, a2)
	out := e.must("", "sessions")
	if !strings.HasPrefix(out, "sess-1") || !strings.Contains(strings.SplitN(out, "\n", 2)[0], "(this session)") ||
		!strings.Contains(out, "sess-0") || !strings.Contains(out, "  2 entries") {
		t.Errorf("sessions output:\n%s", out)
	}
}

func TestFileLinksAndStats(t *testing.T) {
	e := newEnv(t)
	cause := firstID(e.must("Writers hit SQLITE_BUSY.", "add", "--kind", "bug", "--tag", "db", "--file", "internal/store/store.go"))
	fix := firstID(e.must("Use BEGIN IMMEDIATE.", "add", "--kind", "decision", "--tag", "db", "--link", cause[:10]))
	e.must("Tweaked the README.", "add", "--kind", "log", "--file", "README.md")

	sameSet(t, "--file store.go", ids(e.must("", "list", "--file", "store.go")), cause)
	sameSet(t, "--file internal/store/store.go", ids(e.must("", "list", "--file", "internal/store/store.go")), cause)
	if got := ids(e.must("", "list", "--file", "other.go")); len(got) != 0 {
		t.Errorf("--file other.go matched %v", got)
	}
	if show := e.must("", "show", cause); !strings.Contains(show, "linked_from: ["+fix) {
		t.Errorf("show should list the entry linking here:\n%s", show)
	}
	if show := e.must("", "show", fix); !strings.Contains(show, "links: ["+cause) {
		t.Errorf("a link given as a prefix should be stored as the full ID:\n%s", show)
	}
	if _, errOut, err := e.run("x", "add", "--link", "ZZZZ"); err == nil || !strings.Contains(errOut, "no entry") {
		t.Errorf("a link to nothing should be refused, stderr %q", errOut)
	}
	out := e.must("", "stats", "--tag", "db")
	for _, want := range []string{"2 entries", "by kind: ", "bug 1", "decision 1", "by tag: db 2"} {
		if !strings.Contains(out, want) {
			t.Errorf("stats --tag db missing %q:\n%s", want, out)
		}
	}
}

// "What's still open from last week?" — opt-in open state, closed with a
// note that stays searchable.
func TestOpenAndClose(t *testing.T) {
	e := newEnv(t)
	bug := firstID(e.must("Export times out for big tenants.", "add", "--kind", "bug", "--open", "--at", "2026-11-10"))
	todo := firstID(e.must("Write the runbook for restores.", "add", "--kind", "todo", "--at", "2026-11-11"))
	e.must("Fixed a typo.", "add", "--kind", "bug", "--at", "2026-11-12") // not tracked
	e.must("Old open idea.", "add", "--kind", "idea", "--open", "--at", "2026-10-01")

	sameSet(t, "open last week", ids(e.must("", "list", "--status", "open", "--on", "last week")), bug, todo)

	out := e.must("", "close", bug[:12], "--note", "Streamed the CSV instead of building it in memory.")
	if !strings.Contains(out, bug+"  closed") {
		t.Fatalf("close output: %q", out)
	}
	sameSet(t, "open last week after close", ids(e.must("", "list", "--status", "open", "--on", "last week")), todo)
	sameSet(t, "closed", ids(e.must("", "list", "--status", "closed")), bug)

	show := e.must("", "show", bug)
	note := ""
	for _, l := range strings.Split(show, "\n") {
		if strings.HasPrefix(l, "closed_by: ") {
			f := strings.Fields(l)
			note = f[len(f)-1]
		}
	}
	if note == "" || !strings.Contains(show, "linked_from: ["+note) {
		t.Fatalf("show should say closed by the note entry, which links back:\n%s", show)
	}
	if out := e.must("", "search", "--limit", "3", "Streamed the CSV"); !strings.HasPrefix(out, note) {
		t.Errorf("the closing note should be searchable:\n%s", out)
	}

	e.must("", "reopen", bug)
	sameSet(t, "open after reopen", ids(e.must("", "list", "--status", "open", "--on", "last week")), bug, todo)
	if _, errOut, err := e.run("", "list", "--status", "done"); err == nil || !strings.Contains(errOut, "open or closed") {
		t.Errorf("unknown status should be refused, stderr %q", errOut)
	}
}

// A journal written by the first release (schema 1, no status or embedder
// columns) must keep working, with its entries intact.
func TestUpgradeFromSchema1(t *testing.T) {
	e := newEnv(t)
	id := firstID(e.must("Entry from the first release.", "add"))
	db, err := sql.Open("sqlite", filepath.Join(e.home, "test", "journal.db"))
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{
		`ALTER TABLE entries DROP COLUMN status`, `ALTER TABLE entries DROP COLUMN closed_at`,
		`ALTER TABLE entries DROP COLUMN closed_by`, `ALTER TABLE chunks DROP COLUMN embedder`, `PRAGMA user_version = 1`,
	} {
		if _, err := db.Exec(q); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	db.Close()

	if out := e.must("", "list"); !strings.HasPrefix(out, id) {
		t.Fatalf("old entry missing after upgrade:\n%s", out)
	}
	e.must("", "close", id)
	sameSet(t, "closed after upgrade", ids(e.must("", "list", "--status", "closed")), id)
}

// A summary covers exactly the period asked for, cites every entry so the
// reader can open it, is saved and reused, and is redone when an entry is
// added to the period later.
func TestSummarize(t *testing.T) {
	e := newEnv(t)
	var yesterday []string
	for i, at := range []string{"2026-11-18T09:00:00+01:00", "2026-11-18T13:00:00+01:00", "2026-11-18T23:50:00+01:00"} {
		yesterday = append(yesterday, firstID(e.must(fmt.Sprintf("Yesterday's work, part %d.", i), "add", "--at", at)))
	}
	today := firstID(e.must("Today's work.", "add"))

	out := e.must("", "summarize", "--on", "yesterday")
	for _, id := range yesterday {
		if !strings.Contains(out, id[:14]) {
			t.Errorf("summary does not cite %s:\n%s", id[:14], out)
		}
	}
	if strings.Contains(out, today[:14]) || !strings.Contains(out, "Covers 3 entries") || !strings.Contains(out, "Summary: 2026-11-18 → 2026-11-18") {
		t.Errorf("summary should cover exactly yesterday's 3 entries:\n%s", out)
	}
	if strings.Contains(out, "99999") {
		t.Errorf("a citation that matches no entry must be removed:\n%s", out)
	}
	if got := ids(e.must("", "list", "--kind", "summary")); len(got) != 1 {
		t.Fatalf("want 1 saved summary, got %v", got)
	}

	calls := e.ai.summaryCalls.Load()
	e.must("", "summarize", "--on", "yesterday")
	if n := e.ai.summaryCalls.Load() - calls; n != 0 {
		t.Errorf("an unchanged period should reuse the saved summary, made %d calls", n)
	}

	late := firstID(e.must("Logged later, about yesterday.", "add", "--at", "2026-11-18T18:00:00+01:00"))
	out = e.must("", "summarize", "--on", "yesterday")
	if !strings.Contains(out, late[:14]) || !strings.Contains(out, "Covers 4 entries") {
		t.Errorf("summary should be redone with the late entry:\n%s", out)
	}
	if got := ids(e.must("", "list", "--kind", "summary")); len(got) != 1 {
		t.Errorf("the stale summary should be replaced, got %v", got)
	}
	if _, errOut, err := e.run("", "summarize"); err == nil || !strings.Contains(errOut, "which period") {
		t.Errorf("summarize without a period should ask for one, stderr %q", errOut)
	}
}

// A long period gets a short overview citing its day summaries, which cite
// the entries: following the citations reaches every entry. Day summaries
// are saved, and an entry added later redoes only its own day and the
// overview.
func TestSummarizeLongPeriodByDay(t *testing.T) {
	e := newEnv(t)
	body := strings.Repeat("Investigated the export worker and tuned its batch size. ", 9)
	var all []string
	for i := 0; i < 120; i++ {
		all = append(all, firstID(e.must(body, "add", "--at", fmt.Sprintf("2026-11-%02dT10:%02d:00+01:00", 9+i%7, i%60))))
	}
	out := e.must("", "summarize", "--on", "last week")
	if n := e.ai.summaryCalls.Load(); n != 8 {
		t.Errorf("want 7 day summaries + 1 overview = 8 calls, made %d", n)
	}
	if !strings.Contains(out, "Covers 120 entries") || !strings.Contains(out, "Details by day: 2026-11-09 [") || strings.Contains(out, "D99") {
		t.Errorf("overview should cover 120 entries, list the days, and drop an invented day:\n%s", out)
	}
	days := regexp.MustCompile(`\d{4}-\d\d-\d\d \[([0-9A-Z]{14})\]`).FindAllStringSubmatch(out, -1)
	if len(days) != 7 {
		t.Fatalf("want 7 day summaries linked, got %d:\n%s", len(days), out)
	}
	var detail strings.Builder
	for _, d := range days {
		if !strings.Contains(out, "- theme ["+d[1]+"]") {
			t.Errorf("overview bullet should cite day summary %s", d[1])
		}
		detail.WriteString(e.must("", "show", d[1]))
	}
	missing := 0
	for _, id := range all {
		if !strings.Contains(detail.String(), id[:14]) {
			missing++
		}
	}
	if missing > 0 {
		t.Errorf("%d of 120 entries can't be reached from the overview through its day summaries", missing)
	}

	calls := e.ai.summaryCalls.Load()
	late := firstID(e.must("Found the export leak.", "add", "--at", "2026-11-12T18:00:00+01:00"))
	out = e.must("", "summarize", "--on", "last week")
	if n := e.ai.summaryCalls.Load() - calls; n != 2 {
		t.Errorf("an entry added to one day should redo that day and the overview (2 calls), made %d", n)
	}
	if !strings.Contains(out, "Covers 121 entries") {
		t.Errorf("the redone overview should count the late entry:\n%s", out)
	}
	day12 := regexp.MustCompile(`2026-11-12 \[([0-9A-Z]{14})\]`).FindStringSubmatch(out)
	if day12 == nil || !strings.Contains(e.must("", "show", day12[1]), late[:14]) {
		t.Errorf("the late entry should be cited in the redone day summary")
	}
}

// Any OpenAI-compatible chat API works: jrnl sends the configured model, key
// and extra fields.
func TestAnyChatProvider(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "shot.png")
	os.WriteFile(img, []byte("\x89PNG settings page"), 0o644)
	e.must("Login form rejects valid credentials.", "add")
	e.must("rerank-me: Login is slow on Mondays.", "add")

	e.noChat = true
	e.extra = []string{"JRNL_CHAT_URL=" + e.url + "/v1/", "JRNL_CHAT_KEY=k-test", "JRNL_CHAT_MODEL=vendor/some-model",
		`JRNL_CHAT_EXTRA={"reasoning_effort":"low"}`}
	out, errOut, err := e.run("", "search", "Login")
	if err != nil || !strings.Contains(strings.SplitN(out, "\n", 2)[0], "rerank-me") {
		t.Fatalf("search should be reranked by the configured model: %v %q\n%s", err, errOut, out)
	}
	got := e.ai.chatReqs[len(e.ai.chatReqs)-1]
	if got["model"] != "vendor/some-model" || got["auth"] != "Bearer k-test" || got["reasoning_effort"] != "low" {
		t.Errorf("want the configured model, key and extra field, got %v", got)
	}
	if out := e.must("Settings page.", "add", "--attach", img); out == "" {
		t.Fatal("no ID")
	}
	if out := e.must("", "search", "picture"); !strings.Contains(out, "A screenshot showing settings page") && !strings.Contains(out, "matched") {
		t.Errorf("attachment not captioned through the configured model:\n%s", out)
	}

	e.extra = []string{"JRNL_CHAT_URL=" + e.url + "/v1"}
	if _, errOut, err := e.run("", "search", "Login"); err == nil || !strings.Contains(errOut, "JRNL_CHAT_MODEL") {
		t.Errorf("a chat URL without a model should be refused: %v %q", err, errOut)
	}
}

// Switching the embedding model must not mix vectors: entries embedded by
// the old model are not matched until reindex embeds them with the new one.
func TestSwitchEmbeddingModel(t *testing.T) {
	e := newEnv(t)
	id := firstID(e.must("Login form rejects valid credentials.", "add"))
	e.extra = []string{"JRNL_EMBED_URL=" + e.url + "/v1", "JRNL_EMBED_KEY=k", "JRNL_EMBED_MODEL=other-embedder"}
	if out := e.must("", "search", "authentication problem"); strings.Contains(out, id) {
		t.Fatalf("matched a vector from the old model:\n%s", out)
	}
	if doc := e.must("", "doctor"); !strings.Contains(doc, "1 item(s) waiting") || !strings.Contains(doc, "other-embedder") {
		t.Errorf("doctor should show the new model and the entry waiting for it:\n%s", doc)
	}
	if out := e.must("", "reindex"); !strings.Contains(out, "embedded 1 ") {
		t.Fatalf("reindex should re-embed the entry: %q", out)
	}
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Fatalf("not found by meaning after reindex with the new model:\n%s", out)
	}
	if m := e.ai.embedModels[len(e.ai.embedModels)-1]; m != "other-embedder" {
		t.Errorf("last embedding request used %q", m)
	}
}

// A journal from before schema 3 has vectors named by the old scheme. They
// came from the same models and APIs, so they keep working without reindex:
// text from Ollama, and media from Workers AI (now a cloudflare provider).
func TestUpgradeKeepsOldVectors(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "chart.png")
	os.WriteFile(img, []byte("\x89PNG quarterly chart"), 0o644)
	id := firstID(e.must("Login form rejects valid credentials.", "add", "--attach", img))
	db, err := sql.Open("sqlite", filepath.Join(e.home, "test", "journal.db"))
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{
		`UPDATE chunks SET embedder = 'ollama/embeddinggemma-2:270m/768' WHERE modality IN ('text', 'caption')`,
		`UPDATE chunks SET embedder = 'workers-ai/embeddinggemma-2/768' WHERE modality = 'image'`,
		`PRAGMA user_version = 2`} {
		if _, err := db.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Fatalf("old text vectors lost after the upgrade:\n%s", out)
	}
	chats := len(e.ai.chatReqs)
	if out := e.must("", "search", "--like", img); !strings.HasPrefix(out, id) || len(e.ai.chatReqs) != chats {
		t.Fatalf("old media vectors lost after the upgrade:\n%s", out)
	}
	if doc := e.must("", "doctor"); strings.Contains(doc, "waiting") {
		t.Errorf("nothing should need embedding again after the upgrade:\n%s", doc)
	}
}

// Each embedding model gets text in the format its model card asks for:
// Qwen3 wants an instruction on the query and the document plain; models
// jrnl knows nothing about get plain text.
func TestTextFormatFollowsTheModel(t *testing.T) {
	for model, want := range map[string][2]string{
		"qwen3-embedding:0.6b":   {"Instruct: ", "Login\n\nLogin form rejects valid credentials."},
		"text-embedding-3-small": {"authentication problem", "Login\n\nLogin form rejects valid credentials."},
	} {
		e := newEnv(t)
		e.extra = []string{"JRNL_EMBED_MODEL=" + model}
		e.must("Login form rejects valid credentials.", "add", "--title", "Login")
		e.must("", "search", "authentication problem")
		got := strings.Join(e.ai.texts, "\n---\n")
		if !strings.Contains(got, "\n---\n"+want[0]) && !strings.HasPrefix(got, want[0]) || !strings.Contains(got, want[1]) || strings.Contains(got, "title: ") {
			t.Errorf("%s: want query starting %q and document %q, got:\n%s", model, want[0], want[1], got)
		}
	}
}

// A dedicated reranker (Cohere-style /rerank, in Cohere's or Voyage's reply
// shape) orders search results without any chat model.
func TestRerankAPI(t *testing.T) {
	for _, model := range []string{"cohere/rerank-v5-fast", "voyage/rerank-3-lite"} {
		e := newEnv(t)
		e.noChat = true // no chat model at all
		e.extra = []string{"JRNL_RERANK_URL=" + e.url + "/v1", "JRNL_RERANK_KEY=k", "JRNL_RERANK_MODEL=" + model}
		e.must("Login form rejects valid credentials.", "add")
		e.must("rerank-me: Login is slow on Mondays.", "add")
		out, errOut, err := e.run("", "search", "Login")
		if err != nil || errOut != "" || !strings.Contains(strings.SplitN(out, "\n", 2)[0], "rerank-me") || e.ai.rerankCalls.Load() != 1 {
			t.Errorf("%s: want the reranker's pick first, quietly, from 1 call (made %d); stderr %q:\n%s",
				model, e.ai.rerankCalls.Load(), errOut, out)
		}
	}
}

// Keys can come from a .env file, such as a 1Password Environment mounted as
// a named pipe. The environment still wins over the file.
func TestKeysFromEnvFile(t *testing.T) {
	e := newEnv(t)
	e.must("Login form rejects valid credentials.", "add")
	e.must("rerank-me: Login is slow on Mondays.", "add")
	pipe := filepath.Join(t.TempDir(), ".env")
	if err := syscall.Mkfifo(pipe, 0o600); err != nil {
		t.Fatal(err)
	}
	serve := func() {
		go func() {
			// First an empty answer, as 1Password gives a reader that comes
			// while another is being served; then the file.
			if f, err := os.OpenFile(pipe, os.O_WRONLY, 0); err == nil {
				f.Close()
			}
			f, err := os.OpenFile(pipe, os.O_WRONLY, 0)
			if err == nil {
				f.WriteString("# from 1Password\nexport JRNL_CHAT_KEY=\"from-file\"\nJRNL_CHAT_MODEL=vendor/m\n")
				f.Close()
			}
		}()
	}
	e.noChat = true
	e.extra = []string{"JRNL_ENV_FILE=" + pipe, "JRNL_CHAT_URL=" + e.url + "/v1"}

	serve()
	e.must("", "search", "Login")
	if got := e.ai.chatReqs[len(e.ai.chatReqs)-1]; got["auth"] != "Bearer from-file" || got["model"] != "vendor/m" {
		t.Errorf("want the key and model from the file, got %v", got)
	}

	serve()
	e.extra = append(e.extra, "JRNL_CHAT_KEY=from-env")
	e.must("", "search", "Login")
	if got := e.ai.chatReqs[len(e.ai.chatReqs)-1]; got["auth"] != "Bearer from-env" {
		t.Errorf("the environment should win over the file, got %v", got["auth"])
	}
}

// A mounted pipe that nobody answers (1Password closed or its prompt not
// approved) must not hang jrnl.
func TestEnvFileWithoutWriter(t *testing.T) {
	pipe := filepath.Join(t.TempDir(), ".env")
	if err := syscall.Mkfifo(pipe, 0o600); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	loadEnvFile(pipe, 200*time.Millisecond)
	if d := time.Since(start); d > 2*time.Second {
		t.Errorf("waited %v for a pipe with no writer", d)
	}
}

// config.toml picks each role's provider and model. JRNL_* variables win: a
// lone model or key replaces that field, and a URL replaces the whole role,
// so the file's key never goes to another URL.
func TestConfigFile(t *testing.T) {
	e := newEnv(t)
	e.must("Login form rejects valid credentials.", "add")
	e.must("rerank-me: Login is slow on Mondays.", "add")
	e.noChat = true
	e.config = `
[providers.gw]
url = "` + e.url + `/v1"
key = "env:MY_GATEWAY_KEY"

[chat]
provider = "gw"
model    = "conf/model"
extra    = { reasoning_effort = "low", kwargs = { thinking = false } }
`
	lastChat := func(env ...string) map[string]any {
		t.Helper()
		e.extra = append([]string{"MY_GATEWAY_KEY=k-conf"}, env...)
		e.must("", "search", "Login")
		return e.ai.chatReqs[len(e.ai.chatReqs)-1]
	}
	if got := lastChat(); got["model"] != "conf/model" || got["auth"] != "Bearer k-conf" || got["reasoning_effort"] != "low" ||
		got["kwargs"].(map[string]any)["thinking"] != false {
		t.Errorf("want the config's model, key and extra field, got %v", got)
	}
	if got := lastChat("JRNL_CHAT_MODEL=env/model"); got["model"] != "env/model" || got["auth"] != "Bearer k-conf" {
		t.Errorf("JRNL_CHAT_MODEL alone should replace only the model, got %v", got)
	}
	if got := lastChat("JRNL_CHAT_URL="+e.url+"/v1", "JRNL_CHAT_MODEL=env/model"); got["auth"] == "Bearer k-conf" {
		t.Errorf("the config's key went to the URL set by JRNL_CHAT_URL: %v", got)
	}

	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL="} // the config's embedder, not the test default
	for conf, want := range map[string]string{
		"[providers.gw]\nurl = \"x\"\nkey = \"sk-123\"\n[chat]\nprovider = \"gw\"\nmodel = \"m\"\n":           "env:",
		"[chat]\nprovider = \"gw\"\nmodle = \"m\"\n":                                                          "unknown setting",
		"[providers.cf]\ntype = \"cloudflare\"\naccount = \"a\"\n[embed]\nprovider = \"cf\"\nmodel = \"m\"\n": "can't do embed",
	} {
		e.config = conf
		if _, errOut, err := e.run("", "search", "Login"); err == nil || !strings.Contains(errOut, want) {
			t.Errorf("config %q: want an error with %q, got %v %q", conf, want, err, errOut)
		}
	}
}

// The same embedding model at the same URL is the same embedder, whether
// config.toml or JRNL_EMBED_* names it: switching must not re-embed anything.
func TestConfigAndEnvNameTheSameEmbedder(t *testing.T) {
	e := newEnv(t)
	e.config = "[providers.o]\nurl = \"" + e.url + "/v1\"\n[embed]\nprovider = \"o\"\nmodel = \"emb-x\"\n"
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL="} // the config's embedder, not the test default
	id := firstID(e.must("Login form rejects valid credentials.", "add"))
	e.config = ""
	e.extra = []string{"JRNL_EMBED_URL=" + e.url + "/v1", "JRNL_EMBED_MODEL=emb-x"}
	if doc := e.must("", "doctor"); strings.Contains(doc, "waiting") || !strings.Contains(doc, "emb-x") {
		t.Errorf("switching from config to env should not need a reindex:\n%s", doc)
	}
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Errorf("not found by meaning after switching to env:\n%s", out)
	}
}

// list says how long ago each entry was, so an agent needs no date arithmetic.
// JRNL_NOW is 2026-11-19 15:30 +01:00; the expected ages are worked by hand.
func TestListShowsAge(t *testing.T) {
	e := newEnv(t)
	cases := []struct{ at, want string }{
		{"2026-11-19T15:29:30+01:00", "just now"},
		{"2026-11-19T15:05:00+01:00", "25m ago"},
		{"2026-11-18T14:30:00+01:00", "25h ago"},
		{"2026-11-10T15:30:00+01:00", "9d ago"},
		{"2026-10-01T15:30:00+02:00", "7w ago"},  // 49 days
		{"2026-05-19T15:30:00+02:00", "6mo ago"}, // 184 days
		{"2023-11-19T15:30:00+01:00", "3y ago"},
	}
	want := map[string]string{}
	for _, c := range cases {
		want[firstID(e.must("entry "+c.want, "add", "--at", c.at))] = c.want
	}
	for _, l := range strings.Split(strings.TrimSpace(e.must("", "list")), "\n") {
		id := strings.Fields(l)[0]
		if !strings.Contains(l, "  "+want[id]+"  ") {
			t.Errorf("want %q in list line:\n%s", want[id], l)
		}
		if strings.Contains(l, "2026-") || strings.Contains(l, "2023-") {
			t.Errorf("list line still shows a date:\n%s", l)
		}
	}
}

// show prints metadata as YAML front matter and the body as written, once.
// read and open are the same command.
func TestShowIsFrontMatterThenBody(t *testing.T) {
	e := newEnv(t)
	body := "Fix: login loop\n\nRoot cause: the # in the cookie name."
	id := firstID(e.must(body, "add", "--kind", "bug", "--tag", "auth", "--at", "2026-11-19T12:30:00+01:00"))
	out := e.must("", "show", id)
	head, rest, ok := strings.Cut(strings.TrimPrefix(out, "---\n"), "\n---\n")
	if !strings.HasPrefix(out, "---\n") || !ok {
		t.Fatalf("want a front matter block first:\n%s", out)
	}
	if rest != "\n"+body+"\n" {
		t.Errorf("the body after the front matter should be exactly the text added, got %q", rest)
	}
	for _, line := range []string{"id: " + id, `title: "Fix: login loop"`, "time: 2026-11-19 12:30 (3h ago)", "kind: bug", "tags: [auth]"} {
		if !strings.Contains(head, line+"\n") {
			t.Errorf("front matter is missing %q:\n%s", line, head)
		}
	}
	if strings.Contains(out, "# Fix") {
		t.Errorf("the title should not be repeated as a heading:\n%s", out)
	}
	for _, alias := range []string{"read", "open"} {
		if got := e.must("", alias, id[:12]); got != out {
			t.Errorf("%s should print what show prints, got:\n%s", alias, got)
		}
	}
}

// --fast skips the reranker and keeps only hits whose meaning score reaches
// --min-score. In the fake, a sign-in query sits 0.5 from a sign-in entry
// (a document puts half its weight on its concept) and 0 from anything else.
func TestFastSearchSkipsRerankAndCutsByScore(t *testing.T) {
	e := newEnv(t)
	auth := firstID(e.must("Users could not sign in after the token refresh.", "add", "--kind", "bug"))
	auth2 := firstID(e.must("Login page rejected valid credentials on Safari.", "add", "--kind", "bug"))
	e.must("Chose Go over Bun for the CLI.", "add", "--kind", "decision")

	// Two hits survive the cut, so a reranker would be asked.
	chats := len(e.ai.chatReqs)
	sameSet(t, "--fast --min-score 0.4", ids(e.must("", "search", "--fast", "--min-score", "0.4", "login failure")), auth, auth2)
	if got := ids(e.must("", "search", "--fast", "--min-score", "0.6", "login failure")); len(got) != 0 {
		t.Errorf("0.5 is under 0.6, so nothing should be left, got %v", got)
	}
	if len(e.ai.chatReqs) != chats {
		t.Errorf("--fast should not call the chat model to rerank")
	}
}

// A plain repository is read from its .git directory: the project comes from
// the origin remote, and an entry records the branch and short commit, from
// a subfolder and with refs packed. The expected values come from git itself.
func TestProjectBranchAndCommitFromGit(t *testing.T) {
	e := newEnv(t)
	repo := t.TempDir()
	gitc := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		cmd.Env = append(os.Environ(), "GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1", "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	gitc("init", "-q", "-b", "feature/x")
	gitc("remote", "add", "upstream", "git@github.com:other/wrong.git")
	gitc("remote", "add", "origin", "https://github.com/Acme/Widget.git")
	os.MkdirAll(filepath.Join(repo, "sub", "dir"), 0o755)
	e.dir = filepath.Join(repo, "sub", "dir")
	e.path = t.TempDir() // no git for jrnl: it must read .git itself

	// No commit yet: project and branch, no commit.
	if out := e.must("", "where"); !strings.HasSuffix(strings.TrimSpace(out), "acme-widget") {
		t.Errorf("where = %q, want the origin slug acme-widget", out)
	}
	gitc("commit", "-q", "--allow-empty", "-m", "one")
	gitc("pack-refs", "--all")
	want := gitc("rev-parse", "--short=7", "HEAD")
	out := e.must("", "show", firstID(e.must("an entry", "add")))
	for _, line := range []string{`branch: "feature/x"`, "commit: " + want} {
		if !strings.Contains(out, line+"\n") {
			t.Errorf("show is missing %q:\n%s", line, out)
		}
	}
	// Detached HEAD: no branch, the commit still recorded.
	gitc("checkout", "-q", "--detach")
	out = e.must("", "show", firstID(e.must("detached", "add")))
	if strings.Contains(out, "branch:") || !strings.Contains(out, "commit: "+want+"\n") {
		t.Errorf("detached HEAD should record the commit and no branch:\n%s", out)
	}
}

// jrnl serve answers over a Unix socket that only this user can open; sees
// entries other processes add; finds an exact identifier and a reworded
// question; shows nothing for an unrelated one; and never creates a journal
// for a project it is asked about. Uses the real potion model, so it runs
// only where the model is cached.
func TestServe(t *testing.T) {
	needLocalModel(t)
	e := newEnv(t)
	// A short home: a Unix socket path has a 104-byte limit on macOS.
	home, err := os.MkdirTemp("/tmp", "jrs")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(home) })
	e.home = home
	quitServerAt(t, home) // the first add may start one before the test's own
	// The command line embeds with the local model, as the server does.
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL=", "JRNL_MODEL_DIR=" + localDir()}
	login := firstID(e.must("Users could not sign in after the token refresh.", "add", "--kind", "bug"))
	goBun := firstID(e.must("Chose Go over Bun for the CLI.", "add", "--kind", "decision"))

	srv := exec.Command(bin, "serve")
	srv.Env = serveEnv(home)
	if err := srv.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { srv.Process.Kill(); srv.Wait() })
	sock := filepath.Join(home, ".run", "jrnl.sock")
	client := &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", sock)
	}}}
	search := func(project, q string) (int, string) {
		r, err := client.Get("http://jrnl/search?" + url.Values{"project": {project}, "q": {q}}.Encode())
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		b, _ := io.ReadAll(r.Body)
		return r.StatusCode, string(b)
	}
	for i := 0; ; i++ {
		if _, err := client.Get("http://jrnl/health"); err == nil {
			break
		}
		if i == 100 {
			t.Fatal("jrnl serve did not answer within 10 s")
		}
		time.Sleep(100 * time.Millisecond)
	}
	if st, _ := os.Stat(sock); st.Mode().Perm() != 0o600 {
		t.Errorf("socket mode %v, want 0600", st.Mode().Perm())
	}

	if _, out := search("test", "sign in fails after a token refresh"); !strings.HasPrefix(out, login) {
		t.Errorf("a reworded question should find the sign-in entry first:\n%s", out)
	}
	if _, out := search("test", "kubernetes pod keeps restarting"); out != "" {
		t.Errorf("an unrelated question should show nothing:\n%s", out)
	}
	// Written by another process while the server runs: found by meaning,
	// which needs its vector, and by an identifier no embedding can place.
	card := firstID(e.must("Payments failed when the customer's card had expired.", "add", "--kind", "bug"))
	if _, out := search("test", "billing broke because of an expired credit card"); !strings.HasPrefix(out, card) {
		t.Errorf("an entry added meanwhile should be found by meaning:\n%s", out)
	}
	// A long entry, as real ones are: the identifier is a small part of its vector.
	hex := firstID(e.must("Crash on start after the upgrade\n\nThe crash report 7f3e9ab2c4 came from the old build that "+
		"still shipped the previous renderer. Players on older phones saw a black screen for a few seconds, then the "+
		"app closed. We rolled the renderer back, rebuilt the asset bundle, cleared the shader cache on first launch "+
		"and asked the store to expedite the review. Support answered every ticket with the workaround meanwhile.", "add", "--kind", "bug"))
	if _, out := search("test", "7f3e9ab2c4"); !strings.HasPrefix(out, hex) {
		t.Errorf("an exact identifier should be found:\n%s", out)
	}
	if code, _ := search("nobody", "anything at all here"); code != http.StatusNotFound {
		t.Errorf("an unknown project should be 404, got %d", code)
	}
	if _, err := os.Stat(filepath.Join(home, "nobody")); err == nil {
		t.Errorf("serve created a journal for a project it was only asked about")
	}

	// With the server up, a command embeds through it: this one has no
	// model of its own to load.
	own := e.extra
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL=", "JRNL_MODEL_DIR=" + filepath.Join(home, "no-model")}
	refund := firstID(e.must("Refunds failed when the card network timed out.", "add", "--kind", "bug"))
	if doc := e.must("", "doctor"); strings.Contains(doc, "waiting") {
		t.Errorf("an add with the server up should be embedded through it:\n%s", doc)
	}
	if _, out := search("test", "money back did not go through"); !strings.HasPrefix(out, refund) {
		t.Errorf("the entry embedded through the server should be found by meaning:\n%s", out)
	}
	e.extra = own

	// A new decision like an earlier one: jrnl add names the earlier one and
	// the command that marks it replaced, and links nothing itself.
	out, errOut, err := e.run("", "add", "--kind", "decision", "Chose Go over Bun for the CLI again, after trying Bun's compile step")
	if err != nil {
		t.Fatal(err)
	}
	again := firstID(out)
	if !strings.Contains(errOut, "jrnl link "+again+" <ID> --supersedes") || !strings.Contains(errOut, goBun) {
		t.Errorf("want the earlier decision suggested on stderr:\n%s", errOut)
	}
	if show := e.must("", "show", goBun); strings.Contains(show, "superseded_by") {
		t.Errorf("a suggestion must not link by itself:\n%s", show)
	}
	if _, errOut, _ := e.run("", "add", "--kind", "note", "Chose Go over Bun for the CLI, a note"); strings.Contains(errOut, "earlier decisions") {
		t.Errorf("only decisions get suggestions:\n%s", errOut)
	}
}

// The server keeps its index current when the newest entry is deleted and
// another added: SQLite then reuses the deleted row's number, so the new
// row is not "after the last one indexed" and the count is unchanged.
func TestServeIndexAfterDeletingNewest(t *testing.T) {
	ai := localTestAI(t)
	t.Setenv("JRNL_HOME", t.TempDir())
	s, err := Open("p")
	if err != nil {
		t.Fatal(err)
	}
	add := func(body string) string {
		e := Entry{ID: newID(), TS: time.Now().UTC().Format(time.RFC3339), Project: "p", Kind: "note", Title: firstLine(body), Body: body}
		if err := s.Insert(e, []Chunk{{Modality: "text"}}); err != nil {
			t.Fatal(err)
		}
		if _, err := process(s, ai, e.ID, true); err != nil {
			t.Fatal(err)
		}
		return e.ID
	}
	add("Chose Go over Bun for the CLI.")
	old := add("Payments failed when the customer's card had expired.")
	sv := newServer(ai)
	if _, err := sv.search("p", "expired card payments failed", 5); err != nil {
		t.Fatal(err)
	}
	// Another process: delete the newest entry, add one.
	if err := s.Delete(old); err != nil {
		t.Fatal(err)
	}
	fresh := add("Webhook signatures were rejected after rotating the secret.")
	hits, err := sv.search("p", "webhook signatures rejected after rotating the secret", 5)
	if err != nil {
		t.Fatalf("search after delete and add: %v", err)
	}
	if len(hits) == 0 || hits[0].Entry.ID != fresh {
		t.Errorf("want the new entry first, got %v", hits)
	}
}

// A journal not searched for a while is closed and dropped from memory,
// one searched since stays, and the next search of the dropped one reloads
// it with whatever was written meanwhile.
func TestServeUnloadsIdleJournals(t *testing.T) {
	ai := localTestAI(t)
	t.Setenv("JRNL_HOME", t.TempDir())
	add := func(project, body string) string {
		s, err := Open(project)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		e := Entry{ID: newID(), TS: time.Now().UTC().Format(time.RFC3339), Project: project, Kind: "note", Title: firstLine(body), Body: body}
		if err := s.Insert(e, []Chunk{{Modality: "text"}}); err != nil {
			t.Fatal(err)
		}
		if _, err := process(s, ai, e.ID, true); err != nil {
			t.Fatal(err)
		}
		return e.ID
	}
	add("idle", "Chose Go over Bun for the CLI.")
	add("busy", "Users could not sign in after the token refresh.")
	sv := newServer(ai)
	if _, err := sv.search("idle", "why go over bun", 5); err != nil {
		t.Fatal(err)
	}
	cutoff := time.Now()
	if _, err := sv.search("busy", "sign in fails after token refresh", 5); err != nil {
		t.Fatal(err)
	}
	sv.unload(cutoff)
	if _, ok := sv.projects["idle"]; ok {
		t.Error("the journal unsearched since the cutoff is still loaded")
	}
	if _, ok := sv.projects["busy"]; !ok {
		t.Error("the journal searched after the cutoff was unloaded")
	}
	card := add("idle", "Payments failed when the customer's card had expired.")
	hits, err := sv.search("idle", "billing broke because of an expired credit card", 5)
	if err != nil {
		t.Fatalf("search after unload: %v", err)
	}
	if len(hits) == 0 || hits[0].Entry.ID != card {
		t.Errorf("want the entry written while unloaded first, got %v", hits)
	}
}

// One server per journal home. Two started together left both running,
// one unreachable and holding the model until it idled out; and a server
// of an older version kept answering after an upgrade.
func TestServeOneServerOfTheCurrentVersion(t *testing.T) {
	needLocalModel(t)
	home, err := os.MkdirTemp("/tmp", "jrs") // a socket path is limited to 104 bytes
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(home) })
	sock := filepath.Join(home, ".run", "jrnl.sock")
	env := serveEnv(home)
	health := func() string {
		r, err := socketClient(sock).Get("http://jrnl/health")
		if err != nil {
			return ""
		}
		defer r.Body.Close()
		b, _ := io.ReadAll(r.Body)
		return strings.TrimSpace(string(b))
	}
	waitFor := func(want string) {
		t.Helper()
		for range 100 {
			if strings.HasPrefix(health(), want) {
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
		t.Fatalf("health never became %q (last %q)", want, health())
	}

	var procs []*exec.Cmd
	for range 2 {
		c := exec.Command(bin, "serve")
		c.Env = env
		if err := c.Start(); err != nil {
			t.Fatal(err)
		}
		procs = append(procs, c)
	}
	t.Cleanup(func() {
		for _, c := range procs {
			c.Process.Kill()
		}
	})
	waitFor("jrnl dev ")
	exited := make(chan int, 2)
	for i, c := range procs {
		go func() { c.Wait(); exited <- i }()
	}
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Fatal("two servers started together are both still running")
	}
	select {
	case i := <-exited:
		t.Fatalf("both servers exited (second: %d); none is serving", i)
	case <-time.After(300 * time.Millisecond):
	}

	// An upgrade: the new binary's serve --background replaces the old server.
	newer := filepath.Join(t.TempDir(), "jrnl")
	if out, err := exec.Command("go", "build", "-ldflags", "-X main.version=9.9.9", "-o", newer, ".").CombinedOutput(); err != nil {
		t.Fatal(string(out))
	}
	up := exec.Command(newer, "serve", "--background")
	up.Env = env
	if out, err := up.CombinedOutput(); err != nil {
		t.Fatalf("serve --background: %v\n%s", err, out)
	}
	waitFor("jrnl 9.9.9 ")
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Error("the older server is still running after the upgrade")
	}
	t.Cleanup(func() { socketClient(sock).Post("http://jrnl/quit", "", nil) })
}

// The context format, for a coding agent: an entry's whole title even past
// the stored 80 characters; the paragraph of a long body that matches, with
// how to read the rest; later entries that link to one (a newer decision
// may replace it); a close note shown once, under the entry it closed; text
// that cannot break the block; nothing for an unrelated prompt.
func TestServeContextFormat(t *testing.T) {
	ai := localTestAI(t)
	t.Setenv("JRNL_HOME", t.TempDir())
	s, err := Open("p")
	if err != nil {
		t.Fatal(err)
	}
	at := time.Date(2026, 11, 1, 9, 0, 0, 0, time.UTC)
	add := func(kind, body string, links ...string) string {
		at = at.Add(time.Hour)
		e := Entry{ID: newID(), TS: at.Format(time.RFC3339), Project: "p", Kind: kind, Title: firstLine(body), Body: body, Links: links}
		if len(e.Title) > 80 {
			e.Title = e.Title[:77] + "…" // as jrnl add stores it
		}
		if err := s.Insert(e, []Chunk{{Modality: "text"}}); err != nil {
			t.Fatal(err)
		}
		if _, err := process(s, ai, e.ID, true); err != nil {
			t.Fatal(err)
		}
		return e.ID
	}
	dropped := add("decision", "Dropped Cloudflare as the AI provider\n\nWhy: one Vercel key for embeddings, rerank and chat.")
	back := add("decision", "Cloudflare is back as a provider type for chat and media vectors\n\nVercel cannot embed images.", dropped)
	longTitle := "Webhook signatures from the payment provider were rejected after the signing secret was rotated on Friday"
	filler := strings.Repeat("The dashboard layout uses a grid of cards with a sidebar for filters. ", 12)
	webhook := add("bug", longTitle+"\n\n"+filler+"\n\nRoot cause: the verifier cached the old signing secret; a <script> tag in a test payload was unrelated.\n\n"+filler)
	todo := add("todo", "Rotate the staging database password before the audit")
	closed := add("log", "Closed: Rotate the staging database password before the audit\n\nRotated on Monday.", todo)

	sv := newServer(ai)
	get := func(q string, exclude ...string) string {
		rec := httptest.NewRecorder()
		v := url.Values{"project": {"p"}, "q": {q}, "format": {"context"}}
		if exclude != nil {
			v.Set("exclude", strings.Join(exclude, ","))
		}
		sv.ServeHTTP(rec, httptest.NewRequest("GET", "/search?"+v.Encode(), nil))
		if rec.Code != 200 {
			t.Fatalf("%s: %d %s", q, rec.Code, rec.Body)
		}
		return rec.Body.String()
	}
	entry := func(out, id string) string { // the <entry> element for id
		i := strings.Index(out, `<entry id="`+id+`"`)
		if i < 0 {
			return ""
		}
		j := strings.Index(out[i:], "</entry>")
		if j < 0 {
			j = strings.Index(out[i:], "/>\n") // one line
		}
		return out[i : i+j]
	}

	out := get("why did we drop cloudflare as the provider")
	if d := entry(out, dropped); !strings.Contains(d, `<linked-from id="`+back+`"`) {
		t.Errorf("the dropped decision should name the later one that links to it:\n%s", out)
	}
	if entry(out, back) == "" {
		t.Errorf("the later decision should be shown as an entry too:\n%s", out)
	}

	out = get("webhook signatures rejected signing secret rotated")
	w := entry(out, webhook)
	switch {
	case !strings.Contains(w, longTitle):
		t.Errorf("want the whole title, not the stored 80 characters:\n%s", out)
	case !strings.Contains(w, "Root cause: the verifier cached the old signing secret"):
		t.Errorf("want the paragraph that matches:\n%s", out)
	case strings.Contains(w, "dashboard layout"):
		t.Errorf("want the matching paragraph only, not the long rest of the body:\n%s", out)
	case !strings.Contains(w, `rest="jrnl show `+webhook+`"`):
		t.Errorf("a cut body should say how to read the rest:\n%s", out)
	case strings.Contains(out, "<script>") || strings.Count(out, "<entry ") != strings.Count(out, "</entry>")+strings.Count(out, "/>\n"):
		t.Errorf("entry text must not open or close elements:\n%s", out)
	}

	out = get("rotate the staging database password")
	if entry(out, todo) == "" || strings.Count(out, `id="`+closed+`"`) != 1 || entry(out, closed) != "" {
		t.Errorf("the close note should appear once, under the todo it closed:\n%s", out)
	}

	// Entries the agent already has are left out.
	if out := get("rotate the staging database password", todo, "01M00000000000000000000000"); entry(out, todo) != "" {
		t.Errorf("an excluded entry was given again:\n%s", out)
	}

	// A thread: the newer decision names the one it replaces, with the
	// passage that holds its reason; the older one says what replaced it.
	old := add("decision", "Moved the job queue to Redis\n\nWhy: the Postgres queue table locked under 200 writers.")
	cur := add("decision", "Moved the job queue back to Postgres with SKIP LOCKED")
	if err := s.Link(cur, old, "supersedes"); err != nil {
		t.Fatal(err)
	}
	out = get("why did we move the job queue", old) // the newer one alone is found
	if c := entry(out, cur); !strings.Contains(c, `<supersedes id="`+old+`"`) || !strings.Contains(c, "locked under 200 writers") {
		t.Errorf("the newer decision should carry the one it replaces and its reason:\n%s", out)
	}
	out = get("why did we move the job queue", cur) // the older one alone is found
	if o := entry(out, old); !strings.Contains(o, `superseded-by="`+cur+`"`) || !strings.Contains(o, `<superseded-by id="`+cur+`"`) {
		t.Errorf("the older decision should say what replaced it:\n%s", out)
	}

	if out := get("what is the capital of portugal"); out != "" {
		t.Errorf("an unrelated prompt should get nothing:\n%s", out)
	}
}

// A decision can say which earlier one it replaces, at add time or later
// with jrnl link; show gives both directions; a wrong link can be changed
// or removed.
func TestSupersedes(t *testing.T) {
	e := newEnv(t)
	dropped := firstID(e.must("Dropped Cloudflare as the AI provider", "add", "--kind", "decision"))
	kept := firstID(e.must("Cloudflare kept as a provider type", "add", "--kind", "decision", "--supersedes", dropped[:12]))
	if show := e.must("", "show", dropped); !strings.Contains(show, "superseded_by: ["+kept+"]") || strings.Contains(show, "linked_from") {
		t.Errorf("the older decision should say what replaced it, and not as a plain link:\n%s", show)
	}
	if show := e.must("", "show", kept); !strings.Contains(show, "supersedes: ["+dropped+"]") {
		t.Errorf("the newer decision should say what it replaces:\n%s", show)
	}

	a := firstID(e.must("Use Postgres for the ledger", "add", "--kind", "decision"))
	b := firstID(e.must("Use SQLite for the ledger", "add", "--kind", "decision"))
	e.must("", "link", b, a)
	if show := e.must("", "show", a); !strings.Contains(show, "linked_from: ["+b+"]") {
		t.Errorf("jrnl link should relate them:\n%s", show)
	}
	e.must("", "link", b, a, "--supersedes")
	if show := e.must("", "show", a); !strings.Contains(show, "superseded_by: ["+b+"]") || strings.Contains(show, "linked_from") {
		t.Errorf("--supersedes should change the link's type:\n%s", show)
	}
	e.must("", "link", a, b, "--rm") // either direction
	if show := e.must("", "show", a); strings.Contains(show, b) {
		t.Errorf("--rm should remove the link:\n%s", show)
	}
	if _, _, err := e.run("", "link", a, b, "--rm"); err == nil {
		t.Error("removing a link that is not there should fail")
	}
	if _, _, err := e.run("", "link", a, a); err == nil {
		t.Error("an entry linking to itself should fail")
	}
}

// Journals from before typed links: a close note's link becomes closes,
// known from closed_by; other links stay plain.
func TestUpgradeTypesLinks(t *testing.T) {
	e := newEnv(t)
	todo := firstID(e.must("Rotate the staging password", "add", "--kind", "todo"))
	other := firstID(e.must("Staging password policy", "add", "--link", todo))
	e.must("", "close", todo, "--note", "Rotated on Monday")
	db, err := sql.Open("sqlite", filepath.Join(e.home, "test", "journal.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, q := range []string{`ALTER TABLE links DROP COLUMN type`, `PRAGMA user_version = 4`} {
		if _, err := db.Exec(q); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	e.must("", "list")
	types := map[string]string{}
	rows, err := db.Query(`SELECT src, type FROM links WHERE dst = ?`, todo)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var src, typ string
		rows.Scan(&src, &typ)
		types[src] = typ
	}
	rows.Close()
	note := e.must("", "show", todo)
	closedBy := note[strings.Index(note, "closed_by: ")+len("closed_by: "):][:26]
	if types[closedBy] != "closes" || types[other] != "relates" || len(types) != 2 {
		t.Errorf("want the close note's link closes and the other relates, got %v", types)
	}
}

// needLocalModel skips where the local embedding model is not downloaded.
func needLocalModel(t testing.TB) {
	t.Helper()
	if !localDownloaded() {
		t.Skip("the local embedding model is not downloaded; run jrnl reindex once")
	}
}

// localTestAI embeds with the real local model and nothing else.
func localTestAI(t testing.TB) *AI {
	t.Helper()
	needLocalModel(t)
	// serving: as inside jrnl serve, it never asks or starts a server.
	return &AI{local: true, serving: true, embedModel: localModel, embedID: localID, hc: http.DefaultClient, attempts: 1,
		from: map[string]string{}, chatErr: errNoChat, mediaErr: errNoMediaEmbed}
}

// serveEnv runs jrnl serve on home with the local model and none of this
// machine's settings.
func serveEnv(home string) []string {
	return append(os.Environ(), "JRNL_HOME="+home, "JRNL_CONFIG="+filepath.Join(home, "no-config.toml"),
		"JRNL_ENV_FILE="+filepath.Join(home, "no-env"), "JRNL_EMBED_URL=", "JRNL_EMBED_MODEL=", "JRNL_MODEL_DIR="+localDir())
}

// With no embeddings API set up, jrnl embeds in its own process: a reworded
// question finds the entry, and doctor says where the model is. Uses the
// real model, so it runs only where it is downloaded.
func TestBuiltInEmbedder(t *testing.T) {
	needLocalModel(t)
	e := newEnv(t)
	quitServerAt(t, e.home)
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL=", "JRNL_MODEL_DIR=" + localDir()}
	login := firstID(e.must("Users could not sign in after the token refresh.", "add"))
	e.must("Chose Go over Bun for the CLI.", "add", "--kind", "decision")
	if out := e.must("", "search", "--fast", "why do logins fail after refreshing credentials"); !strings.HasPrefix(out, login) {
		t.Errorf("a reworded question should find the sign-in entry first:\n%s", out)
	}
	if doc := e.must("", "doctor"); !strings.Contains(doc, "bge-small-en-v1.5 in this binary") || strings.Contains(doc, "waiting") {
		t.Errorf("doctor should show the built-in model with nothing waiting:\n%s", doc)
	}
}

// Entries with no vector from the server's embedder (saved by an older
// jrnl, by another embedder, or before the model was downloaded) are
// embedded by jrnl serve in the background and stored, and found by
// meaning once they are.
func TestServeBackfill(t *testing.T) {
	ai := localTestAI(t)
	t.Setenv("JRNL_HOME", t.TempDir())
	backfillDelay = 0
	t.Cleanup(func() { backfillDelay = 2 * time.Second })
	s, err := Open("p")
	if err != nil {
		t.Fatal(err)
	}
	login := Entry{ID: newID(), TS: time.Now().UTC().Format(time.RFC3339), Project: "p", Kind: "bug",
		Title: "Users could not sign in after the token refresh", Body: "Users could not sign in after the token refresh"}
	other := Entry{ID: newID(), TS: time.Now().UTC().Format(time.RFC3339), Project: "p", Kind: "decision",
		Title: "Chose Go over Bun for the CLI", Body: "Chose Go over Bun for the CLI"}
	for _, e := range []Entry{login, other} {
		if err := s.Insert(e, []Chunk{{Modality: "text"}}); err != nil {
			t.Fatal(err)
		}
	}
	// One of them has a vector, from an embedder the server does not use.
	if _, err := s.db.Exec(`UPDATE chunks SET vec = ?, embedder = 'old-model@elsewhere' WHERE entry_id = ?`,
		encodeVec(make([]float32, 8)), other.ID); err != nil {
		t.Fatal(err)
	}
	sv := newServer(ai)
	q := "why do logins fail after refreshing credentials"
	for i := 0; ; i++ {
		sv.mu.Lock()
		hits, err := sv.search("p", q, 5)
		sv.mu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
		if len(hits) > 0 && hits[0].Entry.ID == login.ID && slices.Contains(hits[0].Via, "vector") {
			break
		}
		if i == 100 {
			t.Fatalf("not found by meaning after 5 s of backfill: %v", hits)
		}
		time.Sleep(50 * time.Millisecond)
	}
	var n int
	s.db.QueryRow(`SELECT count(*) FROM chunks WHERE embedder = ? AND vec IS NOT NULL`, localID).Scan(&n)
	if n != 2 {
		t.Errorf("want both entries' vectors stored by the server's embedder, got %d", n)
	}
}

// A model file that does not load leaves entries waiting for reindex, not
// marked as text that can never be embedded; with the model fixed, reindex
// embeds them.
func TestBrokenModelWaits(t *testing.T) {
	needLocalModel(t)
	e := newEnv(t)
	quitServerAt(t, e.home)
	broken := t.TempDir()
	os.WriteFile(filepath.Join(broken, "model.onnx"), []byte("junk"), 0o644)
	tok, _ := os.ReadFile(filepath.Join(localDir(), "tokenizer.json"))
	os.WriteFile(filepath.Join(broken, "tokenizer.json"), tok, 0o644)
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL=", "JRNL_MODEL_DIR=" + broken}
	id := firstID(e.must("Users could not sign in after the token refresh.", "add"))
	if doc := e.must("", "doctor"); !strings.Contains(doc, "1 item(s) waiting") {
		t.Errorf("the entry should wait for reindex:\n%s", doc)
	}
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL=", "JRNL_MODEL_DIR=" + localDir()}
	if out := e.must("", "reindex"); !strings.Contains(out, "embedded 1 ") {
		t.Fatalf("reindex with the model fixed: %q", out)
	}
	if out := e.must("", "search", "--fast", "why do logins fail after refreshing credentials"); !strings.HasPrefix(out, id) {
		t.Errorf("not found by meaning after reindex:\n%s", out)
	}
}

// reindex downloads the model's files from the pinned revision, checks
// them, and refuses a file that is not the one expected.
func TestModelDownload(t *testing.T) {
	needLocalModel(t)
	var corrupt bool
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, f := range localFiles {
			if r.URL.Path == "/"+localRepo+"/resolve/"+localRev+"/"+f[1] {
				if corrupt && f[0] == "model.onnx" {
					io.WriteString(w, "<html>please log in to the hotel wifi</html>")
					return
				}
				http.ServeFile(w, r, filepath.Join(localDir(), f[0]))
				return
			}
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(hub.Close)
	e := newEnv(t)
	quitServerAt(t, e.home)
	dir := filepath.Join(t.TempDir(), "model")
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL=", "JRNL_MODEL_DIR=" + dir, "JRNL_MODELS_URL=" + hub.URL}
	corrupt = true // before add, whose background server downloads too
	e.must("Users could not sign in after the token refresh.", "add")
	if _, errOut, _ := e.run("", "reindex"); !strings.Contains(errOut, "SHA-256") {
		t.Errorf("a wrong file should be refused: %q", errOut)
	}
	if _, err := os.Stat(filepath.Join(dir, "model.onnx")); err == nil {
		t.Error("the wrong file was kept")
	}
	corrupt = false
	if out := e.must("", "reindex"); !strings.Contains(out, "embedded 1 ") {
		t.Fatalf("reindex after downloading: %q", out)
	}
}

// Text past the model's 512-token window is cut by tokens, not characters:
// dense code and accented text alike end within the window, as a prefix of
// what was given; short text is left alone.
func TestLocalTokenWindow(t *testing.T) {
	needLocalModel(t)
	p, err := local.load(false)
	if err != nil {
		t.Fatal(err)
	}
	code := strings.Repeat("src/server/services/ServiceMarket.luau:142 0x7f3e9ab2c4 ", 80)
	accents := strings.Repeat("Résumé café naïve façade über Ångström — ", 120)
	short := "Users could not sign in after the token refresh."
	in := []string{code, accents, short}
	out := fitTokens(p, in)
	batch := backends.NewBatch(len(out))
	defer batch.Destroy()
	backends.TokenizeInputs(batch, p.Model.Tokenizer, out)
	for i, x := range batch.Input {
		if n := len(x.TokenIDs); n > 512 {
			t.Errorf("text %d: %d tokens after cutting, want at most 512", i, n)
		}
		if !strings.HasPrefix(in[i], out[i]) || (i < 2 && len(out[i]) < 200) {
			t.Errorf("text %d: want a long prefix of the original, got %d bytes", i, len(out[i]))
		}
	}
	if out[2] != short {
		t.Errorf("short text changed: %q", out[2])
	}
	if _, err := local.embed([]string{code}, false); err != nil {
		t.Errorf("embedding long text: %v", err)
	}
}

// quitServerAt stops the jrnl serve a test's commands started under home,
// waiting for one that is still starting (it holds the lock before it
// listens).
func quitServerAt(t *testing.T, home string) {
	t.Cleanup(func() {
		run := filepath.Join(home, ".run")
		for range 100 {
			if r, err := socketClient(filepath.Join(run, "jrnl.sock")).Post("http://jrnl/quit", "", nil); err == nil {
				r.Body.Close()
				return
			}
			lock, err := os.OpenFile(filepath.Join(run, "jrnl.lock"), os.O_RDWR, 0)
			if err != nil {
				return // no server ever started here
			}
			free := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) == nil
			lock.Close()
			if free {
				return
			}
			time.Sleep(50 * time.Millisecond)
		}
		t.Errorf("a jrnl serve under %s did not quit", home)
	})
}

// The server closes the model after a backfill to free its buffers; a
// search embedding at that moment still gets its vectors.
func TestLocalResetDuringEmbedding(t *testing.T) {
	needLocalModel(t)
	texts := []string{"Users could not sign in after the token refresh.", "Chose Go over Bun for the CLI.",
		"Refunds failed when the card network timed out.", "The tutorial card stayed after the last step.", "Rival studios poach developers."}
	want, err := local.embed(texts, false)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				local.reset()
			}
		}
	}()
	for range 5 {
		got, err := local.embed(texts, false)
		if err != nil {
			t.Fatal(err)
		}
		for i := range texts {
			if c := dot(got[i], want[i]); c < 0.9999 {
				t.Errorf("text %d: vector changed across a reset, cosine %.5f", i, c)
			}
		}
	}
	close(stop)
	wg.Wait()
}

// Only jrnl add starts a journal. A command that reads said nothing and left
// an empty journal behind for every folder it was run in; doctor still works
// before the first entry, since it is how setup is checked.
func TestReadingStartsNoJournal(t *testing.T) {
	e := newEnv(t)
	for _, args := range [][]string{{"list"}, {"search", "anything"}, {"stats"}, {"show", "01M"}, {"reindex"}} {
		if _, errOut, err := e.run("", args...); err == nil || !strings.Contains(errOut, "no journal for test yet") {
			t.Errorf("jrnl %v: want an error saying there is no journal yet, got %v %q", args, err, errOut)
		}
	}
	if doc := e.must("", "doctor"); !strings.Contains(doc, "0 entries") {
		t.Errorf("doctor should work before the first entry:\n%s", doc)
	}
	if _, err := os.Stat(filepath.Join(e.home, "test")); err == nil {
		t.Error("reading created a journal")
	}
	id := firstID(e.must("First entry.", "add"))
	if out := e.must("", "list"); !strings.HasPrefix(out, id) {
		t.Errorf("list after the first add:\n%s", out)
	}
}

// A mistyped kind or an empty tag is refused, not saved as a new kind or
// tag that every later filter and count then carries.
func TestAddRefusesUnknownKindAndBadTag(t *testing.T) {
	e := newEnv(t)
	for _, c := range []struct {
		args []string
		want string
	}{
		{[]string{"add", "--kind", "decison", "text"}, `not "decison"`},
		{[]string{"add", "--kind", "summary", "text"}, `not "summary"`},
		{[]string{"add", "--tag", "", "text"}, "--tag is one word"},
		{[]string{"add", "--tag", "two words", "text"}, "--tag is one word"},
	} {
		if out, errOut, err := e.run("", c.args...); err == nil || !strings.Contains(errOut, c.want) || out != "" {
			t.Errorf("jrnl %v: want an error with %q and no ID, got %v %q %q", c.args, c.want, err, errOut, out)
		}
	}
	if _, err := os.Stat(filepath.Join(e.home, "test")); err == nil {
		t.Error("a refused add started a journal")
	}
	e.must("text", "add", "--kind", "milestone", "--tag", "release")
}

// A config.toml that jrnl refuses is an error from summarize too; it
// crashed with a nil pointer instead.
func TestSummarizeWithBadConfig(t *testing.T) {
	e := newEnv(t)
	e.must("Shipped the importer.", "add")
	e.config, e.noChat = "[chat]\nmodel = \"m\"\n", true // the config's chat, not the test default
	_, errOut, err := e.run("", "summarize", "--on", "today")
	if err == nil || !strings.Contains(errOut, "config [chat] needs a provider") || strings.Contains(errOut, "panic") {
		t.Errorf("want the config error, got %v %q", err, errOut)
	}
}

// jrnl rm is the way out after a secret was written down, so what it
// removes must be gone from the files, not only from the listings: the
// entry's text, an attachment no other entry uses, and a saved summary
// that cites it. An attachment another entry still uses stays.
func TestRmErases(t *testing.T) {
	e := newEnv(t)
	const secret, fileSecret = "Zqxmarker9173", "Wkvattach5520"
	note := filepath.Join(t.TempDir(), "notes.txt")
	os.WriteFile(note, []byte("token "+fileSecret), 0o644)
	bad := firstID(e.must("Deploy key rotated.\n\nThe key is "+secret+" for now.", "add", "--tag", secret, "--attach", note, "--at", "yesterday"))
	keeps := firstID(e.must("Same notes file, kept.", "add", "--attach", note, "--at", "yesterday"))
	todo := firstID(e.must("Rotate the key.", "add", "--kind", "todo"))
	e.must("", "close", todo, "--note", "done, see "+secret)
	closeNote := firstID(e.must("", "list", "--kind", "log"))
	e.must("", "summarize", "--on", "yesterday")
	if out := e.must("", "list", "--kind", "summary"); out == "" {
		t.Fatal("no summary was saved")
	}

	dir := filepath.Join(e.home, "test")
	holds := func(marker string) (where []string) {
		filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				b, _ := os.ReadFile(p)
				if bytes.Contains(bytes.ToLower(b), []byte(strings.ToLower(marker))) {
					where = append(where, filepath.Base(p))
				}
			}
			return nil
		})
		return where
	}
	if len(holds(secret)) == 0 || len(holds(fileSecret)) == 0 {
		t.Fatal("the markers are not in the journal's files before rm; the check below would prove nothing")
	}

	out, errOut, err := e.run("", "rm", bad, closeNote)
	if err != nil || !strings.Contains(out, bad+"  removed") || !strings.Contains(errOut, "1 saved summary") {
		t.Fatalf("rm: %v\n%s\n%s", err, out, errOut)
	}
	if where := holds(secret); len(where) > 0 {
		t.Errorf("the removed text is still in %v", where)
	}
	if where := holds(fileSecret); len(where) != 1 {
		t.Errorf("the attachment another entry uses should be the only copy left, found %v", where)
	}
	if out := e.must("", "list", "--limit", "100"); strings.Contains(out, bad) || strings.Contains(out, "summary") || !strings.Contains(out, keeps) {
		t.Errorf("list after rm:\n%s", out)
	}
	if out := e.must("", "show", todo); strings.Contains(out, "closed_by") || !strings.Contains(out, "status: closed") {
		t.Errorf("the todo should stay closed without its removed note:\n%s", out)
	}
	e.must("", "rm", keeps)
	if where := holds(fileSecret); len(where) > 0 {
		t.Errorf("the attachment no entry uses is still in %v", where)
	}
}

// A server started under one embedder must not undo a reindex done after
// config.toml named another: each chunk holds one vector, and the server
// embedded every entry again its own way on the next search. It quits
// instead, and the vectors stay.
func TestServeQuitsWhenTheEmbedderChanges(t *testing.T) {
	e := newEnv(t)
	home, err := os.MkdirTemp("/tmp", "jrs") // a socket path is limited to 104 bytes
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(home) })
	e.home, e.dir = home, filepath.Join(t.TempDir(), "test")
	os.MkdirAll(e.dir, 0o755)
	quitServerAt(t, home)
	conf := func(model string) string {
		return "[providers.o]\nurl = \"" + e.url + "/v1\"\n[embed]\nprovider = \"o\"\nmodel = \"" + model + "\"\n"
	}
	e.extra = []string{"JRNL_EMBED_URL=", "JRNL_EMBED_MODEL="} // the config's embedder, not the test default
	e.config = conf("emb-a")
	e.must("Login form rejects valid credentials.", "add")
	e.must("Chose Go over Bun for the CLI.", "add")
	e.must("", "serve", "--background")
	sock := filepath.Join(home, ".run", "jrnl.sock")
	up := func() bool {
		_, _, ok := runningServer(sock)
		return ok
	}
	for i := 0; !up(); i++ {
		if i == 100 {
			t.Fatal("the server did not start")
		}
		time.Sleep(50 * time.Millisecond)
	}

	e.config = conf("emb-b")
	e.must("", "reindex")
	embedders := func() string {
		out, err := exec.Command("sqlite3", filepath.Join(home, "test", "journal.db"),
			"select group_concat(distinct embedder) from chunks").Output()
		if err != nil {
			t.Fatal(err)
		}
		return strings.TrimSpace(string(out))
	}
	want := embedders()
	if !strings.HasPrefix(want, "emb-b@") {
		t.Fatalf("reindex should have embedded with the new model, got %q", want)
	}
	if r, err := socketClient(sock).Get("http://jrnl/search?project=test&q=login+credentials"); err == nil {
		r.Body.Close()
	}
	for i := 0; up(); i++ {
		if i == 120 {
			t.Fatal("the server is still running 6 s after a search under changed settings")
		}
		time.Sleep(50 * time.Millisecond)
	}
	if got := embedders(); got != want {
		t.Errorf("the server embedded again with its old model: %q, want %q", got, want)
	}
}
