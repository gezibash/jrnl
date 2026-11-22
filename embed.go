package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	// Text is embedded locally. On the 20,000-entry test, Ollama answered in
	// 40 ms (worst 0.85 s) against 1.5 s (worst 10 s, with 429s) on Workers AI,
	// and found as many right answers (21-22 of 30 against 21).
	ollamaModel = "embeddinggemma-2:270m"
	// The model's window is 8,192 tokens; this keeps long entries well inside it.
	maxDocChars = 24000
	// Media vectors at full size. On a 20,000-entry test, 256 dims put a
	// reworded question's answer first 2 times in 30; 768 did 10 in 30.
	mediaDims = 768
)

// Each vector records the embedder that made it, and search compares only
// vectors from the same one. Vectors from different embedders, or the same
// model reached another way, are not comparable (measured: cosine 0.74-0.81
// for one text sent two ways), and nothing else would catch the mix-up. A
// text embedder is its model and the host serving it.
//
// embedderID names a text embedder: the model and the host that serves it.
func embedderID(base, model string) string {
	host := base
	if u, err := url.Parse(base); err == nil && u.Host != "" {
		host = u.Host
	}
	h, port, err := net.SplitHostPort(host)
	if err == nil && (h == "127.0.0.1" || h == "::1") {
		host = net.JoinHostPort("localhost", port)
	}
	return model + "@" + host
}

// Input is one thing to embed: a document (Title and Text), a search query
// (Query set), or a media file.
type Input struct {
	Title, Text string
	Query       bool
	File, Mime  string
}

// format writes text the way the embedding model was trained to read it.
// Each family has its own convention, from its model card.
func format(model string, in Input) string {
	m := strings.ToLower(model)
	var t string
	switch {
	case m == localModel:
		t = in.Title + "\n\n" + in.Text
		if in.Query {
			t = localQueryPrefix + in.Text
		}
		if len(t) > localMaxChars {
			t = strings.ToValidUTF8(t[:localMaxChars], "")
		}
		return t
	case strings.Contains(m, "embeddinggemma"):
		t = "title: " + in.Title + " | text: " + in.Text
		if in.Query {
			t = "task: search result | query: " + in.Text
		}
	case strings.Contains(m, "qwen3-embedding"):
		// Qwen3 instructs the query and leaves the document plain.
		t = in.Title + "\n\n" + in.Text
		if in.Query {
			t = "Instruct: Given a question about a software project, retrieve the journal entries that answer it\nQuery: " + in.Text
		}
	default:
		t = in.Title + "\n\n" + in.Text
		if in.Query {
			t = in.Text
		}
	}
	if len(t) > maxDocChars {
		t = t[:maxDocChars]
	}
	return t
}

// AI reaches the models jrnl uses, all optional:
//   - text embeddings: any OpenAI-compatible /embeddings API; by default
//     Ollama on this machine.
//   - chat, for captions, reranking and summaries: any OpenAI-compatible
//     /chat/completions API.
//   - a reranker with a Cohere-style /rerank API.
//   - media vectors (search --like by look): Cloudflare Workers AI's own API.
type AI struct {
	embedURL, embedKey, embedModel, embedID string
	embedSet                                bool // set by config or JRNL_EMBED_URL, not a default
	// local: no embeddings API is set up, so bge-small runs in this process.
	// download: this command may download its model (reindex, serve).
	local, download bool

	chatURL, chatKey, chatModel string

	// Vectors of the raw images, video and audio, for search --like. Only
	// Cloudflare Workers AI is supported; without it, --like compares
	// descriptions.
	media    endpoint
	mediaID  string
	mediaErr error

	from map[string]string // role: where it was set, for jrnl doctor

	// A dedicated reranker with a Cohere-style /rerank API, optional.
	rerankURL, rerankKey, rerankModel string
	chatExtra                         map[string]any // merged into each chat request
	chatErr                           error          // set when no chat API is set up

	hc       *http.Client
	attempts int
}

// Interactive returns a client for a search the user is waiting on: one
// attempt, hard deadline. Measured over 61 searches, 1 rerank call in 10
// took 7-13 s and the worst 22 s; past the deadline, search uses its
// fallback instead of waiting.
func (e *AI) Interactive(deadline time.Duration) *AI {
	c := *e
	c.hc = &http.Client{Timeout: deadline}
	c.attempts = 1
	return &c
}

// NewAI reads its settings from the environment:
//
//	(see config.go for config.toml; these variables override it)
//	JRNL_EMBED_URL, JRNL_EMBED_KEY, JRNL_EMBED_MODEL  text embeddings
//	    (default: bge-small-en-v1.5 in this process; a model with no URL
//	    means Ollama at OLLAMA_HOST or localhost:11434)
//	JRNL_CHAT_URL, JRNL_CHAT_KEY, JRNL_CHAT_MODEL     chat
//	JRNL_CHAT_EXTRA                                   JSON fields added to each chat request
//	JRNL_RERANK_URL, JRNL_RERANK_KEY, JRNL_RERANK_MODEL  a reranker with a
//	    Cohere-style /rerank API (Cohere, Voyage, Jina, Vercel AI Gateway)
//	JRNL_ENV_FILE                                     more of these settings (see envFile)

// ollamaURL is the OpenAI-compatible API of Ollama at OLLAMA_HOST.
func ollamaURL() string {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "localhost:11434"
	}
	if !strings.Contains(host, "://") {
		host = "http://" + host
	}
	return strings.TrimSuffix(host, "/") + "/v1"
}

// envFile is where jrnl looks for API keys that are not in the environment:
// JRNL_ENV_FILE, or ~/.config/jrnl/.env. It can be a 1Password Environment
// mounted as a local .env file, which is a named pipe.
func envFile() string {
	if p := os.Getenv("JRNL_ENV_FILE"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "jrnl", ".env")
}

// loadEnvFile sets each KEY=VALUE from the file that the environment does
// not already set. A missing file is fine. A named pipe whose writer does not
// answer within wait (1Password closed, or its prompt not yet approved) is
// skipped with a warning, so jrnl still saves entries, without the keys.
func loadEnvFile(path string, wait time.Duration) {
	info, err := os.Stat(path)
	if err != nil {
		return
	}
	pipe := info.Mode()&os.ModeNamedPipe != 0
	type result struct {
		data []byte
		err  error
	}
	done := make(chan result, 1)
	go func() {
		// 1Password serves one reader at a time: of 5 jrnl processes reading
		// at once, 2 got an empty file. An empty pipe is read again.
		var data []byte
		var err error
		for i := range 20 {
			if data, err = os.ReadFile(path); err != nil || !pipe || len(bytes.TrimSpace(data)) > 0 {
				break
			}
			time.Sleep(time.Duration(25*(i+1)) * time.Millisecond)
		}
		done <- result{data, err}
	}()
	var r result
	select {
	case r = <-done:
	case <-time.After(wait):
		r.err = fmt.Errorf("no answer in %v; is 1Password unlocked, and was its prompt approved?", wait)
	}
	if r.err != nil {
		fmt.Fprintf(os.Stderr, "jrnl: skipped %s: %v\n", path, r.err)
		return
	}
	for _, line := range strings.Split(string(r.data), "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "export "))
		k, v, ok := strings.Cut(line, "=")
		if !ok || strings.HasPrefix(line, "#") {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		if len(v) >= 2 && (v[0] == '"' || v[0] == '\'') && v[len(v)-1] == v[0] {
			v = v[1 : len(v)-1]
		}
		if os.Getenv(k) == "" {
			os.Setenv(k, v)
		}
	}
}

var loadEnvOnce sync.Once

func NewAI() (*AI, error) {
	loadEnvOnce.Do(func() { loadEnvFile(envFile(), 10*time.Second) })
	e := &AI{hc: &http.Client{Timeout: 3 * time.Minute}, attempts: 4, from: map[string]string{}}
	c, err := loadConfig(configFile())
	if err != nil {
		return nil, err
	}

	emb, err := c.resolve("embed", c.Embed, "EMBED")
	if err != nil {
		return nil, err
	}
	e.embedURL, e.embedKey, e.embedModel, e.embedSet = emb.url, emb.key, emb.model, emb.url != ""
	switch {
	case !e.embedSet && e.embedModel == "":
		e.local, e.embedModel, e.embedID = true, localModel, localID
	case !e.embedSet:
		e.embedURL = ollamaURL()
		fallthrough
	default:
		e.embedID = embedderID(e.embedURL, e.embedModel)
	}
	e.from["embed"] = emb.from

	chat, err := c.resolve("chat", c.Chat, "CHAT")
	if err != nil {
		return nil, err
	}
	e.chatURL, e.chatKey, e.chatModel, e.chatExtra, e.from["chat"] = chat.url, chat.key, chat.model, chat.extra, chat.from
	if e.chatURL == "" {
		e.chatErr = errNoChat
	}
	if x := os.Getenv("JRNL_CHAT_EXTRA"); x != "" {
		e.chatExtra = nil
		if err := json.Unmarshal([]byte(x), &e.chatExtra); err != nil {
			return nil, fmt.Errorf("JRNL_CHAT_EXTRA must be a JSON object: %w", err)
		}
	}

	rr, err := c.resolve("rerank", c.Rerank, "RERANK")
	if err != nil {
		return nil, err
	}
	e.rerankURL, e.rerankKey, e.rerankModel, e.from["rerank"] = rr.url, rr.key, rr.model, rr.from

	if e.media, err = c.resolve("media_embed", c.MediaEmbed, ""); err != nil {
		return nil, err
	}
	if e.media.run == "" {
		e.mediaErr = errNoMediaEmbed
	} else {
		// Workers AI gives the same vectors on any account.
		e.mediaID, e.from["media_embed"] = "cloudflare/"+e.media.model, e.media.from
	}
	return e, nil
}

// Embed sends text to the embeddings API and media files to the media
// embedder. The two kinds of vector live in different spaces and are never
// compared with each other.
func (e *AI) Embed(ins []Input) ([][]float32, error) {
	var texts, media []int
	for i, in := range ins {
		if in.File == "" {
			texts = append(texts, i)
		} else {
			media = append(media, i)
		}
	}
	out := make([][]float32, len(ins))
	if len(texts) > 0 {
		input := make([]string, len(texts))
		for j, i := range texts {
			input[j] = format(e.embedModel, ins[i])
		}
		vecs, err := e.embedText(input)
		if err != nil {
			return nil, err
		}
		for j, i := range texts {
			out[i] = vecs[j]
		}
	}
	if len(media) > 0 {
		if e.mediaErr != nil {
			return nil, e.mediaErr
		}
		msgs := make([]any, len(media))
		for j, i := range media {
			content, err := messageContent(ins[i])
			if err != nil {
				return nil, err
			}
			msgs[j] = []any{map[string]any{"role": "user", "content": content}}
		}
		vecs, err := e.embedMedia(map[string]any{"messages": msgs, "dimensions": mediaDims}, len(media))
		if err != nil {
			return nil, err
		}
		for j, i := range media {
			out[i] = vecs[j]
		}
	}
	return out, nil
}

// embedMedia calls Workers AI's own API, which takes images, video and audio
// as data URIs. Text sent this way embeds differently from the same text
// sent as "input" (cosine 0.74-0.81), so only files go here.
func (e *AI) embedMedia(req map[string]any, n int) ([][]float32, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var res struct {
		Success bool `json:"success"`
		Errors  []struct {
			Message string `json:"message"`
		} `json:"errors"`
		Result struct {
			Data []struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		} `json:"result"`
	}
	if err := e.post(e.media.run+"/"+e.media.model, e.media.key, body, &res); err != nil {
		return nil, fmt.Errorf("media embeddings: %w", err)
	}
	if !res.Success {
		var m []string
		for _, er := range res.Errors {
			m = append(m, er.Message)
		}
		return nil, fmt.Errorf("media embeddings: %s", strings.Join(m, "; "))
	}
	out := make([][]float32, n)
	for _, d := range res.Result.Data {
		if d.Index >= 0 && d.Index < n {
			out[d.Index] = normalize(d.Embedding)
		}
	}
	for i, v := range out {
		if len(v) == 0 {
			return nil, fmt.Errorf("media embeddings: no vector for input %d", i)
		}
	}
	return out, nil
}

func messageContent(in Input) (any, error) {
	b, err := os.ReadFile(in.File)
	if err != nil {
		return nil, permanent{err}
	}
	uri := "data:" + in.Mime + ";base64," + base64.StdEncoding.EncodeToString(b)
	for _, kind := range []string{"image", "video", "audio"} {
		if strings.HasPrefix(in.Mime, kind+"/") {
			return []any{map[string]any{"type": kind + "_url", kind + "_url": map[string]string{"url": uri}}}, nil
		}
	}
	return nil, permanent{fmt.Errorf("cannot embed %s", in.Mime)}
}

func (e *AI) embedText(input []string) ([][]float32, error) {
	if e.local {
		vecs, err := local.embed(input, e.download)
		if errors.Is(err, errNoModel) {
			// jrnl serve downloads the model and embeds what waited.
			startServer()
		}
		return vecs, err
	}
	body, _ := json.Marshal(map[string]any{"model": e.embedModel, "input": input})
	var res struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	err := e.post(e.embedURL+"/embeddings", e.embedKey, body, &res)
	var down unreachable
	var he httpError
	switch {
	case errors.As(err, &down):
		return nil, embedDown{e.embedURL, e.embedSet, err}
	case errors.As(err, &he) && he.status == 404 && !e.embedSet:
		return nil, fmt.Errorf("ollama has no %s model (run `ollama pull %s`)", e.embedModel, e.embedModel)
	case err != nil:
		return nil, fmt.Errorf("embeddings at %s: %w", e.embedURL, err)
	}
	out := make([][]float32, len(input))
	for _, d := range res.Data {
		if d.Index >= 0 && d.Index < len(out) {
			out[d.Index] = normalize(d.Embedding)
		}
	}
	for i, v := range out {
		if len(v) == 0 {
			return nil, fmt.Errorf("embeddings at %s: no vector for input %d", e.embedURL, i)
		}
	}
	return out, nil
}

// httpError is a reply other than 200, with the API's own message.
type httpError struct {
	status int
	msg    string
}

func (h httpError) Error() string { return fmt.Sprintf("HTTP %d: %s", h.status, h.msg) }

// unreachable means nothing listens at the address. Retrying won't help,
// and a missing local Ollama must not cost seconds of backoff on every add.
type unreachable struct{ error }

// post sends JSON to an API and decodes the reply. It retries rate limits,
// server errors and network failures with backoff, so a burst from many
// parallel sessions slows down instead of failing.
func (e *AI) post(url, key string, body []byte, out any) error {
	var last error
	for attempt := 0; attempt < e.attempts; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(1<<(attempt-1)) * time.Second)
		}
		req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := e.hc.Do(req)
		if errors.Is(err, syscall.ECONNREFUSED) {
			return unreachable{err}
		}
		if err != nil {
			last = err
			continue
		}
		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			last = httpError{resp.StatusCode, apiMessage(data)}
			continue
		}
		if err != nil {
			return err
		}
		if resp.StatusCode != 200 {
			err := httpError{resp.StatusCode, apiMessage(data)}
			// Only statuses about the input itself are permanent. A bad token
			// (401/403) or a missing model (404) must stay retryable, or every
			// entry written meanwhile would be skipped by reindex forever.
			switch resp.StatusCode {
			case 400, 413, 415, 422:
				return permanent{err}
			}
			return err
		}
		return json.Unmarshal(data, out)
	}
	return last
}

// apiMessage pulls the error message out of a reply in the shapes OpenAI,
// Cloudflare and Ollama use, or else returns the start of the body.
func apiMessage(data []byte) string {
	var r struct {
		Error  json.RawMessage `json:"error"`
		Errors []struct {
			Message string `json:"message"`
		} `json:"errors"`
	}
	if json.Unmarshal(data, &r) == nil {
		var m struct {
			Message string `json:"message"`
		}
		var str string
		switch {
		case json.Unmarshal(r.Error, &m) == nil && m.Message != "":
			return truncate(m.Message, 300)
		case json.Unmarshal(r.Error, &str) == nil && str != "":
			return truncate(str, 300)
		case len(r.Errors) > 0:
			return truncate(r.Errors[0].Message, 300)
		}
	}
	body := strings.TrimSpace(string(data))
	if strings.HasPrefix(body, "<") {
		return "the reply is a web page, not an API answer; check the URL"
	}
	return truncate(body, 300)
}

// Caption asks the chat model to describe one or more images. Video frames
// are sent as images, which every vision chat model takes.
func (e *AI) Caption(images []string, prompt string) (string, error) {
	var content []any
	for _, path := range images {
		b, err := os.ReadFile(path)
		if err != nil {
			return "", permanent{err}
		}
		mt := mediaTypes[strings.ToLower(filepath.Ext(path))]
		content = append(content, map[string]any{"type": "image_url",
			"image_url": map[string]string{"url": "data:" + mt + ";base64," + base64.StdEncoding.EncodeToString(b)}})
	}
	content = append(content, map[string]any{"type": "text", "text": prompt})
	out, err := e.chat(content, 2048) // room to transcribe a full screen of text
	if err == nil && out == "" {
		err = errors.New("empty caption")
	}
	return out, err
}

// chat sends one user message to the chat model and returns the reply text.
func (e *AI) chat(content any, maxTokens int) (string, error) {
	if e.chatErr != nil {
		return "", e.chatErr
	}
	req := map[string]any{
		"model":                 e.chatModel,
		"messages":              []any{map[string]any{"role": "user", "content": content}},
		"max_completion_tokens": maxTokens,
	}
	for k, v := range e.chatExtra {
		req[k] = v
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	var res struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := e.post(e.chatURL+"/chat/completions", e.chatKey, body, &res); err != nil {
		return "", fmt.Errorf("chat at %s: %w", e.chatURL, err)
	}
	if len(res.Choices) == 0 {
		return "", fmt.Errorf("chat at %s: no reply", e.chatURL)
	}
	c := res.Choices[0]
	if c.FinishReason == "length" {
		fmt.Fprintf(os.Stderr, "jrnl: the reply hit its %d-token limit and is cut off\n", maxTokens)
	}
	return strings.TrimSpace(c.Message.Content), nil
}

// Rerank asks the chat model to order candidate entries by how well they answer the
// query, and returns up to k candidate positions, best first. On the 20,000
// entry test, this put the right entry first for 21 of 30 reworded questions,
// against 11 for keyword+vector fusion alone.
//
// k <= 0 asks for every relevant entry instead: for "the latest on X", search
// then sorts those by date. Asking for the 5 best of ~60 near-identical
// matches and sorting those missed the newest one on the 20,000-entry test.
func (e *AI) Rerank(query string, items []string, k int) ([]int, error) {
	// A dedicated reranker scores every candidate, but can't say that none
	// of them answers; "every relevant entry" (k <= 0) stays with chat.
	if e.rerankURL != "" && k > 0 {
		return e.rerankAPI(query, items, k)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Query: %s\n\nJournal entries:\n", query)
	for i, it := range items {
		fmt.Fprintf(&b, "[%d] %s\n", i, it)
	}
	if k > 0 {
		fmt.Fprintf(&b, "\nRank the entries by how well they answer the query. Reply with only a JSON array "+
			"of the %d best entry numbers, best first, for example [3, 0, 7].", k)
	} else {
		b.WriteString("\nWhich entries are about the query? Reply with only a JSON array of the numbers " +
			"of every such entry, for example [3, 0, 7], or [] if none are.")
	}
	out, err := e.chat(b.String(), 400)
	if err != nil {
		return nil, err
	}
	var order []int
	seen := map[int]bool{}
	for _, f := range strings.FieldsFunc(out, func(r rune) bool { return r < '0' || r > '9' }) {
		n, err := strconv.Atoi(f)
		if err == nil && n < len(items) && !seen[n] {
			seen[n] = true
			order = append(order, n)
		}
	}
	// "[]" means none of the candidates answers the query. On the 20,000
	// entry test it came back when the right entry was not among them.
	if len(order) == 0 && strings.TrimSpace(out) != "[]" {
		return nil, fmt.Errorf("reranker gave no usable order: %q", truncate(out, 120))
	}
	return order, nil
}

// rerankAPI calls a Cohere-style /rerank endpoint. Cohere, Jina and Vercel
// answer with "results", Voyage with "data"; both carry index and score.
func (e *AI) rerankAPI(query string, items []string, k int) ([]int, error) {
	body, _ := json.Marshal(map[string]any{"model": e.rerankModel, "query": query, "documents": items})
	type scored struct {
		Index int     `json:"index"`
		Score float64 `json:"relevance_score"`
	}
	var res struct {
		Results []scored `json:"results"`
		Data    []scored `json:"data"`
	}
	if err := e.post(e.rerankURL+"/rerank", e.rerankKey, body, &res); err != nil {
		return nil, fmt.Errorf("rerank at %s: %w", e.rerankURL, err)
	}
	got := append(res.Results, res.Data...)
	if len(got) == 0 {
		return nil, fmt.Errorf("rerank at %s: no scores", e.rerankURL)
	}
	sort.SliceStable(got, func(a, b int) bool { return got[a].Score > got[b].Score })
	var order []int
	seen := map[int]bool{}
	for _, g := range got {
		if g.Index >= 0 && g.Index < len(items) && !seen[g.Index] && len(order) < k {
			seen[g.Index] = true
			order = append(order, g.Index)
		}
	}
	return order, nil
}

// notSetUp marks a part of jrnl that this machine has not been set up for:
// no chat API, no media embedder, no ffmpeg, no parakeet. The work
// waits, quietly, for `jrnl reindex` after setup; `jrnl doctor` says how to turn
// each part on.
type notSetUp struct{ error }

var (
	errNoChat       = notSetUp{errors.New("no chat model is set up ([chat] in config.toml, or JRNL_CHAT_URL and JRNL_CHAT_MODEL)")}
	errNoMediaEmbed = notSetUp{errors.New("no media embedder is set up ([media_embed] in config.toml)")}
)

// embedDown means nothing answered at the embeddings address. With no
// JRNL_EMBED_URL, no ollama installed and no text embedding in the journal yet,
// that is a machine without Ollama, and jrnl stays quiet; otherwise it was set
// up and is now broken.
type embedDown struct {
	addr string
	set  bool // JRNL_EMBED_URL was given
	err  error
}

func (d embedDown) Error() string {
	if d.set {
		return "nothing answers at the embeddings URL " + d.addr
	}
	return "ollama is not reachable at " + strings.TrimSuffix(d.addr, "/v1") + " (start it with `ollama serve`)"
}
func (d embedDown) Unwrap() error { return d.err }

// permanent marks an input that will fail the same way on every retry: the
// API rejected it, or the file is unreadable or of an unsupported type.
type permanent struct{ error }

func isPermanent(err error) bool {
	var p permanent
	return errors.As(err, &p)
}

// EmbedEach embeds a batch in one request. If the batch fails, it retries each
// input alone, so one bad attachment cannot block the others. Result i is
// either vecs[i] or errs[i].
//
// Text and media are embedded separately with separate errors, so trouble
// with the media embedder never holds back text.
func (e *AI) EmbedEach(ins []Input) (vecs [][]float32, errs []error) {
	vecs, errs = make([][]float32, len(ins)), make([]error, len(ins))
	var text, media []int
	for i, in := range ins {
		if in.File == "" {
			text = append(text, i)
		} else {
			media = append(media, i)
		}
	}
	for _, group := range [][]int{text, media} {
		if len(group) == 0 {
			continue
		}
		sub := make([]Input, len(group))
		for j, i := range group {
			sub[j] = ins[i]
		}
		gv, ge := e.embedGroup(sub)
		for j, i := range group {
			vecs[i], errs[i] = gv[j], ge[j]
		}
	}
	return vecs, errs
}

func (e *AI) embedGroup(ins []Input) (vecs [][]float32, errs []error) {
	vecs, errs = make([][]float32, len(ins)), make([]error, len(ins))
	all, err := e.Embed(ins)
	if err == nil {
		return all, errs
	}
	// A transient failure (outage, rate limit after retries) hits every input
	// alike; retrying them one by one would only multiply the wait.
	if len(ins) == 1 || !isPermanent(err) {
		for i := range errs {
			errs[i] = err
		}
		return vecs, errs
	}
	for i, in := range ins {
		v, err := e.Embed([]Input{in})
		if err != nil {
			errs[i] = err
			continue
		}
		vecs[i] = v[0]
	}
	return vecs, errs
}

func normalize(v []float32) []float32 {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	if sum == 0 {
		return v
	}
	n := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= n
	}
	return v
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		if i >= len(b) {
			break
		}
		s += a[i] * b[i]
	}
	return s
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}
