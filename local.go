package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
)

// Text is embedded in this process by BAAI/bge-small-en-v1.5, run in pure Go
// (hugot's GoMLX backend): no server, no key, no network after the model's
// one download (128 MB, into the user cache).
//
// On the tycoon journal's 27 reworded questions and 18 unrelated ones
// (2026-11-22, vectors alone): right entry first for 23, in the top 5 for
// all 27, against 24 and 27 for EmbeddingGemma through Ollama and 18 and 25
// for potion-base-32M; unrelated questions scored lowest of the three.
// About 8 ms a query and 19 ms an entry on Apple M-series with
// GOEXPERIMENT=simd, as releases are built; about twice and six times that
// without.
const (
	localRepo  = "BAAI/bge-small-en-v1.5"
	localRev   = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a" // pinned: main can move
	localModel = "bge-small-en-v1.5"
	// The model card's instruction for search queries; documents go plain.
	localQueryPrefix = "Represent this sentence for searching relevant passages: "
	// The model reads 512 tokens and its tokenizer does not cut the input,
	// so longer text would run past its position table: embed cuts it at
	// this many, after the markers it adds. Keyword search still covers
	// the rest of a long entry. Characters are a poor proxy: of 154 real
	// entries one passed 512 tokens at 1,500 characters, and dense code
	// reaches 667.
	localMaxTokens = 510
	localSize      = "128 MB"
)

// localID names the vectors this embedder makes. Anything that changes them
// (model, revision, the text recipe) needs a new ID, so search never mixes
// them and `jrnl reindex` redoes the old ones.
var localID = localModel + "@local:" + localRev[:7]

// The files the model needs, where they are in its repository, and their
// SHA-256, checked after download (a proxy or portal can answer 200 with a
// page of its own).
var localFiles = [][3]string{
	{"model.onnx", "onnx/model.onnx", "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"},
	{"tokenizer.json", "tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"},
}

func localDir() string {
	if d := os.Getenv("JRNL_MODEL_DIR"); d != "" {
		return d
	}
	c, err := os.UserCacheDir()
	if err != nil {
		c = os.TempDir()
	}
	return filepath.Join(c, "jrnl", "models", localModel+"-"+localRev[:7])
}

// modelsURL is where models are downloaded from; tests point it elsewhere.
func modelsURL() string {
	if u := os.Getenv("JRNL_MODELS_URL"); u != "" {
		return u
	}
	return "https://huggingface.co"
}

// errNoModel: the model is not downloaded, and this command does not
// download it (add and search must not wait minutes for 128 MB). jrnl serve
// downloads it and embeds what waited.
var errNoModel = notSetUp{errors.New("the embedding model is not downloaded yet (`jrnl reindex` downloads it, " + localSize + ", once)")}

// localDownloaded reports whether every model file is in place.
func localDownloaded() bool {
	for _, f := range localFiles {
		if _, err := os.Stat(filepath.Join(localDir(), f[0])); err != nil {
			return false
		}
	}
	return true
}

type localEmbedder struct {
	mu  sync.Mutex // guards p and err
	run sync.Mutex // one model call at a time: a search and a backfill share it
	p   *pipelines.FeatureExtractionPipeline
	err error
}

var local localEmbedder

// load opens the model once per process, downloading it first if allowed.
// A failed load is not remembered when the files were missing, so a later
// call can download them.
func (l *localEmbedder) load(download bool) (*pipelines.FeatureExtractionPipeline, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.p != nil || l.err != nil {
		return l.p, l.err
	}
	if !localDownloaded() {
		if !download {
			return nil, errNoModel
		}
		if err := fetchModel(localDir()); err != nil {
			return nil, err
		}
	}
	s, err := hugot.NewGoSession(context.Background())
	if err == nil {
		l.p, err = hugot.NewPipeline(s, hugot.FeatureExtractionConfig{ModelPath: localDir(), Name: localModel,
			OnnxFilename: "model.onnx", Options: []hugot.FeatureExtractionOption{pipelines.WithNormalization()}})
	}
	if err != nil {
		// Not permanent: the entries wait for reindex instead of being
		// marked as text that can never be embedded.
		l.err = fmt.Errorf("load %s from %s: %w", localModel, localDir(), err)
	}
	return l.p, l.err
}

// embed returns a unit vector for each text, already formatted.
func (l *localEmbedder) embed(texts []string, download bool) ([][]float32, error) {
	p, err := l.load(download)
	if err != nil {
		return nil, err
	}
	texts = fitTokens(p, texts)
	// Groups of 4 keep a search behind a backfill waiting tens of
	// milliseconds, not hundreds.
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += 4 {
		l.run.Lock()
		r, err := p.RunPipeline(context.Background(), texts[i:min(i+4, len(texts))])
		l.run.Unlock()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", localModel, err)
		}
		out = append(out, r.Embeddings...)
	}
	return out, nil
}

// fetchModel downloads the model files into dir. A lock keeps two processes
// from downloading at once; the second waits and finds the files there.
// Each file is written under a temporary name and renamed when complete.
func fetchModel(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	lock, err := os.OpenFile(dir+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
	if localDownloaded() {
		return nil
	}
	fmt.Fprintf(os.Stderr, "jrnl: downloading the embedding model %s (%s, once)\n", localRepo, localSize)
	hc := &http.Client{Timeout: 15 * time.Minute}
	for _, f := range localFiles {
		dst := filepath.Join(dir, f[0])
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		url := fmt.Sprintf("%s/%s/resolve/%s/%s", modelsURL(), localRepo, localRev, f[1])
		r, err := hc.Get(url)
		if err != nil {
			return fmt.Errorf("download %s: %w", f[1], err)
		}
		if r.StatusCode != http.StatusOK {
			r.Body.Close()
			return fmt.Errorf("download %s: HTTP %d", url, r.StatusCode)
		}
		tmp, err := os.CreateTemp(dir, f[0]+".part-*")
		if err != nil {
			r.Body.Close()
			return err
		}
		h := sha256.New()
		_, err = io.Copy(io.MultiWriter(tmp, h), r.Body)
		r.Body.Close()
		if cerr := tmp.Close(); err == nil {
			err = cerr
		}
		if got := hex.EncodeToString(h.Sum(nil)); err == nil && got != f[2] {
			err = fmt.Errorf("got a file with SHA-256 %s, want %s", got, f[2])
		}
		if err == nil {
			err = os.Rename(tmp.Name(), dst)
		}
		if err != nil {
			os.Remove(tmp.Name())
			return fmt.Errorf("download %s: %w", f[1], err)
		}
	}
	return nil
}

// fitTokens cuts each text to the model's window. The tokenizer gives no
// token positions, so a text over the window is cut where its token count
// says, counted again, and shortened until it fits (about two rounds; only
// long texts pay).
func fitTokens(p *pipelines.FeatureExtractionPipeline, texts []string) []string {
	count := func(xs []string) []int {
		batch := backends.NewBatch(len(xs))
		defer batch.Destroy()
		backends.TokenizeInputs(batch, p.Model.Tokenizer, xs)
		n := make([]int, len(xs))
		for i, in := range batch.Input {
			n[i] = len(in.TokenIDs)
		}
		return n
	}
	limit := localMaxTokens + 2 // with the [CLS] and [SEP] markers
	var out []string
	for i, n := range count(texts) {
		t := texts[i]
		for n > limit {
			cut := len(t) * (limit - 8) / n
			for cut > 0 && !utf8.RuneStart(t[cut]) {
				cut--
			}
			t = t[:cut]
			n = count([]string{t})[0]
		}
		if t != texts[i] {
			if out == nil {
				out = slices.Clone(texts)
			}
			out[i] = t
		}
	}
	if out == nil {
		return texts
	}
	return out
}
