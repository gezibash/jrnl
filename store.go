package main

import (
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	mrand "math/rand/v2"
	"mime"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

// Busy timeout comes before WAL so that a process waiting on another's lock
// waits instead of failing. _txlock=immediate makes every BEGIN take the write
// lock up front; a deferred transaction that upgrades later gets SQLITE_BUSY
// with no wait.
const dsnParams = "?_pragma=busy_timeout(15000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=foreign_keys(ON)&_txlock=immediate"

var schema = []string{
	`CREATE TABLE IF NOT EXISTS entries (
		id TEXT PRIMARY KEY,
		ts TEXT NOT NULL,
		project TEXT NOT NULL,
		session TEXT NOT NULL DEFAULT '',
		author TEXT NOT NULL,
		kind TEXT NOT NULL,
		branch TEXT NOT NULL DEFAULT '',
		commit_sha TEXT NOT NULL DEFAULT '',
		title TEXT NOT NULL,
		body TEXT NOT NULL,
		status TEXT NOT NULL DEFAULT '', -- '' (not tracked), open or closed
		closed_at TEXT NOT NULL DEFAULT '',
		closed_by TEXT NOT NULL DEFAULT '' -- the entry holding the closing note
	)`,
	`CREATE INDEX IF NOT EXISTS entries_ts ON entries(ts)`,
	`CREATE TABLE IF NOT EXISTS tags (
		entry_id TEXT NOT NULL REFERENCES entries(id),
		tag TEXT NOT NULL,
		PRIMARY KEY (entry_id, tag)
	)`,
	`CREATE INDEX IF NOT EXISTS tags_tag ON tags(tag)`,
	// type: relates (--link), supersedes (src replaces dst: a thread of
	// decisions) or closes (a close note and the entry it closed).
	`CREATE TABLE IF NOT EXISTS links (
		src TEXT NOT NULL REFERENCES entries(id),
		dst TEXT NOT NULL,
		type TEXT NOT NULL DEFAULT 'relates',
		PRIMARY KEY (src, dst)
	)`,
	`CREATE TABLE IF NOT EXISTS files (
		entry_id TEXT NOT NULL REFERENCES entries(id),
		path TEXT NOT NULL,
		PRIMARY KEY (entry_id, path)
	)`,
	`CREATE TABLE IF NOT EXISTS assets (
		hash TEXT PRIMARY KEY,
		ext TEXT NOT NULL,
		mime TEXT NOT NULL,
		bytes INTEGER NOT NULL,
		description TEXT, -- caption and/or transcript; NULL = not described yet
		describe_error TEXT NOT NULL DEFAULT '' -- set when it can never be described
	)`,
	`CREATE TABLE IF NOT EXISTS entry_assets (
		entry_id TEXT NOT NULL REFERENCES entries(id),
		hash TEXT NOT NULL REFERENCES assets(hash),
		name TEXT NOT NULL,
		caption TEXT NOT NULL DEFAULT '',
		PRIMARY KEY (entry_id, hash)
	)`,
	// One row per embedded piece. vec IS NULL means the embedding is pending.
	// modality: text (the entry), caption (an attachment's description), or
	// image / video / audio (the raw media, used only by search --like).
	`CREATE TABLE IF NOT EXISTS chunks (
		id INTEGER PRIMARY KEY,
		entry_id TEXT NOT NULL REFERENCES entries(id),
		hash TEXT NOT NULL DEFAULT '',
		modality TEXT NOT NULL,
		vec BLOB,
		error TEXT NOT NULL DEFAULT '', -- set when the input can never embed
		embedder TEXT NOT NULL DEFAULT '' -- which model and path made vec
	)`,
	`CREATE UNIQUE INDEX IF NOT EXISTS chunks_piece ON chunks(entry_id, hash, modality)`,
	`CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
		entry_id UNINDEXED, title, body, tags, captions,
		tokenize = 'porter unicode61'
	)`,
}

type Store struct {
	db  *sql.DB
	dir string
}

func journalHome() string {
	if h := os.Getenv("JRNL_HOME"); h != "" {
		return h
	}
	h, _ := os.UserHomeDir()
	return filepath.Join(h, ".journal")
}

func Open(project string) (*Store, error) {
	dir := filepath.Join(journalHome(), project)
	if err := os.MkdirAll(filepath.Join(dir, "assets"), 0o755); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", "file:"+filepath.Join(dir, "journal.db")+dsnParams)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	s := &Store{db: db, dir: dir}
	// Reads skip the write lock once the schema is in place. While another
	// process is creating the database, this read can fail with SQLITE_BUSY
	// without waiting (1 in 500 in the parallel test); the locked path below
	// waits, so any error falls through to it.
	var v int
	if err := db.QueryRow(`PRAGMA user_version`).Scan(&v); err == nil && v == schemaVersion {
		return s, nil
	}
	// Creation and upgrades run under the write lock, so two first runs don't
	// race; the version is read again once the lock is held.
	// Switching a new database to WAL can fail with SQLITE_BUSY at once,
	// without waiting, while another process creates it (10 of 500 parallel
	// first runs); those retry.
	tx, err := db.Begin()
	for i := 0; err != nil && strings.Contains(err.Error(), "SQLITE_BUSY") && i < 100; i++ {
		time.Sleep(time.Duration(10+mrand.IntN(20)) * time.Millisecond)
		tx, err = db.Begin()
	}
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", dir, err)
	}
	defer tx.Rollback()
	if err := tx.QueryRow(`PRAGMA user_version`).Scan(&v); err != nil {
		return nil, err
	}
	if v > schemaVersion {
		return nil, fmt.Errorf("%s was written by a newer jrnl (schema %d, this jrnl knows %d)", dir, v, schemaVersion)
	}
	if v > 0 {
		// Bring an existing journal up to date before creating anything new.
		for _, c := range []struct{ table, col, def string }{
			{"chunks", "embedder", "TEXT NOT NULL DEFAULT ''"},
			{"entries", "status", "TEXT NOT NULL DEFAULT ''"},
			{"entries", "closed_at", "TEXT NOT NULL DEFAULT ''"},
			{"entries", "closed_by", "TEXT NOT NULL DEFAULT ''"},
			{"links", "type", "TEXT NOT NULL DEFAULT 'relates'"},
		} {
			if err := addColumn(tx, c.table, c.col, c.def); err != nil {
				return nil, fmt.Errorf("upgrade: %w", err)
			}
		}
	}
	for _, q := range schema {
		if _, err := tx.Exec(q); err != nil {
			return nil, fmt.Errorf("schema: %w", err)
		}
	}
	if v > 0 && v < 3 {
		// Before 3, Ollama's native API at OLLAMA_HOST made every text vector.
		// Its /v1 API gives the same vectors (cosine 1.0 on a 6,700-token
		// text), so they keep working under the new name.
		if _, err := tx.Exec(`UPDATE chunks SET embedder = ? WHERE embedder = ?`,
			embedderID(ollamaURL(), ollamaModel), "ollama/"+ollamaModel+"/768"); err != nil {
			return nil, fmt.Errorf("upgrade: %w", err)
		}
	}
	if v > 0 && v < 4 {
		// Before 4, media vectors were always Workers AI's EmbeddingGemma 2,
		// named by a constant. Now they are named like text embedders.
		if _, err := tx.Exec(`UPDATE chunks SET embedder = 'cloudflare/@cf/google/embeddinggemma-2'
			WHERE embedder = 'workers-ai/embeddinggemma-2/768'`); err != nil {
			return nil, fmt.Errorf("upgrade: %w", err)
		}
	}
	if v > 0 && v < 5 {
		// Before 5, links had no type. A close note's link to what it closed
		// is known from closed_by; every other link stays relates.
		if _, err := tx.Exec(`UPDATE links SET type = 'closes'
			WHERE EXISTS (SELECT 1 FROM entries e WHERE e.id = links.dst AND e.closed_by = links.src)`); err != nil {
			return nil, fmt.Errorf("upgrade: %w", err)
		}
	}
	if _, err := tx.Exec(fmt.Sprintf(`PRAGMA user_version = %d`, schemaVersion)); err != nil {
		return nil, err
	}
	return s, tx.Commit()
}

// 1: first release. 2: open/closed status on entries. 3: text embedders
// named by model and host. 4: media embedders named the same way. 5: typed
// links (relates, supersedes, closes).
const schemaVersion = 5

func addColumn(tx *sql.Tx, table, col, def string) error {
	var n int
	if err := tx.QueryRow(`SELECT count(*) FROM pragma_table_info(?) WHERE name = ?`, table, col).Scan(&n); err != nil || n > 0 {
		return err
	}
	_, err := tx.Exec(`ALTER TABLE ` + table + ` ADD COLUMN ` + col + ` ` + def)
	return err
}

// Delete removes an entry and everything that hangs off it (used to replace
// a stale summary).
func (s *Store) Delete(id string) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, q := range []string{
		`DELETE FROM chunks WHERE entry_id = ?`, `DELETE FROM entries_fts WHERE entry_id = ?`,
		`DELETE FROM tags WHERE entry_id = ?`, `DELETE FROM links WHERE src = ? OR dst = ?`,
		`DELETE FROM files WHERE entry_id = ?`, `DELETE FROM entry_assets WHERE entry_id = ?`,
		`DELETE FROM entries WHERE id = ?`,
	} {
		args := []any{id}
		if strings.Count(q, "?") == 2 {
			args = append(args, id)
		}
		if _, err := tx.Exec(q, args...); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// SetStatus opens, closes or reopens entries. closedBy is the entry holding
// the closing note, if any.
func (s *Store) SetStatus(ids []string, status, closedBy string) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	closedAt := ""
	if status == "closed" {
		closedAt = utc(now())
	}
	for _, id := range ids {
		if _, err := tx.Exec(`UPDATE entries SET status = ?, closed_at = ?, closed_by = ? WHERE id = ?`,
			status, closedAt, closedBy, id); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// Common media types, in case the system MIME table lacks them.
var mediaTypes = map[string]string{
	".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
	".gif": "image/gif", ".heic": "image/heic", ".avif": "image/avif",
	".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".m4v": "video/mp4",
	".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".flac": "audio/flac",
}

func (s *Store) Close() error { return s.db.Close() }

// Link sets the link from src to dst, replacing its type if it has one.
func (s *Store) Link(src, dst, typ string) error {
	if src == dst {
		return fmt.Errorf("an entry cannot link to itself")
	}
	_, err := s.db.Exec(`INSERT INTO links (src, dst, type) VALUES (?,?,?)
		ON CONFLICT (src, dst) DO UPDATE SET type = excluded.type`, src, dst, typ)
	return err
}

// Unlink removes the link between a and b, in either direction. It says
// whether there was one.
func (s *Store) Unlink(a, b string) (bool, error) {
	r, err := s.db.Exec(`DELETE FROM links WHERE (src = ? AND dst = ?) OR (src = ? AND dst = ?)`, a, b, b, a)
	if err != nil {
		return false, err
	}
	n, err := r.RowsAffected()
	return n > 0, err
}

type Entry struct {
	ID, TS, Project, Session, Author, Kind, Branch, Commit, Title, Body string
	Status, ClosedAt, ClosedBy                                          string
	Tags, Links, LinkedFrom, Files                                      []string
	// Supersedes: the entries this one replaces; SupersededBy: those that
	// replace it. Links and LinkedFrom hold the other types.
	Supersedes, SupersededBy []string
	Closes                   []string // written only: a close note's entries
	Assets                   []Asset
}

type Asset struct {
	Hash, Ext, Mime, Name, Caption, Path string
	Description                          string // generated caption/transcript
	Bytes                                int64
}

func (a Asset) Modality() string {
	switch {
	case strings.HasPrefix(a.Mime, "image/"):
		return "image"
	case strings.HasPrefix(a.Mime, "video/"):
		return "video"
	case strings.HasPrefix(a.Mime, "audio/"):
		return "audio"
	}
	return ""
}

func (s *Store) assetPath(hash, ext string) string {
	return filepath.Join(s.dir, "assets", hash+ext)
}

// Ingest copies a file into the asset store, named by its SHA-256. The copy
// goes to a temp file first and is renamed into place, so parallel ingests of
// the same file never see a half-written asset.
func (s *Store) Ingest(src, caption string) (Asset, error) {
	in, err := os.Open(src)
	if err != nil {
		return Asset{}, err
	}
	defer in.Close()
	tmp, err := os.CreateTemp(filepath.Join(s.dir, "assets"), ".ingest-*")
	if err != nil {
		return Asset{}, err
	}
	defer os.Remove(tmp.Name())
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, h), in)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return Asset{}, err
	}
	ext := strings.ToLower(filepath.Ext(src))
	mt := mediaTypes[ext]
	if mt == "" {
		mt = mime.TypeByExtension(ext)
	}
	if i := strings.IndexByte(mt, ';'); i >= 0 {
		mt = mt[:i]
	}
	if mt == "" {
		mt = "application/octet-stream"
	}
	a := Asset{Hash: hex.EncodeToString(h.Sum(nil)), Ext: ext, Mime: mt, Name: filepath.Base(src), Caption: caption, Bytes: n}
	a.Path = s.assetPath(a.Hash, a.Ext)
	if err := os.Rename(tmp.Name(), a.Path); err != nil {
		return Asset{}, err
	}
	return a, nil
}

type Chunk struct {
	Hash, Modality string
	Vec            []float32 // nil = pending
	Err            string    // set = can never embed; reindex skips it
}

func (s *Store) Insert(e Entry, chunks []Chunk) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`INSERT INTO entries (id, ts, project, session, author, kind, branch, commit_sha, title, body, status)
		VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
		e.ID, e.TS, e.Project, e.Session, e.Author, e.Kind, e.Branch, e.Commit, e.Title, e.Body, e.Status); err != nil {
		return err
	}
	for _, t := range e.Tags {
		if _, err := tx.Exec(`INSERT OR IGNORE INTO tags VALUES (?,?)`, e.ID, t); err != nil {
			return err
		}
	}
	for typ, ids := range map[string][]string{"relates": e.Links, "supersedes": e.Supersedes, "closes": e.Closes} {
		for _, l := range ids {
			if _, err := tx.Exec(`INSERT OR IGNORE INTO links (src, dst, type) VALUES (?,?,?)`, e.ID, l, typ); err != nil {
				return err
			}
		}
	}
	for _, f := range e.Files {
		if _, err := tx.Exec(`INSERT OR IGNORE INTO files VALUES (?,?)`, e.ID, f); err != nil {
			return err
		}
	}
	for _, a := range e.Assets {
		if _, err := tx.Exec(`INSERT OR IGNORE INTO assets (hash, ext, mime, bytes) VALUES (?,?,?,?)`, a.Hash, a.Ext, a.Mime, a.Bytes); err != nil {
			return err
		}
		if _, err := tx.Exec(`INSERT OR IGNORE INTO entry_assets VALUES (?,?,?,?)`, e.ID, a.Hash, a.Name, a.Caption); err != nil {
			return err
		}
	}
	for _, c := range chunks {
		if _, err := tx.Exec(`INSERT OR IGNORE INTO chunks (entry_id, hash, modality) VALUES (?,?,?)`,
			e.ID, c.Hash, c.Modality); err != nil {
			return err
		}
	}
	if err := ensureCaptionChunks(tx, e.ID); err != nil {
		return err
	}
	return tx.Commit()
}

type execer interface {
	Exec(string, ...any) (sql.Result, error)
}

// ensureCaptionChunks gives every described attachment of the entry (or of
// all entries, for "") a caption chunk to embed, and rebuilds the entry's
// keyword index so descriptions are searchable by keyword too. Idempotent, so
// parallel sessions describing the same file can both run it.
func ensureCaptionChunks(tx execer, entryID string) error {
	scope, args := "", []any{}
	if entryID != "" {
		scope, args = " AND ea.entry_id = ?", []any{entryID}
	}
	if _, err := tx.Exec(`INSERT OR IGNORE INTO chunks (entry_id, hash, modality)
		SELECT ea.entry_id, ea.hash, 'caption' FROM entry_assets ea JOIN assets a ON a.hash = ea.hash
		WHERE a.description IS NOT NULL`+scope, args...); err != nil {
		return err
	}
	ftsScope := ""
	if entryID != "" {
		ftsScope = " WHERE e.id = ?"
	}
	if _, err := tx.Exec(`DELETE FROM entries_fts`+strings.Replace(ftsScope, "e.id", "entry_id", 1), args...); err != nil {
		return err
	}
	_, err := tx.Exec(`INSERT INTO entries_fts (entry_id, title, body, tags, captions)
		SELECT e.id, e.title, e.body,
			COALESCE((SELECT group_concat(tag, ' ') FROM tags WHERE entry_id = e.id), ''),
			COALESCE((SELECT group_concat(ea.name || ' ' || ea.caption || ' ' || COALESCE(a.description, ''), ' ')
				FROM entry_assets ea JOIN assets a ON a.hash = ea.hash WHERE ea.entry_id = e.id), '')
		FROM entries e`+ftsScope, args...)
	return err
}

// ToDescribe lists media attachments of the entry (or all entries, for "")
// that have no description yet and have not failed for good.
func (s *Store) ToDescribe(entryID string) ([]Asset, error) {
	q := `SELECT DISTINCT a.hash, a.ext, a.mime, ea.name FROM assets a JOIN entry_assets ea ON ea.hash = a.hash
		WHERE a.description IS NULL AND a.describe_error = ''
		AND (a.mime LIKE 'image/%' OR a.mime LIKE 'video/%' OR a.mime LIKE 'audio/%')`
	args := []any{}
	if entryID != "" {
		q, args = q+` AND ea.entry_id = ?`, append(args, entryID)
	}
	rows, err := s.db.Query(q+` GROUP BY a.hash`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Asset
	for rows.Next() {
		var a Asset
		if err := rows.Scan(&a.Hash, &a.Ext, &a.Mime, &a.Name); err != nil {
			return nil, err
		}
		a.Path = s.assetPath(a.Hash, a.Ext)
		out = append(out, a)
	}
	return out, rows.Err()
}

// SetDescription stores a description and queues caption chunks for every
// entry that uses the file.
func (s *Store) SetDescription(hash, desc string) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE assets SET description = ? WHERE hash = ?`, desc, hash); err != nil {
		return err
	}
	rows, err := tx.Query(`SELECT entry_id FROM entry_assets WHERE hash = ?`, hash)
	if err != nil {
		return err
	}
	ids := scanIDs(rows)
	for _, id := range ids {
		if err := ensureCaptionChunks(tx, id); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (s *Store) SetDescribeError(hash, msg string) error {
	_, err := s.db.Exec(`UPDATE assets SET describe_error = ? WHERE hash = ?`, msg, hash)
	return err
}

// Resolve accepts a full ID or a unique prefix.
func (s *Store) Resolve(prefix string) (string, error) {
	rows, err := s.db.Query(`SELECT id FROM entries WHERE id LIKE ? || '%' LIMIT 2`, strings.ToUpper(prefix))
	if err != nil {
		return "", err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		rows.Scan(&id)
		ids = append(ids, id)
	}
	switch len(ids) {
	case 0:
		return "", fmt.Errorf("no entry %q", prefix)
	case 1:
		return ids[0], nil
	}
	return "", fmt.Errorf("prefix %q matches more than one entry", prefix)
}

func (s *Store) Get(id string) (Entry, error) {
	var e Entry
	err := s.db.QueryRow(`SELECT id, ts, project, session, author, kind, branch, commit_sha, title, body, status, closed_at, closed_by
		FROM entries WHERE id = ?`, id).
		Scan(&e.ID, &e.TS, &e.Project, &e.Session, &e.Author, &e.Kind, &e.Branch, &e.Commit, &e.Title, &e.Body, &e.Status, &e.ClosedAt, &e.ClosedBy)
	if err != nil {
		return e, err
	}
	e.Tags = s.strings(`SELECT tag FROM tags WHERE entry_id = ? ORDER BY tag`, id)
	e.Links = s.strings(`SELECT dst FROM links WHERE src = ? AND type != 'supersedes' ORDER BY dst`, id)
	e.LinkedFrom = s.strings(`SELECT src FROM links WHERE dst = ? AND type != 'supersedes' ORDER BY src`, id)
	e.Supersedes = s.strings(`SELECT dst FROM links WHERE src = ? AND type = 'supersedes' ORDER BY dst`, id)
	e.SupersededBy = s.strings(`SELECT src FROM links WHERE dst = ? AND type = 'supersedes' ORDER BY src`, id)
	e.Files = s.strings(`SELECT path FROM files WHERE entry_id = ? ORDER BY path`, id)
	rows, err := s.db.Query(`SELECT a.hash, a.ext, a.mime, a.bytes, ea.name, ea.caption, COALESCE(a.description, '')
		FROM entry_assets ea JOIN assets a ON a.hash = ea.hash WHERE ea.entry_id = ?`, id)
	if err != nil {
		return e, err
	}
	defer rows.Close()
	for rows.Next() {
		var a Asset
		if err := rows.Scan(&a.Hash, &a.Ext, &a.Mime, &a.Bytes, &a.Name, &a.Caption, &a.Description); err != nil {
			return e, err
		}
		a.Path = s.assetPath(a.Hash, a.Ext)
		e.Assets = append(e.Assets, a)
	}
	return e, rows.Err()
}

func (s *Store) strings(q string, args ...any) []string {
	rows, err := s.db.Query(q, args...)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var v string
		rows.Scan(&v)
		out = append(out, v)
	}
	return out
}

type Filter struct {
	Kind, Session, Author, File, Status string
	Since, Until, On                    string // see parseWhen
	Tags                                []string
}

func utc(t time.Time) string { return t.UTC().Format(time.RFC3339) }

func (f Filter) where() (string, []any, error) {
	var cl []string
	var args []any
	add := func(c string, a ...any) { cl, args = append(cl, c), append(args, a...) }
	if f.Kind != "" {
		add("e.kind = ?", f.Kind)
	}
	if f.Session != "" {
		add("e.session = ?", f.Session)
	}
	if f.Author != "" {
		add("e.author = ?", f.Author)
	}
	switch f.Status {
	case "":
	case "open", "closed":
		add("e.status = ?", f.Status)
	default:
		return "", nil, fmt.Errorf("--status is open or closed, not %q", f.Status)
	}
	if f.File != "" {
		// A bare file name matches it in any directory.
		add("EXISTS (SELECT 1 FROM files fl WHERE fl.entry_id = e.id AND (fl.path = ? OR fl.path LIKE '%/' || ?))", f.File, f.File)
	}
	ref := now()
	if f.Since != "" {
		sp, err := parseWhen(f.Since, ref)
		if err != nil {
			return "", nil, fmt.Errorf("--since: %w", err)
		}
		add("e.ts >= ?", utc(sp.start))
	}
	if f.Until != "" {
		sp, err := parseWhen(f.Until, ref)
		if err != nil {
			return "", nil, fmt.Errorf("--until: %w", err)
		}
		if sp.end.Equal(sp.start) {
			add("e.ts <= ?", utc(sp.end)) // a point: up to and including it
		} else {
			add("e.ts < ?", utc(sp.end)) // a day, week or month: through its end
		}
	}
	if f.On != "" {
		sp, err := parseWhen(f.On, ref)
		if err != nil {
			return "", nil, fmt.Errorf("--on: %w", err)
		}
		if sp.end.Equal(sp.start) {
			return "", nil, fmt.Errorf("--on needs a day, week or month, not a point in time")
		}
		add("e.ts >= ? AND e.ts < ?", utc(sp.start), utc(sp.end))
	}
	for _, t := range f.Tags {
		add("EXISTS (SELECT 1 FROM tags t WHERE t.entry_id = e.id AND t.tag = ?)", t)
	}
	if len(cl) == 0 {
		return "1=1", nil, nil
	}
	return strings.Join(cl, " AND "), args, nil
}

// LastSession is the most recently active session other than skip (the
// caller's own), so "the last session" means the one before this one.
func (s *Store) LastSession(skip string) (string, error) {
	var id string
	err := s.db.QueryRow(`SELECT session FROM entries WHERE session != '' AND session != ?
		GROUP BY session ORDER BY max(ts) DESC LIMIT 1`, skip).Scan(&id)
	if err == sql.ErrNoRows {
		return "", fmt.Errorf("no earlier session in this journal")
	}
	return id, err
}

type SessionInfo struct {
	ID, First, Last, LastTitle string
	Entries                    int
}

func (s *Store) Sessions(f Filter, limit int) ([]SessionInfo, error) {
	w, args, err := f.where()
	if err != nil {
		return nil, err
	}
	rows, err := s.db.Query(`SELECT e.session, min(e.ts), max(e.ts), count(*),
			(SELECT title FROM entries x WHERE x.session = e.session ORDER BY x.ts DESC LIMIT 1)
		FROM entries e WHERE e.session != '' AND `+w+`
		GROUP BY e.session ORDER BY max(e.ts) DESC LIMIT ?`, append(args, limit)...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []SessionInfo
	for rows.Next() {
		var si SessionInfo
		if err := rows.Scan(&si.ID, &si.First, &si.Last, &si.Entries, &si.LastTitle); err != nil {
			return nil, err
		}
		out = append(out, si)
	}
	return out, rows.Err()
}

type Count struct {
	Key string
	N   int
}

// Stats counts matching entries by kind, tag and author.
func (s *Store) Stats(f Filter) (total int, first, last string, by map[string][]Count, err error) {
	w, args, err := f.where()
	if err != nil {
		return
	}
	var fp, lp sql.NullString
	if err = s.db.QueryRow(`SELECT count(*), min(e.ts), max(e.ts) FROM entries e WHERE `+w, args...).Scan(&total, &fp, &lp); err != nil {
		return
	}
	first, last = fp.String, lp.String
	by = map[string][]Count{}
	for name, q := range map[string]string{
		"kind":   `SELECT e.kind, count(*) FROM entries e WHERE ` + w + ` GROUP BY 1 ORDER BY 2 DESC`,
		"tag":    `SELECT t.tag, count(*) FROM entries e JOIN tags t ON t.entry_id = e.id WHERE ` + w + ` GROUP BY 1 ORDER BY 2 DESC`,
		"author": `SELECT e.author, count(*) FROM entries e WHERE ` + w + ` GROUP BY 1 ORDER BY 2 DESC`,
	} {
		rows, qerr := s.db.Query(q, args...)
		if qerr != nil {
			return 0, "", "", nil, qerr
		}
		for rows.Next() {
			var c Count
			rows.Scan(&c.Key, &c.N)
			by[name] = append(by[name], c)
		}
		rows.Close()
	}
	return
}

// List returns matching entries newest first, or oldest first.
func (s *Store) List(f Filter, limit int, oldest bool) ([]Entry, error) {
	w, args, err := f.where()
	if err != nil {
		return nil, err
	}
	order := "DESC"
	if oldest {
		order = "ASC"
	}
	rows, err := s.db.Query(`SELECT e.id FROM entries e WHERE `+w+` ORDER BY e.ts `+order+`, e.id `+order+` LIMIT ?`, append(args, limit)...)
	if err != nil {
		return nil, err
	}
	ids := scanIDs(rows)
	out := make([]Entry, 0, len(ids))
	for _, id := range ids {
		e, err := s.Get(id)
		if errors.Is(err, sql.ErrNoRows) {
			// Deleted since the query, by another session or a summary
			// being replaced; seen 1 run in 6 in the day-summary test.
			continue
		}
		if err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, nil
}

func scanIDs(rows *sql.Rows) []string {
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		rows.Scan(&id)
		ids = append(ids, id)
	}
	return ids
}

type pending struct {
	chunkID  int64
	modality string
	label    string // for messages: "entry text", a file name, or "<file> description"
	in       Input
}

// staleChunks matches chunks that need a vector from the current embedders:
// none yet, or one from another model. Media chunks count only when a media
// embedder is set up (mediaEmb != "").
const staleChunks = `((c.modality IN ('text', 'caption') AND (c.vec IS NULL OR c.embedder != ?))
	OR (? != '' AND c.modality NOT IN ('text', 'caption') AND (c.vec IS NULL OR c.embedder != ?))) AND c.error = ''`

// Pending returns chunks of the entry (or all entries, for "") that need a
// vector (staleChunks): after a switch of embedding model, reindex brings
// every entry back into search by meaning.
func (s *Store) Pending(entryID, textEmb, mediaEmb string) ([]pending, error) {
	q := `SELECT c.id, c.modality, c.hash, e.title, e.body, COALESCE(a.ext, ''), COALESCE(a.mime, ''),
			COALESCE(ea.name, ''), COALESCE(ea.caption, ''), COALESCE(a.description, '')
		FROM chunks c JOIN entries e ON e.id = c.entry_id
		LEFT JOIN assets a ON a.hash = c.hash AND c.hash != ''
		LEFT JOIN entry_assets ea ON ea.entry_id = c.entry_id AND ea.hash = c.hash
		WHERE ` + staleChunks
	args := []any{textEmb, mediaEmb, mediaEmb}
	if entryID != "" {
		q, args = q+` AND c.entry_id = ?`, append(args, entryID)
	}
	rows, err := s.db.Query(q+` ORDER BY c.id`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []pending
	for rows.Next() {
		var p pending
		var hash, title, body, ext, mt, name, caption, desc string
		if err := rows.Scan(&p.chunkID, &p.modality, &hash, &title, &body, &ext, &mt, &name, &caption, &desc); err != nil {
			return nil, err
		}
		switch p.modality {
		case "text":
			p.in, p.label = Input{Title: title, Text: body}, "entry text"
		case "caption":
			p.in, p.label = captionInput(name, caption, desc), name+" description"
		default:
			p.in, p.label = Input{File: s.assetPath(hash, ext), Mime: mt}, name
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// HasTextVectors reports whether any text here has ever been embedded.
func (s *Store) HasTextVectors() bool {
	var n int
	s.db.QueryRow(`SELECT COUNT(*) FROM (SELECT 1 FROM chunks
		WHERE vec IS NOT NULL AND modality IN ('text', 'caption') LIMIT 1)`).Scan(&n)
	return n > 0
}

// Waiting counts attachments not yet described and chunks Pending would
// return, leaving out the ones that can never be done.
func (s *Store) Waiting(textEmb, mediaEmb string) (int, error) {
	var n int
	err := s.db.QueryRow(`SELECT
		(SELECT COUNT(*) FROM chunks c WHERE `+staleChunks+`) +
		(SELECT COUNT(*) FROM assets WHERE description IS NULL AND describe_error = '')`, textEmb, mediaEmb, mediaEmb).Scan(&n)
	return n, err
}

// captionInput is how an attachment's description is embedded, both when it
// is stored and when search --like looks for one like it.
func captionInput(name, caption, desc string) Input {
	return Input{Title: name, Text: strings.TrimSpace(caption + "\n" + desc)}
}

func (s *Store) SetVec(chunkID int64, v []float32, embedder string) error {
	_, err := s.db.Exec(`UPDATE chunks SET vec = ?, embedder = ? WHERE id = ?`, encodeVec(v), embedder, chunkID)
	return err
}

func (s *Store) SetError(chunkID int64, msg string) error {
	_, err := s.db.Exec(`UPDATE chunks SET error = ? WHERE id = ?`, msg, chunkID)
	return err
}

func (s *Store) Backup() (string, error) {
	dir := filepath.Join(s.dir, "backups")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	dst := filepath.Join(dir, "journal-"+time.Now().UTC().Format("20060102T150405Z")+".db")
	_, err := s.db.Exec(`VACUUM INTO ?`, dst)
	return dst, err
}

func encodeVec(v []float32) any {
	if v == nil {
		return nil
	}
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

func decodeVec(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

// newID returns a ULID: 48-bit millisecond time + 80 random bits, Crockford
// base32. IDs sort by creation time.
func newID() string {
	const enc = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
	var b [16]byte
	ms := uint64(time.Now().UnixMilli())
	for i := 5; i >= 0; i-- {
		b[i] = byte(ms)
		ms >>= 8
	}
	rand.Read(b[6:])
	// 128 bits -> 26 base32 chars, taking 5 bits at a time from the top.
	out := make([]byte, 26)
	hi := binary.BigEndian.Uint64(b[:8])
	lo := binary.BigEndian.Uint64(b[8:])
	for i := 25; i >= 0; i-- {
		out[i] = enc[lo&31]
		lo = lo>>5 | hi<<59
		hi >>= 5
	}
	return string(out)
}
