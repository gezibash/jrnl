# jrnl

A journal per project, for coding agents and the people working with them.
Agents write down decisions, bug root causes, findings and screenshots as
they work, then search them later: "what did we do yesterday", "have we seen
this error before", "what's still open", "summarise last week".

- Entries live in SQLite, one journal per git remote (`~/.journal/<project>/`).
  Many sessions can write at once.
- Search mixes keywords and meaning, then reranks. Dates are local calendar
  days: `--on yesterday`, `--since monday`, `--on "last week"`.
- Attachments (images, video, audio) are captioned or transcribed, so search
  finds them by what they show or say.
- Summaries cite the entries they come from; long periods cite day summaries.

## Install

```bash
mise use -g github:gezibash/jrnl
# or
brew install gezibash/tap/jrnl
# or, into ~/.local/bin (JRNL_INSTALL_DIR picks another folder)
curl -fsSL https://raw.githubusercontent.com/gezibash/jrnl/main/install.sh | sh
```

Or download a binary from the releases page, or `go install github.com/gezibash/jrnl@main`.

jrnl is made to run on every agent prompt, so keep a mise shim out of its
way: each shim runs mise before the tool, 100-300 ms a call. With mise, use
`mise activate` in your shell and turn shims off, so the real binary is on
PATH (`jrnl doctor` warns when it finds a shim):

```bash
mise settings set activate_shims false
```

For Claude Code, whose commands don't run your shell's prompt hooks, a
SessionStart hook can put the project's tools on PATH once per session:

```json
{ "type": "command",
  "command": "[ -n \"$CLAUDE_ENV_FILE\" ] && p=$(mise bin-paths 2>/dev/null | paste -sd: -) && [ -n \"$p\" ] && echo \"export PATH=\\\"$p:\\$PATH\\\"\" >> \"$CLAUDE_ENV_FILE\" || true" }
```

Homebrew's own `jrnl` formula is a different program, a Python note taker.
Install from the tap as shown above. Both install a `jrnl` command, so keep one.

## Setup

None. Install the binary and start writing: entries, filters, dates,
open/closed tracking, keyword search and search by meaning all work with
nothing else. The embedding model, BAAI/bge-small-en-v1.5, runs inside jrnl;
`jrnl reindex` (or the first `jrnl serve`) downloads it once, 128 MB.

Each optional part adds more. Turn them on in any order, whenever you like:

| Adds | Needs |
|---|---|
| Search by meaning with another model | Any OpenAI-compatible embeddings API in `[embed]`, such as [Ollama](https://ollama.com) with `embeddinggemma-2:270m`. Without it, the built-in bge-small-en-v1.5 |
| Reranking, captions, summaries | A chat model that takes images |
| Faster reranking | A reranker with a Cohere-style `/rerank` API (Cohere, Voyage, Jina, or any of them through Vercel AI Gateway) |
| `search --like` by look and sound | Media embeddings from Cloudflare Workers AI. Without them, `--like` compares descriptions, which needs the chat model |
| Video and audio attachments | `ffmpeg`; audio transcription uses `parakeet-mlx` through `uvx` (Apple silicon) |

Set them up in `~/.config/jrnl/config.toml` (or `JRNL_CONFIG`). Providers say
where a service is; each role picks a provider and a model:

```toml
[providers.vercel]
type = "openai"                       # any OpenAI-compatible API (the default)
url  = "https://ai-gateway.vercel.sh/v1"
key  = "env:AI_GATEWAY_API_KEY"       # the variable that holds the key

[providers.cloudflare]
type    = "cloudflare"                # Workers AI: chat and media embeddings
account = "env:CLOUDFLARE_ACCOUNT_ID" # the ID, or a variable
key     = "env:CLOUDFLARE_API_TOKEN"

[embed]
provider = "vercel"
model    = "alibaba/qwen3-embedding-4b"

[rerank]
provider = "vercel"
model    = "voyage/rerank-3-lite"

[chat]
provider = "vercel"
model    = "google/gemma-4-26b-a4b-it"
extra    = { reasoning_effort = "none" }   # JSON fields added to each chat request

[media_embed]
provider = "cloudflare"
model    = "@cf/google/embeddinggemma-2"
```

OpenAI, Vercel AI Gateway, OpenRouter, Ollama, LM Studio, vLLM and others
work as `openai` providers. The file holds no keys, only the names of the
variables that do, so it can go in a public dotfiles repository. jrnl refuses
settings it doesn't know, so a typo can't quietly turn a part off.

Variables win over the file: `JRNL_EMBED_*`, `JRNL_CHAT_*` and `JRNL_RERANK_*`
(`_URL`, `_KEY`, `_MODEL`), and `JRNL_CHAT_EXTRA`. A `_URL` replaces the whole
role with an OpenAI-compatible API, so the file's key is never sent to it; a
`_MODEL` or `_KEY` alone replaces just that.

Keys that are not in the environment are read from `~/.config/jrnl/.env` (or
`JRNL_ENV_FILE`), one `KEY=value` per line. That file can be a
[1Password Environment](https://www.1password.dev/environments/local-env-file)
mounted as a local `.env` file, so keys never sit on disk. jrnl reads it
only when a request needs a key, account or model that the environment
lacks, so commands that call no API (list, show, add, `search --fast`) never
wait on it. Put the `_URL` that defines a role in `config.toml` or the
environment, not in this file.

`jrnl doctor` shows which parts are on, which models they use, where each was
set, and how to turn on the rest. After turning one on, or switching
embedding model, `jrnl reindex` indexes everything saved before. Vectors from
different embedding models are never compared, and each model's are kept, so
switching back needs no second reindex; `jrnl reindex --prune` deletes those
of models no longer set up, which `jrnl doctor` lists.

On a real 159-entry journal (2026-11-22), with 27 reworded questions and 18
unrelated ones, `jrnl serve` with the built-in model shows the answer for 25
of the 27, and 3 of the 18 unrelated questions get a hit. Questions of two to
four words do worse: 15 of 22 answered, 3 of 10 unrelated ones get a hit. By
its vectors alone the model puts the right entry first for 22 of the 27; when
the journal had 113 entries it was 24, as for EmbeddingGemma through Ollama
(potion-base-32M: 18). `eval_test.go` measures it, from questions kept
outside the repository; [docs/benchmarks.md](docs/benchmarks.md) has the
figures for other embedding models and rerankers on 20,000 entries.

## Search from a coding-agent hook

`jrnl serve --background` keeps the embedding model and every journal's
index in memory (about 500 MB with the built-in model), and answers over a
Unix socket at `~/.journal/.run/jrnl.sock`. Other jrnl commands embed through
it, so they stay small; one that finds no server loads the model itself and
starts one.

```bash
curl --unix-socket ~/.journal/.run/jrnl.sock 'http://jrnl/search?project=<name>&q=<words>&limit=5'
```

`exclude=ID,ID` leaves entries out. A search takes about 9 ms on 20,000 entries (almost all of it embedding the question), so a hook
can run one on every prompt. It prints what `jrnl search` prints, and shows only clear matches, and nothing for an unrelated
question. The server runs one at a time, replaces a server of another version
when started, frees a journal after 10 idle minutes and exits after 30.
`jrnl search --fast` gives the same answer from the command line: every
search ranks with the same code.

## Use

```bash
jrnl add --kind bug --tag auth "Users logged out after token refresh

Root cause: the refresh handler raced with the session writer."
jrnl search "users logged out"
jrnl list --on yesterday
jrnl list --status open --on "last week"
jrnl summarize --on "last week"
jrnl rm 01M7WEN3        # delete an entry for good, with its attachments
jrnl help
```

Only `jrnl add` starts a journal; the other commands say so when a project
has none. `jrnl rm` rewrites the database, so the removed text is gone from
the file; backups made earlier keep their copies.

A newer decision can say which one it replaces, so the decisions on a topic read
as a thread whose newest entry is the current state:

```bash
jrnl add --kind decision --supersedes 01M7WEN3 "Cloudflare kept as a provider type"
jrnl link NEW OLD --supersedes   # afterwards; --rm removes a wrong link
```

`show` then marks the older entry as superseded and keeps
keep its text, since it often holds the reason. With `jrnl serve` running,
`jrnl add --kind decision` lists similar earlier decisions and the command to
link one; it never links by itself.
