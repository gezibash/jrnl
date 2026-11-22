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

Homebrew's own `jrnl` formula is a different program, a Python note taker.
Install from the tap as shown above. Both install a `jrnl` command, so keep one.

## Setup

None. Install the binary and start writing: entries, filters, dates,
open/closed tracking and keyword search all work with nothing else.

Each optional part adds more. Turn them on in any order, whenever you like:

| Adds | Needs |
|---|---|
| Search by meaning | An embedding model. Default: [Ollama](https://ollama.com) on this machine with `ollama pull embeddinggemma-2:270m` |
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
mounted as a local `.env` file, so keys never sit on disk.

`jrnl doctor` shows which parts are on, which models they use, where each was
set, and how to turn on the rest. After turning one on, or switching
embedding model, `jrnl reindex` indexes everything saved before. Vectors from
different embedding models are never compared.

Measured on the same 20,000 entries and 30 reworded questions:

| Embeddings + reranker | Right entry first | Median / slowest 5% | Notes |
|---|---|---|---|
| EmbeddingGemma 2 (local) + Gemma 4 (Workers AI) | 22 of 30 | 0.8 s / 4.2 s | |
| EmbeddingGemma 2 (local) + Voyage rerank-3-lite | 22 of 30 | 0.56 s / 0.73 s | fastest |
| Qwen3-Embedding-4B (local) + Gemma 4 | 24 of 30 | 0.93 s / 1.8 s | 8.8 GB of memory in Ollama |
| Qwen3-Embedding-4B (Vercel) + Voyage rerank-3-lite | 25 of 30 | 1.3 s / 2.2 s | best answers |

Cohere Rerank 5 Pro and Fast and Voyage rerank-3 scored 20 to 22, like Gemma:
for most of the misses, the right entry was never among the candidates, so
better embeddings help more than a better reranker.

The numbers below were measured with EmbeddingGemma 2 and Gemma 4. Other
models will score differently; the reranking and summary prompts were tuned
on Gemma 4. Keep thinking short or off: a model that reasons at length can
use up its reply budget and return nothing.

Measured on 20,000 entries spread over 3 months, with 30 test questions:

| | No setup | With Ollama and Workers AI |
|---|---|---|
| Question with the entry's own words: right entry first | 30 of 30 | 30 of 30 |
| Question in other words: right entry first | 8 of 30 | 21 of 30 |
| Search time, median / slowest 5% | 0.06 s / 0.07 s | 1.1 s / 4.2 s |
| Adds per second, 16 sessions at once | 156 | 58 |

The reworded questions avoid the entry's words on purpose, so they are the
hard case. Search by meaning is what closes that gap.

## Search from a coding-agent hook

`jrnl serve --background` keeps a small embedding model (potion-base-32M,
downloaded once, 131 MB) and every journal's index in memory, and answers
over a Unix socket at `~/.journal/.run/jrnl.sock`:

```bash
curl --unix-socket ~/.journal/.run/jrnl.sock 'http://jrnl/search?project=<name>&q=<words>&limit=5'
```

Add `format=context` for a block written for a coding agent to read: each
entry's whole title and its text (the passage that matches, when long), how it
matched, when and at which commit it was written, and later entries that link
to it, such as a newer decision; `exclude=ID,ID` leaves out entries the agent
already has. A search takes about a millisecond on 20,000 entries, so a hook
can run one on every prompt. It shows only clear matches, and nothing for an unrelated
question. The server runs one at a time, replaces a server of another version
when started, frees a journal after 10 idle minutes and exits after 30.
`jrnl search --fast` gives the same kind of answer from the command line.

## Use

```bash
jrnl add --kind bug --tag auth "Users logged out after token refresh

Root cause: the refresh handler raced with the session writer."
jrnl search "users logged out"
jrnl list --on yesterday
jrnl list --status open --on "last week"
jrnl summarize --on "last week"
jrnl help
```

A newer decision can say which one it replaces, so the decisions on a topic read
as a thread whose newest entry is the current state:

```bash
jrnl add --kind decision --supersedes 01M7WEN3 "Cloudflare kept as a provider type"
jrnl link NEW OLD --supersedes   # afterwards; --rm removes a wrong link
```

`show` and the agent recall block then mark the older entry as superseded and
keep its text, since it often holds the reason. With `jrnl serve` running,
`jrnl add --kind decision` lists similar earlier decisions and the command to
link one; it never links by itself.
