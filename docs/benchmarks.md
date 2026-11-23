# Benchmarks with other models

These were measured before jrnl had a built-in embedding model, with
EmbeddingGemma 2 through Ollama as the default. They still describe what the
optional `[embed]`, `[rerank]` and `[chat]` roles add. The built-in model's
figures are in the README.

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

| | Keywords only | With Ollama and Workers AI |
|---|---|---|
| Question with the entry's own words: right entry first | 30 of 30 | 30 of 30 |
| Question in other words: right entry first | 8 of 30 | 21 of 30 |
| Search time, median / slowest 5% | 0.06 s / 0.07 s | 1.1 s / 4.2 s |
| Adds per second, 16 sessions at once | 156 | 58 |

The reworded questions avoid the entry's words on purpose, so they are the
hard case. Search by meaning is what closes that gap.
