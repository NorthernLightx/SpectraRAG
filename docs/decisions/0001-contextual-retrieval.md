# ADR 0001 — Contextual retrieval as the parser-robustness layer

**Status:** Rejected (2026-05-01). Without rerank: 3 local A/Bs all regress on
recall@10 (-14.3%). With rerank: contextual is *neutralized* — identical metrics
to rerank-only baseline. Production will use rerank, so contextual provides no
marginal value while costing an LLM call per chunk at ingest time.
**Date:** 2026-04-28 (initial); 2026-04-30 (runs #1–#3); 2026-05-01 (run #4 + verdict).
**Phase:** 1.4b.

## Context

PyMuPDF-based ingestion produces flat text. Our regex-based section-heading
detector matches `^\s*\d+(?:\.\d+)*\s+Title$`, which fails on:

- Two-column layouts (column-flattened text rejoins headings into body).
- Headings without numeric prefixes ("Abstract", "References").
- Wrapped/multi-line headings.

In the live baseline against ArXiv `2604.22753v1` (91 chunks), most chunks ended
up with `section='?'`. Eval baseline:

| Metric | Value |
|---|---|
| nDCG@5 (macro, in-corpus) | 0.5283 |
| recall@10 (macro) | 0.8750 |
| MRR (macro) | 0.5000 |

`recall@10` is high (0.875), but `nDCG@5` and `MRR` show the right chunks are
present in the candidate pool but not ranked at the top.

We considered three families of fix:

1. **Swap parser** (docling, Marker, OpenDataLoader) — adds a heavy dependency
   and is only as good as the new parser. Still has failure modes on
   adversarial PDFs.
2. **PyMuPDF font-size heuristic + `get_toc()` fallback** — cheap (no new
   deps), but no published evidence of magnitude of improvement on retrieval
   metrics; band-aid for one specific failure mode.
3. **Contextual retrieval** ([Anthropic, Sept 2024][1]) — for each chunk, an
   LLM produces a 50-100 token blurb situating the chunk inside the paper.
   The blurb is prepended to the chunk text *before* embedding/BM25 indexing.
   Display text and citations are unchanged.

[1]: https://www.anthropic.com/news/contextual-retrieval

## Decision

Adopt **contextual retrieval** (option 3) as the primary robustness layer.

**Why this over a parser swap:**

- **Parser-agnostic.** Works whether sections come out clean or as `'?'`. Does
  not commit us to a specific PDF library.
- **Evidence-backed.** Anthropic reports ~35% retrieval improvement; the
  technique is in production use in published systems.
- **Smaller blast radius.** No new heavy dependency (we already have
  `OpenRouterClient`); ingest-only LLM calls (offline, can be cached).
- **Composable.** If we later swap parsers, contextual retrieval still helps
  on top.

**Trade-offs accepted:**

- Adds an LLM call per chunk at ingest. For ArXiv-sized papers (~90 chunks)
  with `gpt-4o-mini` on OpenRouter, est. cost ≈ $0.05–0.15/paper.
- Ingest latency dominated by LLM round-trips (mitigated with concurrency=4).
- Display text and citations remain the original chunk text — only embedding
  + BM25 see the situating blurb. (Designed this way intentionally: we never
  want to cite an LLM-generated sentence.)

## Implementation

- `src/types/documents.py` → `Chunk` gained `context: str | None`. New
  `Chunk.indexed_text` property returns `f"{context}\n\n{text}"` when context
  is set, else `text`.
- `src/ingestion/contextualize.py` — new module:
  `contextualize_chunks(chunks, paper_text, llm, model, ...)` returns new
  chunks with `context` populated (uses `model_copy`, originals are not
  mutated).
- `src/ingestion/pipeline.py` — `ingest_paper` accepts optional
  `contextualizer_llm` + `contextualizer_model`. When both are set, runs
  contextualize between chunking and embedding. Both BM25 and the embedder
  now index `chunk.indexed_text` (no behavior change when context is None).
- `src/rag/bm25.py` — tokenizes `chunk.indexed_text` instead of `chunk.text`.
- `scripts/eval_run.py` — new flags `--contextualize`,
  `--contextualize-provider {openrouter,ollama}` (default `openrouter`),
  `--contextualize-model` (default per-provider:
  `openai/gpt-4o-mini` for openrouter, `qwen2.5:7b` for ollama),
  `--contextualize-concurrency`.
- `src/llm/ollama_chat.py` — `OllamaChatClient` duck-types `LLMClient` against
  Ollama's `/api/chat`. No new protocol, no new dep — same `httpx` + `tenacity`
  stack as `OpenRouterClient`. Provides a zero-spend path for the live A/B.

Read paths (retrieval, generation, citations) are unchanged: queries are
embedded as-is; results expose `chunk.text` to the LLM and to citations.

## Verification (already done)

- 137/137 unit tests pass; 6 new tests cover the contextualizer + indexed_text.
- mypy strict + ruff clean.
- Re-ran baseline eval through the new code path with no contextualize:
  nDCG@5=0.5283, recall@10=0.8750, MRR=0.5000 — **identical** to the
  pre-refactor baseline. The refactor is provably non-regressive when
  `context is None`.

## Decision rule for the A/B (pending live run)

Run on the same PDF + same 5 golden queries. Adopt contextual retrieval as
the default if **either**:

1. Macro nDCG@5 improves ≥ 5% (i.e., ≥ 0.555), **or**
2. Macro MRR improves ≥ 10% (i.e., ≥ 0.55).

Reject if recall@10 *drops* by more than 5% (would be a sign the blurb is
introducing noise that hurts BM25).

## A/B run #1 — local Ollama, llama3.2:3b on RTX 3070 (2026-04-30)

Run ID `b156ef45369b` (`data/eval/runs/run-20260430-191600.{json,md}`).
91 chunks contextualized in 327s; all 91 got blurbs. Same paper, same
golden set, same retriever wiring as the baseline.

| Metric (in-corpus, n=4) | Baseline | Contextual (llama3.2:3b) | Δ |
|---|---|---|---|
| nDCG@5 | 0.5283 | 0.4936 | **−6.6%** |
| recall@10 | 0.8750 | 0.7500 | **−14.3%** |
| MRR | 0.5000 | 0.4333 | **−13.3%** |

Per-query: q1/q2 unchanged, q3 lost a relevant chunk from the top-10
(recall 1.0 → 0.5), q4 dropped on both nDCG and MRR.

Decision rule literal reading: recall@10 dropped >5% → **reject**. Per the
local-path caveat below, this is **not** a reject of the technique — it's
strong evidence that a 3B-class local model produces blurbs noisy enough to
hurt retrieval at the BM25 layer (added tokens dilute term frequency without
improving topicality). Anthropic's published 35% lift used Claude Haiku /
Sonnet-class models, not a 3B.

**Status remains Proposed**, blocked on a second A/B with a stronger LLM.

## A/B run #2 — local Ollama, qwen2.5:7b on RTX 3070 (2026-04-30)

Run ID `b2101a28f9d7` (`data/eval/runs/run-20260430-193713.{json,md}`).
91 chunks contextualized in 972s; all 91 got blurbs. Same paper, same
golden set, same retriever wiring as the baseline.

| Metric (in-corpus, n=4) | Baseline | 3B (run #1) | 7B (run #2) | Δ vs baseline |
|---|---|---|---|---|
| nDCG@5 | 0.5283 | 0.4936 | 0.5377 | **+1.8%** |
| recall@10 | 0.8750 | 0.7500 | 0.7500 | **−14.3%** |
| MRR | 0.5000 | 0.4333 | 0.5208 | **+4.2%** |

Per-query: q1/q2 unchanged. q3 *gained* on ranking (nDCG 0.307→0.387,
MRR 0.333→0.500) but *lost* a relevant chunk from the top-10
(recall 1.0→0.5). q4 dropped on nDCG and MRR. The pattern is consistent
with blurbs adding signal that helps surface *some* relevant chunks higher
while displacing *other* relevant chunks below the top-10 cutoff — i.e.,
ranking-metric improvements paid for with coverage loss.

Decision rule literal reading: nDCG@5 +1.8% (below +5%) and MRR +4.2%
(below +10%) so neither adoption clause triggers; recall@10 −14.3% (above
−5% threshold) triggers the reject clause. Per the local-path caveat, a
regression escalates to cloud before flipping `Status:`.

**Critical confound (both runs):** Ollama's default `num_ctx=4096` truncated
the input prompt from ~22k tokens to 4096 on every call (the system prompt
+ chunk are kept; the paper text is *severely* truncated, with only the
tail surviving). The local-blurb regression is therefore not clean signal
on the technique — it's "Anthropic-style contextual retrieval, but the LLM
saw at most ~10 KB of the paper instead of 60 KB." A cloud A/B with
gpt-4o-mini (128k context) would feed the model the full truncated paper
text and produce a cleaner read. A future local A/B with `num_ctx=16384`
on the 7B model (still fits in 8 GB VRAM with bge-m3 evicted) would also
work but isn't wired through `OllamaChatClient` yet.

**Status remains Proposed**, blocked on either (a) a cloud A/B or (b) a
local A/B with bumped `num_ctx`.

## A/B run #3 — local Ollama, qwen2.5:7b @ num_ctx=8192 (2026-04-30)

Run ID `12ae544fb1d0` (`data/eval/runs/run-20260430-202105.{json,md}`).
Doubled the context window vs runs #1/#2 to test whether 4096-token
truncation was the real cause of the regression. 91 chunks contextualized
in 1829s (≈30 min, ~2.4× slower per call than 4k).

Ollama logs confirm `limit=8192` — `num_ctx` wiring works — but prompts
are still ~22k tokens, so we still truncate (now keeping the *last* 8k
instead of the last 4k of paper text + chunk).

| Metric (in-corpus, n=4) | Baseline | 3B@4k | 7B@4k | 7B@8k | Δ vs baseline |
|---|---|---|---|---|---|
| nDCG@5 | 0.5283 | 0.4936 | 0.5377 | **0.3160** | **−40.2%** |
| recall@10 | 0.8750 | 0.7500 | 0.7500 | 0.7500 | −14.3% |
| MRR | 0.5000 | 0.4333 | 0.5208 | **0.3819** | **−23.6%** |

Per-query collapse on q2 (`nDCG@5: 0.500 → 0.000`, MRR `0.333 → 0.167`)
and q3 (`nDCG@5: 0.307 → 0.000`). The relevant chunks for those queries
fell out of the top-5 entirely.

Counter-intuitive: more context made it *worse*, not better. Likely
mechanism — at 4k truncation the model sees mostly the *chunk* + a sliver
of paper tail and writes tightly chunk-focused blurbs. At 8k it sees
~half the paper (skewed to the *end*: discussion, conclusion, references)
plus the chunk, and writes globally-flavoured blurbs that pull in
end-of-paper terminology, diluting the chunk's specific terms (BM25 term
frequency drops; dense embedding drifts toward generic topic vectors).

This is consistent with reports that contextual retrieval gains depend
strongly on a strong instruction-following model that can stay
chunk-focused while given paper context. A 7B Q4-quantized local model
at 8k context apparently doesn't.

**Pattern across all 3 local runs:** recall@10 dropped by exactly 14.3%
in every run, and ranking metrics moved unpredictably with model + ctx.
Together this suggests *some* relevant-chunk displacement is intrinsic to
*any* blurb prepending we tried — the chunk's specific terms get diluted
by general-purpose situating sentences, regardless of who wrote them.

## A/B run #4 — local Ollama qwen2.5:7b @ num_ctx=4096 + cross-encoder rerank (2026-05-01)

Run ID `4afe3afe28ff` (`data/eval/runs/run-20260501-111934.{json,md}`).
Same paper, same golden set, same retriever wiring, but with the BGE
cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, top-50 → top-10) applied
on top of hybrid (BM25+dense+RRF). Compared against rerank-only baseline
(run `379862feedf1`, same retriever stack without contextualization):

| Metric (in-corpus, n=4) | Rerank-only | Rerank+contextual | Δ |
|---|---|---|---|
| nDCG@5 | 0.8160 | 0.8160 | **0.0%** |
| recall@10 | 0.8750 | 0.8750 | **0.0%** |
| MRR | 0.8750 | 0.8750 | **0.0%** |

Per-query results identical to rerank-only across all 5 queries.

**Mechanism:** `BgeReranker.rerank(query, chunk.text)` scores the original
chunk text — not `chunk.indexed_text` — so the situating blurb is invisible
to the cross-encoder. Whatever the blurb does to the RRF candidate-pool
ordering, the reranker's top-50→top-10 selection is dominant and identical.

This is the cleanest possible signal that, *with rerank in the pipeline*,
contextual retrieval provides exactly zero marginal value while charging an
LLM call per chunk at ingest time.

## Decision (final, 2026-05-01): Rejected

Three regressing runs without rerank + one neutralized run with rerank =
the technique provides either negative or zero value on this corpus
across the full grid we tested. Production deployment will include the
reranker (it provides +54.5% nDCG@5 and +75% MRR over hybrid alone — see
`docs/decisions/0002-rerank.md` if/when it lands), so the "with rerank"
cell is the one that matters. Contextual retrieval is therefore rejected.

**Caveats / what could still flip this:**

1. **Larger golden set** (`data/golden/v1.1.yaml`, 15 queries — later
   folded into v2). All current numbers are macro-averaged over 4 in-corpus
   queries — variance is high. If the rerank+contextual delta on a larger
   set is non-zero in either direction, the verdict can be revisited.
2. **Cloud A/B with `gpt-4o-mini` + 128k context** to test whether
   cloud-quality blurbs change the candidate-pool composition enough that
   rerank picks materially different chunks. Only worth running if we get
   a strong reason to believe the candidate pool is a bottleneck for
   recall@10 — currently it isn't (recall@10 = 0.875 already, capped by
   the 4-query golden set).
3. **Reranker-of-`indexed_text`**: an alternative wiring that scores
   `(query, indexed_text)` instead of `(query, text)`. Might let blurbs
   directly influence rerank scoring. Trade-off: longer rerank inputs
   (slower) and unclear signal-to-noise. Not currently planned.

## Pending

Two live A/B paths — pick whichever is more convenient.

**Cloud (highest signal-to-noise):** requires `RAG_OPENROUTER_API_KEY`.

```bash
uv run python -m scripts.eval_run \
    --pdf data/papers/2604.22753v1.pdf \
    --golden data/golden/v1.yaml \
    --collection eval_phase1_ctx_or \
    --contextualize \
    --contextualize-provider openrouter \
    --contextualize-model openai/gpt-4o-mini
```

**Local 7B+ (zero spend, fits in RTX 3070's 8 GB):** pull a 7B-class model
first — e.g. `docker exec rag-ollama ollama pull qwen2.5:7b` (~4.7 GB).
Requires GPU passthrough (already wired in `docker-compose.yml`).

```bash
uv run python -m scripts.eval_run \
    --pdf data/papers/2604.22753v1.pdf \
    --golden data/golden/v1.yaml \
    --collection eval_phase1_ctx_qwen \
    --contextualize \
    --contextualize-provider ollama \
    --contextualize-model qwen2.5:7b
```

Caveat for the local path (validated empirically by run #1): a 3B-class
model produces blurbs noisy enough to *hurt* retrieval at the BM25 layer.
A 7B-class model is the minimum we should trust for this signal. If a 7B+
local A/B still regresses, escalate to cloud before flipping `Status:`.
If the local 7B+ A/B shows a *win*, that's sufficient signal to adopt
(the technique is robust enough that even a small model helps).

Update this ADR with measured numbers from the next run; flip `Status:` to
Accepted or Rejected based on the rule above.

## Alternatives considered

- **Docling A/B** (parser swap): deferred. If contextual retrieval is
  insufficient after measurement, revisit. Note: the two are not mutually
  exclusive.
- **Font-size heuristic in PyMuPDF**: deferred. Lower ceiling, no published
  evidence of retrieval-metric impact.

## References

- Anthropic, *Introducing Contextual Retrieval*, Sept 2024.
  <https://www.anthropic.com/news/contextual-retrieval>
