# ADR 0019 — Agentic retrieval tier: within noise, kept opt-in

**Status:** **Not shipped as the default tier.** Measured against the new
text-only hybrid baseline on v3, agentic-decomposition is **−0.2% on
answer_correctness overall** — within judge noise, not the +5% needed to
ship. Stays in tree opt-in (`--agentic` in `eval_run`), since the
per-category split is real and useful as a future routing input:
**figure +9.2pp**, **factual −5.0pp**, **table −10.0pp**. Fifth honest-
negative ADR in this repo (0012, 0013/0015, 0016, 0018, 0019).
**Date:** 2026-05-20

## Context

ADR 0018 rejected GraphRAG on this corpus after a measured kill-spike
(BM25-with-the-same-LLM beat it 5–1 on global synthesis). The remaining
SOTA-flavoured direction the original revamp named is **agentic** — but
agentic-retrieval works differently from graph-retrieval. It's a *per-query*
LLM-driven step on top of any base retriever; it does not depend on
cross-document graph structure (the precondition the spike showed this
small, sparse corpus does not satisfy). So it is the natural next thing to
measure here.

Two missing prerequisites were also fixed before any retrieval-quality
claim could be made honestly:

1. **`answer_correctness` metric did not exist** in `src/eval` (the
   skeptical-review B1 finding behind ADR 0018, and the metric ADR 0017
   incorrectly claimed was already in the harness). Landed in
   `src/eval/judges.py::LLMJudge.answer_correctness`, wired into
   `GenerationMetrics`, the runner, `eval_run`'s judge construction,
   `check_regression`'s gated metrics, and the markdown report. Tests:
   `tests/unit/test_judges.py::test_judge_answer_correctness_*`.
2. **No committed hybrid baseline on the new metric existed.** Producing it
   is part of this ADR's measurement, not a separate one.

## What "agentic" means here (and what it isn't)

`AgenticRetriever` in `src/rag/retrievers/agentic.py`:

- One LLM call **decomposes** the user's query into 1–N atomic
  sub-questions (`src/prompts/library/decompose_query.yaml`). Distinct
  from `MultiQueryRetriever`, which *paraphrases* one query (rewrite /
  HyDE).
- Each sub-question is retrieved via the wrapped base retriever in
  parallel; the lists fuse with RRF (same as `MultiQueryRetriever`, so
  the fusion mechanic is not the experimental variable here).
- Atomic decomposition (one line back) bypasses the fan-out entirely —
  identical to plain base retrieval, no cost penalty.
- LLM or parse failure falls back gracefully to a single base call on the
  original query.

No iterative self-grading / re-issue loop in this version. Minimum to get
the kill / continue signal (same discipline as ADR 0018's spike):
decomposition alone is the cheap test. If it doesn't beat hybrid here,
adding a self-grading loop won't either.

## Measurement (in flight)

Configuration both arms (text-only, clean attribution):

- 20 papers, post-ADR-0017 corpus (1,973 chunks).
- Golden v3 (33 queries) with `answer_correctness` vs `expected_facts`.
- `bge-m3` embedder, hybrid BM25 + dense + RRF.
- No rerank / router / figures / VLM / contextualisation — isolates the
  retriever change.
- Same `gemma3:4b` for generation and judge, `num_ctx=16384` so the
  prompt is never truncated (ADR 0016 artefact avoided).
- `--paper-id-filter` matches the committed baseline's eval convention.

Differential only: the **agentic** arm wraps the same retriever in
`AgenticRetriever` (decomposition with `max_subqueries=4`).

### Result

Run IDs hybrid `325375af3043` (`data/eval/baseline-text-only.json` +
`.md`), agentic `fd50bbda0212` (`data/eval/agentic-text-only.json` +
`.md`). Both `gemma3:4b` gen + judge, `num_ctx=16384`,
`--paper-id-filter`, no rerank / router / figures / VLM, 39 queries on
v3 (31 in-corpus with `expected_facts`).

| metric (n=in-corpus where applicable) | hybrid baseline | + agentic | Δ |
|---|---|---|---|
| **answer_correctness** (n=31) | **0.7626** | **0.7613** | **−0.0013 (−0.2%)** |
| faithfulness (n=39) | 0.7454 | 0.7538 | +0.0085 (+1.1%) |
| answer_relevance (n=39) | 0.6179 | 0.6308 | +0.0128 (+2.1%) |
| context_precision (n=39) | 0.8077 | 0.8179 | +0.0103 (+1.3%) |
| citation_rate (when cited) | 1.000 | 1.000 | 0.0 |
| p50 latency | 20.3 s | 20.8 s | +2.5% |
| p95 latency | 28.6 s | 25.8 s | −10.0% |
| total tokens out | 16 733 | 10 932 | −34.7% |

Retrieval nDCG/recall/MRR are not reported here: chunk-ids changed in
ADR 0017 and v3's `relevant_chunk_ids` are not re-anchored — both arms
score near 0 by construction. `answer_correctness` is the chunk-id-robust
scoreboard the ADR exists to produce; that is the metric the verdict
rests on.

### `answer_correctness` by category (the real story)

| category | n | hybrid | + agentic | Δ |
|---|---:|---:|---:|---:|
| equation | 1 | 1.000 | 1.000 | 0.000 |
| factual | 13 | **0.831** | 0.781 | **−0.050** |
| figure | 11 | 0.767 | **0.859** | **+0.092** |
| multi_hop | 2 | 0.800 | 0.800 | 0.000 |
| table | 4 | 0.450 | **0.350** | **−0.100** |

The overall flat number hides two real effects pointing in opposite
directions. **Decomposition helps where the question has multiple parts**
(figure: "what does Fig N show + how does X compare to Y" — 11 queries,
+9.2 pp). **Decomposition hurts where the question is atomic** — a
factoid lookup that the LLM fragments into noisier sub-questions whose
top retrievals are off-target (factual −5 pp, table −10 pp). `multi_hop`
is too small (n=2) to read, and the decomposition prompt may have
classified those particular two as atomic.

### Verdict

Decomposition-only agentic on this corpus is **within judge noise on the
headline metric** and the +5 % regression-gate bar is not crossed (or
broken). On-brand outcome: **not shipped as the default**, kept opt-in
behind `--agentic`. The per-category split is itself useful evidence for
a future *router-style* selective agentic (decompose only when a
classifier says the query is multi-part), but building and measuring that
is the next ADR's job, not this one's — exactly the discipline ADR
0013/0015 punished violating.

## What this leaves open

- **Selective / router-style agentic** is the obvious next experiment
  given the +9.2 pp on `figure` and −5/−10 pp on `factual`/`table`.
  A small classifier deciding "decompose y/n" would, on this corpus,
  plausibly recover the figure win without the factoid cost. ADR 0019
  does not build it: per the repo's discipline, one variable per ADR.
- Decomposition-only **without** a self-grading loop is the cheapest
  agentic configuration. Adding a grade-and-re-issue loop is more LLM
  cost; the current data does not justify it.
- Multimodal (figures/tables/visual routing) and contextual retrieval
  are unchanged — this ADR isolates the retrieval-decomposition variable
  exactly because ADR 0013→0015 punished conflated attribution.
- The judge (`gemma3:4b`) has known variance at this corpus size
  (ADR 0016: ±0.07 at n=40). A stronger judge (cloud Sonnet/Opus) would
  tighten the band; the −0.0013 overall delta might survive or might
  invert. Either outcome is honest; the current data says "not better"
  with the available judge.

## Related

- ADR 0018 — GraphRAG rejected on this corpus (sparse-graph precondition
  not met). Agentic does not share that precondition.
- ADR 0017 — corpus clean, the ingestion this measurement assumes.
- ADR 0016 — honest-metric requirement (`answer_correctness` vs
  `expected_facts`) finally implemented in `src/eval` for this ADR.
- ADR 0013 / 0015 — attribution discipline: isolate the retrieval
  variable, don't conflate with rerank/router/VLM.
