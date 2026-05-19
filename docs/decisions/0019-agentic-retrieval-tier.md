# ADR 0019 — Agentic retrieval tier (proposed, measurement in flight)

**Status:** Proposed. The metric (`answer_correctness` vs `expected_facts`)
and the retriever (`src/rag/retrievers/agentic.py`) are landed; the
baseline-vs-agentic comparison runs are in flight as of writing. The ADR
moves to Accepted or Rejected when both runs complete and the numbers go
into the "Measurement" section below.
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

_(Numbers go here when both runs complete.)_

| metric | hybrid baseline | + agentic | Δ |
|---|---|---|---|
| answer_correctness | – | – | – |
| faithfulness | – | – | – |
| answer_relevance | – | – | – |
| context_precision | – | – | – |
| nDCG@5 | – | – | – |
| recall@10 | – | – | – |
| MRR | – | – | – |
| p50 latency | – | – | – |

### Verdict

_(Continue/kill decision lands here against the eval's 5% regression gate
and a frank read of multi_hop-subset behaviour, which is the question
class agentic decomposition should win on.)_

## What this leaves open

- Per-query LLM cost is real: 1 extra decomposition call per query, plus
  N parallel retrieve calls inside the wrapped pipeline. The latency
  delta is reported in the table above.
- A self-grading loop (decompose → retrieve → grade → re-decompose) is
  not built. If the decomposition-only arm shows clear edge here, the
  next ADR can iterate; if not, neither would.
- Multimodal (figures/tables) and visual routing are unchanged here —
  this ADR isolates the retrieval-decomposition variable, exactly the
  attribution issue ADR 0013→0015 punished.

## Related

- ADR 0018 — GraphRAG rejected on this corpus (sparse-graph precondition
  not met). Agentic does not share that precondition.
- ADR 0017 — corpus clean, the ingestion this measurement assumes.
- ADR 0016 — honest-metric requirement (`answer_correctness` vs
  `expected_facts`) finally implemented in `src/eval` for this ADR.
- ADR 0013 / 0015 — attribution discipline: isolate the retrieval
  variable, don't conflate with rerank/router/VLM.
