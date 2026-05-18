# ADR 0012 — Reranker swap investigated; incumbent kept (premise falsified)

**Status:** Rejected — no swap; `BAAI/bge-reranker-v2-m3` stays the default.
**Date:** 2026-05-17.

## Context

A recurring optimisation idea (results.md:44, the ADR 0010 "smaller
cross-encoder" thread) is to replace the 568M `bge-reranker-v2-m3`
cross-encoder with a cheaper one. The stated motivation was latency:
`docs/results.md:39` lists the reranker at **~5,500 ms** and calls it the
dominant whole-query cost.

Before swapping anything, the premise and the accuracy cost were measured.

## Investigation

Instruments (new): `scripts/experiments/bench_rerank.py` (rerank-stage latency +
correctness, faithful to `src/rag/rerank.py:106` — `CrossEncoder.predict`
over the real `--rerank-input-size 50` workload); `scripts/experiments/run_doe.py`
(Phase 1: keyless deterministic retrieval sweep — retrieval metrics are
exact because `eval_run` `run_id` is a content hash); `scripts/experiments/run_doe2.py`
(Phase 2: judged metrics via local Ollama `gemma3:4b`, plus an
incumbent×3 judge-noise floor). No router / no ColQwen2 — isolates the
reranker and avoids the 8 GB OOM.

Models: `bge-reranker-v2-m3` (568M, incumbent), `bge-reranker-base` (278M),
`ms-marco-MiniLM-L-6-v2` (22M). Substrates: golden v3 and MMLongBench.
Raw data: `data/eval/runs/doe-20260517-004816/` (Phase 1) and
`doe2-20260517-014144/` (Phase 2).

## Findings

1. **The ~5.5 s premise is false on the current stack.** Faithfully
   measured on this GPU the incumbent rerank stage is **p50 ≈ 1.0–1.3 s**
   (`bench.txt`), not 5.5 s. `results.md:39` is explicitly tagged "From v2
   baseline" and sourced from `scripts/legacy/profile_latency.py` (RTX 3070,
   n=6); the figure is almost certainly cold-model-load-contaminated (cold
   load measured separately at ~9–27 s). Retrieval-only is <0.2 s;
   whole-query ≈ 3 s. The reranker is ~⅓ of the query, not the dominator.

2. **MMLongBench is null for a text-only reranker comparison.** All three
   models score **0.0000** ndcg@5/recall@10/mrr over 109 in-corpus queries
   (0/0/109, every query). MMLB relevance is page-level and visual; without
   the ColQwen2 leg the reranker cannot affect it. Only v3 carries signal.

3. **v3 retrieval (deterministic — strong evidence).**
   `ms-marco-MiniLM-L-6-v2` is **non-inferior** to the incumbent: ndcg@5
   Δ −0.005 (CI [−0.112, +0.090]), recall@10 **identical** (0/0/31), mrr
   Δ −0.016 — at **~20× lower latency** (60 ms vs ~1.2 s p50).
   `bge-reranker-base` is **decisively regressive**: ndcg@5 Δ −0.138, CI
   [−0.279, −0.009] excludes 0 (−0.18 factual, −1.0 equation, −0.47
   multi_hop). bge-base eliminated; not carried into Phase 2.

4. **v3 judged (weaker evidence — coarse local judge, n=31).** vs incumbent:
   answer_relevance Δ +0.004 (non-inferior); context_precision Δ −0.021
   (~−2.6 %, within the 5 % gate); faithfulness Δ −0.042 (~−5.4 %, above
   the 0.003 noise floor but CI [−0.124, +0.017] crosses 0 — the lone
   yellow flag, inconclusive at n=31). Judge caveat: `gemma3:4b` emitted
   near-constant scores (0.85 / 0.90 repeatedly); the small noise floor
   reflects a low-resolution judge as much as true stability, so the
   judged "non-inferiority" is soft, and absolute values are **not**
   comparable to `baseline.json` (different judge + config) — only the
   within-Phase-2 paired contrast is valid.

5. **v3 is itself reranker-insensitive.** recall@10 is bit-identical across
   a 25× parameter difference (568M vs 22M). The reranker reorders within
   the top-10 on v3; it rarely changes membership. Consistent with v3
   being an easy/saturated chunk-level set.

## Decision

**Keep `bge-reranker-v2-m3`. No swap. No `Settings.rerank_model` knob.**

The earlier plan to add a no-op `Settings.rerank_model` opt-in is
deliberately reversed: with the swap rejected the knob is speculative
config (a setting whose recommended value never changes), which the
project's hygiene bar forbids. The `eval_run --rerank-model` CLI flag
already covers reproducible experimentation.

The decisive result is meta, not the per-model table: **the reranker is
neither the latency lever (premise false) nor a meaningful accuracy lever
(v3 insensitive, MMLB visual) in this repo.** A 22M model matching a 568M
one for ~nothing is not a win to ship — it is evidence that this stage
does not matter here. The high-impact levers remain the visual retriever
(ADR 0004/0008) and the dispatch classifier; this investigation
triangulates that conclusion from a third, independent angle.

## What this leaves open

- **results.md:39,166 is stale/misleading** (~5.5 s "From v2 baseline").
  Worth correcting to the measured ~1 s with provenance — a docs fix,
  flagged here, not done in this ADR.
- **API-vs-eval rerank parity (unverified).** A prior code analysis
  claimed `src/api/main.py` builds `PipelineRetriever` *without* a
  reranker while `eval_run` reranks. Not verified here; flagged as a
  separate question and deliberately **not** silently changed.
- **MMLB reranker effect is untestable text-only.** Would require
  `--router` (GPU-heavy, 8 GB OOM risk) and is low value given the swap
  is rejected.
- **faithfulness −5.4 % on MiniLM is unresolved** (wide CI, coarse judge).
  Resolving it needs a stronger judge (e.g. `qwen3-vl:235b-cloud`) and
  larger n; not worth it since the swap is rejected regardless.
- **Artifacts.** `scripts/experiments/bench_rerank.py` is a reusable rerank latency
  harness (the repo previously had only `scripts/legacy/profile_latency.py`)
  and is worth keeping. `scripts/experiments/run_doe.py` / `run_doe2.py` are one-off
  DoE drivers; keep-or-remove is a maintainer call.

## Related

- ADR 0003 (query expansion): same shape — investigated, rejected with
  data, code kept opt-in-off. This ADR adds a second "measured, not the
  lever" result.
- ADR 0010 (cost-quality cascade): closes its "smaller cross-encoder"
  thread — a smaller cross-encoder does not help because the reranker is
  not the bottleneck the cascade assumed.
- ADR 0004 / 0008 (visual retrieval / routing): the actual high-impact
  lever, reinforced here by elimination.

## Correction (2026-05-17)

Finding 2 ("MMLongBench null for a text-only reranker comparison … the
reranker cannot affect it") is **wrong**. The Phase 1 sweep ran the
MMLongBench golden against the wrong corpus — the 20 `2604.*` v3 arXiv
PDFs in `data/papers/` — instead of MMLongBench's own documents in
`data/mmlongbench/documents/`. The 0.0000 across all models was a
corpus mismatch, not a property of the benchmark or evidence that the
reranker is irrelevant there. The repo's correctly-run numbers
(`docs/results.md` §"Stress test on MMLongBench-Doc";
`data/eval/baseline-mmlongbench.json`) show text-only MMLongBench at
recall@10 ≈ 0.685, not 0.

The **decision is unaffected**: rejecting the reranker swap rested on the
v3 deterministic sweep (whose corpus, `data/papers/`, was correct) and
the falsified ~5.5 s latency premise — not on the MMLongBench arm, which
was never load-bearing here. Finding 2 should be read as void; Findings
1, 3, 4, 5 and the decision stand.
