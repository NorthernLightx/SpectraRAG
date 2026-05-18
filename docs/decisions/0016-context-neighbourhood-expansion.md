# ADR 0016 — Context-neighbourhood expansion: built, inconclusive, not shipped

**Status:** Not shipped — fair test gives a weak, within-noise signal.
`ContextExpansionRetriever` retained, opt-in-off, pending confirmation.
**Date:** 2026-05-18.

## Context

ADR 0013/0015 showed the reranker and routing are marginal or
artifact-driven under honest measurement. The remaining untested
intuition: a retrieved chunk rarely answers alone — pull its
*neighbourhood* (the ±k page-sequential chunks and the figures/tables
its text references) before generation, the way a human reads *around* a
sentence. Measured by the honest metric the saturation failures pointed
to: **answer-correctness vs `expected_facts`, judged, in the realistic
multi-doc setting**.

## Build

`src/rag/retrievers/context_expansion.py` — a `Retriever` decorator
(mypy-strict + ruff clean, functional smoke verified): window neighbours
+ figure/table linking; exact passthrough when `window=0,
link_artifacts=False` (the eval baseline arm). Harness:
`scripts/experiments/study_context.py`, `data/golden/robust-v1.yaml`, 2 decisive arms
(baseline vs `+both`), stratified screen, incremental persistence.

## Findings

1. **First run was a context-window artifact, not a result.** gemma3:4b
   defaults to a 4096-token window; the generator packs up to ~8000
   tokens of context (`+both` ~doubles it) and the prompt is
   question-first — so the model read a truncated prompt and `+both`
   scored −0.35 across *every* bucket uniformly. Diagnosed from the
   uniform-collapse pattern + `ollama ps`. Fixed with
   `OllamaChatClient(num_ctx=16384)` so both arms run untruncated. (Caveat:
   earlier gemma3:4b *judged* numbers in ADR 0013 may have been similarly
   ctx-truncated — a further reason those were already flagged low-confidence.)

2. **Fair run (n=40 stratified, no-rerank shared base, untruncated):**

   | arm | overall | text | figure | table | mixed |
   |---|---|---|---|---|---|
   | baseline | 0.6125 | 0.900 | 0.350 | 0.650 | 0.550 |
   | +both | 0.6500 | 0.950 | 0.450 | 0.750 | **0.450** |

   Δ = **+0.0375**. The harness printed "WIN" (≥ +0.03 bar) but **that
   bar was too crude**: on n=40 with a coarse 0/0.5/1 judge the standard
   error is ~±0.07; each per-bucket 0.10 is a single query flipping; and
   `mixed` *regressed* −0.10. The effect is **within noise and
   non-uniform → inconclusive, not a confirmed win.**

3. Directionally, the gains concentrate where the mechanism predicts —
   `figure` and `table` (+0.10 each), the buckets where pulling a
   referenced artifact's caption should help. So the idea is *plausibly*
   real, unlike the reranker/routing dead-ends, but unproven here.

## Decision

**Do not ship `+both`.** Keep `ContextExpansionRetriever` in-tree,
opt-in-off (clean, reversible, promising-but-unproven). A proper
confirmation — full n≈84, paired test + CI, rerank-on, plus the
`+window` / `+links` ablation to attribute the effect — would settle it.
Given every lever explored this session proved marginal under honest
measurement, whether that confirmation is worth the compute is a
diminishing-returns call left to the maintainer; the evidence does not
justify a default-on change.

## What this leaves open

- Confirmation run (bigger n, paired stats, rerank-on, ablation).
- `num_ctx` was unset for gemma3:4b throughout the session — prior
  gemma3:4b-judged generation numbers (ADR 0013 Phase 2) are likely
  context-truncated; already flagged low-confidence, now with a mechanism.
- A stronger judge than gemma3:4b would tighten the noise band.

## Related

- ADR 0013 / 0015: the marginal/artifact pattern that motivated trying
  this last untested lever and the honest-metric requirement.
- ADR 0011 / 0009: figure-caption aggregation / region evidence — the
  artifact chunks the `link_artifacts` path exploits.
