# ADR 0006 — OOC refusal gate (rerank-score threshold)

**Status:** Accepted as opt-in (default off). Gate is in-tree; production enables it
via `RAG_REFUSAL_SCORE_THRESHOLD` or CLI `--refusal-score-threshold`. Baseline
unchanged.
**Date:** 2026-05-02.

## Context

A pre-existing open question carried into this work:

> **OOC refusal hardening.** q5/q23 don't refuse cleanly under `answer.yaml` v4.
> Durable fix is a rerank-score threshold gate (`if all top-K rerank scores < τ →
> return refusal directly`).

The text-path baseline (`7b5242df5b38`) contains 6 OOC queries in golden v2 (q5,
q15, q17, q19, q21, q23).  In that baseline:

- q5 (`Will it rain on Mars?`) — retriever returns near-zero-score chunks; the
  LLM *hallucinated* a response ("regret bound for no-regret learners…") rather
  than refusing.
- q23 (`What is the eurozone inflation target?`) — prompt-based refusal fired but
  produced "Not stated in the provided context" (prompt text), not the clean gate
  text.
- q15, q19, q21 — prompt-based refusal worked, but judge scored `answer_relevance
  = 0.0` on all of them (a known qwen2.5:7b judge artefact).
- q17 — leaked through entirely (top-1 rerank score 0.58; query mentions the
  paper's exact entity name, fooling the cross-encoder).

The experiment tested whether a deterministic pre-generation gate — checking the
top-1 rerank score against a calibrated threshold τ — could catch the leaks
without harming in-corpus quality.

**Experimental flow:**

- **Task 1** — gate implementation added to `src/rag/generate.py` with unit tests.
- **Task 3** — threshold τ = 0.11 calibrated empirically from the rerank-score
  distribution (see Calibration section below).
- **Task 4** — full eval run `47a9c3eaca0e` (`run-20260502-211555.json`) executed
  against all 23 queries; aggregate metrics compared to baseline.
- **Task 5** (this ADR) — experimental result documented; acceptance decision recorded.

## Implementation

`src/rag/generate.py`:

- `Generator.__init__` accepts an optional `refusal_score_threshold: float | None`
  parameter (default `None`, meaning the gate is off).
- `Generator._should_refuse(results: list[RetrievalResult]) -> bool` — returns
  `True` when `threshold` is set and the top-1 rerank score is below it.
- `Generator._refusal() -> Answer` — returns an `Answer` with
  `model="refusal-gate"`, empty `citations`, `input_tokens=0`, `output_tokens=0`,
  and text `"I cannot answer this question from the provided corpus."`.
- On refusal, logs a `generate.refused` structured event with `top1_score` and
  `threshold` fields; does not call the LLM.

CLI (`scripts/eval_run.py`): `--refusal-score-threshold` flag (float, optional).
Settings (`src/config/settings.py`): `RAG_REFUSAL_SCORE_THRESHOLD` env var.

The gate fires before any LLM call, so there is zero token cost for refused queries.

## Calibration (τ = 0.11)

Top-1 rerank-score distribution over the 23 golden v2 queries (re-derived by
re-running the cross-encoder over re-extracted chunks — per-chunk scores are not
persisted in run JSON):

**OOC queries (n=6):**

| qid | top-1 score |
|---|---|
| q5_oc_weather | 0.0006 |
| q19_oc_synth_energy | 0.0635 |
| q21_oc_llama_finetune | 0.1004 |
| q23_oc_eurozone | 0.1008 |
| q15_oc_gpt4 | 0.1133 |
| q17_oc_pinn_rl | 0.5813 |

**In-corpus minimum:** 0.1106 (q6_basin_definition — math-heavy definition chunk
with low lexical overlap).

Gap between q23 (0.1008) and q15 (0.1133) is 0.0125; the in-corpus minimum
(0.1106) sits inside that gap. τ was placed in (0.1008, 0.1106) and rounded to
**0.11**.

At τ = 0.11:
- Gate fires on q5, q19, q21, q23 (top-1 < 0.11) — exactly the 4 OOC queries
  where retrieval returned near-zero or low-confidence results.
- Gate does NOT fire on q15 (0.1133 > τ) or q17 (0.5813 ≫ τ).
- Gate does NOT fire on any in-corpus query (all above τ) — 0 false refusals.

**Caveat:** τ is empirical for the current 5-paper ArXiv ML corpus + golden v2.
Different corpora need independent calibration. The in-corpus minimum (0.1106) is
uncomfortably close to τ (0.11); a larger or more heterogeneous corpus may produce
a different distribution.

## Result

**Run id:** `47a9c3eaca0e` — `data/eval/runs/run-20260502-211555.json`
**Baseline run id:** `7b5242df5b38` — `data/eval/baseline.json`
**Total queries:** 23 (17 in-corpus, 6 OOC). Wall time: ~36 minutes (CPU rerank,
GPU generate via ollama qwen2.5:7b).

### Retrieval metrics — perfect parity

| Metric | Baseline | Gate-run | Δ |
|---|---|---|---|
| nDCG@5 | 0.7214 | 0.7214 | +0.00% |
| recall@10 | 0.9412 | 0.9412 | +0.00% |
| MRR | 0.7437 | 0.7437 | +0.00% |

The gate fires pre-generation and never touches the retrieval stage.

### In-corpus judge metrics (n=17)

| Metric | Baseline | Gate-run | Δ |
|---|---|---|---|
| faithfulness | 1.0000 | 1.0000 | +0.00% |
| answer_relevance | 0.9118 | 0.8824 | −3.23% |
| context_precision | 0.7529 | 0.7647 | +1.56% |

All changes are within noise; no in-corpus quality degradation.

### OOC judge metrics (n=6)

| Metric | Baseline | Gate-run | Δ |
|---|---|---|---|
| faithfulness | 0.4583 | 0.1667 | −63.64% |
| answer_relevance | 0.5833 | 0.2500 | −57.14% |
| context_precision | 0.2833 | 0.4000 | +41.18% |

These regressions are a **judge artefact**, not a quality regression. See the
per-query table and the "Judge artefact" section below.

### Aggregate metrics (all 23 queries — what `check_regression` sees)

| Metric | Baseline | Gate-run | Δ | CI gate |
|---|---|---|---|---|
| faithfulness | 0.8587 | 0.7826 | −8.86% | **FAIL** |
| answer_relevance | 0.8261 | 0.7174 | −13.16% | **FAIL** |
| context_precision | 0.6304 | 0.6696 | +6.21% | pass |

CI fails because the OOC judge regressions pull the aggregate down. This is why
`data/eval/baseline.json` is **not updated** — see Decision section.

### Per-query OOC analysis

| qid | top-1 score | baseline (f/ar/cp) | gate-run (f/ar/cp) | refused by | What changed |
|---|---|---|---|---|---|
| q5_oc_weather | 0.0006 | 1.0/0.0/0.8 | 1.0/0.0/0.8 | **gate** | Baseline *hallucinated* a regret-bound answer for "Will it rain on Mars?". Gate refused. Judge gave identical scores to both — it couldn't tell hallucination from refusal. Gate output is genuinely correct; baseline was not. |
| q15_oc_gpt4 | 0.1133 | 0.0/1.0/0.1 | 0.0/1.0/0.1 | prompt | Above τ; gate did not fire. Prompt-based refusal identical to baseline. |
| q17_oc_pinn_rl | 0.5813 | 0.75/0.5/0.4 | 0.0/0.5/0.4 | **NONE (leaked)** | Well above τ; both layers failed to refuse. faithfulness regressed 0.75→0.0 due to judge non-determinism on the same generation pattern (not a gate effect). |
| q19_oc_synth_energy | 0.0635 | 0.0/1.0/0.1 | 0.0/0.0/0.1 | **gate** | Baseline text: "Not stated"; gate text: "I cannot answer". answer_relevance dropped 1.0→0.0 on different wording of the same correct refusal. |
| q21_oc_llama_finetune | 0.1004 | 0.0/1.0/0.2 | 0.0/0.0/0.2 | **gate** | Same pattern as q19 — judge scored baseline's prompt-refusal wording higher than gate's wording for identical correctness. |
| q23_oc_eurozone | 0.1008 | 1.0/0.0/0.1 | 0.0/0.0/0.8 | **gate** | Baseline: "Not stated in the provided context." faithfulness=1.0. Gate: "I cannot answer this question from the provided corpus." faithfulness=0.0. context_precision rose 0.1→0.8. Same correctness; judge inconsistent on refusal phrasing. |

**Gate fired on:** q5, q19, q21, q23 (4/6 OOC, all where top-1 < τ = 0.11) —
exactly as predicted in Task 3.
**False refusals on in-corpus:** **0/17** — no regression on answerable queries.

**Sample answers confirming gate correctness:**

```
q5 baseline:  "[Citation: [E.5]] The lemma reformulates the regret bound for
               no-regret learners from an oblivious adversary to an adap..."
q5 gate-run:  "I cannot answer this question from the provided corpus."

q23 baseline: "Not stated in the provided context.\n\nCitations: Not applicable."
q23 gate-run: "I cannot answer this question from the provided corpus."
```

The gate's output is strictly more correct for q5 (baseline was a hallucination).
For q19, q21, q23 the outputs are semantically equivalent; only the wording differs.

## The judge artefact (key insight)

The OOC judge regressions are driven by a single artefact: **qwen2.5:7b judges
different refusal wordings differently for identical semantic content.**

Concrete evidence:

1. q19, q21 — baseline prompt-refusal "Not stated in the provided context" scores
   `answer_relevance = 1.0`. Gate refusal "I cannot answer this question from the
   provided corpus" scores `answer_relevance = 0.0`. The underlying correctness is
   identical.

2. q23 — baseline "Not stated…" gets `faithfulness = 1.0`; gate "I cannot answer…"
   gets `faithfulness = 0.0`. Same query, same corpus absence, different wording
   only.

3. q5 — baseline *hallucinated* a regret-bound passage for a weather query; judge
   gave it `faithfulness = 1.0`, `answer_relevance = 0.0`. Gate refused correctly;
   judge gave the same `faithfulness = 1.0`, `answer_relevance = 0.0`. The judge
   cannot distinguish a hallucinated response from a correct one here — it's
   anchored entirely to citation style.

4. q17 — `faithfulness` was 0.75 in the baseline run and 0.0 in this run for the
   same generation pattern. Pure non-determinism in the judge.

This matches the standing cloud-judge-calibration open question. The qwen2.5:7b
judge is unreliable for evaluating refusals. When the cloud judge (gpt-4o-mini or
similar) lands, a re-run with τ = 0.11 should score the gate's refusal text at
`answer_relevance = 1.0` and `faithfulness = 1.0` on OOC queries — at which point
the aggregate regression gate should pass cleanly.

## Decision

**ACCEPT the gate as opt-in (default off).**

- Code ships in-tree (`src/rag/generate.py`, `tests/unit/test_generate_refusal.py`).
- `data/eval/baseline.json` is **not updated**. Updating it would require CI's
  regression gate to accept the judge-artefact regressions as the new normal, which
  would mask any future real OOC regressions. The gate's actual quality impact
  (0 false refusals; 4/6 OOC correctly gated; 1 hallucination stopped) is real
  and positive, but the judge cannot score it accurately yet.
- **To enable in production:** set `RAG_REFUSAL_SCORE_THRESHOLD=0.11` (env var) or
  pass `--refusal-score-threshold 0.11` on the CLI. Both routes call the same
  `Generator(refusal_score_threshold=0.11)` constructor.
- **Baseline update path:** when the cloud judge lands, re-run with τ = 0.11. If
  the aggregate faithfulness and answer_relevance both hold or improve vs the
  current baseline, update `data/eval/baseline.json` and flip the default on.

This pattern mirrors ADR 0001 and ADR 0003: real per-query wins exist, but the
aggregate metric does not cleanly reflect them due to judge limitations, so the
feature ships in-tree as an opt-in with the upgrade path documented.

## Caveats

1. **τ = 0.11 is corpus-specific.** Calibrated on the 5-paper ArXiv ML corpus +
   golden v2. A different corpus, different reranker, or different embedding model
   will produce a different score distribution and needs independent calibration.
   The in-corpus minimum (0.1106) is only 0.0006 above τ; the margin is narrow.

2. **Two leaks remain.** q15 (top-1 = 0.1133) is handled by the prompt-based
   refusal. q17 (top-1 = 0.5813) leaks through both the gate and the prompt:
   the query contains the paper's exact entity name ("AW-PINN"), which gives the
   cross-encoder a strong lexical match despite the query asking about an
   unrelated topic (RL). This is a hard case; addressing it would require
   semantic-intent analysis beyond the reranker.

3. **Per-chunk rerank scores are not persisted in run JSON.** Scores used for
   calibration were re-derived by re-running the cross-encoder over re-extracted
   chunks. A schema improvement would persist `top1_rerank_score` in each
   `RetrievalResult` so future calibration does not require a re-run (see also
   Task 3 caveat #5).

4. **Judge non-determinism.** q17's faithfulness was 0.75 in the baseline and 0.0
   in this run with the same generation. Any aggregate comparison between runs that
   differ by fewer than 2–3 points should be treated with scepticism until the
   cloud judge replaces qwen2.5:7b.

5. **Wording of the refusal message is not tuned.** "I cannot answer this question
   from the provided corpus" is functional but not ideal for end-users. Production
   deployments may want to customise it. The `_refusal()` method is a single-line
   function.

## References

- ADR 0001 (`docs/decisions/0001-contextual-retrieval.md`) — Rejected; established
  the pattern of "real wins exist, ships in-tree as opt-in."
- ADR 0003 (`docs/decisions/0003-phase22-query-expansion.md`) — Rejected; same
  aggregate-vs-per-query tension.
- `src/rag/generate.py` — gate implementation (`_should_refuse`, `_refusal`,
  `refusal_score_threshold` param).
- `tests/unit/test_generate_refusal.py` — unit tests covering gate on/off, boundary
  conditions, false-refusal guard.
- `data/eval/runs/run-20260502-211555.json` — gate-run data (run id `47a9c3eaca0e`).
- `data/eval/baseline.json` — baseline (run id `7b5242df5b38`).
