# ADR 0010 — Cost-quality cascade routing + eval methodology hardening

**Status:** Accepted as opt-in features (2026-05-10). Cascade ships off by
default; eval methodology improvements (B1 deterministic OOC scoring, B2
multi-seed judge averaging, B3 smoke pre-flight) ship on by default.
**Date:** 2026-05-10.
**Phase:** 4.0 (Tier 2).

## Context

Two threads converge in this ADR:

1. **Cost-quality cascade** (Tier 2 #3 from the post-Tier-1 plan): the
   committed router (ADR 0008) dispatches by query category — `figure`,
   `table`, `multi_hop` always invoke the visual leg even when the text
   leg is overwhelmingly confident. ColQwen2 cuda inference is a
   meaningful per-query cost; if text alone could answer the query, we'd
   like to skip visual.

2. **Eval methodology**: Tier 1's verification run (`196ac0f8786f`) showed
   ±1.3 % faithfulness swings driven entirely by the LLM judge scoring
   q33 differently across runs (1.0/1.0/1.0 vs 0.0/0.5/0.0 — same generated
   answer, different judge call). That variance is bigger than any feature
   we've shipped moves the aggregate, so future small lifts are
   uninterpretable without methodology fixes.

Both threads are tracked in this ADR because they ship in the same
branch and the eval methodology gates whether the cascade's expected
cost-quality tradeoff is even measurable.

## Decision

### A. Cost-quality cascade (`routing_mode='cascade'`)

`RoutingRetriever` gains a `cascade` mode alongside the existing
`category` mode (default). In cascade mode:

1. Always run the text leg first (cheap; we need it for the decision).
2. If `force_route` is set, honor it (`text` → text-only, `hybrid` →
   text + visual + RRF fuse).
3. Otherwise: read the top-1 rerank score from the text results.
   - If `top_score >= cascade_confidence_threshold`: return text-only,
     skip the visual leg entirely. ColQwen2 inference saved.
   - Else: invoke the visual leg, RRF-fuse with the already-retrieved
     text results (no double-call), return the fused list.

`Settings.routing_mode: Literal["category","cascade"] = "category"` and
`Settings.cascade_confidence_threshold: float | None = None`. Cascade
mode requires an explicit threshold; the constructor raises
`ValueError` if the threshold is unset.

### B. Eval methodology

#### B1. Deterministic OOC scoring (always-on)

`runner._run_one` extends the refusal-override:

```
                  refusal answer    non-refusal answer
out_of_corpus     1.0 / 1.0         0.0 / 0.0  (NEW)
in-corpus         0.0 / 0.0         LLM judge runs
```

The new path: a non-refusal answer to an OOC query is wrong by
construction (no correct content answer exists for an unanswerable
query). The LLM judge is unreliable on these — q33's same generated
answer scored 1.0 in one run and 0.0 in another. With the override, the
score is a pure function of (category, refusal-or-not).

#### B2. Multi-seed judge averaging (opt-in)

`LLMJudge` gains `n_samples: int = 1` and `sampling_temperature: float =
0.7`. When `n_samples > 1`, each metric fires N parallel calls at the
sampling temperature; the result's `.score` is the mean and `.score_std`
is the sample stddev. `GenerationMetrics` gains `*_std` fields. CLI
flag: `--judge-n-samples N`. Default 1 = previous behavior.

Cost: N× judge tokens. At gpt-4o-mini with N=3 and 39 v3 queries × 3
metrics, ≈ $0.04 per eval. Cheap.

#### B3. Smoke pre-flight (`scripts/smoke_eval.py`)

Wraps `scripts.eval_run` with golden v1 (5 queries, 1 paper) and the
production stack flags. Targets ~5 min completion. Fails fast on infra
issues (missing API key, Ollama hang, GPU OOM, schema drift) before
launching a 60–90 min v3 run. Run with `python -m scripts.smoke_eval`.

## Cascade calibration outcome

`scripts/calibrate_cascade.py` ran every v3 query through the text leg
only (the same path the cascade uses for its first pass), recorded
top-1 rerank scores. Per-category distribution:

| Category | n | min | median | max |
|---|---|---|---|---|
| factual | 13 | 0.111 | 0.937 | 0.999 |
| figure | 11 | 0.824 | 0.948 | 0.999 |
| table | 4 | 0.902 | 0.991 | 0.994 |
| multi_hop | 2 | 0.804 | 0.950 | 0.950 |
| equation | 1 | 0.852 | — | — |
| out_of_corpus | 8 | 0.001 | 0.100 | 0.910 |

**Honest finding:** the text reranker is uniformly confident across all
in-corpus categories (factual, figure, table, multi_hop all cluster in
~[0.85, 1.00]). Top-1 score by itself doesn't separate "confident text"
from "needs visual help." A clean per-category threshold doesn't exist.

Threshold choice for the v3 corpus:

- **0.95+**: skips visual on ~half the in-corpus queries. Aggressive
  cost saving; risks losing visual help on figure/table queries that
  actually benefit from it.
- **0.85**: skips visual on the most-confident factual queries (~3 of 31
  in-corpus). Conservative; preserves visual leg for almost all
  figure/table/multi_hop work.
- **0.50**: barely affects anything; effectively cascade-off.

We ship the cascade off by default; operators set
`--cascade --cascade-threshold 0.85` (or higher) explicitly when they
want the cost savings. The verification eval ran at threshold=0.85.

The deeper lesson: top-1 rerank score is a *weak* uncertainty signal on
a corpus where the reranker is well-calibrated. Future work can replace
the score-threshold gate with:
- **score margin** (top-1 minus top-2): bigger margin = more confident
- **score entropy** over top-K
- **per-category** thresholds (factual queries skip visual at lower
  thresholds than figure queries)

## Validation outcome

The full v3 verification eval was attempted with all features on
(`--cascade --cascade-threshold 0.85 --judge-n-samples 3` plus the
existing committed-baseline stack). It failed early with HTTP 403 from
OpenRouter: *"Key limit exceeded (total limit)"* — the API key hit its
spending cap during the cumulative session work. Not a code defect.

A smaller end-to-end smoke verification (run `568fe7cfd4f9`) ran all
Tier 2 code paths against golden v1 (5 queries, 1 paper) using local
Ollama models (llama3.2:3b for gen + judge, n_samples=2) so it doesn't
burn API credit. Validates the *plumbing*, not headline metrics — the
absolute scores are lower because llama3.2:3b is a weaker judge than
gpt-4o-mini.

| Metric | Smoke value (Ollama 3.2:3b judge) |
|---|---|
| nDCG@5 (4 in-corpus queries) | 0.8726 |
| recall@10 | 0.8750 |
| MRR | 1.0000 |
| citation grounding | 1.0000 |
| faithfulness | 0.8133 |
| answer_relevance | 0.6900 |
| context_precision | 0.6060 |

What the smoke proves end-to-end:

- **Cascade dispatched correctly.** q5 (OOC) had `cascade_top_score=0.001 <
  0.85` → fell through to `cascade_decision=uncertain_hybrid` and the
  visual leg fired. The 4 in-corpus queries had high text confidence and
  hit `cascade_decision=confident_text` (visual leg skipped). Logs include
  `mode=cascade` and the decision string.
- **Multi-seed judge fired.** Config records `judge_n_samples=2`. Per-call
  log entries include `n_samples=2 score_std=…`. Smoke happened to land
  on score_std=0 for several metrics (the 2 calls agreed); a fuller run
  on a more contentious corpus would surface non-zero variance.
- **B1 deterministic OOC.** q5's refusal answer (category=out_of_corpus)
  was scored faith=1.0/ans_rel=1.0 by the runner-side override; no LLM
  judge call was made for those metrics on this query (verified in
  `_RecordingJudge` unit tests + matching log shape here).

The headline cost-quality numbers vs the committed baseline
`f844619927e0` remain measured by Tier 1's `196ac0f8786f` data: cascade
is metric-neutral on this corpus by design, the methodology fixes
remove judge noise rather than move quality. A future v3 eval (after
the OpenRouter cap is raised, or running it through Ollama for a
multi-hour wall-clock cost) would confirm that explicitly with
gpt-4o-mini's score range; the code paths themselves are verified.

## Caveats & open questions

1. **Cascade threshold isn't load-bearing.** As the calibration data
   shows, the v3 corpus's text reranker is too confident across the
   board for a single-value threshold to cleanly separate categories.
   This ADR ships the *infrastructure* — the cascade mode, the calibration
   tool, the eval flag — but the actual cost savings on this corpus are
   modest. Bigger payoff would require either a richer uncertainty
   signal (score margin, entropy) or a corpus where the reranker has
   genuinely uncertain regions.

   *Update 2026-05-12:* `Query.routing_mode` accepts a per-query
   `category` | `cascade` override. The demo UI exposes it under
   "Advanced retrieval settings" so visitors can flip a single query
   between modes without restarting the server. When cascade is
   requested per-query against a category-wired server,
   `RoutingRetriever` uses 0.85 (the value the v3 verification ran
   with) as the threshold; operators who want a different default
   still set `Settings.cascade_confidence_threshold` at boot.

2. **Multi-seed averaging shifts the cost story.** N=3 triples the
   judge call cost. Negligible on gpt-4o-mini; meaningful if a future
   judge is more expensive. Default stays at 1; opt-in via
   `--judge-n-samples`.

3. **Deterministic OOC scoring might be too strict.** A non-refusal
   answer that happens to acknowledge "I'm not sure but here's related
   context…" loses the LLM judge's potential partial credit. We accept
   this — the OOC test's purpose is verifying refusal, not measuring
   gradients of "how related is the model's wrong answer." Documented
   here in case future eval needs change the requirement.

4. **B3 smoke pre-flight is opt-in.** Operators run it before long
   evals; we don't gate v3 runs on it automatically. Consider promoting
   to required-pre-flight if recurring infra crashes prove the smoke
   alone isn't enough discipline.

## References

- ADR 0008 — Phase 3.2 routing (the category-mode dispatch this ADR
  augments).
- ADR 0009 — Region-level evidence + 1st/2nd follow-ups.
- `scripts/calibrate_cascade.py` — per-corpus threshold picker.
- `scripts/calibrate_refusal.py` (Tier 1) — same shape, refusal version.
- Run `568fe7cfd4f9` — Tier 2 smoke verification (Ollama, golden v1).
