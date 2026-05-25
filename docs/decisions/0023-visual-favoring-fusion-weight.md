# ADR 0023 — Visual-favoring weight for page-level RRF (config knob, default 1.0)

**Status:** Shipped as a config knob. `Settings.visual_fusion_weight`,
**default 1.0**, which reproduces ADR 0008's equal-weight fusion byte-for-byte
(no baseline change). The visual bias is *not* enabled by default. **Cross-corpus eval done
(2026-05-25): w>1 regresses the text-heavy v3 corpus, so the default 1.0 is
confirmed and the knob is per-corpus — see "Cross-corpus validation" below.**
**Date:** 2026-05-25.

## Context

The hybrid path (figure/table/multi_hop queries) fuses its text and visual legs
with Reciprocal Rank Fusion at page granularity in
`RoutingRetriever._fuse_page_level` (ADR 0008 §"Decision" §5, RRF k=60). Both
legs were weighted equally. ADR 0008 §6 listed "per-category RRF weights" as an
open follow-up; this is the first measurement of that lever.

ADR 0013 established that on MMLongBench the visual leg *is* the accuracy lever
(~93 % of answerable queries are figure/table) and that the routing headroom is
a *policy* knob, not a model-size knob — but it also caught the trap: a
degenerate "always-visual" prompt (`classify_query_v2`) scored highest on
MMLongBench precisely because the benchmark rewards ignoring the text leg, and
that strategy would regress the text-heavy arXiv corpus where the visual leg
*loses* ~10 % on definitional/text queries (ADR 0007 / `results.md`). So any
visual bias has to clear a higher bar than "it wins on MMLongBench."

## Finding (measured, CPU re-fusion, no new inference)

Re-fusing the existing depth-50 per-leg results
(`data/eval/runs/depth50-20260525-015216/depth50.json`, `text_top50` /
`visual_top50`) through the weighted RRF — scoring with the canonical page-level
helpers (`scripts.rescore_mmlb_pages`) — sweeps as follows on the figure subset
(n=75) and table subset (n=23):

| w_visual | figure recall@10 | figure nDCG@5 | figure MRR | table nDCG@5 |
|---|---|---|---|---|
| **1.0 (= shipped)** | **0.7293** | 0.5571 | 0.5484 | 0.6838 |
| 2.0 | 0.7338 | 0.5838 | 0.5806 | 0.7307 |
| 3.0 | 0.7538 | 0.5916 | 0.5850 | 0.7429 |
| 4.0 | 0.7938 | 0.6081 | 0.6033 | 0.7334 |
| **5.0** | **0.8071** | **0.6364** | **0.6258** | 0.7334 |
| 6.0 | 0.8071 | 0.6354 | 0.6251 | 0.7315 |

The w=1.0 row reproduces the shipped fused ranking *exactly* — figure
recall@10 0.7293, and per-query the top-10 page sequence is identical to
`fused_top50` on all 149 queries (zero mismatches). That is the proof the
implementation matches the equal-weight RRF it replaces.

At w=5: figure recall@10 **0.729 → 0.807**, nDCG@5 **0.557 → 0.636**, MRR
**0.548 → 0.626**; table nDCG@5 **+0.05** (recall@10 unchanged — tables are
already near-ceiling at 0.826). The curve is monotone up to a w=5–6 plateau.

**This is a fusion gain, not a leg swap.** The w=5 blend (0.807) beats *both*
equal-weight (0.729) *and* visual-only (0.774, from the same legs). If the win
were "the visual leg is just better here," visual-only would be the ceiling and
the blend could not exceed it. It does. Keeping a down-weighted text leg in the
fusion still helps, which is what separates this from the rejected
always-visual v2 prompt in ADR 0013.

### Cross-validation

- **Split-half.** Held-out halves of the figure set improve 0.725→0.802 and
  0.733→0.786, both at a consistent optimal w=4–5 — the optimum is not fit to
  the full set.
- **Broad-based, not one-query.** 8/75 figure queries improve, 1 regresses; the
  lift is distributed, not a single outlier dragging the macro mean.

## Mechanism

RRF score per page = `1/(k+rank_text) + w/(k+rank_visual)`. Raising `w` lets a
page that the visual leg ranks highly but the text leg ranks low (or misses)
climb past a text-favored page. On a corpus where the answer usually lives in a
rendered figure/table, the visual leg's ordering is the more reliable signal for
these query categories, so up-weighting it surfaces the right page within the
top-10 more often — while the still-present text leg breaks ties and rescues the
minority of figure queries whose answer text is indexed. The right frame
(per ADR 0013): favour the modality suited to the routed query, without
discarding the other leg.

## Decision

1. **Ship a single visual-leg weight as a config knob.**
   `Settings.visual_fusion_weight: float = 1.0`, threaded through
   `RoutingRetriever(visual_fusion_weight=...)` into `_fuse_page_level`. The
   text leg keeps the implicit weight 1.0; only the visual leg is scaled. One
   knob matches the finding (only the visual leg was swept) and ADR 0008 §6's
   "per-category RRF weights" framing.

2. **Default = 1.0, on purpose.** 1.0 is the equal-weight fusion ADR 0008
   shipped, verified byte-identical above — so wiring this knob changes *no*
   served result until an operator opts in. MMLongBench is ~93 % visual and
   *wants* w≈5, but ADR 0013 is explicit that a text-heavy corpus (the
   production arXiv corpus) may regress under a visual bias. Flipping the
   default to a visual-favoring value without a text-corpus eval would repeat
   the always-visual overfit ADR 0013 caught. The conservative default keeps
   the win available behind a flag without betting the production corpus on a
   single-benchmark result.

3. **Scope: hybrid-routed queries only.** The weight lives in
   `_fuse_page_level`, which runs solely on the hybrid path (figure / table /
   multi_hop, and the cascade fall-back). Text-routed queries never fuse, so
   they are byte-for-byte unaffected at any weight.

## What this leaves open

- **Cross-corpus validation is the gate before changing the default.** Measure
  the same weight sweep on the text-heavy arXiv corpus (golden v3) where ADR
  0007 documents the visual leg losing on definitional queries. If a moderate
  weight (~3) holds figure/table gains there *without* regressing text-routed
  categories — text-routed queries are unaffected by construction, so the risk
  is mis-routed factual/definitional queries that landed on the hybrid path —
  then promoting the default to ~3 is the recommended next step. Until then the
  default stays 1.0.
- **multi_hop is untested here.** This depth-50 run's classifier labelled 0
  queries `multi_hop` (figure/table only), so the sweep says nothing about
  multi_hop fusion. The knob applies to it by code path; its effect there is
  unmeasured.
- **Per-category weights remain unbuilt.** A figure query and a (near-ceiling)
  table query may want different weights; this ships one global visual weight.
  Per-category weights are the natural follow-up if cross-corpus validation
  justifies the extra surface (ADR 0008 §6).
- **Re-fusion only.** This measured re-ordering of already-retrieved depth-50
  legs, not a fresh end-to-end run. Promotion to a new baseline goes through the
  eval-engineer / tech-lead with the per-query Markdown — this ADR does not
  rebaseline.

## Cross-corpus validation — 2026-05-25 (resolves "what this leaves open" §1)

Ran the same weighted-RRF sweep on the text-heavy v3 arXiv corpus
(`eval_phase32_router` — the 20-paper v3 corpus, full payload; golden v3's 15
figure/table queries), two-pass depth-50, fused through the real
`_fuse_page_level`. Validation driver:
`scripts/experiments/validate_v3_visual_fusion_weight.py`.

| w_visual | v3 fig/table nDCG@5 | MRR |
|---|---|---|
| **1.0** | **0.904** | **0.872** |
| 2.0 | 0.809 | 0.747 |
| 3.0 | 0.809 | 0.747 |
| 5.0 | 0.755 | 0.739 |

**w>1 regresses v3 — monotone down, 6/15 queries worse, 0 better.** recall@10 is
saturated at 1.0 (v3 runs with `paper_id_filter`, scoping retrieval to one paper
so the gold page is always in the text leg's top-10), so the call rests on
nDCG@5/MRR. The w=1.0 row reproduced equal-weight RRF exactly (0/15 page-order
mismatches).

**Mechanism — the mirror image of MMLongBench.** On v3 the *text* leg is the
better ranker: figure/table captions are clean indexed text, so the bge-reranker
puts the gold page at rank 1 for 10/15 queries. Up-weighting the visual leg
displaces those correct rank-1 pages with the visual leg's (often wrong) picks.
On MMLongBench the visual leg is the better ranker, so the same knob helps. **The
lever is real, but its sign depends on which modality carries the answer in a
given corpus** — exactly the per-corpus caveat ADR 0013 predicted.

**Conclusion: default stays 1.0 (now evidence-backed, not just precautionary).**
The knob is per-corpus: visual-heavy / scanned / figure-only corpora set w≈5;
text-heavy arXiv-style corpora keep 1.0. Do **not** flip the global default.
Confound: n=15 (small, but the direction is unambiguous — monotone, 6 worse /
0 better, mechanism verified per-query). Per-category weights (§"what this leaves
open") remain the natural follow-up but are not justified by this evidence alone.

## Related

- ADR 0013: routing is the accuracy lever; the visual leg is the lever, the
  always-visual trap, and the explicit warning that a text-heavy corpus may not
  want a visual bias. This ADR ships the conservative knob that lever implies.
- ADR 0008: page-level RRF (k=60) this weights; §6 listed per-category RRF
  weights as the open follow-up.
- ADR 0007: visual leg loses ~10 % on definitional/text queries — the reason the
  default is not flipped without a text-corpus eval.
- ADR 0010: cascade — the other hybrid entry point `_fuse_page_level` serves, so
  the weight applies to cascade fall-back fusion too.
