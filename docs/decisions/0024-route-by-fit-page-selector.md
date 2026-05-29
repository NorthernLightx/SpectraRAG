# ADR 0024 — Route-by-fit page selector: feed the whole document when it fits context, else top-k RAG

**Status:** Accepted for the eval generation path; shipped as an opt-in policy
in `scripts/experiments/run_mmlb_qa.py` (`route_pages_by_fit`, behind
`--page-budget`, default off — the top-k path is unchanged). Production `/answer`
wiring is explicitly out of scope (see §"What this leaves open"). Not a baseline
change.
**Date:** 2026-05-29.

## Context

The MMLongBench-Doc generation harness feeds the VLM a fixed top-5 of the
retrieved page-images (`run_mmlb_qa.select_pages`, top-K unique pages from the
fused ranking). The 2026-05-29 capstone measured that on documents that fit the
model's context this is an over-tight cut: retrieval-loss dominates the
distraction cost for small docs, so feeding more pages helps.

This sits on the RAG↔long-context spectrum the 2025-26 literature treats as
"no one size fits all" — long-context wins single-doc leaderboards but suffers
lost-in-the-middle, distraction, and per-token cost, while RAG is required once
content exceeds the window. The current frontier framing is route-before-retrieve
(arXiv 2509.21865; OpenReview "Route Before Retrieve"). The question this ADR
settles for this corpus: where does the crossover fall, and is it worth a routing
rule.

## Finding (measured, same model / same extractor)

All arms: generation `gemma-4-31b:free`, extraction `gemma3:4b`, depth-50 w1
retrieval. Full record: `docs/research/2026-05-29-agenda/RESULTS.md`, "Bet 1
capstone — RAG vs whole-document".

Whole-doc vs top-5 RAG on documents that fit (≤50 pages, n=66):

| metric | top-5 RAG | whole-doc | Δ |
|---|---|---|---|
| answerable | 0.326 | 0.441 | +0.116 |
| figure (n=48) | 0.224 | 0.346 | +0.12 |
| table (n=11) | 0.523 | 0.705 | +0.18 |

The page-count sweep is monotone — k1 0.161 / k3 0.269 / k5 0.359 / oracle 0.418
— and conversion *rises* with more pages, so distractor dilution is negligible.
Top-5 was an over-aggressive cut where the doc fits.

Simulated route-by-fit (whole-doc when page-count fits a budget, else top-k):
**0.417 vs all-RAG 0.349, +0.068 answerable**, routing 59% of queries to
whole-doc. Beyond the budget the policy keeps RAG, which is required: on the
>50-page docs whole-doc *lost* (0.192 vs 0.222), and the 166-page doc exceeds the
window entirely. The crossover is real and measured.

## Decision

1. **Add `route_pages_by_fit(...)` to the eval generation harness** beside
   `select_pages`. Given the document's page count and a context budget, return
   all the document's pages when it fits the budget, else fall back to
   `select_pages(fused, top_k)`. The fit test is a closed interval
   (`page_count <= budget` routes to whole-doc).

2. **Opt-in, default off.** Gated behind `--page-budget` (default `None`). Unset
   reproduces the top-5 path byte-for-byte, so the policy changes no existing
   run. This makes the +0.068 reproducible without disturbing the committed
   top-k results — the same conservative-default posture as ADR 0023.

3. **The lever is how much you feed, not a bigger model or a generation trick.**
   The 2026-05-29 follow-up refuted feed-fewer-pages, a stronger extractor
   (tooling-blocked), an anti-refusal prompt (null), and a frontier VLM (flat
   overall; its figure edge did not survive real retrieval). Page count is the
   one generation-side lever that moved end-to-end accuracy on this corpus.

## What this leaves open

- **Production `/answer` is deliberately not wired.** `Generator.answer` consumes
  only `list[RetrievalResult]`, and `RetrievalResult` carries no document
  page-count, so the policy's input does not exist on the production path. Worse,
  the measured win is single-doc QA where "the document" is given by the gold
  label; a corpus-wide endpoint must first *identify* the document, which is
  unsolved and unmeasured here. Wiring prod would mean new ingestion metadata, a
  doc→page-count lookup, retriever→generator plumbing, and doc-identification
  logic — speculative surface for a portfolio demo. Production keeps its
  `_MAX_VISION_IMAGES = 4` cap. Revisit only if a corpus-setting measurement
  justifies it.

- **The +0.116 is on a feasibility-filtered subset.** Whole-doc *failed on
  45/149 queries (30%)* — large page-image payloads choked the free tier — so the
  +0.116 is "whole-doc wins where it ran," not everywhere. The +0.068 routing
  number already folds the >50-page RAG fallback back in, but both rest on the
  free-tier feasibility envelope.

- **Page-count source is the load-bearing input.** Retrieval surfaces only
  top-50, so the document's *true* page count must come from outside the
  retrieved set (e.g. the rendered-pages directory). A wrong source silently
  under-feeds the model and erases the win while tests still pass; the helper
  takes the count as an integer and the caller is responsible for resolving it
  loudly.

- **Single corpus, single model pairing.** Measured on MMLongBench with one
  generation/extraction pair. The crossover budget is corpus- and
  context-window-specific; the rule generalizes, the threshold does not transfer
  unmeasured.

## Related

- ADR 0013: routing is the accuracy lever; the visual leg is the lever. This ADR
  routes a different axis — how many pages to feed — on the same corpus.
- ADR 0023: per-corpus visual-fusion weight, shipped opt-in with an unchanged
  default; this ADR follows the same conservative-default discipline.
- ADR 0010: cost-quality cascade — the existing "spend more only when it pays"
  framing that route-by-fit extends to the generation context budget.
