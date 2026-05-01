# ADR 0004 — Phase 3 visual retrieval (ColQwen2 / ColPali-style)

**Status:** Accepted as a complementary path. Visual and text retrievers have
fundamentally different strengths on this corpus; neither dominates.
Production recommendation is text-path baseline; visual ships in-tree as
`scripts/eval_visual.py` for ablation work and as the foundation for a
future hybrid text+visual fusion (deferred).
**Date:** 2026-05-01.
**Phase:** 3.

## Context

`PROJECT.md §5` Phase 3 calls for "ColQwen2 visual path … pipeline vs visual
comparison — the headline result." After Phases 1, 2.0, 2.1, 2.2 the text
path baseline is `7b5242df5b38` (`data/eval/baseline.json`):

- nDCG@5 0.7214, recall@10 0.9412, MRR 0.7437
- p50 query latency ~73 s (whole-pipeline including generate + judge)
- p50 retrieve-only ~5 s on GPU rerank

The text path's failure modes:
- multi-hop / multi-source queries (q4 0.624, q12 0.613)
- term-mismatch queries (q11 0.000, q20 0.000) where the query vocabulary
  doesn't overlap the paper's
- factual lookups in dense, non-headed sections that BM25+rerank can't pin

ColPali-style retrieval embeds *whole page images* with a vision-language
model and uses late interaction (MaxSim) to match against the query. The
hypothesis is that this should help wherever the text pipeline's chunking
or term-overlap heuristics break down.

## Implementation

- `colpali-engine 0.3.10` (forced — newer 0.3.15 needs transformers 5.x +
  a peft version that doesn't have `_maybe_shard_state_dict_for_tp`,
  unresolvable dependency knot). Pinned `torch 2.6.0+cu124` and
  `torchvision 0.21.0+cu124` to match.
- `src/ingestion/visual.py` — `render_pages()` rasterises each PDF page
  to PNG at configurable DPI (150 default), idempotent on disk.
- `src/rag/retrievers/visual.py` — `VisualRetriever` (in-memory
  multi-vector store of `[n_patches, dim]` bf16 tensors keyed by chunk-id
  `<paper_id>::p<N>::page`) + `build_visual_retriever()` async factory
  that loads ColQwen2 once and embeds every page. Duck-types `Retriever`
  protocol so it slots into `eval.runner.evaluate` cleanly.
- `scripts/eval_visual.py` — sibling to `eval_run.py`. ColQwen2 needs the
  full GPU (~5 GB), can't co-exist with bge-m3+reranker+Ollama, so the
  visual path is its own command. Falls back to deriving relevant pages
  from `relevant_chunk_ids` when the golden set's `relevant_pages` is empty.
- 3 new unit tests (page rendering smoke + idempotence). 213 total pass.

## Headline result — visual vs text on golden v2 (5 papers, 23 queries)

Visual run: `ab16789c4f3b` (`data/eval/runs/run-visual-20260501-215441.json`).

| Metric (in-corpus, n=17) | Text (baseline) | Visual | Δ |
|---|---|---|---|
| nDCG@5 | **0.7214** | 0.6327 | −12.3% |
| **recall@10** | 0.9412 | **1.0000** | **+6.2%** |
| MRR | **0.7437** | 0.6300 | −15.3% |
| **p50 retrieve latency** | ~5 s | **~300 ms** | **17× faster** |
| p95 retrieve latency | ~6 s | ~360 ms | 17× faster |

**Visual retrieves the right page for *every* in-corpus query** — recall@10
is a perfect 1.0 (vs text's 0.94). It just doesn't always rank the right
page first or in the top-5, dragging nDCG and MRR.

## Per-query analysis — the strengths are *different*, not "better/worse"

| query | category | text ndcg5 | visual ndcg5 | Δ |
|---|---|---|---|---|
| q1_inter_basin | factual | 1.000 | 0.631 | −0.369 LOSS |
| q3_approximation_options | factual | 0.877 | 1.000 | +0.123 WIN |
| **q4_target_region** | multi_hop | 0.624 | **1.000** | **+0.376 WIN** |
| q6_basin_definition | factual | 1.000 | 0.000 | −1.000 LOSS |
| q7_posterior_mixture | factual | 1.000 | 0.431 | −0.569 LOSS |
| q8_benchmark_size | factual | 1.000 | 0.431 | −0.569 LOSS |
| **q9_baselines** | factual | 0.387 | **1.000** | **+0.613 WIN** |
| **q10_ablation_terms** | factual | 0.631 | **1.000** | **+0.369 WIN** |
| **q12_multibasin_vs_vopt** | multi_hop | 0.613 | **0.877** | **+0.264 WIN** |
| q13_figure1_lrbsz | figure | 1.000 | 0.387 | −0.613 LOSS |
| q16_aw_pinn_what | factual | 0.631 | 0.000 | −0.631 LOSS |
| **q20_exploration_hacking** | factual | 0.000 | **1.000** | **+1.000 WIN** |
| q22_mase_definition | factual | 0.500 | 0.000 | −0.500 LOSS |

**Visual wins** (5 strong wins): q4, q9, q10, q12 (multi-hop and
multi-source), and dramatically q20 (exploration_hacking, where text path
totally failed because the answer chunk was buried under repeated mentions).

**Visual losses** (6 strong losses): q1, q6, q7, q8, q16, q22 — all
single-fact definitional lookups where text rerank pinpoints the exact
chunk. ColPali sees the whole page and the relevant fact is one sentence
in a sea of others.

**Surprise loss** on q13 (figure-targeted) — the visual retriever picked a
*different* page that visually resembles Figure 1's heatmap structure
rather than the actual page-2 figure. Cross-paper visual similarity
fooled it on a corpus where every paper has at least one heatmap.

## Decision

**Accept visual retrieval as a complementary path, not a replacement.**

- Default eval pipeline stays text-only (`scripts/eval_run.py`).
  `data/eval/baseline.json` unchanged.
- Visual retrieval ships as `scripts/eval_visual.py` for ablation /
  research use.
- The natural next step is **hybrid text + visual** with RRF fusion of
  both retrievers' top-K — text for definitional precision, visual for
  multi-hop / term-mismatch coverage. The `VisualRetriever` already
  duck-types the protocol so a `MultiSourceRetriever` decorator
  (analogous to `MultiQueryRetriever`) is small. Deferred to Phase 3.1
  if and when we want to chase that combined number.
- Latency (~300 ms p50 retrieve) is genuinely production-worthy — visual
  is **17× faster than text at the retrieve stage** on this corpus
  because ColQwen2 query embedding + MaxSim is GPU-bound and direct,
  whereas the text path is dense + sparse + RRF + cross-encoder rerank.

## Caveats & open questions

1. **Granularity mismatch.** Text retrieves chunks; visual retrieves
   pages. We translated `relevant_chunk_ids` to `relevant_pages` for the
   visual eval (every chunk's page becomes a relevant page), so visual
   recall@10 is mechanically easier to saturate. A "fair" comparison
   would either chunk pages into sub-image regions for visual, or score
   text at page-granularity. Not done; the current numbers are a
   reasonable headline read but should not be over-interpreted.
2. **Cross-paper visual similarity** can mislead — q13 was hit by this.
   With a larger corpus this would amplify. Hybrid would mitigate via
   the text-path semantic signal.
3. **Generation + judge not run on visual path.** A text generator on
   page-image chunks would need a vision-language model in the
   generation step (e.g., the Phase 2.1 minicpm-v:8b captioner repurposed
   as generator). Out of scope for this ADR.
4. **`vidore/colqwen2-v1.0`** weights are 7 GB on disk, ~5 GB GPU at bf16.
   Coexists fine with nothing else loaded; can't run alongside the text
   path on 8 GB VRAM. The two paths are run sequentially.

## References

- `PROJECT.md §5` Phase 3 deliverable.
- `src/ingestion/visual.py`, `src/rag/retrievers/visual.py`,
  `scripts/eval_visual.py`, `tests/unit/test_visual_render.py`.
- ADR 0001 (contextual retrieval — Rejected), ADR 0002 (multi-modal
  chunks — Accepted opt-in), ADR 0003 (query expansion — Rejected,
  per-query wins) for the pattern of "real per-query wins, mixed
  aggregate."
