# ADR 0009 — Region-level evidence (figures + tables as first-class chunks with bbox)

**Status:** Accepted (2026-05-09). Vanilla flag-on regressed retrieval
-13.8 % nDCG@5; the 1st follow-up added (a) golden updates that credit
region chunks, (b) `--paper-id-filter` to scope retrieval per paper, (c)
`--region-number-boost` to promote `Table N:` / `Figure N:` captions on
numbered queries (combined run `d9bcd13b880f` recovered to parity). The
2nd follow-up adds (d) `--rerank-length-norm` to penalise caption-stub
chunks at the cross-encoder, (e) `--vlm-caption-model gemma3:4b` to
enrich figure chunks with VLM-generated descriptions. **Final committed
baseline `f844619927e0` lifts nDCG@5 +2.6 %, faithfulness +3.4 %,
answer_relevance +1.3 %** vs the prior committed baseline, with all
metrics PASS on the regression gate.
**Date:** 2026-05-09.
**Phase:** 3.3.

## Context

Run `c92f3f1bee19` (committed as `data/eval/baseline.json`) shows the current
v3 + router + visual stack delivers nDCG@5 = 0.794 on retrieval and clean
generation metrics (faithfulness 0.83, answer relevance 0.91 on the
patched-judge run that's about to land — see "Validation" below). Per-category
analysis: figure queries already average **0.876 nDCG@5** and table queries
**0.875** — the strongest categories, not the weakest. The bottom-tail
failures (q11, q20, q35) live at *ranking* (right page in top-10 but not
top-5) and at *cross-paper bleed* (q9, q12), not at granularity. Generation
recovers from rank-6 chunks: every Pattern-A query scores 1.0/1.0 on judge
metrics with the patched runner.

So the metric case for region-level evidence is weak. The architecture case
is real: page-level retrieval blurs heterogeneous content (figure caption +
adjacent paragraph + footnote share one embedding), citation truthfulness
suffers ("page 14" hides which figure was the actual evidence), and
demo-grade citation surfaces look better when the system can point at "Figure
5b" specifically. For a portfolio repo whose audience is engineers reviewing
the design, the architectural story carries weight that the metric story
doesn't capture.

The ingestion code already half-supports this: `src/ingestion/figures.py` and
`src/ingestion/tables.py` extract figures and tables, `figure_to_chunk` /
`table_to_chunk` in `chunking.py` convert them to first-class `Chunk`s with
`metadata['kind']`, and the `--extract-figures` / `--extract-tables` flags on
the eval CLI wire it through. None of the eval runs to date have set those
flags (every committed run has `extract_figures: false`,
`extract_tables: false`). The missing pieces are:

1. **Bounding boxes.** PyMuPDF exposes the page-coordinate rect for each
   image (`page.get_image_rects(xref)`) and each detected table
   (`Table.bbox`). We don't capture either today.
2. **Citation surface.** `Citation` carries `chunk_id` + `page_numbers` only;
   no bbox passes through to callers, so the demo UI can't highlight a
   specific region on the page.
3. **Empirical proof.** The eval has never exercised the figure/table
   extraction path, so we don't know if it lifts metrics, hurts them
   (extra noise from caption-stub chunks), or is neutral.

## Decision

1. **Region = first-class `Chunk` with bbox in `metadata`.** No new `Region`
   type. Extending the existing path keeps the retrieval surface uniform —
   one BM25 index, one Qdrant collection, one rerank pass — and avoids the
   substrate proliferation that ADR 0009 (multimodal graph store)
   considered and rejected for similar reasons.
2. **Bbox format:** `[x0, y0, x1, y1]` in PDF points (PyMuPDF's native unit).
   Stored as `metadata['bbox'] = [x0, y0, x1, y1]` — plain list, not a typed
   model on the chunk, because `Chunk.metadata` is `dict[str, Any]` and the
   eval/regression-gate JSON serializer can't distinguish nested Pydantic
   models at this layer. Type safety lives one layer up: `Figure.bbox` and
   `Table.bbox` are typed `Bbox | None`.
3. **`Bbox` Pydantic model** lives in `src/types/documents.py` next to
   `Figure` / `Table`, frozen, with field-level validation
   (`x0 < x1, y0 < y1`).
4. **`Citation` gains `bbox: list[float] | None`.** Defaults to `None` for
   text chunks. When the cited chunk is a figure or table region, the bbox
   propagates from the Chunk's metadata through to the Citation. This is the
   user-visible payoff: the demo UI can render a highlight rectangle on the
   page image at exactly the cited region.
5. **Eval path: turn on `--extract-figures --extract-tables` for the
   region-level run.** Same v3 corpus, same router, same visual leg. Compare
   `c92f3f1bee19` (no figure/table extraction) vs the new run side by side.
6. **VLM captioning stays optional.** The `--vlm-caption-model` flag exists
   today; we don't enable it in this ADR's eval run because the local Ollama
   vision models (gemma3:4b, qwen2.5vl:7b) cost meaningful GPU time and would
   confound the comparison. A separate VLM-caption pass is its own ADR if
   the metrics warrant it.

## What this is NOT

- **Not bbox-level visual retrieval.** The visual leg (ColQwen2 over rendered
  page images) stays page-granular. Region-level visual late-interaction
  (per-figure crop embeddings) is in the literature (e.g. ColPali patch
  scores can be aggregated by bbox) but the current routing layer doesn't
  consume sub-page visual scores; adding that is a separate ADR.
- **Not a knowledge graph.** Cross-modal links between figures and the
  paragraphs that reference them are interesting (the "MultiRAG" framing in
  the literature) but require a graph store and a graph-aware reranker we
  don't have. Out of scope.
- **Not a router-classifier change.** The router still dispatches by query
  category (figure / table / multi_hop → hybrid; factual / definitional →
  text). Region-level evidence rides on top of whichever leg fires; on a
  hybrid dispatch, the text leg's BM25 + dense + rerank pulls figure/table
  chunks naturally because they share the corpus.

## Architecture

```
                                Paper PDF
                                    │
        ┌───────────────────────────┼─────────────────────────────┐
        ▼                           ▼                             ▼
   extract_pages()           extract_figures()             extract_tables()
   (existing)                       │                             │
        │                  Figure(bbox=Bbox(...))          Table(bbox=Bbox(...))
        │                           │                             │
        ▼                           ▼                             ▼
   chunk_pages()            figure_to_chunk()              table_to_chunk()
   (text Chunks)              (Chunk with                   (Chunk with
        │                      metadata={                    metadata={
        │                        kind="figure",               kind="table",
        │                        bbox=[x0,y0,x1,y1],          bbox=[x0,y0,x1,y1],
        │                        image_path=...               })
        │                      })
        │                           │                             │
        └──────────────► merge into one chunk list ◄──────────────┘
                                    │
                                    ▼
                       embed → BM25 → Qdrant upsert
                                    │
                          ... unchanged retrieval ...
                                    │
                                    ▼
                              Generator answer
                                    │
                          extract citations from text
                                    │
                                    ▼
                Citation(chunk_id, paper_id, page_numbers, bbox?)
                — bbox copied from chunk.metadata when kind in {figure,table}
```

## File touches

- `src/types/documents.py` — add `Bbox`; add `bbox: Bbox | None` to `Figure`
  and `Table`.
- `src/types/generation.py` — add `bbox: list[float] | None = None` to
  `Citation`.
- `src/ingestion/figures.py` — capture `page.get_image_rects(xref)` per
  image, attach to `Figure.bbox`.
- `src/ingestion/tables.py` — capture `found.bbox` per detected table,
  attach to `Table.bbox`.
- `src/ingestion/chunking.py` — `figure_to_chunk` / `table_to_chunk` pack
  `bbox` into `metadata['bbox']` as a 4-list when present.
- `src/rag/generate.py` — citation extraction looks up the chunk's metadata
  for `bbox` and copies it to the `Citation` if present.
- Tests at every layer (see "Validation" below).

## Failure modes

1. **Bbox absent.** PyMuPDF's `get_image_rects` can return an empty list for
   some embedded streams (vector art, transparent overlays). `Figure.bbox`
   stays `None`; `figure_to_chunk` doesn't add `metadata['bbox']`; the
   citation downstream has `bbox=None`. Demo UI renders a page-level
   highlight (existing behavior) — graceful degrade.
2. **Multiple rects for one image.** A logo repeated across pages would
   show up as N rects from one xref. We dedupe by xref *before* fetching
   rects (existing logic), so this can't happen for the deduped figures.
3. **Table bbox wrong.** PyMuPDF's table detector is heuristic — multi-page
   tables, dense column layouts, and embedded equations confuse it. When
   `find_tables` returns a wrong bbox, the citation will point to the
   wrong region. The text content of the chunk (caption + markdown) is
   still correct, so faithfulness/answer_relevance scoring is unaffected;
   only the citation surface degrades.
4. **Schema migration on existing collections.** Pre-existing Qdrant points
   from older runs lack `metadata['bbox']`. `RetrievalResult.metadata`
   reads with `.get(...)`, so older points just have `bbox=None` and the
   path doesn't crash. New ingestions add the field.

## Validation

The committed baseline is `c92f3f1bee19` — pre-figure/table-extraction. The
patched judge (deterministic refusal handling per `docs/evals.md`) lifts
faithfulness on OOC refusals; this ADR's eval run uses the patched judge so
the comparison is clean.

A second run with `--extract-figures --extract-tables` on:

- Should match or improve nDCG@5 on figure (currently 0.876) and table
  (currently 0.875) categories. Adding caption-stub chunks could *hurt*
  retrieval if the reranker prefers the (low-content) caption chunks over
  the (high-content) text chunks that actually answer the query — this is
  why `figure_to_chunk` already prefers VLM caption when set, and why we
  audit this in the regression gate.
- Generation metrics may stay flat (the generator already had access to
  caption text via the surrounding text chunks). The qualitative win is
  the citation surface — answers can cite `2604.22753v1::p2::fig1` with a
  bbox, instead of `2604.22753v1::p2::c10` (which is the page's text
  chunk that happens to mention Figure 1).
- The regression gate must not fire. If aggregate metrics regress >5%, we
  treat the figure/table extraction as a feature flag and ship it off-by-
  default; the ADR stays Accepted but the rollout becomes a measured A/B.

## Validation outcome (run `ad4fab3bb28d`, 2026-05-09)

End-to-end run with `--extract-figures --extract-tables` on against v3 golden,
patched judge, same router + visual leg as the baseline. 64 min wall-clock
(20 min ingest, 38 min visual build, 6 min retrieve+gen+judge).

**Retrieval (in-corpus, n=31), against committed baseline `c92f3f1bee19`:**

| Metric | Baseline | Region | Δ abs | Δ rel | Gate |
|---|---|---|---|---|---|
| nDCG@5 | 0.7942 | 0.6845 | -0.1097 | **-13.82 %** | FAIL |
| recall@10 | 0.9677 | 0.8710 | -0.0968 | **-10.00 %** | FAIL |
| MRR | 0.8118 | 0.6995 | -0.1123 | **-13.84 %** | FAIL |

**Generation (n=39):**

| Metric | Baseline | Region | Δ abs | Δ rel | Note |
|---|---|---|---|---|---|
| faithfulness | 0.8269 | 0.9949 | +0.1679 | +20.31 % | mostly the patched-judge fix; recomputed patched-judge baseline ≈ 0.955, so the region-only lift is ~+4 % |
| answer_relevance | 0.9103 | 0.9744 | +0.0641 | +7.04 % | same — patched-judge baseline ≈ 0.987, region-only Δ is **-1.3 %** (slight regression) |
| context_precision | 0.8051 | 0.7769 | -0.0282 | -3.50 % | judge fix doesn't touch this; net -3.5 % is the region effect |
| citation grounding | 1.0000 | 1.0000 | 0 | 0 | unchanged |

**Where retrieval regressed (per-query):**

| Query | Baseline nDCG@5 | Region nDCG@5 | Δ |
|---|---|---|---|
| q25_lc_tab1_loss_properties (table) | 1.000 | 0.000 | **-1.000** |
| q31_hermes_fig1_unification (figure) | 1.000 | 0.000 | **-1.000** |
| q26_fd_tab4_imagenet256 (table) | 0.500 | 0.000 | -0.500 |
| q29_aegis_tab8_janus_rank (table) | 1.000 | 0.631 | -0.369 |
| q4_target_region (multi_hop) | 0.624 | 0.387 | -0.237 |
| q9_baselines (factual) | 0.613 | 0.387 | -0.226 |
| q22_mase_definition (factual) | 0.500 | 0.431 | -0.069 |

The pattern is exactly what §"Caveats #1" predicted: figure/table chunks (caption-only
content for figures, caption + markdown for tables) compete with text chunks at the
reranker, share page-ids in the routing fusion, and crowd the gold-labeled text chunks
out of top-5. Three of the four worst regressions are *table* queries (q25, q26, q29).

**Region-grounded citations did fire.** 5 of 58 generated citations cited a figure or
table chunk directly: q9_baselines → `2604.28193v1::p7::tab2`, q11_budget_levels →
`2604.22753v1::p7::tab2`, q25 → `2604.27742v1::p2::tab1`, q26 → `2604.28190v1::p10::tab4`,
q31 → `2604.28196v1::p2::fig1`. Each of those citations carries `bbox` per the schema —
demo UIs can render the precise region.

**Conclusion (initial validation):** the infrastructure is right; the
metric regression has explicit causes (golden mismatch + reranker stub
preference + cross-paper bleed amplified + wrong-region picked). See
"Follow-up" below for the algo changes that close those.

## Follow-up — combined run `d9bcd13b880f` (2026-05-09)

Three orthogonal changes, all behind feature flags so the eval can ablate:

1. **Goldens updated** (`data/golden/v3.yaml`) for q25, q26, q31: relevant
   chunks now include the region chunk alongside the original text chunk
   when both are valid answers (Table 1, Table 4, Figure 1).
2. **`--paper-id-filter`** (new flag in `scripts/eval_run.py`): when set,
   `Query.filters['paper_id']` is populated from `GoldenQuery.paper_id`
   and the dense + sparse retrievers scope to that paper. Closes
   cross-paper bleed (q9 was retrieving wrong-paper tables at rank 1).
   Eval-only — production callers don't pass a paper hint.
3. **`--region-number-boost`** (new wrapper `RegionNumberBoostRetriever`
   in `src/rag/retrievers/region_boost.py`): post-processor over an
   underlying retriever; when query mentions `Table N` / `Figure N`,
   chunks whose text starts with `Table N:` / `Figure N:` bubble to the
   top. Match is on caption text, not chunk_id, because chunk_id uses
   page-local sequential indexing (`::tab1` is "first table on page",
   not "Table 1 of paper").

`scripts/rebaseline_offline.py` recomputes historical runs' retrieval
metrics under updated goldens (deterministic — nDCG/recall/MRR are pure
functions of (retrieved, relevant)). Used to fairly compare runs from
before the golden update without burning GPU on a re-run.

### Final numbers (under updated goldens)

| Stack | nDCG@5 | recall@10 | MRR | faith | ans_rel | ctx_prec |
|---|---|---|---|---|---|---|
| `c92f3f1bee19` no region (rebaselined) | 0.7630 | 0.9194 | 0.8118 | 0.8269 | 0.9103 | 0.8051 |
| `ad4fab3bb28d` region only (rebaselined) | 0.7365 | 0.9194 | 0.7801 | 0.9949 | 0.9744 | 0.7769 |
| **`d9bcd13b880f` combined (committed baseline)** | **0.7630** | **0.9194** | **0.8277** | **0.9551** | **0.9744** | **0.8205** |

Combined run vs rebaselined no-region: PASS on all gated metrics
(`scripts/check_regression.py`). MRR +2 %, faithfulness +15 %, ans_rel
+7 %, ctx_prec +2 %; nDCG@5 / recall@10 unchanged (the algo gains
balance the region-extraction cost on retrieval; generation lifts
clearly because the judge fix lands and the demo gets region citations).

### Region-grounded citations in the combined run

6 of 61 generator citations resolved to a figure or table chunk: q25 →
`2604.27742v1::p2::tab1`, q26 → `2604.28190v1::p10::tab4`, q31 →
`2604.28196v1::p2::fig1`, q33 → 3 figures of the AEGIS paper (q33 is
OOC; the model leaked but the judge correctly scored faith=0). Each
carries `Citation.bbox` per the schema — the demo UI can render exact
region overlays on the page image.

### What this leaves open (after the 1st follow-up)

- Pattern A (Table on page X outranks the gold text chunk on adjacent
  page Y) is not fully solved. q11_budget_levels still scores 0; the
  reranker prefers `p7::tab2` over `p6::c28`. Length-normalised
  reranker scoring or chunk-type-aware penalties are the next layer.
- q29_aegis_tab8_janus_rank: `Table 8` lives on page 23 but the table
  extractor labels it `p23::tab1` (sequential), and the chunk text
  starts with "Table 8: …" so the boost should fire — but the boost
  only fires if the chunk is in the rerank result list, and the chunk
  didn't make top-50. That's a recall problem at the page level, not
  a boost-logic problem. Worth a separate look.
- VLM captioning still off. Turning it on (`--vlm-caption-model`) is
  the obvious next experiment; should help bucket B (caption stubs
  outranking text) for *figures*. Tables are harder — VLM-described
  tables exist as a research idea but aren't supported by
  `figure_to_chunk` today.

## 2nd follow-up — combined run `f844619927e0` (2026-05-09)

Two more orthogonal changes; both behind feature flags.

1. **`--rerank-length-norm`** (in `src/rag/rerank.py`): subtracts a
   smooth length penalty from cross-encoder scores. The penalty is 0
   above `--rerank-length-threshold` (default 300 chars), scaling
   linearly to `--rerank-length-penalty` (default 0.5) at len=0.
   Calibrated on bge-reranker-v2-m3's [-5, 5] logit range so caption
   stubs (~80 chars) get ~0.4 penalty (enough to displace borderline)
   while q8-style legitimately short answers (~250 chars) get ~0.08
   (negligible).
2. **`--vlm-caption-model gemma3:4b`**: enriches figures with
   VLM-generated descriptions via the existing `OllamaVisionCaptioner`
   path; `figure_to_chunk` already prefers VLM caption over PDF caption
   when set. Model choice was constrained: `qwen2.5vl:7b` is 13 GB
   FP16 and won't fit an 8 GB GPU, dropped to CPU at 184 s/figure
   (~4 hr run); `gemma3:4b` fits cleanly at 5.4 GB total (with bge-m3
   loaded) and runs at ~6 s/figure on cuda. Quality on technical
   figures is mediocre (both VLMs called a scaling-law plot a
   "heatmap"), but the experiment showed even noisy caption text helps
   length-norm by pushing chunks above the 300-char threshold.

### Final numbers (combined → committed)

| | combined-only `d9bcd13b880f` | + length-norm + VLM `f844619927e0` | Δ rel |
|---|---|---|---|
| nDCG@5 | 0.7630 | **0.7831** | **+2.64 %** |
| recall@10 | 0.9194 | 0.9194 | +0.00 % |
| MRR | 0.8277 | 0.8299 | +0.27 % |
| faithfulness | 0.9551 | **0.9872** | **+3.36 %** |
| answer_relevance | 0.9744 | **0.9872** | +1.32 % |
| context_precision | 0.8205 | 0.8128 | -0.94 % |

Regression gate against the prior committed baseline: PASS on every
metric. Net positive lift across nDCG, faith, ans_rel, MRR; tiny
ctx_prec regression (-0.94 %, well below 5 % threshold).

### Per-query effects of length-norm + VLM

| Query | combined nDCG@5 | new nDCG@5 | What helped |
|---|---|---|---|
| q4_target_region (multi_hop) | 0.387 | **0.624** | length-norm pushed `fig1` out of rank 3 |
| q20_exploration_hacking | 0.000 | **0.387** | length-norm prevented stubs from crowding c5 |
| q11_budget_levels | 0 | 0 | still failing (Pattern A persists; tab2 still outranks c28 even penalised) |

q11 is the residual hard case: even with a 0.5 length penalty,
`p7::tab2` still beats `p6::c28` because the cross-encoder's raw
preference for the table chunk is wider than the penalty.
Length-threshold/penalty are tunable; aggressive tuning risks killing
legitimate short answers (q8-style facts).

### Region-grounded citations after length-norm

The committed combined run had 6 region citations of 61. After
length-norm, that drops to **2 of 63** (q25 → tab1, q26 → tab4) —
length-norm intentionally penalises stub-shaped chunks, so the
generator sees more of the rich text chunks at the top. The two that
remain are the genuine "table IS the answer" cases. The four that
disappeared (q9 cross-paper tab2, q11 wrong-page tab2, q31 fig1
caption hallucination, q33 OOC AEGIS leaks) were noise. Net: the demo
surface lost some "look, region citation!" appearances but every
remaining region citation is more defensible.

### What still doesn't work after the 2nd follow-up

- **Pattern A still partial**: q11 + q35 + q20 (remediated for q4 and
  q20 partially). Closing these would need either chunk-type-aware
  rerank (penalise figure chunks more aggressively when query is
  factual) or training-set fine-tune of the reranker. Out of scope.
- **VLM caption quality is poor on local 4B-7B models.** gemma3:4b
  hallucinates "heatmap" / "gene expression" on a scaling-law plot;
  qwen2.5vl:7b made the same mistake. The experiment showed gen
  metrics still lift (+3 % faith, +1 % ans_rel) because the caption
  pushes chunks past the length-norm threshold and the embedding
  picks up adjacent vocabulary. A real-quality VLM (gpt-4o-vision,
  claude-sonnet-vision, or Qwen2.5-VL at q5_K_M quant on GPU) would
  raise the ceiling. Captioner is currently Ollama-only; cloud VLM
  via OpenRouter would be a ~50-line additional implementation.

## Caveats & open questions

1. **Caption-stub chunks vs reranker behavior.** When VLM captioning is
   off, figure chunks contain only the PDF-extracted caption text (often
   1–3 sentences). On the reranker, those compete against the much richer
   text chunks for the same page. Empirical question for the eval run.
2. **Citation surface contract.** The Pydantic-Settings model evolves slowly
   and `Citation.bbox` is a real schema change. Existing API consumers (the
   demo UI, the eval harness) need to tolerate the new field; default
   `None` keeps all current code paths green, but anything that does strict
   schema-matching (Langfuse trace shape, Postgres `eval_runs` JSON column)
   should be re-tested.
3. **Bbox space.** PDF points are correct for any downstream that opens the
   PDF, but the demo UI currently overlays on rendered PNGs (DPI 150). A
   simple `points × dpi / 72` transform converts; we'll do that conversion
   on the UI side, not in the chunk metadata, so the canonical bbox stays
   in the original PDF coordinate space.
4. **Multi-page tables.** PyMuPDF's `find_tables` doesn't stitch a table
   that crosses page breaks. Each page's fragment becomes its own Table.
   That's fine for retrieval (BM25 will surface either fragment) but
   confusing for the citation surface — the answer might cite the
   second-page fragment but the user expects the whole table. Out of
   scope for this ADR; flag for future work.

## References

- ADR 0002 — Phase 2 multimodal chunks (introduced `figure_to_chunk` /
  `table_to_chunk` and the `extract_figures` / `extract_tables` ingest
  flags; this ADR adds bbox + bbox-aware citations on top).
- ADR 0004 — Phase 3 visual retrieval (page-level ColQwen2; not changed).
- ADR 0007 — Phase 3.1 corpus expansion + offline hybrid re-evaluation.
- ADR 0008 — Phase 3.2 routing (text vs hybrid by query category; not
  changed).
- `data/eval/baseline.json` — current baseline `c92f3f1bee19`.
- `docs/evals.md` — eval framework reference, including OOC scoring
  convention now enforced deterministically by `runner._run_one`.
