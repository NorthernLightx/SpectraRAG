# ADR 0021 — Docling as the text-chunking source too (proposed, eval in flight)

**Status:** Proposed. Implementation landed (`src/ingestion/docling_chunker.py`
+ pipeline wiring + 13 tests), default flipped (`use_docling=True` everywhere),
final eval against the pre-Docling-text baseline running in background. The
ADR moves to Accepted (or back to Proposed with a re-scope) when the
measured `answer_correctness` delta lands in the "Measurement" section.
**Date:** 2026-05-20

## Context

ADR 0020 swapped PyMuPDF for Docling on **figure / table extraction only**
— audit flag rate halved, heterogeneous formats held. The text path was
untouched: `pdf.py::extract_pages` (PyMuPDF) + `chunking.py::chunk_pages`
(ADR 0017's regex section splitter) still ran the show. The follow-up
probe `scripts/experiments/docling_text_probe.py` surfaced five concrete
gaps in that path that the audit on figures/tables never reached:

1. **Section labels are broken.** On `2604.22753v1` the existing chunker
   labels chunks 1 → 12 as `"Abstract"` because PyMuPDF's column
   merging glued `"1 Introduction"` into body text, defeating the
   regex. Docling labels every section header (`Abstract`, `1
   Introduction`, …) as `section_header` directly.
2. **Figure-interior numbers leak into body chunks.** Chunk 5 of that
   paper is `"3 2.095 2.103 2.112 2.137 …"` — the heatmap values from
   Figure 1, sitting under section "Abstract". `is_soup` doesn't catch
   the mixed case. Docling places each axis-tick and grid value as its
   own `text` block *inside the figure's bbox*, so a bbox-containment
   filter cleanly removes them.
3. **Math is anonymous prose in PyMuPDF, labelled in Docling.** 104
   `formula` blocks on this paper. The old chunker mashes them into
   prose; Docling labels them so they can be handled deliberately.
4. **Page furniture is labelled in Docling, guessed in PyMuPDF.** 26
   `page_header` + 25 `page_footer` blocks here, all labelled
   directly. `clean.py::detect_running_header` from ADR 0017 has to
   guess them by word-prefix frequency across pages.
5. **No bbox on text chunks.** Every Docling text block has
   `prov.bbox`. The PyMuPDF chunker discards this; region-precise
   citations (ADR 0009) are figure/table-only today.

## Decision (subject to measurement)

`src/ingestion/docling_chunker.py::chunk_with_docling` consumes
`DoclingDocument.texts` in reading order, with three deterministic
filters:

- **Label whitelist for body content** — `{text, list_item, formula,
  footnote, code, paragraph}` only. Page furniture (`page_header`,
  `page_footer`) and captions are dropped at the boundary; captions
  belong to `figure_to_chunk` / `table_to_chunk` already.
- **Figure / table-region containment filter** — any body block whose
  bbox sits inside (≥80 % area overlap with) a figure or table bbox on
  the same page is excluded. No more axis-tick leakage.
- **Section accumulation** — `section_header` blocks become the
  `Chunk.section` label; everything between two `section_header`s
  becomes that section's body, joined with `\n`, then windowed to
  `target_chars` via the same `_window_spans` as the pre-Docling
  chunker.

Each emitted `Chunk` carries `metadata['bbox']` (the union of
contributing-block bboxes) when the chunk is single-page. Cross-page
chunks omit bbox — the project's `Bbox` is page-local (ADR 0009).

`pipeline.py::ingest_paper`'s `use_docling=True` path now:
1. Runs Docling once (`convert_with_docling`).
2. Feeds the converted `DoclingDocument` to `chunk_with_docling` for
   text, and (when enabled) to `parse_with_docling(doc=doc)` for
   figures/tables. **One conversion per paper, not two.**
3. Builds `paper_text` for contextualization from the same labelled
   body blocks (`paper_text_from_docling`), so the long-context prompt
   doesn't carry page furniture either.

`eval_run --use-docling` switched from `store_true` (opt-in) to
`BooleanOptionalAction` with `default=True`; pass `--no-use-docling`
to fall back to PyMuPDF for repeatability of pre-ADR-0021 baselines.

## Measurement (in flight)

Background run, same v3 / `gemma3:4b` gen + judge / `paper-id-filter` /
no-rerank / no-router as `baseline-text-only.json`. Only the chunker
changes. Result lands at `data/eval/baseline-docling-text.json`.

| metric | pre-Docling-text | Docling text | Δ |
|---|---:|---:|---:|
| answer_correctness | 0.7626 | _(pending)_ | _(pending)_ |
| faithfulness | 0.7454 | _(pending)_ | _(pending)_ |
| answer_relevance | 0.6179 | _(pending)_ | _(pending)_ |
| context_precision | 0.8077 | _(pending)_ | _(pending)_ |
| p50 latency | 20.3 s | _(pending)_ | _(pending)_ |

The structural wins should not regress headline `answer_correctness`
on this corpus — the new chunks are *cleaner* (no figure-text leak, no
page furniture), section labels are correct, and the windowing budget
is unchanged. Expected directional read: flat or modest improvement;
any clear regression triggers an investigation before ADR moves to
Accepted.

## What this leaves open

- **Per-block label exploitation.** Right now `formula`, `list_item`,
  `footnote` all flow into the same windowed text body. Future work
  could give equations their own retrieval path or format markdown
  lists explicitly. Not in scope for ADR 0021 — first establish that
  the cleaner text path doesn't regress.
- **Bbox-aware citation UX.** The bbox is now in `Chunk.metadata`;
  rendering it as a region highlight on text answers is a UI change
  (`/figures.html` already does this for figure chunks; same surface
  would work for text).
- **Cross-page chunks losing bbox.** A chunk that straddles a page
  break has no single bbox (page-local invariant). Could emit a
  per-page bbox list in metadata; not now.
- **Near-duplicate chunks from `overlap_chars=200`.** Still produces
  ~17 % overlap between adjacent chunks. Separately tracked; would
  benefit from sentence-aware boundaries that exist in `_window_spans`
  but are tunable.

## Related

- ADR 0017 — corpus clean; the regex chunker this ADR supersedes for
  the Docling path. PyMuPDF + ADR-0017 chunker stays as the fallback
  when `use_docling=False`, so pre-ADR-0021 baselines remain
  reproducible.
- ADR 0020 — Docling for figure/table extraction; this is the obvious
  follow-on that uses the same parse for text too, no second pass.
- ADR 0009 — region-precise citations; ADR 0021 extends the
  bbox-on-chunk invariant from figures/tables to text.
- ADR 0019 — `answer_correctness` is the scoreboard this ADR's
  measurement uses.
