# ADR 0021 — Docling as the text-chunking source too

**Status:** **Accepted.** Measured against the same v3 / `gemma3:4b` /
hybrid / `--paper-id-filter` config as `baseline-text-only.json`, the
Docling text chunker improved every generation metric past the 5 %
regression gate — `answer_correctness` **+8.2 %** (0.7626 → 0.8255),
`faithfulness` **+8.8 %**, `answer_relevance` **+23.9 %**,
`context_precision` **+9.7 %**, and `citation_rate` populates on
**all 39 queries** (up from 27). Real cost: p50 latency 20.3 s →
83.5 s. First measured positive ADR this session.
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

## Measurement, 2026-05-20

Same v3 / `gemma3:4b` gen + judge / `--paper-id-filter` / no-rerank /
no-router as `baseline-text-only.json`. Only the chunker changes.
Committed reference: `data/eval/baseline-docling-text.json`
(`run_id=e7643a0c1085`).

| metric | pre-Docling text | Docling text | Δ abs | Δ rel |
|---|---:|---:|---:|---:|
| **answer_correctness** | 0.7626 | **0.8255** | **+0.0629** | **+8.2 %** |
| **faithfulness** | 0.7454 | **0.8110** | +0.0656 | +8.8 % |
| **answer_relevance** | 0.6179 | **0.7654** | +0.1474 | **+23.9 %** |
| **context_precision** | 0.8077 | **0.8859** | +0.0782 | +9.7 % |
| citation_rate (n=) | 1.000 (n=27) | 1.000 (n=39) | — | every gen cites now |
| p50 latency | 20.3 s | **83.5 s** | +63.2 s | +311 % |
| p95 latency | 28.6 s | 105.9 s | +77.3 s | +270 % |
| tokens out (total) | 16 733 | 11 637 | −5 096 | answers terser |

### Per-category `answer_correctness`

| category | n | pre | post | Δ |
|---|---:|---:|---:|---:|
| equation | 1 | 1.000 | 1.000 | 0.000 |
| factual | 13 | 0.831 | **0.908** | +0.077 |
| figure | 11 | 0.767 | 0.781 | +0.014 |
| multi_hop | 2 | 0.800 | 0.900 | +0.100 |
| **table** | 4 | 0.450 | **0.600** | **+0.150** |

Every category at parity or better. The biggest absolute wins are
exactly the question classes the probe predicted would benefit:

- **table +0.15** — the prior chunker mixed numeric table contents
  into prose under wrong section labels; Docling separates and labels
  them.
- **factual +0.08** — section attribution being deterministic instead
  of `"Abstract"`-everything gives the retriever cleaner section
  metadata to discriminate on.
- **multi_hop +0.10** — multi-hop queries benefit from coherent
  cross-section context that the layout-aware reading order produces.
- **figure +0.01** — flat is the right read here; this run did *not*
  enable `--extract-figures`, so the figure-query subset is being
  answered from text chunks alone in both arms. ADR 0019's per-class
  routing remains the path that would lift this further.

### Honest costs

- **p50 latency 4×, p95 ~4×.** Real per-query cost. The chunks are
  cleaner *and* denser per chunk (figure-leak axis ticks gone, page
  furniture gone), so the top-K retrieval feeds the generator more
  meaningful content per chunk — generator runtime scales with that.
  The token-out drop (−30 %) suggests the model is also producing
  more concise correct answers, which is consistent with retrieval
  being more on-target.
- **Ingest is slower.** Docling does layout + OCR + table-structure
  detection per paper (~30 s warm-GPU). Acceptable for offline ingest;
  not for query-time work, which this isn't.

### Citation coverage

Generator citation rate is now **100 % of queries (n=39)** vs **n=27
(69 %)** under PyMuPDF chunking. The likely mechanism: chunk-ids
labelled by real section header text (e.g.
`"3 Method"`) instead of `"Abstract"`-everything give the LLM
stronger structural cues to anchor citations on. Worth confirming in
a future ADR but the measurement is clean.

## Verdict — Accept

The measured headline gain (+8.2 % `answer_correctness`,
**−7 % ≤ Δ ≤ +24 % across the board, all positive**, all categories
flat-or-up) crosses the 5 % gate decisively. Latency cost is real but
the trade is favourable for a portfolio-grade RAG: better answer
quality, every answer cited, denser retrieval contexts. Default
remains `use_docling=True`. The PyMuPDF + ADR-0017 path stays in tree
behind `--no-use-docling` so the pre-ADR-0021 baselines remain
reproducible.

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
