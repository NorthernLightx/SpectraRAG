# ADR 0011 — Figure caption aggregation (one Figure per `Figure N:` caption span)

**Status:** Accepted (2026-05-10).
**Date:** 2026-05-10.

## Context

Run `b8r2w5kc4` (Tier 1 + Tier 2 eval on the $0 Nemotron stack) hit a
pathological extraction on paper `2604.28190v1` (FD-loss): the PDF holds a
single appendix figure — `Figure E.1: Uncurated paired samples on ImageNet
256×256` — encoded as a 5×2 grid of class-panels, each panel a 4×4 grid of
16 model-output thumbnails. PyMuPDF's `page.get_images(full=True)` returns
each of those ~220 sub-thumbnails as a distinct XREF. The pre-ADR-0011
extractor emitted one `Figure` per XREF, so a single logical figure became
220 chunks. Pages 20–23 each had a similar grid; in total the paper
contributed ~1000 figure chunks where ~10 was correct.

Downstream this corrupted retrieval:
- BM25 + dense both got flooded with caption-stub chunks that share the same
  caption text — every sub-thumbnail emitted a chunk whose `text` was the
  PDF caption (since `figure_to_chunk` falls back to it).
- The reranker spent its budget on near-identical near-duplicate chunks.
- VLM captioning got hammered (one VLM call per sub-thumbnail).
- Ingestion time on this paper alone took ~45 min on the 8 GB GPU before we
  killed the run.

The architectural problem is a layer conflation. ColQwen2 (our visual leg)
already does micro-attention right: it embeds each ~14×14 patch of the
rendered page and scores via late interaction (MaxSim) — fine-grained
features for *scoring*, coarse units for *retrieval*. Our `extract_figures`
inverted that: fine-grained units (XREFs at the visual-element level) became
retrieval units, while losing the aggregate caption context.

A band-aid landed first (`min_dim` 64 → 256) which dropped most
sub-thumbnails by size. That works for 28190 specifically (its sub-cells
are ~120 px) but is fragile: a composite with larger sub-panels would still
flood, and the band-aid drops information that the principled fix can
preserve.

## Decision

**One `Figure` per logical caption span**, not per XREF.

1. **Parse `Figure N:` labels with bbox.** `_captions_with_bboxes(page)`
   returns `{label: (caption_text, label_bbox)}`. Text body comes from the
   full-text regex (so multi-line bodies survive); bbox comes from the
   PyMuPDF text block that contains the label.

2. **Broaden the label regex.** Appendix and supplementary figures use
   letter-prefixed labels: `Figure E.1`, `Figure A.3`, `Figure S1`, etc.
   The prior `\d+`-only pattern missed `Figure E.1:` entirely, which on
   28190 meant pages 20–23 had 220 XREFs each and *zero* parsed captions
   — every XREF fell through to the per-XREF fallback. Label keys are now
   strings (`"1"`, `"E.1"`, `"S1"`) instead of ints.

3. **Assign XREFs to captions by nearest-y.** `_assign_xrefs_to_captions`
   maps each XREF whose bbox we know to the caption whose label bbox has
   the closest y-center. Captions partition the page into vertical bands;
   composite XREFs all cluster around one caption naturally. XREFs without
   bboxes (or pages without bboxed captions) fall through to a per-XREF
   fallback path.

4. **One Figure per non-empty caption group.** Representative `image_path`
   = the largest member XREF (by area), saved as the PNG. `bbox` = union of
   all member XREFs (so citation rectangles point at the whole composite,
   not one cell). PDF caption text from the parsed body.

5. **Revert `min_dim` to 64.** With aggregation, the volume problem is
   solved at the right layer; tiny XREFs that pass `min_dim` get bundled
   into the right Figure instead of dropped. The floor stays at 64 to skip
   pure noise (1-px separators, vector-art outlines PyMuPDF reports as
   "images").

## Validation

Unit tests cover the primitives (`_union_bbox`, `_assign_xrefs_to_captions`
with single-caption, multi-caption, no-caption, and missing-bbox cases) and
the appendix-label regex (`Figure E.1:`).

The integration test `test_extract_figures_aggregates_composite_paper`
asserts that paper `2604.28190v1` extracts to **< 50** figures. Actual:
**12** figures (one per logical caption: `Figure 1, 4, 5, 7, A.1, C.1,
E.1-E.4, G.1, G.2`). Pre-ADR-0011 with `min_dim=64`: ~1000. With the
`min_dim=256` band-aid: ~50–80 (the sub-thumb floor effect). With
caption-anchored aggregation: 12 — the right number.

Logged per-figure manifest sizes confirm the aggregation is grouping
correctly:
```
caption_label=E.1 n_members=216
caption_label=E.2 n_members=218
caption_label=E.3 n_members=216
caption_label=E.4 n_members=217
```

## What this leaves open

- **VLM captioning operates on the representative XREF only.** For a
  composite, the rep is one panel — the VLM sees one cell, not the grid.
  PDF caption text covers the aggregate, so retrieval signal is preserved;
  the loss is on VLM caption fidelity for composites specifically. Fix
  would be rendering the union-bbox region of the page as a single composite
  PNG and feeding that to the VLM. Out of scope here.
- **Caption-less composite pages still over-decompose.** If a paper encodes
  a composite without any `Figure N:` label on the page, all XREFs fall
  through to the per-XREF fallback. None of the v3 corpus papers exhibit
  this; if a future paper does, the fix is spatial clustering on the
  fallback path.
- **Nearest-y assignment can mis-bin XREFs near a caption boundary.** On a
  page with two captions stacked tightly, an XREF equidistant between them
  binds to the lexically-first one. Rare; not worth bbox-IoU-style logic
  unless eval evidence appears.

## Related

- ADR 0009: region-level evidence with bbox citations. This ADR fixes the
  unit of aggregation; ADR 0009's bbox surface continues to work — the
  citation just points at a union rect now instead of one sub-cell.
- The band-aid `min_dim=256` is replaced by `min_dim=64` + aggregation.
