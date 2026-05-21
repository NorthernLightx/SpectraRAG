# ADR 0022 — Figure role classification at ingestion

**Status:** Accepted. Code-only change; no eval movement expected on the
text baseline (`eval_docling_text` has no figure chunks). The new field
exists for the figures gallery and for any future role-aware retriever.
**Date:** 2026-05-20

## Context

Spot-checking the figures gallery on `eval_docling_mm` (Docling +
`--extract-figures` ingest) surfaced an over-recall problem: of 304
"picture" chunks Docling emits across the 20-paper arXiv-2604 corpus,
**42 are below 1000 pt²** — affiliation email/social icons, the
Microsoft logo recurring on every page of `2604.28181v1`, CC-BY
license badges, and inline red-✗ status markers from `2604.28177v1`.
None of them is a publication figure in any meaningful sense, but
Docling's RT-DETR layout model fires `picture` on all of them.

Two failure modes follow:

1. The gallery (`/figures.html`) shows pages of junk thumbnails ahead
   of real figures, undermining the "browse the figures" use case.
2. When multi-modal chunks were enabled in retrieval (ADR 0020), these
   junk chunks contributed to the −6.7 % `answer_correctness`
   regression — a chunk whose text is the placeholder
   `[2604.28181v1::p1::fig1]` and image is the Microsoft logo doesn't
   help retrieve anything except the user typing "Microsoft".

The naive fix — drop everything below a size threshold at ingestion —
loses information the user might legitimately ask about ("what
license is this paper under?", "whose affiliation appears here?").
The right shape is to **characterise, then filter by context**: keep
all picture-detections in the index, tag them with a role, and let
the gallery and the retriever filter where appropriate.

## Decision

`src/types/documents.py::Figure` gains a `role: FigureRole` field with
three values: `figure`, `decoration`, `unlabeled`. The classifier in
`src/ingestion/docling_parser.py::_classify_figure_role` is pure and
deterministic:

1. **Caption-first.** If the paper itself captioned the picture with
   `^(Figure|Fig\.?)\s*\d`, it's a `figure` — regardless of how small
   the crop is. This rescues the lone 906-pt² real figure in this
   corpus ("Figure 3: Screenshots of the artifacts ...") that an
   area-only cut would have dropped.
2. **Sub-threshold uncaptioned pictures are `decoration`.** The cut at
   **5000 pt²** is data-derived from the corpus characterisation, not
   guessed — see the table below.
3. **Above-threshold uncaptioned pictures are `unlabeled`.** Real
   pictures the paper didn't formally caption (an inset diagram inside
   an equation block, a captionless schematic). Kept in the index,
   hidden from the gallery's default view.

`figure_to_chunk` forwards `figure.role` into `chunk.metadata["role"]`
so the API can surface it without re-deriving. `/figures` returns the
field; `figures.html`'s default view filters to `role=figure`, with a
"Show all detections" option that lifts the filter.

A migration cushion in `src/api/routes/figures.py::_derive_role` mirrors
the classifier and runs against the chunk's emitted caption + stored
bbox when the metadata `role` is absent — so the gallery works against
existing collections (e.g. `eval_docling_mm`) without a re-ingest.

## Data behind the 5000 pt² cut

20-paper arXiv-2604 corpus, Docling figure extraction, 304 picture
chunks total.

| area bucket (pt²) | n | what's in there |
|---|---:|---|
| <1 000 | 42 | email/social icons (✉, 😀), CC-BY logo, Microsoft logo × 23 across `2604.28181v1`, red ✗ status markers |
| 1 000 – 5 000 | 1 | another red ✗ icon at 1130 pt² |
| 3 000 – 8 000 | 0 | clean valley |
| 5 000 – 20 000 | 32 | real figures; smallest sampled was the 8 789-pt² SWAP-test quantum circuit |
| 20 000 – 100 000 | 169 | mostly real figures |
| ≥ 100 000 | 60 | full-page figures |

The bimodal distribution with a zero-density valley between ~1 k and
~5 k is the empirical justification for the cut. Of the 261 chunks at
or above 5 000 pt², **175 (67 %)** also match the `^Figure N` caption
pattern; the remainder are uncaptioned real pictures, which the
`unlabeled` role preserves.

## What this does **not** change

- **Eval baseline.** The text-only baseline (`baseline-docling-text.json`,
  ADR 0021) has no figure chunks, so this change can't move it.
- **Retrieval behaviour.** Existing retrievers don't filter by role
  yet. The metadata is in place; introducing a role-aware
  retrieval-side filter would be a separate ADR with its own
  measurement.
- **Drop-at-ingestion.** No picture-detection is dropped. The user's
  guidance — "junk is fine, the user can ask about it" — is preserved.

## What it costs

- One classifier function, ~12 lines.
- One field on `Figure` and on `FigureBrowseItem`.
- One UI dropdown on `/figures.html` and a per-role breakdown in the
  status text.
- Tests: 9 cases pinning the classifier (caption rescue, small icons,
  logo-sized decorations, missing bbox, etc.).

No measured cost on retrieval. Gallery default view on `eval_docling_mm`
becomes **209 / 304** items (started at 304, dropped 42 decorations and
53 unlabeled — see below).

## Amendment, same day — use Docling's built-in figure classifier

User pushback: "isn't Docling supposed to label all this correctly?
Docling uses models." Correct — and we weren't using them. Docling 2.94
ships `DocumentFigureClassifier-v2.5` (an EfficientNet trained on 28
document-picture classes: `logo`, `icon`, `bar_chart`, `box_plot`,
`flow_chart`, `line_chart`, `pie_chart`, `scatter_plot`, `photograph`,
`engineering_drawing`, `chemistry_structure`, `screenshot_from_*`,
`signature`, `stamp`, etc.) but it ships **off** in the default
`PdfPipelineOptions`. The first-pass implementation didn't enable it.

### Setup costs

- New dep: `onnxruntime` declared in `pyproject.toml` (Docling pulls it
  in already for OCR; the figure classifier needs it too).
- The classifier's default Transformers engine triggers `torch.compile`
  / dynamo, which needs Triton, unavailable on Windows-CPU. Setting
  `TORCHDYNAMO_DISABLE=1` at import time in `docling_parser.py`
  sidesteps the compile path and lets the model run on plain PyTorch.
  Tried the ONNX engine first — Docling 2.94 currently returns
  near-uniform predictions (~0.09 for every class) through that path,
  which we suspect is a preprocessing bug we didn't track down.
- Per-paper ingest cost: classifier adds ~5–10 s to a paper-conversion
  that was already ~45 s. Negligible at the 20-paper corpus scale.

### Priority order

Three signals feed the role, evaluated in order:

1. **Paper-authored `Figure N` caption** wins. The classifier called
   the 906-pt² "Figure 3: Screenshots of the artifacts" thumbnail
   `logo(1.00)` because the visual evidence really did look like one
   at thumbnail size, but the paper's own caption is the higher
   authority. Caption-first prevents that mis-tag.
2. **Docling classifier label at ≥ 0.30 confidence**, mapped through
   `_DOCLING_LABEL_TO_ROLE`. 28 labels collapse to 3 roles: logos /
   icons / signatures → `decoration`; everything chart-shaped or
   diagram-shaped → `figure`; `table` and `other` → `unlabeled`
   (Docling already extracts tables via a separate model, picture-side
   table hits are duplicates).
3. **Area heuristic** as the final fallback: <5000 pt² uncaptioned →
   `decoration`, else `unlabeled`.

### Measurement on `2604.28181v1` (single-paper probe collection)

46 picture-detections; classifier labels (top class per detection):

| docling label | n | role |
|---|---:|---|
| logo | 33 | decoration (Microsoft affiliation block × 33) |
| bar_chart | 5 | figure (occupation / artifact-type / rubric distributions) |
| flow_chart | 3 | figure (method-overview diagrams) |
| table | 4 | unlabeled (picture-side hits of real tables Docling already extracts separately) |
| icon | 1 | decoration (page-1 email-envelope glyph) |

Role distribution: **9 figure / 33 decoration / 4 unlabeled**. The
Figure-3 thumbnail (real caption, tiny crop) is rescued by the
caption-first priority.

Stored fields on each figure chunk's `metadata`: `role` (3-bucket),
`docling_label` (one of 28), `docling_label_confidence` (float).
Future filters / retrievers can read the rich label directly without
re-running ingestion.

### Tests added

5 new cases pin the classifier-integration logic: high-confidence
logo overrides size heuristic, high-confidence bar_chart marks as
figure, low-confidence label falls back to heuristic, `table` label
stays `unlabeled`, unknown future label falls through. Plus the
paper-caption-beats-Docling-mislabel case. 20/20 classifier tests
pass, 581/581 full unit suite passes.

## Amendment — caption regex tightened

First-pass regex `^Figure\s*\d` was too narrow against the live corpus.
Recharacterising the initial `unlabeled` bucket (86 chunks) found 38
real figures with non-standard prefixes:

- 15 with subfigure-letter captions: `"(a) CDF of prediction errors..."`,
  `"(b) Qwen3-14B with locking strategies..."`
- 21 with letter-numbered or page-prefixed shapes:
  `"Figure C.1: Screenshot..."`, `"Figure F. Samples of AMD"`,
  `"1 Figure 9: The trade-off..."` (the leading `1` is a column-merge
  artifact from PDF text extraction)
- 2 with table-style captions on picture-side detections of tables —
  intentionally left as `unlabeled` since Docling's separate tables
  loop already emits a proper `Table N` chunk

Extended pattern:

```
^\s*(?:\d+\s+)?(?:(?:figure|fig\.?)\s+[A-Z0-9]|\([a-z]\)\s)
```

This rescues all 36 captioned cases. After tightening: **209 figure /
53 unlabeled / 42 decoration**. The remaining 53 unlabeled all carry
the `[paper::p::fig]` placeholder — i.e. zero caption text — so a
caption-only classifier can't push further. A VLM-based labeller
could, at the usual cost; left as out-of-scope here.

## Amendment — role-aware retrieval filter + gallery floor for tiny decorations

**Date:** 2026-05-21

Resolves the retrieval-side filter deferred under "What this does **not**
change" ("a role-aware retrieval-side filter would be a separate ADR with its
own measurement"), and adds a gallery floor for the glyph-sized decorations
that prompted it.

### Retrieval-side filter — measured recall-neutral

`PipelineRetriever` now drops candidates whose `metadata["role"] ==
"decoration"` from **both** legs before rerank/return, behind
`Settings.exclude_decoration_chunks` (default-on; `RAG_EXCLUDE_DECORATION_CHUNKS`);
`scripts/eval_run.py` exposes `--exclude-decoration` / `--no-exclude-decoration`
for A/B.

Filtered vs unfiltered on the only decoration-bearing collection
(`eval_docling_classified_tables`): nDCG@5, recall@10, MRR all **+0.00 %**,
retrieved IDs **byte-identical** for every query, every subset flat.
baseline.json unchanged — nothing to rebaseline.

Mechanism: a decoration chunk's only indexed text is its id-stub
`[paper::p::figN]` (no caption; the VLM captioner skips decorations), which
scores near-zero on both BM25 and dense, so decorations never enter the top-50
candidate pool — confirmed by pool probes and the retriever's own "nothing
dropped" logs. The −6.7 % `answer_correctness` harm in the original Context was
a **generation-side** effect (decorations reaching the answer context), not a
retrieval-metric one. So this lands as an **explicit default-on guardrail** —
making role-awareness intentional rather than relying on the accident that
id-stubs score below the cut — not a retrieval win; the gate passes because it
cannot worsen retrieval.

Caveat: no Qdrant collection has *both* a `role=decoration` population *and*
golden-v3 chunk-id alignment (the role field postdates the golden-aligned
collections), so the A/B ran filtered-vs-unfiltered on the same collection,
where anchor drift cancels in the delta.

### Gallery floor for tiny decorations

The role + default-off Decorative bucket keep glyphs out of the gallery's
default view, but the *opt-in* Decorative view still showed 12-pt table-emoji
detections (e.g. `2604.28177v1` p2, 6 chunks at 146 pt²). `/figures` now also
drops `role=="decoration"` items below `_MIN_DECORATION_AREA_PT2` (500 pt²).
Decoration-only, so it cannot touch a real figure — the smallest in-corpus
figure is a 514-pt² *caption-rescued* `figure`, and real figures are never
`role=decoration`.

## Amendment — bbox-less pictures default to `unlabeled`, not `decoration`

**Date:** 2026-05-21

A figures-gallery report that *looked* like real figures misclassified as
`decoration` turned out to be a **stale corpus**: the deployed `rag_corpus`
predates this classifier (built by the old PyMuPDF extractor — 0 docling
labels), so the gallery's `_derive_role` cushion was re-deriving roles at view
time. The current Docling pipeline classifies those papers' figures correctly
(verified on `2604.28196v1`/`2604.28197v1`: 5/5 and 7/7 real figures detected,
all `figure`).

The investigation did surface a genuine latent defect in the fallback ladder,
fixed here: the terminal `bbox is None` branch returned `decoration`.
`decoration` is the one role removed from the gallery content view, excluded by
the role-aware retrieval filter, *and* skipped by the VLM captioner — so a real
figure whose bbox is ever absent would be caption-starved and dropped from both
surfaces, invisibly. The fallback now returns `unlabeled` (kept in retrieval,
hidden only from the gallery's default view) in both `_classify_figure_role`
and the `_derive_role` migration cushion, which stay in sync. A picture is
`decoration` only on positive evidence now: a confident logo/icon-class label
or a measured sub-5000-pt² area.

Defensive — no bbox-less figure was observed in the current pipeline on the
sampled papers; the change removes a silent figure-loss path before a corpus
rebuild. (Also hardened: `parse_with_docling` now clears the per-paper crop dir
each run, so re-ingests don't accumulate a palimpsest of stale crops.)

## Related

- ADR 0009 — region-precise bboxes; this ADR rides on the same bbox
  the figure chunk already carries.
- ADR 0020 — Docling for figure/table extraction; this ADR addresses
  the over-recall the layout model produces.
- ADR 0021 — Docling text chunker. Unaffected — text chunks have no
  role.
