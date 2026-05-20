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

No measured cost on retrieval. Gallery default view drops from 304 to
~261 visible items on `eval_docling_mm`, mostly removing the 23
Microsoft-logo duplicates and the inline icons.

## Related

- ADR 0009 — region-precise bboxes; this ADR rides on the same bbox
  the figure chunk already carries.
- ADR 0020 — Docling for figure/table extraction; this ADR addresses
  the over-recall the layout model produces.
- ADR 0021 — Docling text chunker. Unaffected — text chunks have no
  role.
