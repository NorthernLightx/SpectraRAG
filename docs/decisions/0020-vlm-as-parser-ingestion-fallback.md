# ADR 0020 — VLM-as-parser ingestion fallback (proposed; spike says continue)

**Status:** Proposed; the kill-spike on the ADR-0017-amendment audit-flagged
miss class recovered **5 of 5 known misses with no control-page
hallucination**. Next step is wiring it as a deterministic-first / VLM-
fallback **cascade** behind `extract_figures` + `extract_tables`
(mirroring the ADR 0010 cost-quality philosophy), and measuring on a
deliberately heterogeneous corpus before flipping it on by default.
**Date:** 2026-05-20

## Context

ADR 0017's "−19 %, zero content loss" claim was scoped honestly in its
2026-05-20 amendment: it covers *text* content; the figure / table path
inherits PyMuPDF's known limits. The overlay audit
(`scripts/audit_ingestion_overlay.py`) made the gap concrete — across 4
diverse papers (87 pages) **~14 % of pages have at least one figure or
table the extractor silently missed**, dominated by two mechanisms:

1. vector-drawn figures invisible to `page.get_images()` (matplotlib-
   style stroked plots saved without rasterisation), and
2. `page.find_tables()` heuristic failures on tight numeric tables
   (plus the inverse: false-positive table regions on dense-numeric
   non-tables).

Cross-format the miss rate is expected to be higher — IEEE templates
with Roman-numeral table labels, slide-deck PDFs with no structural
conventions, non-English captions (`Abbildung` / `图`), OCR'd scans.
**The regex-label strategy is structurally brittle.** The repo already
has `qwen3-vl:235b-cloud` pulled via Ollama (matches the
cloud-via-Ollama operating preference), so a VLM-as-parser fallback for
the miss class can be tested with no new dependency.

## Spike

`scripts/experiments/vlm_layout_spike.py`. Six pages from
`2604.22753v1`: five from the audit's flagged-misses list plus one
known-good control. Each rendered at 150 DPI, sent to
`qwen3-vl:235b-cloud` with a strict JSON-array prompt ("list every
figure and table with `{type, label, bbox, caption, summary}`; empty
array if none; no prose"). One call per page, `temperature=0.0`,
`num_predict=2000`. Raw results in
`data/eval/ingestion/vlm-spike/spike-results.json`.

## Result

| page | ground truth | VLM output | verdict |
|---|---|---|---|
| p02 (control) | Figure 1 only | `figure 1`, bbox `[177, 90, 817, 256]`, caption "Our method identifies the extrapolation optimum…" | ✓ correct, no hallucination |
| p06 | Table 1 | `table 1`, bbox `[193, 100, 794, 245]`, caption "Task statistics for the scaling-law benchmark." | ✓ recovered |
| p07 | Figure 2 + Table 2 | `figure 2` + `table 2`, both bboxes, both captions | ✓ both recovered |
| p08 | Figure 3 | `figure 3`, bbox `[150, 95, 750, 290]`, "Parameter-space visualization…" | ✓ recovered |
| p09 | Table 3 | `table 3`, bbox `[114, 81, 879, 242]`, "Ablation study of the acquisition function." | ✓ recovered |
| p13 | Table 4 | `table 4`, bbox `[172, 823, 823, 924]`, "Collected scaling laws grouped by task." | ✓ recovered |

**5 / 5 misses recovered, 0 hallucinations on the control page.** Bboxes
are within page bounds (1275 × 1650 px), aspect ratios match the
visible artifacts in the audit overlay PNGs, and captions are
textually faithful to what is on the rendered pages. JSON parse rate
on the spike: 6 / 6.

## Verdict — continue

Cascade-style integration is justified:

- **Fast path (unchanged):** `extract_figures` + `extract_tables` run
  first; both are deterministic, fast, no LLM cost, sufficient for the
  86 % of pages they already handle.
- **Fallback path (new):** for each page where the overlay-audit logic
  flags a missed caption label (`Figure N:` / `Table N:` in text with no
  matching extracted artifact on that page), send the rendered 150 DPI
  page image to a VLM with the spike's prompt. Parse the JSON, emit
  `Figure` / `Table` objects (with bboxes converted from image px to
  PDF points). Fail-soft: per-page errors fall through to existing
  behaviour, like `captioner.py`.
- **Hallucination guard:** the fallback only runs on flagged pages —
  it never adds artifacts the extractor was confident about. The
  control-page result confirms the VLM also does not invent artifacts
  when there genuinely are none, but the audit-gated invocation is the
  belt-and-suspenders defence.

## Cost and latency, honest

Six cloud calls finished in roughly one to two minutes total — bounded
and acceptable. At the audit's 14 % flag rate, a 20-paper corpus
(~420 pages) would trigger ~60 fallback calls, not 420. Per-paper
fallback cost is bounded by the *miss rate*, not the page count. This
keeps the cascade economics close to the deterministic fast path while
making robustness opt-in only where it is needed.

## What this leaves open

- **Bbox precision.** Spike bboxes look plausible by eye against the
  audit overlays; nothing has yet *measured* IoU vs hand-drawn ground
  truth. Before flipping the fallback on by default, sample-label a
  handful of pages and report bbox-IoU; cap at the precision PyMuPDF
  delivers for its own hits (which is the bar the citation surface in
  ADR 0009 already accepts).
- **Heterogeneous-format measurement.** The spike was on one paper. A
  proper integration ADR should ingest **one IEEE paper, one slide-deck
  PDF, and one OCR'd scan** in addition to the existing ArXiv corpus,
  re-run the overlay audit, and report the flag rate. That is the
  honest test of "feeds whatever documents possible."
- **VLM-JSON robustness.** The tolerant parser in `graph_extract.py`
  (fence-stripping, outermost-object slicing) is the template; reuse it
  rather than re-implementing. Per-page failure must degrade to
  existing behaviour, never abort the batch.
- **Cascade for tables specifically.** `find_tables()` also produces
  *false-positive* table regions (audit on `2604.22753v1` reports 8
  extracted tables for a paper with 4 numbered tables visible). The
  fallback only addresses *misses*; a future iteration could also let
  the VLM cull obvious false positives, but that needs its own
  evidence and is not in scope here.

## Related

- ADR 0017 — corpus clean; its 2026-05-20 amendment introduced the
  overlay audit that makes this fallback's invocation gate possible.
- ADR 0010 — cost-quality cascade; this is the same pattern applied to
  ingestion rather than retrieval.
- ADR 0002 — multi-modal chunks; the VLM caption path; this fallback
  reuses the same `Figure`/`Table` types so nothing downstream changes.
- ADR 0018 — kill-spike methodology (one decisive experiment before
  committing to a build); this ADR is the same shape.
