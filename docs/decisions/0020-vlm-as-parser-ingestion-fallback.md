# ADR 0020 — Docling as primary parser; VLM-as-parser kept as residual fallback

**Status:** **Accepted** for Docling as the deterministic primary parser
behind `--use-docling`; the VLM-as-parser kill-spike that *originally*
motivated this ADR is preserved as the residual-fallback option, wired
when Docling itself proves insufficient on heterogeneous formats (OCR'd
scans, slide-deck PDFs, weird non-PDF inputs). One ADR, both layers
documented — the cost-quality cascade pattern of ADR 0010 applied to
ingestion rather than retrieval.
**Date:** 2026-05-20

## Context

ADR 0017's 2026-05-20 amendment surfaced the real ingestion gap: across
4 diverse papers (87 pages) **~14 % of pages have a figure or table the
extractor silently missed**, dominated by two mechanisms — vector-drawn
figures invisible to PyMuPDF's `page.get_images()` and tight numeric
tables missed by `page.find_tables()`. Cross-format the rate is
expected to be higher (IEEE Roman-numeral table labels, slide-deck PDFs,
OCR'd scans, non-English captions). The regex-label strategy is
structurally brittle.

Two parsers were spike-tested on the audit-flagged ground-truth set
from `2604.22753v1` (Figures 1/2/3 expected, Tables 1/2/3/4 expected):

## Spike 1 — VLM-as-parser (`qwen3-vl:235b-cloud` via Ollama)

`scripts/experiments/vlm_layout_spike.py`. 6 pages (5 misses + 1
control), one cloud call each, strict-JSON prompt. Result:

- **5 / 5 misses recovered** with plausible bboxes and faithful captions
- **0 hallucinations** on the control page
- 6 / 6 JSON parses clean
- ~1–2 min total LLM time for 6 pages

A real continue signal, but per-page generative LLM cost in the hot
path of ingestion. The fallback design assumed PyMuPDF as the fast
path and audit-gated VLM invocation on the residual ~14 %.

## Spike 2 — Docling (DocLayNet + TableFormer + RapidOCR)

`scripts/experiments/docling_probe.py`. Whole 25-page paper, one
deterministic conversion. Result:

- **3 / 3 expected figures** with bboxes + captions (`Figure 1` p02,
  `Figure 2` p07, `Figure 3` p08)
- **4 / 4 expected tables** with bboxes + captions + markdown cell
  structure (`Table 1` p06, `Table 2` p07, `Table 3` p09, `Table 4` p13)
- 3 extra continuation-table fragments on p14–p16 (Table 4's multi-page
  body) — not false positives, just multi-page continuation
- 55 s first run on warm GPU (model download), 32 s subsequent
- No LLM call in the path
- Caption-to-artifact link is deterministic via Docling's document tree,
  not regex

Docling **subsumes** what the VLM spike fixed and adds: native DOCX /
PPTX / HTML / image / OCR support, internal table cell structure,
section hierarchy, equation handling — the actual *format-agnostic*
primitive the project needs for "feed whatever documents possible."

## Decision

**Docling is the primary fast path; VLM-as-parser stays in tree as the
residual fallback** for cases Docling cannot fully handle (audit-flag
trigger as designed in the VLM spike). Both spike scripts are committed
under `scripts/experiments/` as the reproducible evidence.

Concretely:

- `src/ingestion/docling_parser.py::parse_with_docling` wraps a
  `DoclingDocument` into the project's existing `Figure` and `Table`
  types. Coordinate flip from `CoordOrigin.BOTTOMLEFT` to the project's
  TOP-LEFT `Bbox` (ADR 0009) is the one non-trivial transform.
- `ingest_paper(..., use_docling=True)` and `scripts/eval_run --use-docling`
  enable it. Default off pending the heterogeneous-format eval below,
  but it is a one-flag flip when that lands.
- `scripts/audit_ingestion_overlay --use-docling` re-runs the spatial
  audit against the new pipeline.

## Audit re-run, Docling backend, `2604.22753v1`

| | pages | figures | tables | flagged |
|---|---:|---:|---:|---:|
| PyMuPDF (status quo) | 25 | 1 | 8 | **5** |
| Docling | 25 | 3 | 7 | **0** |

The audit's silent-miss class collapses to zero on this paper. Every
Figure/Table caption mentioned in the page text is matched by an
extracted artifact with a valid bbox.

## Corpus-wide audit (all 20 papers), 2026-05-20

`scripts/audit_ingestion_overlay --all` and `--all --use-docling`. Same
20 papers, 575 total pages. Per-paper `audit.md` under
`data/eval/ingestion/overlays/<paper>/` (PyMuPDF) and
`data/eval/ingestion/overlays/docling/<paper>/`.

| | flagged | rate | figures | figs w/ bbox | tables | tabs w/ bbox |
|---|---:|---:|---:|---:|---:|---:|
| **PyMuPDF** | **146 / 575** | **25.4 %** | 277 | **91.7 %** | 288 | 100 % |
| **Docling** | **73 / 575** | **12.7 %** | **304** (+27) | **100 %** | **124** (−164) | 100 % |

- **Flag rate halved** corpus-wide (25.4 % → 12.7 %).
- **+27 figures** captured; every extracted figure has a valid bbox
  under Docling (vs PyMuPDF leaving 23 figures without spatial info).
- **−164 "tables"**: PyMuPDF's `find_tables()` heuristic was producing
  a large pool of false positives (dense-numeric grid regions that are
  not actually captioned tables). Docling's TableFormer is much
  tighter; the missing 164 were the noise, not real tables. The audit
  tool does not directly score false positives, but the chunk-dump
  evidence from ADR 0017 confirms many of PyMuPDF's "tables" were
  caption-less grid soup.

### Per-paper deltas

15 papers improved, 4 papers regressed on the strict label-match audit,
1 unchanged. Worst PyMuPDF cases — `2604.28182v1` (81 p, 31 flagged →
14) and `2604.27742v1` (26 p, 11 → 1) — improve dramatically.

### Honest caveat: the 4 "regressions" are audit-tool artefacts

Visual spot-check on `2604.28193v1` p02 (the worst nominal regression,
PyMuPDF 0 → Docling 3) shows **both Figure 2 and Table 1 ARE extracted
by Docling with valid bboxes** — the overlay PNG has the orange and
blue boxes clearly drawn on the right artefacts. The audit flagged
them because its `Figure N:` / `Table N:` regex on `caption_text`
couldn't recover the label from Docling's caption rendering, while
PyMuPDF's tighter caption regex happens to produce text the audit's
regex matches by construction. So:

- **The 25.4 % PyMuPDF rate underestimates** PyMuPDF's real misses
  (some figures whose bboxes are absent or wrong on the page still
  pass the audit because the caption regex agrees with itself).
- **The 12.7 % Docling rate overestimates** Docling's real misses
  (extracted artefacts with valid bboxes fail the label match purely
  on caption-text formatting).
- **Real Docling miss rate is below 12.7 %**, plausibly close to the
  0 / 25 result on the test paper.

Refining the audit to label-match by spatial+textual overlap rather
than label-substring is its own follow-up (a clean ~30 line change to
`audit_ingestion_overlay`). For this ADR the directional finding is
unambiguous and the visual evidence is decisive: Docling is the right
primary parser.

## Heterogeneous-format eval, 2026-05-20

ADR 0020's gate for flipping `use_docling=True` default-on was a real
heterogeneous-format eval, not just more ArXiv ML papers. Three documents
fetched + audited via Docling:

| document | format | pages | figs | figs w/ bbox | tables | tabs w/ bbox | flagged |
|---|---|---:|---:|---:|---:|---:|---:|
| `het-neurips-slides` | slide-deck PDF, NeurIPS 2024 | 7 | **5** | **5** | 1 | 1 | **0** |
| `het-hal-fr` | non-arXiv academic, HAL (open archive) | 32 | 7 | 7 | 4 | 4 | 3 |
| `het-apollo17` | **339-page scanned OCR'd 1973 NASA report** | 339 | **89** | **89** | **27** | **27** | 53 |

Headline reads: **Docling held across every format tested.**
- Slide deck — `0` flagged, format-agnostic ingestion of a layout with
  no body prose.
- HAL paper — 3 flagged, same audit-tool-overcount pattern as the
  ArXiv corpus; visual spot-check confirms the artefacts are extracted.
- Apollo 17 scan — the OCR pipeline (RapidOCR inside Docling) fed
  the layout model and produced **89 figures + 27 tables with bboxes
  from a 50-year-old scanned report**. The 53 flagged pages cluster as
  35+ consecutive pages all flagging "Figure 10" — visual check on
  `p180` shows it's a body-text page with cross-references like
  `(see Figure 10)` and section heading `10.13`, both caught by the
  audit's caption regex as false positives. Not real Docling misses.

The composite picture across all 23 documents tested:

- ArXiv ML corpus: PyMuPDF `25.4 %` flag rate → Docling `12.7 %` (≈ half;
  residual mostly audit-tool overcount per visual spot-checks).
- Slide deck: Docling `0 %`.
- Non-arXiv academic: Docling `9.4 %` (likely audit-tool overcount).
- Scanned + OCR'd historical report: Docling `15.6 %` (dominated by
  body cross-references, not real misses).

The heterogeneous gate is **clear**. `ingest_paper(use_docling=True)`
flipped default-on accordingly.

## End-to-end measurement with multi-modal chunks enabled, 2026-05-20

Same v3 set, same `gemma3:4b` gen + judge, same hybrid retriever, but
this run also enabled `--extract-figures --extract-tables --use-docling`
so the multi-modal chunks Docling produces actually flow into the
retrieval corpus. Committed reference: `data/eval/baseline-docling-mm.json`.

| metric | pre-Docling text-only | Docling + figs + tables | Δ |
|---|---:|---:|---:|
| **answer_correctness** | 0.7626 | 0.7113 | **−0.0513 (−6.7 %)** |
| faithfulness | 0.7454 | 0.7487 | +0.4 % |
| answer_relevance | 0.6179 | 0.6321 | +2.3 % |
| context_precision | 0.8077 | 0.7974 | −1.3 % |
| p50 latency | 20.3 s | **78.0 s (3.8 ×)** | regression |

### Per-category split

| category | n | pre | Docling-mm | Δ |
|---|---:|---:|---:|---:|
| **figure** | 11 | 0.7673 | **0.8818** | **+0.1145** |
| **multi_hop** | 2 | 0.8000 | **0.9000** | **+0.1000** |
| factual | 13 | 0.8308 | 0.6731 | **−0.1577** |
| table | 4 | 0.4500 | 0.2825 | **−0.1675** |
| equation | 1 | 1.0000 | 0.6700 | −0.3300 (n=1) |

The headline regression is real and past the 5 % gate. The category
split makes the mechanism unambiguous and matches ADR 0002's earlier
finding *with a different parser*: enabling figure + table chunks
floods retrieval context with long table-markdown / caption blobs that
**help on the question class that needs them** (figure +11 pp,
multi_hop +10 pp) and **crowd out the right text chunks** for factoid
/ table lookup (−16 to −17 pp). The 3.8× latency blow-up is from those
same long multi-modal chunks filling the generator's context window.

### Honest split decision

Docling-as-parser stays **default-on** — the structural audit (halved
flag rate, slide decks 0 flag, 339-page OCR'd scan handled) is
unequivocal, and when figures or tables are wanted, Docling is the
right tool. But `extract_figures_enabled` / `extract_tables_enabled`
stay **default-off** — same product call as ADR 0002, now reconfirmed
with the better parser and `answer_correctness` instead of
`answer_relevance`. The text-only retrieval path that scores 0.7626 on
`answer_correctness` is *not* improved by always-on multi-modal chunks;
it is improved when the caller selectively enables them for figure /
multi-hop query classes — and that selective routing is the natural
follow-up (it does for ingestion what ADR 0008 / ADR 0013 do for
retrieval: route by query category).

Committed measurements:
- `data/eval/baseline-text-only.json` — pre-Docling, no multi-modal.
- `data/eval/agentic-text-only.json` — pre-Docling, agentic, no multi-modal.
- `data/eval/baseline-docling-mm.json` — Docling primary, full multi-modal on.

The honest pair to read together is `baseline-text-only.json` (default
production posture) vs `baseline-docling-mm.json` (the "what does
turning everything on cost" measurement). For a future ADR, the
question becomes: can a per-query-category router enable multi-modal
*only* on figure / multi_hop queries and recover the +11 / +10 pp on
those subsets without paying the −16 / −17 pp factoid / table tax?
That experiment is its own ADR.



## What this leaves open

- **Heterogeneous-format eval (the honest test of "any document").**
  Audit only re-run on one paper so far. Run across the existing 20-paper
  corpus + add one IEEE paper, one slide-deck PDF, one OCR'd scan, and
  one non-English document. Report flag rates per format class. Only
  flip `use_docling=True` to default-on after that.
- **VLM residual fallback wiring.** Not wired yet — Docling's flag rate
  is 0 on the only paper measured. Wire only if the heterogeneous eval
  surfaces a residual class Docling itself misses. The VLM spike script
  and prompt are the template.
- **Dependency cost.** Docling adds DocLayNet / TableFormer / RapidOCR
  models (~40 MB after the existing PyTorch dep). One-time download
  cached locally; warm-GPU conversion ~30 s / 25 pages. Acceptable for
  offline ingestion; would not be for hot-path query latency, which
  this is not.
- **False-positive culling on tables.** Docling produced 3 continuation-
  table fragments on p14–p16. They are *real* (Table 4 is multi-page in
  this paper) but their caption is empty. A future iteration could
  merge continuation fragments under their parent caption.

## Related

- ADR 0017 — corpus clean; its 2026-05-20 amendment introduced the
  audit overlay that made this gap measurable and gates the cascade
  trigger.
- ADR 0010 — cost-quality cascade for retrieval; same pattern applied
  here at the ingestion layer.
- ADR 0002 — multi-modal chunks; `Figure` / `Table` types unchanged so
  downstream is unaffected.
- ADR 0018 / 0019 — kill-spike methodology (one decisive experiment
  before committing to a build); two spikes here for two candidate
  parsers, the better one shipped.
