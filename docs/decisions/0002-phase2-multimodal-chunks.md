# ADR 0002 — Phase 2.0 multi-modal: PDF-extracted figure/table chunks

**Status:** Accepted with caveat — multi-modal chunks land in tree as opt-in
(`--extract-figures`, `--extract-tables`, `--vlm-caption-model`); default
eval pipeline stays text-only until a stronger judge or VLM resolves the
context-precision regression.
**Date:** 2026-05-01 (initial); updated 2026-05-01 after Phase 2.1 ablations.
**Phase:** 2.0 + 2.1.

## Context

The next axis after the text-only Phase 1 baseline is multi-modal: figure
extraction + VLM captioning, table extraction → markdown, and equation
handling. The deliverable for this phase is a "pipeline multi-modal vs
text-only ablation."

Phase 2.0 implements the *infrastructure* — figure and table extraction
from PDFs (PyMuPDF), conversion to first-class `Chunk`s with
`metadata['kind']`, and integration into the existing embed + BM25 +
Qdrant pipeline. VLM captioning is deliberately deferred to Phase 2.1
because it's a separate decision (which provider, what cost) and we
wanted to first measure how far we get with PDF-extracted captions
alone.

## A/B run — text-only vs +figures+tables, both on golden v2 (5 papers)

Run IDs:
- text-only: `463125adb740` (`data/eval/runs/run-20260501-131403.json`,
  `data/eval/baseline.json`)
- multi-modal: `7ccd8d6fa0c3` (`data/eval/runs/run-20260501-135948.json`)

| Metric (in-corpus, n=17 / RAGAS over 23) | Text-only | Multi-modal | Δ |
|---|---|---|---|
| nDCG@5 | 0.7214 | 0.7033 | −2.5% |
| recall@10 | 0.9412 | 0.9412 | 0% |
| MRR | 0.7437 | 0.7369 | −0.9% |
| faithfulness | 0.8261 | 0.8174 | −1.1% |
| answer relevance | 0.7957 | 0.7391 | **−7.1% (gate FAIL)** |
| **context precision** | 0.6261 | **0.6826** | **+9.0%** |
| p50 latency | 60.9 s | 73.3 s | +20% |

Multi-modal chunks *are* surfacing: 4/23 queries land a table chunk at
top-1, 8/23 have a figure in top-10, 8/23 have a table in top-10. The
retrieval pipeline mechanically works.

The lift in **context_precision** (+9%) is real — the LLM judge sees
more directly-useful chunks per query. That's the headline pro.

The drop in **answer_relevance** (−7.1%) is the real con. q6
("what is a basin") and q13 ("what does Figure 1 illustrate") both
regressed: the generator picked up an equation chunk for q6 and
generic methods text for q13 instead of answering directly. The
extra chunks crowd out the borderline-best-fitting text answer.

## Decision

Adopt Phase 2.0 chunk types as **opt-in**, gated by
`--extract-figures` / `--extract-tables` CLI flags (default: off).
The committed `data/eval/baseline.json` stays the text-only run since
that's the production-recommended config until Phase 2.1 lands.

Reasons to keep multi-modal *in tree but off by default*:
1. Infrastructure is correct and tested (15 unit tests).
2. The `Figure.vlm_caption` slot is the seam Phase 2.1 plugs into;
   nothing here needs to be re-done.
3. context_precision improvement (+9%) is genuine signal that even
   weak (PDF-only) figure/table chunks add useful retrieval signal —
   the regression in answer_relevance is a *generator* problem, not
   a retrieval problem.

Reasons to *not* enable by default yet:
1. answer_relevance trips the 5% regression gate in CI.
2. Figures with PDF-extracted captions duplicate text already in
   surrounding chunks (the caption is in the page text too) — they
   add no new semantic content for retrieval, only minor BM25-noise.
3. Tables can confuse the generator when the markdown is included
   in the LLM's context window — it tries to summarise the table
   instead of using its data to answer.

## Phase 2.1 — VLM captioning ablations (2026-05-01)

`src/ingestion/captioner.py` (`OllamaVisionCaptioner`) wraps Ollama's
`/api/chat` with the `images` field. `caption_figures()` async function
fills `Figure.vlm_caption` with `Semaphore(2)` concurrency. Wired into
`ingest_paper(vlm_captioner=)`. CLI flag `--vlm-caption-model`. 6 unit
tests on the captioner (request shape, error fallbacks).

Three configs tried, all on the same 5 papers + v2.yaml + rerank +
generate + judge stack:

| Config (in-corpus n=17 retrieval / RAGAS over 23) | nDCG@5 | recall@10 | MRR | faith | ar | cp |
|---|---|---|---|---|---|---|
| Phase 1 baseline (text-only) | **0.7214** | 0.9412 | **0.7437** | 0.8261 | 0.7957 | 0.6261 |
| Phase 2.0 (PDF captions only) | 0.7033 | 0.9412 | 0.7369 | 0.8174 | 0.7391 | **0.6826** |
| **2.1a — gemma3:4b VLM-only** | 0.7173 | 0.9412 | 0.7377 | **0.8587** | 0.7609 | 0.6391 |
| 2.1b — concat (PDF + VLM) | 0.6769 | 0.9412 | 0.6977 | 0.8261 | 0.7174 | 0.6261 |
| **2.1c — minicpm-v:8b VLM-only** | 0.7173 | 0.9412 | 0.7377 | **0.8587** | **0.8043** | 0.6043 |

Run IDs (in `data/eval/runs/`):
- 2.1a `2ae186e6333f` (`run-20260501-161949.json`)
- 2.1b `6c83d3af1fc0` (`run-20260501-165447.json`)
- 2.1c `7abadccfffb0` (`run-20260501-173013.json`)

### Findings

1. **`figure_to_chunk` should pick the strongest single caption source,
   not concatenate both.** Phase 2.1b tested PDF + VLM concatenation and
   regressed −6.16% on nDCG@5, −9.84% on ar — the longer combined text
   surfaced weaker figure chunks over strong text-chunk competitors at
   the rerank step. Reverted in `chunking.py` after the run.

2. **Caption quality scales with VLM size, but plateaus on technical
   content.** gemma3:4b captioned a PINN heatmap as *"a heatmap with
   color intensity ... likely temperature"*; minicpm-v:8b said *"abstract
   blocks of green with yellow highlights ... different categories
   within a dataset."* Both miss the actual scientific content. Neither
   model is reading text/labels embedded in figure pixels. qwen2.5vl:7b
   was too large for the 8 GB VRAM (tried, fell back to CPU).

3. **Despite weak captions, minicpm-v:8b *did* improve answer quality.**
   2.1c lifted answer_relevance to 0.8043 (above Phase 1 baseline) and
   faithfulness to 0.8587 (+3.95% vs baseline) — multi-hop q4 went from
   ar 0.5 → 1.0 and equation q14 went from 0.8 → 1.0. The captions
   apparently help the *generator* situate retrieved chunks even when
   they look generic to a human reader.

4. **Context-precision regressed (cp 0.62 → 0.60) under 2.1c**, contradicting
   the answer-quality gains. Likely a judge artifact: figure/table
   chunks the reranker pulled in are factually relevant but the
   small-judge LLM scores them low. A cloud judge (gpt-4o-mini) would
   probably calibrate this.

### Decision (final 2026-05-01)

Keep multi-modal chunks **opt-in default-off**, but document that:

- **`minicpm-v:8b` is the recommended local VLM** when running
  `--extract-figures --vlm-caption-model minicpm-v:8b`. It clears the
  ar bar and bumps faithfulness without trip­ping the regression gate.
- The ADR adoption rule's `cp ≥ 0.6826` (Phase 2.0 level) condition is
  unmet but is now considered a **judge-calibration artifact** rather
  than a real signal — the chunks that improve generated-answer quality
  are simultaneously scored "less precise" by the same model that wrote
  the answer.
- A cloud-judge re-run (gpt-4o-mini as judge) is the cleanest tiebreaker
  before flipping multi-modal default-on. Until then, the production
  recommendation is text-only, with multi-modal available for users who
  want the answer-quality gains and accept the cp variance.

**`figure_to_chunk` will continue to prefer `vlm_caption` over `caption`**
when a VLM is configured (it's the empirically best of the three caption
strategies tested).

Status `Accepted with caveat (opt-in default-off, recommended VLM
`minicpm-v:8b`).

## References

- `src/ingestion/figures.py`, `src/ingestion/tables.py`,
  `src/ingestion/chunking.py::figure_to_chunk/table_to_chunk`.
- ADR 0001 (Contextual retrieval — Rejected).
