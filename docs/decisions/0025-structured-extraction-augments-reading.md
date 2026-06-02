# ADR 0025 — Structured extraction (tables/charts → text) augments the reader; a local 1.2B VLM matches the cloud ceiling

**Status:** **Directional positive, demo-scoped — not significant at n.** Feeding
offline-extracted structured text (tables/charts transcribed to markdown)
*alongside* the page image lifts answer accuracy on the post-retrieval failure set
**+0.12 (strict 0.16→0.30, fair 0.40→0.51)**, but on only **~6 discordant pairs
(sign-test p≈0.07–0.13)** — directional, not significant; two of the raw wins were
on cards where extraction had *failed* (reader noise, not the lever). The extractor
is **MinerU2.5-1.2B running locally on the 8 GB GPU** — extraction recall **0.591
on the 22 reliable structured-object cards, a TIE with cloud qwen3-vl-235b (0.591),
above OCR-only docling (0.500)**. So a free local 1.2B *matches* the cloud 235B; it
does not beat it (an earlier 0.625-vs-0.542 "+3" read was inflated by single-digit
gold-value matcher false positives — see Caveats). The operator-level backend
selector (`RAG_EXTRACTOR_BACKEND`, `src/extraction/`) is wired; the ingest-time
consumer that feeds the reader is gated on the lever clearing significance.
**Date:** 2026-06-02

## Context

The binding end-to-end constraint on this corpus is **post-retrieval
perception**, not retrieval (the 2026-05-29 agenda: gold page is in front of the
reader ~72 % of the time, but the reader converts only ~46-48 %). Every lever
that tried to make the *reader* smarter at query time was refuted — agentic
decomposition (ADR 0019), crop/zoom "thinking with images" (reset ADR), bigger
VLM, higher DPI. The 2025-26 frontier (SIMPLOT, TALENT, DocVLM) points a
different way: don't make the reader re-read pixels — **hand it the structured
data, extracted once offline where you can spend compute and verify.**

## What was measured

**(1) Does structured-text-alongside-image help the reader?** Paired A/B on the
43 gold-present post-retrieval failures (`struct_extract_probe.py`): same reader
(gemma-4-31b:free), same pages; the only difference is whether tables/charts
transcribed by a strong VLM are prepended to the prompt.

| metric | page image only | + structured text | Δ |
|---|---|---|---|
| strict ACC | 0.163 | 0.302 | +0.14 |
| fair ACC | 0.395 | 0.512 | +0.12 |
| table subset | 0.222 | 0.444 | +0.22 |
| figure subset | 0.167 | 0.267 | +0.10 |

**Power, stated honestly.** n=43 but only ~8 *discordant* pairs (7 strict wins / 1
loss, 6 fair / 1). An exact sign test is **p≈0.07 (strict), p≈0.13 (fair)** —
*directional, not significant at 0.05*. Two of the strict wins are on cards where
the extraction actually **failed** (`__extract_failed__` placeholder fed), i.e.
reader nondeterminism, not the lever; dropping them gives 5/1 (p≈0.22). The paired
design does make the delta common-mode-noise-robust against bad-gold/format
artifacts (both arms score 0 there), and the +0.12 does clear the ±0.08 per-card
judge band — but clearing that band is necessary, not sufficient, at this flip
count. Read this as a promising lever to power up, not a settled win (cf. the
agentic probe shelved at the same p≈0.13).

**(2) Which extractor — efficiently and near/above SOTA?** Extraction recall (gold
value present in the extraction = a label-free TEDS proxy; matcher hardened with
word-boundary matching, single-char golds excluded as unreliable) on the 22
reliable structured-object cards (`extract_bench.py`, `extract_compare.py`):

| extractor | recall (n=22 reliable) | head-to-head vs qwen | nature |
|---|---|---|---|
| **MinerU2.5-1.2B (local)** | **0.591** | **tie (+1, −1)** | local, free, fits 8 GB, reads charts |
| qwen3-vl-235b (cloud) | 0.591 | — | cloud, slow, the prior extractor |
| docling (local OCR) | 0.500 | −2 | local, fast, **chart-blind** |

So a free local 1.2B model **matches** the cloud 235B (not beats — an earlier
"0.625 vs 0.542, +3/−1" read was inflated: 2 of those 3 "wins" were single-digit
gold values ("9", "4") matching incidentally inside other tokens, fixed by the
word-boundary matcher). External corroboration (frontier survey): MinerU2.5 sits at
**88-92 table TEDS**, near the ~94 SOTA (GLM-OCR, PaddleOCR-VL) and at/above
qwen3-vl-235b's 86 — i.e. near-SOTA on-box. The matcher recall here is a relative
proxy, not TEDS; the absolute SOTA framing rests on those external bars.

**The mechanism that decides the architecture:** OCR parsers (docling,
MinerU-pipeline) transcribe tables but are **chart-blind** — they emit charts as
`![](image.jpg)` with no data. Only a VLM reads chart data points. Our hard cards
are mostly charts, so the extractor *must* be VLM-based. MinerU2.5-1.2B is the one
option that is simultaneously VLM-based, chart-capable, and 8 GB-feasible.

## Decision

1. **Adopt structured-extraction-augmentation** as a measured generation-side
   lever: transcribe a page's tables/charts to text offline, feed that text
   alongside the page image at answer time.
2. **Extractor = MinerU2.5-1.2B served locally** via `mineru-api
   --enable-vlm-preload` (model loaded once; pages POSTed to `/file_parse`). This
   is the efficient form — the per-page CLI reloads the model each page and is
   not viable for batch.
3. **Ship the operator selector now; gate the ingest-time consumer on the
   lever.** The backend is an operator/deploy choice, so it lives in `Settings`
   as `RAG_EXTRACTOR_BACKEND={none|qwen-cloud|mineru-local}` (default `none`) with
   a small factory in `src/extraction/` returning the configured backend or
   `None`. This is how production doc-extraction systems expose it — a deploy
   knob, not a per-query toggle. The heavier ingest-time wiring (run the extractor
   at ingest, cache the structured text in `chunk.metadata`, thread it into the
   prompt) is deliberately *not* built yet: it's production-grade effort that
   should wait until the lever clears significance (the QA-lift is still
   directional, p≈0.07–0.13). The probe + benches stay in `scripts/experiments/`.

## Honest caveats

- **"Efficient" means cost/privacy, not latency.** MinerU2.5-1.2B via transformers
  (no vllm — vllm doesn't fit 8 GB cleanly) is ~1-3 min/page. It is free, local,
  and private, but not fast per page. A vllm/sglang host on a bigger GPU is the
  throughput path; on this box, offline-at-ingest (not per-query) is how you'd use
  it.
- **Recall is a proxy, not TEDS.** "0.591" is gold-value-present on the 22 reliable
  cards, not the official TEDS metric; the *relative* ranking is sound (identical
  hardened scorer across backends) and the *absolute* SOTA framing rests on the
  external TEDS bars, not this number.
- **Small n, matcher caveat.** 22 reliable structured-object cards (25 minus 3
  single-char golds the presence-matcher can't score reliably). MinerU vs qwen is a
  **tie (+1/−1)**, not a win — the earlier "+3" was single-digit matcher false
  positives, now fixed with word-boundary matching (`extraction_recall._present`).
- **QA-lift is directional, not significant.** ~8 discordant pairs, sign-test
  p≈0.07–0.13; 2 raw wins were on failed-extraction cards. Power it up (more cards,
  or a fixed extractor with no failures) before treating +0.12 as established.
- **The gold is ~25-30 % noisy** on the failure slice (a human audit was started,
  not finished). The paired A/B is robust to this; the absolute accuracies are not.

## Related

- ADR 0024 — route-by-fit (feed more pages); this ADR feeds *better-represented*
  pages. Complementary generation-side levers.
- ADR 0020 — VLM-as-parser at ingest produced prose captions for retrieval; this
  is structured (not prose) extraction fed to the *reader* (not retrieval), which
  is why ADR 0020's retrieval-crowding regression doesn't apply.
- The reset agentic/thinking work (crop-zoom, DPI) — refuted query-time
  reader-smartening; this ADR is the representation-change alternative that worked.
