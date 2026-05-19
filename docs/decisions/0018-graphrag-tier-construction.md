# ADR 0018 — GraphRAG tier: rejected on measured kill-spike

**Status:** **Rejected** on a cheap kill-spike (M2). The construction core
(S1.1 extraction + S1.2 graph build / communities + S1.3 minimal community
summaries) stays in tree as an opt-in artifact since the LLM `is_reference_list`
filter has standalone value (ADR 0017 deferred bib removal here). S1.4–S1.6
(persistence, GraphRetriever, full 20-paper build) are *not* built. The
hybrid baseline + agentic tier (ADR 0019, Step 2) is the direction.
**Date:** 2026-05-19

## Context

Step 1 of the agentic + graph revamp: a GraphRAG retrieval tier alongside the
untouched hybrid pipeline (tier, not replacement). This ADR is opened *before*
the verdict on purpose: the repo's prior is that plausible techniques come
back within noise on this corpus — reranker rejected (0012), routing a
measurement artifact (0013 → 0015, +0.0066 vs a +0.05 gate), context
expansion inconclusive (0016, +0.0375 within ±0.07 noise at n=40). Full
Microsoft-style GraphRAG is a strictly larger and more expensive bet than any
of those. So the construction choices are recorded here, the measurement
section is deliberately empty, and the build is gated on a cheap signal.

## Construction decisions (S1.1 / S1.2)

- **Extraction.** One LLM call per clean chunk → entities + relations
  (`src/ingestion/graph_extract.py`). `is_reference_list` is the bibliography
  filter ADR 0017 deferred: moved from a lexical heuristic that provably
  cannot separate a reference list from citation-dense prose (ADR 0017's
  intro-vs-bib calibration) to an LLM judgment. **This is unmeasured** — that
  the LLM actually separates them on this corpus is an assumption the spike
  must check, not a result.
- **Entity merge by lowercased name.** Known loss: a case-only-distinct
  acronym pair (`mAP` the metric vs `MAP` the method) collapses into one node
  with an arbitrarily tie-broken type. Accepted for a 20-paper demo *pending*
  a measured occurrence count on this corpus; pinned by
  `test_case_only_distinct_acronyms_collapse_known_loss` so any change is
  deliberate.
- **Communities: networkx Louvain, recursive for 2 levels** — *not*
  Microsoft GraphRAG's hierarchical Leiden + map-reduce community answering.
  Leiden needs an igraph/graspologic dependency not justified on a demo
  corpus. This is **"GraphRAG-style", not "Full GraphRAG"**; earlier plan
  language overclaimed and is corrected here.

## Measurement — kill-spike (M2), 2026-05-19

3 diverse papers (2604.22753v1 scaling-laws / 2604.28173v1 S-JEPA
skeletal-action / 2604.28192v1 LaST-R1 VLA), 250 clean chunks, gemma3:4b
local model both arms, same answer prompt, only retrieval differs. Control
arm: in-process BM25 over the same chunks. 34 min total LLM time:
`scripts/experiments/graphrag_spike.py`, results
`data/eval/ingestion/spike.md` + `spike-graph.json`.

### Graph-ingestion metrics (the early structural signal)

| metric | value |
|---|---|
| chunks | 250 |
| `is_reference_list` flagged | **53 (21.2%)** |
| zero-extraction chunks | 0.8% |
| entities / chunk | 4.29 |
| relations / chunk | 3.14 |
| nodes | 787 |
| edges | 672 (avg degree ≈ 1.7) |
| isolates | 5.8% |
| communities | 240 (189 L0 + 51 L1) |
| singleton communities | **19.2%** |

A graph that fragmented (degree ~1.7, 1/5 singleton communities) means
community summaries are paper-localised and thin — and that is the precise
structural precondition under which "global" search over reports cannot
beat passage retrieval. The structural signal predicted the verdict before
the side-by-side ran.

### Side-by-side verdict (8 global-synthesis queries)

BM25-RAG with the same LLM **wins 5 of 8 queries**, GraphRAG-global wins
1, 2 are wash. On the very class of question GraphRAG is supposed to win
(cross-paper synthesis), BM25 gives the LLM more cross-paper material to
work with and produces *richer, more accurate, more cross-document*
answers most of the time. The one GraphRAG win (Q6 "contributions") is
real — it is the only answer in the spike that genuinely spans all 3
papers — but it is outweighed by:

- **Hallucination on Q1** ("shared problem domains"): GraphRAG asserted a
  fake shared theme ("mental health: schizophrenia and bipolar disorder")
  that none of these papers is about. A confident wrong cross-paper claim
  is a worse failure mode than BM25's narrowness.
- **Entity-typing errors on Q5** ("datasets"): GraphRAG listed MPJPE (a
  metric), "All Data" (a method name), SMPL (a model) as datasets — the
  closed-vocabulary extractor mislabels under gemma3:4b.
- **Narrower coverage** on Q2/Q3/Q5/Q7/Q8: GraphRAG fixated on one paper's
  community summaries while BM25's 6 top chunks span the corpus.

### Cost objection validated

34 min for 250 chunks (~6.4 s/chunk effective at concurrency 4). The full
20-paper corpus would be ~4.5 h of extraction alone, before community
summarisation and global answering. The cheap spike saved that spend.

## Decision

Reject GraphRAG-as-built on this corpus, on the same honest-measurement
grounds as ADR 0013 (routing artefact) and ADR 0016 (context expansion
within noise). The construction core stays in tree opt-in because:

- the LLM `is_reference_list` filter has standalone value (ADR 0017
  deferred bibliography removal here; the spike confirmed it fires at
  21.2% — precision still needs a human-labelled check),
- the extraction + graph code is mypy-strict, ruff-clean, 21 unit tests,
  no maintenance cost while dormant.

S1.4 (Qdrant entity embeddings), S1.5 (GraphRetriever), S1.6 (20-paper
build + measurement) are **not built**. Pivot to Step 2 (agentic retrieval
over the existing hybrid) — ADR 0019.

## What this teaches

- The repo's prior held: ADRs 0012, 0013/0015, 0016 are all "plausible
  technique, within noise on this corpus." 0018 makes four.
- The early structural signal from the ingestion scorecard (sparse graph,
  many singleton communities) predicted the verdict in ~3 min of compute,
  before the 31 min of LLM work confirmed it qualitatively. The
  user-asked-for "dynamics + transparency early enough" is exactly what
  bought the kill confidence.
- A stronger extractor (cloud Sonnet/Opus) + a denser corpus *might*
  change this, but those are different preconditions, not the question
  this spike was authorised to answer.

## Related

- ADR 0016 — honest-metric requirement and the within-noise pattern this ADR
  refuses to repeat blindly.
- ADR 0017 — corpus clean; its "answer-quality delta measured in Step 1"
  line is corrected (the metric did not exist).
- ADR 0013 / 0015 — the look-promising-then-evaporate precedent.
