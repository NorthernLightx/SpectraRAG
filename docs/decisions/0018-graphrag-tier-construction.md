# ADR 0018 — GraphRAG tier: construction (proposed, unmeasured)

**Status:** Proposed. The construction core (S1.1 extraction + S1.2 graph
build / communities) is landed and unit-tested. The GraphRAG-vs-hybrid
comparison is **not run and cannot be yet** — the metric the plan named does
not exist in the eval harness (see "Measurement"). No signal exists; this is
not a shipped technique. The ADR moves to Accepted or Rejected only after the
kill-spike below produces a number.
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

## Measurement

**Not run.** The plan named "answer-correctness vs `expected_facts` (the
ADR 0016 honest metric)" as the success criterion. That metric **is not
implemented**: `GenerationMetrics` has no `answer_correctness` field,
`expected_facts` is a `GoldenQuery` field no code in `src/` consumes, and
`baseline.json` does not contain it. ADR 0016 only ever used a throwaway
`scripts/experiments/study_context.py` harness with a `+0.03` bar it itself
diagnosed as too crude. Citing this metric as if it existed (in ADR 0017 and
the Step-1 plan) was an unverified assertion; ADR 0017 is amended to correct
it.

Prerequisites before any 20-paper build (S1.4–S1.6):

1. A real `answer_correctness` judge metric in `src/eval` (not a
   `scripts/experiments` one-off), wired into `GenerationMetrics`, with a
   committed hybrid baseline on v3's `expected_facts` queries and its
   resolvable-effect floor stated (0015/0016 show this corpus may not resolve
   a +0.05 at this n with a coarse judge).
2. A cheap kill-spike: graph + community summaries on **2–3 papers**
   (~200 calls, not ~2,000), 8–10 genuinely global/aggregative queries,
   GraphRAG-style answer read side-by-side against hybrid. Continue to
   S1.4–S1.6 only if there is a visible qualitative edge on the question
   class GraphRAG is *supposed* to win (global synthesis), since hybrid
   already saturates factoid lookup on this corpus.

## Decision

Keep S1.1 / S1.2: clean, low-risk construction, and the bib-filter has
standalone value. **Do not build S1.3–S1.6 or run the 20-paper extraction
until the spike returns a continue signal.** If the spike shows no edge,
GraphRAG is rejected here on the same honest-measurement grounds as 0013 /
0016, and that is a valid, on-brand outcome — not a failure to paper over.

## Related

- ADR 0016 — honest-metric requirement and the within-noise pattern this ADR
  refuses to repeat blindly.
- ADR 0017 — corpus clean; its "answer-quality delta measured in Step 1"
  line is corrected (the metric did not exist).
- ADR 0013 / 0015 — the look-promising-then-evaporate precedent.
