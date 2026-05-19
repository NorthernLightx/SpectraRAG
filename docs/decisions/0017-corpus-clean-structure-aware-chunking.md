# ADR 0017 — Corpus clean: document-level structure-aware chunking

**Status:** Accepted. Header stripping, soup filtering, and document-level
structure-aware chunking shipped (−19% corpus, zero content loss).
Bibliography removal and the answer-quality delta are deferred to Step 1
(GraphRAG ingest), where the corpus is re-ingested under an LLM pass anyway.
**Date:** 2026-05-19

## Context

Step 0 of the agentic + graph + multi-modal revamp: "no good ingestion, no
good RAG." Before building a knowledge graph on top, size and remove the
noise in the raw corpus, because a reference-list or figure-number-grid
chunk is wasted entity-extraction cost and pollutes the graph.

`scripts/experiments/quantify_corpus_junk.py` measured the raw corpus (the
pre-change `chunk_pages` over all 20 papers, 2,436 chunks):

- **Numeric/symbol soup ~8.5%** (207 chunks). Vector-drawn figures and
  tables leak into the PDF text layer as axis ticks and value grids
  ("2.127 2.126 2.134"); the raster figure path never sees them.
- **Bibliography ~10–12% of text** (measured floor 8.8%; the three largest
  papers' "References" headings are glued to body text by PyMuPDF's
  two-column extraction and went uncounted).
- Per-page character chunking also split a section at every page break,
  cut sentences mid-way, and let the running header ("Preprint. Under
  review.", journal lines) prefix the first chunk of nearly every page.

## What shipped

- `src/ingestion/clean.py`: running-header + page-number detection and
  stripping (PyMuPDF emits both as their own lines, so line-based removal
  is exact), and an `is_soup` content predicate.
- `src/ingestion/chunking.py` rewritten: pages are concatenated into one
  document with a char→page map, split on numbered/appendix/named
  headings, and windowed across page boundaries. A chunk now carries every
  page its text spans; soup windows are dropped.

Chunk-ids renumber by design. 14 new unit tests; full suite 517 green;
mypy strict and ruff clean. Result: **2,436 → 1,973 chunks (−19%) with no
real content lost** — appendices survive through the last page, verified
by chunk-dump eyeball and tests.

## What did not ship, and why

**Bibliography region-excision.** Built, then caught deleting
golden-anchored appendix content: a one-sentence appendix A sitting
between References and a citation-dense appendix B defeats every boundary
heuristic (heading, density, prose-collapse, entry-shape). Calibration
disproved the premise outright: no lexical feature separates a reference
list from citation-dense academic prose on this corpus. In 2604.22753v1
the introduction carries 6.7 year-citations per 1,000 chars while its own
reference list carries 3.3. This is the problem ML document-structure
parsers exist for. Deferred to Step 1: the GraphRAG pass already runs an
LLM over every chunk for entity extraction, a reference-list chunk yields
no entities, and the LLM is exactly the judge the lexical features could
not be. No new dependency.

**Golden re-anchor.** Chunk-ids changed, so v3's `relevant_chunk_ids` are
stale. An automated lexical re-anchor produced confident but wrong anchors
(a "benchmark size" query matched a performance-discussion chunk at 0.80
term recall). Shipping a machine-relabeled golden corrupts the eval's
ground truth, which `scripts/promote_candidates.py` exists to forbid.
Chunk-id retrieval metrics are being retired for the graph/agentic tiers
(ADR 0016 line of reasoning), so Step 0 is measured on the
chunk-id-robust generation metrics instead, and that old-vs-new delta is
folded into Step 1's eval — the corpus is re-ingested and run through the
LLM pipeline there regardless, so measuring it there is not redundant.

## Decision

Ship the three structural wins now. They stand on their own: coherent
cross-page chunks and 19% less noise into every downstream consumer,
independent of anything graph/agentic. Defer bibliography removal to the
Step-1 LLM filter and the answer-quality verification to Step-1's eval.
Status is honest: an enabling change, structurally verified, answer
quality pending Step 1.

## What this leaves open

- The bibliography (~10–12%) is still in the corpus until the Step-1 LLM
  filter removes it.
- The old-vs-new answer-quality delta (faithfulness, answer relevance,
  answer-correctness vs `expected_facts`) is measured in Step 1.
- v3 chunk-id anchors are not re-anchored; the chunk-id retrieval metric
  is deprecated for the new tiers.

## Related

- ADR 0016 — the honest-metric requirement (answer-correctness vs
  `expected_facts`) and the case for retiring chunk-id retrieval metrics.
- ADR 0002 — text-only attribution methodology; figure/table chunks come
  from a separate converter and are unaffected by this change.
- `scripts/experiments/quantify_corpus_junk.py` — reproducible
  before/after corpus measurement.
