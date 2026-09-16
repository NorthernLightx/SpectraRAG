# Where SpectraRAG fits

A short read on what SpectraRAG does differently from other document-RAG
tools, and what it deliberately doesn't do. The goal isn't to win a feature
checklist; it's to be clear about the niche.

## The niche

SpectraRAG is a **multi-modal document RAG that treats a page's pixels as a
first-class retrieval signal, runs on a single 8 GB GPU, and measures every
change.** It targets PDFs whose answers live in figures, charts, and tables,
where a text-only index misses the page entirely.

## What's actually different

- **Visual retrieval is late-interaction over page images, not OCR-flattened
  text.** Most "multi-modal" RAG engines (RAGFlow, kotaemon, LlamaParse) do
  layout/OCR *well* and then index the extracted text. SpectraRAG keeps a
  ColQwen2 (ColPali-family) multi-vector index over rendered pages and scores it
  with MaxSim, so a chart with no useful text layer is still retrievable. On
  figure/chart pages that's the difference between finding the page and not.
- **A per-query router, with a closed-loop proof it captured the available
  lift.** A small classifier sends each query to the text leg or to text+visual.
  The repo measured the ceiling: oracle routing equals the shipped router on the
  benchmark (ADR 0013). Peers that always run one path (always-ColPali, or
  always-text) don't make (or measure) that decision.
- **An eval behind every change, including the negatives.** Committed golden
  sets, a >5% regression gate, and a wall of *measured* dead ends (GraphRAG lost
  to plain RAG, agentic decomposition hurt retrieval, rerankers were a wash) plus
  a scorer-bug audit that recovered ~0.11 of apparent accuracy. The discipline is
  the point as much as any single number.
- **Runs on consumer hardware, served on CPU.** ColQwen2 fits an 8 GB card for
  the offline page-encode, and the persisted index serves on a CPU-only box
  (ADR 0028). Most ColPali demos assume a bigger GPU or a managed cloud.
- **BYOK privacy.** Generation runs browser-direct with the visitor's own key;
  the server never sees it.

## What it deliberately doesn't do

This is a focused, single-maintainer project, not a platform. If you need these,
the tools in parentheses are the better fit:

- **Multi-user, accounts, shared collections** (kotaemon, Open WebUI, Onyx).
  SpectraRAG is single-corpus; runtime upload is local-only and flag-gated off on
  the demo (ADR 0029).
- **Enterprise connectors**: Slack, Drive, Confluence ingestion (Onyx, Quivr).
- **Billion-document scale** with quantized late-interaction indexes (Vespa's
  binary-quantized ColPali, Weaviate MUVERA). SpectraRAG holds the page index in
  one embedded Qdrant; that's right for a demo corpus, not a warehouse.
- **A turnkey hosted product.** The hosted demo is a fixed corpus with no upload;
  it showcases the retrieval behaviour, it isn't a SaaS.

## When to reach for it

Reach for SpectraRAG when the documents are figure- and table-heavy, the hardware
budget is a single consumer GPU, and you want a retrieval pipeline whose every
knob has a measured effect rather than a default you have to trust. Reach for one
of the alternatives above when you need multi-tenancy, connectors, or scale.

## Sources

The landscape read behind this page: RAGFlow, kotaemon, Morphik, the ColPali
paper and the byaldi/ColQwen libraries, Vespa's and Weaviate's late-interaction
work, Onyx, and PaperQA2. See each project's own docs for current capabilities.
This page describes mechanics, not popularity, because stars and feature lists
move faster than a committed file should claim to track.
