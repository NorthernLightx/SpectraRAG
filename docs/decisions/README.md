# Architecture decisions

The decision log for SpectraRAG: what was chosen, why, and what the measurement
showed. Records are amended rather than rewritten, so the long ones carry their
history. Several document bets that were measured and dropped; those are kept on
purpose.

## Start here

These five explain the system as it is served.

- [0032](./0032-routing-is-a-cost-lever.md): Every query runs both search legs and fuses them. On MMDocIR that beats a per-query router by 0.16 recall@10 at the same latency, and the reader, not retrieval, limits the answers.
- [0028](./0028-persisted-visual-index-cpu-serve.md): The visual leg's page vectors are built once on a GPU and ship in the image; the CPU server only encodes the question.
- [0033](./0033-one-reader-context.md): The chat, `/answer` and the eval build the reader's input in one place, so the eval measures what users get.
- [0014](./0014-api-reranker-parity.md): The API and the eval build retrieval from one config, and equal fingerprints in `/health` and the run files mean identical retrieval.
- [0031](./0031-provider-menu-byok-or-ollama.md): Answers are generated in the browser on the visitor's own provider (an OpenRouter key or local Ollama); the hosted page also answers free through Gemini.

## Also shipped

- [0004](./0004-phase3-visual-retrieval.md): Visual retrieval with ColQwen2 page embeddings
- [0008](./0008-phase32-routing.md): Per-query routing between text and both legs, still available as a mode
- [0009](./0009-region-level-evidence.md): Figures and tables as chunks that carry their page region
- [0011](./0011-figure-caption-aggregation.md): One figure per caption, so multi-panel figures stay whole
- [0017](./0017-corpus-clean-structure-aware-chunking.md): Structure-aware chunking with page furniture stripped (19 % smaller corpus, no content lost)
- [0020](./0020-vlm-as-parser-ingestion-fallback.md): Docling parses the PDFs, with a vision-model parser kept as the fallback
- [0021](./0021-docling-text-chunker.md): Docling chunks the text too (+8 % answer correctness)
- [0022](./0022-figure-role-classification.md): Each detected picture is tagged figure, decoration or unlabeled; decorations stay out of retrieval and the gallery
- [0023](./0023-visual-favoring-fusion-weight.md): The visual fusion weight is set per corpus; the text-heavy demo keeps 1
- [0034](./0034-recordings-as-time-segment-documents.md): Talks and audio become documents whose pages are time segments
- [0029](./0029-runtime-document-upload.md): PDF upload into the live corpus, behind a flag for local use
- [0030](./0030-frontend-backend-split.md): The frontend ships separately from the API image
- [0005](./0005-phase4-deploy-and-observability.md): Deploy and observability scaffold (logging, request ids, tracing)
- [0006](./0006-ooc-refusal-gate.md): Out-of-corpus refusal on the rerank score, off by default

## Measured, not made the default

- [0001](./0001-contextual-retrieval.md): Contextual retrieval: no gain once results are reranked
- [0003](./0003-phase22-query-expansion.md): Query expansion (LLM rewrite, HyDE): wins on some queries cancel losses on others
- [0012](./0012-reranker-swap-investigated.md): Reranker swap: the incumbent stays
- [0016](./0016-context-neighbourhood-expansion.md): Pulling in neighbouring chunks: within noise
- [0018](./0018-graphrag-tier-construction.md): GraphRAG: rejected after a small trial build
- [0019](./0019-agentic-retrieval-tier.md): Agentic retrieval: within noise overall, kept opt-in
- [0025](./0025-structured-extraction-augments-reading.md): Tables and charts transcribed to text for the reader: a gain in the right direction, not significant
- [0024](./0024-route-by-fit-page-selector.md): Feeding the whole document when it fits: opt-in for questions about one paper
- [0026](./0026-dci-evaluated-experimental-opt-in.md): An agent that greps the raw corpus (DCI): experimental opt-in
- [0010](./0010-cost-quality-cascade.md): Cascade routing by confidence: opt-in; the eval fixes it came with are on
- [0002](./0002-phase2-multimodal-chunks.md): Figure and table chunks from PDF extraction: opt-in at the time
- [0015](./0015-routing-fair-eval-set.md): An eval set meant to judge routing fairly: built, failed its validation

## Superseded

- [0013](./0013-routing-is-the-accuracy-lever.md): Routing as the accuracy lever, superseded by 0032
- [0027](./0027-keyless-demo-chat.md): Keyless chat through a server-held key, superseded by 0031
- [0007](./0007-phase31-corpus-expansion-and-hybrid-fusion.md): Corpus expansion and an offline fusion re-evaluation that led to 0008
