# Architecture decisions

The decision log for SpectraRAG: what was chosen, why, and what the measurement
showed. Several records document bets that were measured and dropped; those are
kept on purpose.

- [0001](./0001-contextual-retrieval.md): Contextual retrieval as the parser-robustness layer
- [0002](./0002-phase2-multimodal-chunks.md): Multi-modal: PDF-extracted figure/table chunks
- [0003](./0003-phase22-query-expansion.md): Query expansion (LLM rewrite + HyDE + combo)
- [0004](./0004-phase3-visual-retrieval.md): Visual retrieval (ColQwen2 / ColPali-style)
- [0005](./0005-phase4-deploy-and-observability.md): Deploy + observability scaffold
- [0006](./0006-ooc-refusal-gate.md): OOC refusal gate (rerank-score threshold)
- [0007](./0007-phase31-corpus-expansion-and-hybrid-fusion.md): Corpus expansion + golden v3 + offline hybrid re-evaluation
- [0008](./0008-phase32-routing.md): Per-query routing (text-only vs hybrid)
- [0009](./0009-region-level-evidence.md): Region-level evidence (figures + tables as first-class chunks with bbox)
- [0010](./0010-cost-quality-cascade.md): Cost-quality cascade routing + eval methodology hardening
- [0011](./0011-figure-caption-aggregation.md): Figure caption aggregation
- [0012](./0012-reranker-swap-investigated.md): Reranker swap investigated; incumbent kept (premise falsified)
- [0013](./0013-routing-is-the-accuracy-lever.md): Routing is the accuracy lever; LLM-classifier router shipped (superseded by 0032)
- [0014](./0014-api-reranker-parity.md): Wire the reranker into the API retrieval path
- [0015](./0015-routing-fair-eval-set.md): A routing-fair evaluation set (`robust-v1`)
- [0016](./0016-context-neighbourhood-expansion.md): Context-neighbourhood expansion: built, inconclusive, not shipped
- [0017](./0017-corpus-clean-structure-aware-chunking.md): Document-level structure-aware chunking
- [0018](./0018-graphrag-tier-construction.md): GraphRAG tier: rejected on measured kill-spike
- [0019](./0019-agentic-retrieval-tier.md): Agentic retrieval tier: within noise, kept opt-in
- [0020](./0020-vlm-as-parser-ingestion-fallback.md): Docling as primary parser; VLM-as-parser kept as residual fallback
- [0021](./0021-docling-text-chunker.md): Docling as the text-chunking source too
- [0022](./0022-figure-role-classification.md): Figure role classification at ingestion
- [0023](./0023-visual-favoring-fusion-weight.md): Visual-favoring weight for page-level RRF (config knob)
- [0024](./0024-route-by-fit-page-selector.md): Route-by-fit: feed the whole document when it fits context, else top-k RAG
- [0025](./0025-structured-extraction-augments-reading.md): Structured extraction (tables/charts → text) augments the reader
- [0026](./0026-dci-evaluated-experimental-opt-in.md): Direct Corpus Interaction (DCI): a text-IR method, shipped as an experimental opt-in
- [0027](./0027-keyless-demo-chat.md): Keyless demo chat through a caged server-side OpenRouter key (superseded by 0031)
- [0028](./0028-persisted-visual-index-cpu-serve.md): Persisted ColQwen2 page index for CPU serving
- [0029](./0029-runtime-document-upload.md): Runtime document upload (flag-gated, text-leg, incremental)
- [0030](./0030-frontend-backend-split.md): Split the frontend off the backend image
- [0031](./0031-provider-menu-byok-or-ollama.md): Provider menu: generation on the visitor's own provider (OpenRouter BYOK or local Ollama)
- [0032](./0032-routing-is-a-cost-lever.md): Routing is a cost lever, not an accuracy lever (supersedes 0013)
- [0033](./0033-one-reader-context.md): One reader context for the chat, /answer and the eval
