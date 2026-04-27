# Multi-modal Paper RAG — Project Specification

> Handoff document for Claude Code. This is the source of truth for scope,
> stack, architecture, and what NOT to build. Read in full before scaffolding.

---

## 1. Pitch and framing

A production-grade RAG system for scientific papers (ArXiv ML corpus) that
**compares visual document retrieval (ColQwen2) against a traditional
multi-modal pipeline (text + figure captioning + table extraction)** on the
same corpus, wrapped in a full LLMOps stack.

The differentiator is **the comparison itself**, not "we handle figures." The
field has moved past pipeline approaches for visually-rich documents
(see ColPali / ViDoRe). Building both paths and benchmarking them on
QASPER-style queries is the defensible technical story.

The secondary story is the **production engineering**: provider abstraction,
prompt versioning, hybrid retrieval, eval harness with regression gates,
observability, IaC, CI/CD.

---

## 2. Core decisions

These are settled. Do not relitigate during scaffolding.

### LLM provider layer
- **Default provider: OpenRouter** (one API, one key, one bill, OpenAI-compatible).
- **Thin internal interface** (`LLMClient` Protocol) with three implementations:
  - `OpenRouterClient` — default for hosted models
  - `AnthropicClient` — direct SDK, used when prompt caching matters
    (long-context paper queries — verify cache-hit savings empirically)
  - `OllamaClient` — local models
- Do **not** rebuild LiteLLM. The interface is escape-hatch insurance, not
  a routing/fallback/cost-optimization layer. Roughly 50 lines.

### Retrieval paths (the headline)
Two retrievers, swappable via config, both evaluated on the same golden set:

1. **Pipeline path** (`rag/retrievers/pipeline.py`)
   - Text: chunked + dense embeddings + BM25 hybrid + cross-encoder rerank
   - Figures: vision-model captions, embedded as text alongside the figure ref
   - Tables: structured extraction (markdown), embedded as text
   - Equations: LaTeX preserved in chunks

2. **Visual path** (`rag/retrievers/visual.py`)
   - ColQwen2 (start with `vidore/colqwen2-v1.0`)
   - Render PDF pages to images, embed via `colpali-engine`
   - Late-interaction MaxSim scoring
   - Page images fed to multi-modal generator (no chunk reassembly)

### Embeddings
- **Default**: `BAAI/bge-m3` via Ollama (local, free, competitive, multilingual).
- **Alternative**: Voyage or OpenAI embeddings via OpenRouter-style call —
  config-switchable, but changing this triggers a full reindex.
- Treat embedding choice as a **deployment-time config**, not a runtime swap.

### Vector storage
- **Qdrant** for vectors (text path AND ColQwen2 multi-vectors — Qdrant
  supports late interaction natively; verify in scaffolding).
- **Postgres + pgvector** for metadata, search history, eval runs, user
  feedback. Not for primary retrieval.

### Reranking (text path)
- BGE reranker v2 (`BAAI/bge-reranker-v2-m3`) as default.
- Rerank top-50 from hybrid retrieval down to top-5.

### Generation
- Default: Claude Sonnet via OpenRouter (or direct Anthropic if caching).
- Visual path requires multi-modal generator. Must accept page images.

### Observability
- **Langfuse** for traces, prompt versions, evals.
- OpenTelemetry for infra-level metrics (latency, error rates).
- Token cost tracking via Langfuse generations.

### Infra
- FastAPI (async) + Uvicorn
- Docker + docker-compose for local
- Terraform → Azure Container Apps + Key Vault for cloud
- GitHub Actions: lint, type-check, unit tests, integration tests, Docker build

---

## 3. Architectural principles (extensibility without over-engineering)

The project must be **extendable for unknown future features** without
turning into a plugin framework. This is enforced by discipline, not
abstraction.

### Rules

1. **Clean module boundaries.** `ingestion/`, `rag/`, `llm/`, `eval/`,
   `api/` are independent. `rag/` does not import from `ingestion/` —
   they communicate via shared types in `src/types/`.

2. **Protocols only at real seams.** Define `Protocol` for:
   - `LLMClient` (chat + embed)
   - `Retriever` (query → ranked results with scores + provenance)
   - `Embedder` (texts → vectors) — separate from LLMClient
   - `Reranker` (query + candidates → reranked)
   - Nothing else. No protocol for prompts, guardrails, parsers, or
     metrics until a second concrete implementation forces it.

3. **Config over code.** Pydantic Settings + YAML for everything that
   varies between runs: model names, chunk size, top-k, rerank on/off,
   retriever choice. New experiments are config diffs.

4. **Pipeline = sequence of pure-ish functions.** No god objects.
   - Ingestion: `pdf → pages → chunks → embeddings → indexed`
   - Retrieval: `query → retrieved → reranked → context → answer`
   - Each step has typed inputs/outputs. Each step is independently
     testable. New steps slot in.

5. **Serializable pipeline stages.** Every intermediate artifact
   (chunks, retrieval results, contexts, generations) is a Pydantic
   model dumpable to JSON. You can reconstruct or fork any stage from
   disk. This is the cheapest thing that buys the most flexibility.

6. **Eval harness as forcing function.** Every retriever, chunker,
   prompt, and reranker registers itself with the eval framework. New
   variants run against the golden set automatically. Extensibility +
   regression safety in one mechanism.

7. **Architecture Decision Records.** `docs/decisions/NNNN-*.md` for
   any non-obvious choice. Future-self needs to know *why*.

### Anti-rules — explicitly do NOT build

- Plugin / registry systems with auto-discovery
- Event buses, hooks "for future use"
- Generic `Strategy` patterns where there is currently one strategy
- Configurable everything — leave hardcoded what is not actually varying
- Abstract base classes with one concrete implementation
- A custom retry / fallback / cost-routing layer (use OpenRouter or LiteLLM)
- A custom prompt templating engine (Jinja2 is fine; YAML files are fine)
- Caching abstractions before there's a measured cache problem

**Rule of three**: write concrete the first time, copy-paste with
variation the second time, abstract on the third. Two retrievers
(pipeline + visual) is the third — abstract that.

---

## 4. Project structure

```
multimodal-paper-rag/
├── README.md                  # Pitch, architecture diagram (Mermaid), quickstart
├── PROJECT_SPEC.md            # This file
├── Dockerfile
├── docker-compose.yml         # api + qdrant + postgres + langfuse + ollama
├── pyproject.toml             # uv / hatchling; ruff + mypy + pytest configured
├── .env.example
├── .github/
│   └── workflows/
│       ├── ci.yml             # lint, typecheck, unit tests
│       ├── integration.yml    # docker-compose up + integration tests
│       └── docker.yml         # build + push image
├── terraform/                 # Azure: Container Apps, Key Vault, Postgres
├── docs/
│   ├── architecture.md        # Detailed arch + Mermaid diagrams
│   ├── prompts.md             # Versioning policy
│   ├── evals.md               # Golden set methodology, metrics
│   └── decisions/             # ADRs
│       ├── 0001-openrouter-default.md
│       ├── 0002-colqwen2-vs-pipeline.md
│       └── 0003-qdrant-multi-vector.md
├── src/
│   ├── types/                 # Pydantic models shared across modules
│   │   ├── documents.py       # Paper, Page, Chunk, Figure, Table
│   │   ├── retrieval.py       # Query, RetrievalResult, RankedChunk
│   │   └── generation.py      # Context, Answer, Citation
│   ├── api/
│   │   ├── main.py            # FastAPI app
│   │   ├── routes/            # /ingest, /query, /eval
│   │   └── deps.py            # DI for clients
│   ├── ingestion/
│   │   ├── pdf.py             # PyMuPDF: pages, text, layout
│   │   ├── figures.py         # Figure extraction + VLM captioning
│   │   ├── tables.py          # Structured extraction (markdown)
│   │   ├── equations.py       # LaTeX preservation
│   │   ├── chunking.py        # Section-aware chunking
│   │   └── pipeline.py        # Orchestration
│   ├── llm/
│   │   ├── protocol.py        # LLMClient Protocol
│   │   ├── openrouter.py
│   │   ├── anthropic.py
│   │   └── ollama.py
│   ├── embeddings/
│   │   ├── protocol.py        # Embedder Protocol
│   │   ├── ollama_bge.py      # default
│   │   └── voyage.py          # alt
│   ├── rag/
│   │   ├── retrievers/
│   │   │   ├── protocol.py    # Retriever Protocol
│   │   │   ├── pipeline.py    # text + figures + tables, hybrid + rerank
│   │   │   └── visual.py      # ColQwen2 late interaction
│   │   ├── rerank.py          # BGE reranker
│   │   ├── hybrid.py          # BM25 + dense fusion (RRF)
│   │   └── generate.py        # Context assembly + LLM call
│   ├── prompts/
│   │   ├── loader.py          # YAML loader with version tracking
│   │   └── library/           # *.yaml — versioned, hashed
│   ├── eval/
│   │   ├── golden_set.py      # QASPER subset + custom queries
│   │   ├── metrics.py         # RAGAS: faithfulness, relevance, precision
│   │   ├── runner.py          # Run config × retriever × golden set
│   │   └── report.py          # Markdown + JSON output
│   ├── guardrails/
│   │   ├── citation_check.py  # Generated answer cites retrieved chunks
│   │   └── refusal.py         # Out-of-corpus query handling
│   ├── observability/
│   │   ├── langfuse.py        # Trace decorators
│   │   └── otel.py            # OpenTelemetry setup
│   └── config/
│       ├── settings.py        # Pydantic Settings
│       └── *.yaml             # default.yaml, local.yaml, prod.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/              # Sample papers + expected outputs
├── data/
│   ├── papers/                # ArXiv PDFs (gitignored, fetch script)
│   └── golden/                # Eval set
└── scripts/
    ├── fetch_papers.py        # ArXiv ML papers
    ├── ingest.py              # CLI ingestion
    └── eval.py                # CLI eval runner
```

---

## 5. MVP phasing

**Each phase ends with a working end-to-end system.** Do not start phase N+1
until phase N is deployed, evaluated, and documented.

### Phase 1 — Text-only baseline (Week 1)
- FastAPI scaffold, docker-compose with Qdrant + Postgres + Langfuse
- PyMuPDF text extraction + section-aware chunking
- Ollama BGE-M3 embeddings
- Dense retrieval + BM25 hybrid (RRF fusion)
- BGE reranker
- OpenRouter generation
- Langfuse traces wired
- Golden set: 20 manually-written queries against 5 ArXiv papers
- RAGAS metrics computed and reported
- **Deliverable: deployable, evaluable text RAG**

### Phase 2 — Pipeline multi-modal (Week 2)
- Figure extraction + VLM captioning
- Table extraction → markdown
- Equation handling (LaTeX preserved in chunks)
- Same retriever, expanded chunks
- Re-run golden set, compare to Phase 1
- **Deliverable: pipeline multi-modal vs text-only ablation**

### Phase 3 — ColQwen2 visual path (Week 3)
- `colpali-engine` integration
- Page rendering pipeline
- Qdrant multi-vector collection for late interaction
- Multi-modal generator (Claude Sonnet with image inputs)
- Re-run golden set
- **Deliverable: pipeline vs visual comparison — the headline result**

### Phase 4 — Production polish (Week 4)
- Terraform → Azure Container Apps deploy
- GitHub Actions: full CI/CD
- Guardrails (citation check, refusal handling)
- Caching layer (only if measurements show it's needed)
- Architecture docs, ADRs, README polish, demo recording
- **Deliverable: deployed, documented, demo-able**

---

## 6. Eval framework

### Golden set
- 20–30 queries against 5–10 ArXiv ML papers
- Sourced from QASPER where overlap exists; supplement with custom queries
- Each query labeled: relevant chunks/pages, expected answer key facts
- Categories: factual lookup, multi-hop, figure-dependent, table-dependent,
  equation-dependent, out-of-corpus (refusal expected)

### Metrics
- **Retrieval**: nDCG@5, recall@10, MRR
- **Generation** (RAGAS): faithfulness, answer relevance, context precision
- **Citation**: % of generated claims that cite retrieved sources
- **Latency**: p50, p95 end-to-end
- **Cost**: tokens in / out, $ per query

### Regression gates
- CI fails if any metric drops more than 5% vs. last main-branch baseline
- Eval results stored in Postgres, plotted in a simple dashboard

---

## 7. What to scaffold first (Phase 1, day 1)

1. `pyproject.toml` with: fastapi, uvicorn, pydantic, pydantic-settings,
   qdrant-client, psycopg, sqlalchemy, langfuse, openai (for OpenRouter),
   anthropic, ollama, sentence-transformers, rank-bm25, pymupdf, httpx,
   tenacity, pytest, ruff, mypy
2. `docker-compose.yml` with qdrant, postgres, langfuse, ollama services
3. `src/types/` with all shared Pydantic models
4. `src/config/settings.py` Pydantic Settings reading from YAML + env
5. `src/llm/protocol.py` + `OpenRouterClient` (only this one in Phase 1)
6. `src/embeddings/ollama_bge.py`
7. `src/api/main.py` with `/health` and a placeholder `/query`
8. `tests/unit/` skeleton with one passing test
9. `.github/workflows/ci.yml` — lint + typecheck + tests
10. `README.md` skeleton with Mermaid architecture diagram

Get that green before writing ingestion or retrieval code.

---

## 8. Open decisions deferred to implementation

- Specific Qdrant collection schema for ColQwen2 multi-vectors (verify
  Qdrant version supports MaxSim natively in the version we pin)
- Whether to use `unstructured.io` or pure PyMuPDF + custom logic for
  pipeline path — start PyMuPDF, add unstructured if quality is poor
- Specific VLM for figure captioning in pipeline path — Claude with vision
  is the safe default
- Token budget per retrieval context — start 8k, tune from eval results

---

## 9. Style conventions

- Python 3.12+
- Type hints everywhere, mypy strict
- Ruff for lint + format
- Async by default in `api/` and `llm/`; sync OK in `ingestion/` and `eval/`
- Pydantic v2 models for all data crossing module boundaries
- No bare `except`
- Logging via `structlog`, JSON in prod, pretty in dev
- Secrets only via env / Key Vault — never in YAML

---

## 10. Definition of done (project)

- All four phases deployed
- Eval results reproducible from scratch with one command
- README has: 30-second pitch, architecture diagram, quickstart, eval
  results table, demo GIF or video link
- Deployed instance accessible at a public URL
- ADRs cover all non-obvious decisions
- A blog post or write-up explaining the pipeline-vs-visual comparison
  result (this is the portfolio payoff)