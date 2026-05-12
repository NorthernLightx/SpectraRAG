# ADR 0008 — Phase 3.2 per-query routing (text-only vs hybrid)

**Status:** Accepted (2026-05-03 design; implementation closed against this design — see run `6447247ef8e7` referenced in the README).
**Date:** 2026-05-03.
**Phase:** 3.2.

## Context

ADR 0007 closed Phase 3.1 with a sign-flip on the v3-only subset:
hybrid (RRF over text + visual) edges text-only by **+1.9 % nDCG@5** on
the 14 figure/table/image-grounded queries, while losing **−10.6 %** on
the 17 definitional/factual queries (`docs/decisions/0007-...md` §"Per-subset").
On aggregate text-only still wins (0.8628 vs 0.8226), so blanket-on
hybrid is rejected. ADR 0007 promoted **per-query routing** to Phase 3.2
priority and sketched the heuristic: "detect category via length / lexical
heuristics on the query text, and only invoke the visual path when the
classifier signals figure / table / multi-hop."

This ADR pins the design before code lands so the router's behaviour is
reviewable against the empirical case in one place. The headline of the
project (pipeline-vs-visual on the same corpus) only becomes user-visible
in the deployed app once routing is wired into `/answer` — so this ADR
also unblocks ADR 0005 caveat #1 ("the deploy serves text-only").

## Decision

1. **Binary dispatch.** Two destinations: text-only or RRF-fused
   text+visual. Visual-only is not a destination — the v3 numbers don't
   justify it (visual alone scored 0.768 nDCG@5 on the v3-only subset
   vs 0.924 for hybrid; visual never wins a subset on its own).
2. **Regex/keyword classifier — no LLM, no embedding similarity.** ADR 0007
   pre-committed to "simplest viable." A misclassification routes to
   text-only, which is the strong baseline; the worst case loses some
   recall on a figure query but never produces wrong answers (citations
   still ground correctly per ADR 0006). Latency is sub-millisecond, cost
   is zero. Upgrade path is "add LLM/embedding fallback in 3.2.1 if
   measured misclass rate > X %."
3. **Five categories, page-level RRF fusion.** Categories `figure`,
   `table`, `multi_hop` route to hybrid; `factual`, `definitional` route
   to text-only. Default is `definitional`. Same vocabulary as ADR 0007's
   per-subset analysis so router logs cross-reference cleanly with eval
   reports without a relabeling pass.
4. **Per-request override.** `Query.force_route: Literal["text", "hybrid"] | None = None`.
   Bypasses the classifier when set. Used by the eval harness (run every
   query through hybrid for comparison), A/B testing, and debugging.
5. **Reuse the existing RRF in `src/rag/hybrid.py`.** No new fusion code.
   Page-level keys to match ADR 0007's offline methodology — text chunks
   are mapped to their page id before fusion; without this normalisation,
   text-and-visual hits on the same page never merge and double-count.

## Architecture

```
                  Query
                    │
          ┌─────────┴───────────┐
          │  RoutingRetriever   │  # implements src.rag.retrievers.protocol.Retriever
          └─────────┬───────────┘
                    │
         classify_query(query.text) ──► category ∈ {figure, table, multi_hop, factual, definitional}
                    │
              ┌─────┴──────┐
   text-only  │            │  hybrid (figure/table/multi_hop OR forced)
              │            │
   PipelineRetriever      asyncio.gather(
   (existing)               PipelineRetriever.retrieve(query),
              │              VisualRetriever.retrieve(query),
              │            )
              │              │
              │              ▼
              │           RRF at PAGE granularity:
              │             text chunk x → page-id (paper::p<n>)
              │             visual page  → page-id (paper::p<n>)
              │             reciprocal_rank_fusion(...)
              │              │
              │              ▼
              │           Map fused page-ids back:
              │             page has text hit → top-scoring text RetrievalResult
              │             page visual-only  → visual RetrievalResult (source="visual")
              │
              └─────────────►──────────► list[RetrievalResult] (top-k)
```

**File layout:**

- `src/rag/retrievers/routing.py` (new) — `RoutingRetriever`, `classify_query`, `Category` literal, page-id normalisation helper.
- `src/rag/retrievers/__init__.py` — re-export `RoutingRetriever` and `classify_query`.
- `src/types/retrieval.py` — add `force_route: Literal["text", "hybrid"] | None = None` to `Query`.
- `src/api/main.py` — `_wire_retriever_from_settings` builds either `PipelineRetriever` or `RoutingRetriever(text=..., visual=...)` based on `Settings.enable_routing` (new field, default True). Replaces the deferred-retriever-wiring stub from ADR 0005.
- `src/config/settings.py` — add `enable_routing: bool = True` and `visual_model: str = "vidore/colqwen2-v1.0"`.

## Classifier (precedence-ordered)

`classify_query(text)` runs the patterns in this order, returning the first match:

| Order | Category | Regex (case-insensitive) | Route |
|---|---|---|---|
| 1 | `table` | `\btable\s+\d+\|\bcell\b\|\brow\b\|\bcolumn\b` | hybrid |
| 2 | `figure` | `\bfigure\s+\d+\|\bfig\.\s*\d+\|\bplot\b\|\bdiagram\b\|\bchart\b` | hybrid |
| 3 | `multi_hop` | `\bcompare\b\|\bvs\.?\b\|\bversus\b\|\bdifferences?\b\|\bbetween\b` | hybrid |
| 4 | `factual` | `\b\d+(?:\.\d+)?\b\|\b[A-Z]{2,}\b` (numeric span or ≥2-char acronym) | text-only |
| 5 | `definitional` | default — no match above | text-only |

Precedence is intentional: a query like *"compare Figure 3 vs Figure 4"*
classifies as `figure` and not `multi_hop`. Both route to hybrid, so the
choice only affects observability labels.

## Failure modes

- **Visual leg raises** (GPU OOM, model-load error during ColQwen2
  inference, cold-start tensor weirdness): caught in `RoutingRetriever`,
  logged at warning, treated as if the query had routed text-only.
  `routing.visual_failed=true` set on the current OTel span. Demos do not
  die from GPU hiccups.
- **Both legs return empty**: standard upstream behaviour — `/answer`'s
  refusal gate (ADR 0006) handles it; no special routing logic.
- **`force_route="hybrid"` but visual retriever is None** (Phase 3.2.1 toggle
  off): error 400 from the API layer. Don't silently degrade — the caller
  asked for hybrid explicitly.

## Observability

- `structlog` event `routing.dispatched` per query: `category`, `path`
  (`text` | `hybrid`), `forced` (bool), `text_n` (count from text leg),
  `visual_n` (count from visual leg, or 0 when text-only), `fused_pages`
  (count of unique pages after fusion when path=hybrid).
- Current OTel span (`rag.retrieve`, parented under
  `POST /answer` per `src/api/routes/answer.py:25`) gains attrs:
  `routing.category`, `routing.path`, `routing.forced`,
  `routing.visual_failed` (only set when true).
- `Langfuse` trace metadata: `category`, `path` — captured on the
  existing `rag.query` trace. Wire-up details belong to the
  implementation plan, not this ADR.

## Caveats & open questions

1. **Regex brittleness vs phrasing.** A figure query phrased as
   *"the bottom panel of the network architecture"* (no `figure`/`fig.`/`plot`
   token) classifies as `definitional` and routes text-only. False
   negatives are recoverable (text @ page is the strong baseline) but
   measurable. Phase 3.2.1 adds a misclassification telemetry counter
   and revisits if the rate exceeds an as-yet-unset threshold.
2. **Factual heuristic is heuristic.** Year mentions ("after 2024") match
   the numeric pattern and label as `factual` rather than `definitional`.
   Same dispatch destination, so this is observability-only noise — the
   ADR 0007 cross-reference loses fidelity for 5–10 % of queries that
   look factual-by-numerics but are conceptually definitional. Acceptable.
3. **RRF k=60 is the literature default, not tuned.** ADR 0007's offline
   eval used the same default. Tuning k against the v3 hybrid subset is
   a Phase 3.2.1 candidate — could be worth ±0.5 % nDCG, not enough to
   block production routing.
4. **Page-level fusion changes the production result granularity.**
   Pre-routing `/answer` returned chunk-level results. Routed-hybrid
   returns page-level (text-leg chunks collapsed to their page; visual
   pages as-is). Implication for the generator: the LLM gets fewer
   candidate items but each spans a full page. We did not rerun the
   end-to-end generation eval at page granularity in 3.1 — only the
   retrieval-only metrics. Post-implementation we should re-score
   end-to-end faithfulness/precision on golden v3 with the router on
   and confirm there's no regression vs the 5-paper baseline.
5. **`force_route` not in the public OpenAPI schema.** The field is
   added to `Query` so it works via the JSON body, but we won't
   document it in `/docs` until the eval harness consumes it. Avoids
   committing to a public contract on a debug-shaped knob.

   *Update 2026-05-12:* the demo UI (`web/chat.html` and `web/index.html`)
   now exposes `force_route` via an "Advanced retrieval settings" panel
   so visitors can A/B compare text-only vs hybrid dispatch on the same
   query. The field remains absent from `/docs` — the demo UI is a
   first-party consumer, not a public API contract.
6. **Phase 3.2.1 vs 3.2.** The router lands in 3.2 with regex
   classification + production wiring. Open candidates for 3.2.1:
   per-category RRF weights, LLM/embedding fallback for unclassified
   queries, expose `force_route` in OpenAPI, golden v3.1 with edge
   queries (panel-of-figure phrasings, equation-without-LaTeX), and
   tuning RRF k.
7. **Visual retriever build is one-shot at startup**, currently the
   slowest part of the boot sequence (ADR 0004 caveat). The router
   doesn't change this — it just adds a wiring path. Phase 4.x can
   revisit lazy-load if cold-start latency becomes a Container Apps
   concern.

## References

- ADR 0004 — Phase 3 visual retrieval (visual accepted as complementary,
  hybrid deferred).
- ADR 0006 — OOC refusal gate (interaction with empty-both-legs case).
- ADR 0007 — Phase 3.1 corpus expansion + offline hybrid re-evaluation
  (the empirical case for routing — read this for the +1.9 % subset
  number and the visual-vs-hybrid per-query analysis).
- `src/rag/hybrid.py:reciprocal_rank_fusion` — existing RRF, reused as-is.
- `src/rag/retrievers/protocol.py` — `Retriever` Protocol; `RoutingRetriever` is a drop-in.
- `src/rag/retrievers/visual.py:33` — visual page-id format
  `<paper>::p<n>::page` (page granularity, not chunk).
- `src/types/retrieval.py:14` — `Query` model gains the `force_route`
  field per Decision §4.
- `data/golden/v3.yaml` — the corpus the router was re-evaluated on
  after wiring (router-on retrieval-only run; per the README, the run
  matched the v3 oracle bound from ADR 0007 §"Implications"). Promoting
  v3 to the regression baseline is a separate decision, not part of this
  ADR.
