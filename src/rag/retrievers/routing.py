"""Per-query routing — classify queries to text-only or hybrid (text+visual).

ADR 0008 pins the original design: regex/keyword classifier emits one of five
categories; {figure, table, multi_hop} route to hybrid (RRF over text + visual
at page granularity), {factual, definitional} route to text-only.

ADR 0010 adds a second mode — **cascade** — that dispatches by *retrieval
confidence* instead of query category. Always runs the text leg first; only
invokes the visual leg if the top-1 rerank score falls below
`cascade_confidence_threshold`. This is a cost-quality knob: when text is
already confident, ColQwen2 inference is skipped (~30 % of total per-query
latency on the v3 corpus). The threshold is calibrated via
`scripts/calibrate_cascade.py`.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Literal

from opentelemetry import trace
from opentelemetry.trace import Span

from src.observability.logging import get_logger
from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.rag.retrievers.protocol import Retriever
from src.types import Query, RetrievalResult

if TYPE_CHECKING:
    from src.rag.retrievers.classifier_llm import LLMQueryClassifier

_log = get_logger(__name__)

# RRF k from the original Cormack & Buettcher 2009 paper. Same default
# scripts/eval_hybrid.py used to produce ADR 0007's offline numbers, so the
# production router's fused ranking tracks that eval methodology.
_RRF_K = 60

Category = Literal["table", "figure", "multi_hop", "factual", "definitional"]
RoutingPath = Literal["text", "hybrid"]
RoutingMode = Literal["category", "cascade"]

# Precedence-ordered patterns. Order matters: a query like "compare Figure 3 vs
# Figure 4" matches both `figure` and `multi_hop`; precedence picks `figure`.
# Both route to hybrid so the choice only affects the observability label.
_TABLE_RE = re.compile(r"\btable\s+\d+|\bcell\b|\brow\b|\bcolumn\b", re.IGNORECASE)
_FIGURE_RE = re.compile(
    r"\bfigure\s+\d+|\bfig\.\s*\d+|\bplot\b|\bdiagram\b|\bchart\b", re.IGNORECASE
)
_MULTIHOP_RE = re.compile(
    r"\bcompare\b|\bvs\.?\b|\bversus\b|\bdifferences?\b|\bbetween\b", re.IGNORECASE
)
# Factual = numeric span OR ≥2-char uppercase acronym. NO IGNORECASE — the
# acronym half needs case sensitivity (otherwise every word would match).
_FACTUAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b|\b[A-Z]{2,}\b")


def classify_query(text: str) -> Category:
    """Map a query string to one of the five categories per ADR 0008.

    Pure function — no I/O, no side effects, deterministic. Patterns are
    intentionally small; ADR 0008 §"Caveats" covers the trade-offs.
    """
    if _TABLE_RE.search(text):
        return "table"
    if _FIGURE_RE.search(text):
        return "figure"
    if _MULTIHOP_RE.search(text):
        return "multi_hop"
    if _FACTUAL_RE.search(text):
        return "factual"
    return "definitional"


def route_for_category(category: Category) -> RoutingPath:
    """Map a category to its dispatch destination per ADR 0008 §"Decision"."""
    if category in ("figure", "table", "multi_hop"):
        return "hybrid"
    return "text"


def _to_page_id(chunk_id: str) -> str:
    """Normalise any chunk_id to a page-id of the form 'paper::pN'.

    Text chunks are formatted 'paper::pN::cM'; visual page chunks are
    'paper::pN::page' (per src/rag/retrievers/visual.py). Both collapse to
    'paper::pN' so RRF can merge text + visual hits on the same page —
    page-level fusion per ADR 0008 §"Decision" §5.
    """
    parts = chunk_id.split("::")
    return "::".join(parts[:2])


class RoutingRetriever:
    """Dispatches queries to text-only or RRF-fused (text+visual) per ADR 0008.

    Always implements the Retriever protocol, so it slots into the existing
    `set_retriever()` wiring as a drop-in replacement for `PipelineRetriever`.

    Hybrid path runs both legs concurrently via asyncio.gather, fuses their
    rankings at page granularity using src.rag.hybrid.reciprocal_rank_fusion,
    and maps each fused page back to a single RetrievalResult — the
    highest-scoring text chunk on that page when text contributed, else the
    visual page result.

    Visual-leg failures (GPU OOM, model-load errors) fall back to text-only;
    `routing.visual_failed=true` is set on the current OTel span so the
    failure is observable without breaking the response. ADR 0008 §"Failure modes".

    Classifier upgrade: pass an `LLMQueryClassifier` via `classifier=` to
    replace the regex with an LLM-based zero-shot classifier. This is the fix
    for corpora where natural-language queries don't carry "Figure X" / "Table N"
    keywords (MMLongBench-style) — the regex under-fired by ~75 % on that
    corpus per the run-cc45831697b6 diagnostic. Default classifier is the
    regex (fast, deterministic, free).
    """

    def __init__(
        self,
        *,
        text: Retriever,
        visual: Retriever,
        classifier: LLMQueryClassifier | None = None,
        mode: RoutingMode = "category",
        cascade_confidence_threshold: float | None = None,
    ) -> None:
        if mode == "cascade" and cascade_confidence_threshold is None:
            raise ValueError(
                "RoutingRetriever(mode='cascade') requires a "
                "cascade_confidence_threshold (float in [0, 1] is typical)."
            )
        self._text = text
        self._visual = visual
        self._classifier = classifier
        self._mode = mode
        self._cascade_threshold = cascade_confidence_threshold

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        if self._mode == "cascade":
            return await self._retrieve_cascade(query)

        if self._classifier is not None:
            category = await self._classifier.classify(query.text)
        else:
            category = classify_query(query.text)
        forced = query.force_route is not None
        path: RoutingPath = (
            query.force_route if query.force_route is not None else route_for_category(category)
        )

        span = trace.get_current_span()
        span.set_attribute("routing.category", category)
        span.set_attribute("routing.path", path)
        span.set_attribute("routing.forced", forced)
        span.set_attribute("routing.mode", "category")

        if path == "text":
            text_results = await self._text.retrieve(query)
            _log.info(
                "routing.dispatched",
                mode="category",
                category=category,
                path=path,
                forced=forced,
                text_n=len(text_results),
                visual_n=0,
                fused_pages=len(text_results),
            )
            return text_results

        # Hybrid: gather text + visual concurrently so latency is max(text, visual)
        text_task = asyncio.create_task(self._text.retrieve(query))
        visual_task = asyncio.create_task(self._safe_visual_retrieve(query, span))
        text_results, visual_results = await asyncio.gather(text_task, visual_task)

        if visual_results is None:
            # Visual failed; degrade to text-only so demos don't die from GPU hiccups
            _log.info(
                "routing.dispatched",
                category=category,
                path=path,
                forced=forced,
                text_n=len(text_results),
                visual_n=0,
                fused_pages=len(text_results),
                visual_failed=True,
            )
            return text_results

        fused = self._fuse_page_level(text_results, visual_results, top_k=query.top_k)
        _log.info(
            "routing.dispatched",
            mode="category",
            category=category,
            path=path,
            forced=forced,
            text_n=len(text_results),
            visual_n=len(visual_results),
            fused_pages=len(fused),
        )
        return fused

    async def _retrieve_cascade(self, query: Query) -> list[RetrievalResult]:
        """Cascade dispatch: text first, fall back to hybrid only if uncertain.

        ADR 0010. Always runs the text leg first; if its top-1 rerank score
        meets `cascade_confidence_threshold`, return text-only (skipping the
        visual leg entirely). Else run the visual leg and RRF-fuse like the
        `category` hybrid path. `force_route="hybrid"` short-circuits to
        always-hybrid; `force_route="text"` short-circuits to always-text.
        """
        assert self._cascade_threshold is not None  # narrowed by __init__
        span = trace.get_current_span()
        span.set_attribute("routing.mode", "cascade")

        # Always run text first; cheap and we need it for the decision.
        text_results = await self._text.retrieve(query)

        # force_route overrides the confidence-based decision.
        if query.force_route == "text":
            self._log_cascade(
                path="text", decision="forced_text", forced=True,
                text_results=text_results, visual_results=None, fused_n=len(text_results),
            )
            span.set_attribute("routing.path", "text")
            return text_results
        if query.force_route == "hybrid":
            return await self._cascade_run_visual_and_fuse(
                query, span, text_results, decision="forced_hybrid", forced=True, top_score=0.0,
            )

        top_score = text_results[0].score if text_results else 0.0
        span.set_attribute("routing.cascade_top_score", float(top_score))
        span.set_attribute("routing.cascade_threshold", self._cascade_threshold)

        if top_score >= self._cascade_threshold:
            self._log_cascade(
                path="text", decision="confident_text", forced=False,
                text_results=text_results, visual_results=None,
                fused_n=len(text_results), top_score=top_score,
            )
            span.set_attribute("routing.path", "text")
            return text_results

        return await self._cascade_run_visual_and_fuse(
            query, span, text_results, decision="uncertain_hybrid",
            forced=False, top_score=top_score,
        )

    async def _cascade_run_visual_and_fuse(
        self,
        query: Query,
        span: Span,
        text_results: list[RetrievalResult],
        *,
        decision: str,
        forced: bool,
        top_score: float,
    ) -> list[RetrievalResult]:
        """Cascade fall-back: run the visual leg, RRF-fuse with text_results."""
        visual_results = await self._safe_visual_retrieve(query, span)
        if visual_results is None:
            self._log_cascade(
                path="text", decision=f"{decision}_visual_failed", forced=forced,
                text_results=text_results, visual_results=None, fused_n=len(text_results),
                top_score=top_score, visual_failed=True,
            )
            span.set_attribute("routing.path", "text")
            return text_results
        fused = self._fuse_page_level(text_results, visual_results, top_k=query.top_k)
        self._log_cascade(
            path="hybrid", decision=decision, forced=forced,
            text_results=text_results, visual_results=visual_results, fused_n=len(fused),
            top_score=top_score,
        )
        span.set_attribute("routing.path", "hybrid")
        return fused

    def _log_cascade(
        self,
        *,
        path: RoutingPath,
        decision: str,
        forced: bool,
        text_results: list[RetrievalResult],
        visual_results: list[RetrievalResult] | None,
        fused_n: int,
        top_score: float = 0.0,
        visual_failed: bool = False,
    ) -> None:
        kwargs: dict[str, object] = {
            "mode": "cascade",
            "path": path,
            "forced": forced,
            "cascade_decision": decision,
            "cascade_top_score": float(top_score),
            "cascade_threshold": self._cascade_threshold,
            "text_n": len(text_results),
            "visual_n": len(visual_results) if visual_results is not None else 0,
            "fused_pages": fused_n,
        }
        if visual_failed:
            kwargs["visual_failed"] = True
        _log.info("routing.dispatched", **kwargs)

    async def _safe_visual_retrieve(self, query: Query, span: Span) -> list[RetrievalResult] | None:
        """Run the visual retriever; on any exception, log and return None to
        signal the caller to fall back to text-only. ADR 0008 §"Failure modes"."""
        try:
            return await self._visual.retrieve(query)
        except Exception as exc:
            _log.warning(
                "routing.visual_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            span.set_attribute("routing.visual_failed", True)
            return None

    def _fuse_page_level(
        self,
        text_results: list[RetrievalResult],
        visual_results: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """RRF over page-keyed rankings; map fused pages back to RetrievalResults.

        Each leg's rank-by-page list is built by walking the leg's results in
        rank order and recording each unique page once at its first appearance.
        That preserves the leg-internal ordering that RRF expects while
        avoiding double-counting multiple text chunks on the same page.
        """
        # Best text RetrievalResult per page (highest score among chunks on that page).
        # Used to pick the "preferred" RetrievalResult to return for each fused page.
        best_text_per_page: dict[str, RetrievalResult] = {}
        for r in text_results:
            page_id = _to_page_id(r.chunk_id)
            existing = best_text_per_page.get(page_id)
            if existing is None or r.score > existing.score:
                best_text_per_page[page_id] = r

        visual_by_page: dict[str, RetrievalResult] = {
            _to_page_id(r.chunk_id): r for r in visual_results
        }

        # Build per-leg rank-by-page lists (each page once at its first appearance).
        text_pages_in_rank: list[str] = []
        seen: set[str] = set()
        for r in text_results:
            page_id = _to_page_id(r.chunk_id)
            if page_id not in seen:
                seen.add(page_id)
                text_pages_in_rank.append(page_id)
        visual_pages_in_rank = [_to_page_id(r.chunk_id) for r in visual_results]

        text_ranked = [RankedItem(id=p, score=1.0) for p in text_pages_in_rank]
        visual_ranked = [RankedItem(id=p, score=1.0) for p in visual_pages_in_rank]

        fused = reciprocal_rank_fusion([text_ranked, visual_ranked], k=_RRF_K, top_k=top_k)

        out: list[RetrievalResult] = []
        for fused_item in fused:
            page_id = fused_item.id
            if page_id in best_text_per_page:
                out.append(best_text_per_page[page_id])
            elif page_id in visual_by_page:
                out.append(visual_by_page[page_id])
        return out
