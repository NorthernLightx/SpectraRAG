"""Phase 3.2 per-query routing — classify queries to text-only or hybrid (text+visual).

ADR 0008 pins the design: regex/keyword classifier emits one of five categories;
{figure, table, multi_hop} route to hybrid (RRF over text + visual at page
granularity), {factual, definitional} route to text-only. Misclassification cost
is bounded — the worst case routes a figure query through text-only, which is
the strong baseline (text @ page nDCG@5 = 0.86 per ADR 0007).
"""

from __future__ import annotations

import asyncio
import re
from typing import Literal

from opentelemetry import trace
from opentelemetry.trace import Span

from src.observability.logging import get_logger
from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.rag.retrievers.protocol import Retriever
from src.types import Query, RetrievalResult

_log = get_logger(__name__)

# RRF k from the original Cormack & Buettcher 2009 paper. Same default
# scripts/eval_hybrid.py used to produce ADR 0007's offline numbers, so the
# production router's fused ranking should track that eval methodology.
# ADR 0008 caveat §3 — Phase 3.2.1 candidate to tune against golden v3.
_RRF_K = 60

Category = Literal["table", "figure", "multi_hop", "factual", "definitional"]
RoutingPath = Literal["text", "hybrid"]

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
    """

    def __init__(self, *, text: Retriever, visual: Retriever) -> None:
        self._text = text
        self._visual = visual

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        category = classify_query(query.text)
        forced = query.force_route is not None
        path: RoutingPath = (
            query.force_route if query.force_route is not None else route_for_category(category)
        )

        span = trace.get_current_span()
        span.set_attribute("routing.category", category)
        span.set_attribute("routing.path", path)
        span.set_attribute("routing.forced", forced)

        if path == "text":
            text_results = await self._text.retrieve(query)
            _log.info(
                "routing.dispatched",
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
            category=category,
            path=path,
            forced=forced,
            text_n=len(text_results),
            visual_n=len(visual_results),
            fused_pages=len(fused),
        )
        return fused

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
