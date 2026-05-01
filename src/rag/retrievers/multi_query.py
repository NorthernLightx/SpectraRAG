"""MultiQueryRetriever — Retriever decorator that fuses results from query
variants (LLM rewrites + optional HyDE passage). Concrete second impl after
`PipelineRetriever`; still under the rule-of-three threshold so no new
abstraction.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from src.observability.logging import get_logger, timed_event
from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.rag.query_expansion import QueryExpander
from src.rag.retrievers.protocol import Retriever
from src.types import Query, RetrievalResult

_log = get_logger(__name__)

ExpansionMode = Literal["rewrite", "hyde", "combo"]


class MultiQueryRetriever:
    """Wraps a base `Retriever`. Generates query variants via `QueryExpander`,
    retrieves for each in parallel, and fuses with reciprocal rank fusion.

    The original query is always retrieved as one of the variants — even if
    the LLM expander returns nothing (empty list / empty string), retrieval
    still happens for the original.
    """

    def __init__(
        self,
        *,
        base: Retriever,
        expander: QueryExpander,
        mode: ExpansionMode = "rewrite",
        n_rewrites: int = 3,
        rrf_k: int = 60,
        concurrency: int = 2,
    ) -> None:
        self._base = base
        self._expander = expander
        self._mode = mode
        self._n_rewrites = n_rewrites
        self._rrf_k = rrf_k
        self._concurrency = concurrency

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        with timed_event(
            _log,
            "multi_query.done",
            query=query.text,
            mode=self._mode,
            n_rewrites=self._n_rewrites,
        ) as ctx:
            variants = await self._gather_variants(query.text)
            ctx["n_variants"] = len(variants)
            ctx["variants"] = variants

            # Retrieve for the original + each variant, with concurrency capped.
            # Default is 2 — anything higher saturated Ollama embed + Qdrant
            # connection pools and one variant timed out per query, aborting
            # the whole asyncio.gather. We also tolerate per-variant failures
            # via return_exceptions, falling back to the variants that did work.
            sub_top_k = max(query.top_k * 2, query.top_k)
            queries = [Query(text=q, top_k=sub_top_k) for q in [query.text, *variants]]
            sem = asyncio.Semaphore(self._concurrency)

            async def _one(q: Query) -> list[RetrievalResult]:
                async with sem:
                    return await self._base.retrieve(q)

            raw = await asyncio.gather(*(_one(q) for q in queries), return_exceptions=True)
            results_per_variant: list[list[RetrievalResult]] = []
            for q, item in zip(queries, raw, strict=True):
                if isinstance(item, BaseException):
                    _log.warning(
                        "multi_query.variant_failed",
                        query=q.text,
                        error=type(item).__name__,
                    )
                    continue
                results_per_variant.append(item)

            # Build a chunk_id -> RetrievalResult map keyed by the *first* time
            # we see each chunk (preserves the metadata/score from whichever
            # variant ranked it). Then fuse the surviving ranked lists with RRF.
            seen: dict[str, RetrievalResult] = {}
            ranked_lists: list[list[RankedItem]] = []
            for results in results_per_variant:
                for result in results:
                    seen.setdefault(result.chunk_id, result)
                ranked_lists.append([RankedItem(id=r.chunk_id, score=r.score) for r in results])

            fused = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k, top_k=query.top_k)
            out = [seen[item.id] for item in fused if item.id in seen]
            ctx["returned"] = len(out)
            ctx["top_chunk"] = out[0].chunk_id if out else None
            ctx["variants_succeeded"] = len(results_per_variant)
            return out

    async def _gather_variants(self, query: str) -> list[str]:
        """Return the LLM-generated variants for the configured mode."""
        if self._mode == "rewrite":
            return await self._expander.rewrite(query, n=self._n_rewrites)
        if self._mode == "hyde":
            hyde = await self._expander.hyde(query)
            return [hyde] if hyde else []
        # combo
        rewrites_task = self._expander.rewrite(query, n=self._n_rewrites)
        hyde_task = self._expander.hyde(query)
        rewrites, hyde = await asyncio.gather(rewrites_task, hyde_task)
        return [*rewrites, hyde] if hyde else list(rewrites)
