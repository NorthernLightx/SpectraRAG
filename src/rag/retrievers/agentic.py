"""Agentic retriever: LLM-decomposes the query, retrieves per sub-question,
fuses with RRF (ADR 0019).

Distinct from `MultiQueryRetriever`: that decorator *paraphrases* one query
(rewrite / HyDE) — same intent, multiple surface forms. This one *decomposes*
a complex multi-part query into atomic sub-questions and retrieves each one
separately, so a multi-hop question like "compare X and Y on Z" splits into
"what is X on Z?" + "what is Y on Z?" and each leg pulls its own evidence.

Per-query cost (1 decomposition call + up to N base retrievals); no
per-corpus indexing cost — unlike GraphRAG (ADR 0018), this does not rely
on a corpus-wide graph that the spike showed does not pay off on this set.
"""

from __future__ import annotations

import asyncio

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger, timed_event
from src.prompts.loader import Prompt, load_prompt_by_name
from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.rag.retrievers.protocol import Retriever
from src.types import Query, RetrievalResult

_log = get_logger(__name__)


def _parse_subqueries(text: str, *, original: str, max_subqueries: int) -> list[str]:
    """Pull non-empty, non-numbered lines off the LLM response. Falls back
    to the original query when the model produces nothing usable."""
    lines = [
        line.strip().lstrip("-*0123456789.) ").strip()
        for line in text.strip().splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        key = ln.lower()
        if not ln or key in seen or len(ln) < 3:
            continue
        seen.add(key)
        out.append(ln)
        if len(out) >= max_subqueries:
            break
    return out or [original]


class AgenticRetriever:
    """Decompose → retrieve-per-sub-question → RRF-fuse. Wraps a base Retriever.

    Graceful: on any LLM / parse failure, falls back to a single base
    retrieval on the original query. A query whose decomposition reduces to
    [original] is *identical* to plain base retrieval — there is no cost
    penalty for atomic queries beyond the one decomposition LLM call (which
    can be skipped entirely by passing `decompose=False`).
    """

    def __init__(
        self,
        *,
        base: Retriever,
        llm: LLMClient,
        model: str,
        decompose_prompt: Prompt | None = None,
        max_subqueries: int = 4,
        rrf_k: int = 60,
        decompose_max_tokens: int = 200,
    ) -> None:
        self._base = base
        self._llm = llm
        self._model = model
        self._prompt = decompose_prompt or load_prompt_by_name("decompose_query")
        self._max_subqueries = max(1, max_subqueries)
        self._rrf_k = rrf_k
        self._decompose_max_tokens = decompose_max_tokens

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        with timed_event(
            _log,
            "agentic.retrieve",
            query=query.text[:80],
            model=self._model,
            max_subqueries=self._max_subqueries,
        ) as ctx:
            subqueries = await self._decompose(query.text)
            ctx["n_subqueries"] = len(subqueries)
            # Skip the multi-retrieve fan-out when decomposition reduced to a
            # single atomic question: it would just RRF a single list against
            # itself, which is identity. Direct base call preserves any
            # routing-info / scores the base populates on the result.
            if len(subqueries) <= 1:
                return await self._base.retrieve(query)

            async def _one(sub: str) -> list[RetrievalResult]:
                sub_q = query.model_copy(update={"text": sub})
                return await self._base.retrieve(sub_q)

            results_per_sub = await asyncio.gather(*(_one(s) for s in subqueries))
            by_id: dict[str, RetrievalResult] = {}
            ranked_lists: list[list[RankedItem]] = []
            for results in results_per_sub:
                ranked_lists.append([RankedItem(id=r.chunk_id, score=r.score) for r in results])
                for r in results:
                    # First-seen-wins keeps the highest-ranked variant's text /
                    # source metadata, matching MultiQueryRetriever's pattern.
                    by_id.setdefault(r.chunk_id, r)
            fused = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k, top_k=query.top_k)
            ctx["fused_unique"] = len(by_id)
            return [by_id[item.id] for item in fused if item.id in by_id]

    async def _decompose(self, query_text: str) -> list[str]:
        system, user = self._prompt.render(query=query_text)
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))
        try:
            response = await self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=0.0,
                max_tokens=self._decompose_max_tokens,
            )
        except (RuntimeError, OSError) as exc:
            # Fall back to a single-pass retrieval on the original query.
            _log.warning("agentic.decompose_failed", error=str(exc))
            return [query_text]
        return _parse_subqueries(
            response.text, original=query_text, max_subqueries=self._max_subqueries
        )
