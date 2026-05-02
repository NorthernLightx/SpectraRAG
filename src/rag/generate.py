"""Generator: assemble context from retrieved chunks, call the LLM, parse citations."""

from __future__ import annotations

import re
import time

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger, timed_event
from src.observability.metrics import (
    GENERATE_LATENCY_MS,
    TOKENS_IN,
    TOKENS_OUT,
)
from src.prompts.loader import Prompt
from src.types import Answer, Citation, RetrievalResult

_log = get_logger(__name__)

# Match `[<id>]` and `[chunk_id <id>]` (some local models inline the literal "chunk_id"
# despite the prompt). The id can contain dots — ArXiv paper ids like `2604.22753v1`
# have them, and chunk ids are `<paper_id>::p<n>::c<n>`. Without `.` the regex would
# silently truncate `2604.22753v1::p5::c24` to `2604`.
_CITATION_RE = re.compile(r"\[(?:chunk_id\s+)?([A-Za-z0-9.:_\-]+)\]")
_CHARS_PER_TOKEN = 4  # rough approximation; replace with tokenizer when needed


class Generator:
    """LLM-backed answer generator. Renders a prompt over retrieved chunks and parses citations."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompt: Prompt,
        model: str,
        temperature: float = 0.2,
        max_context_tokens: int = 8000,
        refusal_score_threshold: float | None = None,
        refusal_text: str = "I cannot answer this question from the provided corpus.",
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._model = model
        self._temperature = temperature
        self._max_context_chars = max_context_tokens * _CHARS_PER_TOKEN
        self._refusal_score_threshold = refusal_score_threshold
        self._refusal_text = refusal_text

    async def answer(self, query: str, retrieved: list[RetrievalResult]) -> Answer:
        if self._refusal_score_threshold is not None and self._should_refuse(retrieved):
            _log.info(
                "generate.refused",
                reason="rerank_score_below_threshold",
                threshold=self._refusal_score_threshold,
                top_score=max((r.score for r in retrieved), default=None),
                n_retrieved=len(retrieved),
            )
            return self._refusal()
        context, used = self._build_context(retrieved)
        system, user = self._prompt.render(query=query, context=context)

        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))

        with timed_event(
            _log,
            "generate.done",
            model=self._model,
            prompt_version=self._prompt.version,
            context_chunks=len(used),
        ) as ctx:
            started = time.monotonic()
            response = await self._llm.chat(
                messages=messages, model=self._model, temperature=self._temperature
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            citations = self._extract_citations(response.text, used)
            ctx["model"] = response.model
            ctx["tokens_in"] = response.tokens_in
            ctx["tokens_out"] = response.tokens_out
            ctx["citations"] = len(citations)
            attrs = {"model": response.model, "prompt_version": self._prompt.version}
            TOKENS_IN.add(response.tokens_in, attributes=attrs)
            TOKENS_OUT.add(response.tokens_out, attributes=attrs)
            GENERATE_LATENCY_MS.record(latency_ms, attributes=attrs)
        return Answer(
            text=response.text,
            citations=citations,
            model=response.model,
            prompt_version=self._prompt.version,
            latency_ms=latency_ms,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )

    def _build_context(self, retrieved: list[RetrievalResult]) -> tuple[str, list[RetrievalResult]]:
        used: list[RetrievalResult] = []
        parts: list[str] = []
        budget = 0
        for r in retrieved:
            pages = ", ".join(str(p) for p in r.page_numbers)
            block = f"[{r.chunk_id}] (pages {pages}) {r.text}"
            if budget + len(block) > self._max_context_chars and parts:
                break
            parts.append(block)
            used.append(r)
            budget += len(block) + 2
        return "\n\n".join(parts), used

    def _extract_citations(self, text: str, used: list[RetrievalResult]) -> list[Citation]:
        cited_ids = set(_CITATION_RE.findall(text))
        by_id = {r.chunk_id: r for r in used}
        citations: list[Citation] = []
        for cid in cited_ids:
            r = by_id.get(cid)
            if r is None:
                continue
            citations.append(
                Citation(
                    chunk_id=r.chunk_id,
                    paper_id=r.paper_id,
                    page_numbers=r.page_numbers,
                )
            )
        return citations

    def _should_refuse(self, retrieved: list[RetrievalResult]) -> bool:
        if not retrieved:
            return True
        threshold = self._refusal_score_threshold
        assert threshold is not None  # narrowed by caller's check
        return all(r.score < threshold for r in retrieved)

    def _refusal(self) -> Answer:
        return Answer(
            text=self._refusal_text,
            citations=[],
            model="refusal-gate",
            prompt_version="refusal-v1",
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
        )
