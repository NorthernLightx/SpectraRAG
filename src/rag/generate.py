"""Generator: assemble context from ranked chunks, call the LLM, parse citations."""

from __future__ import annotations

import re
import time

from src.llm.protocol import LLMClient, Message
from src.prompts.loader import Prompt
from src.types import Answer, Chunk, Citation, RankedChunk

_CITATION_RE = re.compile(r"\[([A-Za-z0-9:_\-]+)\]")
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
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._model = model
        self._temperature = temperature
        self._max_context_chars = max_context_tokens * _CHARS_PER_TOKEN

    async def answer(self, query: str, ranked_chunks: list[RankedChunk]) -> Answer:
        context, used = self._build_context(ranked_chunks)
        system, user = self._prompt.render(query=query, context=context)

        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))

        started = time.monotonic()
        response = await self._llm.chat(
            messages=messages, model=self._model, temperature=self._temperature
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        citations = self._extract_citations(response.text, used)
        return Answer(
            text=response.text,
            citations=citations,
            model=response.model,
            prompt_version=self._prompt.version,
            latency_ms=latency_ms,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )

    def _build_context(self, ranked: list[RankedChunk]) -> tuple[str, list[Chunk]]:
        used: list[Chunk] = []
        parts: list[str] = []
        budget = 0
        for ranked_chunk in ranked:
            chunk = ranked_chunk.chunk
            pages = ", ".join(str(p) for p in chunk.page_numbers)
            block = f"[{chunk.chunk_id}] (pages {pages}) {chunk.text}"
            if budget + len(block) > self._max_context_chars and parts:
                break
            parts.append(block)
            used.append(chunk)
            budget += len(block) + 2  # account for "\n\n" separator
        return "\n\n".join(parts), used

    def _extract_citations(self, text: str, used: list[Chunk]) -> list[Citation]:
        cited_ids = set(_CITATION_RE.findall(text))
        by_id = {c.chunk_id: c for c in used}
        citations: list[Citation] = []
        for cid in cited_ids:
            chunk = by_id.get(cid)
            if chunk is None:
                continue
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    page_numbers=chunk.page_numbers,
                )
            )
        return citations
