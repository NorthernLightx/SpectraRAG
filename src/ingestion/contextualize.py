"""Contextual retrieval (Anthropic, Sept 2024).

For each chunk, an LLM produces a 50-100 token blurb situating the chunk inside
the paper. The blurb is prepended to the chunk text *before* embedding/BM25
indexing, so retrieval works regardless of whether section detection succeeded.

Display/citation still uses the original chunk text — `Chunk.text` is unchanged;
the blurb lives in `Chunk.context` and is combined via `Chunk.indexed_text`.
"""

from __future__ import annotations

import asyncio

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger, timed_event
from src.types import Chunk

_log = get_logger(__name__)

_MAX_PAPER_CHARS = 60_000
_DEFAULT_CONCURRENCY = 4

_SYSTEM_PROMPT = (
    "You situate fragments of a research paper inside the broader document. "
    "Given the full paper and a single fragment from it, write 1-2 short sentences "
    "(50-100 tokens) describing where this fragment sits in the paper's argument: "
    "which section/topic, what claim or definition it advances, and any key terms. "
    "Be specific and terse. Do not paraphrase the fragment. Do not add new facts. "
    "Output only the situating sentences — no preamble, no quotes."
)

_USER_TEMPLATE = (
    "<paper>\n{paper_text}\n</paper>\n\n"
    "<fragment>\n{chunk_text}\n</fragment>\n\n"
    "Situate the fragment within the paper in 1-2 sentences."
)


def _truncate_paper(paper_text: str, max_chars: int = _MAX_PAPER_CHARS) -> str:
    if len(paper_text) <= max_chars:
        return paper_text
    head = paper_text[: max_chars // 2]
    tail = paper_text[-max_chars // 2 :]
    return f"{head}\n\n[... truncated ...]\n\n{tail}"


async def _contextualize_one(
    chunk: Chunk,
    paper_text: str,
    *,
    llm: LLMClient,
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> Chunk:
    async with semaphore:
        user = _USER_TEMPLATE.format(paper_text=paper_text, chunk_text=chunk.text)
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user),
        ]
        response = await llm.chat(
            messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        return chunk.model_copy(update={"context": response.text.strip() or None})


async def contextualize_chunks(
    chunks: list[Chunk],
    paper_text: str,
    *,
    llm: LLMClient,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> list[Chunk]:
    """Return new chunks with `context` populated by an LLM situating blurb.

    `paper_text` is the full extracted paper (concatenated pages). It is
    truncated to ~60k chars before being sent on each call. Concurrency limits
    in-flight LLM calls; `model_copy` keeps the originals immutable.
    """
    if not chunks:
        return []
    truncated = _truncate_paper(paper_text)
    semaphore = asyncio.Semaphore(concurrency)
    with timed_event(
        _log,
        "contextualize.done",
        n_chunks=len(chunks),
        paper_chars=len(paper_text),
        model=model,
    ) as ctx:
        results = await asyncio.gather(
            *(
                _contextualize_one(
                    c,
                    truncated,
                    llm=llm,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
                for c in chunks
            )
        )
        ctx["with_context"] = sum(1 for c in results if c.context)
    return list(results)
