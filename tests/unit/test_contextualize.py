"""Contextualizer: prepends LLM-generated situating blurb to each chunk."""

from __future__ import annotations

from typing import Any

from src.ingestion.contextualize import contextualize_chunks
from src.llm.protocol import ChatResponse, Message
from src.types import Chunk


class _StubLLM:
    """Records calls and returns a deterministic blurb derived from the chunk."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], str]] = []

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append((messages, model))
        # Echo a tag so tests can verify per-chunk wiring.
        user_msg = messages[-1].content
        marker = "FRAG:" + str(len(user_msg))
        return ChatResponse(text=f"context for {marker}", model=model, tokens_in=1, tokens_out=1)


def _chunk(cid: str, text: str = "fragment text") -> Chunk:
    return Chunk(chunk_id=cid, paper_id="p1", page_numbers=[1], text=text)


async def test_contextualize_populates_context_for_each_chunk() -> None:
    chunks = [_chunk("c1"), _chunk("c2", "another fragment")]
    llm = _StubLLM()

    result = await contextualize_chunks(
        chunks, paper_text="The full paper text.", llm=llm, model="cheap-model"
    )

    assert len(result) == 2
    assert all(c.context for c in result)
    assert all(c.context and c.context.startswith("context for FRAG:") for c in result)
    # Originals are not mutated — model_copy returns new instances.
    assert chunks[0].context is None


async def test_contextualize_empty_input_short_circuits() -> None:
    llm = _StubLLM()
    result = await contextualize_chunks([], paper_text="x", llm=llm, model="m")
    assert result == []
    assert llm.calls == []


async def test_contextualize_uses_paper_text_in_prompt() -> None:
    [chunk] = [_chunk("c1", text="key fragment")]
    llm = _StubLLM()
    await contextualize_chunks([chunk], paper_text="THE PAPER TEXT", llm=llm, model="m")

    [(messages, _)] = llm.calls
    user_content = messages[-1].content
    assert "THE PAPER TEXT" in user_content
    assert "key fragment" in user_content


async def test_contextualize_truncates_long_paper() -> None:
    # 200k chars > 60k limit — should be truncated.
    long_paper = "A" * 200_000
    llm = _StubLLM()
    await contextualize_chunks([_chunk("c1")], paper_text=long_paper, llm=llm, model="m")

    [(messages, _)] = llm.calls
    user_content = messages[-1].content
    assert "[... truncated ...]" in user_content
    assert len(user_content) < 100_000  # well under 200k


async def test_indexed_text_prepends_context() -> None:
    chunk = Chunk(
        chunk_id="c1",
        paper_id="p1",
        page_numbers=[1],
        text="raw chunk text",
        context="this is the situating blurb",
    )
    assert chunk.indexed_text == "this is the situating blurb\n\nraw chunk text"


async def test_indexed_text_falls_back_to_text_when_no_context() -> None:
    chunk = _chunk("c1", text="just text")
    assert chunk.context is None
    assert chunk.indexed_text == "just text"
