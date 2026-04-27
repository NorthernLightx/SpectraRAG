"""Generator: assembles context, calls LLM, parses citations."""

from __future__ import annotations

from typing import Any

from src.llm.protocol import ChatResponse, Message
from src.prompts.loader import Prompt
from src.rag.generate import Generator
from src.types import Chunk, RankedChunk


class _StubLLM:
    """LLMClient stub that records calls and returns a canned response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[list[Message], str, float]] = []

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append((messages, model, temperature))
        return ChatResponse(
            text=self.response_text,
            model=model,
            tokens_in=42,
            tokens_out=21,
            raw={},
        )


def _ranked(cid: str, text: str, score: float = 0.5) -> RankedChunk:
    chunk = Chunk(chunk_id=cid, paper_id="p1", page_numbers=[1], text=text)
    return RankedChunk(chunk=chunk, score=score, rank=1)


def _prompt(template: str = "Q: {query}\nC:\n{context}", system: str | None = None) -> Prompt:
    return Prompt(name="t", version="v1-abc", system=system, user_template=template)


async def test_generator_calls_llm_with_rendered_prompt() -> None:
    llm = _StubLLM("Answer using [c1] and [c2].")
    gen = Generator(
        llm=llm, prompt=_prompt(system="be helpful"), model="anthropic/claude-3.5-sonnet"
    )

    answer = await gen.answer(
        "What is X?", [_ranked("c1", "First fact."), _ranked("c2", "Second fact.")]
    )

    assert answer.text.startswith("Answer using")
    assert answer.model == "anthropic/claude-3.5-sonnet"
    assert answer.tokens_in == 42 and answer.tokens_out == 21
    assert answer.prompt_version == "v1-abc"

    [(messages, model, _temp)] = llm.calls
    assert model == "anthropic/claude-3.5-sonnet"
    assert messages[0].role == "system" and messages[0].content == "be helpful"
    assert messages[1].role == "user"
    assert "What is X?" in messages[1].content
    assert "[c1]" in messages[1].content and "[c2]" in messages[1].content


async def test_generator_extracts_citations_for_chunks_referenced_in_answer() -> None:
    llm = _StubLLM("Per [c1] this is true. [c99] does not exist in our context.")
    gen = Generator(llm=llm, prompt=_prompt(), model="m")

    answer = await gen.answer("q", [_ranked("c1", "x"), _ranked("c2", "y")])

    cited_ids = {c.chunk_id for c in answer.citations}
    assert cited_ids == {"c1"}  # c99 dropped (not in used set), c2 not cited


async def test_generator_truncates_context_to_token_budget() -> None:
    llm = _StubLLM("ok")
    # Each block is ~50 chars; with budget=10 tokens (≈40 chars), only the first should fit.
    gen = Generator(llm=llm, prompt=_prompt(), model="m", max_context_tokens=10)

    chunks = [_ranked(f"c{i}", "padding text " * 5) for i in range(5)]
    await gen.answer("q", chunks)

    [(messages, _, _)] = llm.calls
    # Only the first chunk fits — second chunk wouldn't, so it's dropped.
    assert "[c0]" in messages[0].content
    assert "[c1]" not in messages[0].content


async def test_generator_runs_with_no_chunks() -> None:
    llm = _StubLLM("I don't know.")
    gen = Generator(llm=llm, prompt=_prompt(), model="m")
    answer = await gen.answer("q", [])
    assert answer.text == "I don't know."
    assert answer.citations == []
