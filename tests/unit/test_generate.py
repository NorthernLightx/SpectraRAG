"""Generator: assembles context, calls LLM, parses citations."""

from __future__ import annotations

from typing import Any

from src.llm.protocol import ChatResponse, Message
from src.prompts.loader import Prompt
from src.rag.generate import Generator
from src.types import RetrievalResult


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


def _result(
    cid: str,
    text: str,
    score: float = 0.5,
    *,
    metadata: dict[str, Any] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid,
        paper_id="p1",
        score=score,
        text=text,
        page_numbers=[1],
        source="pipeline",
        metadata=metadata or {},
    )


def _prompt(template: str = "Q: {query}\nC:\n{context}", system: str | None = None) -> Prompt:
    return Prompt(name="t", version="v1-abc", system=system, user_template=template)


async def test_generator_calls_llm_with_rendered_prompt() -> None:
    llm = _StubLLM("Answer using [c1] and [c2].")
    gen = Generator(
        llm=llm, prompt=_prompt(system="be helpful"), model="anthropic/claude-3.5-sonnet"
    )

    answer = await gen.answer(
        "What is X?", [_result("c1", "First fact."), _result("c2", "Second fact.")]
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

    answer = await gen.answer("q", [_result("c1", "x"), _result("c2", "y")])

    cited_ids = {c.chunk_id for c in answer.citations}
    assert cited_ids == {"c1"}  # c99 dropped (not in used set), c2 not cited


async def test_generator_truncates_context_to_token_budget() -> None:
    llm = _StubLLM("ok")
    # Each block is ~50 chars; with budget=10 tokens (≈40 chars), only the first should fit.
    gen = Generator(llm=llm, prompt=_prompt(), model="m", max_context_tokens=10)

    chunks = [_result(f"c{i}", "padding text " * 5) for i in range(5)]
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


async def test_generator_extracts_citations_with_chunk_id_prefix() -> None:
    """Some local models inline `[chunk_id <id>]` despite the prompt; regex tolerates it."""
    llm = _StubLLM("Per [chunk_id c1] this is true. Also [c2] and [chunk_id c99-not-in-set].")
    gen = Generator(llm=llm, prompt=_prompt(), model="m")
    answer = await gen.answer("q", [_result("c1", "x"), _result("c2", "y")])
    cited_ids = {c.chunk_id for c in answer.citations}
    assert cited_ids == {"c1", "c2"}


async def test_generator_extracts_realistic_arxiv_chunk_ids() -> None:
    """ArXiv paper ids contain dots: `2604.22753v1`. Chunk ids look like `<paper>::p5::c24`."""
    cid_a = "2604.22753v1::p5::c24"
    cid_b = "2604.22753v1::p19::c77"
    llm = _StubLLM(f"Per [{cid_a}] and Citations: [{cid_b}], [{cid_a}].")
    gen = Generator(llm=llm, prompt=_prompt(), model="m")
    answer = await gen.answer("q", [_result(cid_a, "x"), _result(cid_b, "y")])
    cited_ids = {c.chunk_id for c in answer.citations}
    assert cited_ids == {cid_a, cid_b}


# ADR 0009: bbox propagation through citation extraction. When a cited
# chunk's metadata carries a bbox (figures + tables only), the Citation
# returned by the generator includes it so downstream UIs can render
# region-precise highlights.


async def test_citation_picks_up_bbox_from_figure_chunk_metadata() -> None:
    fig_id = "2604.22753v1::p3::fig1"
    llm = _StubLLM(f"As shown [{fig_id}].")
    gen = Generator(llm=llm, prompt=_prompt(), model="m")
    answer = await gen.answer(
        "q",
        [
            _result(
                fig_id,
                "fig caption",
                metadata={"kind": "figure", "bbox": [10.0, 20.0, 110.0, 220.0]},
            )
        ],
    )
    [cit] = answer.citations
    assert cit.bbox == [10.0, 20.0, 110.0, 220.0]


async def test_citation_picks_up_bbox_from_table_chunk_metadata() -> None:
    tab_id = "p1::p4::tab1"
    llm = _StubLLM(f"From [{tab_id}].")
    gen = Generator(llm=llm, prompt=_prompt(), model="m")
    answer = await gen.answer(
        "q",
        [
            _result(
                tab_id, "table md", metadata={"kind": "table", "bbox": [50.0, 100.0, 450.0, 300.0]}
            )
        ],
    )
    [cit] = answer.citations
    assert cit.bbox == [50.0, 100.0, 450.0, 300.0]


async def test_citation_bbox_none_for_text_chunks() -> None:
    """Text chunks have no bbox; Citation.bbox stays None."""
    llm = _StubLLM("Per [c1].")
    gen = Generator(llm=llm, prompt=_prompt(), model="m")
    answer = await gen.answer("q", [_result("c1", "text", metadata={"section": "Intro"})])
    [cit] = answer.citations
    assert cit.bbox is None


async def test_citation_ignores_malformed_bbox_metadata() -> None:
    """Defensive: chunk metadata might have a bbox of wrong shape (e.g. older
    payload schema). Citation.bbox stays None rather than crashing."""
    llm = _StubLLM("Per [c1].")
    gen = Generator(llm=llm, prompt=_prompt(), model="m")
    bad_metadata_cases: list[dict[str, Any]] = [
        {"bbox": "not a list"},
        {"bbox": [1.0, 2.0, 3.0]},  # wrong length
        {"bbox": [1.0, 2.0, 3.0, "x"]},  # non-numeric
        {"bbox": None},
    ]
    for bad in bad_metadata_cases:
        answer = await gen.answer("q", [_result("c1", "text", metadata=bad)])
        [cit] = answer.citations
        assert cit.bbox is None, f"should reject malformed bbox metadata: {bad!r}"
