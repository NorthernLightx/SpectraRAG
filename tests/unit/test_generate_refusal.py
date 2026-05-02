"""Generator's optional rerank-score refusal gate."""

from __future__ import annotations

from src.llm.protocol import ChatResponse
from src.prompts.loader import Prompt
from src.rag.generate import Generator
from src.types import RetrievalResult


class _CountingLLM:
    """Records whether chat() was called."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *, messages, model, temperature):  # type: ignore[no-untyped-def]
        self.calls += 1
        return ChatResponse(text="answer [c1]", model=model, tokens_in=10, tokens_out=20)


def _chunk(score: float, chunk_id: str = "c1") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id="p",
        score=score,
        text="content",
        page_numbers=[1],
        source="pipeline",
    )


def _prompt() -> Prompt:
    return Prompt(name="answer", version="v0", system=None, user_template="{query} {context}")


async def test_generator_refuses_when_all_scores_below_threshold() -> None:
    llm = _CountingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", refusal_score_threshold=0.5)  # type: ignore[arg-type]

    answer = await gen.answer("q", [_chunk(0.1), _chunk(0.4, "c2")])

    assert llm.calls == 0  # gate fired before LLM
    assert answer.citations == []
    assert answer.tokens_in == 0
    assert answer.tokens_out == 0
    assert answer.model == "refusal-gate"
    assert "cannot answer" in answer.text.lower() or "out of corpus" in answer.text.lower()


async def test_generator_does_not_refuse_when_any_score_meets_threshold() -> None:
    llm = _CountingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", refusal_score_threshold=0.5)  # type: ignore[arg-type]

    answer = await gen.answer("q", [_chunk(0.1), _chunk(0.6, "c2")])

    assert llm.calls == 1  # LLM was invoked
    assert answer.model == "m"
    assert answer.tokens_in == 10
    assert answer.tokens_out == 20


async def test_generator_with_no_threshold_never_refuses() -> None:
    """Regression: default behaviour (threshold=None) is unchanged."""
    llm = _CountingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m")  # type: ignore[arg-type]  # no threshold

    answer = await gen.answer("q", [_chunk(0.0, "c1")])  # score 0.0 — would refuse if gated

    assert llm.calls == 1
    assert answer.model == "m"


async def test_generator_refuses_on_empty_retrieval_when_threshold_set() -> None:
    """Edge case: zero retrieved chunks => refuse (threshold semantics imply 'no confident match')."""
    llm = _CountingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", refusal_score_threshold=0.5)  # type: ignore[arg-type]

    answer = await gen.answer("q", [])

    assert llm.calls == 0
    assert answer.model == "refusal-gate"
