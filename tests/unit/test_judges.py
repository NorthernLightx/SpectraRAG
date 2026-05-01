"""LLMJudge: parser robustness and end-to-end judging via a stub LLM."""

from __future__ import annotations

from typing import Any

import pytest

from src.eval.judges import LLMJudge, _parse_score
from src.llm.protocol import ChatResponse, Message
from src.prompts.loader import load_prompt_by_name
from src.types import RetrievalResult


class _StubLLM:
    """Records calls and replies with a queue of canned ChatResponses."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append(
            {"messages": messages, "model": model, "temperature": temperature, "kwargs": kwargs}
        )
        text = self._replies.pop(0) if self._replies else "0.0\nfallback"
        return ChatResponse(text=text, model=model, tokens_in=10, tokens_out=5)


def _retrieval_result(chunk_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id="paper1",
        score=1.0,
        text=text,
        page_numbers=[1],
        source="pipeline",
    )


@pytest.mark.parametrize(
    ("raw", "expected_score"),
    [
        ("0.8\nrationale here", 0.8),
        ("1.0", 1.0),
        ("0", 0.0),
        ("0.5\nthe answer is partially supported.\nmore detail.", 0.5),
        ("  0.7  \nleading whitespace ok", 0.7),
        ("score: 0.6\nmoot — first decimal is in line 1 anyway", 0.6),
        ("1.5\nshould clamp to 1.0", 1.0),
        ("-0.3\nshould clamp to 0.0", 0.0),
    ],
)
def test_parse_score_extracts_first_decimal(raw: str, expected_score: float) -> None:
    score, _rationale = _parse_score(raw)
    assert score == expected_score


def test_parse_score_returns_zero_for_unparseable() -> None:
    score, rationale = _parse_score("totally not a number\nignore me")
    assert score == 0.0
    assert "totally not a number" in rationale


def test_parse_score_handles_empty_input() -> None:
    assert _parse_score("") == (0.0, "[judge returned empty response]")
    assert _parse_score("   \n  ")[0] == 0.0


async def test_judge_faithfulness_calls_llm_and_returns_score() -> None:
    llm = _StubLLM(["0.85\nMost claims supported."])
    judge = LLMJudge(
        llm=llm,
        model="qwen2.5:7b",
        faithfulness_prompt=load_prompt_by_name("judge_faithfulness"),
        answer_relevance_prompt=load_prompt_by_name("judge_answer_relevance"),
        context_precision_prompt=load_prompt_by_name("judge_context_precision"),
    )

    result = await judge.faithfulness(
        query="What is X?",
        answer="X is Y because of Z.",
        retrieved=[_retrieval_result("c1", "X is defined as Y in section 2.")],
    )

    assert result.score == 0.85
    assert "Most claims supported" in result.rationale
    assert llm.calls[0]["model"] == "qwen2.5:7b"
    user_msg = llm.calls[0]["messages"][-1].content
    assert "X is Y because of Z." in user_msg
    assert "[c1]" in user_msg


async def test_judge_answer_relevance_does_not_send_context() -> None:
    llm = _StubLLM(["0.4\nPartially relevant."])
    judge = LLMJudge(
        llm=llm,
        model="qwen2.5:7b",
        faithfulness_prompt=load_prompt_by_name("judge_faithfulness"),
        answer_relevance_prompt=load_prompt_by_name("judge_answer_relevance"),
        context_precision_prompt=load_prompt_by_name("judge_context_precision"),
    )

    result = await judge.answer_relevance(
        query="What is X?", answer="X is unrelated to your question."
    )
    assert result.score == 0.4
    user_msg = llm.calls[0]["messages"][-1].content
    assert "What is X?" in user_msg
    assert "X is unrelated to your question." in user_msg
    # Context section should NOT appear
    assert "Retrieved chunks" not in user_msg
    assert "Source context" not in user_msg


async def test_judge_context_precision_inlines_chunks() -> None:
    llm = _StubLLM(["0.5\nHalf are relevant."])
    judge = LLMJudge(
        llm=llm,
        model="qwen2.5:7b",
        faithfulness_prompt=load_prompt_by_name("judge_faithfulness"),
        answer_relevance_prompt=load_prompt_by_name("judge_answer_relevance"),
        context_precision_prompt=load_prompt_by_name("judge_context_precision"),
    )

    result = await judge.context_precision(
        query="What is X?",
        retrieved=[
            _retrieval_result("c1", "X is defined as Y."),
            _retrieval_result("c2", "Unrelated digression about Q."),
        ],
    )

    assert result.score == 0.5
    user_msg = llm.calls[0]["messages"][-1].content
    assert "[c1]" in user_msg and "[c2]" in user_msg
    assert "X is defined as Y." in user_msg


async def test_judge_clamps_unparseable_to_zero() -> None:
    llm = _StubLLM(["I am a small model and will not output numbers"])
    judge = LLMJudge(
        llm=llm,
        model="qwen2.5:7b",
        faithfulness_prompt=load_prompt_by_name("judge_faithfulness"),
        answer_relevance_prompt=load_prompt_by_name("judge_answer_relevance"),
        context_precision_prompt=load_prompt_by_name("judge_context_precision"),
    )

    result = await judge.answer_relevance(query="Q?", answer="A.")
    assert result.score == 0.0
    assert "I am a small model" in result.rationale
