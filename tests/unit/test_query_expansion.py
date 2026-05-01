"""QueryExpander: rewrite parsing + e2e via stub LLM."""

from __future__ import annotations

from typing import Any

from src.llm.protocol import ChatResponse, Message
from src.rag.query_expansion import QueryExpander, _parse_rewrites


class _StubLLM:
    """Records calls and replies with canned ChatResponses (queue-style)."""

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
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        text = self._replies.pop(0) if self._replies else ""
        return ChatResponse(text=text, model=model, tokens_in=10, tokens_out=5)


def test_parse_rewrites_strips_numbering_and_bullets() -> None:
    raw = """1. How is the inter-basin gain defined?
2) Which acquisition criterion measures variance reduction across basins?
- What objective does inter-basin acquisition optimise?
* Yet another phrasing here?
"How does the paper define inter-basin gain?"
"""
    out = _parse_rewrites(raw, expected=10)
    assert out == [
        "How is the inter-basin gain defined?",
        "Which acquisition criterion measures variance reduction across basins?",
        "What objective does inter-basin acquisition optimise?",
        "Yet another phrasing here?",
        "How does the paper define inter-basin gain?",
    ]


def test_parse_rewrites_dedupes_case_insensitively() -> None:
    raw = """How is X defined?
how is X defined?
What is X?"""
    out = _parse_rewrites(raw, expected=10)
    assert out == ["How is X defined?", "What is X?"]


def test_parse_rewrites_caps_at_expected() -> None:
    raw = "\n".join(f"Phrasing {i}?" for i in range(10))
    out = _parse_rewrites(raw, expected=3)
    assert len(out) == 3


def test_parse_rewrites_handles_empty_lines() -> None:
    raw = "\n\nFirst rephrase?\n\nSecond rephrase?\n"
    out = _parse_rewrites(raw, expected=5)
    assert out == ["First rephrase?", "Second rephrase?"]


def test_parse_rewrites_returns_empty_for_blank() -> None:
    assert _parse_rewrites("", expected=3) == []
    assert _parse_rewrites("   \n  \n", expected=3) == []


async def test_expander_rewrite_calls_llm_and_parses() -> None:
    llm = _StubLLM(["Rephrase one?\nRephrase two?\nRephrase three?"])
    expander = QueryExpander(llm=llm, model="qwen2.5:7b")
    rewrites = await expander.rewrite("What is X?", n=3)
    assert rewrites == ["Rephrase one?", "Rephrase two?", "Rephrase three?"]
    user_msg = llm.calls[0]["messages"][-1].content
    assert "What is X?" in user_msg
    assert "exactly 3" in user_msg


async def test_expander_rewrite_with_zero_returns_empty() -> None:
    llm = _StubLLM(["should not be called"])
    expander = QueryExpander(llm=llm, model="qwen2.5:7b")
    assert await expander.rewrite("Q?", n=0) == []
    assert llm.calls == []


async def test_expander_hyde_returns_passage() -> None:
    llm = _StubLLM(
        [
            "The inter-basin gain quantifies expected variance reduction across multiple plausible scaling-law basins after observing a candidate experiment."
        ]
    )
    expander = QueryExpander(llm=llm, model="qwen2.5:7b")
    passage = await expander.hyde("What is the inter-basin gain?")
    assert passage.startswith("The inter-basin gain quantifies")
    assert "across multiple plausible" in passage


async def test_expander_hyde_returns_empty_on_blank() -> None:
    llm = _StubLLM([""])
    expander = QueryExpander(llm=llm, model="qwen2.5:7b")
    assert await expander.hyde("Q?") == ""
