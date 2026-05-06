"""LLMQueryClassifier — LLM-based query classifier alternative to the regex.

Used by RoutingRetriever when MMLongBench-style natural-language queries don't
carry "Figure X" / "Table N" keywords the regex looks for.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.llm.protocol import ChatResponse, Message
from src.prompts.loader import Prompt
from src.rag.retrievers.classifier_llm import LLMQueryClassifier, _parse_category


def _prompt() -> Prompt:
    return Prompt(name="classify_query", version="v0", system=None, user_template="{query}")


class _CannedLLM:
    """Returns whatever response_text we set; records calls."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[list[Message], str]] = []

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        images: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append((messages, model))
        return ChatResponse(text=self.response_text, model=model, tokens_in=10, tokens_out=2)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("table", "table"),
        ("figure", "figure"),
        ("multi_hop", "multi_hop"),
        ("factual", "factual"),
        ("definitional", "definitional"),
        ("Table.", "table"),  # trailing punctuation tolerated
        ("'figure'", "figure"),  # surrounding quotes tolerated
        ("FIGURE", "figure"),  # uppercase normalised
        ("figure\n\nrationale text", "figure"),  # only first line read
    ],
)
def test_parse_category_happy_paths(raw: str, expected: str) -> None:
    assert _parse_category(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "neither_a_known_token",
        "Here is the answer: figure",  # extra prose before token — not the rubric
    ],
)
def test_parse_category_falls_back_to_definitional(raw: str) -> None:
    """Unparseable / unknown / multi-token responses degrade to text-only routing."""
    assert _parse_category(raw) == "definitional"


@pytest.mark.asyncio
async def test_classifier_calls_llm_and_returns_category() -> None:
    llm = _CannedLLM(response_text="figure")
    clf = LLMQueryClassifier(llm=llm, model="gpt-4o-mini-test", prompt=_prompt())
    cat = await clf.classify("What does the chart on page 5 show?")
    assert cat == "figure"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_classifier_handles_unexpected_output_gracefully() -> None:
    """Real models occasionally ramble. Classifier must not crash; defaults
    to definitional so routing degrades to text-only."""
    llm = _CannedLLM(response_text="Sure, this is a figure-related query.")
    clf = LLMQueryClassifier(llm=llm, model="gpt-4o-mini-test", prompt=_prompt())
    cat = await clf.classify("Q?")
    # Multi-token first line → not in valid set → falls back
    assert cat == "definitional"
