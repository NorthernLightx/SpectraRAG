"""Community summarization: title/summary parse + graceful drop (ADR 0018)."""

from __future__ import annotations

from typing import Any

from src.graph import build_graph, detect_communities, summarize_communities
from src.graph.summarize import _parse
from src.llm.protocol import ChatResponse, Message
from src.types import ChunkExtraction, Community, GraphEntity, GraphRelation


class _StubLLM:
    def __init__(
        self,
        text: str = "Scaling Laws\nThis cluster covers scaling-law fitting.",
        *,
        boom: bool = False,
    ) -> None:
        self._text = text
        self._boom = boom

    async def chat(self, messages: list[Message], model: str, **kwargs: Any) -> ChatResponse:
        if self._boom:
            raise RuntimeError("ollama down")
        return ChatResponse(text=self._text, model=model, tokens_in=1, tokens_out=1)


def _graph() -> Any:
    ex = ChunkExtraction(
        chunk_id="c0",
        entities=[
            GraphEntity(name="Scaling Law", type="concept", description="loss vs scale"),
            GraphEntity(name="Budget", type="concept", description="compute budget"),
        ],
        relations=[
            GraphRelation(
                source="Scaling Law", target="Budget", description="constrained by", weight=5.0
            )
        ],
    )
    return build_graph([ex])


def test_parse_title_then_summary() -> None:
    r = _parse("My Title\nLine one. Line two.", "L0_0")
    assert r is not None
    assert r.title == "My Title"
    assert r.summary == "Line one. Line two."


def test_parse_empty_returns_none() -> None:
    assert _parse("   \n  ", "L0_0") is None


async def test_summarize_emits_one_report_per_community() -> None:
    g = _graph()
    comms = detect_communities(g)
    reports = await summarize_communities(g, comms, llm=_StubLLM(), model="m")
    assert reports
    assert reports[0].title == "Scaling Laws"
    assert "scaling-law" in reports[0].summary


async def test_summarize_drops_failed_calls() -> None:
    g = _graph()
    comms = detect_communities(g)
    assert await summarize_communities(g, comms, llm=_StubLLM(boom=True), model="m") == []


async def test_summarize_drops_community_absent_from_graph() -> None:
    g = _graph()
    ghost = Community(community_id="L0_9", level=0, entity_names=["nonexistent"], parent_id=None)
    assert await summarize_communities(g, [ghost], llm=_StubLLM(), model="m") == []


async def test_summarize_empty_input() -> None:
    assert await summarize_communities(_graph(), [], llm=_StubLLM(), model="m") == []
