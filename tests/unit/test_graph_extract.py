"""Graph extraction: tolerant JSON parsing + graceful per-chunk failure (ADR 0018)."""

from __future__ import annotations

from typing import Any

from src.ingestion.graph_extract import _parse_extraction, extract_graph
from src.llm.protocol import ChatResponse, Message
from src.types import Chunk

_GOOD = (
    '{"is_reference_list": false,'
    ' "entities": [{"name": "BGE-M3", "type": "model", "description": "an embedder"},'
    ' {"name": "Recall", "type": "metric", "description": "retrieval metric"}],'
    ' "relations": [{"source": "BGE-M3", "target": "Recall",'
    ' "description": "is evaluated by", "weight": 7}]}'
)


class _StubLLM:
    """Minimal LLMClient: returns a fixed response, or raises if `boom`."""

    def __init__(self, text: str = _GOOD, *, boom: bool = False) -> None:
        self._text = text
        self._boom = boom

    async def chat(self, messages: list[Message], model: str, **kwargs: Any) -> ChatResponse:
        if self._boom:
            raise RuntimeError("ollama down")
        return ChatResponse(text=self._text, model=model, tokens_in=1, tokens_out=1)


def _chunk(text: str = "BGE-M3 is scored by recall.") -> Chunk:
    return Chunk(chunk_id="p::p1::c0", paper_id="p", page_numbers=[1], text=text)


def test_parse_clean_json() -> None:
    ex = _parse_extraction(_GOOD, "c1")
    assert not ex.is_reference_list
    assert {e.name for e in ex.entities} == {"BGE-M3", "Recall"}
    assert ex.relations[0].source == "BGE-M3" and ex.relations[0].weight == 7.0


def test_parse_fenced_and_prefixed() -> None:
    wrapped = f"Sure, here is the graph:\n```json\n{_GOOD}\n```\nDone."
    ex = _parse_extraction(wrapped, "c1")
    assert len(ex.entities) == 2


def test_parse_reference_list_flag() -> None:
    ex = _parse_extraction('{"is_reference_list": true, "entities": [], "relations": []}', "c1")
    assert ex.is_reference_list and not ex.entities


def test_parse_unknown_type_coerced_and_bad_weight() -> None:
    raw = (
        '{"entities": [{"name": "X", "type": "wizardry", "description": "d"}],'
        ' "relations": [{"source": "X", "target": "Y", "description": "r", "weight": "lots"}]}'
    )
    ex = _parse_extraction(raw, "c1")
    assert ex.entities[0].type == "concept"
    assert ex.relations[0].weight == 1.0


def test_parse_garbage_is_graceful() -> None:
    ex = _parse_extraction("the model could not produce json", "c9")
    assert ex.chunk_id == "c9" and not ex.entities and not ex.relations


async def test_extract_graph_aggregates() -> None:
    out = await extract_graph([_chunk(), _chunk()], llm=_StubLLM(), model="m")
    assert len(out) == 2
    assert sum(len(e.entities) for e in out) == 4


async def test_extract_graph_llm_failure_is_per_chunk() -> None:
    out = await extract_graph([_chunk()], llm=_StubLLM(boom=True), model="m")
    assert len(out) == 1 and not out[0].entities  # degraded, not raised


async def test_extract_graph_empty_input() -> None:
    assert await extract_graph([], llm=_StubLLM(), model="m") == []
