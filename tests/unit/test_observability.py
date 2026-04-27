"""Langfuse tracing wiring: no-op when not configured; emits a trace when client is set."""

from __future__ import annotations

from typing import Any

import pytest

from src.observability.langfuse import make_langfuse_client, trace_query
from src.types import Answer, Citation, Query, RetrievalResult


class _FakeLangfuse:
    """Records calls to verify trace shape."""

    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []
        self.flush_count = 0

    def trace(self, *, name: str, input: dict[str, Any], output: dict[str, Any]) -> Any:
        self.traces.append({"name": name, "input": input, "output": output})

    def flush(self) -> None:
        self.flush_count += 1


def _query() -> Query:
    return Query(text="What is X?", top_k=3)


def _retrieved() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="c1",
            paper_id="p1",
            score=0.9,
            text="snippet",
            page_numbers=[1],
            source="pipeline",
        )
    ]


def _answer() -> Answer:
    return Answer(
        text="The answer.",
        citations=[Citation(chunk_id="c1", paper_id="p1", page_numbers=[1])],
        model="anthropic/claude-3.5-sonnet",
        prompt_version="v1-abc",
        latency_ms=500,
        tokens_in=120,
        tokens_out=30,
    )


def test_trace_query_no_op_when_client_is_none() -> None:
    trace_query(None, query=_query(), retrieved=_retrieved(), answer=_answer())  # must not raise


def test_trace_query_emits_one_trace_with_query_and_answer() -> None:
    fake = _FakeLangfuse()
    trace_query(fake, query=_query(), retrieved=_retrieved(), answer=_answer())

    assert len(fake.traces) == 1
    trace = fake.traces[0]
    assert trace["name"] == "rag.query"
    assert trace["input"]["query"] == "What is X?"
    assert trace["output"]["n_retrieved"] == 1
    assert trace["output"]["retrieved_chunk_ids"] == ["c1"]
    assert trace["output"]["answer_model"] == "anthropic/claude-3.5-sonnet"
    assert trace["output"]["prompt_version"] == "v1-abc"
    assert trace["output"]["cited_chunk_ids"] == ["c1"]
    assert fake.flush_count == 1


def test_trace_query_without_answer_omits_generation_fields() -> None:
    fake = _FakeLangfuse()
    trace_query(fake, query=_query(), retrieved=_retrieved(), answer=None)

    output = fake.traces[0]["output"]
    assert "answer_text" not in output
    assert output["n_retrieved"] == 1


def test_make_langfuse_client_returns_none_when_keys_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("RAG_LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("RAG_LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("RAG_LANGFUSE_HOST", raising=False)
    assert make_langfuse_client() is None
