"""End-to-end: configure OTel with an in-memory exporter, drive /health, see spans."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import src.observability.otel as otel_mod


def _reset_otel_global_provider(provider: TracerProvider) -> None:
    """Force-replace the OTel global tracer provider, bypassing the write-once guard.

    `opentelemetry.trace._set_tracer_provider` uses a `Once` object that blocks
    repeated calls. In tests we need to supply a fresh in-memory provider; we
    reset both the guard flag and the global pointer directly.
    """
    import opentelemetry.trace as _trace_mod

    _trace_mod._TRACER_PROVIDER_SET_ONCE._done = False
    _trace_mod._TRACER_PROVIDER = None
    trace.set_tracer_provider(provider)


@pytest.fixture
def in_memory_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Force an in-memory tracer provider before create_app()."""
    importlib.reload(otel_mod)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Override before create_app() runs so FastAPI auto-instr binds to ours.
    _reset_otel_global_provider(provider)
    # Also short-circuit configure_otel so it doesn't replace our provider.
    otel_mod._configured = True
    return exporter


def test_health_request_emits_span(in_memory_spans: InMemorySpanExporter) -> None:
    from src.api.main import create_app

    app = create_app(log_file=None)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    spans: list[Any] = list(in_memory_spans.get_finished_spans())
    span_names = [s.name for s in spans]
    # FastAPI auto-instrumentation names the span after the route or method.
    assert any("/health" in name or "GET" in name for name in span_names), span_names


def test_answer_route_emits_retrieve_and_generate_spans(
    in_memory_spans: InMemorySpanExporter,
) -> None:
    from src.api.deps import set_generator, set_retriever
    from src.api.main import create_app
    from src.types import Answer, Query, RetrievalResult

    class FakeRetriever:
        async def retrieve(self, q: Query) -> list[RetrievalResult]:
            return []

    class FakeGenerator:
        async def answer(self, text: str, retrieved: list[RetrievalResult]) -> Answer:
            return Answer(
                text="ok",
                citations=[],
                model="fake",
                prompt_version="v0",
                latency_ms=1,
                tokens_in=1,
                tokens_out=1,
            )

    set_retriever(FakeRetriever())
    set_generator(FakeGenerator())  # type: ignore[arg-type]
    app = create_app(log_file=None)
    client = TestClient(app)

    response = client.post("/answer", json={"text": "hi", "top_k": 5})
    assert response.status_code == 200, response.text

    span_names = {s.name for s in in_memory_spans.get_finished_spans()}
    assert "rag.retrieve" in span_names, span_names
    assert "rag.generate" in span_names, span_names
