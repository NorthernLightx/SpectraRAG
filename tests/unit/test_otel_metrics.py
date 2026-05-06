"""Generator emits token counters + latency histogram."""

from __future__ import annotations

import importlib

import opentelemetry.metrics._internal as _otel_metrics_internal
import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import src.observability.metrics as metrics_mod
from src.llm.protocol import ChatResponse
from src.prompts.loader import Prompt
from src.rag.generate import Generator
from src.types import RetrievalResult


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> InMemoryMetricReader:
    # OTel only allows set_meter_provider once per process by default.
    # Reset the internal guard so the fixture can install a fresh provider.
    monkeypatch.setattr(_otel_metrics_internal._METER_PROVIDER_SET_ONCE, "_done", False)
    monkeypatch.setattr(_otel_metrics_internal, "_METER_PROVIDER", None)

    rdr = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[rdr])
    metrics.set_meter_provider(provider)
    importlib.reload(metrics_mod)  # rebind module-level instruments to new provider
    return rdr


class _FakeLLM:
    async def chat(self, messages, model, *, temperature, images=None):  # type: ignore[no-untyped-def]
        return ChatResponse(text="ok [c1]", model=model, tokens_in=10, tokens_out=20)


async def test_generator_emits_token_metrics(reader: InMemoryMetricReader) -> None:
    prompt = Prompt(name="t", version="v0", user_template="{query} {context}", system=None)
    gen = Generator(llm=_FakeLLM(), prompt=prompt, model="fake")  # type: ignore[arg-type]
    chunk = RetrievalResult(
        chunk_id="c1", paper_id="p", page_numbers=[1], text="t", score=1.0, source="pipeline"
    )
    answer = await gen.answer("q", [chunk])
    assert answer.tokens_in == 10

    data = reader.get_metrics_data()
    assert data is not None
    metric_names = {
        m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
    }
    assert "rag.tokens.in" in metric_names
    assert "rag.tokens.out" in metric_names
    assert "rag.generate.latency_ms" in metric_names
