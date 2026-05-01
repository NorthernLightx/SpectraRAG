"""configure_otel: no-op without endpoint; tracer/meter retrievable when configured."""

from __future__ import annotations

import importlib

import pytest

import src.observability.otel as otel_mod
from src.observability.otel import configure_otel, get_meter, get_tracer


@pytest.fixture(autouse=True)
def _reset_otel_module() -> None:
    importlib.reload(otel_mod)


def test_configure_otel_noop_without_endpoint() -> None:
    assert configure_otel() is False


def test_configure_otel_with_endpoint_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "rag-test")
    assert configure_otel() is True


def test_configure_otel_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    assert configure_otel() is True
    assert configure_otel() is True  # second call is a no-op, still True


def test_tracer_and_meter_always_returnable() -> None:
    # Even without configure_otel(), the OTel API returns no-op tracers/meters.
    tracer = get_tracer()
    meter = get_meter()
    assert tracer is not None
    assert meter is not None
    span = tracer.start_span("noop-test")
    span.end()
