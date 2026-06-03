"""OpenTelemetry SDK init. No-op when OTEL_EXPORTER_OTLP_ENDPOINT is unset.

Idempotent. Mirrors the langfuse / sentry pattern: SDK env vars stay out of
`Settings`. After `configure_otel()`, FastAPI / httpx auto-instrumentation
emit spans automatically; manual spans use `get_tracer()`; metrics use
`get_meter()`.
"""

from __future__ import annotations

import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from src.observability.logging import get_logger

_log = get_logger(__name__)
_configured = False
_INSTRUMENTATION_NAME = "src.rag"


def configure_otel() -> bool:
    """Initialise tracer and meter providers if OTLP endpoint is set.

    Returns True when configured (either now or previously), False when no
    endpoint is set. Auto-instrumentation for FastAPI / httpx is wired in
    `create_app()` — splitting it keeps this fn pure and testable.
    """
    global _configured
    if _configured:
        return True
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    service_name = os.environ.get("OTEL_SERVICE_NAME", "spectrarag")
    resource = Resource.create({"service.name": service_name})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))],
    )
    metrics.set_meter_provider(meter_provider)

    _configured = True
    _log.info("otel.configured", endpoint=endpoint, service_name=service_name)
    return True


def get_tracer() -> Tracer:
    """Return a tracer for manual spans. Returns a no-op tracer if uninitialised."""
    return trace.get_tracer(_INSTRUMENTATION_NAME)


def get_meter() -> metrics.Meter:
    """Return a meter for manual instruments. Returns a no-op meter if uninitialised."""
    return metrics.get_meter(_INSTRUMENTATION_NAME)
