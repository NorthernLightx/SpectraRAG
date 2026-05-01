"""Module-level OTel instruments. Importing this module registers the names.

Usage:
    from src.observability.metrics import (
        TOKENS_IN, TOKENS_OUT, GENERATE_LATENCY_MS, ERRORS
    )
    TOKENS_IN.add(response.tokens_in, attributes={"model": response.model})

The instruments bind to whatever MeterProvider is current at import time;
tests can swap the provider then `importlib.reload` this module.
"""

from __future__ import annotations

from src.observability.otel import get_meter

_meter = get_meter()

TOKENS_IN = _meter.create_counter(
    "rag.tokens.in",
    unit="tokens",
    description="Total prompt tokens sent to the LLM.",
)
TOKENS_OUT = _meter.create_counter(
    "rag.tokens.out",
    unit="tokens",
    description="Total completion tokens produced by the LLM.",
)
GENERATE_LATENCY_MS = _meter.create_histogram(
    "rag.generate.latency_ms",
    unit="ms",
    description="LLM generate call latency in milliseconds.",
)
ERRORS = _meter.create_counter(
    "rag.errors",
    unit="errors",
    description="Errors raised inside the RAG pipeline, by exception class.",
)
