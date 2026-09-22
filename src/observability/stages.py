"""Wall-clock time per retrieval stage, for one query.

`collect_stages()` opens a collector; `stage(name)` adds its block's duration
to it and records an OTel span. The collector lives in a context variable and
holds a dict, so the router's concurrent legs (asyncio tasks) and the work they
hand to `asyncio.to_thread` (which copies the context) all write to the same
one. Outside a collector, `stage` only records the span.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from src.observability.otel import get_tracer

_stages_var: ContextVar[dict[str, float] | None] = ContextVar("stage_ms", default=None)


@contextmanager
def stage(name: str) -> Iterator[None]:
    started = time.perf_counter()
    with get_tracer().start_as_current_span(f"rag.{name}"):
        try:
            yield
        finally:
            sink = _stages_var.get()
            if sink is not None:
                sink[name] = sink.get(name, 0.0) + (time.perf_counter() - started) * 1000


@contextmanager
def collect_stages() -> Iterator[dict[str, float]]:
    """Milliseconds per stage name for the block, summed when a stage repeats."""
    sink: dict[str, float] = {}
    token = _stages_var.set(sink)
    try:
        yield sink
    finally:
        _stages_var.reset(token)


def rounded(stages: dict[str, float]) -> dict[str, float]:
    return {name: round(ms, 1) for name, ms in stages.items()}
