"""Per-IP rate limit on /answer (Phase 2.2 — slowapi at 10/minute).

Tests wire FakeRetriever + a stub generator via dependency_overrides so /answer
returns 200 instead of the 503 from the unset-retriever guard. The slowapi
decorator only counts requests that reach the route function; if Depends raises
HTTPException(503) before the route runs, the bucket never fills. Production
behaviour is correct (real deploys have a wired retriever) — this just makes
the test exercise the rate-limit code path explicitly.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.deps import _GeneratorState, _RetrieverState, get_generator, get_retriever
from src.api.main import create_app
from src.api.rate_limit import limiter
from src.types import Answer, RetrievalResult
from tests.fakes import FakeRetriever


class _StubGenerator:
    """Returns a canned Answer regardless of input — keeps /answer hot so the
    slowapi decorator counts every call."""

    async def answer(self, query: str, retrieved: list[RetrievalResult]) -> Answer:
        return Answer(
            text="ok",
            citations=[],
            model="stub",
            prompt_version="stub-v1",
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
        )


def _wire_app() -> FastAPI:
    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever(results=[])
    app.dependency_overrides[get_generator] = lambda: _StubGenerator()
    return app


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The limiter is module-level — reset between tests so attempts from
    earlier cases don't bleed into later ones (TestClient routes everything
    through the same `testclient` host)."""
    monkeypatch.delenv("RAG_PUBLIC_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OPENROUTER_API_KEY", raising=False)
    _GeneratorState.instance = None
    _RetrieverState.instance = None
    limiter.reset()
    yield
    limiter.reset()


def test_eleventh_request_in_window_returns_429() -> None:
    """First 10 succeed; 11th hits the limit."""
    app = _wire_app()
    client = TestClient(app)
    for i in range(10):
        r = client.post("/answer", json={"text": "anything"})
        assert r.status_code == 200, f"req {i + 1}: expected 200, got {r.status_code}"
    r = client.post("/answer", json={"text": "anything"})
    assert r.status_code == 429


def test_other_endpoints_are_not_rate_limited() -> None:
    """The 10/min cap is on /answer specifically — /query and /health stay open
    even after /answer's bucket is exhausted."""
    app = _wire_app()
    client = TestClient(app)
    for _ in range(11):
        client.post("/answer", json={"text": "anything"})
    assert client.post("/query", json={"text": "anything"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_429_response_is_json_with_detail() -> None:
    """The 429 body is JSON — clients can parse the error programmatically.
    Retry-After is intentionally not asserted: slowapi only adds it under
    `headers_enabled=True`, which requires every route to return Response
    directly. We return Pydantic Answer models, so it stays off; adding
    Retry-After would need a custom 429 handler — Phase 3.2.1 candidate."""
    app = _wire_app()
    client = TestClient(app)
    for _ in range(10):
        client.post("/answer", json={"text": "anything"})
    r = client.post("/answer", json={"text": "anything"})
    assert r.status_code == 429
    assert r.headers.get("content-type", "").startswith("application/json")
    body = r.json()
    assert "error" in body or "detail" in body
