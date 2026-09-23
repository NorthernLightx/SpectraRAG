"""Per-IP rate limit on /answer (slowapi at 10/minute).

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
from starlette.requests import Request

from src.api.deps import _GeneratorState, _RetrieverState, get_generator, get_retriever
from src.api.main import create_app
from src.api.rate_limit import client_key, limiter
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
    directly. We return Pydantic Answer models, so it stays off."""
    app = _wire_app()
    client = TestClient(app)
    for _ in range(10):
        client.post("/answer", json={"text": "anything"})
    r = client.post("/answer", json={"text": "anything"})
    assert r.status_code == 429
    assert r.headers.get("content-type", "").startswith("application/json")
    body = r.json()
    assert "error" in body or "detail" in body


def _burn_answer_bucket(client: TestClient, **headers: str) -> int:
    for _ in range(10):
        client.post("/answer", json={"text": "anything"}, headers=headers)
    status: int = client.post("/answer", json={"text": "anything"}, headers=headers).status_code
    return status


def test_on_cloud_run_each_forwarded_client_gets_its_own_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every request reaches the container from Google's front end, so keying
    on the socket peer would share one bucket across all visitors."""
    monkeypatch.setenv("K_SERVICE", "spectrarag")
    client = TestClient(_wire_app())
    assert _burn_answer_bucket(client, **{"X-Forwarded-For": "203.0.113.1"}) == 429
    r = client.post("/answer", json={"text": "x"}, headers={"X-Forwarded-For": "203.0.113.2"})
    assert r.status_code == 200


def test_a_client_supplied_forwarded_for_does_not_open_a_new_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The front end appends the real address after anything the client sent,
    so rotating the client-supplied entry changes nothing."""
    monkeypatch.setenv("K_SERVICE", "spectrarag")
    client = TestClient(_wire_app())
    assert _burn_answer_bucket(client, **{"X-Forwarded-For": "1.1.1.1, 203.0.113.1"}) == 429
    r = client.post(
        "/answer", json={"text": "x"}, headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.1"}
    )
    assert r.status_code == 429


def _request(*forwarded_for: str, peer: str = "10.0.0.1") -> Request:
    headers = [(b"x-forwarded-for", value.encode()) for value in forwarded_for]
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_on_cloud_run_a_second_forwarded_for_line_does_not_hide_the_appended_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated header lines are one comma-joined list, and the front end's hop
    is the last entry of it."""
    monkeypatch.setenv("K_SERVICE", "spectrarag")
    assert client_key(_request("9.9.9.9", "1.1.1.1, 203.0.113.1")) == "203.0.113.1"


def test_ipv6_clients_share_one_bucket_per_64(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client holds a whole /64 and can rotate addresses inside it freely."""
    monkeypatch.setenv("K_SERVICE", "spectrarag")
    first = client_key(_request("2001:db8:1:2::1"))
    assert first == client_key(_request("2001:db8:1:2:ffff::9"))
    assert first != client_key(_request("2001:db8:1:3::1"))
    monkeypatch.delenv("K_SERVICE")
    assert client_key(_request(peer="2001:db8:1:2::1")) == first


def test_off_cloud_run_the_header_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    client = TestClient(_wire_app())
    assert _burn_answer_bucket(client, **{"X-Forwarded-For": "203.0.113.1"}) == 429
    r = client.post("/answer", json={"text": "x"}, headers={"X-Forwarded-For": "203.0.113.2"})
    assert r.status_code == 429
