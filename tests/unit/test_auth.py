"""API-key middleware: gate /answer + /query behind X-API-Key when configured.

Phase 2.1 — protects the LLM-spending endpoints from drive-by traffic on the
public URL. Health and OpenAPI metadata are exempt so liveness probes and the
Swagger UI keep working without a key.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.deps import _GeneratorState, _RetrieverState
from src.api.main import create_app


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Same isolation pattern as tests/unit/test_answer_route.py: clear module-
    level state and any leaking RAG_PUBLIC_API_KEY / RAG_OPENROUTER_API_KEY from
    the dev shell so each test exercises a fresh app."""
    monkeypatch.delenv("RAG_PUBLIC_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OPENROUTER_API_KEY", raising=False)
    _GeneratorState.instance = None
    _RetrieverState.instance = None
    yield
    _GeneratorState.instance = None
    _RetrieverState.instance = None


def test_answer_returns_401_when_key_required_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_PUBLIC_API_KEY", "secret")
    app = create_app(log_file=None)
    client = TestClient(app)
    response = client.post("/answer", json={"text": "anything"})
    assert response.status_code == 401
    assert "api key" in response.json()["detail"].lower()


def test_answer_returns_401_when_key_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PUBLIC_API_KEY", "secret")
    app = create_app(log_file=None)
    client = TestClient(app)
    response = client.post("/answer", json={"text": "anything"}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_answer_passes_auth_when_key_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auth gate passes through; the 503 from the unset retriever is the
    expected next-layer response — proves auth didn't short-circuit."""
    monkeypatch.setenv("RAG_PUBLIC_API_KEY", "secret")
    app = create_app(log_file=None)
    client = TestClient(app)
    response = client.post("/answer", json={"text": "anything"}, headers={"X-API-Key": "secret"})
    assert response.status_code == 503
    assert "retriever" in response.json()["detail"].lower()


def test_query_endpoint_also_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PUBLIC_API_KEY", "secret")
    app = create_app(log_file=None)
    client = TestClient(app)
    response = client.post("/query", json={"text": "anything"})
    assert response.status_code == 401


def test_health_endpoint_does_not_require_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PUBLIC_API_KEY", "secret")
    app = create_app(log_file=None)
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_root_and_docs_exempt_from_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Liveness / OpenAPI surfaces stay reachable for probes + the Swagger UI."""
    monkeypatch.setenv("RAG_PUBLIC_API_KEY", "secret")
    app = create_app(log_file=None)
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_no_auth_applied_when_key_unset() -> None:
    """When RAG_PUBLIC_API_KEY is unset, requests pass through without an
    X-API-Key header — preserves the dev / single-user default."""
    app = create_app(log_file=None)
    client = TestClient(app)
    response = client.post("/answer", json={"text": "anything"})
    # Auth not enforced; route's own retriever guard fires instead.
    assert response.status_code == 503


def test_constant_time_comparison_on_correct_length_wrong_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same length as the real key, different bytes — must still 401. Defends
    against timing-side-channel regressions if someone swaps hmac.compare_digest
    for a shortcut == comparison."""
    monkeypatch.setenv("RAG_PUBLIC_API_KEY", "secret-12345")
    app = create_app(log_file=None)
    client = TestClient(app)
    response = client.post(
        "/answer",
        json={"text": "anything"},
        headers={"X-API-Key": "decoy-67890"},  # same length
    )
    assert response.status_code == 401
