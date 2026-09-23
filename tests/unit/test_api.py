"""API surface: /health is reachable, /query routes to the configured retriever."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_retriever
from src.api.main import create_app
from src.types import RetrievalResult
from tests.fakes import FakeRetriever


def _make_client(retriever: FakeRetriever | None = None) -> TestClient:
    app = create_app(log_file=None)
    if retriever is not None:
        app.dependency_overrides[get_retriever] = lambda: retriever
    return TestClient(app)


def test_health_returns_ok() -> None:
    client = _make_client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "env" in body
    # pages_available drives the BYOK frontend's decision to attach image
    # content blocks to OpenRouter calls. Always present as a bool.
    assert isinstance(body["pages_available"], bool)
    # routing_available tells the UI whether force_route/routing_mode do
    # anything on this deployment (False on CPU-only deploys with no router).
    assert isinstance(body["routing_available"], bool)


def test_query_returns_results_from_retriever() -> None:
    fake = FakeRetriever(
        results=[
            RetrievalResult(
                chunk_id="c1",
                paper_id="p1",
                score=0.9,
                text="hit text",
                page_numbers=[1],
                source="pipeline",
            )
        ]
    )
    client = _make_client(retriever=fake)
    response = client.post("/query", json={"text": "x", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["chunk_id"] == "c1"
    assert body["results"][0]["source"] == "pipeline"


def test_query_validates_input() -> None:
    client = _make_client(retriever=FakeRetriever(results=[]))
    response = client.post("/query", json={"text": ""})
    assert response.status_code == 422


def test_query_rejects_text_longer_than_the_context_question_cap() -> None:
    # BM25 and ColQwen2 cost grows with query length; one huge body would pin the
    # single serving instance.
    client = _make_client(retriever=FakeRetriever(results=[]))
    assert client.post("/query", json={"text": "a" * 4000}).status_code == 200
    assert client.post("/query", json={"text": "a" * 4001}).status_code == 422


def test_query_returns_503_when_retriever_unset() -> None:
    client = TestClient(create_app(log_file=None))
    response = client.post("/query", json={"text": "x"})
    assert response.status_code == 503
    assert "ingest a corpus" in response.json()["detail"].lower()


def test_root_redirects_to_api_docs() -> None:
    """The repo ships no UI, so `/` points a browser at the interactive API docs."""
    client = _make_client()
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"
    assert client.get("/docs").status_code == 200


def test_pages_mount_serves_png_when_pages_dir_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When settings.pages_dir points at a populated directory, page PNGs are
    served at /pages/<paper>/<file> so the BYOK frontend can include them as
    OpenRouter image content blocks."""
    from src.api.deps import get_settings

    paper_dir = tmp_path / "paper-x"
    paper_dir.mkdir()
    png = paper_dir / "paper-x_p1.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    monkeypatch.setenv("RAG_PAGES_DIR", str(tmp_path))
    # get_settings is @lru_cache; previous tests may have cached a Settings
    # without RAG_PAGES_DIR set. Clear so /health re-reads the env-derived
    # value the new app instance also sees.
    get_settings.cache_clear()

    client = TestClient(create_app(log_file=None))
    response = client.get("/pages/paper-x/paper-x_p1.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    health = client.get("/health").json()
    assert health["pages_available"] is True


def test_pages_mount_absent_when_pages_dir_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """No RAG_PAGES_DIR → /pages mount is skipped → /pages/* falls through
    to the static-frontend mount → 404 for non-index paths. Health flag is
    False so the BYOK frontend skips image content blocks entirely."""
    from src.api.deps import get_settings

    monkeypatch.delenv("RAG_PAGES_DIR", raising=False)
    get_settings.cache_clear()
    client = TestClient(create_app(log_file=None))
    response = client.get("/pages/paper-x/paper-x_p1.png")
    assert response.status_code == 404
    health = client.get("/health").json()
    assert health["pages_available"] is False
