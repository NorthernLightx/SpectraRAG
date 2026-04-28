"""API surface: /health is reachable, /query routes to the configured retriever."""

from __future__ import annotations

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
    assert body[0]["chunk_id"] == "c1"
    assert body[0]["source"] == "pipeline"


def test_query_validates_input() -> None:
    client = _make_client(retriever=FakeRetriever(results=[]))
    response = client.post("/query", json={"text": ""})
    assert response.status_code == 422


def test_query_returns_503_when_retriever_unset() -> None:
    client = TestClient(create_app(log_file=None))
    response = client.post("/query", json={"text": "x"})
    assert response.status_code == 503
    assert "ingest a corpus" in response.json()["detail"].lower()


def test_root_returns_service_name() -> None:
    client = _make_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "multi-modal" in response.json()["service"].lower()
