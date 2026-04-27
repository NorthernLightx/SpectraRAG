"""API surface: /health is reachable and /query is wired but not yet implemented."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "env" in body


def test_query_placeholder_returns_501() -> None:
    client = TestClient(create_app())
    response = client.post("/query", json={"text": "What is X?"})
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()


def test_query_validates_input() -> None:
    client = TestClient(create_app())
    response = client.post("/query", json={"text": ""})
    assert response.status_code == 422


def test_root_returns_service_name() -> None:
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "multi-modal" in response.json()["service"].lower()
