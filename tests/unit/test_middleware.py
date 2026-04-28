"""Request-ID middleware: mint or propagate, bind to structlog, echo header."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import REQUEST_ID_HEADER, request_context_middleware


def _build_app() -> tuple[FastAPI, dict[str, Any]]:
    """Return an app whose /echo route captures the contextvars bound during the request."""
    app = FastAPI()
    app.middleware("http")(request_context_middleware)
    captured: dict[str, Any] = {}

    @app.get("/echo")
    def echo() -> dict[str, str]:
        captured.update(structlog.contextvars.get_contextvars())
        return {"ok": "yes"}

    return app, captured


def test_minted_when_header_absent() -> None:
    app, captured = _build_app()
    client = TestClient(app)
    response = client.get("/echo")

    assert response.status_code == 200
    assert REQUEST_ID_HEADER in response.headers
    rid = response.headers[REQUEST_ID_HEADER]
    assert len(rid) == 32  # uuid4().hex

    assert captured["request_id"] == rid
    assert captured["method"] == "GET"
    assert captured["path"] == "/echo"


def test_inbound_header_propagated_unchanged() -> None:
    app, captured = _build_app()
    client = TestClient(app)
    response = client.get("/echo", headers={REQUEST_ID_HEADER: "trace-abc-123"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"
    assert captured["request_id"] == "trace-abc-123"


def test_contextvars_cleared_after_response() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    client.get("/echo", headers={REQUEST_ID_HEADER: "rid-x"})

    # Middleware clears contextvars in `finally` after each response.
    assert structlog.contextvars.get_contextvars() == {}


def test_each_request_gets_distinct_id() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    a = client.get("/echo").headers[REQUEST_ID_HEADER]
    b = client.get("/echo").headers[REQUEST_ID_HEADER]
    assert a != b
