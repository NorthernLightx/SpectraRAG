"""POST /demo/chat: the caged keyless demo proxy (ADR 0027).

Upstream OpenRouter is faked with httpx.MockTransport via the module's
`_make_client` hook, so the cage properties are asserted on the actual
outbound request: server-chosen ":free" model, max_price pinned to 0,
client-sent model ignored, fallback chain, and the global daily counter.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_settings
from src.api.main import create_app
from src.api.rate_limit import limiter
from src.api.routes import demo
from src.config.settings import Settings

_SSE_BODY = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    """Quota and limiter are module-level state; reset around each test."""
    demo._DemoQuota.day = ""
    demo._DemoQuota.used = 0
    limiter.reset()
    yield
    demo._DemoQuota.day = ""
    demo._DemoQuota.used = 0
    limiter.reset()


def _client(settings: Settings) -> TestClient:
    app = create_app(log_file=None)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def _demo_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"demo_openrouter_key": "sk-or-v1-demo"}
    base.update(overrides)
    return Settings(**base)


def _install_upstream(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[dict[str, Any]]:
    """Route the demo module's outbound httpx through a MockTransport.

    Returns a list that accumulates each upstream request's JSON body.
    """
    seen: list[dict[str, Any]] = []

    def _handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen.append(body)
        result = handler(body)
        assert isinstance(result, httpx.Response)
        return result

    def _fake_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(_handle))

    monkeypatch.setattr(demo, "_make_client", _fake_client)
    return seen


def test_503_when_no_demo_key() -> None:
    client = _client(Settings())
    res = client.post("/demo/chat", json={"messages": [{"role": "user", "content": "q"}]})
    assert res.status_code == 503


def test_streams_with_server_chosen_free_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_upstream(monkeypatch, lambda body: httpx.Response(200, content=_SSE_BODY))
    client = _client(_demo_settings())

    # A client-sent "model" must be dropped (no such field on the request).
    res = client.post(
        "/demo/chat",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "q"}]},
    )

    assert res.status_code == 200
    assert res.content == _SSE_BODY
    assert len(seen) == 1
    sent = seen[0]
    assert sent["model"] == "google/gemma-4-26b-a4b-it:free"
    assert sent["stream"] is True
    assert sent["max_tokens"] == demo._MAX_TOKENS
    # The cage: every price axis pinned to zero.
    assert sent["provider"]["max_price"] == {
        "prompt": 0,
        "completion": 0,
        "request": 0,
        "image": 0,
    }


def test_falls_back_when_primary_model_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(body: dict[str, Any]) -> httpx.Response:
        if body["model"].startswith("google/"):
            return httpx.Response(429, json={"error": "slammed"})
        return httpx.Response(200, content=_SSE_BODY)

    seen = _install_upstream(monkeypatch, handler)
    client = _client(_demo_settings())

    res = client.post("/demo/chat", json={"messages": [{"role": "user", "content": "q"}]})

    assert res.status_code == 200
    assert [b["model"] for b in seen] == [
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
    ]


def test_502_when_all_models_down(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_upstream(monkeypatch, lambda body: httpx.Response(503, json={"error": "down"}))
    client = _client(_demo_settings())
    res = client.post("/demo/chat", json={"messages": [{"role": "user", "content": "q"}]})
    assert res.status_code == 502


def test_non_free_models_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_upstream(monkeypatch, lambda body: httpx.Response(200, content=_SSE_BODY))
    client = _client(_demo_settings(demo_models="openai/gpt-4o, foo/bar:free"))

    res = client.post("/demo/chat", json={"messages": [{"role": "user", "content": "q"}]})

    assert res.status_code == 200
    assert [b["model"] for b in seen] == ["foo/bar:free"]


def test_503_when_only_paid_models_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_upstream(monkeypatch, lambda body: httpx.Response(200, content=_SSE_BODY))
    client = _client(_demo_settings(demo_models="openai/gpt-4o"))
    res = client.post("/demo/chat", json={"messages": [{"role": "user", "content": "q"}]})
    assert res.status_code == 503


def test_daily_cap_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_upstream(monkeypatch, lambda body: httpx.Response(200, content=_SSE_BODY))
    client = _client(_demo_settings(demo_daily_cap=1))

    first = client.post("/demo/chat", json={"messages": [{"role": "user", "content": "q"}]})
    second = client.post("/demo/chat", json={"messages": [{"role": "user", "content": "q"}]})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "demo_quota_exhausted"


def test_quota_resets_on_new_day() -> None:
    demo._DemoQuota.day = "2000-01-01"
    demo._DemoQuota.used = 300
    assert demo._DemoQuota.take(300) is True
    assert demo._DemoQuota.used == 1


def test_health_reports_demo_availability() -> None:
    on = _client(_demo_settings()).get("/health").json()
    off = _client(Settings()).get("/health").json()
    assert on["demo_available"] is True
    assert off["demo_available"] is False
