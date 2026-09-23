"""POST /query/dci: per-client rate limit and provider rejections."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routes.dci as dci_route
from src.api.deps import get_chunks, get_settings
from src.api.main import create_app
from src.api.rate_limit import limiter
from src.config.settings import Settings
from src.dci.agent import DciResult
from src.dci.tools import CorpusTools
from src.types import Query, RetrievalResult


class _StubRetriever:
    """Stands in for DciRetriever: no model calls, optional provider failure."""

    fail_with: int | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def run(self, query: Query) -> tuple[list[RetrievalResult], DciResult]:
        if self.fail_with is not None:
            request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            response = httpx.Response(self.fail_with, request=request)
            raise httpx.HTTPStatusError("provider error", request=request, response=response)
        return [], DciResult(question=query.text, mode="retrieval", answer=None, ranked_doc_ids=[])


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(dci_route, "DciRetriever", _StubRetriever)
    monkeypatch.setattr(dci_route, "build_dci_corpus", lambda chunks: (CorpusTools({}), {}))
    monkeypatch.setattr(dci_route._DciCorpusState, "tools", None)
    monkeypatch.setattr(_StubRetriever, "fail_with", None)
    limiter.reset()
    yield
    limiter.reset()


def _client() -> TestClient:
    app: FastAPI = create_app(log_file=None)
    app.dependency_overrides[get_settings] = lambda: Settings(enable_dci=True)
    app.dependency_overrides[get_chunks] = lambda: {}
    return TestClient(app)


def _ask(client: TestClient) -> httpx.Response:
    response: httpx.Response = client.post(
        "/query/dci", json={"text": "What is X?"}, headers={"X-OpenRouter-Key": "sk-test"}
    )
    return response


def test_query_dci_limits_each_client_to_five_a_minute() -> None:
    client = _client()
    for i in range(5):
        assert _ask(client).status_code == 200, f"request {i + 1}"
    assert _ask(client).status_code == 429


def test_query_dci_returns_502_naming_the_provider_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_StubRetriever, "fail_with", 401)

    response = _ask(_client())

    assert response.status_code == 502
    assert "HTTP 401" in response.json()["detail"]
