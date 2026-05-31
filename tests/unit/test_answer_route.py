"""POST /answer: retrieve + generate, returns Answer; 503 when generator not set."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.deps import (
    _GeneratorState,
    _RetrieverState,
    get_generator,
    get_retriever,
    get_settings,
    get_tracer,
)
from src.api.main import create_app
from src.config.settings import Settings
from src.types import Answer, Citation, Query, RetrievalResult
from tests.fakes import FakeRetriever


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Module-level _GeneratorState / _RetrieverState leak across tests; reset around each.

    Also clears RAG_OPENROUTER_API_KEY so create_app() default tests don't see a key
    leaked from the dev shell.
    """
    monkeypatch.delenv("RAG_OPENROUTER_API_KEY", raising=False)
    _GeneratorState.instance = None
    _RetrieverState.instance = None
    yield
    _GeneratorState.instance = None
    _RetrieverState.instance = None


class _StubGenerator:
    """Generator stub: returns a canned Answer regardless of input."""

    def __init__(self, answer: Answer) -> None:
        self._answer = answer
        self.calls: list[tuple[str, list[RetrievalResult]]] = []

    async def answer(self, query: str, retrieved: list[RetrievalResult]) -> Answer:
        self.calls.append((query, retrieved))
        return self._answer


def _retrieved() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="c1",
            paper_id="p1",
            score=0.9,
            text="snippet one",
            page_numbers=[1],
            source="pipeline",
        )
    ]


def _answer_payload() -> Answer:
    return Answer(
        text="The answer cites [c1].",
        citations=[Citation(chunk_id="c1", paper_id="p1", page_numbers=[1])],
        model="anthropic/claude-3.5-sonnet",
        prompt_version="answer/v1-abc",
        latency_ms=200,
        tokens_in=80,
        tokens_out=24,
    )


def test_answer_route_calls_retriever_then_generator_then_returns_answer() -> None:
    retriever = FakeRetriever(results=_retrieved())
    generator = _StubGenerator(_answer_payload())
    captured_traces: list[dict[str, Any]] = []

    def fake_tracer() -> Any:
        class _T:
            def trace(self, *, name: str, input: dict[str, Any], output: dict[str, Any]) -> Any:
                captured_traces.append({"name": name, "input": input, "output": output})

            def flush(self) -> None:
                pass

        return _T()

    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_tracer] = fake_tracer

    client = TestClient(app)
    response = client.post("/answer", json={"text": "What is X?", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "The answer cites [c1]."
    assert body["citations"][0]["chunk_id"] == "c1"
    assert body["model"] == "anthropic/claude-3.5-sonnet"

    # Generator received the retrieved results
    assert len(generator.calls) == 1
    assert generator.calls[0][0] == "What is X?"
    assert generator.calls[0][1][0].chunk_id == "c1"

    # Tracer fired one trace named rag.query
    assert len(captured_traces) == 1
    assert captured_traces[0]["name"] == "rag.query"


def test_answer_route_returns_503_when_generator_unset() -> None:
    retriever = FakeRetriever(results=_retrieved())
    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever
    # Don't set generator → should 503

    client = TestClient(app)
    response = client.post("/answer", json={"text": "x"})
    assert response.status_code == 503
    assert "generator" in response.json()["detail"].lower()


def test_answer_route_validates_input() -> None:
    retriever = FakeRetriever(results=[])
    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: _StubGenerator(_answer_payload())

    client = TestClient(app)
    response = client.post("/answer", json={"text": ""})
    assert response.status_code == 422


def test_answer_route_works_without_tracer() -> None:
    """No tracer configured → trace_query receives None → no-op (request still 200)."""
    retriever = FakeRetriever(results=_retrieved())
    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: _StubGenerator(_answer_payload())
    # tracer left as default (None)

    client = TestClient(app)
    response = client.post("/answer", json={"text": "x"})
    assert response.status_code == 200


def _unused_query() -> Query:
    return Query(text="placeholder")


def test_create_app_wires_generator_when_openrouter_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG_OPENROUTER_API_KEY in env → create_app() builds + registers a Generator.

    No HTTP traffic asserted; we're verifying the wiring path
    Settings → OpenRouterClient → Generator → set_generator().
    """
    from src.rag.generate import Generator

    monkeypatch.setenv("RAG_OPENROUTER_API_KEY", "sk-or-v1-test")
    create_app(log_file=None)
    assert isinstance(_GeneratorState.instance, Generator)


def test_create_app_does_not_wire_generator_when_key_unset() -> None:
    """Without the key, _GeneratorState stays None — /answer returns 503 by design."""
    create_app(log_file=None)
    assert _GeneratorState.instance is None


# --- ADR 0024 route-by-fit on /answer ------------------------------------


class _TrackingRetriever:
    """Retriever stub that records whether retrieve() ran — lets the route-by-fit
    tests assert the whole-doc path skipped retrieval entirely. FakeRetriever
    doesn't track calls, so the skip assertion needs this."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[Query] = []

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        self.calls.append(query)
        return self._results[: query.top_k]


def _two_page_doc(pages_dir: Path, paper_id: str, n_pages: int) -> None:
    paper_dir = pages_dir / paper_id
    paper_dir.mkdir(parents=True)
    for p in range(1, n_pages + 1):
        (paper_dir / f"{paper_id}_p{p}.png").write_bytes(b"\x89PNG\r\n")


def test_answer_whole_doc_when_paper_scoped_and_fits(tmp_path: Path) -> None:
    # Paper-scoped query + budget + doc fits -> feed the whole document's page
    # images and SKIP retrieval entirely (route-by-fit, ADR 0024).
    _two_page_doc(tmp_path, "paperX", 2)
    retriever = _TrackingRetriever(_retrieved())
    generator = _StubGenerator(_answer_payload())

    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_settings] = lambda: Settings(page_budget=5, pages_dir=tmp_path)

    client = TestClient(app)
    resp = client.post("/answer", json={"text": "q", "filters": {"paper_id": "paperX"}})

    assert resp.status_code == 200
    assert retriever.calls == []  # retrieval skipped on the whole-doc path
    fed = generator.calls[0][1]
    assert len(fed) == 2
    assert all(r.source == "visual" for r in fed)
    assert {r.chunk_id for r in fed} == {"paperX::p1::page", "paperX::p2::page"}


def test_answer_falls_back_to_rag_when_not_paper_scoped(tmp_path: Path) -> None:
    # Budget set but the query names no paper -> RAG path (retriever is called).
    retriever = _TrackingRetriever(_retrieved())
    generator = _StubGenerator(_answer_payload())

    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_settings] = lambda: Settings(page_budget=5, pages_dir=tmp_path)

    client = TestClient(app)
    resp = client.post("/answer", json={"text": "q"})

    assert resp.status_code == 200
    assert len(retriever.calls) == 1
    assert generator.calls[0][1][0].chunk_id == "c1"


def test_answer_falls_back_to_rag_when_doc_over_budget(tmp_path: Path) -> None:
    # Paper-scoped but the doc exceeds the budget -> RAG (whole-doc would blow context).
    _two_page_doc(tmp_path, "paperBig", 3)
    retriever = _TrackingRetriever(_retrieved())
    generator = _StubGenerator(_answer_payload())

    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_settings] = lambda: Settings(page_budget=2, pages_dir=tmp_path)

    client = TestClient(app)
    resp = client.post("/answer", json={"text": "q", "filters": {"paper_id": "paperBig"}})

    assert resp.status_code == 200
    assert len(retriever.calls) == 1


def test_answer_falls_back_to_rag_when_budget_unset(tmp_path: Path) -> None:
    # Default (page_budget None): the top-k path is unchanged even for a scoped query.
    _two_page_doc(tmp_path, "paperX", 2)
    retriever = _TrackingRetriever(_retrieved())
    generator = _StubGenerator(_answer_payload())

    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_settings] = lambda: Settings(pages_dir=tmp_path)

    client = TestClient(app)
    resp = client.post("/answer", json={"text": "q", "filters": {"paper_id": "paperX"}})

    assert resp.status_code == 200
    assert len(retriever.calls) == 1
