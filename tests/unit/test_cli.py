"""spectrarag CLI: dispatch + self-contained serve defaults."""

import os
from unittest import mock

import pytest

from src.cli import main
from src.config.settings import load_settings


def test_serve_sets_self_contained_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("RAG_PROFILE", "RAG_QDRANT_URL", "RAG_PAGES_DIR"):
        monkeypatch.delenv(var, raising=False)
    with mock.patch("uvicorn.run") as run:
        rc = main(["serve", "--port", "9000"])
    assert rc == 0
    run.assert_called_once()
    assert run.call_args.args[0] == "src.api.main:app"
    assert run.call_args.kwargs["port"] == 9000
    assert os.environ["RAG_PROFILE"] == "cpu"
    assert os.environ["RAG_QDRANT_URL"] == "path:./qdrant_local"
    settings = load_settings()
    assert settings.embedder_backend == "sentence_transformers"
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_serve_respects_user_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDER_BACKEND", "ollama")
    with mock.patch("uvicorn.run"):
        main(["serve"])
    # The profile fills defaults; an explicit env var still wins over it.
    assert load_settings().embedder_backend == "ollama"


def test_serve_overrides_docker_default_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    # .env.example ships the docker default; the self-contained serve must still
    # use the embedded snapshot rather than a Qdrant server that isn't running.
    monkeypatch.setenv("RAG_QDRANT_URL", "http://localhost:6333")
    with mock.patch("uvicorn.run"):
        main(["serve"])
    assert os.environ["RAG_QDRANT_URL"] == "path:./qdrant_local"


def test_serve_respects_custom_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_QDRANT_URL", "http://my-qdrant:6333")
    with mock.patch("uvicorn.run"):
        main(["serve"])
    assert os.environ["RAG_QDRANT_URL"] == "http://my-qdrant:6333"


def test_fetch_invokes_fetch_papers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, list[str]]] = []

    def fake_run(module: str, args: list[str]) -> int:
        captured.append((module, args))
        return 0

    monkeypatch.setattr("src.cli._run_module", fake_run)
    rc = main(["fetch", "--manifest", "m.txt"])
    assert rc == 0
    assert len(captured) == 1
    module, args = captured[0]
    assert module == "scripts.fetch_papers"
    assert "--manifest" in args
    assert "m.txt" in args


def test_ingest_invokes_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, list[str]]] = []

    def fake_run(module: str, args: list[str]) -> int:
        captured.append((module, args))
        return 0

    monkeypatch.setattr("src.cli._run_module", fake_run)
    rc = main(["ingest", "--pdf-dir", "mydocs", "--collection", "c1", "--force"])
    assert rc == 0
    module, args = captured[0]
    assert module == "scripts.bootstrap_corpus"
    assert "--pdf-dir" in args and "mydocs" in args
    assert "--collection" in args and "c1" in args
    assert "--force" in args


def test_no_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        main([])
