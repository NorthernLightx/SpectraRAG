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


def _snapshot(root: "os.PathLike[str]") -> None:
    snap = os.path.join(root, "qdrant_local", "collection")
    os.makedirs(snap)
    with open(os.path.join(snap, "storage.sqlite"), "w") as fh:
        fh.write("committed")


def test_serve_with_uploads_uses_a_working_copy_of_the_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "os.PathLike[str]"
) -> None:
    # Uploads write into the store; the committed snapshot must stay clean.
    _snapshot(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    monkeypatch.setenv("RAG_ENABLE_UPLOAD", "true")
    with mock.patch("uvicorn.run"):
        main(["serve"])
    assert os.environ["RAG_QDRANT_URL"] == "path:./qdrant_uploads"
    assert os.path.isfile(os.path.join(tmp_path, "qdrant_uploads", "collection", "storage.sqlite"))


def test_serve_keeps_an_existing_working_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "os.PathLike[str]"
) -> None:
    _snapshot(tmp_path)
    os.makedirs(os.path.join(tmp_path, "qdrant_uploads"))
    with open(os.path.join(tmp_path, "qdrant_uploads", "uploaded.txt"), "w") as fh:
        fh.write("earlier upload")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    monkeypatch.setenv("RAG_ENABLE_UPLOAD", "1")
    with mock.patch("uvicorn.run"):
        main(["serve"])
    assert os.path.isfile(os.path.join(tmp_path, "qdrant_uploads", "uploaded.txt"))


def test_serve_without_uploads_reads_the_snapshot_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "os.PathLike[str]"
) -> None:
    _snapshot(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    monkeypatch.delenv("RAG_ENABLE_UPLOAD", raising=False)
    with mock.patch("uvicorn.run"):
        main(["serve"])
    assert os.environ["RAG_QDRANT_URL"] == "path:./qdrant_local"
    assert not os.path.exists(os.path.join(tmp_path, "qdrant_uploads"))


def test_serve_redoes_an_interrupted_working_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "os.PathLike[str]"
) -> None:
    # A half-copied store would otherwise be served on every later start.
    _snapshot(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    monkeypatch.setenv("RAG_ENABLE_UPLOAD", "true")

    def dies(src: str, dst: str, **kwargs: object) -> None:
        os.makedirs(dst, exist_ok=True)
        raise KeyboardInterrupt

    with mock.patch("src.cli.shutil.copytree", dies), pytest.raises(KeyboardInterrupt):
        main(["serve"])
    with mock.patch("uvicorn.run"):
        main(["serve"])
    assert os.path.isfile(os.path.join(tmp_path, "qdrant_uploads", "collection", "storage.sqlite"))
    assert os.listdir(os.path.join(tmp_path, "qdrant_uploads")) == ["collection"]
