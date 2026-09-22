"""RetrievalConfig: one description of the retrieval stack for the API and the eval."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.bootstrap import _wire_retriever_from_settings
from src.api.deps import _RetrievalConfigState, _RetrieverState, get_settings
from src.api.main import create_app
from src.config.settings import Settings, load_settings
from src.rag.bm25 import Bm25Index
from src.rag.retrieval_config import RetrievalConfig, build_text_retriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk
from tests.fakes import FakeEmbedder

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    _RetrieverState.instance = None
    _RetrievalConfigState.instance = None
    yield
    _RetrieverState.instance = None
    _RetrievalConfigState.instance = None


def test_fingerprint_is_stable_and_tracks_every_knob() -> None:
    base = RetrievalConfig()
    assert base.fingerprint() == RetrievalConfig().fingerprint()
    assert base.fingerprint() != RetrievalConfig(reranker_model="other").fingerprint()
    assert base.fingerprint() != RetrievalConfig(candidate_pool=20).fingerprint()


def test_irrelevant_knobs_do_not_change_the_fingerprint() -> None:
    """Knobs another setting switches off retrieve identically, so they must
    fingerprint identically."""
    assert (
        RetrievalConfig(visual_model=None, routing_mode="cascade", cascade_threshold=0.5)
        == RetrievalConfig()
    )
    assert RetrievalConfig(reranker_model=None, rerank_length_norm=True) == RetrievalConfig(
        reranker_model=None
    )
    hybrid = RetrievalConfig(visual_model="v", routing_mode="hybrid", classifier="llm:x")
    assert hybrid.classifier is None
    assert RetrievalConfig(visual_model="v").routing_mode == "category"
    assert RetrievalConfig(visual_model="v").classifier == "regex"


def test_text_only_drops_the_visual_leg() -> None:
    config = RetrievalConfig(visual_model="v", routing_mode="hybrid", visual_fusion_weight=5.0)
    assert config.text_only() == RetrievalConfig()


def test_from_settings_reads_the_served_knobs() -> None:
    settings = Settings(
        reranker_model="r",
        rerank_top_k=20,
        rerank_input_size=30,
        enable_multimodal=True,
        routing_mode="hybrid",
        visual_fusion_weight=2.0,
    )
    config = RetrievalConfig.from_settings(settings)
    assert config.reranker_model == "r"
    assert config.candidate_pool == 20
    assert config.rerank_input_size == 30
    assert config.rerank_length_norm is True
    assert config.visual_model == settings.visual_model
    assert config.routing_mode == "hybrid"
    assert config.visual_fusion_weight == 2.0


def test_multimodal_off_is_text_only() -> None:
    config = RetrievalConfig.from_settings(Settings(enable_multimodal=False))
    assert config.visual_model is None
    assert config.routing_mode is None


def test_cpu_profile_overlays_defaults() -> None:
    settings = load_settings(profile="cpu")
    assert settings.profile == "cpu"
    assert settings.embedder_backend == "sentence_transformers"
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.rerank_top_k == 20


def test_env_beats_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PROFILE", "cpu")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "7")
    settings = load_settings()
    assert settings.profile == "cpu"
    assert settings.rerank_top_k == 7


def test_unknown_profile_is_an_error() -> None:
    with pytest.raises(ValueError, match="Unknown settings profile"):
        load_settings(profile="nope")


def test_dockerfile_bakes_the_reranker_the_cpu_profile_names() -> None:
    """The image pre-downloads one cross-encoder and serves the `cpu` profile.
    If they disagree, the first /query downloads a model at request time."""
    dockerfile = (_REPO / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^ENV RAG_PROFILE=cpu\b", dockerfile, re.MULTILINE)
    baked = re.search(r"CrossEncoder\('([^']+)'\)", dockerfile)
    assert baked is not None
    assert baked.group(1) == load_settings(profile="cpu").reranker_model


def test_builder_threads_the_config_into_the_text_leg() -> None:
    config = RetrievalConfig(
        reranker_model="r", candidate_pool=20, rerank_input_size=30, exclude_decoration=False
    )
    retriever = build_text_retriever(
        config,
        embedder=FakeEmbedder(dim=8),
        vectorstore=QdrantVectorStore(url=":memory:", collection_name="c", dim=8),
        bm25=Bm25Index(),
        chunks_by_id={},
    )
    assert retriever._candidate_pool == 20
    assert retriever._rerank_input_size == 30
    assert retriever._exclude_decoration is False
    assert retriever._reranker is not None
    assert retriever._reranker._model_name == "r"


async def test_health_reports_the_wired_fingerprint() -> None:
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="rc", dim=8)
    await store.ensure_collection()
    await store.upsert_chunks(
        [Chunk(chunk_id="p::p1::c0", paper_id="p", page_numbers=[1], text="x")], [[0.1] * 8]
    )
    settings = Settings(corpus_collection="rc", qdrant_url=":memory:", reranker_model="r")
    assert await _wire_retriever_from_settings(settings, embedder=embedder, vectorstore=store)

    app = create_app(log_file=None)
    app.dependency_overrides[get_settings] = lambda: settings
    body = TestClient(app).get("/health").json()
    expected = RetrievalConfig.from_settings(settings).text_only()
    assert body["retrieval"]["fingerprint"] == expected.fingerprint()
    assert body["retrieval"]["config"]["reranker_model"] == "r"
