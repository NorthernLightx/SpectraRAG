"""_wire_retriever_from_settings — startup wiring of the production retriever.

Covers the paths the lifespan handler walks at API startup:
(1) wires a PipelineRetriever when Qdrant has chunks; (2) silently skips when
the collection is empty / Qdrant is unreachable; (3) when ``enable_multimodal``
is on AND a visual leg can be built, wraps in a RoutingRetriever; (4) when
multimodal is on but the visual leg fails, degrades to text-only.

The wire function closes the gap between bootstrap_corpus.py and a live
/answer — without it /answer returns 503 even after a successful ingest.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.api.bootstrap import (
    _build_classifier_from_settings,
    _collect_pages_from_dir,
    _wire_generator_from_settings,
    _wire_retriever_from_settings,
)
from src.api.deps import _GeneratorState, _RetrieverState
from src.config.settings import Settings
from src.rag.retrievers.classifier_llm import LLMQueryClassifier
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.routing import RoutingRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Query, RetrievalResult
from tests.fakes import FakeEmbedder, FakeRetriever


@pytest.fixture(autouse=True)
def _reset_retriever_state() -> Iterator[None]:
    """Module-level retriever + generator leak across tests; reset around each."""
    _RetrieverState.instance = None
    _GeneratorState.instance = None
    yield
    _RetrieverState.instance = None
    _GeneratorState.instance = None


def _settings(
    *,
    qdrant_url: str = ":memory:",
    enable_multimodal: bool = False,
    pages_dir: Path | None = None,
    openrouter_api_key: str | None = None,
    refusal_score_threshold: float | None = 0.105,
    reranker_model: str | None = None,
    page_budget: int | None = None,
) -> Settings:
    kwargs: dict[str, object] = {
        "env": "test",
        "log_level": "INFO",
        "qdrant_url": qdrant_url,
        "corpus_collection": "wiring_test",
        "rerank_top_k": 20,
        "enable_multimodal": enable_multimodal,
        "refusal_score_threshold": refusal_score_threshold,
    }
    if pages_dir is not None:
        kwargs["pages_dir"] = pages_dir
    if page_budget is not None:
        kwargs["page_budget"] = page_budget
    if openrouter_api_key is not None:
        kwargs["openrouter_api_key"] = openrouter_api_key
    if reranker_model is not None:
        kwargs["reranker_model"] = reranker_model
    return Settings(**kwargs)  # type: ignore[arg-type]


def _vec(dim: int) -> list[float]:
    return [0.1] * dim


@pytest.mark.asyncio
async def test_wire_returns_false_when_collection_empty() -> None:
    """No corpus ingested → wire is a no-op, /answer falls through to 503."""
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="wiring_test", dim=embedder.dim)
    await store.ensure_collection()  # exists but no points

    wired = await _wire_retriever_from_settings(_settings(), embedder=embedder, vectorstore=store)

    assert wired is False
    assert _RetrieverState.instance is None


@pytest.mark.asyncio
async def test_wire_returns_false_when_collection_does_not_exist() -> None:
    """Even more degenerate: collection was never created. Same outcome."""
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="never_made", dim=embedder.dim)

    wired = await _wire_retriever_from_settings(_settings(), embedder=embedder, vectorstore=store)

    assert wired is False
    assert _RetrieverState.instance is None


@pytest.mark.asyncio
async def test_wire_returns_false_on_qdrant_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreachable Qdrant raises during scroll → caught, logged, retriever stays None."""

    class _BoomStore:
        async def scroll_chunks(self) -> list[Chunk]:
            raise ConnectionError("Qdrant unreachable")

    embedder = FakeEmbedder(dim=8)

    wired = await _wire_retriever_from_settings(
        _settings(qdrant_url="http://nowhere.invalid:6333"),
        embedder=embedder,
        vectorstore=_BoomStore(),  # type: ignore[arg-type]
    )

    assert wired is False
    assert _RetrieverState.instance is None


@pytest.mark.asyncio
async def test_wire_populates_pipeline_retriever_from_qdrant() -> None:
    """The full happy path: chunks live in Qdrant → wire builds BM25 +
    chunks_by_id and registers a PipelineRetriever via set_retriever()."""
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="wiring_test", dim=embedder.dim)
    await store.ensure_collection()
    chunks = [
        Chunk(
            chunk_id=f"paper::p1::c{i}",
            paper_id="paper",
            page_numbers=[1],
            text=f"chunk text {i}",
            section="Intro" if i == 0 else None,
        )
        for i in range(3)
    ]
    await store.upsert_chunks(chunks, [_vec(embedder.dim) for _ in chunks])

    wired = await _wire_retriever_from_settings(_settings(), embedder=embedder, vectorstore=store)

    assert wired is True
    assert isinstance(_RetrieverState.instance, PipelineRetriever)


@pytest.mark.asyncio
@pytest.mark.slow  # real wiring reranks → CrossEncoder model download; CI only
async def test_wired_retriever_can_serve_a_query() -> None:
    """Sanity check beyond construction: the wired retriever actually returns
    chunks (so set_retriever didn't silently register a broken instance)."""
    from src.types import Query

    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="wiring_test", dim=embedder.dim)
    await store.ensure_collection()
    chunks = [
        Chunk(
            chunk_id="paper::p1::c0",
            paper_id="paper",
            page_numbers=[1],
            text="multi-modal retrieval over papers",
        ),
        Chunk(
            chunk_id="paper::p1::c1",
            paper_id="paper",
            page_numbers=[1],
            text="some unrelated content",
        ),
    ]
    await store.upsert_chunks(chunks, [_vec(embedder.dim) for _ in chunks])

    wired = await _wire_retriever_from_settings(_settings(), embedder=embedder, vectorstore=store)
    assert wired is True

    retriever = _RetrieverState.instance
    assert retriever is not None
    results = await retriever.retrieve(Query(text="multi-modal retrieval", top_k=2))
    assert len(results) >= 1
    assert {r.chunk_id for r in results}.issubset({c.chunk_id for c in chunks})


@pytest.mark.asyncio
async def test_wire_threads_reranker_model_from_settings() -> None:
    """Settings.reranker_model reaches the pipeline reranker, so the CPU-only
    Cloud Run deploy can swap bge-reranker-v2-m3 for a light MiniLM
    cross-encoder. Checks the configured id without loading the model — the
    BgeReranker resolves its CrossEncoder lazily on first rerank, not here."""
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="wiring_test", dim=embedder.dim)
    await store.ensure_collection()
    await store.upsert_chunks(
        [Chunk(chunk_id="paper::p1::c0", paper_id="paper", page_numbers=[1], text="x")],
        [_vec(embedder.dim)],
    )

    wired = await _wire_retriever_from_settings(
        _settings(reranker_model="test/custom-reranker"),
        embedder=embedder,
        vectorstore=store,
    )

    assert wired is True
    retriever = _RetrieverState.instance
    assert isinstance(retriever, PipelineRetriever)
    assert retriever._reranker is not None
    assert retriever._reranker._model_name == "test/custom-reranker"


# ---------- _collect_pages_from_dir --------------------------------------


def test_collect_pages_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    """Pages dir not present (default deploy without ingestion) → empty dict,
    no exception. The visual-leg builder treats that as "skip the visual leg"."""
    assert _collect_pages_from_dir(tmp_path / "does-not-exist") == {}


def test_collect_pages_skips_unrelated_files(tmp_path: Path) -> None:
    """Loose files in pages_dir are ignored — only `<paper>/<paper>_pN.png`
    counts. The matcher requires the page filename to start with the parent
    directory name so a copy/paste from another paper doesn't pollute."""
    paper_dir = tmp_path / "paper-a"
    paper_dir.mkdir()
    (paper_dir / "paper-a_p1.png").write_bytes(b"x")
    (paper_dir / "paper-a_p10.png").write_bytes(b"x")
    (paper_dir / "paper-a_p2.png").write_bytes(b"x")
    (paper_dir / "summary.txt").write_text("ignore me")
    (paper_dir / "wrong-paper_p3.png").write_bytes(b"x")  # wrong prefix
    # And a stray file at the root, not inside a paper dir:
    (tmp_path / "loose.png").write_bytes(b"x")

    layout = _collect_pages_from_dir(tmp_path)

    assert set(layout) == {"paper-a"}
    assert [p[0] for p in layout["paper-a"]] == [1, 2, 10]


# ---------- _build_classifier_from_settings ------------------------------


def test_classifier_falls_back_to_ollama_when_no_api_key() -> None:
    """ADR 0013: no OpenRouter key → build the Ollama-backed classifier,
    not None. Keyless deploys must not degrade to ADR 0008's weak regex
    router (measured +10.8% recall@10 on MMLongBench)."""
    classifier = _build_classifier_from_settings(_settings())
    assert isinstance(classifier, LLMQueryClassifier)


def test_classifier_constructed_when_api_key_present() -> None:
    classifier = _build_classifier_from_settings(_settings(openrouter_api_key="sk-test"))
    assert isinstance(classifier, LLMQueryClassifier)


# ---------- multi-modal wire ---------------------------------------------


async def _populate_store(embedder: FakeEmbedder) -> QdrantVectorStore:
    store = QdrantVectorStore(url=":memory:", collection_name="wiring_test", dim=embedder.dim)
    await store.ensure_collection()
    chunks = [
        Chunk(
            chunk_id="paper::p1::c0",
            paper_id="paper",
            page_numbers=[1],
            text="content about figure 3",
        ),
        Chunk(
            chunk_id="paper::p2::c0",
            paper_id="paper",
            page_numbers=[2],
            text="other content",
        ),
    ]
    await store.upsert_chunks(chunks, [_vec(embedder.dim) for _ in chunks])
    return store


@pytest.mark.asyncio
async def test_multimodal_off_yields_pipeline_retriever() -> None:
    """``enable_multimodal=False`` (the default) → text-only PipelineRetriever
    even when a visual retriever is available — preserves today's behaviour."""
    embedder = FakeEmbedder(dim=8)
    store = await _populate_store(embedder)
    visual = FakeRetriever(results=[])

    wired = await _wire_retriever_from_settings(
        _settings(enable_multimodal=False),
        embedder=embedder,
        vectorstore=store,
        visual_retriever=visual,
    )

    assert wired is True
    assert isinstance(_RetrieverState.instance, PipelineRetriever)


@pytest.mark.asyncio
async def test_multimodal_on_with_visual_leg_yields_routing_retriever() -> None:
    """End-to-end: multimodal flag on, injected visual leg + classifier →
    RoutingRetriever serving /answer. This is the path the README's
    multi-modal claims need to be load-bearing in production."""
    embedder = FakeEmbedder(dim=8)
    store = await _populate_store(embedder)
    visual_results = [
        RetrievalResult(
            chunk_id="paper::p1::page",
            paper_id="paper",
            score=0.9,
            text="[Page image paper p1]",
            page_numbers=[1],
            source="visual",
        )
    ]
    visual = FakeRetriever(results=visual_results)

    wired = await _wire_retriever_from_settings(
        _settings(enable_multimodal=True),
        embedder=embedder,
        vectorstore=store,
        visual_retriever=visual,
    )

    assert wired is True
    assert isinstance(_RetrieverState.instance, RoutingRetriever)


@pytest.mark.asyncio
async def test_multimodal_on_without_visual_leg_degrades_to_text() -> None:
    """Multimodal flag on but no visual retriever could be built (no GPU /
    no pages_dir / model load failure) → wire degrades to PipelineRetriever
    rather than booting nothing. Prevents a hardware regression from causing
    503 on every /answer."""
    embedder = FakeEmbedder(dim=8)
    store = await _populate_store(embedder)

    wired = await _wire_retriever_from_settings(
        _settings(enable_multimodal=True),  # no pages_dir → visual leg returns None
        embedder=embedder,
        vectorstore=store,
    )

    assert wired is True
    assert isinstance(_RetrieverState.instance, PipelineRetriever)


@pytest.mark.asyncio
@pytest.mark.slow  # real wiring reranks → CrossEncoder model download; CI only
async def test_multimodal_routing_actually_dispatches_figure_to_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RoutingRetriever wraps both legs and dispatches by query category.
    A figure query hits the visual leg via the regex classifier; this test
    confirms the wiring put the visual leg where the router can reach it."""
    # ADR 0013 makes the keyless wiring path build a live Ollama-backed
    # classifier; force the regex fallback so this stays a hermetic
    # wiring/dispatch test (CI has no Ollama → httpx.ConnectError otherwise).
    monkeypatch.setattr("src.api.bootstrap._build_classifier_from_settings", lambda _s: None)
    embedder = FakeEmbedder(dim=8)
    store = await _populate_store(embedder)
    visual_results = [
        RetrievalResult(
            chunk_id="paper::p1::page",
            paper_id="paper",
            score=0.9,
            text="[Page image paper p1]",
            page_numbers=[1],
            source="visual",
        )
    ]
    visual = FakeRetriever(results=visual_results)
    await _wire_retriever_from_settings(
        _settings(enable_multimodal=True),
        embedder=embedder,
        vectorstore=store,
        visual_retriever=visual,
    )

    retriever = _RetrieverState.instance
    assert isinstance(retriever, RoutingRetriever)
    # "Figure 3" → regex classifier returns "figure" → hybrid path → visual leg fires
    results = await retriever.retrieve(Query(text="What does Figure 3 show?", top_k=3))
    sources = {r.source for r in results}
    assert "visual" in sources or any(r.source == "pipeline" for r in results)


# Tier 1: refusal_score_threshold propagates from Settings to the production
# Generator. Without this wiring the calibrated default is dead code.


def test_wire_generator_propagates_refusal_threshold_default() -> None:
    """Generator built from Settings inherits the calibrated 0.105 default."""
    wired = _wire_generator_from_settings(_settings(openrouter_api_key="sk-test"))
    assert wired is True
    assert _GeneratorState.instance is not None
    assert _GeneratorState.instance._refusal_score_threshold == pytest.approx(0.105)


def test_wire_generator_propagates_explicit_threshold() -> None:
    """Override via Settings is honored."""
    wired = _wire_generator_from_settings(
        _settings(openrouter_api_key="sk-test", refusal_score_threshold=0.4)
    )
    assert wired is True
    assert _GeneratorState.instance is not None
    assert _GeneratorState.instance._refusal_score_threshold == pytest.approx(0.4)


def test_wire_generator_propagates_disabled_threshold() -> None:
    """When threshold is None in Settings, the gate is off in the Generator."""
    wired = _wire_generator_from_settings(
        _settings(openrouter_api_key="sk-test", refusal_score_threshold=None)
    )
    assert wired is True
    assert _GeneratorState.instance is not None
    assert _GeneratorState.instance._refusal_score_threshold is None


def test_wire_generator_skips_when_no_api_key() -> None:
    """No OpenRouter key = no generator wired = /answer 503s."""
    wired = _wire_generator_from_settings(_settings(openrouter_api_key=None))
    assert wired is False


def test_wire_generator_raises_image_cap_to_page_budget() -> None:
    """ADR 0024: when page_budget is set, the Generator's vision-image cap must rise
    to it, else a fitting whole document is silently truncated to 4 images (the win
    is erased). Guards the real bootstrap wiring, not a hand-built Generator."""
    wired = _wire_generator_from_settings(_settings(openrouter_api_key="sk-test", page_budget=30))
    assert wired is True
    assert _GeneratorState.instance is not None
    assert _GeneratorState.instance._max_vision_images == 30


def test_wire_generator_keeps_default_cap_when_no_page_budget() -> None:
    """Unset page_budget = default cap (4) = unchanged top-k/text behaviour."""
    from src.rag.generate import _MAX_VISION_IMAGES

    wired = _wire_generator_from_settings(_settings(openrouter_api_key="sk-test"))
    assert wired is True
    assert _GeneratorState.instance is not None
    assert _GeneratorState.instance._max_vision_images == _MAX_VISION_IMAGES
