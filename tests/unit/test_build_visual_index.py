"""Smoke test for scripts/build_visual_index orchestration — no GPU, no model.

Catches the two issues static checks (ruff/mypy/synthetic unit tests) missed and
that only surfaced at runtime:
  1. non-corpus PDFs (ingestion fixtures like het-apollo17) leaking into the
     visual index — the build must scope to the served text corpus;
  2. two live Qdrant clients on one embedded path — local mode rejects it, so the
     read/encode/upsert flow must open and close sequentially.

The model and page rendering are stubbed; this exercises the orchestration only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from qdrant_client.http.models import VectorParams

import scripts.build_visual_index as bvi
import src.ingestion.visual as ingestion_visual
from src.ingestion.visual import RenderedPage
from src.rag.vectorstore import QdrantVectorStore
from src.rag.visual_store import QdrantVisualStore
from src.types import Chunk


class _StubRetriever:
    """Mimics build_visual_retriever's return contract: the two dicts the build
    script reads."""

    def __init__(
        self, page_embeds: dict[str, torch.Tensor], page_meta: dict[str, tuple[str, int]]
    ) -> None:
        self._page_embeds = page_embeds
        self._page_meta = page_meta


async def test_build_scopes_to_corpus_and_reopens_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qurl = f"path:{tmp_path / 'q'}"

    # Seed a one-paper text corpus (paperA). hetX is deliberately NOT in it.
    ts = QdrantVectorStore(url=qurl, collection_name="rag_corpus", dim=4)
    await ts.ensure_collection()
    await ts.upsert_chunks(
        [Chunk(chunk_id="paperA::p1::c0", paper_id="paperA", page_numbers=[1], text="x")],
        [[0.1, 0.2, 0.3, 0.4]],
    )
    await ts._client.close()  # release the embedded lock for the script

    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    (pdf_dir / "paperA.pdf").write_bytes(b"%PDF-fake")
    (pdf_dir / "hetX.pdf").write_bytes(b"%PDF-fake")  # the fixture that must be skipped

    encoded_papers: list[str] = []

    def _fake_render(
        paper_id: str, pdf_path: Path, *, out_dir: Path, dpi: int
    ) -> list[RenderedPage]:
        return [RenderedPage(paper_id=paper_id, page_number=1, image_path=Path("x.png"))]

    async def _fake_build(
        pages_by_paper: dict[str, list[tuple[int, Path]]], *, model_name: str, device: str
    ) -> _StubRetriever:
        encoded_papers.extend(pages_by_paper.keys())
        embeds = {f"{p}::p1::page": torch.ones((2, 4)) for p in pages_by_paper}
        meta = {f"{p}::p1::page": (p, 1) for p in pages_by_paper}
        return _StubRetriever(embeds, meta)

    monkeypatch.setattr(ingestion_visual, "render_pages", _fake_render)
    monkeypatch.setattr(bvi, "build_visual_retriever", _fake_build)

    await bvi._main(
        pdf_dir=pdf_dir,
        pdfs=None,
        qdrant_url=qurl,
        collection="rag_corpus_visual",
        corpus_collection="rag_corpus",
        pages_dir=tmp_path / "pages",
        visual_model="stub",
        device="cpu",
        dpi=150,
        force=False,
    )

    # hetX was filtered out before the encoder ran; only the corpus paper reached it.
    assert encoded_papers == ["paperA"]

    # The visual collection persisted (the reopen-for-upsert didn't deadlock on
    # the embedded path lock) and holds only the corpus paper.
    vs = QdrantVisualStore(url=qurl, collection_name="rag_corpus_visual")
    count = await vs.count()
    hits = await vs.search([[1.0, 1.0, 1.0, 1.0]], top_k=5)
    await vs.close()
    assert count == 1
    assert {h.paper_id for h in hits} == {"paperA"}


async def test_pages_only_build_sizes_the_collection_from_the_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collection takes the per-token dim the encoder emitted (320 for
    Vultron), not ColQwen2's 128. Pins the build's existing sizing now that the
    store has no default dim to fall back on."""
    paper_dir = tmp_path / "pages" / "paperA"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paperA_p1.jpg").write_bytes(b"not decoded: the encoder is stubbed")

    async def _fake_build(
        pages_by_paper: dict[str, list[tuple[int, Path]]], *, model_name: str, device: str
    ) -> _StubRetriever:
        return _StubRetriever(
            {"paperA::p1::page": torch.ones((3, 320))}, {"paperA::p1::page": ("paperA", 1)}
        )

    monkeypatch.setattr(bvi, "build_visual_retriever", _fake_build)
    qurl = f"path:{tmp_path / 'q'}"

    await bvi._main(
        pdf_dir=tmp_path,
        pdfs=None,
        qdrant_url=qurl,
        collection="pages_visual",
        corpus_collection="rag_corpus",
        pages_dir=tmp_path / "pages",
        visual_model="stub",
        device="cpu",
        dpi=150,
        force=False,
        pages_only=True,
    )

    vs = QdrantVisualStore(url=qurl, collection_name="pages_visual")
    info = await vs._client.get_collection("pages_visual")
    await vs.close()
    vectors = info.config.params.vectors
    assert isinstance(vectors, VectorParams)
    assert vectors.size == 320
