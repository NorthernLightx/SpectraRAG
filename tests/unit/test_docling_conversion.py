"""Docling page failures must reach the ingest result instead of vanishing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fitz
import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_settings
from src.api.main import create_app
from src.config.settings import Settings
from src.ingestion.docling_parser import summarize_conversion


def _result(
    *,
    page_count: int,
    completed: list[int],
    errors: list[str],
    page_range: tuple[int, int] = (1, 2**31 - 1),
    document: object = "doc",
) -> SimpleNamespace:
    """Shaped like docling's ConversionResult after a run."""
    return SimpleNamespace(
        input=SimpleNamespace(page_count=page_count, limits=SimpleNamespace(page_range=page_range)),
        pages=[SimpleNamespace(page_no=n) for n in completed],
        errors=[SimpleNamespace(error_message=e) for e in errors],
        document=document,
    )


def _pdf(tmp_path: Path, n_pages: int) -> Path:
    doc = fitz.open()
    for i in range(n_pages):
        doc.new_page().insert_text((72, 72), f"Page {i + 1} text.", fontsize=11)
    pdf = tmp_path / f"doc{n_pages}.pdf"
    doc.save(pdf)
    doc.close()
    return pdf


def test_dropped_pages_are_reported() -> None:
    conversion = summarize_conversion(
        _result(page_count=5, completed=[1, 2, 4], errors=["Page 3: std::bad_alloc"])
    )
    assert conversion.document == "doc"
    assert conversion.failed_pages == [3, 5]
    assert conversion.errors == ["Page 3: std::bad_alloc"]
    assert conversion.partial


def test_clean_conversion_is_not_partial() -> None:
    conversion = summarize_conversion(_result(page_count=2, completed=[1, 2], errors=[]))
    assert conversion.failed_pages == []
    assert not conversion.partial


def test_ingest_route_returns_failed_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(log_file=None)
    app.dependency_overrides[get_settings] = lambda: Settings(enable_upload=True)

    async def fake_ingest(**kwargs: object) -> mock.Mock:
        return mock.Mock(chunk_count=1, chunks=[mock.Mock(chunk_id="d::p1::c0")], failed_pages=[2])

    monkeypatch.setattr(
        "src.api.routes.ingest.get_corpus_handles",
        lambda: (mock.Mock(), mock.Mock(), mock.Mock()),
    )
    monkeypatch.setattr("src.api.routes.ingest.get_chunks", lambda: {})
    monkeypatch.setattr("src.api.routes.ingest.ingest_paper", fake_ingest)

    resp = TestClient(app).post(
        "/ingest", files={"file": ("d.pdf", b"%PDF stub", "application/pdf")}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["pages_failed"] == [2]


@pytest.mark.slow
def test_real_conversion_exposes_the_fields_read(tmp_path: Path) -> None:
    """Guards the attribute paths `summarize_conversion` reads on docling's real type."""
    from src.ingestion.docling_parser import convert_with_docling

    doc = fitz.open()
    for i in range(2):
        doc.new_page().insert_text((72, 72), f"Page {i + 1} text.", fontsize=11)
    pdf = tmp_path / "two.pdf"
    doc.save(pdf)
    doc.close()

    conversion = convert_with_docling(pdf)
    assert conversion.failed_pages == []
    assert not conversion.partial
    assert len(conversion.document.pages) == 2


def test_a_long_pdf_converts_in_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each window is converted on its own; the failures of every window are
    reported with their true page numbers and the window documents merged."""
    from src.ingestion import docling_parser

    ranges: list[tuple[int, int]] = []

    class FakeConverter:
        def convert(self, path: Path, page_range: tuple[int, int]) -> SimpleNamespace:
            ranges.append(page_range)
            start, end = page_range
            completed = [n for n in range(start, end + 1) if n != 45]
            return _result(
                page_count=70,
                completed=completed,
                errors=["Page 45: std::bad_alloc"] if 45 in range(start, end + 1) else [],
                page_range=page_range,
                document=f"doc{start}",
            )

    monkeypatch.setattr(docling_parser, "_build_converter", FakeConverter)
    monkeypatch.setattr(docling_parser, "_merge_documents", lambda docs: tuple(docs))

    conversion = docling_parser.convert_with_docling(_pdf(tmp_path, 70))

    assert ranges == [(1, 30), (31, 60), (61, 70)]
    assert conversion.document == ("doc1", "doc31", "doc61")
    assert conversion.failed_pages == [45]
    assert conversion.errors == ["Page 45: std::bad_alloc"]


def test_a_short_pdf_converts_in_one_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ingestion import docling_parser

    class FakeConverter:
        def convert(self, path: Path, page_range: tuple[int, int]) -> SimpleNamespace:
            assert page_range == (1, 3)
            return _result(page_count=3, completed=[1, 2, 3], errors=[], page_range=page_range)

    monkeypatch.setattr(docling_parser, "_build_converter", FakeConverter)
    conversion = docling_parser.convert_with_docling(_pdf(tmp_path, 3))
    assert conversion.document == "doc"
    assert not conversion.partial


_LONG_PAPER = Path("data/papers/2604.28182v1.pdf")


@pytest.mark.slow
@pytest.mark.skipif(not _LONG_PAPER.exists(), reason="demo papers not fetched")
def test_a_long_paper_keeps_every_page() -> None:
    """One Docling run over this 81-page paper dropped pages 47 to 81 to a
    cumulative std::bad_alloc; converted in windows it keeps all of them."""
    from src.ingestion.docling_parser import convert_with_docling

    conversion = convert_with_docling(_LONG_PAPER)
    assert conversion.failed_pages == []
    assert sorted(conversion.document.pages) == list(range(1, 82))
