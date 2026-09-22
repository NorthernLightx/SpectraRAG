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


def _result(*, page_count: int, completed: list[int], errors: list[str]) -> SimpleNamespace:
    """Shaped like docling's ConversionResult after a run."""
    return SimpleNamespace(
        input=SimpleNamespace(
            page_count=page_count, limits=SimpleNamespace(page_range=(1, 2**31 - 1))
        ),
        pages=[SimpleNamespace(page_no=n) for n in completed],
        errors=[SimpleNamespace(error_message=e) for e in errors],
        document="doc",
    )


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
