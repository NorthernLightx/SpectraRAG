"""Page rendering: idempotence + smoke against a real paper."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.visual import render_pages


@pytest.mark.integration
def test_render_pages_against_real_paper(tmp_path: Path) -> None:
    pdf = Path("data/papers/2604.22753v1.pdf")
    if not pdf.exists():
        pytest.skip("paper not present")
    pages = render_pages("2604.22753v1", pdf, out_dir=tmp_path, dpi=120)
    assert len(pages) == 25  # the paper has 25 pages
    for p in pages:
        assert p.image_path.exists()
        assert p.image_path.stat().st_size > 1024  # non-trivial PNG


@pytest.mark.integration
def test_render_pages_is_idempotent(tmp_path: Path) -> None:
    pdf = Path("data/papers/2604.22753v1.pdf")
    if not pdf.exists():
        pytest.skip("paper not present")
    first = render_pages("2604.22753v1", pdf, out_dir=tmp_path, dpi=120)
    mtimes_before = {p.image_path: p.image_path.stat().st_mtime_ns for p in first}
    second = render_pages("2604.22753v1", pdf, out_dir=tmp_path, dpi=120)
    mtimes_after = {p.image_path: p.image_path.stat().st_mtime_ns for p in second}
    # Nothing should have been re-rendered.
    assert mtimes_before == mtimes_after


def test_render_pages_raises_for_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        render_pages("nope", tmp_path / "missing.pdf")
