"""Docling converts PDFs only, whatever a file's name says."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from docling.exceptions import ConversionError

from src.ingestion.docling_parser import _build_converter


def test_converter_refuses_a_non_pdf_named_pdf(tmp_path: Path) -> None:
    # Docling picks the backend from the content, so a gzip tarball saved as
    # .pdf would reach its METS-GBS archive reader unless only PDF is allowed.
    path = tmp_path / "upload.pdf"
    member = b'<mets xmlns="http://www.loc.gov/METS/"></mets>'
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("x.xml")
        info.size = len(member)
        tar.addfile(info, io.BytesIO(member))
    with pytest.raises(ConversionError, match="not allowed"):
        _build_converter().convert(path, raises_on_error=True)
