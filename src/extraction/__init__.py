"""Structured extraction: tables/charts → text, fed to the reader (ADR 0025)."""

from __future__ import annotations

from src.extraction.structured import (
    StructuredExtractor,
    build_extractor,
)

__all__ = ["StructuredExtractor", "build_extractor"]
