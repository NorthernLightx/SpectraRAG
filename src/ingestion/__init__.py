"""Ingestion: PDFs → pages → chunks → indexed."""

from src.ingestion.chunking import Section, chunk_pages, split_into_sections
from src.ingestion.pdf import extract_pages
from src.ingestion.pipeline import IngestedPaper, ingest_paper

__all__ = [
    "IngestedPaper",
    "Section",
    "chunk_pages",
    "extract_pages",
    "ingest_paper",
    "split_into_sections",
]
