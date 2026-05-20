"""End-to-end ingestion: paper → pages → chunks → indexed in BM25 + vectorstore.

Optional figure + table extraction is supported: when enabled, figures and
tables are converted to first-class Chunks (with `metadata['kind']`) and join
the text chunks in the same embedding + BM25 + Qdrant pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.embeddings.protocol import Embedder
from src.ingestion.captioner import _Captioner, caption_figures
from src.ingestion.chunking import chunk_pages, figure_to_chunk, table_to_chunk
from src.ingestion.contextualize import contextualize_chunks
from src.ingestion.figures import extract_figures
from src.ingestion.pdf import extract_pages
from src.ingestion.tables import extract_tables
from src.llm.protocol import LLMClient
from src.observability.logging import get_logger, timed_event
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Paper

_log = get_logger(__name__)


@dataclass(frozen=True)
class IngestedPaper:
    """Outcome of ingesting one paper."""

    paper_id: str
    chunk_count: int
    chunks: list[Chunk]


async def ingest_paper(
    *,
    paper: Paper,
    embedder: Embedder,
    vectorstore: QdrantVectorStore,
    bm25: Bm25Index,
    target_chars: int = 1200,
    overlap_chars: int = 200,
    contextualizer_llm: LLMClient | None = None,
    contextualizer_model: str | None = None,
    contextualizer_concurrency: int = 4,
    extract_figures_enabled: bool = False,
    extract_tables_enabled: bool = False,
    use_docling: bool = True,
    figures_out_dir: Path = Path("data/figures"),
    vlm_captioner: _Captioner | None = None,
) -> IngestedPaper:
    """Full pipeline: extract pages, chunk, optionally contextualize, embed, index.

    If `contextualizer_llm` and `contextualizer_model` are both provided, each
    chunk gets an LLM-generated situating blurb prepended at index time
    (Anthropic-style contextual retrieval). Display text is unchanged.

    When `extract_figures_enabled` / `extract_tables_enabled` are True, figures
    and tables are extracted from the PDF and added to the chunk list as
    first-class chunks (with `metadata['kind']` = "figure" / "table"). They go
    through the same embed + BM25 + Qdrant path as text chunks.

    `use_docling=True` (the default after ADR 0020's heterogeneous-format
    eval, 2026-05-20) uses Docling's deterministic layout + table-
    structure pipeline instead of PyMuPDF's `extract_figures` (XREF-only,
    misses vector plots) and `extract_tables` (`find_tables()` heuristic,
    misses tight numeric tables). Halved the corpus-wide audit flag rate
    (25.4 % → 12.7 %) on the 20-paper ArXiv corpus and held cleanly on
    heterogeneous formats — slide-deck PDFs (0 flags), HAL-style non-
    arXiv papers, and a 339-page scanned OCR'd NASA Apollo 17 report
    (89 figures + 27 tables recovered with bboxes). Set `use_docling=False`
    to fall back to PyMuPDF (preserved for repeatability of pre-ADR-0020
    measurements). `extract_figures_enabled` / `extract_tables_enabled`
    still gate whether figures / tables are emitted as chunks; the VLM
    captioner still runs if set (its caption is preferred over Docling's
    at `figure_to_chunk` time per ADR 0002).
    """
    with timed_event(
        _log, "ingest.done", paper_id=paper.paper_id, pdf_path=str(paper.pdf_path)
    ) as ctx:
        # Single Docling conversion when enabled — shared by text chunking
        # (ADR 0021) and figure / table extraction (ADR 0020) so the
        # layout + OCR pipeline runs once per paper, not twice.
        docling_doc = None
        paper_text = ""
        if use_docling:
            from src.ingestion.docling_chunker import chunk_with_docling, paper_text_from_docling
            from src.ingestion.docling_parser import convert_with_docling

            docling_doc = convert_with_docling(paper.pdf_path)
            chunks = chunk_with_docling(
                paper.paper_id,
                docling_doc,
                target_chars=target_chars,
                overlap_chars=overlap_chars,
            )
            paper_text = paper_text_from_docling(docling_doc)
            ctx["pages"] = len(getattr(docling_doc, "pages", {}) or {})
        else:
            pages = extract_pages(paper_id=paper.paper_id, pdf_path=paper.pdf_path)
            chunks = chunk_pages(pages, target_chars=target_chars, overlap_chars=overlap_chars)
            paper_text = "\n\n".join(p.text for p in pages)
            ctx["pages"] = len(pages)
        ctx["text_chunks"] = len(chunks)

        # Multi-modal extraction. When Docling already converted, reuse
        # the same `DoclingDocument` to avoid a second slow pass.
        figure_count = 0
        figures_captioned = 0
        table_count = 0
        if use_docling and (extract_figures_enabled or extract_tables_enabled):
            from src.ingestion.docling_parser import parse_with_docling

            docling_figs, docling_tabs = parse_with_docling(
                paper.paper_id,
                paper.pdf_path,
                out_dir=figures_out_dir,
                doc=docling_doc,
            )
            if extract_figures_enabled:
                if vlm_captioner is not None and docling_figs:
                    docling_figs = await caption_figures(docling_figs, captioner=vlm_captioner)
                    figures_captioned = sum(1 for f in docling_figs if f.vlm_caption)
                chunks.extend(figure_to_chunk(f) for f in docling_figs)
                figure_count = len(docling_figs)
            if extract_tables_enabled:
                chunks.extend(table_to_chunk(t) for t in docling_tabs)
                table_count = len(docling_tabs)
        elif not use_docling:
            if extract_figures_enabled:
                figures = extract_figures(paper.paper_id, paper.pdf_path, out_dir=figures_out_dir)
                if vlm_captioner is not None and figures:
                    figures = await caption_figures(figures, captioner=vlm_captioner)
                    figures_captioned = sum(1 for f in figures if f.vlm_caption)
                chunks.extend(figure_to_chunk(f) for f in figures)
                figure_count = len(figures)
            if extract_tables_enabled:
                tables = extract_tables(paper.paper_id, paper.pdf_path)
                chunks.extend(table_to_chunk(t) for t in tables)
                table_count = len(tables)
        ctx["figure_chunks"] = figure_count
        ctx["figures_captioned"] = figures_captioned
        ctx["table_chunks"] = table_count
        ctx["use_docling"] = use_docling

        ctx["chunks"] = len(chunks)
        if not chunks:
            ctx["embedding_dim"] = 0
            ctx["contextualized"] = False
            return IngestedPaper(paper_id=paper.paper_id, chunk_count=0, chunks=[])

        contextualized = contextualizer_llm is not None and contextualizer_model is not None
        if contextualized:
            assert contextualizer_llm is not None
            assert contextualizer_model is not None
            chunks = await contextualize_chunks(
                chunks,
                paper_text,
                llm=contextualizer_llm,
                model=contextualizer_model,
                concurrency=contextualizer_concurrency,
            )
        ctx["contextualized"] = contextualized

        embeddings = await embedder.embed_texts([c.indexed_text for c in chunks])
        await vectorstore.upsert_chunks(chunks, embeddings)
        bm25.add(chunks)
        ctx["embedding_dim"] = len(embeddings[0]) if embeddings else 0
        return IngestedPaper(paper_id=paper.paper_id, chunk_count=len(chunks), chunks=chunks)
