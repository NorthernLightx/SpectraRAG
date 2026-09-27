"""End-to-end ingestion: paper → pages → chunks → indexed in BM25 + vectorstore.

Optional figure + table extraction is supported: when enabled, figures and
tables are converted to first-class Chunks (with `metadata['kind']`) and join
the text chunks in the same embedding + BM25 + Qdrant pipeline.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.embeddings.protocol import Embedder
from src.ingestion import media
from src.ingestion.captioner import (
    _Captioner,
    caption_figures,
    relatex_captions,
    relatex_table_captions,
)
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


def document_id(filename: str, *, fallback: str) -> str:
    """A document id from a file name: the stem, with characters outside
    `[A-Za-z0-9._-]` replaced. Leading dots go, since pages and figures are
    written under `<dir>/<id>/` and `.` or `..` would name the dir or its parent."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).stem).lstrip(".") or fallback


@dataclass(frozen=True)
class IngestedPaper:
    """Outcome of ingesting one paper."""

    paper_id: str
    chunk_count: int
    chunks: list[Chunk]
    # 1-based pages Docling could not process. Their content is missing from
    # `chunks` even though the paper still ingested.
    failed_pages: list[int] = field(default_factory=list)


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
    pages_dir: Path = Path("data/pages"),
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
    heterogeneous formats: slide-deck PDFs (0 flags), HAL-style non-
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
        # Single Docling conversion when enabled, shared by text chunking
        # (ADR 0021) and figure / table extraction (ADR 0020) so the
        # layout + OCR pipeline runs once per paper, not twice.
        docling_doc = None
        paper_text = ""
        failed_pages: list[int] = []
        if use_docling:
            from src.ingestion.docling_chunker import chunk_with_docling, paper_text_from_docling
            from src.ingestion.docling_parser import convert_with_docling

            # Docling's layout + OCR pass takes seconds to minutes of CPU. Off the
            # event loop so POST /ingest doesn't freeze every other request.
            conversion = await asyncio.to_thread(convert_with_docling, paper.pdf_path)
            docling_doc = conversion.document
            failed_pages = conversion.failed_pages
            chunks = chunk_with_docling(
                paper.paper_id,
                docling_doc,
                target_chars=target_chars,
                overlap_chars=overlap_chars,
            )
            paper_text = paper_text_from_docling(docling_doc)
            ctx["pages"] = len(getattr(docling_doc, "pages", {}) or {})
            ctx["pages_failed"] = len(failed_pages)
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

            docling_figs, docling_tabs = await asyncio.to_thread(
                parse_with_docling,
                paper.paper_id,
                paper.pdf_path,
                out_dir=figures_out_dir,
                doc=docling_doc,
            )
            if extract_figures_enabled:
                if vlm_captioner is not None and docling_figs:
                    docling_figs = await caption_figures(docling_figs, captioner=vlm_captioner)
                    docling_figs = await relatex_captions(docling_figs, captioner=vlm_captioner)
                    figures_captioned = sum(1 for f in docling_figs if f.vlm_caption)
                chunks.extend(figure_to_chunk(f) for f in docling_figs)
                figure_count = len(docling_figs)
            if extract_tables_enabled:
                if vlm_captioner is not None and docling_tabs:
                    docling_tabs = await relatex_table_captions(
                        docling_tabs, captioner=vlm_captioner, pages_dir=pages_dir
                    )
                chunks.extend(table_to_chunk(t) for t in docling_tabs)
                table_count = len(docling_tabs)
        elif not use_docling:
            if extract_figures_enabled:
                figures = extract_figures(paper.paper_id, paper.pdf_path, out_dir=figures_out_dir)
                if vlm_captioner is not None and figures:
                    figures = await caption_figures(figures, captioner=vlm_captioner)
                    figures = await relatex_captions(figures, captioner=vlm_captioner)
                    figures_captioned = sum(1 for f in figures if f.vlm_caption)
                chunks.extend(figure_to_chunk(f) for f in figures)
                figure_count = len(figures)
            if extract_tables_enabled:
                tables = extract_tables(paper.paper_id, paper.pdf_path)
                if vlm_captioner is not None and tables:
                    tables = await relatex_table_captions(
                        tables, captioner=vlm_captioner, pages_dir=pages_dir
                    )
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
            return IngestedPaper(
                paper_id=paper.paper_id, chunk_count=0, chunks=[], failed_pages=failed_pages
            )

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

        ctx["embedding_dim"] = await _index_chunks(chunks, embedder, vectorstore, bm25)
        return IngestedPaper(
            paper_id=paper.paper_id,
            chunk_count=len(chunks),
            chunks=chunks,
            failed_pages=failed_pages,
        )


async def _index_chunks(
    chunks: list[Chunk], embedder: Embedder, vectorstore: QdrantVectorStore, bm25: Bm25Index
) -> int:
    """Embed, upsert and BM25-index `chunks`; returns the embedding dim."""
    embeddings = await embedder.embed_texts([c.indexed_text for c in chunks])
    await vectorstore.upsert_chunks(chunks, embeddings)
    bm25.add(chunks)
    return len(embeddings[0]) if embeddings else 0


async def ingest_media(
    *,
    doc_id: str,
    media_path: Path,
    embedder: Embedder,
    vectorstore: QdrantVectorStore,
    bm25: Bm25Index,
    transcriber: media.Transcriber,
    pages_dir: Path = Path("data/pages"),
    params: media.SegmentationParams | None = None,
    audio_params: media.AudioSegmentationParams | None = None,
    target_chars: int = 1200,
) -> IngestedPaper:
    """A recording as a document whose pages are its time segments.

    A video's pages are its slide segments, each with a keyframe written as the
    page image. An audio recording's pages are time windows over its
    transcript, with no image. Either way the segment manifest goes beside the
    pages and the transcript chunks are indexed like any text chunk. Decoding
    and transcription take minutes of CPU, so both run off the event loop.
    """
    params = params or media.SegmentationParams()
    audio_params = audio_params or media.AudioSegmentationParams()
    with timed_event(_log, "ingest_media.done", doc_id=doc_id, media_path=str(media_path)) as ctx:
        probe = await asyncio.to_thread(media.probe_media, media_path)
        words: list[media.Word] = []
        if probe.has_audio:
            source = await asyncio.to_thread(media.media_fingerprint, media_path)
            cached = media.load_words(pages_dir, doc_id, transcriber.name, source=source)
            if cached is None:
                cached = await asyncio.to_thread(transcriber.transcribe, media_path)
                media.save_words(pages_dir, doc_id, transcriber.name, cached, source=source)
            words = cached
        segmentation: dict[str, Any]
        if probe.has_video:
            segments = await asyncio.to_thread(
                media.segment_video, media_path, doc_id, pages_dir, params
            )
            segmentation = params.as_dict()
        else:
            segments = media.segment_words(words, duration_s=probe.duration_s, params=audio_params)
            segmentation = audio_params.as_dict()
        segments = media.cover_words(segments, words)
        chunks = media.transcript_chunks(doc_id, segments, words, target_chars=target_chars)
        if not chunks and not probe.has_video:
            # An audio document is only its transcript; checked before the old
            # chunks go, so a failed re-ingest keeps the previous index.
            raise ValueError(f"no speech found in {media_path.name}")
        ctx["kind"] = "video" if probe.has_video else "audio"
        ctx["segments"] = len(segments)
        per_minute = len(segments) / max(segments[-1].end_s / 60, 1e-9)
        ctx["segments_per_min"] = round(per_minute, 2)
        ctx["words"] = len(words)
        ctx["chunks"] = len(chunks)
        ctx["empty_pages"] = len(segments) - len({c.page_numbers[0] for c in chunks})
        # The manifest marks a recording as done, so it goes last: a failure
        # before it leaves no marker and the next run retries. The old chunks
        # go first, since a new segmentation changes their ids.
        media.manifest_path(pages_dir, doc_id).unlink(missing_ok=True)
        await vectorstore.delete_paper(doc_id)
        if chunks:
            ctx["embedding_dim"] = await _index_chunks(chunks, embedder, vectorstore, bm25)
        media.write_manifest(
            pages_dir,
            media.MediaManifest(
                doc_id=doc_id,
                source=media_path.name,
                duration_s=segments[-1].end_s,
                transcriber=transcriber.name,
                segmentation=segmentation,
                segments=segments,
                kind="video" if probe.has_video else "audio",
            ),
        )
        return IngestedPaper(paper_id=doc_id, chunk_count=len(chunks), chunks=chunks)
