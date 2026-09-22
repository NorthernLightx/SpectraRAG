"""Runtime document upload (ADR 0029). Flag-gated off on the public demo."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from src.api.deps import get_chunks, get_corpus_handles, get_settings
from src.config.settings import Settings
from src.ingestion.pipeline import ingest_paper
from src.observability.logging import get_logger
from src.types import Paper

router = APIRouter()
log = get_logger(__name__)

# 25 MB: a research PDF beyond this is unusual, and the cap bounds the parse +
# embed memory cost on the CPU demo box.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class IngestResult(BaseModel):
    """Outcome of a successful upload."""

    paper_id: str
    chunks_added: int
    corpus_chunks: int
    # Pages Docling could not process: the document ingested without them.
    pages_failed: list[int] = []


@router.post("/ingest", response_model=IngestResult)
async def ingest(
    file: UploadFile, request: Request, settings: Settings = Depends(get_settings)
) -> IngestResult:
    if not settings.enable_upload:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document upload is disabled on this deployment (set RAG_ENABLE_UPLOAD).",
        )
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only .pdf uploads are supported."
        )
    # Reject an oversize upload on its declared length before reading the body
    # into memory; the post-read check still bounds an absent/lying header.
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload cap.",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty upload.")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload cap.",
        )

    embedder, vectorstore, bm25 = get_corpus_handles()
    chunks_by_id = get_chunks()
    paper_id = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).stem) or "upload"

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / f"{paper_id}.pdf"
        pdf_path.write_bytes(data)
        paper = Paper(paper_id=paper_id, title=paper_id, pdf_path=pdf_path)
        try:
            result = await ingest_paper(
                paper=paper,
                embedder=embedder,
                vectorstore=vectorstore,
                bm25=bm25,
                contextualizer_llm=None,
                contextualizer_model=None,
                contextualizer_concurrency=4,
                extract_figures_enabled=True,
                extract_tables_enabled=True,
                vlm_captioner=None,
            )
        except Exception as exc:
            log.warning(
                "api.ingest.failed", paper=paper_id, error=str(exc), error_type=type(exc).__name__
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not parse the uploaded PDF.",
            ) from exc

    for chunk in result.chunks:
        chunks_by_id[chunk.chunk_id] = chunk
    log.info(
        "api.ingest.done",
        paper=paper_id,
        chunks=result.chunk_count,
        corpus=len(chunks_by_id),
        pages_failed=result.failed_pages,
    )
    return IngestResult(
        paper_id=paper_id,
        chunks_added=result.chunk_count,
        corpus_chunks=len(chunks_by_id),
        pages_failed=result.failed_pages,
    )
