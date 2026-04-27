"""Query endpoint: placeholder until retrieval/generation lands in later tasks."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.types import Query

router = APIRouter()


@router.post("/query")
def query(payload: Query) -> None:
    """Placeholder. Wired to validate input shape; full pipeline arrives in Phase 1 retrieval."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            f"Query endpoint not implemented yet (received text='{payload.text[:40]}...', "
            f"top_k={payload.top_k})."
        ),
    )
