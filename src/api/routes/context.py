"""POST /context: the reader's messages for one chat turn (ADR 0033).

The chat UI generates in the browser on the visitor's own key (ADR 0031), so it
needs the messages without the server calling a model. This route runs the
builder /answer and the eval use and returns its output; the browser swaps each
page image ref for the image bytes and sends the result to the provider.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from src.api.deps import get_settings, peek_figures
from src.api.rate_limit import limiter
from src.config.settings import Settings
from src.prompts.loader import load_prompt_by_name
from src.rag.context import (
    READER_PROMPT_NAME,
    PriorTurn,
    ReaderMessage,
    build_reader_context,
)
from src.types import RetrievalResult

router = APIRouter()


class ContextRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # The /query results for this turn, as the UI received them.
    results: list[RetrievalResult] = Field(max_length=100)
    # The chat sends its whole history; the cap only rejects absurd bodies.
    prior_turns: list[PriorTurn] = Field(default_factory=list, max_length=500)
    use_images: bool = True


class InjectedFigure(BaseModel):
    chunk_id: str
    paper_id: str
    page: int
    caption: str
    bbox: list[float] | None = None


class ContextResponse(BaseModel):
    messages: list[ReaderMessage]
    injected: list[InjectedFigure]
    context_ids: list[str]
    prompt_version: str


def _has_page_render(pages_dir: Path, paper: str, page: int) -> bool:
    """Whether the render exists. `paper` comes from the request body, so the
    path must resolve inside pages_dir or a crafted id could probe for files
    elsewhere on disk."""
    root = pages_dir.resolve()
    candidate = (root / paper / f"{paper}_p{page}.png").resolve()
    return candidate.is_relative_to(root) and candidate.is_file()


# One call per chat turn, like /query; the looser limit leaves room for retries.
@router.post("/context", response_model=ContextResponse)
@limiter.limit("30/minute")
def context(
    request: Request, payload: ContextRequest, settings: Settings = Depends(get_settings)
) -> ContextResponse:
    prompt = load_prompt_by_name(READER_PROMPT_NAME)
    pages_dir = settings.pages_dir if settings.pages_dir and settings.pages_dir.is_dir() else None
    ctx = build_reader_context(
        payload.question,
        payload.results,
        prompt=prompt,
        prior_turns=payload.prior_turns,
        figures=peek_figures(),
        include_images=payload.use_images and pages_dir is not None,
        max_context_tokens=settings.max_context_tokens,
        # The browser fetches /pages/<paper>/<paper>_p<N>.png; a page with no
        # render must not take one of the image slots.
        image_available=lambda paper, page: (
            pages_dir is not None and _has_page_render(pages_dir, paper, page)
        ),
    )
    return ContextResponse(
        messages=ctx.messages,
        injected=[
            InjectedFigure(
                chunk_id=f.chunk_id,
                paper_id=f.paper_id,
                page=f.page,
                caption=f.caption,
                bbox=f.bbox,
            )
            for f in ctx.injected
        ],
        context_ids=ctx.context_ids,
        prompt_version=prompt.version,
    )
