"""Papers endpoint: lists the demo corpus surfaced via /pages.

Derives the catalogue from the on-disk `pages_dir` layout
(`<pages_dir>/<paper_id>/<paper_id>_p<N>.png`) so the result tracks
whatever's actually been baked into the deployed image — no separate
manifest to drift out of sync.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_settings
from src.config.settings import Settings

router = APIRouter()


# arXiv preprint IDs look like `YYMM.NNNNN[vN]` (post-2007 format). Older
# `category/YYMMNNN` IDs aren't in the curated demo, so we don't try to
# detect them here.
_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


class PaperInfo(BaseModel):
    paper_id: str
    is_arxiv: bool
    arxiv_url: str | None = None
    pdf_url: str | None = None
    page_count: int


@router.get("/papers", response_model=list[PaperInfo])
def list_papers(settings: Settings = Depends(get_settings)) -> list[PaperInfo]:
    pages_dir = settings.pages_dir
    if pages_dir is None or not pages_dir.is_dir():
        return []
    out: list[PaperInfo] = []
    for subdir in sorted(pages_dir.iterdir()):
        if not subdir.is_dir():
            continue
        paper_id = subdir.name
        page_count = sum(1 for _ in subdir.glob(f"{paper_id}_p*.png"))
        if page_count == 0:
            continue
        is_arxiv = bool(_ARXIV_RE.match(paper_id))
        out.append(
            PaperInfo(
                paper_id=paper_id,
                is_arxiv=is_arxiv,
                arxiv_url=f"https://arxiv.org/abs/{paper_id}" if is_arxiv else None,
                pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf" if is_arxiv else None,
                page_count=page_count,
            )
        )
    return out
