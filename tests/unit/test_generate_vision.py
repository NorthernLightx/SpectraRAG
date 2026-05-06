"""Generator vision wiring: when pages_dir is set and a visual RetrievalResult
is in the context, the LLM call gets `images=[...]`. When pages_dir is None
or no visual sources are present, no images are sent (text-only back-compat).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.llm.protocol import ChatResponse, Message
from src.prompts.loader import Prompt
from src.rag.generate import Generator
from src.types import RetrievalResult


class _RecordingLLM:
    """Captures the `images` argument passed to chat() so tests can assert."""

    def __init__(self) -> None:
        self.last_images: list[Path] | None = None

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        images: list[Path] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.last_images = images
        return ChatResponse(text="ok [c1]", model=model, tokens_in=10, tokens_out=20)


def _prompt() -> Prompt:
    return Prompt(name="answer", version="v0", system=None, user_template="{query} {context}")


def _text(cid: str = "p1::p1::c0") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid,
        paper_id="p1",
        score=0.9,
        text="t",
        page_numbers=[1],
        source="pipeline",
    )


def _visual(paper: str = "p1", page: int = 5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper}::p{page}::page",
        paper_id=paper,
        score=0.8,
        text=f"[Page image {paper} p{page}]",
        page_numbers=[page],
        source="visual",
    )


@pytest.mark.asyncio
async def test_no_images_sent_when_pages_dir_unset(tmp_path: Path) -> None:
    """Even with a visual RetrievalResult, pages_dir=None means text-only call."""
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=None)
    await gen.answer("q?", [_visual()])
    assert llm.last_images is None


@pytest.mark.asyncio
async def test_no_images_sent_when_no_visual_results(tmp_path: Path) -> None:
    """pages_dir is set but all retrievals are text — still text-only call."""
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    await gen.answer("q?", [_text()])
    assert llm.last_images is None


@pytest.mark.asyncio
async def test_images_passed_when_visual_result_and_image_exists(tmp_path: Path) -> None:
    """pages_dir set + visual RetrievalResult + PNG present → llm.chat receives images."""
    paper, page = "paper-x", 7
    img_dir = tmp_path / paper
    img_dir.mkdir()
    img_path = img_dir / f"{paper}_p{page}.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal png signature, content unused by Generator
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    await gen.answer("q?", [_visual(paper=paper, page=page)])
    assert llm.last_images == [img_path]


@pytest.mark.asyncio
async def test_missing_image_skipped_gracefully(tmp_path: Path) -> None:
    """pages_dir set + visual RetrievalResult + PNG missing → image skipped, call still made."""
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    await gen.answer("q?", [_visual(paper="ghost", page=99)])
    # No image found → images param falls back to None
    assert llm.last_images is None


@pytest.mark.asyncio
async def test_images_capped_at_max(tmp_path: Path) -> None:
    """More than _MAX_VISION_IMAGES (4) visual results → only first 4 attached."""
    paper = "many-pages"
    img_dir = tmp_path / paper
    img_dir.mkdir()
    visuals: list[RetrievalResult] = []
    for page in range(1, 7):  # 6 visual results, only 4 should be attached
        (img_dir / f"{paper}_p{page}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        visuals.append(_visual(paper=paper, page=page))
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    await gen.answer("q?", visuals)
    assert llm.last_images is not None
    assert len(llm.last_images) == 4
