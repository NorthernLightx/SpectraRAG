"""Page images in the reader's context (ADR 0033).

With pages_dir set, every retrieved result's page image goes inline after its
chunk, behind a citable label, the way the chat UI sends it. Text results carry
their page too: a page both legs found reaches the reader as its text chunk, and
without this it would lose its pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.llm.protocol import ChatResponse, ContentPart, ImagePart, Message, TextPart
from src.prompts.loader import Prompt
from src.rag.context import MAX_PAGE_IMAGES
from src.rag.generate import Generator
from src.types import RetrievalResult


class _RecordingLLM:
    """Captures the messages passed to chat()."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.messages = messages
        return ChatResponse(text="ok [p1::p1::c0]", model=model, tokens_in=10, tokens_out=20)

    def user_parts(self) -> list[ContentPart]:
        content = self.messages[-1].content
        assert isinstance(content, list)
        return content

    def images(self) -> list[Path]:
        return [p.path for p in self.user_parts() if isinstance(p, ImagePart)]


def _prompt() -> Prompt:
    return Prompt(name="reader", version="v0", system=None, user_template="{query}")


def _text(cid: str = "p1::p1::c0", page: int = 1, paper: str = "p1") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid,
        paper_id=paper,
        score=0.9,
        text="t",
        page_numbers=[page],
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


def _page(pages_dir: Path, paper: str, page: int, suffix: str = ".png") -> Path:
    path = pages_dir / paper / f"{paper}_p{page}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


async def test_no_images_when_pages_dir_unset() -> None:
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=None)
    await gen.answer("q?", [_visual()])
    assert llm.images() == []


async def test_text_results_carry_their_page_image(tmp_path: Path) -> None:
    path = _page(tmp_path, "p1", 1)
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    answer = await gen.answer("q?", [_text()])
    assert llm.images() == [path]
    assert "p1::p1::page" in answer.context_ids


async def test_each_image_follows_its_label(tmp_path: Path) -> None:
    path = _page(tmp_path, "paper-x", 7)
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    await gen.answer("q?", [_visual(paper="paper-x", page=7)])
    parts = llm.user_parts()
    i = next(n for n, p in enumerate(parts) if isinstance(p, ImagePart))
    label = parts[i - 1]
    assert isinstance(label, TextPart) and label.text == "[page image paper-x::p7::page]"
    assert parts[i] == ImagePart(path=path)


async def test_jpeg_pages_are_found(tmp_path: Path) -> None:
    path = _page(tmp_path, "doc", 2, suffix=".jpg")
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    await gen.answer("q?", [_visual(paper="doc", page=2)])
    assert llm.images() == [path]


async def test_missing_image_drops_with_its_label(tmp_path: Path) -> None:
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    answer = await gen.answer("q?", [_text("ghost::p99::c0", page=99, paper="ghost")])
    assert llm.images() == []
    assert all("[page image" not in p.text for p in llm.user_parts() if isinstance(p, TextPart))
    assert "ghost::p99::page" not in answer.context_ids


async def test_one_image_per_page(tmp_path: Path) -> None:
    """Two chunks and the visual result on the same page attach it once."""
    _page(tmp_path, "p1", 1)
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    await gen.answer("q?", [_text("p1::p1::c0"), _text("p1::p1::c1"), _visual(page=1)])
    assert len(llm.images()) == 1


async def test_images_capped(tmp_path: Path) -> None:
    visuals = []
    for page in range(1, MAX_PAGE_IMAGES + 3):
        _page(tmp_path, "many", page)
        visuals.append(_visual(paper="many", page=page))
    llm = _RecordingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="m", pages_dir=tmp_path)
    await gen.answer("q?", visuals)
    assert len(llm.images()) == MAX_PAGE_IMAGES


async def test_page_image_citation_resolves(tmp_path: Path) -> None:
    """The prompt tells the reader to cite page ids for what it read off a page."""
    _page(tmp_path, "p1", 1)

    class _PageCiter(_RecordingLLM):
        async def chat(self, messages: list[Message], model: str, **kwargs: Any) -> ChatResponse:
            self.messages = messages
            return ChatResponse(
                text="The plot rises [p1::p1::page].", model=model, tokens_in=1, tokens_out=1
            )

    gen = Generator(llm=_PageCiter(), prompt=_prompt(), model="m", pages_dir=tmp_path)
    answer = await gen.answer("q?", [_text()])
    [citation] = answer.citations
    assert citation.chunk_id == "p1::p1::page"
    assert citation.page_numbers == [1]
