"""The reader's messages for one turn, built once for the chat UI, /answer and the eval.

ADR 0033. The chat UI sends these messages to the visitor's provider from the
browser (it fetches them from POST /context); the Generator sends them through
an LLMClient; eval_run measures the Generator. All three read the same prompt,
the same image policy and the same figure-caption injection.

Page images appear as `PageImageRef` parts, each preceded by its citable label.
Whoever sends the messages resolves a ref to the image bytes and drops the
label with it when the image is missing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from src.llm.protocol import Role, TextPart
from src.prompts.loader import Prompt
from src.types import Chunk, RetrievalResult

# The prompt in src/prompts/library the reader runs on everywhere.
READER_PROMPT_NAME = "chat"
# Page images attached for retrieved chunks, before the ones for figures the
# question names. A page image averages ~0.5 MB as base64.
MAX_PAGE_IMAGES = 6
# Named figures whose caption and page join the context.
_MAX_INJECTED = 2
_INJECTED_CAPTION_CHARS = 700
# How many top chunks decide which papers a "Figure N" can refer to.
_INJECT_FROM_TOP = 3
_CHARS_PER_TOKEN = 4

# "Figure 2", "figs. 2 and 3", "tables 1, 2": one keyword, a list of numbers.
_FIGURE_REF_RE = re.compile(
    "\\b(figs?\\.?|figures?|tables?)\\s*(\\d+(?:\\s*(?:,|and|&|\N{EN DASH}|-)\\s*\\d+)*)\\b",
    re.IGNORECASE,
)


class PageImageRef(BaseModel):
    """A page image to attach, named by paper and 1-based page."""

    type: Literal["page_image"] = "page_image"
    paper_id: str
    page: int
    label: str


ReaderPart = TextPart | PageImageRef


class ReaderMessage(BaseModel):
    role: Role
    content: str | list[ReaderPart]


class PriorTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class FigureRef:
    """A figure or table chunk the question can name by number."""

    chunk_id: str
    paper_id: str
    page: int
    caption: str
    bbox: list[float] | None = None


@dataclass(frozen=True)
class ReaderContext:
    messages: list[ReaderMessage]
    # Retrieved results whose text fit the budget, in order.
    used: list[RetrievalResult]
    injected: list[FigureRef]
    # Every id the reader was shown and may cite: used chunks, injected
    # captions, attached page images.
    context_ids: list[str] = field(default_factory=list)


def page_image_id(paper_id: str, page: int) -> str:
    return f"{paper_id}::p{page}::page"


def figure_refs(chunks: Iterable[Chunk]) -> list[FigureRef]:
    """Figure and table chunks in (paper, page, chunk id) order, the order the
    /figures gallery lists them in."""
    refs: list[FigureRef] = []
    for chunk in chunks:
        if chunk.metadata.get("kind") not in {"figure", "table"} or not chunk.page_numbers:
            continue
        bbox_raw = chunk.metadata.get("bbox")
        bbox = (
            [float(v) for v in bbox_raw]
            if isinstance(bbox_raw, list)
            and len(bbox_raw) == 4
            and all(isinstance(v, (int, float)) for v in bbox_raw)
            else None
        )
        refs.append(
            FigureRef(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                page=chunk.page_numbers[0],
                caption=chunk.text,
                bbox=bbox,
            )
        )
    refs.sort(key=lambda r: (r.paper_id, r.page, r.chunk_id))
    return refs


def named_figures(
    question: str, retrieved: Sequence[RetrievalResult], figures: Sequence[FigureRef]
) -> list[FigureRef]:
    """Figures and tables the question names by number, found in the papers of
    the top retrieved chunks. Retrieval tends to return text that mentions a
    figure from another page, and captions rarely share words with the question,
    so without this the page that shows the figure never reaches the reader."""
    if not figures:
        return []
    refs: list[tuple[bool, str]] = []
    for match in _FIGURE_REF_RE.finditer(question):
        is_table = match.group(1).lower().startswith("t")
        refs.extend((is_table, num) for num in re.findall(r"\d+", match.group(2)))
    if not refs:
        return []
    papers = list(dict.fromkeys(r.paper_id for r in retrieved[:_INJECT_FROM_TOP]))
    out: list[FigureRef] = []
    for is_table, num in refs:
        pattern = re.compile(
            rf"^table\.?\s*{num}\b" if is_table else rf"^fig(?:ure)?\.?\s*{num}\b",
            re.IGNORECASE,
        )
        for paper_id in papers:
            found = next(
                (f for f in figures if f.paper_id == paper_id and pattern.match(f.caption.strip())),
                None,
            )
            if found is not None:
                # One page per reference: the same "Figure 2" in a second paper
                # would crowd out the question's other references.
                out.append(found)
                break
    return out[:_MAX_INJECTED]


def build_reader_context(
    question: str,
    retrieved: Sequence[RetrievalResult],
    *,
    prompt: Prompt,
    prior_turns: Sequence[PriorTurn] = (),
    figures: Sequence[FigureRef] = (),
    include_images: bool = True,
    max_images: int = MAX_PAGE_IMAGES,
    max_context_tokens: int | None = None,
    image_available: Callable[[str, int], bool] | None = None,
) -> ReaderContext:
    """System prompt, prior turns, then one user message: each retrieved chunk
    followed by its page images, the captions and pages of figures the question
    names, and the prompt's question block.

    `max_context_tokens` bounds the chunk text (~4 chars per token; the first
    chunk always goes in). `image_available` lets a caller that knows which
    page files exist skip missing ones before they count against `max_images`.
    """
    messages: list[ReaderMessage] = []
    if prompt.system:
        messages.append(ReaderMessage(role="system", content=prompt.system))
    messages.extend(ReaderMessage(role=t.role, content=t.text) for t in prior_turns)

    parts: list[ReaderPart] = []
    context_ids: list[str] = []
    seen_pages: set[tuple[str, int]] = set()

    def attach(paper_id: str, page: int) -> bool:
        key = (paper_id, page)
        if not include_images or key in seen_pages:
            return False
        if image_available is not None and not image_available(paper_id, page):
            return False
        seen_pages.add(key)
        pid = page_image_id(paper_id, page)
        parts.append(PageImageRef(paper_id=paper_id, page=page, label=f"[page image {pid}]"))
        context_ids.append(pid)
        return True

    budget = None if max_context_tokens is None else max_context_tokens * _CHARS_PER_TOKEN
    used: list[RetrievalResult] = []
    spent = 0
    n_images = 0
    for r in retrieved:
        pages = ",".join(str(p) for p in r.page_numbers)
        block = f"[chunk {r.chunk_id}] paper={r.paper_id} pages={pages}\n{r.text or ''}"
        if budget is not None and used and spent + len(block) > budget:
            break
        spent += len(block)
        used.append(r)
        parts.append(TextPart(text=block))
        context_ids.append(r.chunk_id)
        for page in r.page_numbers:
            if n_images >= max_images:
                break
            if attach(r.paper_id, page):
                n_images += 1

    injected = named_figures(question, retrieved, figures)
    for fig in injected:
        caption = fig.caption[:_INJECTED_CAPTION_CHARS]
        parts.append(
            TextPart(
                text=f"[chunk {fig.chunk_id}] paper={fig.paper_id} pages={fig.page}: caption "
                f"of the figure/table named in the question\n{caption}"
            )
        )
        context_ids.append(fig.chunk_id)
        attach(fig.paper_id, fig.page)

    parts.append(TextPart(text=prompt.render(query=question)[1]))
    messages.append(ReaderMessage(role="user", content=parts))
    return ReaderContext(
        messages=messages,
        used=used,
        injected=injected,
        context_ids=list(dict.fromkeys(context_ids)),
    )
