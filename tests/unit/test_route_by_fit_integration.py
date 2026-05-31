"""ADR 0024 route-by-fit: resolver -> REAL Generator -> image attachment, one flow.

The route tests in test_answer_route.py use a stub generator (they prove which
PATH fires); the resolver tests in test_page_budget.py prove the page selection.
Neither exercises the seam between them: whether the real Generator actually
parses the resolver's `paper::pN::page` chunk_ids and attaches every whole-doc
page image to the LLM call. This test closes that gap with a real Generator and
an image-capturing fake LLM — no network.

It is the regression guard for two agreements that live in separate modules and
must stay in sync: the chunk_id format the resolver emits
(`page_budget.resolve_whole_doc_pages`) and the format the Generator re-parses
(`generate._PAGE_RE`), plus the cap bootstrap raises to the page budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.llm.protocol import ChatResponse, Message
from src.prompts.loader import Prompt
from src.rag.generate import Generator
from src.rag.page_budget import resolve_whole_doc_pages


class _ImageCapturingLLM:
    """LLMClient fake that records the `images` attached to the chat call."""

    def __init__(self) -> None:
        self.images: list[Path] | None = None

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
        self.images = images
        return ChatResponse(text="ok", model=model, tokens_in=1, tokens_out=1, raw={})


def _prompt() -> Prompt:
    return Prompt(
        name="answer", version="v1", system="be helpful", user_template="{query}\n{context}"
    )


def _make_doc(pages_dir: Path, paper_id: str, n_pages: int) -> None:
    paper_dir = pages_dir / paper_id
    paper_dir.mkdir(parents=True)
    for p in range(1, n_pages + 1):
        (paper_dir / f"{paper_id}_p{p}.png").write_bytes(b"\x89PNG\r\n")


async def test_whole_doc_pages_reach_the_llm_as_images(tmp_path: Path) -> None:
    # The end-to-end seam: a 6-page doc resolved by route-by-fit, fed to a REAL
    # Generator whose cap is raised to the budget (as bootstrap does when
    # page_budget is set), attaches all 6 page images to the LLM call.
    _make_doc(tmp_path, "paperX", 6)
    resolved = resolve_whole_doc_pages("paperX", tmp_path, budget=10)
    assert resolved is not None and len(resolved) == 6

    llm = _ImageCapturingLLM()
    gen = Generator(
        llm=llm,
        prompt=_prompt(),
        model="vision-model",
        pages_dir=tmp_path,
        max_vision_images=10,  # bootstrap raises the cap to page_budget
    )
    answer = await gen.answer("what does the doc say?", resolved)

    assert answer.text == "ok"
    assert llm.images is not None
    # All 6 whole-doc page images flowed through — resolver chunk_ids parsed by
    # the real Generator and resolved to real PNG paths on disk.
    assert len(llm.images) == 6
    assert {p.name for p in llm.images} == {f"paperX_p{n}.png" for n in range(1, 7)}


async def test_default_cap_truncates_whole_doc_without_raised_budget(tmp_path: Path) -> None:
    # Guard for the silent-truncation trap: the SAME 6-page resolved doc fed to a
    # default-cap Generator (no page_budget wiring) attaches only 4 — proving the
    # raised cap is load-bearing, not decorative.
    _make_doc(tmp_path, "paperX", 6)
    resolved = resolve_whole_doc_pages("paperX", tmp_path, budget=10)
    assert resolved is not None

    llm = _ImageCapturingLLM()
    gen = Generator(llm=llm, prompt=_prompt(), model="vision-model", pages_dir=tmp_path)
    await gen.answer("q", resolved)

    assert llm.images is not None
    assert len(llm.images) == 4  # default _MAX_VISION_IMAGES — would erase the win
