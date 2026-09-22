"""The reader context builder and POST /context (ADR 0033)."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from src.api.deps import _ChunksState, get_settings
from src.api.main import create_app
from src.config.settings import Settings
from src.llm.ollama_chat import OllamaChatClient
from src.llm.openrouter import OpenRouterClient
from src.llm.protocol import ImagePart, Message, TextPart
from src.prompts.loader import Prompt, load_prompt_by_name
from src.rag.context import (
    MAX_PAGE_IMAGES,
    READER_PROMPT_NAME,
    FigureRef,
    PageImageRef,
    PriorTurn,
    build_reader_context,
    figure_refs,
    named_figures,
)
from src.types import Chunk, RetrievalResult

_PROMPT = Prompt(name="reader", version="v0", system="sys", user_template="Q: {query}")


def _result(paper: str, page: int, n: int = 0, text: str = "body") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper}::p{page}::c{n}",
        paper_id=paper,
        score=0.5,
        text=text,
        page_numbers=[page],
        source="pipeline",
    )


def _fig(paper: str, page: int, caption: str, n: int = 1) -> FigureRef:
    return FigureRef(
        chunk_id=f"{paper}::p{page}::fig{n}", paper_id=paper, page=page, caption=caption
    )


def _user_parts(messages: list[Any]) -> list[Any]:
    content = messages[-1].content
    assert isinstance(content, list)
    return list(content)


def test_the_reader_prompt_is_the_chat_prompt() -> None:
    prompt = load_prompt_by_name(READER_PROMPT_NAME)
    assert prompt.name == "chat"
    assert "Not stated in the provided context." in (prompt.system or "")
    assert prompt.render(query="Why?")[1].startswith("\nQuestion: Why?\n(Reminder:")


def test_layout_is_system_then_turns_then_chunks_then_question() -> None:
    ctx = build_reader_context(
        "what?",
        [_result("a", 1), _result("a", 2)],
        prompt=_PROMPT,
        prior_turns=[PriorTurn(role="user", text="hi"), PriorTurn(role="assistant", text="yo")],
    )
    assert [m.role for m in ctx.messages] == ["system", "user", "assistant", "user"]
    parts = _user_parts(ctx.messages)
    assert parts[0] == TextPart(text="[chunk a::p1::c0] paper=a pages=1\nbody")
    assert isinstance(parts[1], PageImageRef) and parts[1].label == "[page image a::p1::page]"
    assert parts[-1] == TextPart(text="Q: what?")
    assert ctx.context_ids == ["a::p1::c0", "a::p1::page", "a::p2::c0", "a::p2::page"]


def test_images_off_sends_text_only() -> None:
    ctx = build_reader_context("q", [_result("a", 1)], prompt=_PROMPT, include_images=False)
    assert not any(isinstance(p, PageImageRef) for p in _user_parts(ctx.messages))


def test_unavailable_pages_do_not_take_an_image_slot() -> None:
    results = [_result("a", p) for p in range(1, MAX_PAGE_IMAGES + 3)]
    ctx = build_reader_context(
        "q", results, prompt=_PROMPT, image_available=lambda paper, page: page != 1
    )
    pages = [p.page for p in _user_parts(ctx.messages) if isinstance(p, PageImageRef)]
    assert pages == list(range(2, MAX_PAGE_IMAGES + 2))


def test_budget_keeps_the_first_chunk_and_stops_before_overrun() -> None:
    results = [_result("a", p, text="x" * 100) for p in range(1, 5)]
    ctx = build_reader_context("q", results, prompt=_PROMPT, max_context_tokens=10)
    assert [r.chunk_id for r in ctx.used] == ["a::p1::c0"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does Figure 2 show?", ["a::p5::fig2"]),
        ("Compare figures 2 and 3", ["a::p5::fig2", "a::p7::fig3"]),
        ("What is in Table 1?", ["a::p3::tab1"]),
        ("Explain Fig. 3", ["a::p7::fig3"]),
        ("What is the method?", []),
    ],
)
def test_named_figures(question: str, expected: list[str]) -> None:
    figures = [
        _fig("a", 5, "Figure 2: training loss", n=2),
        _fig("a", 7, "Fig. 3. Accuracy over time", n=3),
        FigureRef(chunk_id="a::p3::tab1", paper_id="a", page=3, caption="Table 1: datasets"),
        _fig("b", 2, "Figure 2: other paper", n=2),
    ]
    found = named_figures(question, [_result("a", 1)], figures)
    assert [f.chunk_id for f in found] == expected


def test_named_figures_look_only_in_the_top_papers() -> None:
    figures = [_fig("z", 4, "Figure 2: unrelated paper", n=2)]
    top = [_result("a", 1), _result("b", 1), _result("c", 1), _result("z", 1)]
    assert named_figures("Figure 2?", top, figures) == []


def test_injected_figure_adds_caption_and_page() -> None:
    fig = _fig("a", 5, "Figure 2: training loss", n=2)
    ctx = build_reader_context(
        "What does Figure 2 show?", [_result("a", 1)], prompt=_PROMPT, figures=[fig]
    )
    parts = _user_parts(ctx.messages)
    captions = [
        p.text for p in parts if isinstance(p, TextPart) and "named in the question" in p.text
    ]
    assert captions == [
        "[chunk a::p5::fig2] paper=a pages=5: caption of the figure/table named in the "
        "question\nFigure 2: training loss"
    ]
    assert "a::p5::page" in ctx.context_ids
    assert ctx.injected == [fig]


def test_figure_refs_come_from_figure_and_table_chunks() -> None:
    chunks = [
        Chunk(chunk_id="a::p2::c0", paper_id="a", page_numbers=[2], text="body"),
        Chunk(
            chunk_id="a::p4::fig1",
            paper_id="a",
            page_numbers=[4],
            text="Figure 1: x",
            metadata={"kind": "figure", "bbox": [1, 2, 3, 4]},
        ),
        Chunk(
            chunk_id="a::p1::tab1",
            paper_id="a",
            page_numbers=[1],
            text="Table 1: y",
            metadata={"kind": "table"},
        ),
    ]
    refs = figure_refs(chunks)
    assert [r.chunk_id for r in refs] == ["a::p1::tab1", "a::p4::fig1"]
    assert refs[1].bbox == [1.0, 2.0, 3.0, 4.0]


# ---------- POST /context -------------------------------------------------


@pytest.fixture
def _corpus() -> Iterator[None]:
    fig = Chunk(
        chunk_id="a::p5::fig2",
        paper_id="a",
        page_numbers=[5],
        text="Figure 2: training loss",
        metadata={"kind": "figure"},
    )
    _ChunksState.instance = {fig.chunk_id: fig}
    yield
    _ChunksState.instance = None


def _client(pages_dir: Path | None) -> TestClient:
    app = create_app(log_file=None)
    app.dependency_overrides[get_settings] = lambda: Settings(pages_dir=pages_dir)
    return TestClient(app)


def _post(client: TestClient, **body: Any) -> dict[str, Any]:
    payload = {
        "question": "What does Figure 2 show?",
        "results": [_result("a", 1).model_dump()],
        "prior_turns": [{"role": "user", "text": "earlier"}],
        **body,
    }
    resp = client.post("/context", json=payload)
    assert resp.status_code == 200, resp.text
    body_out: dict[str, Any] = resp.json()
    return body_out


def test_context_route_returns_the_builders_messages(tmp_path: Path, _corpus: None) -> None:
    for page in (1, 5):
        (tmp_path / "a").mkdir(exist_ok=True)
        (tmp_path / "a" / f"a_p{page}.png").write_bytes(b"png")
    body = _post(_client(tmp_path))
    assert [m["role"] for m in body["messages"]] == ["system", "user", "user"]
    parts = body["messages"][-1]["content"]
    assert {
        "type": "page_image",
        "paper_id": "a",
        "page": 1,
        "label": "[page image a::p1::page]",
    } in parts
    assert body["injected"][0]["chunk_id"] == "a::p5::fig2"
    assert body["prompt_version"].startswith("v1-")
    assert "a::p5::page" in body["context_ids"]


def test_context_route_skips_images_without_rendered_pages(_corpus: None) -> None:
    body = _post(_client(None))
    parts = body["messages"][-1]["content"]
    assert not [p for p in parts if p["type"] == "page_image"]


def test_context_route_skips_a_missing_page_render(tmp_path: Path, _corpus: None) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a_p5.png").write_bytes(b"png")
    body = _post(_client(tmp_path))
    pages = [p["page"] for p in body["messages"][-1]["content"] if p["type"] == "page_image"]
    assert pages == [5]


def test_context_route_takes_a_long_conversation(_corpus: None) -> None:
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"t{i}"} for i in range(60)]
    body = _post(_client(None), prior_turns=turns)
    assert len(body["messages"]) == 1 + 60 + 1


def test_context_route_does_not_probe_outside_the_pages_dir(tmp_path: Path, _corpus: None) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    (tmp_path / "outside_p1.png").write_bytes(b"png")
    traversal = _result("a", 1).model_copy(update={"paper_id": "../outside"})
    body = _post(_client(pages), results=[traversal.model_dump()])
    assert not [p for p in body["messages"][-1]["content"] if p["type"] == "page_image"]


# ---------- LLM clients: interleaved parts ---------------------------------


def _parts_message(tmp_path: Path) -> Message:
    png = tmp_path / "p.png"
    png.write_bytes(b"png-bytes")
    jpg = tmp_path / "p.jpg"
    jpg.write_bytes(b"jpg-bytes")
    return Message(
        role="user",
        content=[
            TextPart(text="[page image a::p1::page]"),
            ImagePart(path=png),
            TextPart(text="[page image a::p2::page]"),
            ImagePart(path=jpg),
            TextPart(text="Q?"),
        ],
    )


@respx.mock
async def test_openrouter_sends_parts_in_order_with_their_mime(tmp_path: Path) -> None:
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "model": "m"}
        )
    )
    await OpenRouterClient(api_key="k").chat([_parts_message(tmp_path)], "m")
    blocks = json.loads(route.calls[0].request.content)["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image_url", "text", "image_url", "text"]
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert blocks[3]["image_url"]["url"] == (
        "data:image/jpeg;base64," + base64.standard_b64encode(b"jpg-bytes").decode()
    )


@respx.mock
async def test_ollama_joins_text_and_keeps_image_order(tmp_path: Path) -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "ok"}, "model": "m"})
    )
    await OllamaChatClient().chat([_parts_message(tmp_path)], "m")
    [msg] = json.loads(route.calls[0].request.content)["messages"]
    assert msg["content"] == "[page image a::p1::page]\n[page image a::p2::page]\nQ?"
    assert msg["images"] == [
        base64.b64encode(b"png-bytes").decode(),
        base64.b64encode(b"jpg-bytes").decode(),
    ]
