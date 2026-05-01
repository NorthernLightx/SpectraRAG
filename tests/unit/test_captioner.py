"""OllamaVisionCaptioner: image POST shape, fallback on errors, caption_figures concurrency."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

from src.ingestion.captioner import OllamaVisionCaptioner, caption_figures
from src.types import Figure


def _figure(figure_id: str, image_path: Path, caption: str = "") -> Figure:
    return Figure(
        figure_id=figure_id,
        paper_id="paper1",
        page_number=1,
        caption=caption,
        image_path=image_path,
    )


def _write_png(path: Path) -> Path:
    """Write a 1x1 PNG so respx-mocked calls have something to base64-encode."""
    # Smallest valid PNG: 8-byte signature + IHDR + IDAT + IEND.
    minimal_png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000007cdd"
        "5d4c0000000a49444154789c63000100000005000100"
        "0d0a2db40000000049454e44ae426082"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(minimal_png)
    return path


@respx.mock
async def test_captioner_posts_base64_image_to_chat_endpoint(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path / "fig1.png")
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gemma3:4b",
                "message": {"role": "assistant", "content": "A scatter plot of X vs Y."},
                "done": True,
            },
        )
    )

    captioner = OllamaVisionCaptioner(model="gemma3:4b")
    caption = await captioner.caption(image_path)

    assert route.called
    assert caption == "A scatter plot of X vs Y."
    body = route.calls.last.request.content
    assert b'"images":' in body  # JSON has images field
    assert b'"model":"gemma3:4b"' in body.replace(b" ", b"")


@respx.mock
async def test_captioner_returns_empty_string_on_404(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path / "fig1.png")
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(404))
    captioner = OllamaVisionCaptioner(model="missing-model:1b")
    assert await captioner.caption(image_path) == ""


async def test_captioner_returns_empty_when_image_missing(tmp_path: Path) -> None:
    captioner = OllamaVisionCaptioner(model="gemma3:4b")
    assert await captioner.caption(tmp_path / "does_not_exist.png") == ""


@respx.mock
async def test_caption_figures_populates_vlm_caption(tmp_path: Path) -> None:
    figs = [
        _figure("f1", _write_png(tmp_path / "f1.png"), caption="Figure 1: original."),
        _figure("f2", _write_png(tmp_path / "f2.png"), caption="Figure 2: original."),
    ]
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "model": "gemma3:4b",
                    "message": {"content": "VLM caption for fig 1."},
                    "done": True,
                },
            ),
            httpx.Response(
                200,
                json={
                    "model": "gemma3:4b",
                    "message": {"content": "VLM caption for fig 2."},
                    "done": True,
                },
            ),
        ]
    )

    captioner = OllamaVisionCaptioner(model="gemma3:4b")
    out = await caption_figures(figs, captioner=captioner, concurrency=1)

    assert [f.vlm_caption for f in out] == [
        "VLM caption for fig 1.",
        "VLM caption for fig 2.",
    ]
    # Original captions are preserved on the new objects (model_copy).
    assert [f.caption for f in out] == ["Figure 1: original.", "Figure 2: original."]


@respx.mock
async def test_caption_figures_keeps_original_when_vlm_returns_empty(tmp_path: Path) -> None:
    figs = [_figure("f1", _write_png(tmp_path / "f1.png"))]
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"model": "gemma3:4b", "message": {"content": "  "}, "done": True}
        )
    )
    captioner = OllamaVisionCaptioner(model="gemma3:4b")
    [out] = await caption_figures(figs, captioner=captioner)
    assert out.vlm_caption is None  # whitespace-only strips to empty → original kept


async def test_caption_figures_returns_empty_for_no_figures() -> None:
    captioner = OllamaVisionCaptioner(model="gemma3:4b")
    assert await caption_figures([], captioner=captioner) == []
