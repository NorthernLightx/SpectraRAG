"""Structured-extraction selector: factory dispatch and the two backends.

The factory is the operator's `RAG_EXTRACTOR_BACKEND` seam (ADR 0025). These
tests pin that it dispatches to the right backend, returns None when off, and
that each backend hits the right endpoint and degrades to a failure marker
rather than raising on a bad page.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

from src.config.settings import Settings
from src.extraction import StructuredExtractor, build_extractor
from src.extraction.structured import (
    _FAILED,
    MinerULocalExtractor,
    QwenCloudExtractor,
)


def _png(tmp_path: Path, name: str = "p1.png") -> Path:
    img = tmp_path / name
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake png bytes")
    return img


def test_build_extractor_none_returns_none() -> None:
    assert build_extractor(Settings(extractor_backend="none")) is None


def test_build_extractor_dispatches_qwen_cloud() -> None:
    ex = build_extractor(
        Settings(extractor_backend="qwen-cloud", extractor_model="qwen3-vl:235b-cloud")
    )
    assert isinstance(ex, QwenCloudExtractor)


def test_build_extractor_dispatches_mineru_local() -> None:
    ex = build_extractor(Settings(extractor_backend="mineru-local"))
    assert isinstance(ex, MinerULocalExtractor)


def test_backends_satisfy_protocol() -> None:
    assert isinstance(QwenCloudExtractor("http://x", "m"), StructuredExtractor)
    assert isinstance(MinerULocalExtractor("http://x"), StructuredExtractor)


@respx.mock
async def test_qwen_extract_transcribes_each_page(tmp_path: Path) -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"role": "assistant", "content": "col\tval\n1\t2"}}
        )
    )
    ex = QwenCloudExtractor("http://localhost:11434", "qwen3-vl:235b-cloud")

    out = await ex.extract([_png(tmp_path, "a.png"), _png(tmp_path, "b.png")])

    assert route.call_count == 2  # one request per page image
    assert out == "col\tval\n1\t2\n---\ncol\tval\n1\t2"
    body = route.calls.last.request.content
    assert b"Transcribe ALL structured visual content" in body
    assert b'"temperature": 0' in body or b'"temperature":0' in body


@respx.mock
async def test_qwen_extract_marks_failed_page_on_http_error(tmp_path: Path) -> None:
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(500))
    ex = QwenCloudExtractor("http://localhost:11434", "m")

    out = await ex.extract([_png(tmp_path)])

    assert out == _FAILED


@respx.mock
async def test_mineru_extract_reads_md_content(tmp_path: Path) -> None:
    route = respx.post("http://127.0.0.1:8011/file_parse").mock(
        return_value=httpx.Response(
            200, json={"results": {"p1": {"md_content": "| a | b |\n|---|---|\n| 1 | 2 |"}}}
        )
    )
    ex = MinerULocalExtractor("http://127.0.0.1:8011")

    out = await ex.extract([_png(tmp_path)])

    assert route.called
    assert out == "| a | b |\n|---|---|\n| 1 | 2 |"


@respx.mock
async def test_mineru_extract_marks_failed_when_no_md(tmp_path: Path) -> None:
    respx.post("http://127.0.0.1:8011/file_parse").mock(
        return_value=httpx.Response(200, json={"results": {"p1": {}}})
    )
    ex = MinerULocalExtractor("http://127.0.0.1:8011")

    out = await ex.extract([_png(tmp_path)])

    assert out == _FAILED
