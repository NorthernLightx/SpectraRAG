"""Structured-extraction backends (ADR 0025).

A structured extractor transcribes a page's tables and charts to plain text
offline, so the reader can be handed the *data* alongside the page image
instead of re-reading pixels at query time (the +0.12 lever, ADR 0025).

The backend is an operator/deploy choice (`RAG_EXTRACTOR_BACKEND`), not a
per-query one: extraction is slow and runs once per page at ingest. Two
backends are offered, mirroring the eval bench (`scripts/experiments/
extract_bench.py`):

- ``qwen-cloud``   — page images through Ollama to a vision model. Accurate,
  remote, the prior extractor. Routed via Ollama (incl. its ``:cloud``
  passthrough), never a third-party API.
- ``mineru-local`` — page images POSTed to a local MinerU2.5 API server that
  fits the 8 GB GPU. Free, private, matches the cloud on recall (ADR 0025),
  but ~1-3 min/page.

``build_extractor(settings)`` returns the configured backend, or ``None`` when
``extractor_backend == "none"`` (the default — pipeline unchanged). The
ingest-time consumer that calls ``extract`` is gated on the lever clearing
significance (ADR 0025 "Decision" §3); this module is the selector seam.

A per-page failure yields a ``__extract_failed__`` marker in that page's slot
rather than raising, so one unreadable page can't abort a whole document — the
same convention the recall scorer understands.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from src.config.settings import Settings

# Slow per page (local VLM ~1-3 min/page; cloud model seconds), so the read
# timeout is generous — extraction is an offline ingest step, not a hot path.
_DEFAULT_TIMEOUT_SECONDS = 300.0

# Joins the per-page transcriptions of a multi-page extraction.
_PAGE_SEP = "\n---\n"

# Marks a page whose extraction failed (retries exhausted). Kept identical to
# the eval bench so the recall scorer treats it as not-scorable, not a 0.
_FAILED = "__extract_failed__"

# Transcription-only prompt (verbatim from the eval bench so production and the
# measured ADR 0025 recall use the same instruction). Charts are the hard case:
# OCR parsers are chart-blind, so we explicitly ask for series + data points.
_EXTRACT_PROMPT = (
    "Transcribe ALL structured visual content on this page as plain text:\n"
    "- Tables: tab-separated, header row then data rows, every cell.\n"
    "- Charts/plots: each series and its data points (label: value), axis labels, legend.\n"
    "- Maps/colour figures: the legend (what each colour represents) and labelled regions.\n"
    "Do NOT answer any question — only transcribe. Reply 'NONE' if no structured content."
)


@runtime_checkable
class StructuredExtractor(Protocol):
    """Transcribes a page's structured content (tables/charts) to plain text."""

    async def extract(self, images: Sequence[Path]) -> str:
        """Transcribe each page image; return the joined per-page text.

        Pages are joined by ``_PAGE_SEP``. A page that fails after retries
        contributes a ``_FAILED`` marker in its slot instead of raising.
        """
        ...


class QwenCloudExtractor:
    """Transcribe via a vision model served by Ollama (incl. ``:cloud`` tags).

    POSTs one ``/api/chat`` request per page image at temperature 0. The
    optional ``client`` lets tests inject a fake transport; production passes
    none and a short-lived client is created per call.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = client

    async def extract(self, images: Sequence[Path]) -> str:
        out: list[str] = []
        for image in images:
            out.append(await self._extract_one(image))
        return _PAGE_SEP.join(out)

    async def _extract_one(self, image: Path) -> str:
        raw = await asyncio.to_thread(image.read_bytes)
        b64 = base64.standard_b64encode(raw).decode()
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [{"role": "user", "content": _EXTRACT_PROMPT, "images": [b64]}],
            "options": {"temperature": 0, "num_predict": 900},
        }
        try:
            data = await self._post("/api/chat", payload)
        except httpx.HTTPError:
            return _FAILED
        return (data.get("message", {}).get("content", "") or "").strip()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.RemoteProtocolError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        if self._client is not None:
            response = await self._client.post(url, json=payload, timeout=self._timeout)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Ollama returned non-object response: {type(data).__name__}")
        return data


class MinerULocalExtractor:
    """Transcribe via a local MinerU2.5 API server (model preloaded once).

    POSTs each page image to ``/file_parse`` and reads back the markdown. The
    preloaded server is the efficient form — the per-page CLI reloads the 1.2B
    model every page (ADR 0025 "Decision" §2). ``client`` is injectable for tests.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._timeout = timeout
        self._client = client

    async def extract(self, images: Sequence[Path]) -> str:
        out: list[str] = []
        for image in images:
            out.append(await self._extract_one(image))
        return _PAGE_SEP.join(out)

    async def _extract_one(self, image: Path) -> str:
        try:
            raw = await asyncio.to_thread(image.read_bytes)
            files = {"files": (image.name, raw, "image/png")}
            data = {"backend": "vlm-auto-engine", "return_md": "true", "source": "local"}
            if self._client is not None:
                response = await self._client.post(
                    f"{self._url}/file_parse", files=files, data=data, timeout=self._timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(f"{self._url}/file_parse", files=files, data=data)
            response.raise_for_status()
            results = response.json().get("results", {})
        except (httpx.HTTPError, OSError):
            return _FAILED
        md = next(
            (
                v["md_content"]
                for v in results.values()
                if isinstance(v, dict) and "md_content" in v
            ),
            "",
        )
        return md or _FAILED


def build_extractor(settings: Settings) -> StructuredExtractor | None:
    """Construct the operator-selected structured extractor, or None for "none".

    Reads ``settings.extractor_backend`` and wires the matching backend from the
    related settings (``ollama_base_url`` + ``extractor_model`` for qwen-cloud,
    ``mineru_url`` for mineru-local). Returns None when extraction is off.
    """
    backend = settings.extractor_backend
    if backend == "none":
        return None
    if backend == "qwen-cloud":
        return QwenCloudExtractor(settings.ollama_base_url, settings.extractor_model)
    if backend == "mineru-local":
        return MinerULocalExtractor(settings.mineru_url)
    # Unreachable: extractor_backend is a closed Literal validated by pydantic.
    raise ValueError(f"unknown extractor_backend: {backend!r}")
