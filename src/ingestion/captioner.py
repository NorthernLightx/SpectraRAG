"""Vision-language captioning for extracted figures.

Wraps Ollama's `/api/chat` with the `images` field (base64-encoded PNGs) so a
local VLM (gemma3, qwen2.5vl, llava-llama3, …) can generate richer captions than
the PDF-extracted ones. Captions populate `Figure.vlm_caption`; the existing
`figure_to_chunk` then preferentially uses them at index time.

Why a dedicated module instead of reusing `OllamaChatClient`:
- Ollama's vision schema puts `images` at the message level, not in `options`,
  so it doesn't fit the kwargs forwarding the chat client uses.
- VLM captioning is an ingest-time concern; we don't want to broaden
  `LLMClient` until a *second* concrete VLM impl forces a real protocol seam.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.observability.logging import get_logger, timed_event
from src.types import Figure

_log = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_CONCURRENCY = 2

_DEFAULT_PROMPT = (
    "You are captioning a figure from a scientific paper. In 1-3 sentences, describe: "
    "(a) the type of plot or diagram, (b) what variables/axes/categories appear, and "
    "(c) the key quantitative or qualitative takeaway shown. Be specific. Avoid "
    "generic phrases like 'this is a figure' or 'a chart is shown'. Do not include "
    "preamble like 'Here's a description'."
)


class OllamaVisionCaptioner:
    """Single-image captioner backed by Ollama's vision-capable models.

    `model` must be a vision-capable Ollama tag (e.g. `gemma3:4b`,
    `qwen2.5vl:7b`, `llava-llama3:8b`). Wrong model → either an obvious
    error from Ollama or, worse, an empty caption — handle both.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
        prompt: str = _DEFAULT_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = client
        self._prompt = prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.RemoteProtocolError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def caption(self, image_path: Path) -> str:
        """Return a caption string for `image_path`. Empty string on permanent failure.

        Errors that won't get better on retry (404, malformed image, vision-model
        not present) become a logged warning and an empty string — caller decides
        whether to use the original PDF caption or accept no caption.
        """
        # Path I/O is blocking; offload so we don't stall the event loop while a
        # batch of figures is being captioned concurrently.
        if not await asyncio.to_thread(image_path.exists):
            _log.warning("vlm_caption.image_missing", path=str(image_path))
            return ""

        image_bytes = await asyncio.to_thread(image_path.read_bytes)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": self._prompt, "images": [b64]}],
            "stream": False,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
        }

        url = f"{self._base_url}/api/chat"
        client = self._client
        if client is not None:
            response = await client.post(url, json=payload, timeout=self._timeout)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as ad_hoc:
                response = await ad_hoc.post(url, json=payload)

        if response.status_code != 200:
            _log.warning(
                "vlm_caption.http_error",
                model=self._model,
                status=response.status_code,
                path=str(image_path),
            )
            return ""

        data = response.json()
        if not isinstance(data, dict):
            _log.warning("vlm_caption.bad_response", path=str(image_path))
            return ""
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        return text


async def caption_figures(
    figures: list[Figure],
    *,
    captioner: OllamaVisionCaptioner,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> list[Figure]:
    """Populate `Figure.vlm_caption` for every figure (in-place clones via model_copy).

    Concurrency-limited so a small VRAM device can serve the VLM model without
    thrashing. Errors on individual figures are logged and the figure is
    returned with its original `vlm_caption` (None) — `figure_to_chunk` will
    fall back to the PDF-extracted caption.
    """
    if not figures:
        return []

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(figure: Figure) -> Figure:
        async with semaphore:
            try:
                caption = await captioner.caption(figure.image_path)
            except (httpx.HTTPError, RuntimeError) as exc:
                _log.warning(
                    "vlm_caption.failed",
                    figure_id=figure.figure_id,
                    error=str(exc),
                )
                return figure
        if not caption:
            return figure
        return figure.model_copy(update={"vlm_caption": caption})

    with timed_event(_log, "vlm_caption.done", n_figures=len(figures)) as ctx:
        results = await asyncio.gather(*(_one(f) for f in figures))
        ctx["with_caption"] = sum(1 for f in results if f.vlm_caption)
    return list(results)
