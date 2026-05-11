"""Vision-language captioning for extracted figures.

Two backends share the same `caption(image_path) -> str` surface:

- `OllamaVisionCaptioner`: local Ollama (`gemma3:4b`, `qwen2.5vl:7b`, …).
  Free, fast on a small GPU, but quality on technical figures is mediocre
  (tested gemma3:4b + qwen2.5vl:7b: both hallucinated "heatmap" / "gene
  expression" on a scaling-law plot).
- `OpenRouterVisionCaptioner`: cloud VLM via OpenRouter (`openai/gpt-4o-mini`,
  `anthropic/claude-3.5-sonnet`, etc). Real cost, much higher caption
  fidelity. ADR 0009 §"What this leaves open" flagged this as the next
  layer; this is that layer.

Captions populate `Figure.vlm_caption`; the existing `figure_to_chunk`
preferentially uses them at index time. The `caption_figures` helper is
provider-agnostic — it duck-types on `.caption(image_path)`.

Why a dedicated module instead of reusing `OllamaChatClient` /
`OpenRouterClient`:
- Both providers' chat APIs work, but the captioner's prompt + parsing
  is a single ingest-time concern; collapsing it into the LLMClient
  protocol would broaden the protocol just to fit one user.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.observability.logging import get_logger, timed_event
from src.types import Figure


def _should_retry_vlm_request(exc: BaseException) -> bool:
    """Same retry policy as OpenRouterClient: transport errors + HTTP 429."""
    if isinstance(exc, (httpx.TransportError, httpx.RemoteProtocolError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


_log = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_CONCURRENCY = 2
# Reasoning-style VLMs (Nemotron Nano 12B VL, gemini-thinking, etc.) emit
# chain-of-thought tokens before the final caption. With the old 200-token
# cap they truncated mid-reasoning and returned empty `content`. 400 covers
# observed reasoning-heavy outputs (~150-300 reasoning + ~100 caption) and
# is still bounded for cost. gpt-4o-mini and gemma3:4b finish earlier so
# this is a no-op for non-reasoning models.
_DEFAULT_MAX_TOKENS = 400

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
        max_tokens: int = _DEFAULT_MAX_TOKENS,
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


@runtime_checkable
class _Captioner(Protocol):
    """Duck-typed interface both Ollama and OpenRouter captioners satisfy.

    `caption_figures` accepts any object exposing this single async method.
    Used so the eval harness can pick a backend at runtime without the
    helper caring which provider is in use.
    """

    async def caption(self, image_path: Path) -> str: ...


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterVisionCaptioner:
    """Cloud-VLM captioner via OpenRouter's OpenAI-compatible chat API.

    Mirrors `OllamaVisionCaptioner.caption(image_path) -> str`. Uses
    OpenAI's vision schema (`content` is a list of `text` + `image_url`
    blocks) — same shape `OpenRouterClient.chat` uses for in-context
    images on `/answer`. No streaming; one image per request.

    Permanent failures (auth error, model-not-found, malformed image)
    return an empty string and log a warning — caller falls back to the
    PDF caption. Transport errors (network blips) get up to 3 retries
    with exponential backoff.

    Cost-shape: gpt-4o-mini at vision is ~$0.0002 per low-detail image,
    ~$0.001 per high-detail. For the v3 corpus (~80 figures) that's
    $0.02-0.10 per ingestion pass. Worth it on a portfolio repo where
    caption fidelity affects the demo story.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "openai/gpt-4o-mini",
        base_url: str = _OPENROUTER_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
        prompt: str = _DEFAULT_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client
        self._prompt = prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    @retry(
        retry=retry_if_exception(_should_retry_vlm_request),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def caption(self, image_path: Path) -> str:
        """Return a caption for `image_path`. Empty string on permanent failure.

        Free-tier OpenRouter VLMs commonly return 429 under burst load;
        we raise on 429 so tenacity retries with exponential backoff.
        Other non-200 (401, 404, 5xx after retries) become a logged
        warning and an empty string — caller falls back to the PDF caption.
        """
        if not await asyncio.to_thread(image_path.exists):
            _log.warning("vlm_caption.image_missing", path=str(image_path))
            return ""

        image_bytes = await asyncio.to_thread(image_path.read_bytes)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        client = self._client
        if client is not None:
            response = await client.post(url, json=payload, headers=headers, timeout=self._timeout)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as ad_hoc:
                response = await ad_hoc.post(url, json=payload, headers=headers)

        if response.status_code == 429:
            # Raise so tenacity retries with backoff. _should_retry_vlm_request
            # returns True for HTTPStatusError(429); other 4xx/5xx fall through
            # to the warning + empty-string graceful path below.
            response.raise_for_status()
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
        choices = data.get("choices") or []
        if not choices:
            _log.warning("vlm_caption.no_choices", model=self._model, path=str(image_path))
            return ""
        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        return text


async def caption_figures(
    figures: list[Figure],
    *,
    captioner: _Captioner,
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
