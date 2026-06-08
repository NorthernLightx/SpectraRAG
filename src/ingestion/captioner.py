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
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import (
    retry,
    retry_if_exception,
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
# Serial by default: cloud VLMs (Ollama :cloud, OpenRouter) return HTTP 429 under
# concurrent bursts. One request in flight plus the 429 backoff-retry on each
# captioner keeps a bake under the rate limit; local models just run a bit slower.
_DEFAULT_CONCURRENCY = 1
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
    "preamble like 'Here's a description'. "
    # Math is destroyed by PDF text extraction (a superscript 2 reads as 'R 2');
    # the VLM sees the rendered image, so have it re-encode math as LaTeX so the
    # indexed caption is faithful and stays consistent for downstream generation.
    "Render every mathematical symbol, variable, super/subscript and equation as inline "
    "LaTeX delimited by $...$ — e.g. write $R^2$ not 'R 2', $\\Delta V_{\\text{inter}}$, "
    "$\\{L_i(r)\\}_{i=1}^{5}$. Use LaTeX only for the mathematics; leave ordinary prose as text."
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
        retry=retry_if_exception(_should_retry_vlm_request),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def caption(self, image_path: Path, *, prompt: str | None = None) -> str:
        """Return a caption string for `image_path`. Empty string on permanent failure.

        `prompt` overrides the instance prompt for a single call (used by the
        caption-relatex pass, which feeds the existing caption back in).

        Errors that won't get better on retry (404, malformed image, vision-model
        not present) become a logged warning and an empty string — caller decides
        whether to use the original PDF caption or accept no caption.
        """
        effective_prompt = self._prompt if prompt is None else prompt
        # Path I/O is blocking; offload so we don't stall the event loop while a
        # batch of figures is being captioned concurrently.
        if not await asyncio.to_thread(image_path.exists):
            _log.warning("vlm_caption.image_missing", path=str(image_path))
            return ""

        image_bytes = await asyncio.to_thread(image_path.read_bytes)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": effective_prompt, "images": [b64]}],
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

        if response.status_code == 429:
            # Raise so tenacity backs off + retries (cloud rate limit), matching
            # the OpenRouter captioner. Other non-200 fall through to the graceful
            # empty-string path below.
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

    async def caption(self, image_path: Path, *, prompt: str | None = None) -> str: ...


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
    async def caption(self, image_path: Path, *, prompt: str | None = None) -> str:
        """Return a caption for `image_path`. Empty string on permanent failure.

        `prompt` overrides the instance prompt for a single call (used by the
        caption-relatex pass, which feeds the existing caption back in).

        Free-tier OpenRouter VLMs commonly return 429 under burst load;
        we raise on 429 so tenacity retries with exponential backoff.
        Other non-200 (401, 404, 5xx after retries) become a logged
        warning and an empty string — caller falls back to the PDF caption.
        """
        effective_prompt = self._prompt if prompt is None else prompt
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
                        {"type": "text", "text": effective_prompt},
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


_DEFAULT_SKIP_ROLES: frozenset[str] = frozenset({"decoration"})


def _needs_vlm_caption(
    figure: Figure, *, skip_when_captioned: bool, skip_roles: frozenset[str]
) -> bool:
    """Decide whether `figure` is worth a VLM call.

    Default policy (ADR 0022 follow-up): skip figures that already carry a
    real PDF caption — VLM adds no retrieval value there — and skip every
    `decoration` role (logos, icons, signatures): a "this is the Microsoft
    logo" caption is wasted compute when the gallery has already binned
    those out.
    """
    if figure.role in skip_roles:
        return False
    return not (skip_when_captioned and figure.caption.strip())


async def caption_figures(
    figures: list[Figure],
    *,
    captioner: _Captioner,
    concurrency: int = _DEFAULT_CONCURRENCY,
    skip_when_captioned: bool = True,
    skip_roles: frozenset[str] = _DEFAULT_SKIP_ROLES,
) -> list[Figure]:
    """Populate `Figure.vlm_caption` only where it would change the chunk text.

    Concurrency-limited so a small VRAM device can serve the VLM model without
    thrashing. Errors on individual figures are logged and the figure is
    returned with its original `vlm_caption` (None) — `figure_to_chunk` will
    fall back to whatever else it has.

    The filter:
    - Figures whose PDF caption is already populated are passed through
      unchanged (the existing caption already feeds retrieval; replacing it
      adds latency without measured value).
    - Figures whose `role` is in `skip_roles` (default: `decoration`) are
      passed through unchanged — no point in captioning logos and icons.
    - Everything else (real figures with no caption, plus `unlabeled` real
      pictures) gets a VLM call.

    Pass `skip_when_captioned=False` to restore the pre-ADR-0022 "caption
    every figure" behaviour for baseline reproducibility.
    """
    if not figures:
        return []

    eligible_indices = [
        i
        for i, f in enumerate(figures)
        if _needs_vlm_caption(f, skip_when_captioned=skip_when_captioned, skip_roles=skip_roles)
    ]
    if not eligible_indices:
        # Nothing to do — return the input list unchanged.
        _log.info(
            "vlm_caption.skipped_all",
            n_figures=len(figures),
            reason="all figures already captioned or decoration-only",
        )
        return list(figures)

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

    with timed_event(
        _log,
        "vlm_caption.done",
        n_figures=len(figures),
        n_eligible=len(eligible_indices),
    ) as ctx:
        results = list(figures)
        captioned = await asyncio.gather(*(_one(figures[i]) for i in eligible_indices))
        for slot, fig in zip(eligible_indices, captioned, strict=True):
            results[slot] = fig
        ctx["with_caption"] = sum(1 for f in results if f.vlm_caption)
    return results


# Built by concatenation, not str.format, because the LaTeX examples contain
# literal braces ({\text{...}}) that format() would read as field names.
_RELATEX_PROMPT = (
    "The text below is the caption for the attached scientific figure, but its "
    "mathematical notation was flattened by PDF text extraction (for example a "
    "superscript two reads as 'R 2'). Using the image to disambiguate, output the "
    "caption again with exactly the same wording, order and meaning, but written so "
    "every mathematical symbol, variable, super/subscript and equation is inline LaTeX "
    "delimited by $...$ (e.g. $R^2$, $\\Delta V_{\\text{inter}}$). Change only the math "
    "formatting. Add and remove nothing else, and output no preamble.\n\nCaption:\n"
)

# Conservative flag for "this caption carries math the text layer flattened".
# A lone CAPITAL variable then a single digit (R 2, V 2) is the flattened
# super/subscript case. Capital-only so "a 1B-parameter" / "a 3D view" do not
# match, and "Figure 2" does not (the digit follows a whole word). Plus equals
# signs, math operators and Greek letters. The noqa lines below silence ruff's
# ambiguous-unicode check (it reads Greek letters as Latin look-alikes), which
# is exactly the signal we want. Surviving unicode super/subscripts already
# render, so they are not flagged.
_FLAT_MATH_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Z]\s\d(?![0-9])"
    r"|[=≤≥≈≠→↦∑∫√∈∇±×∆∞]"  # noqa: RUF001 — intentional math operators, not typos
    r"|[Α-ω]"  # noqa: RUF001 — Greek letters are the math signal, not a Latin-A typo
)


def _caption_has_math(figure: Figure) -> bool:
    """True when a figure's PDF caption looks like it carries text-flattened math.

    Skips figures that already have a VLM caption (produced with the LaTeX-aware
    prompt, so its math is already encoded). Conservative on purpose: a false
    positive costs one extra VLM call that returns the caption unchanged; a false
    negative just leaves one caption flat.
    """
    if figure.vlm_caption:
        return False
    caption = figure.caption.strip()
    if not caption:
        return False
    return bool(_FLAT_MATH_RE.search(caption))


async def relatex_captions(
    figures: list[Figure],
    *,
    captioner: _Captioner,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> list[Figure]:
    """Re-encode text-flattened math in PDF captions as LaTeX, preserving wording.

    Targets only math-looking captions (`_caption_has_math`), so clean author
    captions stay untouched and the VLM cost stays bounded. The VLM reads the
    rendered image (where a superscript is visibly raised) and re-emits the same
    caption with `$...$` math. The result is written back to `figure.caption` —
    still the author's caption, only the math fixed — so `figure_to_chunk` picks
    it up without swapping in a from-scratch VLM description.
    """
    if not figures:
        return []
    eligible = [i for i, f in enumerate(figures) if _caption_has_math(f)]
    if not eligible:
        return list(figures)

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(figure: Figure) -> Figure:
        async with semaphore:
            try:
                relatexed = await captioner.caption(
                    figure.image_path,
                    prompt=_RELATEX_PROMPT + figure.caption,
                )
            except (httpx.HTTPError, RuntimeError) as exc:
                _log.warning("relatex_caption.failed", figure_id=figure.figure_id, error=str(exc))
                return figure
        if not relatexed:
            return figure
        return figure.model_copy(update={"caption": relatexed})

    with timed_event(
        _log, "relatex_caption.done", n_figures=len(figures), n_eligible=len(eligible)
    ) as ctx:
        results = list(figures)
        fixed = await asyncio.gather(*(_one(figures[i]) for i in eligible))
        for slot, fig in zip(eligible, fixed, strict=True):
            results[slot] = fig
        ctx["relatexed"] = sum(1 for i in eligible if results[i].caption != figures[i].caption)
    return results
