"""OpenRouter implementation of LLMClient. Uses raw httpx to keep the dependency surface tight."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.llm.protocol import ChatResponse, ImagePart, Message, image_mime

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT_SECONDS = 60.0


class OpenRouterUpstreamError(RuntimeError):
    """A 200 response carrying an error/rate-limit body instead of `choices`.
    Free-tier models emit this under load (surfaced as KeyError: 'choices'
    downstream). Retryable: it usually clears on the next attempt."""


def _should_retry_request(exc: BaseException) -> bool:
    """Retry on transport errors and on HTTP 429 (rate limit). 4xx other than
    429 are not retryable (auth errors, model not found, etc); 5xx are
    retryable but typically transient. We cover transport + 429 explicitly;
    other 5xx surface immediately so they fail fast for the operator.

    Free-tier OpenRouter models commonly return 429 under burst eval load;
    without 429 retry the eval crashes on the first burst (run b30k0s5pu
    hit this on call ~120 of a v3 run). Exponential backoff with a longer
    cap (60s) covers minute-bounded rate windows.
    """
    if isinstance(exc, (httpx.TransportError, httpx.RemoteProtocolError, OpenRouterUpstreamError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def _encode_image(path: Path) -> str:
    """Image path -> data URL for an image_url block, typed by extension."""
    encoded = base64.standard_b64encode(path.read_bytes()).decode()
    return f"data:{image_mime(path)};base64,{encoded}"


def _message_dict(message: Message) -> dict[str, Any]:
    """OpenAI-compat message: string content as is, parts as content blocks."""
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}
    blocks: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _encode_image(part.path)}}
        if isinstance(part, ImagePart)
        else {"type": "text", "text": part.text}
        for part in message.content
    ]
    return {"role": message.role, "content": blocks}


class OpenRouterClient:
    """LLMClient backed by OpenRouter (OpenAI-compatible API)."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _OPENROUTER_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client

    @retry(
        retry=retry_if_exception(_should_retry_request),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Retry wraps only the network round-trip: chat() builds the (possibly
        # image-laden) payload once, then this retries the POST on transport
        # errors / 429 without re-encoding images on every attempt.
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}{path}"
        if self._client is not None:
            response = await self._client.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"OpenRouter returned non-object response: {type(data).__name__}")
        # Free-tier models intermittently return a 200 whose body is an
        # error/rate-limit object with no `choices`. Raise a retryable error so
        # the backoff above gets another attempt instead of crashing the run on
        # `data["choices"]` (the agenda's KeyError: 'choices').
        if not data.get("choices"):
            raise OpenRouterUpstreamError(
                f"OpenRouter response had no choices (transient upstream body): {data.get('error')}"
            )
        return data

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
        # `images`, when given, are appended to the LAST user message as
        # image_url blocks (the OpenAI-compat schema for vision input), after
        # whatever text or parts it already carries.
        msg_dicts: list[dict[str, Any]] = [_message_dict(m) for m in messages]
        if images:
            for m in reversed(msg_dicts):
                if m["role"] == "user":
                    content = m["content"]
                    blocks: list[dict[str, Any]] = (
                        content
                        if isinstance(content, list)
                        else [{"type": "text", "text": content}]
                    )
                    for img_path in images:
                        blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": _encode_image(img_path)},
                            }
                        )
                    m["content"] = blocks
                    break
        payload: dict[str, Any] = {
            "model": model,
            "messages": msg_dicts,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        data = await self._post("/chat/completions", payload)

        choice = data["choices"][0]["message"]
        usage = data.get("usage", {}) or {}
        return ChatResponse(
            text=choice.get("content", "") or "",
            model=data.get("model", model),
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            raw=data,
        )
