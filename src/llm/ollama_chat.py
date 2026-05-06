"""Ollama chat over HTTP. Duck-typed against LLMClient. Local-only, no API key."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.protocol import ChatResponse, Message

# Local CPU-served chat models can take 30s-3min per generation on cold start
# or under client-side concurrency (Ollama serializes by default), so we use a
# generous read timeout. Cloud LLMs return in <10s and aren't sensitive to this.
_DEFAULT_TIMEOUT_SECONDS = 600.0


class OllamaChatClient:
    """LLMClient backed by Ollama's /api/chat. Local validation alternative to cloud LLMs."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
        num_ctx: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client
        self._num_ctx = num_ctx

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

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.RemoteProtocolError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        images: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        # `images` is part of LLMClient for OpenRouter vision support; this
        # local-Ollama client doesn't pipe images through `/api/chat` (Ollama
        # has a different per-message base64 image field that isn't worth
        # bridging until a local vision generator becomes a goal). Silently
        # ignored so the protocol stays uniform.
        del images  # unused
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        # Forward arbitrary Ollama-specific options (top_p, num_ctx, …) without re-mapping.
        for key, value in kwargs.items():
            options.setdefault(key, value)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": options,
        }

        data = await self._post("/api/chat", payload)

        message = data.get("message") or {}
        return ChatResponse(
            text=message.get("content", "") or "",
            model=data.get("model", model),
            tokens_in=int(data.get("prompt_eval_count", 0) or 0),
            tokens_out=int(data.get("eval_count", 0) or 0),
            raw=data,
        )
