"""Ollama chat over HTTP. Duck-typed against LLMClient. Local-only, no API key."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.protocol import ChatResponse, ImagePart, Message

# Local CPU-served chat models can take 30s-3min per generation on cold start
# or under client-side concurrency (Ollama serializes by default), so we use a
# generous read timeout. Cloud LLMs return in <10s and aren't sensitive to this.
_DEFAULT_TIMEOUT_SECONDS = 600.0


def _encode_image(image: Any) -> str:
    """Base64-encode one image for Ollama: a path, raw bytes, or an existing string."""
    if isinstance(image, str):
        return image
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")
    path = Path(image)
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _message_dict(message: Message) -> dict[str, Any]:
    """Ollama-native message: text parts joined in order, images as base64 in
    `images`. Labels in the text keep the order the images arrive in."""
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}
    texts: list[str] = []
    images: list[str] = []
    for part in message.content:
        if isinstance(part, ImagePart):
            encoded = _encode_image(part.path)
            if encoded:
                images.append(encoded)
        else:
            texts.append(part.text)
    out: dict[str, Any] = {"role": message.role, "content": "\n".join(texts)}
    if images:
        out["images"] = images
    return out


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
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        # Forward arbitrary Ollama-specific options (top_p, num_ctx, …) without re-mapping.
        for key, value in kwargs.items():
            options.setdefault(key, value)

        # Ollama takes images as a per-message list of base64 strings, not as
        # OpenAI-style content blocks. They attach to the last user message, the
        # one carrying the question and context.
        payload_messages: list[dict[str, Any]] = [_message_dict(m) for m in messages]
        if images:
            encoded = [_encode_image(i) for i in images]
            encoded = [e for e in encoded if e]
            if encoded:
                for msg in reversed(payload_messages):
                    if msg.get("role") == "user":
                        msg["images"] = [*msg.get("images", []), *encoded]
                        break

        payload: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
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
