"""LLMClient Protocol: the only chat interface upstream code depends on."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image file sent inline, in the message where it appears."""

    type: Literal["image"] = "image"
    path: Path


ContentPart = TextPart | ImagePart


class Message(BaseModel):
    """A single chat message. A list of parts keeps text and images interleaved,
    so a label placed before an image stays attached to it."""

    role: Role
    content: str | list[ContentPart]


def image_mime(path: Path) -> str:
    """MIME type for an inline image, by extension. Page renders are PNG;
    benchmark corpora ship JPEG."""
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


class ChatResponse(BaseModel):
    """Result of a chat completion call."""

    text: str
    model: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    raw: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat protocol. Embedding is the Embedder's job, kept separate by design.

    `images` is an optional list of PNG paths attached to the LAST user message
    when the underlying provider supports vision (currently OpenRouter via the
    OpenAI-compat content-block schema). Implementations that don't support
    vision should ignore the parameter or raise. Defaults to None for back-compat
    with the text-only path.
    """

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        images: list[Path] | None = None,
        **kwargs: Any,
    ) -> ChatResponse: ...
