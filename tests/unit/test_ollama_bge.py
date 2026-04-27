"""OllamaBgeEmbedder: BGE-M3 served by a local Ollama instance."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.embeddings.protocol import Embedder


def test_protocol_runtime_check() -> None:
    embedder = OllamaBgeEmbedder(base_url="http://localhost:11434")
    assert isinstance(embedder, Embedder)


@respx.mock
async def test_embed_texts_calls_ollama_per_text() -> None:
    route = respx.post("http://localhost:11434/api/embeddings").mock(
        side_effect=[
            httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]}),
            httpx.Response(200, json={"embedding": [0.4, 0.5, 0.6]}),
        ]
    )

    embedder = OllamaBgeEmbedder(base_url="http://localhost:11434", model="bge-m3")
    vectors = await embedder.embed_texts(["hello", "world"])

    assert route.call_count == 2
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    sent = route.calls[0].request
    assert b'"model":"bge-m3"' in sent.content.replace(b" ", b"")
    assert b"hello" in sent.content


@respx.mock
async def test_embed_empty_input_returns_empty_list() -> None:
    embedder = OllamaBgeEmbedder(base_url="http://localhost:11434")
    assert await embedder.embed_texts([]) == []


@respx.mock
async def test_embed_propagates_http_errors() -> None:
    respx.post("http://localhost:11434/api/embeddings").mock(
        return_value=httpx.Response(500, json={"error": "model not loaded"})
    )
    embedder = OllamaBgeEmbedder(base_url="http://localhost:11434")
    with pytest.raises(httpx.HTTPStatusError):
        await embedder.embed_texts(["hi"])
