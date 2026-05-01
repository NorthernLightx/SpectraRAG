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
async def test_embed_skips_500_with_zero_vector() -> None:
    """Ollama bge-m3 occasionally emits NaN and returns 500 — skip-with-zero, not abort."""
    respx.post("http://localhost:11434/api/embeddings").mock(
        return_value=httpx.Response(500, json={"error": "json: unsupported value: NaN"})
    )
    embedder = OllamaBgeEmbedder(base_url="http://localhost:11434")
    [vec] = await embedder.embed_texts(["hi"])
    assert vec == [0.0] * embedder.dim


@respx.mock
async def test_embed_skips_nan_or_inf_in_payload() -> None:
    """Even on 200 OK, NaN/Inf in returned floats is replaced with a zero vector.

    Python's json module rejects NaN by default; Ollama emits it via Go's encoder.
    We test by feeding a raw bytes body that Python's json *will* parse (it allows
    NaN by default on read, only rejects on write).
    """
    respx.post("http://localhost:11434/api/embeddings").mock(
        return_value=httpx.Response(200, content=b'{"embedding":[NaN,0.1,0.2]}')
    )
    embedder = OllamaBgeEmbedder(base_url="http://localhost:11434")
    [vec] = await embedder.embed_texts(["hi"])
    assert vec == [0.0] * embedder.dim


@respx.mock
async def test_embed_propagates_other_http_errors() -> None:
    """4xx and non-500 5xx still raise (e.g., auth issues, bad requests)."""
    respx.post("http://localhost:11434/api/embeddings").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    embedder = OllamaBgeEmbedder(base_url="http://localhost:11434")
    with pytest.raises(httpx.HTTPStatusError):
        await embedder.embed_texts(["hi"])
