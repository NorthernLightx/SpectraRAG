"""Ollama chat client: request shape and response parsing."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.llm.ollama_chat import OllamaChatClient
from src.llm.protocol import LLMClient, Message


def test_protocol_runtime_check() -> None:
    client = OllamaChatClient()
    assert isinstance(client, LLMClient)


@respx.mock
async def test_chat_returns_assistant_text() -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen2.5:7b",
                "created_at": "2026-04-29T00:00:00Z",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
                "prompt_eval_count": 11,
                "eval_count": 4,
            },
        )
    )

    client = OllamaChatClient()
    response = await client.chat(
        messages=[Message(role="user", content="Hi")],
        model="qwen2.5:7b",
        temperature=0.0,
        max_tokens=128,
    )

    assert route.called
    assert response.text == "Hello!"
    assert response.tokens_in == 11
    assert response.tokens_out == 4
    assert response.model == "qwen2.5:7b"

    sent = route.calls.last.request
    body = sent.content
    assert b'"stream": false' in body or b'"stream":false' in body
    assert b'"num_predict": 128' in body or b'"num_predict":128' in body
    assert b'"temperature": 0.0' in body or b'"temperature":0.0' in body
    assert b"Hi" in body


@respx.mock
async def test_chat_handles_missing_usage_fields() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen2.5:7b",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
            },
        )
    )

    client = OllamaChatClient()
    response = await client.chat(
        messages=[Message(role="user", content="Hi")],
        model="qwen2.5:7b",
    )
    assert response.text == "ok"
    assert response.tokens_in == 0
    assert response.tokens_out == 0


@respx.mock
async def test_chat_forwards_num_ctx_when_set() -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen2.5:7b",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
            },
        )
    )
    client = OllamaChatClient(num_ctx=8192)
    await client.chat(messages=[Message(role="user", content="Hi")], model="qwen2.5:7b")

    body = route.calls.last.request.content
    assert b'"num_ctx": 8192' in body or b'"num_ctx":8192' in body


@respx.mock
async def test_chat_omits_num_ctx_by_default() -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen2.5:7b",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
            },
        )
    )
    client = OllamaChatClient()
    await client.chat(messages=[Message(role="user", content="Hi")], model="qwen2.5:7b")

    body = route.calls.last.request.content
    assert b"num_ctx" not in body


@respx.mock
async def test_chat_raises_on_http_error() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    client = OllamaChatClient()
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat(
            messages=[Message(role="user", content="Hi")],
            model="qwen2.5:7b",
        )
