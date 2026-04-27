"""OpenRouter LLM client: shape of requests and parsing of responses."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.llm.openrouter import OpenRouterClient
from src.llm.protocol import LLMClient, Message


def test_protocol_runtime_check() -> None:
    client = OpenRouterClient(api_key="sk-test")
    assert isinstance(client, LLMClient)


@respx.mock
async def test_chat_returns_assistant_text() -> None:
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "anthropic/claude-3.5-sonnet",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
    )

    client = OpenRouterClient(api_key="sk-test")
    response = await client.chat(
        messages=[Message(role="user", content="Hi")],
        model="anthropic/claude-3.5-sonnet",
        max_tokens=128,
    )

    assert route.called
    assert response.text == "Hello!"
    assert response.tokens_in == 10
    assert response.tokens_out == 5
    assert response.model == "anthropic/claude-3.5-sonnet"

    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer sk-test"
    assert b"Hi" in sent.content


@respx.mock
async def test_chat_raises_on_http_error() -> None:
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    client = OpenRouterClient(api_key="sk-test")
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat(
            messages=[Message(role="user", content="Hi")],
            model="anthropic/claude-3.5-sonnet",
        )
