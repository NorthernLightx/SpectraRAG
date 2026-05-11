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


@respx.mock
async def test_chat_retries_on_429_then_succeeds() -> None:
    """Free-tier OpenRouter models hit 429 under burst eval load (verified
    empirically on run b30k0s5pu). The retry decorator catches 429 with
    exponential backoff up to 6 attempts so the eval survives rate windows."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limit"}),
            httpx.Response(429, json={"error": "rate limit"}),
            httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "nvidia/nemotron-3-super-120b-a12b:free",
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
        ]
    )
    # Patch tenacity's sleep so the test runs fast (no real backoff wait).
    import tenacity

    original = tenacity.nap.sleep
    tenacity.nap.sleep = lambda _: None  # type: ignore[assignment]
    try:
        client = OpenRouterClient(api_key="sk-test")
        resp = await client.chat(
            messages=[Message(role="user", content="ping")],
            model="nvidia/nemotron-3-super-120b-a12b:free",
        )
    finally:
        tenacity.nap.sleep = original  # type: ignore[assignment]
    assert resp.text == "OK"
    assert route.call_count == 3


@respx.mock
async def test_chat_does_not_retry_on_401() -> None:
    """4xx other than 429 is non-retryable (auth errors won't get better)."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    client = OpenRouterClient(api_key="sk-bad")
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat(
            messages=[Message(role="user", content="ping")],
            model="any/model",
        )
    assert route.call_count == 1  # no retries
