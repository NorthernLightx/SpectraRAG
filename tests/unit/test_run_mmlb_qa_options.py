"""Ollama options passthrough in the MMLongBench-Doc QA generation harness
(scripts/experiments/run_mmlb_qa.py:_chat_vision).

Asserts the /api/chat payload shape: num_gpu / num_ctx are injected into
options ONLY when set, and are absent by default (omit-when-None, the same
convention as src/llm/ollama_chat.py). The httpx POST is respx-mocked, so no
Ollama server and no GPU model is touched — this runs CPU-only in CI.
"""

from __future__ import annotations

import json

import httpx
import respx

from scripts.experiments.run_mmlb_qa import _chat_vision

_OLLAMA_URL = "http://localhost:11434"
_OK_RESPONSE = httpx.Response(
    200,
    json={
        "model": "qwen2.5vl:7b",
        "message": {"role": "assistant", "content": "42"},
        "done": True,
        "prompt_eval_count": 1300,
        "eval_count": 3,
    },
)


async def _call(**chat_kwargs: object) -> dict[str, object]:
    """Run one _chat_vision call through a respx-mocked POST; return the parsed
    request body the harness actually sent."""
    route = respx.post(f"{_OLLAMA_URL}/api/chat").mock(return_value=_OK_RESPONSE)
    async with httpx.AsyncClient() as client:
        answer, tin, tout = await _chat_vision(
            client,
            _OLLAMA_URL,
            "qwen2.5vl:7b",
            "system",
            "user",
            ["<b64-png>"],
            temperature=0.0,
            max_tokens=512,
            **chat_kwargs,  # type: ignore[arg-type]
        )
    assert route.called
    assert (answer, tin, tout) == ("42", 1300, 3)
    body = json.loads(route.calls.last.request.content)
    assert isinstance(body, dict)
    return body


@respx.mock
async def test_options_omitted_by_default() -> None:
    """No --num-gpu / --num-ctx: options carry only the existing temperature +
    num_predict keys (current behavior is preserved byte-for-byte)."""
    body = await _call()
    options = body["options"]
    assert isinstance(options, dict)
    assert "num_gpu" not in options
    assert "num_ctx" not in options
    assert options == {"temperature": 0.0, "num_predict": 512}


@respx.mock
async def test_num_gpu_present_only_when_set() -> None:
    body = await _call(num_gpu=99)
    options = body["options"]
    assert isinstance(options, dict)
    assert options["num_gpu"] == 99
    assert "num_ctx" not in options  # the other flag stays absent when only one is set


@respx.mock
async def test_num_ctx_present_only_when_set() -> None:
    body = await _call(num_ctx=8192)
    options = body["options"]
    assert isinstance(options, dict)
    assert options["num_ctx"] == 8192
    assert "num_gpu" not in options


@respx.mock
async def test_both_options_present_when_both_set() -> None:
    body = await _call(num_gpu=99, num_ctx=4096)
    options = body["options"]
    assert isinstance(options, dict)
    # Both injected alongside the existing keys; nothing dropped.
    assert options == {
        "temperature": 0.0,
        "num_predict": 512,
        "num_gpu": 99,
        "num_ctx": 4096,
    }
    # Images still ride on the user message (not displaced by the options change).
    messages = body["messages"]
    assert isinstance(messages, list)
    assert messages[1]["images"] == ["<b64-png>"]


@respx.mock
async def test_zero_num_gpu_is_distinct_from_unset() -> None:
    """num_gpu=0 (force CPU) is a real value, not the same as omitting the flag:
    it must appear in the payload. Guards against a falsy `if num_gpu:` regression."""
    body = await _call(num_gpu=0)
    options = body["options"]
    assert isinstance(options, dict)
    assert options["num_gpu"] == 0
