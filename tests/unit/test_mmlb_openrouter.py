"""OpenRouter provider path for the MMLongBench-Doc QA harness.

Covers the three pieces added so the QA protocol can run on OpenRouter when
Ollama's daily cloud quota is exhausted:
  - scripts/experiments/_openrouter_client.resolve_openrouter_key: env-first,
    then a .env fallback; SystemExit when neither has a key.
  - run_mmlb_qa._chat_vision_openrouter: builds OpenAI-compat image_url blocks
    on the user message and returns the (answer, tokens_in, tokens_out) contract.
  - score_mmlb_qa._extract_one: a persistent 429 (already retried inside the
    client) is a SOFT failure (_EXTRACT_FAILED), not an abort.

OpenRouter HTTP is respx-mocked, so no network and no key are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from scripts.experiments._openrouter_client import _load_key_from_dotenv, resolve_openrouter_key
from scripts.experiments.run_mmlb_qa import _chat_vision_openrouter
from scripts.experiments.score_mmlb_qa import _EXTRACT_FAILED, _extract_one
from src.llm.openrouter import OpenRouterClient

_OR_URL = "https://openrouter.ai/api/v1/chat/completions"


# --------------------------------------------------------------------------
# Key resolution
# --------------------------------------------------------------------------
def test_key_from_env_takes_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_OPENROUTER_API_KEY", "sk-or-env")
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_OPENROUTER_API_KEY=sk-or-file\n", encoding="utf-8")
    assert resolve_openrouter_key(env_file) == "sk-or-env"


def test_key_falls_back_to_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RAG_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nOTHER=x\nRAG_OPENROUTER_API_KEY="sk-or-quoted"\n', encoding="utf-8"
    )
    assert resolve_openrouter_key(env_file) == "sk-or-quoted"


def test_bare_openrouter_var_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RAG_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-bare")
    assert resolve_openrouter_key(tmp_path / "missing.env") == "sk-or-bare"


def test_missing_key_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RAG_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        resolve_openrouter_key(tmp_path / "missing.env")


def test_dotenv_reader_ignores_unrelated_lines(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("\n# header\nFOO=bar\nNOEQUALS\n", encoding="utf-8")
    assert _load_key_from_dotenv(env_file) is None


# --------------------------------------------------------------------------
# Vision generation over OpenRouter
# --------------------------------------------------------------------------
@respx.mock
async def test_chat_vision_openrouter_builds_image_blocks(tmp_path: Path) -> None:
    """The page PNG path is encoded into an image_url content block on the user
    message, and the (answer, tokens_in, tokens_out) contract is returned."""
    png = tmp_path / "p1.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")  # bytes are base64'd; content is irrelevant here
    route = respx.post(_OR_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "google/gemma-4-31b-it:free",
                "choices": [{"message": {"role": "assistant", "content": "Berlin"}}],
                "usage": {"prompt_tokens": 273, "completion_tokens": 1},
            },
        )
    )
    client = OpenRouterClient(api_key="sk-test")
    answer, tin, tout = await _chat_vision_openrouter(
        client,
        "google/gemma-4-31b-it:free",
        "system prompt",
        "user question",
        [png],
        temperature=0.0,
        max_tokens=512,
    )
    assert (answer, tin, tout) == ("Berlin", 273, 1)

    body = json.loads(route.calls.last.request.content)
    messages = body["messages"]
    assert messages[0] == {"role": "system", "content": "system prompt"}
    # The user message content became a block list: one text block + one image_url.
    user_blocks = messages[1]["content"]
    assert user_blocks[0] == {"type": "text", "text": "user question"}
    assert user_blocks[1]["type"] == "image_url"
    assert user_blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 512


# --------------------------------------------------------------------------
# Extraction soft-fail on a persistent 429
# --------------------------------------------------------------------------
@respx.mock
async def test_extract_one_soft_fails_on_persistent_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 that survives the client's own retries must NOT abort the run: it
    returns _EXTRACT_FAILED so the caller leaves the query uncached for a resume.

    asyncio.sleep is stubbed to a no-op so neither the client's tenacity backoff
    (min=2s, exp to 60s, 6 attempts) nor _extract_one's own backoff makes the
    test sleep ~60s; max_attempts=1 keeps _extract_one's loop to a single pass."""

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    respx.post(_OR_URL).mock(return_value=httpx.Response(429, json={"error": "rate limited"}))
    client = OpenRouterClient(api_key="sk-test")
    pred = await _extract_one(client, "deepseek/deepseek-v4-flash:free", "q", "a", max_attempts=1)
    assert pred == _EXTRACT_FAILED


@respx.mock
async def test_extract_one_raises_on_auth_error() -> None:
    """A 401 is a real config error (bad key) — it must propagate, not be
    silently swallowed across a 149-call run."""
    respx.post(_OR_URL).mock(return_value=httpx.Response(401, json={"error": "no auth"}))
    client = OpenRouterClient(api_key="sk-bad")
    with pytest.raises(httpx.HTTPStatusError):
        await _extract_one(client, "deepseek/deepseek-v4-flash:free", "q", "a", max_attempts=1)
