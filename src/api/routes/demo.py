"""POST /demo/chat: keyless generation through the server's caged demo key.

ADR 0027. BYOK browser-direct generation stays the primary path; this route
exists so a visitor without an OpenRouter key still gets a generated answer.
The cage has three independent layers:

- the model comes from a server-side chain of ":free" ids — the client cannot
  name one,
- every upstream request pins `provider.max_price` to 0, so OpenRouter itself
  refuses to route to any endpoint that would charge,
- the key carries a provider-side credit limit, bounding a leak.

Abuse therefore can't spend money; it can only exhaust the account's shared
daily free-model quota. The per-IP limit plus the global daily counter keep
demo traffic well under that ceiling.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.deps import get_settings
from src.api.rate_limit import limiter
from src.config.settings import Settings
from src.observability.logging import get_logger

_log = get_logger(__name__)
router = APIRouter()

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Generous read timeout: free endpoints queue under load, and the response
# streams token-by-token once it starts.
_TIMEOUT = httpx.Timeout(90.0, connect=15.0)
# Mirrors the BYOK client's cap (web/app/api.js streamChat).
_MAX_TOKENS = 800


def _make_client() -> httpx.AsyncClient:
    """Factory hook so tests can swap in a MockTransport-backed client."""
    return httpx.AsyncClient(timeout=_TIMEOUT)


class DemoChatRequest(BaseModel):
    """OpenAI-format chat messages (string content or content-block lists).

    There is deliberately no `model` field: the demo chain is server-chosen,
    and pydantic drops any extra keys a client sends.
    """

    messages: list[dict[str, Any]] = Field(min_length=1, max_length=64)


class _DemoQuota:
    """In-process per-UTC-day counter for the global demo cap.

    Single-replica state (the deploy runs max-instances=1) that resets on
    restart — losing the count on a redeploy just re-opens the demo early,
    which is the harmless direction. Reset in tests via `_DemoQuota.day = ""`.
    """

    day: str = ""
    used: int = 0

    @classmethod
    def take(cls, cap: int) -> bool:
        today = dt.datetime.now(dt.UTC).date().isoformat()
        if cls.day != today:
            cls.day, cls.used = today, 0
        if cls.used >= cap:
            return False
        cls.used += 1
        return True


@router.post("/demo/chat")
@limiter.limit("30/hour")
async def demo_chat(
    request: Request,
    payload: DemoChatRequest,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    key = settings.demo_openrouter_key.get_secret_value() if settings.demo_openrouter_key else None
    # Drop anything that isn't a ":free" id — a misconfigured RAG_DEMO_MODELS
    # must fail closed, not route the demo key to a paid model.
    models = [m.strip() for m in settings.demo_models.split(",") if m.strip().endswith(":free")]
    if not key or not models or settings.demo_daily_cap <= 0:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Keyless demo generation is not configured on this server.",
        )
    if not _DemoQuota.take(settings.demo_daily_cap):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "demo_quota_exhausted")

    body: dict[str, Any] = {
        "messages": payload.messages,
        "temperature": 0.2,
        "max_tokens": _MAX_TOKENS,
        "stream": True,
        "usage": {"include": True},
        # The cage's hard floor: even if a paid id ever slipped past the
        # ":free" filter above, OpenRouter refuses to route a request whose
        # price cap is zero on every axis (images included — we send pages).
        "provider": {"max_price": {"prompt": 0, "completion": 0, "request": 0, "image": 0}},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    client = _make_client()
    upstream: httpx.Response | None = None
    used_model = ""
    for model in models:
        try:
            candidate = await client.send(
                client.build_request(
                    "POST", _OPENROUTER_URL, json={**body, "model": model}, headers=headers
                ),
                stream=True,
            )
        except httpx.HTTPError:
            _log.info("demo.chat.fallback", model=model, reason="transport_error")
            continue
        if candidate.status_code == 200:
            upstream, used_model = candidate, model
            break
        await candidate.aclose()
        _log.info("demo.chat.fallback", model=model, reason=candidate.status_code)
    if upstream is None:
        await client.aclose()
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "The free demo models are all unavailable right now. Retry in a minute, "
            "or add your own OpenRouter key (top-right).",
        )

    final = upstream

    async def passthrough() -> AsyncIterator[bytes]:
        try:
            async for chunk in final.aiter_bytes():
                yield chunk
        finally:
            await final.aclose()
            await client.aclose()

    # Log the model and quota state, never the key or the message content.
    _log.info(
        "demo.chat",
        model=used_model,
        used_today=_DemoQuota.used,
        cap=settings.demo_daily_cap,
    )
    return StreamingResponse(passthrough(), media_type="text/event-stream")
