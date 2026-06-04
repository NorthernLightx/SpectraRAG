"""X-API-Key middleware for /answer + /query.

Compares the inbound `X-API-Key` header against the configured shared secret
in constant time (hmac.compare_digest). When `RAG_PUBLIC_API_KEY` is unset,
the middleware is a no-op — preserves the dev/single-user default where
auth gets in the way of curl-and-iterate.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

# Routes that must always be reachable, even when auth is otherwise required:
# liveness probes, the auto-generated OpenAPI schema, and the interactive docs.
# Adding a route name here is a deliberate "this is safe to call without a
# key" choice; default-deny everything else.
_EXEMPT_PATHS = frozenset({"/", "/health", "/docs", "/redoc", "/openapi.json"})


def make_api_key_middleware(
    api_key: str | None,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Return an ASGI HTTP middleware that 401s requests without a valid key.

    Returning a no-op middleware when `api_key` is None keeps the dev workflow
    friction-free without a separate `if settings.public_api_key:` guard at the
    call site.
    """

    async def _middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Gate on scope["path"] (the routed path), not request.url.path:
        # request.url is rebuilt from the Host header and can be poisoned by a
        # malformed one (CVE-2026-48710), making url.path diverge from the path
        # routing actually dispatched on. scope["path"] is what the router uses.
        if api_key is None or request.scope["path"] in _EXEMPT_PATHS:
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        # hmac.compare_digest avoids leaking the key length / prefix via timing.
        if not hmac.compare_digest(provided, api_key):
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        return await call_next(request)

    return _middleware
