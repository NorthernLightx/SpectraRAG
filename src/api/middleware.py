"""HTTP middleware. Currently: per-request structlog contextvar binding."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

import structlog
from fastapi import Request, Response

REQUEST_ID_HEADER = "X-Request-ID"


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind request_id + method + path to structlog contextvars for the request lifetime.

    Honours an inbound `X-Request-ID` header so callers (load balancers, gateways,
    other services) can correlate across systems. Always echoes the header back.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers[REQUEST_ID_HEADER] = request_id
    return response
