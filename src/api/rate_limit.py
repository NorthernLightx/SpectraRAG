"""Per-client rate limiter for /answer and /query.

Module-level Limiter instance imported by both the route decorator and
`create_app()` (where it gets registered as `app.state.limiter` and the
RateLimitExceeded exception handler is hooked up). Storage defaults to
in-memory, fine for a single replica; revisit if we ever scale out.

Reset state between tests with `limiter.reset()` (see tests/unit/test_rate_limit.py).
"""

from __future__ import annotations

import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def client_key(request: Request) -> str:
    """The address one visitor's requests share.

    On Cloud Run (`K_SERVICE` is set by the platform) the socket peer is
    Google's front end for every visitor, so keying on it puts everyone in one
    bucket. The front end appends the address it received the request from to
    `X-Forwarded-For`; that rightmost entry is the one a client cannot forge,
    while anything to its left is whatever the client sent. Off Cloud Run the
    header is untrusted and the socket peer is the client.
    """
    if os.environ.get("K_SERVICE"):
        hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",")]
        hops = [h for h in hops if h]
        if hops:
            return hops[-1]
    return get_remote_address(request)


# headers_enabled is left at the default False because slowapi's header
# injection requires every route to return `Response` directly (raises on
# Pydantic-model returns), and we return Answer models.
limiter = Limiter(key_func=client_key)
