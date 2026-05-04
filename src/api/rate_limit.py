"""Per-IP rate limiter for /answer (Phase 2.2).

Module-level Limiter instance imported by both the route decorator and
`create_app()` (where it gets registered as `app.state.limiter` and the
RateLimitExceeded exception handler is hooked up). Storage defaults to
in-memory — fine for a single replica; revisit if we ever scale out.

Reset state between tests with `limiter.reset()` (see tests/unit/test_rate_limit.py).
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# IP-keyed limiter. /answer is the LLM-spending endpoint; everything else
# stays unlimited for now (read-only retrieval is cheap). headers_enabled is
# left at the default False — slowapi's header injection requires every route
# to return `Response` directly (raises on Pydantic-model returns), and we
# return Answer models. Phase 3.2.1+ candidate: custom 429 handler that adds
# Retry-After without the global injection requirement.
limiter = Limiter(key_func=get_remote_address)
