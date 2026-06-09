"""FastAPI app factory."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import Response
from starlette.types import Scope

from src.api.auth import make_api_key_middleware
from src.api.bootstrap import _wire_generator_from_settings, _wire_retriever_from_settings
from src.api.deps import set_tracer
from src.api.middleware import request_context_middleware
from src.api.rate_limit import limiter
from src.api.routes import answer, dci, demo, figures, health, papers, query
from src.config.settings import load_settings
from src.observability.langfuse import make_langfuse_client
from src.observability.logging import configure_logging, get_logger
from src.observability.otel import configure_otel
from src.observability.sentry import configure_sentry


class _NoCacheStatic(StaticFiles):
    """StaticFiles that forces revalidation of the no-build SPA's assets.

    The frontend ships its source directly — index.html plus app/*.jsx
    transpiled in-browser and *.css, none content-hash-named. With only
    ETag / Last-Modified the browser applies heuristic caching (often
    hours) and after an edit or deploy serves a stale mix of old and new
    files (e.g. a new figures.jsx against an old shared.jsx). Setting
    `Cache-Control: no-cache` forces a revalidation on every load; the
    server still returns a cheap 304 when the file is unchanged. Page
    images use the separate /pages mount and keep normal caching.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(*, log_file: Path | None = Path("logs/api.log")) -> FastAPI:
    settings = load_settings()
    configure_logging(level=settings.log_level, env=settings.env, log_file=log_file)
    log = get_logger(__name__)

    sentry_on = configure_sentry()
    otel_on = configure_otel()
    generator_on = _wire_generator_from_settings(settings)
    # Env-gated like sentry/otel above: make_langfuse_client() returns a real
    # client only when RAG_LANGFUSE_* keys are set, else None (trace_query is
    # then a no-op). Without this wire the tracer is never registered and the
    # /answer trace is dead regardless of keys.
    langfuse_client = make_langfuse_client()
    set_tracer(langfuse_client)
    log.info(
        "api.startup",
        env=settings.env,
        log_level=settings.log_level,
        sentry=sentry_on,
        otel=otel_on,
        generator=generator_on,
        langfuse=langfuse_client is not None,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Wire the retriever during startup, before yielding. uvicorn awaits
        # the lifespan startup before it opens the listening socket
        # (Server.startup runs lifespan.startup() ahead of create_server), so
        # the container reports ready only once the corpus is wired.
        #
        # This must NOT be a fire-and-forget background task. Wiring loads the
        # in-process ~2.3 GB bge-m3 weights and scrolls the index — CPU-heavy.
        # On Cloud Run a deferred task is CPU-throttled to ~0 the instant the
        # container reports ready, so it never finishes and /query, /answer,
        # /figures 503 forever. Awaiting here keeps the work inside the
        # startup-cpu-boost window while the startup probe waits for the port.
        # `_wire_retriever_from_settings` swallows its own failures (returns
        # False), so a missing/empty corpus still boots — those routes 503,
        # same contract as before. Tests using TestClient(app) without `with`
        # skip lifespan and inject a retriever via dependency_overrides.
        retriever_on = await _wire_retriever_from_settings(settings)
        log.info("api.lifespan.startup", retriever=retriever_on)
        yield

    app = FastAPI(
        title="SpectraRAG",
        version="0.1.0",
        description="Multi-modal PDF RAG comparing text-pipeline vs visual retrieval.",
        lifespan=lifespan,
    )
    # slowapi needs:
    #  1. limiter on app.state (read by the @limiter.limit decorator on routes)
    #  2. a handler for RateLimitExceeded so it returns 429 instead of 500
    #  3. SlowAPIMiddleware — without it the rate check fires AFTER Depends
    #     resolution, so endpoint-level guards (e.g. the unset-retriever 503)
    #     short-circuit before the limiter counts the request and the bucket
    #     never fills. Middleware moves the check above the Depends chain.
    # The type-ignore is the standard slowapi workaround: Starlette types the
    # handler arg as Exception but slowapi narrows to RateLimitExceeded —
    # covariant in practice, mypy strict can't see across the inheritance.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)
    app.middleware("http")(request_context_middleware)
    # Auth runs OUTERMOST so unauthenticated requests get short-circuited
    # before request_context allocates an X-Request-ID or downstream code does
    # any work. Pass None when no key is configured — the middleware no-ops
    # and the endpoint-level guards take over.
    api_key = settings.public_api_key.get_secret_value() if settings.public_api_key else None
    app.middleware("http")(make_api_key_middleware(api_key))

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(dci.router)
    app.include_router(demo.router)
    app.include_router(answer.router)
    app.include_router(papers.router)
    app.include_router(figures.router)

    # Page PNGs served at /pages/<paper>/<paper>_pN.png. The browser pulls
    # these URLs into OpenRouter `image_url` content blocks so a vision-
    # capable model (gpt-4o, claude, qwen3-vl) sees the pixels directly. This
    # is the deploy-side equivalent of `Generator._collect_image_paths`'s
    # server-side attachment: same data, different transport. Mounted from
    # settings.pages_dir when set (defaults to None — no pages served).
    if settings.pages_dir is not None and settings.pages_dir.is_dir():
        app.mount("/pages", StaticFiles(directory=settings.pages_dir), name="pages")

    # Static frontend mounted LAST at "/" so it doesn't shadow API routes —
    # FastAPI matches explicit routes before mounted apps. `html=True` makes
    # GET / serve index.html (instead of a directory listing). When the web/
    # directory isn't present (e.g., a stripped runtime image) the mount
    # silently skips so the API still boots.
    #
    # `Cache-Control: no-cache` is set on HTML responses so browsers
    # revalidate on every reload (cheap 304 via ETag) instead of serving
    # stale HTML after a deploy. Without this, the FastAPI static handler's
    # default heuristic caching keeps old UIs visible for hours.
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.is_dir():
        app.mount("/", _NoCacheStatic(directory=web_dir, html=True), name="web")

    # Auto-instrumentation must run after routers are added so per-route
    # spans are named correctly. HTTPXClientInstrumentor is a singleton
    # and BaseInstrumentor.instrument() is internally idempotent, so a
    # repeat call is a no-op (logs a warning, no exception).
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    return app


_log_file: Path | None = None if os.getenv("RAG_ENV") == "prod" else Path("logs/api.log")
app = create_app(log_file=_log_file)
