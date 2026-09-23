"""FastAPI app factory."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.auth import make_api_key_middleware
from src.api.bootstrap import (
    _warm_retriever,
    _wire_generator_from_settings,
    _wire_retriever_from_settings,
)
from src.api.deps import set_tracer
from src.api.middleware import request_context_middleware
from src.api.rate_limit import limiter
from src.api.routes import answer, context, dci, figures, health, ingest, papers, query
from src.config.settings import load_settings
from src.observability.langfuse import make_langfuse_client
from src.observability.logging import configure_logging, get_logger
from src.observability.otel import configure_otel
from src.observability.sentry import configure_sentry


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
        # in-process ~2.3 GB bge-m3 weights and scrolls the index, so it is
        # CPU-heavy. On Cloud Run a deferred task is CPU-throttled to ~0 the
        # instant the container reports ready, so it never finishes and /query,
        # /answer, /figures 503 forever. Awaiting here keeps the work inside the
        # startup-cpu-boost window while the startup probe waits for the port.
        # `_wire_retriever_from_settings` swallows its own failures (returns
        # False), so a missing/empty corpus still boots; those routes 503,
        # same contract as before. Tests using TestClient(app) without `with`
        # skip lifespan and inject a retriever via dependency_overrides.
        retriever_on = await _wire_retriever_from_settings(settings)
        await _warm_retriever()
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
    #  3. SlowAPIMiddleware. Without it the rate check fires AFTER Depends
    #     resolution, so endpoint-level guards (the unset-retriever 503, for
    #     example) short-circuit before the limiter counts the request and the
    #     bucket never fills. Middleware moves the check above the Depends chain.
    # The type-ignore is the standard slowapi workaround: Starlette types the
    # handler arg as Exception but slowapi narrows to RateLimitExceeded. It is
    # covariant in practice; mypy strict can't see across the inheritance.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)
    app.middleware("http")(request_context_middleware)
    # Auth runs OUTERMOST so unauthenticated requests get short-circuited
    # before request_context allocates an X-Request-ID or downstream code does
    # any work. Pass None when no key is configured, and the middleware no-ops
    # and the endpoint-level guards take over.
    api_key = settings.public_api_key.get_secret_value() if settings.public_api_key else None
    app.middleware("http")(make_api_key_middleware(api_key))

    # CORS only when the frontend is served from a separate origin (decoupled
    # deploy): origins from RAG_CORS_ORIGINS (comma-separated), empty = same-
    # origin/no CORS. Added last so it sits outermost and answers the preflight
    # OPTIONS before the auth middleware can short-circuit it.
    cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(dci.router)
    app.include_router(answer.router)
    app.include_router(context.router)
    app.include_router(papers.router)
    app.include_router(figures.router)

    # Page PNGs served at /pages/<paper>/<paper>_pN.png. The browser pulls
    # these URLs into OpenRouter `image_url` content blocks so a vision-
    # capable model (gpt-4o, claude, qwen3-vl) sees the pixels directly. The
    # server-side Generator attaches the same files from disk (ADR 0033). Mounted from
    # settings.pages_dir when set (defaults to None, so no pages are served).
    if settings.pages_dir is not None and settings.pages_dir.is_dir():
        app.mount("/pages", StaticFiles(directory=settings.pages_dir), name="pages")

    # A browser opening the root lands on the interactive API docs.
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    # Auto-instrumentation must run after routers are added so per-route
    # spans are named correctly. HTTPXClientInstrumentor is a singleton
    # and BaseInstrumentor.instrument() is internally idempotent, so a
    # repeat call is a no-op (logs a warning, no exception).
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    return app


_log_file: Path | None = None if os.getenv("RAG_ENV") == "prod" else Path("logs/api.log")
app = create_app(log_file=_log_file)
