"""FastAPI app factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.auth import make_api_key_middleware
from src.api.deps import set_generator
from src.api.middleware import request_context_middleware
from src.api.rate_limit import limiter
from src.api.routes import answer, health, query
from src.config.settings import Settings, load_settings
from src.llm.openrouter import OpenRouterClient
from src.observability.logging import configure_logging, get_logger
from src.observability.otel import configure_otel
from src.observability.sentry import configure_sentry
from src.prompts.loader import load_prompt_by_name
from src.rag.generate import Generator


def _wire_generator_from_settings(settings: Settings) -> bool:
    """Build OpenRouterClient + Generator and register, when the API key is configured.

    Returns True if a Generator was wired, False if the key is unset (no-op).
    Retriever wiring is intentionally deferred — it needs a populated Qdrant
    collection. Eval scripts and tests wire their own retrievers via
    ``set_retriever``.
    """
    if settings.openrouter_api_key is None:
        return False
    client = OpenRouterClient(api_key=settings.openrouter_api_key.get_secret_value())
    set_generator(
        Generator(
            llm=client,
            prompt=load_prompt_by_name("answer"),
            model=settings.default_chat_model,
            temperature=settings.temperature,
            max_context_tokens=settings.max_context_tokens,
            # When pages_dir is set the Generator attaches the rendered page PNG
            # for any visual RetrievalResult so a vision-capable default_chat_model
            # can read images directly. None = text-only behaviour (back-compat).
            pages_dir=settings.pages_dir,
        )
    )
    return True


def create_app(*, log_file: Path | None = Path("logs/api.log")) -> FastAPI:
    settings = load_settings()
    configure_logging(level=settings.log_level, env=settings.env, log_file=log_file)
    log = get_logger(__name__)

    sentry_on = configure_sentry()
    otel_on = configure_otel()
    generator_on = _wire_generator_from_settings(settings)
    log.info(
        "api.startup",
        env=settings.env,
        log_level=settings.log_level,
        sentry=sentry_on,
        otel=otel_on,
        generator=generator_on,
    )

    app = FastAPI(
        title="Multi-modal Paper RAG",
        version="0.1.0",
        description="RAG over scientific papers comparing pipeline vs visual retrieval.",
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

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "Multi-modal Paper RAG", "docs": "/docs"}

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(answer.router)

    # Auto-instrumentation must run after routers are added so per-route
    # spans are named correctly. HTTPXClientInstrumentor is a singleton
    # and BaseInstrumentor.instrument() is internally idempotent, so a
    # repeat call is a no-op (logs a warning, no exception).
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    return app


_log_file: Path | None = None if os.getenv("RAG_ENV") == "prod" else Path("logs/api.log")
app = create_app(log_file=_log_file)
