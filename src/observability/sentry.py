"""Sentry SDK init. No-op when SENTRY_DSN is unset."""

from __future__ import annotations

import os

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from src.observability.logging import get_logger

_log = get_logger(__name__)
_configured = False


def configure_sentry() -> bool:
    """Initialise Sentry if `SENTRY_DSN` is set. Returns True if initialised, else False.

    Idempotent: subsequent calls are no-ops once Sentry is up. SDK reads the
    DSN, environment, and traces sample rate directly from env per its own
    conventions — they are NOT routed through `Settings` (see CLAUDE.md).
    """
    global _configured
    if _configured:
        return True
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    sample_rate_raw = os.environ.get("SENTRY_TRACES_SAMPLE_RATE")
    sample_rate = float(sample_rate_raw) if sample_rate_raw else 0.0
    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "local"),
        traces_sample_rate=sample_rate,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
        ],
        send_default_pii=False,
    )
    _configured = True
    _log.info("sentry.configured", environment=os.environ.get("SENTRY_ENVIRONMENT", "local"))
    return True
