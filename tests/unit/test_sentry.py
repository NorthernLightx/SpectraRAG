"""configure_sentry: no-op without DSN; honours env when set."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

import src.observability.sentry as sentry_mod
from src.observability.sentry import configure_sentry


@pytest.fixture(autouse=True)
def _reset_sentry_module() -> None:
    """Reload the module so the `_configured` flag is fresh per-test."""
    importlib.reload(sentry_mod)


def test_configure_sentry_noop_without_dsn() -> None:
    with patch("src.observability.sentry.sentry_sdk.init") as init:
        configure_sentry()
        init.assert_not_called()


def test_configure_sentry_initialises_with_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@example.ingest.sentry.io/1")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "test")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")
    with patch("src.observability.sentry.sentry_sdk.init") as init:
        configure_sentry()
        init.assert_called_once()
        kwargs = init.call_args.kwargs
        assert kwargs["dsn"] == "https://abc@example.ingest.sentry.io/1"
        assert kwargs["environment"] == "test"
        assert kwargs["traces_sample_rate"] == 0.25


def test_configure_sentry_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@example.ingest.sentry.io/1")
    with patch("src.observability.sentry.sentry_sdk.init") as init:
        configure_sentry()
        configure_sentry()
        assert init.call_count == 1
