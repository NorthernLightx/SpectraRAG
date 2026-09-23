"""configure_sentry: no-op without DSN; honours env when set."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import patch

import pytest
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

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


def test_configure_sentry_scrubs_the_visitors_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@example.ingest.sentry.io/1")
    with patch("src.observability.sentry.sentry_sdk.init") as init:
        configure_sentry()
    scrubber = init.call_args.kwargs["event_scrubber"]
    headers = {"x-openrouter-key": "sk-or-v1-visitor", "content-type": "application/json"}
    event: Any = {"request": {"headers": headers}}

    scrubber.scrub_event(event)

    assert headers["x-openrouter-key"] != "sk-or-v1-visitor"
    assert headers["content-type"] == "application/json"


class _CapturingTransport(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.sent = b""

    def capture_envelope(self, envelope: Envelope) -> None:
        self.sent += envelope.serialize()


def test_configure_sentry_keeps_a_key_in_a_local_variable_out_of_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@example.ingest.sentry.io/1")
    transport = _CapturingTransport()
    real_init = sentry_sdk.init
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: real_init(transport=transport, **kw))
    configure_sentry()
    # Built at runtime: events carry source lines, and a key never sits in source.
    secret = "-".join(["sk", "or", "v1", "visitor"])

    def call_provider(key: str) -> None:
        raise RuntimeError(f"rejected {len(key)}-char key")

    try:
        call_provider(secret)
    except RuntimeError:
        sentry_sdk.capture_exception()
    sentry_sdk.flush()
    sentry_sdk.get_client().close()

    assert b"rejected" in transport.sent
    assert secret.encode() not in transport.sent


def test_configure_sentry_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://abc@example.ingest.sentry.io/1")
    with patch("src.observability.sentry.sentry_sdk.init") as init:
        configure_sentry()
        configure_sentry()
        assert init.call_count == 1
