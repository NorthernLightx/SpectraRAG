"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

_PREFIXES = ("RAG_", "OTEL_", "SENTRY_")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip RAG_/OTEL_/SENTRY_ env vars so tests start from a clean slate."""
    for key in list(os.environ):
        if key.startswith(_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    yield
