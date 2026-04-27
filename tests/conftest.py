"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolate_rag_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip RAG_* env vars so tests start from a clean slate."""
    for key in list(os.environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key, raising=False)
    yield
