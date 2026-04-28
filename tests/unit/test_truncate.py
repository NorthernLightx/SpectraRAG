"""truncate_long_strings: shortens long string values, leaves others alone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.observability.logging import (
    DEFAULT_MAX_STRING_LEN,
    configure_logging,
    get_logger,
    truncate_long_strings,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> object:
    yield
    import logging

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def test_processor_truncates_long_string() -> None:
    proc = truncate_long_strings(max_len=10)
    out = proc(None, "info", {"event": "x", "text": "abcdefghijKLMNOP"})
    assert out["text"].startswith("abcdefghij")
    assert "[+6 chars]" in out["text"]


def test_processor_passes_short_string_unchanged() -> None:
    proc = truncate_long_strings(max_len=10)
    out = proc(None, "info", {"event": "x", "text": "short"})
    assert out["text"] == "short"


def test_processor_ignores_non_string_values() -> None:
    proc = truncate_long_strings(max_len=2)
    out = proc(None, "info", {"event": "x", "n": 12345, "items": ["a", "b", "c"]})
    assert out["n"] == 12345
    assert out["items"] == ["a", "b", "c"]


def test_truncation_applies_in_real_log_pipeline(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"
    configure_logging(level="INFO", env="local", log_file=log_file)
    log = get_logger("test")

    long_query = "x" * (DEFAULT_MAX_STRING_LEN + 50)
    log.info("retrieve.done", query=long_query, top_k=3)

    [record] = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(record["query"]) < len(long_query)
    assert "[+50 chars]" in record["query"]
    assert record["top_k"] == 3
