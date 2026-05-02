"""Logging configuration: file sink emits JSON, stdout emits human-readable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.observability.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _restore_logging_after_test() -> object:
    """Reset to defaults after each test so we don't pollute later tests."""
    yield
    import logging

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def test_log_file_emits_json_with_event_and_level(tmp_path: Path) -> None:
    log_file = tmp_path / "rag.log"
    configure_logging(level="INFO", env="local", log_file=log_file)

    log = get_logger("test")
    log.info("hello", paper_id="p1", chunks=5)

    contents = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])
    assert record["event"] == "hello"
    assert record["paper_id"] == "p1"
    assert record["chunks"] == 5
    assert record["level"] == "info"
    assert "timestamp" in record


def test_log_level_filters_below_threshold(tmp_path: Path) -> None:
    log_file = tmp_path / "rag.log"
    configure_logging(level="WARNING", env="local", log_file=log_file)

    log = get_logger("test")
    log.info("ignored")
    log.warning("kept")

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert "ignored" not in events
    assert "kept" in events


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    log_file = tmp_path / "rag.log"
    configure_logging(level="INFO", env="local", log_file=log_file)
    configure_logging(level="INFO", env="local", log_file=log_file)

    log = get_logger("test")
    log.info("once")

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    # One configure call replaces handlers, so we should see exactly one record.
    assert len(lines) == 1


def test_no_file_sink_when_log_file_is_none(tmp_path: Path) -> None:
    configure_logging(level="INFO", env="local", log_file=None)
    log = get_logger("test")
    log.info("stdout-only")
    # The log file we didn't create must not exist
    assert not (tmp_path / "rag.log").exists()


def test_stdout_handler_handles_non_cp1252_chars_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: bge-m3 NaN warnings include chunk-text snippets that may
    contain CJK / fullwidth characters (e.g. U+FF08 fullwidth left paren).
    Windows stdout defaults to cp1252 + strict, which fails to encode these
    — Python logging swallows the error via Handler.handleError so the
    process keeps going, but the offending log line is *truncated* at the
    first un-encodable char and noisy `--- Logging error ---` tracebacks
    flood stderr. configure_logging must reconfigure stdout's error policy
    so the full message reaches the stream."""
    import io
    import sys

    raw = io.BytesIO()
    cp1252_strict = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", cp1252_strict)

    configure_logging(level="WARNING", env="local")
    log = get_logger("test")
    log.warning(
        "embed.skip_500",
        text_head="prefix （fullwidth） marker_after_bad_char",  # noqa: RUF001
    )

    decoded = raw.getvalue().decode("cp1252", errors="replace")
    # Without the fix: cp1252 encode crashes mid-message; "marker_after_bad_char"
    # never reaches the buffer because handleError swallows after the partial write.
    assert "embed.skip_500" in decoded
    assert "marker_after_bad_char" in decoded
