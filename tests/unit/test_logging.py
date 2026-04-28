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
