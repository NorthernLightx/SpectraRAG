"""timed_event: emits a single log record with duration_ms on success or error."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.observability.logging import configure_logging, get_logger, timed_event


@pytest.fixture(autouse=True)
def _reset_logging() -> object:
    yield
    import logging

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def _read_records(log_file: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line]


def test_timed_event_emits_duration_ms_on_success(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"
    configure_logging(level="INFO", env="local", log_file=log_file)
    log = get_logger("test")

    with timed_event(log, "retrieve.done", query="x", top_k=3) as ctx:
        ctx["returned"] = 2

    [record] = _read_records(log_file)
    assert record["event"] == "retrieve.done"
    assert record["query"] == "x"
    assert record["top_k"] == 3
    assert record["returned"] == 2
    assert isinstance(record["duration_ms"], int)
    assert record["duration_ms"] >= 0
    assert record["level"] == "info"


def test_timed_event_emits_error_record_and_reraises(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"
    configure_logging(level="INFO", env="local", log_file=log_file)
    log = get_logger("test")

    with (
        pytest.raises(ValueError, match="boom"),
        timed_event(log, "ingest.done", paper_id="p1") as ctx,
    ):
        ctx["pages"] = 5
        raise ValueError("boom")

    [record] = _read_records(log_file)
    assert record["event"] == "ingest.done"
    assert record["paper_id"] == "p1"
    assert record["pages"] == 5
    assert record["level"] == "error"
    assert "duration_ms" in record
    assert "exception" in record  # structlog format_exc_info adds this


def test_timed_event_extra_fields_can_overlap_initial_fields(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"
    configure_logging(level="INFO", env="local", log_file=log_file)
    log = get_logger("test")

    with timed_event(log, "x", n=1) as ctx:
        ctx["n"] = 99  # overrides the initial value

    [record] = _read_records(log_file)
    assert record["n"] == 99
