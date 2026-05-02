"""Structured logging via structlog. Pretty stdout in dev, JSON file always.

Call `configure_logging()` once at process startup (FastAPI `create_app`,
or at the top of any CLI script). Then `get_logger(__name__)` anywhere.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger

DEFAULT_MAX_STRING_LEN = 500
"""Cap on a single string field's length before logging. Coarse PII guard;
real PII redaction lives in Phase 4 (see PROJECT.md §5)."""

ProcessorFn = Callable[[Any, str, dict[str, Any]], dict[str, Any]]


def truncate_long_strings(max_len: int = DEFAULT_MAX_STRING_LEN) -> ProcessorFn:
    """structlog processor: shorten any string value > max_len with a count suffix.

    This is intentionally blunt — it caps the size of the worst offenders
    (full chunk text, full answers, raw paper bodies) before they hit disk
    or stdout. Field-name-aware redaction is a Phase 4 concern.
    """

    def _processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        for key, value in list(event_dict.items()):
            if isinstance(value, str) and len(value) > max_len:
                event_dict[key] = value[:max_len] + f"...[+{len(value) - max_len} chars]"
        return event_dict

    return _processor


def configure_logging(
    *,
    level: str = "INFO",
    env: str = "local",
    log_file: Path | None = None,
) -> None:
    """Wire structlog through stdlib logging.

    - stdout: colour console renderer in dev, JSON in prod
    - log_file (optional): always JSON, one record per line, UTF-8

    Idempotent: replaces any pre-existing root handlers.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Windows stdout defaults to cp1252 + errors='strict', which crashes when
    # the bge-m3 NaN-warning path tries to log a chunk_text snippet containing
    # CJK/fullwidth chars. Python logging swallows the encode error via
    # Handler.handleError, but the line is truncated and stderr fills with
    # `--- Logging error ---` traces. Switching to errors='replace' lets the
    # full message reach the stream with `?` substituted for un-encodable chars.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    foreign_pre_chain: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        truncate_long_strings(),
        timestamper,
    ]

    handlers: list[logging.Handler] = []

    # stdout: pretty in dev (rich tracebacks render natively), JSON in prod.
    if env == "prod":
        stdout_processors: list[Any] = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        stdout_processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=foreign_pre_chain, processors=stdout_processors
        )
    )
    handlers.append(stdout_handler)

    if log_file is not None:
        # File sink is always JSON; format exc_info into an "exception" string.
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=foreign_pre_chain,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )
        handlers.append(file_handler)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(log_level)

    structlog.configure(
        processors=[
            *foreign_pre_chain,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    # Tame third-party noise but keep app logs visible.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> BoundLogger:
    """Bound logger keyed by module name. Use as `log = get_logger(__name__)`."""
    return cast(BoundLogger, structlog.get_logger(name))


@contextmanager
def timed_event(logger: BoundLogger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Emit a single log record on exit with `duration_ms`. Replaces start/done pairs.

    Usage:
        with timed_event(log, "retrieve.done", query=q.text, top_k=q.top_k) as ctx:
            results = do_retrieve()
            ctx["returned"] = len(results)
            ctx["top_chunk"] = results[0].chunk_id if results else None

    On a raised exception: emits the same event at ERROR level with `exc_info=True`
    and `duration_ms`, then re-raises.
    """
    extra: dict[str, Any] = {}
    started = time.monotonic()
    try:
        yield extra
    except Exception:
        duration_ms = int((time.monotonic() - started) * 1000)
        logger.error(event, duration_ms=duration_ms, **{**fields, **extra}, exc_info=True)
        raise
    duration_ms = int((time.monotonic() - started) * 1000)
    logger.info(event, duration_ms=duration_ms, **{**fields, **extra})
