"""Structured logging via structlog. Pretty stdout in dev, JSON file always.

Call `configure_logging()` once at process startup (FastAPI `create_app`,
or at the top of any CLI script). Then `get_logger(__name__)` anywhere.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger


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

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    foreign_pre_chain: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        timestamper,
    ]

    stdout_renderer: Any = (
        structlog.processors.JSONRenderer()
        if env == "prod"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    handlers: list[logging.Handler] = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=foreign_pre_chain,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                stdout_renderer,
            ],
        )
    )
    handlers.append(stdout_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=foreign_pre_chain,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
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
