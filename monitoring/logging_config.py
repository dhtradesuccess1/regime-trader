"""Structured logging configuration built on structlog.

Emits newline-delimited JSON (``.jsonl``) to
``{LOG_DIR}/trading_{YYYYMMDD}.jsonl``, rotated daily with 30 days retained.
``LOG_LEVEL`` is read from the environment (``.env``).

Every event automatically carries a fixed set of trading-context fields, filled
from a bound context (see :func:`bind_trading_context`) or sensible defaults:

    ts, event, regime, confidence, portfolio_value, drawdown_pct, active_positions

Error-level events additionally carry ``error_type``, ``stack_trace`` (formatted
traceback), and ``last_bar_data`` (last known OHLCV values, from context).

Both structlog loggers and stdlib ``logging.getLogger`` loggers (used elsewhere
in the codebase) are routed through the same JSON pipeline, so all output is
consistent. Nothing in the codebase uses ``print()`` -- all output is via
structlog.
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import structlog
from dotenv import load_dotenv

# Trading-context fields injected into every event, with their defaults.
TRADING_CONTEXT_DEFAULTS = {
    "regime": "unknown",
    "confidence": None,
    "portfolio_value": None,
    "drawdown_pct": None,
    "active_positions": None,
}

# Log levels considered errors for the extra error-field requirement.
_ERROR_LEVELS = {"error", "critical", "exception"}

LOG_FILE_PREFIX = "trading_"
LOG_FILE_EXT = ".jsonl"
RETENTION_DAYS = 30


# --------------------------------------------------------------- processors
def add_trading_defaults(logger, method_name, event_dict):
    """Ensure every event carries the fixed trading-context fields."""
    for key, default in TRADING_CONTEXT_DEFAULTS.items():
        event_dict.setdefault(key, default)
    return event_dict


def _normalize_exc_info(value):
    """Coerce a structlog ``exc_info`` value into a (type, val, tb) tuple."""
    if isinstance(value, BaseException):
        return (type(value), value, value.__traceback__)
    if isinstance(value, tuple) and len(value) == 3:
        return value
    if value:  # True -> use the current exception
        return sys.exc_info()
    return None


def add_error_fields(logger, method_name, event_dict):
    """Attach error_type, stack_trace, and last_bar_data to error events.

    Always removes the raw ``exc_info`` (a traceback object is not JSON
    serializable) after extracting what we need.
    """
    level = event_dict.get("level", method_name)
    exc_info = event_dict.pop("exc_info", None)

    if level in _ERROR_LEVELS:
        ei = _normalize_exc_info(exc_info)
        if ei and ei[0] is not None:
            event_dict["error_type"] = ei[0].__name__
            event_dict["stack_trace"] = "".join(traceback.format_exception(*ei))
        else:
            event_dict.setdefault("error_type", None)
            event_dict.setdefault("stack_trace", None)
        # last_bar_data comes from bound context if present, else empty.
        event_dict.setdefault("last_bar_data", {})

    return event_dict


# --------------------------------------------------------------- file handler
class DailyJsonlFileHandler(TimedRotatingFileHandler):
    """Daily-rotating handler writing to ``trading_{YYYYMMDD}.jsonl`` files.

    The active file is always named for the current UTC date; at midnight a new
    dated file is opened and files older than ``RETENTION_DAYS`` are pruned.
    """

    def __init__(self, log_dir, retention_days: int = RETENTION_DAYS, encoding: str = "utf-8"):
        self._log_dir = Path(log_dir)
        super().__init__(
            self._current_path(),
            when="midnight",
            backupCount=retention_days,
            utc=True,
            encoding=encoding,
        )

    def _current_path(self) -> str:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        return str(self._log_dir / f"{LOG_FILE_PREFIX}{date}{LOG_FILE_EXT}")

    def _prune_old(self) -> None:
        if self.backupCount <= 0:
            return
        files = sorted(self._log_dir.glob(f"{LOG_FILE_PREFIX}*{LOG_FILE_EXT}"))
        for stale in files[: -self.backupCount]:
            try:
                stale.unlink()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass

    def doRollover(self) -> None:  # noqa: N802 (stdlib casing)
        if self.stream:
            self.stream.close()
            self.stream = None
        self.baseFilename = os.path.abspath(self._current_path())
        if not self.delay:
            self.stream = self._open()
        self._prune_old()
        # Schedule the next midnight rollover.
        current = int(time.time())
        next_at = self.computeRollover(current)
        while next_at <= current:
            next_at += self.interval
        self.rolloverAt = next_at


# --------------------------------------------------------------- public API
def configure_logging(log_level: str | None = None, log_dir: str | None = None) -> None:
    """Configure structlog + stdlib logging for JSONL output. Idempotent-safe."""
    load_dotenv()
    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    directory = log_dir or os.getenv("LOG_DIR", "./logs")
    Path(directory).mkdir(parents=True, exist_ok=True)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        add_trading_defaults,
        add_error_fields,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = DailyJsonlFileHandler(directory)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str | None = None):
    """Return a structlog logger (configure once via configure_logging)."""
    return structlog.get_logger(name)


def get_console_logger():
    """Return a structlog logger that renders human-readable lines to stdout.

    Used for CLI summaries. Output goes through structlog (no ``print`` calls in
    application code), separate from the JSONL file pipeline.
    """
    return structlog.wrap_logger(
        structlog.PrintLogger(file=sys.stdout),
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
    )


def bind_trading_context(**fields) -> None:
    """Bind trading-context fields (regime, confidence, ...) for all later events.

    Accepts any of the context keys plus ``last_bar_data``. Call as state
    changes (e.g. on each new bar / regime update).
    """
    structlog.contextvars.bind_contextvars(**fields)


def clear_trading_context() -> None:
    """Clear all bound trading-context fields."""
    structlog.contextvars.clear_contextvars()
