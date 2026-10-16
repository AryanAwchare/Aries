"""Structured logging with optional file output.

Every stage of the pipeline logs to a timestamped file under ``logs/``
plus the console. Rotating file handler keeps disk bounded.
"""
from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Query-string parameters that must never reach the logs (httpx logs the full
# request URL at INFO, which would expose Myfxbook email/password/session).
_SECRET_PARAM_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|token|api[_-]?key|apikey|secret|session|email)"
    r"=([^&\s'\"]+)"
)


def scrub_secrets(text: str) -> str:
    """Redact credential-bearing query parameters from a log line."""
    return _SECRET_PARAM_RE.sub(lambda m: f"{m.group(1)}=***", text)


class _RedactingFilter(logging.Filter):
    """Scrubs secrets from every record, in case any library logs a URL."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if isinstance(record.msg, str):
            record.msg = scrub_secrets(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (scrub_secrets(v) if isinstance(v, str) else v) for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    scrub_secrets(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def setup_logging(level: int = logging.INFO, log_dir: Path | None = None) -> None:
    # httpx/httpcore log the full request URL (including credentials) at INFO;
    # raise their threshold so those lines are never emitted.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    redactor = _RedactingFilter()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT))
    console.addFilter(redactor)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "platform.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(_FMT))
        fh.addFilter(redactor)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)