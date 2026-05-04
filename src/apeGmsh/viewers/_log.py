"""ResultsViewer action logger.

Centralises every gesture, dispatch, pick, and error into a single
log surface. Two destinations:

* **stderr** — INFO+ only. Replaces the ``[viewer-dispatch]`` trace
  spam with a cleaner human-readable line per gesture.
* **per-session file** — DEBUG+, written to
  ``~/.apegmsh/viewer-logs/session-YYYYMMDD-HHMMSS.log``. Captures
  the full action trail. Self-contained: a bug report attaches the
  most recent file and we can replay every gesture the user made.

Use :func:`log_action` from any handler / observer / signal wire-up::

    from apeGmsh.viewers._log import log_action

    def _on_apply_clicked(self):
        log_action("ui.settings", "apply_clicked", layer=id(self._d))
        ...

The category should mirror the matrix (``ui.outline.*``,
``ui.geometry.*``, ``ui.settings.*``, ``ui.scrubber.*``,
``ui.shortcut.*``, ``pick.*``, ``probe.*``, ``file.*``,
``dispatch.*``, ``error.*``, ``session.*``). Action is the verb.
Payload is whatever's relevant — keep it short, ``str(value)`` for
non-primitives.

Env vars:

* ``APEGMSH_LOG_DIR``  — override the log directory (default
  ``~/.apegmsh/viewer-logs``).
* ``APEGMSH_LOG_KEEP`` — number of session logs to retain (default
  ``20``). Older are pruned at logger init.
* ``APEGMSH_LOG_LEVEL`` — minimum level for the file handler
  (default ``DEBUG``). Accepts ``DEBUG`` / ``INFO`` / ``WARNING`` /
  ``ERROR`` / ``CRITICAL`` (case-insensitive).
* ``APEGMSH_LOG_CONSOLE_LEVEL`` — minimum level for stderr handler
  (default ``INFO``).
* ``APEGMSH_LOG_OFF`` — set to any non-empty value to disable
  logging entirely (no file, no console).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


_LOGGER_NAME = "apegmsh.viewer"
_DEFAULT_KEEP = 20
_DEFAULT_FILE_LEVEL = "DEBUG"
_DEFAULT_CONSOLE_LEVEL = "INFO"
_DEFAULT_LOG_DIR = Path.home() / ".apegmsh" / "viewer-logs"

# Singleton state — populated on first ``get_logger()`` call.
_INITIALIZED: bool = False
_SESSION_FILE: Optional[Path] = None


def _resolve_level(value: Optional[str], default: str) -> int:
    """Map a level string to the ``logging`` constant."""
    name = (value or default).upper().strip()
    return getattr(logging, name, logging.INFO)


def _resolve_log_dir() -> Path:
    """Pick the log directory from env or default."""
    override = os.environ.get("APEGMSH_LOG_DIR")
    return Path(override) if override else _DEFAULT_LOG_DIR


def _prune_old_sessions(log_dir: Path, keep: int) -> None:
    """Keep the ``keep`` most recent ``session-*.log`` files; delete the rest."""
    try:
        sessions = sorted(
            log_dir.glob("session-*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in sessions[keep:]:
            try:
                stale.unlink()
            except Exception:
                pass
    except Exception:
        pass


def get_logger() -> logging.Logger:
    """Singleton viewer action logger.

    Initialises file + console handlers on first call, prunes old
    session logs per ``APEGMSH_LOG_KEEP``, and writes a session
    header line. Subsequent calls return the same logger.
    """
    global _INITIALIZED, _SESSION_FILE

    logger = logging.getLogger(_LOGGER_NAME)
    if _INITIALIZED:
        return logger

    if os.environ.get("APEGMSH_LOG_OFF"):
        # Hard off — leave logger silent (no handlers).
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        _INITIALIZED = True
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False    # don't bleed into root logger

    # Console — INFO+ to stderr, compact format.
    console_level = _resolve_level(
        os.environ.get("APEGMSH_LOG_CONSOLE_LEVEL"), _DEFAULT_CONSOLE_LEVEL,
    )
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("[viewer] %(message)s"))
    logger.addHandler(ch)

    # File — DEBUG+ to per-session log, with timestamps.
    log_dir = _resolve_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Disk write blocked — skip the file handler, keep console.
        _INITIALIZED = True
        return logger

    try:
        keep = int(os.environ.get("APEGMSH_LOG_KEEP", _DEFAULT_KEEP))
    except ValueError:
        keep = _DEFAULT_KEEP
    _prune_old_sessions(log_dir, keep)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"session-{timestamp}.log"
    file_level = _resolve_level(
        os.environ.get("APEGMSH_LOG_LEVEL"), _DEFAULT_FILE_LEVEL,
    )
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
        _SESSION_FILE = log_file
    except Exception:
        pass

    _INITIALIZED = True

    # Session header — version + log path.
    try:
        from apeGmsh import __version__ as _ver
    except Exception:
        _ver = "unknown"
    logger.info(f"session.start  apeGmsh={_ver} log={_SESSION_FILE}")
    return logger


def log_action(category: str, action: str, **payload: Any) -> None:
    """Log a single user / dispatch / pick action.

    Format: ``<category>.<action>  k1=v1 k2=v2 ...``

    Level is INFO by default. Pass ``_level="debug"`` (or warning /
    error) in the payload to override.
    """
    logger = get_logger()
    level = payload.pop("_level", "info")
    if isinstance(level, str):
        level_int = _resolve_level(level, "INFO")
    else:
        level_int = int(level)
    if not logger.isEnabledFor(level_int):
        return
    msg = f"{category}.{action}"
    if payload:
        msg += "  " + " ".join(f"{k}={_format(v)}" for k, v in payload.items())
    logger.log(level_int, msg)


def log_error(category: str, action: str, exc: BaseException, **payload: Any) -> None:
    """Log an error from a handler / dispatcher / pick.

    Includes the exception type + message; full traceback goes to the
    file handler at DEBUG level.
    """
    logger = get_logger()
    msg = f"{category}.{action}  exc={type(exc).__name__}({exc!r})"
    if payload:
        msg += "  " + " ".join(f"{k}={_format(v)}" for k, v in payload.items())
    logger.error(msg)
    logger.debug(f"{category}.{action}  traceback follows", exc_info=exc)


def session_file() -> Optional[Path]:
    """Return the path of the current session log file (or None)."""
    return _SESSION_FILE


def _format(v: Any) -> str:
    """Compact repr suitable for one-line log output."""
    if v is None:
        return "None"
    if isinstance(v, (str, int, float, bool)):
        return repr(v)
    # Long objects → type + id only, full payload would explode the line.
    try:
        return f"{type(v).__name__}#{id(v):x}"
    except Exception:
        return "<unrepr>"


def shutdown() -> None:
    """Flush + close all handlers. Called by ResultsViewer._on_close."""
    logger = logging.getLogger(_LOGGER_NAME)
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
            logger.removeHandler(h)
        except Exception:
            pass
    global _INITIALIZED, _SESSION_FILE
    _INITIALIZED = False
    _SESSION_FILE = None
