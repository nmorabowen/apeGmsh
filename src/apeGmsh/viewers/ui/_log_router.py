"""Log router — central message handler for the OutputDock.

Captures messages from three sources and routes them through a Qt
signal that the :class:`OutputDock` consumes:

1. **Python ``logging``** — a custom :class:`logging.Handler` attached
   to the root logger at WARNING level. Any ``logging.warning(...)`` /
   ``logging.error(...)`` / ``logging.exception(...)`` from anywhere
   in the process surfaces in the dock.
2. **``sys.excepthook``** — replaces the default with one that emits
   the traceback as severity ``"error"`` and *also* delegates to the
   original so the REPL / terminal still gets the message. Covers
   unhandled exceptions in the main thread.
3. **``sys.unraisablehook``** — Python 3.8+'s hook for "unraisable"
   exceptions (PyQt/PySide signal-slot errors land here on PyQt6 /
   PySide6). Same severity routing.

VTK ``vtkOutputWindow`` capture is intentionally NOT installed here —
it's a deferred follow-up because Python subclassing of
``vtkOutputWindow`` is wrapper-version-dependent and the failure modes
are silent. The Python channels above cover the daily-pain case
(tracebacks during diagram updates).

The router is **idempotent**: calling :meth:`install` twice or
:meth:`uninstall` before :meth:`install` is a no-op. On :meth:`uninstall`
the originals are restored so the viewer leaves the process in the
same state it found it.
"""
from __future__ import annotations

import logging
import sys
import traceback
from typing import Any, Optional


# Severities (str values, not an enum — keeps the signal payload
# trivially serialisable for Qt's marshaling).
SEVERITY_INFO    = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR   = "error"


def _severity_for_level(levelno: int) -> str:
    """Map a ``logging`` level → severity string."""
    if levelno >= logging.ERROR:
        return SEVERITY_ERROR
    if levelno >= logging.WARNING:
        return SEVERITY_WARNING
    return SEVERITY_INFO


class _RouterLogHandler(logging.Handler):
    """A ``logging.Handler`` that emits records through a LogRouter."""

    def __init__(self, router: "LogRouter") -> None:
        super().__init__()
        self._router = router

    def emit(self, record: logging.LogRecord) -> None:
        severity = _severity_for_level(record.levelno)
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        if record.exc_info:
            try:
                msg = msg + "\n" + "".join(
                    traceback.format_exception(*record.exc_info)
                )
            except Exception:
                pass
        try:
            self._router._emit(msg, severity)
        except Exception:
            # Never let logging itself crash the app.
            pass


class LogRouter:
    """Per-viewer log router. One instance per OutputDock.

    Construct, call :meth:`install` to begin capturing, connect a
    consumer to :attr:`message`, call :meth:`uninstall` on viewer
    close.

    Attributes
    ----------
    message
        Qt ``Signal(str, str)`` — ``(text, severity)``. Connect a
        slot to it (typically :meth:`OutputDock.append`).
    """

    def __init__(self) -> None:
        # Late-import qtpy so this module is safe to import in headless
        # contexts (script generation, doc builds).
        from qtpy import QtCore

        # The signal lives on a private QObject — LogRouter itself is
        # not a QObject so it can be constructed before QApplication.
        class _Signaller(QtCore.QObject):
            message = QtCore.Signal(str, str)

        self._signaller = _Signaller()
        # Public attribute — clients connect to router.message.connect(...)
        self.message = self._signaller.message

        self._installed: bool = False
        self._original_excepthook: Optional[Any] = None
        self._original_unraisablehook: Optional[Any] = None
        self._log_handler: Optional[_RouterLogHandler] = None
        # Minimum level the log handler captures — overridable via
        # set_log_level() before install().
        self._log_level: int = logging.WARNING

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_log_level(self, level: int) -> None:
        """Set the minimum ``logging`` level captured.

        Defaults to ``WARNING``. Call before :meth:`install`; changes
        after install are applied to the live handler.
        """
        self._log_level = level
        if self._log_handler is not None:
            self._log_handler.setLevel(level)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        """Hook Python ``logging`` / ``sys.excepthook`` /
        ``sys.unraisablehook``. Idempotent."""
        if self._installed:
            return
        self._installed = True

        # 1. Python logging
        self._log_handler = _RouterLogHandler(self)
        self._log_handler.setLevel(self._log_level)
        logging.getLogger().addHandler(self._log_handler)

        # 2. sys.excepthook
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

        # 3. sys.unraisablehook (Py 3.8+; defensive guard)
        original_unraisable = getattr(sys, "unraisablehook", None)
        if original_unraisable is not None:
            self._original_unraisablehook = original_unraisable
            sys.unraisablehook = self._unraisablehook

    def uninstall(self) -> None:
        """Restore the originals. Idempotent."""
        if not self._installed:
            return
        self._installed = False

        if self._log_handler is not None:
            try:
                logging.getLogger().removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None

        if self._original_excepthook is not None:
            try:
                sys.excepthook = self._original_excepthook
            except Exception:
                pass
            self._original_excepthook = None

        if self._original_unraisablehook is not None:
            try:
                sys.unraisablehook = self._original_unraisablehook
            except Exception:
                pass
            self._original_unraisablehook = None

    # ------------------------------------------------------------------
    # Manual emission — exposed for direct use (status messages, etc.)
    # ------------------------------------------------------------------

    def info(self, text: str) -> None:
        self._emit(text, SEVERITY_INFO)

    def warning(self, text: str) -> None:
        self._emit(text, SEVERITY_WARNING)

    def error(self, text: str) -> None:
        self._emit(text, SEVERITY_ERROR)

    def _emit(self, text: str, severity: str) -> None:
        """Emit through the Qt signal. Thread-safe — Qt marshals to UI thread."""
        try:
            self._signaller.message.emit(text, severity)
        except Exception:
            # If Qt is shut down mid-emit (process teardown), don't crash.
            pass

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def _excepthook(self, exctype, value, tb) -> None:
        """Replacement ``sys.excepthook`` — emit then delegate to original."""
        try:
            text = "".join(traceback.format_exception(exctype, value, tb))
            self._emit(text, SEVERITY_ERROR)
        except Exception:
            pass
        # Forward to the original so terminal output is preserved.
        if self._original_excepthook is not None:
            try:
                self._original_excepthook(exctype, value, tb)
            except Exception:
                pass

    def _unraisablehook(self, unraisable) -> None:
        """Replacement ``sys.unraisablehook`` — Qt-slot errors on PyQt6/PySide6.

        ``unraisable`` is a named tuple ``(exc_type, exc_value,
        exc_traceback, err_msg, object)``.
        """
        try:
            parts: list[str] = []
            err_msg = getattr(unraisable, "err_msg", None)
            if err_msg:
                parts.append(str(err_msg))
            exc_type = getattr(unraisable, "exc_type", None)
            exc_value = getattr(unraisable, "exc_value", None)
            exc_tb = getattr(unraisable, "exc_traceback", None)
            if exc_type is not None:
                parts.append("".join(
                    traceback.format_exception(exc_type, exc_value, exc_tb)
                ))
            self._emit("\n".join(parts) if parts else "Unraisable exception",
                       SEVERITY_ERROR)
        except Exception:
            pass
        # Forward to original.
        if self._original_unraisablehook is not None:
            try:
                self._original_unraisablehook(unraisable)
            except Exception:
                pass
