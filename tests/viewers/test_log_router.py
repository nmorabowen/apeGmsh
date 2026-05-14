"""Unit tests for :class:`LogRouter`.

Covers install/uninstall, idempotency, signal routing for each of
the three capture channels (Python logging, sys.excepthook,
sys.unraisablehook), and minimum-level configuration.
"""
from __future__ import annotations

import logging
import sys

import pytest

pytest.importorskip("qtpy.QtCore")

from apeGmsh.viewers.ui._log_router import (
    LogRouter,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def captured(qapp):
    """Yield (router, collected_messages_list).

    Tests use the list to verify what the router emitted. The router
    is installed at fixture entry, uninstalled at exit — so global
    sys hooks + logging handlers are restored even on test failure.
    """
    router = LogRouter()
    msgs: list[tuple[str, str]] = []
    router.message.connect(lambda text, sev: msgs.append((text, sev)))
    router.install()
    try:
        yield router, msgs
    finally:
        router.uninstall()


def _drain(qapp):
    """Process any queued Qt events so signals fire."""
    qapp.processEvents()


# =====================================================================
# install / uninstall lifecycle
# =====================================================================


def test_install_uninstall_roundtrip_restores_excepthook(qapp):
    original = sys.excepthook
    router = LogRouter()
    router.install()
    assert sys.excepthook is not original
    router.uninstall()
    assert sys.excepthook is original


def test_install_uninstall_roundtrip_restores_unraisablehook(qapp):
    original = sys.unraisablehook
    router = LogRouter()
    router.install()
    assert sys.unraisablehook is not original
    router.uninstall()
    assert sys.unraisablehook is original


def test_install_uninstall_roundtrip_removes_log_handler(qapp):
    root = logging.getLogger()
    before = len(root.handlers)
    router = LogRouter()
    router.install()
    assert len(root.handlers) == before + 1
    router.uninstall()
    assert len(root.handlers) == before


def test_install_is_idempotent(qapp):
    router = LogRouter()
    router.install()
    root = logging.getLogger()
    handlers_after_first = len(root.handlers)
    router.install()    # second call should no-op
    assert len(root.handlers) == handlers_after_first
    router.uninstall()


def test_uninstall_before_install_is_noop(qapp):
    router = LogRouter()
    # Should not raise.
    router.uninstall()


def test_is_installed_property(qapp):
    router = LogRouter()
    assert router.is_installed is False
    router.install()
    assert router.is_installed is True
    router.uninstall()
    assert router.is_installed is False


# =====================================================================
# Python logging — handler routes records to signal
# =====================================================================


def test_logging_warning_emits_warning_severity(captured, qapp):
    router, msgs = captured
    logging.getLogger("apeGmsh.test").warning("hello-warn")
    _drain(qapp)
    assert any("hello-warn" in t and s == SEVERITY_WARNING for t, s in msgs)


def test_logging_error_emits_error_severity(captured, qapp):
    router, msgs = captured
    logging.getLogger("apeGmsh.test").error("hello-err")
    _drain(qapp)
    assert any("hello-err" in t and s == SEVERITY_ERROR for t, s in msgs)


def test_logging_below_threshold_is_suppressed(captured, qapp):
    router, msgs = captured
    # Default threshold is WARNING — info/debug shouldn't reach the dock.
    logging.getLogger("apeGmsh.test").info("invisible-info")
    logging.getLogger("apeGmsh.test").debug("invisible-debug")
    _drain(qapp)
    assert not any("invisible-info" in t for t, _s in msgs)
    assert not any("invisible-debug" in t for t, _s in msgs)


def test_set_log_level_changes_threshold(qapp):
    """set_log_level() only changes the HANDLER level — the logger
    itself must also allow the level through. We set both to INFO and
    verify the message reaches the dock.
    """
    router = LogRouter()
    msgs: list[tuple[str, str]] = []
    router.message.connect(lambda t, s: msgs.append((t, s)))
    router.set_log_level(logging.INFO)
    router.install()
    test_logger = logging.getLogger("apeGmsh.test.set_log_level")
    prior_level = test_logger.level
    test_logger.setLevel(logging.INFO)
    try:
        test_logger.info("hello-info")
        _drain(qapp)
        assert any("hello-info" in t and s == SEVERITY_INFO for t, s in msgs)
    finally:
        test_logger.setLevel(prior_level)
        router.uninstall()


def test_logging_exception_includes_traceback(captured, qapp):
    router, msgs = captured
    try:
        raise ValueError("boom-via-logging")
    except ValueError:
        logging.getLogger("apeGmsh.test").exception("with-traceback")
    _drain(qapp)
    # The message should contain BOTH the user text and the traceback.
    matched = [(t, s) for t, s in msgs if "with-traceback" in t]
    assert matched
    text, severity = matched[-1]
    assert severity == SEVERITY_ERROR
    assert "boom-via-logging" in text


# =====================================================================
# sys.excepthook — unhandled exceptions
# =====================================================================


def test_excepthook_emits_error_with_traceback(captured, qapp):
    router, msgs = captured
    try:
        raise RuntimeError("boom-via-excepthook")
    except RuntimeError:
        exc_type, exc_value, exc_tb = sys.exc_info()
    # Call excepthook directly with the captured info.
    sys.excepthook(exc_type, exc_value, exc_tb)
    _drain(qapp)
    matched = [(t, s) for t, s in msgs if "boom-via-excepthook" in t]
    assert matched
    text, severity = matched[-1]
    assert severity == SEVERITY_ERROR
    assert "RuntimeError" in text


def test_excepthook_delegates_to_original(qapp):
    """Original hook still fires — we don't swallow the message."""
    seen = []

    def fake_original(et, ev, tb):
        seen.append(et)

    original_excepthook = sys.excepthook
    sys.excepthook = fake_original
    try:
        router = LogRouter()
        router.install()
        try:
            try:
                raise RuntimeError("forward-me")
            except RuntimeError:
                ei = sys.exc_info()
            sys.excepthook(*ei)
            assert seen == [RuntimeError]
        finally:
            router.uninstall()
    finally:
        sys.excepthook = original_excepthook


# =====================================================================
# sys.unraisablehook — Qt-slot errors on PyQt6/PySide6
# =====================================================================


def test_unraisablehook_emits_error(captured, qapp):
    import warnings

    router, msgs = captured
    # Simulate sys.unraisablehook's parameter — a named tuple-like
    # object with exc_type / exc_value / exc_traceback / err_msg.
    try:
        raise IndexError("boom-via-unraisable")
    except IndexError:
        et, ev, tb = sys.exc_info()

    class _Unraisable:
        exc_type = et
        exc_value = ev
        exc_traceback = tb
        err_msg = "from a Qt slot"
        object = None

    # Suppress the PytestUnraisableExceptionWarning that pytest
    # synthesises for this kind of intentional test invocation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.unraisablehook(_Unraisable())
    _drain(qapp)
    matched = [(t, s) for t, s in msgs if "boom-via-unraisable" in t]
    assert matched
    _text, severity = matched[-1]
    assert severity == SEVERITY_ERROR


# =====================================================================
# Manual emission API
# =====================================================================


def test_info_warning_error_methods(qapp):
    router = LogRouter()
    msgs: list[tuple[str, str]] = []
    router.message.connect(lambda t, s: msgs.append((t, s)))
    router.info("i")
    router.warning("w")
    router.error("e")
    _drain(qapp)
    assert ("i", SEVERITY_INFO) in msgs
    assert ("w", SEVERITY_WARNING) in msgs
    assert ("e", SEVERITY_ERROR) in msgs


def test_manual_emit_works_without_install(qapp):
    """info/warning/error don't require install()."""
    router = LogRouter()
    msgs: list[tuple[str, str]] = []
    router.message.connect(lambda t, s: msgs.append((t, s)))
    router.warning("standalone")
    _drain(qapp)
    assert ("standalone", SEVERITY_WARNING) in msgs
