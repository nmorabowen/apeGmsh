"""Unit tests for :class:`OutputDock`.

Covers the append + count + clear contract and integration with
:class:`LogRouter`'s message signal. No ResultsViewer / VTK
construction — all tests use a vanilla LogRouter (no install) plus
the OutputDock widget directly.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtCore")

from apeGmsh.viewers.ui._log_router import (
    LogRouter,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
)
from apeGmsh.viewers.ui._output_dock import OutputDock, make_output_dock


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def dock_pair(qapp):
    """Yield ``(router, output_dock)`` — both alive for the test."""
    router = LogRouter()
    output = OutputDock(router)
    try:
        yield router, output
    finally:
        try:
            output.close()
        except Exception:
            pass


def _drain(qapp):
    qapp.processEvents()


# =====================================================================
# Construction
# =====================================================================


def test_outputdock_widget_is_qwidget(dock_pair):
    _router, output = dock_pair
    from qtpy import QtWidgets
    assert isinstance(output.widget, QtWidgets.QWidget)


def test_outputdock_text_view_is_qplaintextedit(dock_pair):
    _router, output = dock_pair
    from qtpy import QtWidgets
    assert isinstance(output.text_view, QtWidgets.QPlainTextEdit)
    assert output.text_view.isReadOnly()


def test_outputdock_starts_with_zero_counts(dock_pair):
    _router, output = dock_pair
    assert output.counts == {"info": 0, "warning": 0, "error": 0}


# =====================================================================
# append() — direct calls + via router signal
# =====================================================================


def test_append_direct_increments_count(dock_pair):
    _router, output = dock_pair
    output.append("hello", SEVERITY_WARNING)
    output.append("oops", SEVERITY_ERROR)
    assert output.counts == {"info": 0, "warning": 1, "error": 1}


def test_append_via_router_signal(dock_pair, qapp):
    router, output = dock_pair
    router.warning("from-signal")
    _drain(qapp)
    assert output.counts["warning"] == 1
    assert "from-signal" in output.text_view.toPlainText()


def test_append_unknown_severity_treated_as_info(dock_pair):
    _router, output = dock_pair
    output.append("mystery", "fatal")    # unknown severity
    assert output.counts == {"info": 1, "warning": 0, "error": 0}


def test_append_empty_text_is_noop(dock_pair):
    _router, output = dock_pair
    output.append("", SEVERITY_WARNING)
    output.append(None, SEVERITY_WARNING)  # type: ignore[arg-type]
    assert output.counts == {"info": 0, "warning": 0, "error": 0}


def test_append_html_escaped(dock_pair):
    _router, output = dock_pair
    # If this were rendered as raw HTML, the <b> tag would NOT appear
    # as text. Escaping means the literal characters survive.
    output.append("<b>not bold</b>", SEVERITY_INFO)
    plain = output.text_view.toPlainText()
    assert "<b>not bold</b>" in plain


def test_append_newlines_preserved(dock_pair):
    _router, output = dock_pair
    output.append("line1\nline2", SEVERITY_INFO)
    plain = output.text_view.toPlainText()
    assert "line1" in plain
    assert "line2" in plain


# =====================================================================
# clear()
# =====================================================================


def test_clear_resets_counts_and_text(dock_pair):
    _router, output = dock_pair
    output.append("a", SEVERITY_WARNING)
    output.append("b", SEVERITY_ERROR)
    assert sum(output.counts.values()) == 2
    output.clear()
    assert output.counts == {"info": 0, "warning": 0, "error": 0}
    assert output.text_view.toPlainText() == ""


# =====================================================================
# on_append observers (for the future status-bar badge)
# =====================================================================


def test_on_append_observer_fires(dock_pair):
    _router, output = dock_pair
    captured: list[tuple[str, str]] = []
    output.on_append(lambda t, s: captured.append((t, s)))
    output.append("hi", SEVERITY_WARNING)
    output.append("bye", SEVERITY_ERROR)
    assert captured == [
        ("hi", SEVERITY_WARNING),
        ("bye", SEVERITY_ERROR),
    ]


def test_on_append_observer_exception_does_not_break_append(dock_pair):
    """A buggy observer doesn't take down the log channel."""
    _router, output = dock_pair
    output.on_append(lambda t, s: 1 / 0)
    # Should not raise.
    output.append("survived", SEVERITY_INFO)
    assert output.counts["info"] == 1


# =====================================================================
# close() — idempotent, no errors on double-call
# =====================================================================


def test_close_disconnects_from_router_signal(dock_pair, qapp):
    router, output = dock_pair
    output.close()
    router.warning("after-close")
    _drain(qapp)
    # Counts unchanged after disconnect.
    assert output.counts["warning"] == 0


def test_close_is_idempotent(dock_pair):
    _router, output = dock_pair
    output.close()
    output.close()  # second call should be a no-op


# =====================================================================
# make_output_dock helper
# =====================================================================


def test_make_output_dock_returns_dock_and_spec(qapp):
    from apeGmsh.viewers.ui._dock_registry import DockSpec
    router = LogRouter()
    output, spec = make_output_dock(router)
    assert isinstance(output, OutputDock)
    assert isinstance(spec, DockSpec)
    assert spec.dock_id == "dock_output"
    assert spec.title == "Output"
    assert spec.default_area == "bottom"
    assert spec.default_visible is False


def test_make_output_dock_factory_returns_widget(qapp):
    from qtpy import QtWidgets
    router = LogRouter()
    output, spec = make_output_dock(router)
    fake_parent = QtWidgets.QWidget()
    returned = spec.factory(fake_parent)
    # Factory must return a QWidget that can be set on a QDockWidget.
    assert isinstance(returned, QtWidgets.QWidget)
    # And it must be the SAME widget that lives inside the OutputDock,
    # so the caller's reference to `output` is connected to what's in
    # the dock.
    assert returned is output.widget
