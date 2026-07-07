"""Regression: the ResultsViewer launch path must ensure a QApplication
before any QWidget is constructed.

Before this guard, ``_show_impl`` built the Output / Color-Map dock
QWidgets while the only ``QApplication`` creation lived deep inside
``ViewerWindow.__init__``. The ``python -m apeGmsh.viewers`` subprocess
(spawned by ``Results.viewer(blocking=False)``) therefore aborted with::

    QWidget: Must construct a QApplication before a QWidget

These tests pin the helper contract; no VTK / window construction.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.viewers.results_viewer import _ensure_qapplication


def test_ensure_qapplication_returns_instance():
    from qtpy import QtWidgets

    app = _ensure_qapplication()
    assert app is not None
    assert QtWidgets.QApplication.instance() is app


def test_ensure_qapplication_is_idempotent():
    """Second call reuses the instance — never a second QApplication."""
    first = _ensure_qapplication()
    second = _ensure_qapplication()
    assert first is second


def test_ensure_qapplication_runs_platform_guard_before_qt(monkeypatch):
    """Regression: the Wayland/kvantum guard must run inside
    ``_ensure_qapplication``, *before* qtpy is touched.

    Qt reads ``QT_QPA_PLATFORM`` / ``QT_STYLE_OVERRIDE`` once, at
    ``QApplication`` construction — and on the results-viewer path
    that construction happens here, long before ``ViewerWindow``
    (where the guard also lives) ever loads. Without this ordering,
    ``python -m apeGmsh.viewers`` on a Linux Wayland session crashes
    in VTK's X layer (BadWindow on X_ConfigureWindow).
    """
    import sys
    import types

    calls: list[str] = []

    monkeypatch.setattr(
        "apeGmsh.viewers.ui._qt_env.prepare_qt_environment",
        lambda: calls.append("guard"),
    )

    sentinel = object()

    class _QApplication:
        @staticmethod
        def instance():
            calls.append("qt")
            return sentinel

    fake_qtpy = types.ModuleType("qtpy")
    fake_qtpy.QtWidgets = types.SimpleNamespace(QApplication=_QApplication)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qtpy", fake_qtpy)

    app = _ensure_qapplication()

    assert app is sentinel
    assert calls == ["guard", "qt"]
