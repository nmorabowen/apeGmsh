"""Unit tests for :class:`LayoutPersistence`.

Each test uses a unique ``window_key`` (with the test's own pid+name
suffix) so concurrent test runs don't share QSettings state. The
``persist`` fixture also calls ``reset()`` on teardown to keep the
QSettings store clean.
"""
from __future__ import annotations

import os
import uuid

import pytest

from apeGmsh.viewers.ui._layout_persistence import LayoutPersistence


# =====================================================================
# Constructor validation (no Qt)
# =====================================================================


def test_persistence_rejects_empty_key():
    with pytest.raises(ValueError, match="non-empty"):
        LayoutPersistence(window_key="")


def test_persistence_rejects_path_separators_in_key():
    with pytest.raises(ValueError, match="path separators"):
        LayoutPersistence(window_key="bad/key")
    with pytest.raises(ValueError, match="path separators"):
        LayoutPersistence(window_key="bad\\key")


def test_persistence_stores_window_key():
    p = LayoutPersistence(window_key="results")
    assert p.window_key == "results"


# =====================================================================
# save / restore / reset — require Qt
# =====================================================================


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def persist():
    """Unique-key LayoutPersistence; reset before AND after each test."""
    key = f"test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    p = LayoutPersistence(window_key=key)
    p.reset()
    yield p
    p.reset()


def _make_window(qapp, *, with_dock=True):
    from qtpy import QtCore, QtWidgets
    win = QtWidgets.QMainWindow()
    win.setCentralWidget(QtWidgets.QWidget(win))
    if with_dock:
        dock = QtWidgets.QDockWidget("Test", win)
        dock.setObjectName("test_dock")
        dock.setWidget(QtWidgets.QWidget(dock))
        win.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
    return win


# ----- restore on empty state -------------------------------------------


def test_restore_returns_false_when_no_state(persist, qapp):
    win = _make_window(qapp)
    assert persist.restore(win) is False


def test_has_saved_state_false_initially(persist):
    assert persist.has_saved_state() is False


# ----- save + restore roundtrip -----------------------------------------


def test_save_then_restore_returns_true(persist, qapp):
    win1 = _make_window(qapp)
    win1.resize(1234, 567)
    persist.save(win1)
    win1.close()

    win2 = _make_window(qapp)
    assert persist.restore(win2) is True
    # Size restored.
    assert win2.size().width() == 1234
    assert win2.size().height() == 567


def test_save_restores_dock_visibility(persist, qapp):
    from qtpy import QtCore, QtWidgets

    win1 = _make_window(qapp)
    # Add a second dock and hide it.
    dock2 = QtWidgets.QDockWidget("Two", win1)
    dock2.setObjectName("dock2")
    dock2.setWidget(QtWidgets.QWidget(dock2))
    win1.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock2)
    dock2.setVisible(False)
    persist.save(win1)
    win1.close()

    # Fresh window with the same docks, both visible by default —
    # restore should hide dock2 again.
    win2 = _make_window(qapp)
    dock2_new = QtWidgets.QDockWidget("Two", win2)
    dock2_new.setObjectName("dock2")
    dock2_new.setWidget(QtWidgets.QWidget(dock2_new))
    win2.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock2_new)
    dock2_new.setVisible(True)

    assert persist.restore(win2) is True
    assert dock2_new.isVisible() is False


def test_has_saved_state_true_after_save(persist, qapp):
    win = _make_window(qapp)
    persist.save(win)
    assert persist.has_saved_state() is True


# ----- reset ------------------------------------------------------------


def test_reset_clears_state(persist, qapp):
    win = _make_window(qapp)
    persist.save(win)
    assert persist.has_saved_state() is True
    persist.reset()
    assert persist.has_saved_state() is False

    # And a subsequent restore returns False.
    win2 = _make_window(qapp)
    assert persist.restore(win2) is False


# ----- schema mismatch --------------------------------------------------


def test_restore_returns_false_on_schema_mismatch(persist, qapp, monkeypatch):
    """If SCHEMA_VERSION moves under an existing saved state, restore
    falls back to defaults rather than applying a stale blob."""
    win = _make_window(qapp)
    persist.save(win)
    assert persist.has_saved_state() is True

    # Simulate a schema bump by patching the class constant.
    monkeypatch.setattr(LayoutPersistence, "SCHEMA_VERSION", 999)

    win2 = _make_window(qapp)
    assert persist.restore(win2) is False


def test_restore_returns_false_on_garbage_schema(persist, qapp):
    """If the schema key holds a non-int, treat as no-state."""
    s = persist._settings()
    s.setValue("schema", "not-a-number")
    s.sync()

    win = _make_window(qapp)
    assert persist.restore(win) is False


# ----- isolation --------------------------------------------------------


def test_different_window_keys_are_isolated(qapp):
    """Two LayoutPersistences with different keys don't see each other's state."""
    key_a = f"test-iso-{uuid.uuid4().hex[:8]}"
    key_b = f"test-iso-{uuid.uuid4().hex[:8]}"
    pa = LayoutPersistence(window_key=key_a)
    pb = LayoutPersistence(window_key=key_b)
    pa.reset(); pb.reset()
    try:
        win = _make_window(qapp)
        pa.save(win)
        assert pa.has_saved_state() is True
        assert pb.has_saved_state() is False
    finally:
        pa.reset(); pb.reset()
