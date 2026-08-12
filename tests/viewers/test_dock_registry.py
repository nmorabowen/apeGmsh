"""Unit tests for :class:`DockSpec` and :func:`sanitize_dock_placement`.

Tests split into two groups:

* No-Qt — validates the pure-Python parts (DockSpec construction and
  field validation). Run without Qt.
* With Qt — validates :func:`sanitize_dock_placement` heals degenerate
  nav-dock placements without disturbing healthy ones.
"""
from __future__ import annotations

import pytest

from apeGmsh.viewers.ui._dock_registry import DockSpec


# =====================================================================
# DockSpec validation (no Qt)
# =====================================================================


def test_dockspec_basic_construction():
    spec = DockSpec(
        dock_id="outline",
        title="Outline",
        factory=lambda parent: object(),
    )
    assert spec.dock_id == "outline"
    assert spec.title == "Outline"
    assert spec.default_area == "right"
    assert spec.default_visible is True
    assert spec.default_floating is False
    assert spec.tabify_with is None


def test_dockspec_rejects_empty_id():
    with pytest.raises(ValueError, match="non-empty"):
        DockSpec(dock_id="", title="x", factory=lambda p: None)


def test_dockspec_rejects_invalid_chars_in_id():
    # Spaces, slashes, etc. would corrupt QSettings paths.
    with pytest.raises(ValueError, match="alphanumeric"):
        DockSpec(dock_id="bad id", title="x", factory=lambda p: None)
    with pytest.raises(ValueError, match="alphanumeric"):
        DockSpec(dock_id="bad/id", title="x", factory=lambda p: None)


def test_dockspec_accepts_separators_in_id():
    # Underscores, hyphens, dots all OK.
    for ok_id in ("foo_bar", "foo-bar", "foo.bar", "foo123"):
        spec = DockSpec(dock_id=ok_id, title="x", factory=lambda p: None)
        assert spec.dock_id == ok_id


def test_dockspec_rejects_invalid_area():
    with pytest.raises(ValueError, match="default_area"):
        DockSpec(
            dock_id="x", title="x", factory=lambda p: None,
            default_area="middle",
        )


# =====================================================================
# Qt fixtures — shared by the mount / sanitize groups below
# =====================================================================


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_window(qapp):
    from qtpy import QtWidgets
    win = QtWidgets.QMainWindow()
    # Central widget needed or QMainWindow warns when adding docks.
    win.setCentralWidget(QtWidgets.QWidget(win))
    return win


# =====================================================================
# sanitize_dock_placement — idempotent per-launch heal for a degenerate
# nav-dock placement (the recurring "stuck on the upper edge" bug).
# Requires Qt.
# =====================================================================

# Floors mirroring LAYOUT.outline_* so the assertions are self-contained.
_MINW, _INITW, _MINH, _INITH = 180, 260, 120, 400


def _new_dock(win, object_name="dock_mesh_outline"):
    from qtpy import QtWidgets
    d = QtWidgets.QDockWidget("Outline", win)
    d.setObjectName(object_name)
    d.setWidget(QtWidgets.QWidget(win))
    return d


def _sanitize(win, dock):
    from apeGmsh.viewers.ui._dock_registry import sanitize_dock_placement
    sanitize_dock_placement(
        win, dock,
        min_width=_MINW, initial_width=_INITW,
        min_height=_MINH, initial_height=_INITH,
    )


def test_sanitize_heals_top_to_left(qapp):
    from qtpy import QtCore
    win = _make_window(qapp)
    dock = _new_dock(win)
    win.addDockWidget(QtCore.Qt.TopDockWidgetArea, dock)
    _sanitize(win, dock)
    assert win.dockWidgetArea(dock) == QtCore.Qt.LeftDockWidgetArea
    assert dock.isFloating() is False


def test_sanitize_heals_bottom_to_left(qapp):
    from qtpy import QtCore
    win = _make_window(qapp)
    dock = _new_dock(win)
    win.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
    _sanitize(win, dock)
    assert win.dockWidgetArea(dock) == QtCore.Qt.LeftDockWidgetArea


def test_sanitize_unfloats_to_left(qapp):
    """The original 50884941 floating symptom."""
    from qtpy import QtCore
    win = _make_window(qapp)
    dock = _new_dock(win)
    win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    dock.setFloating(True)
    assert dock.isFloating() is True
    _sanitize(win, dock)
    assert dock.isFloating() is False
    assert win.dockWidgetArea(dock) == QtCore.Qt.LeftDockWidgetArea


def test_sanitize_reasserts_features(qapp):
    """A corrupt state that stripped the drag handles is re-armed."""
    from qtpy import QtCore, QtWidgets
    win = _make_window(qapp)
    dock = _new_dock(win)
    win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
    _sanitize(win, dock)
    feats = dock.features()
    assert bool(feats & QtWidgets.QDockWidget.DockWidgetMovable)
    assert bool(feats & QtWidgets.QDockWidget.DockWidgetFloatable)


def test_sanitize_sets_both_floors(qapp):
    from qtpy import QtCore
    win = _make_window(qapp)
    dock = _new_dock(win)
    win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    _sanitize(win, dock)
    assert dock.minimumWidth() == _MINW
    assert dock.minimumHeight() == _MINH


def test_sanitize_heals_collapsed_height_to_min(qapp):
    """THE recurring bug: a Left, non-floating, full-width dock crushed
    vertically to a title-bar sliver. The sticky height floor reclaims
    space from the sibling WITHOUT tearing the dock out of its split."""
    from qtpy import QtCore, QtWidgets
    win = _make_window(qapp)
    win.resize(1200, 800)
    top = _new_dock(win, "dock_mesh_outline")
    bottom = QtWidgets.QDockWidget("Sib", win)
    bottom.setObjectName("dock_sibling")
    bottom.setWidget(QtWidgets.QWidget(win))
    win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, top)
    win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, bottom)
    win.splitDockWidget(top, bottom, QtCore.Qt.Vertical)
    win.show()
    qapp.processEvents()
    # Crush the bottom dock to a sliver.
    win.resizeDocks([top, bottom], [760, 30], QtCore.Qt.Vertical)
    qapp.processEvents()
    assert bottom.height() < _MINH  # genuinely crushed
    _sanitize(win, bottom)
    qapp.processEvents()
    assert bottom.height() >= _MINH          # reclaimed by the floor
    assert win.dockWidgetArea(bottom) == QtCore.Qt.LeftDockWidgetArea
    assert bottom.isFloating() is False      # NOT torn out of the split
    win.close()


def test_sanitize_preserves_healthy_left(qapp):
    """A healthy Left dock keeps its area; the sanitizer must not fire
    the re-dock branch."""
    from qtpy import QtCore
    win = _make_window(qapp)
    dock = _new_dock(win)
    win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    _sanitize(win, dock)
    assert win.dockWidgetArea(dock) == QtCore.Qt.LeftDockWidgetArea
    assert dock.isFloating() is False


def test_sanitize_preserves_healthy_right(qapp):
    """A user who docks the nav panel on the RIGHT keeps it there —
    Right is a legitimate vertical-tree area, only Top/Bottom/floating
    are degenerate."""
    from qtpy import QtCore
    win = _make_window(qapp)
    dock = _new_dock(win)
    win.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    _sanitize(win, dock)
    assert win.dockWidgetArea(dock) == QtCore.Qt.RightDockWidgetArea


def test_sanitize_preserves_user_width_on_healthy_left(qapp):
    """Plan-08 persistence: a healthy Left dock the user widened is NOT
    reset to the initial width (the re-dock/resize branch never fires)."""
    from qtpy import QtCore
    win = _make_window(qapp)
    win.resize(1200, 800)
    dock = _new_dock(win)
    win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    win.show()
    qapp.processEvents()
    win.resizeDocks([dock], [420], QtCore.Qt.Horizontal)
    qapp.processEvents()
    widened = dock.width()
    assert widened > _INITW
    _sanitize(win, dock)
    qapp.processEvents()
    # Width preserved (floor only clamps UP, and the dock is above it).
    assert dock.width() >= _INITW
    assert dock.width() == widened
    win.close()
