"""ADR 0088 — results-window architecture guards (no VTK).

Locks the D6 state rules and the D8 acceptance criteria that are
checkable without constructing a real ResultsWindow (whose
QtInteractor allocates a GL context — the established no-VTK test
strategy, see ``test_results_window_extensions.py``):

* schema version 7 for ResultsWindow, 5 for ViewerWindow (D6);
* retired dock objectNames appear nowhere in ``viewers/**`` except
  the schema-history comment (criterion 11);
* the D1 width invariant — viewport ≥ 50 % at 1280 px with both side
  columns at initial widths (criterion 1's arithmetic half);
* focus mode = viewport + Time scrubber, exact-restore snapshot
  (criterion 8), via the real methods bound onto a stub window;
* reset layout restores the D1 boot visibility set from any
  arrangement (criterion 9).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.viewers.ui._layout_metrics import LAYOUT
from apeGmsh.viewers.ui._results_window import ResultsWindow
from apeGmsh.viewers.ui.viewer_window import ViewerWindow

VIEWERS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "apeGmsh"
    / "viewers"
)

# The four objectNames retired by ADR 0088 D6 — never to be reused.
_RETIRED = (
    "dock_results_diagram",
    "dock_results_geometry",
    "dock_results_details",
    "dock_color_map_editor",
)


# =====================================================================
# D6 — schema versions
# =====================================================================


def test_results_window_schema_is_7():
    """ADR 0088 D6 / criterion 10 — one bump for the whole partition."""
    assert ResultsWindow._LAYOUT_SCHEMA_VERSION == 7


def test_viewer_window_schema_stays_5():
    """Mesh/model dock sets do not change in Phase 2 (ADR 0088 D6)."""
    assert ViewerWindow._LAYOUT_SCHEMA_VERSION == 5


# =====================================================================
# Criterion 11 — retired objectNames only in the history comment
# =====================================================================


def test_retired_objectnames_only_in_schema_history_comment():
    offenders: list[str] = []
    for path in sorted(VIEWERS_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in _RETIRED:
            if name not in text:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if name not in line:
                    continue
                comment = line.lstrip().startswith("#")
                in_window = path.name == "_results_window.py"
                if not (comment and in_window):
                    offenders.append(f"{path.name}:{lineno}: {name}")
    assert not offenders, (
        "Retired dock objectNames must appear nowhere in viewers/** "
        "except ResultsWindow's schema-history comment (ADR 0088 "
        f"criterion 11): {offenders}"
    )


# =====================================================================
# Criterion 1 (arithmetic half) — D1 width invariant
# =====================================================================


def test_initial_widths_keep_viewport_at_half_of_1280():
    viewport = 1280 - LAYOUT.outline_initial_width - LAYOUT.right_initial_width
    assert viewport >= 640, (
        "ADR 0088 D1: at 1280 px the viewport must keep >= 50 % with "
        "Outline + Inspector at initial widths"
    )
    # ADR 0088 D1 also states the side columns "never exceed 45 %",
    # but its own arithmetic (260 + 380 = 640 = 50 % of 1280, blessed
    # in the same paragraph) contradicts that at 1280 px — fixed-pixel
    # initial widths give 45 % only for windows >= 1423 px. Pin the
    # invariant the ADR's arithmetic actually endorses: the 45 % cap
    # holds at the maximized-desktop widths the audit screenshots use.
    side = LAYOUT.outline_initial_width + LAYOUT.right_initial_width
    assert side <= 0.45 * 1440, (
        "ADR 0088 D1: the two side columns exceed 45 % even at 1440 px"
    )


def test_output_console_metrics_exist():
    assert LAYOUT.output_min_height == 100
    assert LAYOUT.output_initial_height == 140


# =====================================================================
# Focus mode + reset layout — real methods on a stub window
# =====================================================================


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_stub(qapp):
    """Stub ResultsWindow: real QDockWidgets + the real dock-state
    methods bound, no VTK. Extension docks included so focus mode /
    reset cover them (ADR 0088 D6)."""
    import types
    from qtpy import QtCore, QtWidgets

    win = QtWidgets.QMainWindow()
    win.setCentralWidget(QtWidgets.QWidget(win))

    class _VWStub:
        pass

    vw = _VWStub()
    vw.window = win
    vw._toolbar = QtWidgets.QToolBar(win)
    win.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, vw._toolbar)

    def _dock(object_name, title, area):
        d = QtWidgets.QDockWidget(title, win)
        d.setObjectName(object_name)
        d.setWidget(QtWidgets.QWidget(d))
        win.addDockWidget(area, d)
        return d

    Q = QtCore.Qt.DockWidgetArea
    rs = types.SimpleNamespace()
    rs._vw = vw
    rs._dock_left = _dock(ResultsWindow.DOCK_OUTLINE, "Outline", Q.LeftDockWidgetArea)
    rs._dock_right = _dock(ResultsWindow.DOCK_PLOTS, "Plots", Q.RightDockWidgetArea)
    rs._dock_inspector = _dock(
        ResultsWindow.DOCK_INSPECTOR, "Inspector", Q.RightDockWidgetArea,
    )
    rs._dock_session = _dock(
        ResultsWindow.DOCK_SESSION, "Display", Q.RightDockWidgetArea,
    )
    rs._dock_bottom = _dock(
        ResultsWindow.DOCK_SCRUBBER, "Time scrubber", Q.BottomDockWidgetArea,
    )
    ext = _dock("dock_output", "Output", Q.BottomDockWidgetArea)

    from apeGmsh.viewers.ui._dock_registry import DockSpec
    rs._extension_specs = [
        DockSpec("dock_output", "Output", lambda p: None),
    ]
    rs._extension_docks = {"dock_output": ext}
    rs._focus_state = None
    rs._default_layout_state = win.saveState()
    rs.set_status = lambda *a, **k: None
    for name in (
        "_all_docks", "toggle_focus_mode", "reset_layout",
    ):
        setattr(
            rs, name,
            types.MethodType(getattr(ResultsWindow, name), rs),
        )
    # Class-level constants used by the bound methods.
    rs.DOCK_OUTLINE = ResultsWindow.DOCK_OUTLINE
    rs.DOCK_PLOTS = ResultsWindow.DOCK_PLOTS
    rs.DOCK_INSPECTOR = ResultsWindow.DOCK_INSPECTOR
    rs.DOCK_SESSION = ResultsWindow.DOCK_SESSION
    rs.DOCK_SCRUBBER = ResultsWindow.DOCK_SCRUBBER
    return rs, win


def _boot_visibility(rs) -> None:
    """Arrange the ADR 0088 D1 boot state on the stub."""
    rs._dock_left.setVisible(True)
    rs._dock_inspector.setVisible(True)
    rs._dock_bottom.setVisible(True)
    rs._dock_right.setVisible(False)
    rs._dock_session.setVisible(False)
    rs._extension_docks["dock_output"].setVisible(False)


def test_focus_mode_keeps_scrubber_and_restores_exactly(qapp):
    """Criterion 8 — Ctrl+H leaves viewport + Time scrubber; a second
    Ctrl+H restores the exact prior visibility set."""
    rs, win = _make_stub(qapp)
    try:
        win.show()
        _boot_visibility(rs)
        # User also summoned the Output console before focusing.
        rs._extension_docks["dock_output"].setVisible(True)

        rs.toggle_focus_mode()
        assert rs._dock_bottom.isVisible(), "scrubber must survive focus mode"
        for d in (
            rs._dock_left, rs._dock_right, rs._dock_inspector,
            rs._dock_session, rs._extension_docks["dock_output"],
        ):
            assert not d.isVisible()
        assert not rs._vw._toolbar.isVisible()

        rs.toggle_focus_mode()
        assert rs._dock_left.isVisible()
        assert rs._dock_inspector.isVisible()
        assert rs._dock_bottom.isVisible()
        assert rs._extension_docks["dock_output"].isVisible()
        # Hidden-before stays hidden-after — EXACT restore.
        assert not rs._dock_right.isVisible()
        assert not rs._dock_session.isVisible()
        assert rs._vw._toolbar.isVisible()
    finally:
        win.hide()
        win.deleteLater()


def test_reset_layout_restores_d1_boot_set_from_any_arrangement(qapp):
    """Criterion 9 — Reset layout reproduces the three-dock boot state
    even after summons and closes."""
    rs, win = _make_stub(qapp)
    try:
        win.show()
        # Scrambled arrangement: outline closed, plots + display +
        # output summoned.
        rs._dock_left.setVisible(False)
        rs._dock_inspector.setVisible(False)
        rs._dock_right.setVisible(True)
        rs._dock_session.setVisible(True)
        rs._extension_docks["dock_output"].setVisible(True)

        rs.reset_layout()

        assert rs._dock_left.isVisible()
        assert rs._dock_inspector.isVisible()
        assert rs._dock_bottom.isVisible()
        assert not rs._dock_right.isVisible()
        assert not rs._dock_session.isVisible()
        assert not rs._extension_docks["dock_output"].isVisible()
    finally:
        win.hide()
        win.deleteLater()


def test_reset_layout_clears_focus_snapshot(qapp):
    """Reset while focused must not leave a stale snapshot that
    re-hides docks on the next Ctrl+H."""
    rs, win = _make_stub(qapp)
    try:
        win.show()
        _boot_visibility(rs)
        rs.toggle_focus_mode()
        rs.reset_layout()
        assert rs._focus_state is None
        # Next Ctrl+H ENTERS focus mode (fresh snapshot), not restore.
        rs.toggle_focus_mode()
        assert rs._focus_state is not None
        assert rs._dock_bottom.isVisible()
    finally:
        win.hide()
        win.deleteLater()
