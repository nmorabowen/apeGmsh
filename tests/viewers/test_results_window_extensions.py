"""Tests for the module-level dock-extension helpers (plan 08 step 2).

These cover the logic that powers :class:`ResultsWindow`'s
``extension_docks=`` constructor argument, :meth:`add_extension_dock`
runtime API, and View-menu auto-population — but exercise the
underlying helpers (:func:`mount_dock_spec`, :func:`build_view_menu`,
:func:`add_view_menu_toggle`) directly against a vanilla ``QMainWindow``.

**Why not test ResultsWindow directly?** ResultsWindow's underlying
``QtInteractor`` allocates a VTK/OpenGL context. Constructing many
ResultsWindow instances in tests (one per test) accumulates GL state
that, after enough prior viewer-suite tests have run, can segfault on
Windows. The mount + view-menu logic is what we want to verify; it
lives in module-level helpers precisely so we don't pay the VTK cost
in tests. ResultsWindow's integration is exercised end-to-end every
time the viewer is launched manually.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.viewers.ui._dock_registry import (
    DockSpec,
    add_view_menu_toggle,
    build_view_menu,
    mount_dock_spec,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def main_window(qapp):
    """A minimal QMainWindow with a central widget — vanilla, no VTK."""
    from qtpy import QtWidgets
    win = QtWidgets.QMainWindow()
    win.setCentralWidget(QtWidgets.QWidget(win))
    try:
        yield win
    finally:
        win.deleteLater()


def _make_widget(parent=None):
    """Tiny dock content widget."""
    from qtpy import QtWidgets
    return QtWidgets.QLabel("ext", parent=parent)


# =====================================================================
# mount_dock_spec — the module helper that powers ResultsWindow's
# _mount_extension_dock and DockRegistry.mount
# =====================================================================


def test_mount_sets_object_name_and_title(main_window):
    spec = DockSpec(
        dock_id="ext_one",
        title="Ext One",
        factory=_make_widget,
    )
    dock = mount_dock_spec(main_window, spec)
    assert dock.objectName() == "ext_one"
    assert dock.windowTitle() == "Ext One"


def test_mount_applies_default_area(main_window):
    from qtpy import QtCore

    spec_left = DockSpec(
        dock_id="d_left", title="L", factory=_make_widget,
        default_area="left",
    )
    spec_bottom = DockSpec(
        dock_id="d_bot", title="B", factory=_make_widget,
        default_area="bottom",
    )
    d_left = mount_dock_spec(main_window, spec_left)
    d_bot = mount_dock_spec(main_window, spec_bottom)
    assert main_window.dockWidgetArea(d_left) == \
        QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
    assert main_window.dockWidgetArea(d_bot) == \
        QtCore.Qt.DockWidgetArea.BottomDockWidgetArea


def test_mount_applies_default_visibility(main_window):
    spec_hidden = DockSpec(
        dock_id="d_hidden", title="H", factory=_make_widget,
        default_visible=False,
    )
    spec_visible = DockSpec(
        dock_id="d_visible", title="V", factory=_make_widget,
        default_visible=True,
    )
    d_h = mount_dock_spec(main_window, spec_hidden)
    d_v = mount_dock_spec(main_window, spec_visible)
    main_window.show()
    try:
        assert not d_h.isVisible()
        assert d_v.isVisible()
    finally:
        main_window.hide()


def test_mount_rejects_reserved_id(main_window):
    spec = DockSpec("dup", "Dup", _make_widget)
    with pytest.raises(ValueError, match="Duplicate"):
        mount_dock_spec(main_window, spec, reserved_ids={"dup", "other"})


def test_mount_tabify_with_resolves_by_object_name(main_window):
    """When tabify_with is set, target is found by findChild on objectName."""
    first = mount_dock_spec(main_window, DockSpec(
        "first", "First", _make_widget,
    ))
    second = mount_dock_spec(main_window, DockSpec(
        "second", "Second", _make_widget,
        tabify_with="first",
    ))
    # Both end up in the same Qt tab group. (Outline of any window —
    # docks added to the default area with tabify_with set get
    # tabified.)
    assert first is not None
    assert second is not None
    # Lookup contract — findChild returns the named dock.
    from qtpy import QtWidgets
    assert main_window.findChild(QtWidgets.QDockWidget, "first") is first


def test_mount_tabify_with_unknown_id_raises(main_window):
    with pytest.raises(ValueError, match="not found"):
        mount_dock_spec(main_window, DockSpec(
            "orphan", "Orphan", _make_widget,
            tabify_with="never_registered",
        ))


def test_mount_factory_receives_window_as_parent(main_window):
    captured = []

    def factory(parent):
        captured.append(parent)
        from qtpy import QtWidgets
        return QtWidgets.QWidget(parent)

    mount_dock_spec(main_window, DockSpec("f", "F", factory))
    assert captured == [main_window]


# =====================================================================
# build_view_menu + add_view_menu_toggle
# =====================================================================


def test_view_menu_creates_one_toggle_per_dock(main_window):
    d1 = mount_dock_spec(main_window, DockSpec("a", "Alpha", _make_widget))
    d2 = mount_dock_spec(main_window, DockSpec("b", "Beta", _make_widget))
    d3 = mount_dock_spec(main_window, DockSpec("c", "Gamma", _make_widget))

    menu, _sep = build_view_menu(
        main_window.menuBar(),
        docks=[d1, d2, d3],
        on_reset_layout=lambda: None,
    )
    toggles = [a for a in menu.actions() if a.isCheckable()]
    assert len(toggles) == 3
    # Order matches docks argument.
    assert [a.text() for a in toggles] == ["Alpha", "Beta", "Gamma"]


def test_view_menu_includes_reset_layout(main_window):
    """Reset Layout appears after a separator and binds the callback."""
    d = mount_dock_spec(main_window, DockSpec("a", "A", _make_widget))
    fired = []
    menu, _sep = build_view_menu(
        main_window.menuBar(),
        docks=[d],
        on_reset_layout=lambda: fired.append(1),
    )
    reset = next(
        a for a in menu.actions()
        if not a.isCheckable() and not a.isSeparator() and a.text() == "Reset layout"
    )
    assert reset is not None
    reset.trigger()
    assert fired == [1]


def test_view_menu_reset_layout_disabled_when_no_callback(main_window):
    d = mount_dock_spec(main_window, DockSpec("a", "A", _make_widget))
    menu, _sep = build_view_menu(
        main_window.menuBar(),
        docks=[d],
        on_reset_layout=None,
    )
    reset = next(
        a for a in menu.actions()
        if not a.isCheckable() and not a.isSeparator() and a.text() == "Reset layout"
    )
    assert reset.isEnabled() is False


def test_view_menu_replaces_existing_menu_with_same_title(main_window):
    """Idempotent — calling twice keeps only the latest View menu."""
    d = mount_dock_spec(main_window, DockSpec("a", "A", _make_widget))
    build_view_menu(main_window.menuBar(), docks=[d], on_reset_layout=lambda: None)
    build_view_menu(main_window.menuBar(), docks=[d], on_reset_layout=lambda: None)

    view_menus = [a for a in main_window.menuBar().actions() if a.text() == "View"]
    assert len(view_menus) == 1


def test_view_menu_toggle_inverts_dock_visibility(main_window):
    """Triggering a toggle action flips the underlying dock's visibility."""
    d = mount_dock_spec(main_window, DockSpec("a", "A", _make_widget))
    main_window.show()
    try:
        menu, _sep = build_view_menu(
            main_window.menuBar(),
            docks=[d],
            on_reset_layout=lambda: None,
        )
        toggle = next(a for a in menu.actions() if a.isCheckable())
        initial = d.isVisible()
        toggle.trigger()
        assert d.isVisible() != initial
        toggle.trigger()
        assert d.isVisible() == initial
    finally:
        main_window.hide()


def test_add_view_menu_toggle_inserts_before_separator(main_window):
    """Later toggles go above the Reset Layout separator — Reset Layout
    stays pinned at the bottom."""
    d1 = mount_dock_spec(main_window, DockSpec("a", "A", _make_widget))
    menu, sep = build_view_menu(
        main_window.menuBar(),
        docks=[d1],
        on_reset_layout=lambda: None,
    )
    d2 = mount_dock_spec(main_window, DockSpec("b", "B", _make_widget))
    add_view_menu_toggle(menu, sep, d2)

    actions = menu.actions()
    # The Reset Layout action should be the LAST non-separator action.
    last_non_sep = next(
        a for a in reversed(actions) if not a.isSeparator()
    )
    assert last_non_sep.text() == "Reset layout"
    # Both toggles should appear above the separator.
    sep_idx = actions.index(sep)
    toggles_above = [a for a in actions[:sep_idx] if a.isCheckable()]
    assert [a.text() for a in toggles_above] == ["A", "B"]


# =====================================================================
# View menu — Focus mode / Camera / Theme (ADR 0087 Appendix B)
#
# Same no-VTK strategy as the rest of this file: ResultsWindow's menu
# construction is exercised by binding the REAL methods onto a stub
# (the ``test_toolbar_extensibility.py`` pattern), so drift in the
# production code surfaces here without paying the QtInteractor cost.
# =====================================================================


def _make_view_menu_stub(qapp):
    """A stub ResultsWindow with the real menu-building methods bound."""
    import types

    from qtpy import QtGui, QtWidgets

    from apeGmsh.viewers.ui._results_window import ResultsWindow
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    win = QtWidgets.QMainWindow()

    class _VWStub:
        pass

    vw = _VWStub()
    vw.window = win
    vw._window = win
    vw._QtWidgets = QtWidgets
    vw._theme_callbacks = []
    vw._act_parallel = QtGui.QAction(win)
    vw._act_parallel.setCheckable(True)
    vw._snap_view = lambda direction: None
    vw._fit_view = lambda: None
    for name in (
        "populate_camera_menu", "populate_theme_menu", "on_theme_changed",
    ):
        setattr(vw, name, types.MethodType(getattr(ViewerWindow, name), vw))

    class _RSStub:
        DOCK_OUTLINE = ResultsWindow.DOCK_OUTLINE
        DOCK_PLOTS = ResultsWindow.DOCK_PLOTS
        DOCK_DIAGRAM = ResultsWindow.DOCK_DIAGRAM
        DOCK_GEOMETRY = ResultsWindow.DOCK_GEOMETRY
        DOCK_DETAILS = ResultsWindow.DOCK_DETAILS
        DOCK_SESSION = ResultsWindow.DOCK_SESSION
        DOCK_SCRUBBER = ResultsWindow.DOCK_SCRUBBER

    rs = _RSStub()
    rs._vw = vw
    for attr in (
        "_dock_left", "_dock_right", "_dock_diagram", "_dock_geometry",
        "_dock_details", "_dock_session", "_dock_bottom",
    ):
        setattr(rs, attr, None)
    rs._extension_specs = []
    rs._extension_docks = {}
    rs._view_menu = None
    rs._view_menu_reset_separator = None
    rs._view_menu_theme_separator = None
    rs._focus_state = None
    rs._default_layout_state = None
    rs._act_focus_mode = None
    rs._act_close_window = None
    rs.set_status = lambda *a, **k: None
    for name in (
        "_all_docks", "_build_view_menu", "_install_focus_shortcut",
        "add_view_menu_submenu", "toggle_focus_mode", "_on_q_pressed",
        "reset_layout",
    ):
        setattr(rs, name, types.MethodType(getattr(ResultsWindow, name), rs))
    rs._install_focus_shortcut()
    rs._build_view_menu()
    return rs, win


def test_view_menu_has_focus_mode_camera_and_theme(qapp):
    """ADR 0087 Appendix B: the results View menu carries Focus mode
    (Ctrl+H rendered in the item), a Camera submenu wired with the
    seven presets + Fit view + Orthographic, and a radio-checked Theme
    submenu ending in Theme editor…."""
    rs, win = _make_view_menu_stub(qapp)
    try:
        actions = rs._view_menu.actions()
        texts = [a.text() for a in actions]
        assert "Focus mode" in texts
        focus = next(a for a in actions if a.text() == "Focus mode")
        assert focus.shortcut().toString() == "Ctrl+H"

        submenus = {a.text(): a.menu() for a in actions if a.menu()}
        assert "Camera" in submenus
        assert "Theme" in submenus

        cam_texts = [a.text() for a in submenus["Camera"].actions()]
        for label in (
            "Top", "Bottom", "Front", "Back", "Left", "Right",
            "Isometric", "Fit view", "Orthographic",
        ):
            assert label in cam_texts, f"Camera menu missing {label!r}"

        theme_actions = submenus["Theme"].actions()
        radios = [a for a in theme_actions if a.isCheckable()]
        # All ten built-ins present (plus any user themes).
        assert len(radios) >= 10
        # Exactly one radio checked — the current theme.
        assert sum(1 for a in radios if a.isChecked()) == 1
        assert theme_actions[-1].text() == "Theme editor…"
    finally:
        win.deleteLater()


def test_orbit_axis_submenu_inserts_between_camera_and_theme(qapp):
    """Viewer-added submenus (Orbit axis) land ABOVE the Theme section
    so the menu reads Camera / Orbit axis / Theme (Appendix B order)."""
    rs, win = _make_view_menu_stub(qapp)
    try:
        rs.add_view_menu_submenu("Orbit axis")
        submenu_order = [
            a.text() for a in rs._view_menu.actions() if a.menu()
        ]
        assert submenu_order == ["Camera", "Orbit axis", "Theme"]
    finally:
        win.deleteLater()


def test_close_window_action_owns_q_shortcut(qapp):
    """The window-level Close window action exists with shortcut Q —
    the File menu lists this same action rather than re-registering."""
    rs, win = _make_view_menu_stub(qapp)
    try:
        act = rs._act_close_window
        assert act is not None
        assert act.text() == "Close window"
        assert act.shortcut().toString() == "Q"
    finally:
        win.deleteLater()


# =====================================================================
# ResultsWindow integration — minimal sanity check that the helpers
# are wired up. This test DOES construct a ResultsWindow; isolated to
# a single call so VTK accumulation is minimal.
# =====================================================================


def test_results_window_imports_extension_api(qapp):
    """ResultsWindow exposes the extension API surface promised in the
    plan-doc — DOCK_* constants, extension_docks param, add_extension_dock
    method. Verifies the API contract without constructing the window."""
    from apeGmsh.viewers.ui._results_window import ResultsWindow

    # The class-level constants exist and are distinct strings.
    object_names = {
        ResultsWindow.DOCK_OUTLINE,
        ResultsWindow.DOCK_PLOTS,
        ResultsWindow.DOCK_DIAGRAM,
        ResultsWindow.DOCK_GEOMETRY,
        ResultsWindow.DOCK_DETAILS,
        ResultsWindow.DOCK_SESSION,
        ResultsWindow.DOCK_SCRUBBER,
    }
    assert len(object_names) == 7

    # extension_docks accepted in __init__ signature.
    import inspect
    sig = inspect.signature(ResultsWindow.__init__)
    assert "extension_docks" in sig.parameters

    # Public methods exist.
    assert hasattr(ResultsWindow, "add_extension_dock")
    assert hasattr(ResultsWindow, "extension_dock")
