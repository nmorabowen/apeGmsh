"""Plan 08 — ViewerWindow extension docks (mesh/model viewer rollout).

Verifies the ``extension_docks`` constructor argument,
:meth:`ViewerWindow.add_extension_dock` runtime API, and View-menu
auto-population — mirroring :mod:`test_results_window_extensions`
but stub-binding the new methods so we don't pay the VTK
``QtInteractor`` overhead in tests.

The module-level helpers (``mount_dock_spec`` /
``add_view_menu_toggle``) are already covered by the ResultsWindow
test file; here we only check that ViewerWindow's class-level API
surface is present and that the per-instance methods route through
those helpers as expected.
"""
from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# =====================================================================
# API contract — class-level constants + constructor signature
# =====================================================================


def test_viewer_window_exposes_dock_id_constants():
    """``ViewerWindow.DOCK_TABS`` / ``DOCK_CONSOLE`` exist as strings so
    extension specs can ``tabify_with`` them by name without reaching
    into private state."""
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    assert isinstance(ViewerWindow.DOCK_TABS, str)
    assert isinstance(ViewerWindow.DOCK_CONSOLE, str)
    assert ViewerWindow.DOCK_TABS != ViewerWindow.DOCK_CONSOLE


def test_viewer_window_constructor_accepts_extension_docks():
    """``__init__`` signature exposes the keyword argument."""
    import inspect
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    sig = inspect.signature(ViewerWindow.__init__)
    assert "extension_docks" in sig.parameters


def test_viewer_window_exposes_extension_dock_methods():
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    assert hasattr(ViewerWindow, "add_extension_dock")
    assert hasattr(ViewerWindow, "extension_dock")
    assert hasattr(ViewerWindow, "_mount_extension_dock_inner")


def test_builtin_dock_ids_set_contains_constants():
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    assert ViewerWindow.DOCK_TABS in ViewerWindow._BUILTIN_DOCK_IDS
    assert ViewerWindow.DOCK_CONSOLE in ViewerWindow._BUILTIN_DOCK_IDS


# =====================================================================
# Stub-bound method tests — exercise the logic without VTK
# =====================================================================


def _make_stub_window(qapp):
    """Stand-in window exposing the extension-dock API without
    constructing a full ``ViewerWindow``.

    ViewerWindow's ``__init__`` builds a PyVista ``QtInteractor`` for
    the central widget; running many of those in one test session
    accumulates OpenGL state and risks segfaults on Windows. The
    extension-dock logic only needs a ``QMainWindow``, the extension
    state dicts, and the bound methods. Same pattern as
    :mod:`test_toolbar_extensibility`.
    """
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    class _Stub:
        pass

    stub = _Stub()
    stub._window = QtWidgets.QMainWindow()
    stub._window.setCentralWidget(QtWidgets.QWidget(stub._window))
    stub._extension_specs = []
    stub._extension_docks = {}
    stub._view_menu = None
    stub._view_menu_reset_separator = None
    # ``_window_key=None`` selects the legacy non-persistent path so the
    # menu's Reset Layout entry is disabled. Tests that exercise the
    # persistent path override this attribute on the returned stub.
    stub._window_key = None
    # Mirror class attributes the bound methods read via ``self``.
    stub._BUILTIN_DOCK_IDS = ViewerWindow._BUILTIN_DOCK_IDS

    # Bind the real methods so any drift in production code surfaces
    # in the tests.
    for name in (
        "add_extension_dock", "_mount_extension_dock_inner",
        "extension_dock", "_ensure_view_menu",
    ):
        method = getattr(ViewerWindow, name)
        setattr(stub, name, method.__get__(stub))

    return stub


def test_add_extension_dock_mounts_and_returns_qdockwidget(qapp):
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui._dock_registry import DockSpec

    stub = _make_stub_window(qapp)
    spec = DockSpec(
        dock_id="ext_one",
        title="Ext One",
        factory=lambda parent: QtWidgets.QLabel("ext", parent=parent),
    )
    dock = stub.add_extension_dock(spec)

    assert isinstance(dock, QtWidgets.QDockWidget)
    assert dock.objectName() == "ext_one"
    assert "ext_one" in stub._extension_docks
    assert spec in stub._extension_specs


def test_add_extension_dock_rejects_builtin_collision(qapp):
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui._dock_registry import DockSpec
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    stub = _make_stub_window(qapp)
    spec = DockSpec(
        dock_id=ViewerWindow.DOCK_TABS,
        title="Conflict",
        factory=lambda parent: QtWidgets.QLabel("x", parent=parent),
    )
    with pytest.raises(ValueError, match="built-in"):
        stub.add_extension_dock(spec)


def test_add_extension_dock_rejects_duplicate(qapp):
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui._dock_registry import DockSpec

    stub = _make_stub_window(qapp)
    factory = lambda parent: QtWidgets.QLabel("x", parent=parent)
    stub.add_extension_dock(DockSpec("dup", "Dup", factory))
    with pytest.raises(ValueError, match="Duplicate"):
        stub.add_extension_dock(DockSpec("dup", "Other", factory))


def test_extension_dock_lookup_returns_mounted_dock(qapp):
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui._dock_registry import DockSpec

    stub = _make_stub_window(qapp)
    factory = lambda parent: QtWidgets.QLabel("x", parent=parent)
    dock = stub.add_extension_dock(DockSpec("lookup_me", "L", factory))
    assert stub.extension_dock("lookup_me") is dock


def test_extension_dock_unknown_id_raises_key_error(qapp):
    stub = _make_stub_window(qapp)
    with pytest.raises(KeyError):
        stub.extension_dock("nope")


def test_add_extension_dock_lazily_builds_view_menu(qapp):
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui._dock_registry import DockSpec

    stub = _make_stub_window(qapp)
    assert stub._view_menu is None

    factory = lambda parent: QtWidgets.QLabel("x", parent=parent)
    stub.add_extension_dock(DockSpec("first_ext", "First", factory))

    assert stub._view_menu is not None
    # Exactly one checkable toggle, matching the dock's title.
    toggles = [a for a in stub._view_menu.actions() if a.isCheckable()]
    assert len(toggles) == 1
    assert toggles[0].text() == "First"


def test_add_extension_dock_appends_to_existing_view_menu(qapp):
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui._dock_registry import DockSpec

    stub = _make_stub_window(qapp)
    factory = lambda parent: QtWidgets.QLabel("x", parent=parent)
    stub.add_extension_dock(DockSpec("a", "Alpha", factory))
    stub.add_extension_dock(DockSpec("b", "Beta", factory))

    toggles = [a.text() for a in stub._view_menu.actions() if a.isCheckable()]
    assert toggles == ["Alpha", "Beta"]


def test_tabify_with_resolves_against_dock_tabs_constant(qapp):
    """An extension spec can ``tabify_with`` ``ViewerWindow.DOCK_TABS``
    when the tabs dock has been mounted under that objectName."""
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui._dock_registry import DockSpec
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    stub = _make_stub_window(qapp)
    # Simulate the tabs dock being on the window with the public
    # objectName constant.
    tabs_dock = QtWidgets.QDockWidget("Tabs", stub._window)
    tabs_dock.setObjectName(ViewerWindow.DOCK_TABS)
    stub._window.addDockWidget(
        QtWidgets.Qt.RightDockWidgetArea
        if hasattr(QtWidgets, "Qt") else None,
        tabs_dock,
    ) if False else None    # placeholder — actual add below
    from qtpy import QtCore
    stub._window.addDockWidget(
        QtCore.Qt.RightDockWidgetArea, tabs_dock,
    )

    factory = lambda parent: QtWidgets.QLabel("x", parent=parent)
    dock = stub.add_extension_dock(DockSpec(
        "tabified", "T", factory, tabify_with=ViewerWindow.DOCK_TABS,
    ))
    assert dock.objectName() == "tabified"
