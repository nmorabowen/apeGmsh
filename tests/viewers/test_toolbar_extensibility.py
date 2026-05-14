"""Plan 02 — toolbar extensibility hook tests.

Verifies :meth:`ViewerWindow.add_toolbar_action` /
:meth:`remove_toolbar_action` — the public surface that lets
diagrams and overlays register their own buttons at runtime.

Tests construct a real :class:`ViewerWindow` (no pyvista plotter
needed for toolbar wiring) and exercise the action lifecycle:
add, fire callback, toggle checked state, remove.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtCore")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_stub_window(qapp):
    """Stand-in window that exposes the toolbar API without
    constructing a full ``ViewerWindow``.

    Why: ``ViewerWindow.__init__`` builds a real PyVista
    ``QtInteractor`` to host the VTK plotter. Running the toolbar
    tests after many VTK-heavy tests in the same process exhausts the
    OpenGL context allocator and crashes (Windows access violation).
    The toolbar code under test only needs a ``QToolBar``, an
    ``_icon_actions`` list, and the ``_make_icon`` painter — no VTK.

    Bind the real methods from ``ViewerWindow`` onto this stub so any
    drift in the production code surfaces in the tests.
    """
    from qtpy import QtWidgets, QtGui, QtCore
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    class _Stub:
        pass

    stub = _Stub()
    stub._QtGui = QtGui
    stub._QtCore = QtCore
    stub._toolbar = QtWidgets.QToolBar()
    stub._icon_actions = []
    # Bind unbound functions to the stub instance.
    import types
    for name in (
        "_make_icon", "_add_toolbar_action",
        "add_toolbar_action", "add_toolbar_button",
        "remove_toolbar_action",
    ):
        fn = getattr(ViewerWindow, name)
        setattr(stub, name, types.MethodType(fn, stub))
    return stub


@pytest.fixture
def window(qapp):
    """Fresh toolbar stub per test."""
    return _make_stub_window(qapp)


# =====================================================================
# add_toolbar_action — basic API
# =====================================================================


def test_add_toolbar_action_returns_qaction(window):
    from qtpy import QtGui
    action = window.add_toolbar_action(
        "Custom button", "X", lambda: None,
    )
    assert isinstance(action, QtGui.QAction)


def test_add_toolbar_action_appends_to_toolbar(window):
    """The new action is reachable via the toolbar's actions list."""
    before = list(window._toolbar.actions())
    action = window.add_toolbar_action(
        "Custom button", "X", lambda: None,
    )
    after = list(window._toolbar.actions())
    assert len(after) == len(before) + 1
    assert action in after


def test_add_toolbar_action_callback_fires_on_trigger(window):
    seen: list = []
    action = window.add_toolbar_action(
        "Custom", "X", lambda: seen.append("clicked"),
    )
    action.trigger()
    assert seen == ["clicked"]


def test_add_toolbar_action_tooltip_set(window):
    action = window.add_toolbar_action(
        "Hello tooltip", "X", lambda: None,
    )
    assert action.toolTip() == "Hello tooltip"


# =====================================================================
# Checkable / toggle behavior
# =====================================================================


def test_checkable_toggle_action(window):
    """``checkable=True`` + ``triggered_signal='toggled'`` delivers the
    new bool state to the callback."""
    states: list = []
    action = window.add_toolbar_action(
        "Toggle thing", "T",
        lambda checked: states.append(bool(checked)),
        checkable=True,
        triggered_signal="toggled",
    )
    assert action.isCheckable()
    action.setChecked(True)
    action.setChecked(False)
    assert states == [True, False]


def test_non_checkable_default(window):
    action = window.add_toolbar_action(
        "Momentary", "M", lambda: None,
    )
    assert not action.isCheckable()


# =====================================================================
# remove_toolbar_action
# =====================================================================


def test_remove_toolbar_action_drops_from_toolbar(window):
    action = window.add_toolbar_action(
        "Custom", "X", lambda: None,
    )
    assert action in window._toolbar.actions()
    window.remove_toolbar_action(action)
    assert action not in window._toolbar.actions()


def test_remove_toolbar_action_unregisters_from_theme_refresh(window):
    """After removal, the action must be absent from
    ``_icon_actions`` so a later palette change doesn't poke it."""
    action = window.add_toolbar_action(
        "Custom", "X", lambda: None,
    )
    assert any(a is action for (a, _) in window._icon_actions)
    window.remove_toolbar_action(action)
    assert not any(a is action for (a, _) in window._icon_actions)


def test_remove_toolbar_action_idempotent(window):
    action = window.add_toolbar_action(
        "Custom", "X", lambda: None,
    )
    window.remove_toolbar_action(action)
    # Second call is a no-op — must not raise.
    window.remove_toolbar_action(action)


def test_remove_toolbar_action_none_is_noop(window):
    # Passing None is documented as a no-op.
    window.remove_toolbar_action(None)


# =====================================================================
# Legacy add_toolbar_button still works
# =====================================================================


def test_legacy_add_toolbar_button_appends(window):
    """The pre-existing ``add_toolbar_button`` keeps its no-return
    contract; the action goes onto the toolbar via the new internal
    path."""
    before = len(window._toolbar.actions())
    window.add_toolbar_button("Legacy", "L", lambda: None)
    after = len(window._toolbar.actions())
    assert after == before + 1
