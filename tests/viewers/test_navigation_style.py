"""Viewport mouse-navigation styles — mapping + preference + menu.

The behavioural oracle synthesizes real button events on an offscreen
plotter's interactor and reads the interactor style's STATE — the
thing a drag would act on. Expected values are SELF-CALIBRATED from
the stock VTK trackball in the same process (rotate = what stock
left does, pan = stock middle, dolly = stock right), so the tests
survive a VTK renumbering of its state constants.

apeCAD convention (the default): LEFT unbound — selection only —
MIDDLE orbits, RIGHT pans. The unbound left is the assertion that
matters most: the pyvista wrapper's capture mixin pre-registers a
left observer that rotates regardless of mapping, and these tests
would catch it regressing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pyvista")

import pyvista as pv

from apeGmsh.viewers.ui._navigation import MAPPINGS, apply_navigation_style
from apeGmsh.viewers.ui.preferences_manager import (
    NAVIGATION_CHOICES,
    Preferences,
)


def _plotter():
    try:
        p = pv.Plotter(off_screen=True)
    except Exception:  # pragma: no cover
        pytest.skip("no offscreen render context")
    p.add_mesh(pv.Cube())
    p.render()
    return p


def _press_state(plotter, button: str) -> int:
    """Synthesize a ``button`` press, read the style state, release."""
    iren = plotter.iren.interactor
    iren.SetEventInformation(5, 5)
    iren.InvokeEvent(f"{button}ButtonPressEvent")
    state = plotter.iren.style.GetState()
    iren.InvokeEvent(f"{button}ButtonReleaseEvent")
    return state


@pytest.fixture(scope="module")
def stock_states():
    """(idle, rotate, pan, dolly) as THIS VTK build numbers them."""
    p = _plotter()
    apply_navigation_style(p, "vtk")
    idle = p.iren.style.GetState()
    rotate = _press_state(p, "Left")
    pan = _press_state(p, "Middle")
    dolly = _press_state(p, "Right")
    p.close()
    assert len({idle, rotate, pan, dolly}) == 4  # oracle sanity
    return idle, rotate, pan, dolly


# =====================================================================
# The three conventions, behaviourally
# =====================================================================


def test_apecad_left_is_unbound_middle_orbits_right_pans(stock_states):
    idle, rotate, pan, _dolly = stock_states
    p = _plotter()
    apply_navigation_style(p, "apecad")
    try:
        assert _press_state(p, "Left") == idle      # selection only
        assert _press_state(p, "Middle") == rotate  # orbit
        assert _press_state(p, "Right") == pan      # pan
        assert p.iren.style.GetState() == idle      # releases restore
    finally:
        p.close()


def test_gmsh_left_orbits_middle_zooms_right_pans(stock_states):
    idle, rotate, pan, dolly = stock_states
    p = _plotter()
    apply_navigation_style(p, "gmsh")
    try:
        assert _press_state(p, "Left") == rotate
        assert _press_state(p, "Middle") == dolly
        assert _press_state(p, "Right") == pan
        assert p.iren.style.GetState() == idle
    finally:
        p.close()


def test_vtk_token_restores_the_stock_trackball(stock_states):
    _idle, rotate, pan, dolly = stock_states
    p = _plotter()
    apply_navigation_style(p, "apecad")
    apply_navigation_style(p, "vtk")  # live switch back
    try:
        assert _press_state(p, "Left") == rotate
        assert _press_state(p, "Middle") == pan
        assert _press_state(p, "Right") == dolly
    finally:
        p.close()


def test_reapply_is_idempotent_never_stacks(stock_states):
    """Live switching installs a FRESH style each call — N switches
    must not stack observers into double-started actions."""
    idle, rotate, _pan, dolly = stock_states
    p = _plotter()
    for _ in range(3):
        apply_navigation_style(p, "gmsh")
        apply_navigation_style(p, "apecad")
    try:
        assert _press_state(p, "Left") == idle
        assert _press_state(p, "Middle") == rotate
    finally:
        p.close()


def test_unknown_token_refuses_loudly():
    p = _plotter()
    try:
        with pytest.raises(KeyError):
            apply_navigation_style(p, "blender")
    finally:
        p.close()


# =====================================================================
# Preference vocabulary
# =====================================================================


def test_preference_default_is_apecad():
    assert Preferences().mouse_navigation == "apecad"


def test_choices_and_mappings_agree():
    assert set(NAVIGATION_CHOICES) == set(MAPPINGS)


# =====================================================================
# qt lane — the live View → Navigation switch on a real window
# =====================================================================


@pytest.mark.qt
def test_viewer_window_navigation_menu_switches_live(tmp_path, monkeypatch):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from apeGmsh.viewers.ui import preferences_manager as pm
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    # Isolate preferences: the menu PERSISTS its choice, and a test
    # must never write the user's real preferences.json.
    monkeypatch.setattr(
        pm, "PREFERENCES",
        pm.PreferencesManager(path=tmp_path / "preferences.json"),
    )

    win = ViewerWindow(title="nav-smoke")
    try:
        assert win.navigation_style == "apecad"  # the boot preference
        menu = win.install_navigation_menu()
        actions = {a.text(): a for a in menu.actions()}
        gmsh_action = next(
            a for text, a in actions.items() if text.startswith("Gmsh")
        )
        gmsh_action.trigger()
        assert win.navigation_style == "gmsh"
        assert pm.PREFERENCES.current.mouse_navigation == "gmsh"
        # The apeCAD entry is exclusive-radio unchecked now.
        apecad_action = next(
            a for text, a in actions.items() if text.startswith("apeCAD")
        )
        assert not apecad_action.isChecked()
        assert gmsh_action.isChecked()
    finally:
        win.window.close()
        qapp.processEvents()
