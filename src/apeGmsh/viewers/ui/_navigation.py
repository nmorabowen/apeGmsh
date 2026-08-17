"""Viewport mouse-navigation styles — one mapping for every window.

The owner found the stock VTK trackball awkward; the product's own
convention is apeCAD's (``apeCAD/scratchpad/static/app.js``:
``mouseButtons.LEFT = -1; MIDDLE = ROTATE; RIGHT = PAN``): the LEFT
button belongs to *selection*, never to the camera — orbit rides the
MIDDLE button, pan the RIGHT, zoom the wheel. That is also the
friendliest mapping for the results pick loop (ADR 0098 S4): a left
drag can become a rubber-band window without fighting the camera.

Three conventions ship, keyed by the ``mouse_navigation`` preference
(:data:`~.preferences_manager.NAVIGATION_CHOICES`):

========  =========================  ==================  ============
token     left                       middle              right
========  =========================  ==================  ============
apecad    — (selection only)         orbit               pan
gmsh      orbit                      zoom-drag           pan
vtk       orbit                      pan                 zoom-drag
========  =========================  ==================  ============

The wheel zooms in all three. ``"vtk"`` is the stock
``vtkInteractorStyleTrackballCamera`` (the pre-2026-08 behaviour) and
installs no observers.

Mechanism: a fresh ``InteractorStyleTrackballCamera`` with per-button
observers, the same idiom pyvista's ``enable_custom_trackball_style``
uses — VTK invokes a style's observer INSTEAD of its built-in button
handler, so an empty observer is an unbound button (which the pyvista
helper cannot express; its action vocabulary has no "none"). The
press callbacks are state-guarded exactly like pyvista's: ``StartX``
engages only from the NONE state, and the built-in ``OnXButtonDown``
still runs afterwards for its ``FindPokedRenderer`` bookkeeping.

Applied per window at ``ViewerWindow`` construction (both results
windows, the mesh and the model viewer all wrap it) and re-applied
live by the View → Navigation submenu.
"""
from __future__ import annotations

from typing import Any, Optional

#: token → per-button camera action (``None`` = unbound; modifiers do
#: not change the action). ``"vtk"`` maps to no table at all: the
#: stock trackball keeps its own richer modifier behaviour
#: (ctrl-left spin, shift-left pan).
MAPPINGS: dict[str, "Optional[dict[str, Optional[str]]]"] = {
    "apecad": {"left": None, "middle": "rotate", "right": "pan"},
    "gmsh": {"left": "rotate", "middle": "dolly", "right": "pan"},
    "vtk": None,
}


def apply_navigation_style(plotter: Any, token: str) -> None:
    """Apply the ``token`` mouse mapping to ``plotter``'s interactor.

    ``plotter`` is any pyvista plotter (the ``QtInteractor`` included).
    Idempotent and live: each call installs a FRESH interactor style,
    so switching conventions mid-session never stacks observers.
    Raises ``KeyError`` on an unknown token — the preference vocabulary
    is closed (``NAVIGATION_CHOICES``).
    """
    mapping = MAPPINGS[token]
    # Always start from a FRESH stock trackball via the public API
    # (``enable_trackball_style`` constructs a new style each call) —
    # that is both the "vtk" convention verbatim and the idempotence
    # guarantee for live switching, and it keeps this ui/ module free
    # of pyvista imports (ADR 0056 INV-5 G-IMPORT guard).
    plotter.enable_trackball_style()
    if mapping is None:
        return

    style = plotter.iren.style
    # The pyvista style wrapper's CaptureMixin pre-registers LEFT
    # observers that call OnLeftButtonDown unconditionally
    # (multi-viewport capture — see its docstring). VTK runs ALL
    # observers of an event, so those would rotate the camera
    # underneath any left mapping (including "unbound"). Our windows
    # are single-viewport; drop them and let the table below own
    # every button.
    style.RemoveObservers("LeftButtonPressEvent")
    style.RemoveObservers("LeftButtonReleaseEvent")

    start_map = {
        "rotate": style.StartRotate,
        "pan": style.StartPan,
        "dolly": style.StartDolly,
        "spin": style.StartSpin,
    }
    end_map = (
        style.EndRotate, style.EndPan, style.EndDolly, style.EndSpin,
    )
    button_down = {
        "left": style.OnLeftButtonDown,
        "middle": style.OnMiddleButtonDown,
        "right": style.OnRightButtonDown,
    }
    button_up = {
        "left": style.OnLeftButtonUp,
        "middle": style.OnMiddleButtonUp,
        "right": style.OnRightButtonUp,
    }

    def _noop(_obj: Any, _event: Any) -> None:
        return None

    for button, action in mapping.items():
        event = f"{button.capitalize()}ButtonPressEvent"
        release_event = f"{button.capitalize()}ButtonReleaseEvent"
        if action is None:
            # Unbound: the observer's presence alone suppresses the
            # built-in handler; selection machinery listens on the
            # interactor, not the style, and is untouched.
            style.add_observer(event, _noop)
            style.add_observer(release_event, _noop)
            continue

        def _press(
            _obj: Any, _event: Any,
            _start=start_map[action], _down=button_down[button],
        ) -> None:
            # StartX engages only from the NONE state; the built-in
            # button-down that follows finds the poked renderer and
            # its own StartY is then a state-guarded no-op.
            _start()
            _down()

        def _release(
            _obj: Any, _event: Any, _up=button_up[button],
        ) -> None:
            # Every EndX is state-guarded — only the active one acts.
            for end in end_map:
                end()
            _up()

        style.add_observer(event, _press)
        style.add_observer(release_event, _release)


__all__ = ["MAPPINGS", "apply_navigation_style"]
