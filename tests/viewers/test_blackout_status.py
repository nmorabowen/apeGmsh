"""The "all elements hidden" status toast (blackout hint).

Every hide path — threshold empty range, scope box outside the mesh,
dim filter, stage mask, manual hide — funnels through
``ElementVisibility``, so one payload-independent subscriber on
``ELEMENT_VISIBILITY_CHANGED`` can explain any silently empty
viewport. The transition truth table lives in the module-level
``results_viewer._sync_blackout_hints`` (same headless-testable
pattern as ``_geometries_occluded_by_diagrams``):

* toast once on the transition INTO blackout;
* no re-toast while a geometry stays blacked-out;
* recovery drops the geometry from the dedupe set, so a later
  re-blackout toasts again;
* the message names the culprit layer(s), or "combined filters" when
  only the union covers the grid.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from apeGmsh.viewers.core.element_visibility import (
    ElementVisibility,
    LAYER_SCOPE,
    LAYER_THRESHOLD,
)
from apeGmsh.viewers.results_viewer import _sync_blackout_hints


# ---------------------------------------------------------------------
# Stub director — geometries + materialized scenes, nothing else.
# ---------------------------------------------------------------------


class _Geom:
    def __init__(self, gid: str, name: str) -> None:
        self.id = gid
        self.name = name


class _Scene:
    def __init__(self, ev) -> None:
        self.element_visibility = ev


class _Director:
    """The two reads _sync_blackout_hints performs, and nothing more."""

    def __init__(self, entries) -> None:
        # entries: list of (geom, scene-or-None)
        self._scenes = {g.id: s for g, s in entries}

        class _GeomMgr:
            geometries = [g for g, _s in entries]

        self.geometries = _GeomMgr()

    def materialized_scene_for(self, geom):
        return self._scenes.get(geom.id)


def _ev(n_side: int = 3) -> ElementVisibility:
    grid = pv.ImageData(
        dimensions=(n_side, n_side, n_side),
    ).cast_to_unstructured_grid()
    return ElementVisibility(grid)


def _full(ev) -> np.ndarray:
    return np.ones(ev._n, dtype=bool)


# ---------------------------------------------------------------------
# Transition truth table
# ---------------------------------------------------------------------


def test_toast_fires_once_on_transition_into_blackout():
    ev = _ev()
    geom = _Geom("g1", "Deck")
    director = _Director([(geom, _Scene(ev))])
    blacked_out: set = set()
    msgs: list[str] = []

    # Fresh: nothing hidden, no toast.
    _sync_blackout_hints(director, blacked_out, msgs.append)
    assert msgs == []
    assert blacked_out == set()

    # Threshold hides everything → one toast, geometry tracked.
    ev.set_layer(LAYER_THRESHOLD, _full(ev))
    _sync_blackout_hints(director, blacked_out, msgs.append)
    assert len(msgs) == 1
    assert blacked_out == {"g1"}

    # Still blacked out on the next event → no re-toast.
    _sync_blackout_hints(director, blacked_out, msgs.append)
    _sync_blackout_hints(director, blacked_out, msgs.append)
    assert len(msgs) == 1


def test_toast_refires_after_recovery_and_reblackout():
    ev = _ev()
    geom = _Geom("g1", "Deck")
    director = _Director([(geom, _Scene(ev))])
    blacked_out: set = set()
    msgs: list[str] = []

    ev.set_layer(LAYER_THRESHOLD, _full(ev))
    _sync_blackout_hints(director, blacked_out, msgs.append)
    assert len(msgs) == 1

    # Recovery: threshold cleared → dropped from the dedupe set,
    # silently (no "everything is back" toast).
    ev.clear_layer(LAYER_THRESHOLD)
    _sync_blackout_hints(director, blacked_out, msgs.append)
    assert len(msgs) == 1
    assert blacked_out == set()

    # Re-blackout → toasts again.
    ev.set_layer(LAYER_THRESHOLD, _full(ev))
    _sync_blackout_hints(director, blacked_out, msgs.append)
    assert len(msgs) == 2


def test_message_names_geometry_and_culprit_layers():
    ev = _ev()
    geom = _Geom("g1", "Deck")
    director = _Director([(geom, _Scene(ev))])
    msgs: list[str] = []

    ev.set_layer(LAYER_THRESHOLD, _full(ev))
    ev.set_layer(LAYER_SCOPE, _full(ev))
    _sync_blackout_hints(director, set(), msgs.append)

    # Layer keys map to user-facing names ("scope" → "scope box").
    assert msgs == [
        "Geometry 'Deck': all elements hidden (threshold, scope box)"
    ]


def test_message_reports_combined_filters_when_no_single_culprit():
    ev = _ev()
    n = ev._n
    half = np.zeros(n, dtype=bool)
    half[: n // 2] = True
    geom = _Geom("g1", "Deck")
    director = _Director([(geom, _Scene(ev))])
    msgs: list[str] = []

    ev.set_layer(LAYER_THRESHOLD, half)
    ev.set_layer(LAYER_SCOPE, ~half)
    _sync_blackout_hints(director, set(), msgs.append)

    assert msgs == [
        "Geometry 'Deck': all elements hidden (combined filters)"
    ]


def test_walks_all_geometries_payload_independent():
    """ADR 0084 D3 — the handler must not trust any payload: every
    materialized geometry is checked, unmaterialized ones skipped."""
    ev_a, ev_b = _ev(), _ev()
    ga, gb, gc = _Geom("a", "A"), _Geom("b", "B"), _Geom("c", "C")
    director = _Director(
        [(ga, _Scene(ev_a)), (gb, _Scene(ev_b)), (gc, None)],
    )
    blacked_out: set = set()
    msgs: list[str] = []

    ev_a.set_layer(LAYER_THRESHOLD, _full(ev_a))
    ev_b.set_layer(LAYER_SCOPE, _full(ev_b))
    _sync_blackout_hints(director, blacked_out, msgs.append)

    assert len(msgs) == 2
    assert blacked_out == {"a", "b"}
    assert any("'A'" in m and "threshold" in m for m in msgs)
    assert any("'B'" in m and "scope" in m for m in msgs)


def test_partial_hide_never_toasts():
    ev = _ev()
    geom = _Geom("g1", "Deck")
    director = _Director([(geom, _Scene(ev))])
    msgs: list[str] = []

    ev.hide(list(range(ev._n - 1)))    # all but one cell
    _sync_blackout_hints(director, set(), msgs.append)
    assert msgs == []


def test_viewer_handler_shell_binds_set_status():
    """The RENDER-lane subscriber shell: binds the viewer's dedupe set
    and the window status bar; no director / no window → no-op. Bound
    to a namespace stub (the test_outline_visibility_state idiom) so
    no Qt window is needed."""
    from apeGmsh.viewers.results_viewer import ResultsViewer

    class _Win:
        def __init__(self) -> None:
            self.statuses: list[tuple[str, int]] = []

        def set_status(self, msg, timeout=0):
            self.statuses.append((msg, timeout))

    class _NS:
        pass

    ev = _ev()
    ns = _NS()
    ns._director = _Director([(_Geom("g1", "Deck"), _Scene(ev))])
    ns._win = _Win()
    ns._blackout_geom_ids = set()
    ns._blackout_toast_at = {}
    handler = ResultsViewer._on_element_blackout.__get__(ns)

    # No blackout → no status.
    handler()
    assert ns._win.statuses == []

    ev.set_layer(LAYER_THRESHOLD, _full(ev))
    handler(None, None)    # (kind, payload) — payload ignored
    assert ns._win.statuses == [
        ("Geometry 'Deck': all elements hidden (threshold)", 6000),
    ]

    # Pre-show / post-close: director or window missing → silent no-op
    # (no raise, no state change).
    ns._director = None
    handler()
    ns._director = _Director([])
    ns._win = None
    handler()
    assert ns._blackout_geom_ids == {"g1"}    # untouched by the no-ops

# ---------------------------------------------------------------------
# Adversarial-review fixes: cooldown, hidden-geometry guard
# ---------------------------------------------------------------------


def test_cooldown_caps_reblackout_toasts_per_geometry():
    """Scrubbing across a step whose data empties a threshold is a
    transition per crossing — the cooldown caps that at one toast per
    ``cooldown_s`` per geometry, while a crossing after the window
    re-toasts."""
    ev = _ev()
    geom = _Geom("g1", "Deck")
    director = _Director([(geom, _Scene(ev))])
    blacked_out: set = set()
    last_at: dict = {}
    msgs: list[str] = []
    clock = {"t": 100.0}

    def _tick(dt: float):
        clock["t"] += dt

    kw = dict(last_toast_at=last_at, now_fn=lambda: clock["t"],
              cooldown_s=10.0)

    ev.set_layer(LAYER_THRESHOLD, _full(ev))
    _sync_blackout_hints(director, blacked_out, msgs.append, **kw)
    assert len(msgs) == 1

    # Oscillate within the cooldown window: transitions happen (the
    # dedupe set re-arms) but the toast is suppressed.
    for _ in range(3):
        _tick(1.0)
        ev.clear_layer(LAYER_THRESHOLD)
        _sync_blackout_hints(director, blacked_out, msgs.append, **kw)
        ev.set_layer(LAYER_THRESHOLD, _full(ev))
        _sync_blackout_hints(director, blacked_out, msgs.append, **kw)
    assert len(msgs) == 1

    # Past the window, the next transition toasts again.
    _tick(11.0)
    ev.clear_layer(LAYER_THRESHOLD)
    _sync_blackout_hints(director, blacked_out, msgs.append, **kw)
    ev.set_layer(LAYER_THRESHOLD, _full(ev))
    _sync_blackout_hints(director, blacked_out, msgs.append, **kw)
    assert len(msgs) == 2


def test_hidden_geometry_never_toasts():
    """A geometry the user hid on purpose has nothing to explain — its
    masks are frozen at the step it was last rendered, so a toast off
    them would attribute stale state."""
    ev = _ev()
    geom = _Geom("g1", "Deck")
    geom.visible = False
    director = _Director([(geom, _Scene(ev))])
    blacked_out: set = set()
    msgs: list[str] = []

    ev.set_layer(LAYER_SCOPE, _full(ev))
    _sync_blackout_hints(director, blacked_out, msgs.append)
    assert msgs == []
    assert blacked_out == set()

    # Re-shown → the next scan explains it.
    geom.visible = True
    _sync_blackout_hints(director, blacked_out, msgs.append)
    assert len(msgs) == 1
