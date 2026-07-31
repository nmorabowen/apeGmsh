"""One pin-aware STEP path when pumps are bound (ADR 0084 D2 / PR 10).

Until this PR, ``Director.set_step`` / ``set_stage`` ran a direct,
pin-BLIND ``registry.update_to_step(...)`` + ``_render()`` and *then*
fired the observers that drive the pin-AWARE dispatcher pumps and render
again. Three consequences, one test each below:

* **~2x work per scrub tick** — every visible diagram was stepped twice
  and the viewport painted twice, with a pin-blind intermediate frame.
* **A crash** — ``registry.update_to_step`` pushes the raw global cursor
  to every attached diagram, so a geometry pinned to a SHORTER stage read
  a step that does not exist and ``results/_time.py`` raised
  ``IndexError: List time_slice contains out-of-range indices``. Stage
  pin + short stage + scrubbing past its end was a live user-facing
  crash.
* **The mirror-image bug in combined mode** — there the direct path
  pushed the TRANSLATED local step while the dispatcher pump pushed the
  RAW GLOBAL ``director.step_index``, and the pump ran LAST, so the wrong
  value won.

D2's fix is a ``pumps_bound`` flag recorded by ``Dispatcher.bind()``:
when it is set, the Director's STEP/STAGE paths are intent + observers
only, and ``PumpSet.effective_step_for`` is the ONE step resolver —
pin-aware *and* combined-mode-aware. The unbound branch (WebViewer/trame,
bare-director export scripts, headless director tests) keeps the direct
path; its contract lives in ``test_results_director.py``.

The fixture is PR 8's: two stages of deliberately different length
(``push`` 4 steps, ``grav`` 2) with a per-stage value offset baked into
the data, so every assertion here is on VALUES — reading the right step
of the wrong stage is off by 500 000, not off by zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers._pump_set import PumpSet
from apeGmsh.viewers.diagrams import ResultsDirector
from apeGmsh.viewers.diagrams._director import COMBINED_STAGE_ID
from apeGmsh.viewers.diagrams._dispatch import STAGE_CHANGED, STEP_CHANGED
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

# PR 8's fixture, imported rather than re-typed: the two-stage beam with
# per-stage value offsets is exactly the shape this PR needs, and a
# second copy of a 70-line HDF5 writer would be a second thing to keep
# honest. Re-exporting ``beam_two_stage`` makes it a fixture here too.
from tests.viewers.test_scrub_and_session_roundtrip import (  # noqa: F401
    _N_STEPS_GRAV,
    _N_STEPS_PUSH,
    _contour,
    _nodal_disp_z,
    _stage_id,
    beam_two_stage,
)


# =====================================================================
# Harness — a real director wired the way the shell wires it
# =====================================================================


def _stage_pin_for(director):
    """The resolver ``results_viewer`` hands ``bind_plotter`` (S3b)."""
    def _resolve(diagram):
        owner = director.geometries.geometry_for_layer(diagram)
        return getattr(owner, "stage_id", None) if owner is not None else None
    return _resolve


class _Wiring:
    """Counters for everything a scrub tick is allowed to do once."""

    def __init__(self) -> None:
        self.step = 0
        self.deform = 0
        self.gate = 0
        self.dispatch_render = 0
        self.director_render = 0
        self.direct_update_to_step: list[int] = []


def _wire(director, scene, *, bind_pumps: bool = True):
    """Bind the plotter, then optionally bind real pumps + the shell's
    observer bridge. Returns ``(pumps, wiring)``.

    ``bind_pumps=False`` is the WebViewer / bare-director shape: a bound
    plotter and a render callback, but ``Dispatcher.bind`` never called,
    so the pumps stay no-op and ``pumps_bound`` stays False.
    """
    from tests.viewers.conftest import RecordingBackend

    w = _Wiring()
    director.bind_plotter(
        RecordingBackend(),
        scene=scene,
        render_callback=lambda: setattr(
            w, "director_render", w.director_render + 1,
        ),
        stage_pin_resolver=_stage_pin_for(director),
    )
    # Spy the pin-BLIND direct push. With pumps bound nothing may reach
    # it; that single assertion is the whole of D2 in one line.
    registry = director.registry
    inner = registry.update_to_step

    def _spy_update_to_step(step_index: int) -> None:
        w.direct_update_to_step.append(int(step_index))
        inner(step_index)

    registry.update_to_step = _spy_update_to_step  # type: ignore[method-assign]

    pumps = PumpSet(
        director=director,
        scene=scene,
        read_deform_field=lambda *a, **k: None,
        render_geometries=lambda: [
            geo for geo in director.geometries.geometries if geo.visible
        ],
        sync_node_cloud=lambda _pts: None,
        sync_diagram_substrate_points=lambda *a, **k: None,
    )
    if bind_pumps:
        def _counted(name, fn):
            def _run(*args, **kwargs):
                setattr(w, name, getattr(w, name) + 1)
                return fn(*args, **kwargs)
            return _run

        director.dispatcher.bind(
            pump_step=_counted("step", pumps.pump_step),
            pump_deform=_counted("deform", pumps.pump_deform),
            pump_gate=_counted("gate", pumps.pump_gate),
            pump_restack=pumps.pump_restack,
            render=_counted("dispatch_render", lambda: None),
        )
        # The shell's observer bridge (``results_viewer.py:1416-1421``).
        dispatcher = director.dispatcher
        director.subscribe_step(lambda _s: dispatcher.fire(STEP_CHANGED))
        director.subscribe_stage(lambda _s: dispatcher.fire(STAGE_CHANGED))
    return pumps, w


def _own(director, geom, layer, *, name: str = "C") -> None:
    """Put ``layer`` in a composition of ``geom``, then register it.

    Ownership first, registry second: the registry attaches on ``add``
    when it is bound, and a diagram resolves its effective stage (the
    owning geometry's pin, ADR 0058 S3b) at attach — so a layer added
    before it has an owner attaches against the ACTIVE stage and stays
    there. Same ordering rule PR 8's ``_pinned_pair`` documents.
    """
    comp = _composition(geom, name)
    geom.compositions.add_layer(comp.id, layer)
    director.registry.add(layer)


def _composition(geom, name: str):
    for c in geom.compositions.compositions:
        if c.name == name:
            return c
    return geom.compositions.add(name=name, make_active=True)


def _spy_steps(layer) -> list[int]:
    """Record every step application on ``layer``, in order."""
    seen: list[int] = []
    inner = layer.update_to_step

    def _wrapped(step_index: int) -> None:
        seen.append(int(step_index))
        inner(step_index)

    layer.update_to_step = _wrapped
    return seen


def _values(layer) -> np.ndarray:
    return np.asarray(layer._layer.field_named("displacement_z").values)


# =====================================================================
# 1 — the render-count spy: one tick, one STEP per diagram, one paint
# =====================================================================


def test_one_scrub_tick_is_one_step_one_deform_one_render(
    beam_two_stage,
):
    """ADR 0084 D1's freeze criterion, asserted.

    Two visible diagrams, one ``set_step``. Each diagram must be stepped
    EXACTLY once (twice was the dual path), the DEFORM pump must run
    once, the viewport must be painted once, and the Director's own
    ``render_callback`` must not fire at all — the dispatcher owns the
    paint now.
    """
    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    director.set_stage(_stage_id(results, "push"))

    _pumps, w = _wire(director, scene)
    layers = [_contour(results), _contour(results)]
    for layer in layers:
        _own(director, director.geometries.active, layer, name="Both")

    director.set_step(0)
    seen = [_spy_steps(layer) for layer in layers]
    w.__init__()                              # counters from zero

    director.set_step(3)

    assert seen == [[3], [3]], (
        "each visible diagram must be stepped exactly once per tick"
    )
    assert (w.step, w.deform, w.dispatch_render) == (1, 1, 1)
    assert w.gate == 0, "a scrub tick does not re-run the composition gate"
    assert w.director_render == 0, (
        "the Director must not paint while pumps are bound"
    )
    assert w.direct_update_to_step == [], (
        "the pin-blind direct push must not run while pumps are bound"
    )
    # And the values landed, not just the call counts.
    for layer in layers:
        np.testing.assert_array_equal(
            _values(layer),
            _nodal_disp_z(layer._fem_ids_to_read, 3, "push"),
        )


# =====================================================================
# 2 — the crash: a geometry pinned to a SHORTER stage, scrubbed past it
# =====================================================================


def test_scrub_past_a_pinned_shorter_stage_does_not_raise(beam_two_stage):
    """The ADR 0084 D2 crash-path amendment, as a regression test.

    ``grav`` has 2 steps, ``push`` has 4. Scrubbing the global cursor to
    step 3 with a geometry pinned to ``grav`` used to push the raw 3 into
    a 2-step stage and raise ``IndexError: List time_slice contains
    out-of-range indices`` straight out of ``set_step``.

    Both diagrams are attached BEFORE the scrub — that is the whole
    point; PR 8's version of this case had to position the cursor while
    nothing was attached to dodge the crash.

    Asserted on VALUES: the pinned geometry must land on ``grav``'s
    clamped local step (1), the unpinned one on ``push`` step 3. The
    per-stage offset means reading step 1 of the WRONG stage fails.
    """
    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    director.set_stage(_stage_id(results, "push"))

    pumps, w = _wire(director, scene)
    geoms = director.geometries
    free_geom = geoms.active
    pinned_geom = geoms.add("Pinned", make_active=False)
    geoms.set_stage_pin(pinned_geom.id, _stage_id(results, "grav"))

    free_layer, pinned_layer = _contour(results), _contour(results)
    for geom, layer in ((free_geom, free_layer), (pinned_geom, pinned_layer)):
        _own(director, geom, layer)

    # ``set_stage`` lands on the end of history, so park at 0 first —
    # otherwise the interesting move is a no-op early return.
    director.set_step(0)
    director.set_step(3)                      # must not raise

    assert director.step_index == 3
    assert pumps.effective_step_for(free_layer) == 3
    assert pumps.effective_step_for(pinned_layer) == _N_STEPS_GRAV - 1

    np.testing.assert_array_equal(
        _values(free_layer),
        _nodal_disp_z(free_layer._fem_ids_to_read, 3, "push"),
        err_msg="unpinned geometry did not follow the global cursor",
    )
    np.testing.assert_array_equal(
        _values(pinned_layer),
        _nodal_disp_z(pinned_layer._fem_ids_to_read, 1, "grav"),
        err_msg="pinned geometry did not clamp inside its shorter stage",
    )
    assert w.direct_update_to_step == []


def test_scrub_past_a_pinned_shorter_stage_still_crashes_unbound(
    beam_two_stage,
):
    """The same gesture with NO pumps bound still raises — deliberately.

    The unbound branch is the pin-blind one by contract (ADR 0084 D2):
    nothing in a WebViewer / bare-director consumer sets a stage pin, so
    the direct path has no pin to be aware of. Pinning one anyway is
    outside that contract, and this test pins the failure mode in place
    so the two branches can't be confused for each other.
    """
    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    director.set_stage(_stage_id(results, "push"))

    _pumps, _w = _wire(director, scene, bind_pumps=False)
    geoms = director.geometries
    pinned_geom = geoms.add("Pinned", make_active=False)
    geoms.set_stage_pin(pinned_geom.id, _stage_id(results, "grav"))
    pinned_layer = _contour(results)
    _own(director, pinned_geom, pinned_layer)

    director.set_step(0)                      # inside grav's range
    with pytest.raises(IndexError):
        director.set_step(3)                  # past grav's end


# =====================================================================
# 3 — combined mode: the translated local step, not the raw global
# =====================================================================


def test_combined_mode_pushes_the_translated_local_step(beam_two_stage):
    """ADR 0084 D2, the third consequence.

    Combined mode concatenates ``push`` (4) + ``grav`` (2) into a 6-step
    global timeline. Global step 5 is ``grav``'s local step 1. The
    dispatcher pump used to push the RAW GLOBAL 5 at a diagram reading a
    2-step stage, and — running after the Director's own translated push
    — it won.

    Asserted on values at both ends of a boundary crossing, so the
    translation has to be right in both directions, and the per-stage
    offset proves which stage was actually read.
    """
    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    director.set_stage(_stage_id(results, "push"))

    pumps, w = _wire(director, scene)
    layer = _contour(results)
    _own(director, director.geometries.active, layer)

    director.set_stage(COMBINED_STAGE_ID)
    assert director.combined_active
    assert director.n_steps == _N_STEPS_PUSH + _N_STEPS_GRAV

    # --- inside the FIRST stage's segment: global == local ------------
    director.set_step(1)
    assert pumps.effective_step_for(layer) == 1
    np.testing.assert_array_equal(
        _values(layer),
        _nodal_disp_z(layer._fem_ids_to_read, 1, "push"),
        err_msg="combined step 1 should be push step 1",
    )

    # --- across the boundary: global 5 is grav's local 1 --------------
    director.set_step(5)                      # must not raise
    assert director.step_index == 5           # the cursor stays global
    assert pumps.effective_step_for(layer) == 1
    np.testing.assert_array_equal(
        _values(layer),
        _nodal_disp_z(layer._fem_ids_to_read, 1, "grav"),
        err_msg="combined step 5 should be grav step 1, not raw step 5",
    )

    # --- and back ------------------------------------------------------
    director.set_step(0)
    np.testing.assert_array_equal(
        _values(layer),
        _nodal_disp_z(layer._fem_ids_to_read, 0, "push"),
    )
    assert w.direct_update_to_step == []


# =====================================================================
# 4 — the unbound branch keeps working (web/trame + bare director)
# =====================================================================


def test_unbound_director_still_steps_and_renders(beam_two_stage):
    """No ``Dispatcher.bind`` — the direct path IS the contract.

    This is the WebViewer/trame shape (``web_viewer.py`` binds a plotter
    and a render callback but never binds pumps) and the bare-director
    export shape. ``set_step`` must still push values into the diagrams
    and still call the render callback, because there is no reconciler
    behind the observers to do it.
    """
    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    director.set_stage(_stage_id(results, "push"))

    _pumps, w = _wire(director, scene, bind_pumps=False)
    assert director.dispatcher.pumps_bound is False

    layer = _contour(results)
    _own(director, director.geometries.active, layer)

    director.set_step(0)
    w.__init__()
    director.set_step(2)

    assert w.direct_update_to_step == [2], "the direct push is the contract here"
    assert w.director_render == 1
    assert (w.step, w.deform, w.dispatch_render) == (0, 0, 0)
    np.testing.assert_array_equal(
        _values(layer),
        _nodal_disp_z(layer._fem_ids_to_read, 2, "push"),
    )
