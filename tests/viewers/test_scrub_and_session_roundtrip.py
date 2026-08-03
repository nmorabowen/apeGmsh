"""The two interactions that had no headless test before ADR 0084 PR 7.

ADR 0084 D7 / PR 8 — the "hard half" of the interaction net. Both of
these were unreachable without a real window until the ``PumpSet`` /
``SessionController`` carve, because the pump closures and the
session-restore body lived inside ``ResultsViewer._show_impl``:

**Test 1 — scrub with a contour AND a line_force attached at once.**
Every existing scrub test drives one diagram kind. The pairing is the
point: a contour reads a nodal slab and scatters scalars, a line_force
reads a line-station slab and moves geometry, and they share one STEP
pump and one DEFORM pump. This asserts the *values* each kind carries
at each step and the *positions* the substrate lands on
(``reference + scale·u``) — not that the pumps were called.

**Test 2 — a saved session restored onto a live director.**
``tests/viewers/test_session_persistence.py`` round-trips the spec
JSON; nothing asserted that a payload actually rebuilds the diagram →
composition → geometry graph. This drives the real serialize side into
``SessionController.apply`` with fakes only at the shell boundary (the
status-bar window and the late-resolved clip/legend providers), then
asserts the restored graph.

Both run in the default lane: no ``QApplication``, no ``show()``, no GL
(the shared ``backend`` fixture is the no-GL ``RecordingBackend``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers._pump_set import PumpSet
from apeGmsh.viewers._session_apply import SessionController
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DiagramSpec,
    LineForceDiagram,
    LineForceStyle,
    ResultsDirector,
    SlabSelector,
)
from apeGmsh.viewers.diagrams._session import (
    CompositionSnapshot,
    GeometrySnapshot,
    deserialize_session,
    serialize_session,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


# =====================================================================
# Fixture — one beam carrying BOTH a nodal slab and a line-station slab
# =====================================================================
#
# A single model that feeds a contour and a line_force simultaneously
# is what makes the pairing testable at all. Two stages so the stage
# pin (ADR 0058 S3b) has somewhere to point.

_N_STEPS_PUSH = 4          # active stage
_N_STEPS_GRAV = 2          # pinned stage — deliberately SHORTER
_STATIONS = np.array([-1.0, 0.0, 1.0])      # 3-pt Lobatto


# Per-stage offset, so a value identifies the stage it came from as
# well as the step — otherwise a pin test that reads the WRONG stage at
# the RIGHT step would still pass.
_STAGE_OFFSET = {"push": 0.0, "grav": 500_000.0}


def _nodal_disp_z(
    node_ids: np.ndarray, step: int, stage: str = "push",
) -> np.ndarray:
    """Known nodal scalar: ``node_id + 1000·step + stage offset``."""
    return (
        node_ids.astype(np.float64)
        + step * 1000.0
        + _STAGE_OFFSET[stage]
    )


def _moment(ei: int, si: int, step: int) -> float:
    """Known per-station value: ``(ei+1)·100 + step·10 + si``."""
    return (ei + 1) * 100.0 + step * 10.0 + si


@pytest.fixture
def beam_two_stage(g, tmp_path: Path):
    """3-segment beam with nodal ``displacement_z`` + line-station
    ``bending_moment_z``, written for two stages of different length.

    Returns ``(results, node_ids, line_eids)``.
    """
    pts = [
        g.model.geometry.add_point(float(x), 0.0, 0.0, label=f"p{x}")
        for x in range(4)
    ]
    for i in range(3):
        g.model.geometry.add_line(pts[i], pts[i + 1], label=f"seg{i}")
    g.physical.add_curve([f"seg{i}" for i in range(3)], name="Beam")
    g.mesh.sizing.set_global_size(10.0)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    line_eids = sorted(
        int(x)
        for group in fem.elements
        if group.element_type.dim == 1
        for x in group.ids
    )
    assert line_eids, "fixture produced no line elements"
    n_beams = len(line_eids)

    def _slabs(n_steps: int, stage: str):
        disp = np.stack(
            [_nodal_disp_z(node_ids, s, stage) for s in range(n_steps)],
        )
        moments = np.zeros(
            (n_steps, n_beams, _STATIONS.size), dtype=np.float64,
        )
        for s in range(n_steps):
            for ei in range(n_beams):
                for si in range(_STATIONS.size):
                    moments[s, ei, si] = _moment(ei, si, s)
        return disp, moments

    path = tmp_path / "beam_two_stage.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        for name, n_steps in (
            ("push", _N_STEPS_PUSH), ("grav", _N_STEPS_GRAV),
        ):
            disp, moments = _slabs(n_steps, name)
            sid = w.begin_stage(
                name=name, kind="static",
                time=np.arange(n_steps, dtype=np.float64),
            )
            w.write_nodes(
                sid, "partition_0", node_ids=node_ids,
                components={"displacement_z": disp},
            )
            w.write_line_stations_group(
                sid, "partition_0", group_id="g0",
                class_tag=10, int_rule=0,
                element_index=np.asarray(line_eids, dtype=np.int64),
                station_natural_coord=_STATIONS,
                components={"bending_moment_z": moments},
            )
            w.end_stage()

    results = Results.from_native(path, model=_open_model_from_h5(path))
    return results, node_ids, line_eids


def _contour(results) -> ContourDiagram:
    return ContourDiagram(
        DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="displacement_z"),
            style=ContourStyle(topology="nodes"),
        ),
        results,
    )


def _line_force(results, *, scale: float = 1.0) -> LineForceDiagram:
    return LineForceDiagram(
        DiagramSpec(
            kind="line_force",
            selector=SlabSelector(component="bending_moment_z"),
            style=LineForceStyle(scale=scale),
        ),
        results,
    )


def _stage_id(results, name: str) -> str:
    for stage in results.stages:
        if stage.name == name:
            return stage.id
    raise AssertionError(f"no stage named {name!r}")


def _stamp_stage_pin_resolver(director) -> None:
    """Install the stage-pin resolver the SHELL normally supplies.

    ADR 0058 S3b: ``results_viewer`` hands this to
    ``director.bind_plotter(stage_pin_resolver=...)``, and the registry
    stamps it on each diagram at add/attach so
    ``Diagram._scoped_results`` scopes reads to the owning geometry's
    pinned stage. These tests never bind a plotter, so they install the
    same resolver directly — it is a shell-boundary collaborator, and
    this is the verbatim body from ``results_viewer._stage_pin_for``.

    Must be called BEFORE ``registry.add`` — that is where the stamp
    happens.
    """
    def _stage_pin_for(diagram):
        owner = director.geometries.geometry_for_layer(diagram)
        return getattr(owner, "stage_id", None) if owner is not None else None

    director.registry._stage_pin_resolver = _stage_pin_for


def _pumps(director, scene, **overrides) -> PumpSet:
    """A PumpSet over a REAL director with inert shell collaborators."""
    kwargs = dict(
        director=director,
        scene=scene,
        read_deform_field=lambda *a, **k: None,
        render_geometries=lambda: [
            geo for geo in director.geometries.geometries if geo.visible
        ],
        sync_node_cloud=lambda _pts: None,
        sync_diagram_substrate_points=lambda *a, **k: None,
        # As the shell wires it (ADR 0084 D1) — inert until a threshold
        # is actually set on some geometry.
        thresholds=director.thresholds,
    )
    kwargs.update(overrides)
    return PumpSet(**kwargs)


# =====================================================================
# Test 1 — contour + line_force + deform, scrubbed together
# =====================================================================


def test_scrub_drives_contour_values_line_force_values_and_points(
    beam_two_stage, backend,
):
    """The adversarial pairing: one contour and one line_force attached
    at the same time, scrubbed across every step.

    At each step this asserts three independent things:

    * the contour's scattered scalars equal ``node_id + 1000·step`` for
      the exact fem ids its submesh reads;
    * the line_force's station offsets equal ``scale · moment(ei, si,
      step)`` along the fill direction — i.e. the OTHER diagram kind
      read the OTHER slab type at the same cursor;
    * the substrate points equal ``reference + scale·u(step)``.

    Before PR 7 none of this was reachable without a window.
    """
    results, _node_ids, line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    reference = np.asarray(scene.reference_points, dtype=np.float64).copy()

    director = ResultsDirector(results)
    director.set_stage(_stage_id(results, "push"))

    contour = _contour(results)
    line_force = _line_force(results, scale=1.0)
    for d in (contour, line_force):
        director.registry.add(d)
        d.attach(backend, results.fem, scene)
    comp = director.geometries.active.compositions.add(
        name="Both", make_active=True,
    )
    for d in (contour, line_force):
        director.geometries.active.compositions.add_layer(comp.id, d)

    # A known deformation field: node i moves +i in Y at step 0, and
    # grows by +1 per step. Read through the injected reader — that
    # callback IS the seam contract (PR 7 froze it).
    n_nodes = reference.shape[0]
    deform_scale = 3.0

    def field_at(step: int) -> np.ndarray:
        u = np.zeros((n_nodes, 3), dtype=np.float64)
        u[:, 1] = np.arange(n_nodes, dtype=np.float64) + float(step)
        return u

    reads: list[tuple[str, int]] = []

    def read_deform_field(name, step, stage_id=None):
        reads.append((name, int(step)))
        return field_at(int(step))

    director.geometries.set_deformation(
        director.geometries.active_id,
        enabled=True, field="displacement", scale=deform_scale,
    )
    pumps = _pumps(
        director, scene, read_deform_field=read_deform_field,
    )

    fem_ids = contour._fem_ids_to_read
    n_stations = line_force._n_stations
    n_per_beam = _STATIONS.size

    for step in (0, 2, 3, 1, 0):        # forward, back, and home again
        director.set_step(step)
        pumps.pump_step(None)
        pumps.pump_deform(None)

        # --- contour: the nodal scalars actually scattered -----------
        field = contour._layer.field_named("displacement_z")
        np.testing.assert_array_equal(
            np.asarray(field.values),
            _nodal_disp_z(fem_ids, step),
            err_msg=f"contour scalars wrong at step {step}",
        )

        # --- line_force: the station offsets actually built ----------
        pts = np.asarray(line_force._current_points, dtype=np.float64)
        offsets = pts[n_stations:] - pts[:n_stations]
        expected = np.array([
            _moment(ei, si, step)
            for ei in range(len(line_eids))
            for si in range(n_per_beam)
        ])
        # Project back onto the (unit) fill directions to recover the
        # signed value the diagram read, station by station.
        recovered = np.einsum(
            "ij,ij->i", offsets, line_force._fill_directions,
        ) / line_force.current_scale()
        np.testing.assert_allclose(
            recovered, expected, rtol=1e-9, atol=1e-9,
            err_msg=f"line_force station values wrong at step {step}",
        )

        # --- substrate: reference + scale·u --------------------------
        np.testing.assert_allclose(
            np.asarray(scene.grid.points, dtype=np.float64),
            reference + deform_scale * field_at(step),
            rtol=1e-9, atol=1e-9,
            err_msg=f"deformed points wrong at step {step}",
        )

    # The DEFORM pump read the field once per scrub step, at the step
    # the cursor was on — no stale reads, no double reads.
    assert [s for _name, s in reads] == [0, 2, 3, 1, 0]


def _pinned_pair(results, scene, director, backend):
    """A pinned geometry (``grav``) + an unpinned one (``push``), each
    owning one contour, wired the way the shell wires them."""
    geoms = director.geometries
    free_geom = geoms.active
    pinned_geom = geoms.add("Pinned", make_active=False)
    geoms.set_stage_pin(pinned_geom.id, _stage_id(results, "grav"))

    _stamp_stage_pin_resolver(director)      # before add — that stamps

    free_layer = _contour(results)
    pinned_layer = _contour(results)
    for geom, layer in (
        (free_geom, free_layer), (pinned_geom, pinned_layer),
    ):
        director.registry.add(layer)
        comp = geom.compositions.add(name="C", make_active=True)
        geom.compositions.add_layer(comp.id, layer)
        # Attach AFTER ownership exists so the first read already
        # resolves through the pin.
        layer.attach(backend, results.fem, scene)
    return free_layer, pinned_layer


def _assert_pin_split(pumps, free_layer, pinned_layer, global_step, pinned_step):
    """Both geometries stepped from ONE cursor, asserted on values.

    The per-stage offset baked into the fixture makes the two
    assertions independent: reading the right step of the WRONG stage
    is off by 500 000, not off by zero.
    """
    assert pumps.effective_step_for(free_layer) == global_step
    assert pumps.effective_step_for(pinned_layer) == pinned_step

    pumps.pump_step(None)

    fem_ids = free_layer._fem_ids_to_read
    np.testing.assert_array_equal(
        np.asarray(free_layer._layer.field_named("displacement_z").values),
        _nodal_disp_z(fem_ids, global_step, "push"),
        err_msg=f"unpinned geometry wrong at global step {global_step}",
    )
    np.testing.assert_array_equal(
        np.asarray(pinned_layer._layer.field_named("displacement_z").values),
        _nodal_disp_z(fem_ids, pinned_step, "grav"),
        err_msg=f"pinned geometry wrong at global step {global_step}",
    )


def test_scrub_is_pin_aware_across_two_geometries(beam_two_stage, backend):
    """ADR 0058 S3b — a stage-pinned geometry and an unpinned one
    scrubbed by the SAME global cursor read different stages.

    Asserted on the values each geometry's contour carries: the pinned
    one must show ``grav`` data, the unpinned one ``push`` data, at the
    same cursor.
    """
    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    director.set_stage(_stage_id(results, "push"))

    free_layer, pinned_layer = _pinned_pair(
        results, scene, director, backend,
    )
    pumps = _pumps(director, scene)

    # Both stages cover these cursors, so the pin shows up purely as a
    # stage difference.
    for step in (0, 1):
        director.set_step(step)
        _assert_pin_split(pumps, free_layer, pinned_layer, step, step)


def test_pinned_geometry_clamps_at_the_end_of_its_shorter_stage(
    beam_two_stage, backend,
):
    """ADR 0058 S3b — ``grav`` has 2 steps, ``push`` has 4. At global
    step 3 the pinned geometry holds at its last step (1) while the
    unpinned one reads 3.

    The cursor is moved BEFORE the diagrams attach, on purpose. The
    Director's direct STEP path (``set_step`` → ``registry.
    update_to_step(raw)``) is not pin-aware — it pushes the raw global
    step to every attached diagram, and step 3 does not exist in
    ``grav``, so it raises ``IndexError`` out of ``set_step``. That is
    the ADR 0084 D2 dual path, which PR 10 collapses; this test drives
    the PUMP seam, so it positions the cursor while nothing is attached
    and lets the pin-aware pump do the stepping.
    """
    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    director.set_stage(_stage_id(results, "push"))
    director.set_step(3)                     # nothing attached yet

    free_layer, pinned_layer = _pinned_pair(
        results, scene, director, backend,
    )
    pumps = _pumps(director, scene)

    assert director.step_index == 3
    _assert_pin_split(pumps, free_layer, pinned_layer, 3, 1)


# =====================================================================
# Test 2 — save → apply → restored graph
# =====================================================================


class _FakeWindow:
    """The only thing ``SessionController.apply`` wants from a window."""

    def __init__(self) -> None:
        self.status: list[str] = []

    def set_status(self, msg: str, timeout: int = 0) -> None:
        self.status.append(msg)


def _controller(director, results) -> SessionController:
    """A SessionController with the shell boundary faked out.

    Every late-resolved collaborator is a zero-arg provider by design
    (PR 7) — returning ``None`` from each is exactly the "``show()``
    has not wired it yet" path the constructor documents.
    """
    return SessionController(director=director, results=results)


def _membership(geom_mgr) -> dict[tuple[str, str], tuple[str, ...]]:
    """``{(geometry name, composition name): (diagram kinds...)}``."""
    return {
        (geo.name, comp.name): tuple(
            layer.kind for layer in comp.layers
        )
        for geo in geom_mgr.geometries
        for comp in geo.compositions.compositions
    }


def _saved_payload(results, tmp_path: Path, *, retire_middle: bool = False):
    """The serialize side: two geometries, three specs, two
    compositions — then optionally retire the MIDDLE spec (the PR 3
    hole case)."""
    specs = [
        DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="displacement_z"),
            style=ContourStyle(topology="nodes"),
            label="Disp Z",
        ),
        DiagramSpec(
            kind="line_force",
            selector=SlabSelector(component="bending_moment_z"),
            style=LineForceStyle(scale=2.5),
            label="Mz",
        ),
        DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="displacement_z"),
            style=ContourStyle(topology="nodes"),
            label="Disp Z again",
        ),
    ]
    payload = serialize_session(
        specs=specs,
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        active_stage_id=_stage_id(results, "push"),
        active_step=2,
        geometries=[
            GeometrySnapshot(
                id="g0",
                name="Substrate",
                deform_enabled=True,
                deform_field="displacement",
                deform_scale=4.0,
                offset=(1.0, 2.0, 3.0),
                visible=True,
                active_composition_id="c0",
                compositions=(
                    CompositionSnapshot(
                        id="c0", name="Displacements", layer_indices=(0, 1),
                    ),
                    CompositionSnapshot(
                        id="c1", name="Forces", layer_indices=(2,),
                    ),
                ),
            ),
            GeometrySnapshot(
                id="g1",
                name="Pinned copy",
                stage_id=_stage_id(results, "grav"),
                visible=False,
                active_composition_id=None,
                compositions=(),
            ),
        ],
        active_geometry_id="g1",
    )
    if retire_middle:
        # What a pre-ADR-0058-S4 save looks like once ``deformed_shape``
        # left the kind registry — index 1 becomes a hole.
        payload["diagrams"][1] = {
            "kind": "deformed_shape",
            "selector": {"component": "displacement"},
            "style": {},
        }
    return payload


def test_session_apply_restores_the_whole_graph(beam_two_stage, tmp_path):
    """save → deserialize → ``SessionController.apply`` → assert state.

    The state-level restore test the spec-JSON round-trip file could
    never be: diagrams really constructed against the bound Results,
    compositions really populated, geometry deform / offset / stage-pin
    / visibility really applied to the live ``GeometryManager``.
    """
    results, _node_ids, _line_eids = beam_two_stage
    director = ResultsDirector(results)
    win = _FakeWindow()

    session = deserialize_session(_saved_payload(results, tmp_path))
    _controller(director, results).apply(session, win)

    geoms = director.geometries

    # --- the diagrams themselves --------------------------------------
    restored = list(director.registry.diagrams())
    assert [d.kind for d in restored] == ["contour", "line_force", "contour"]
    assert [d.spec.selector.component for d in restored] == [
        "displacement_z", "bending_moment_z", "displacement_z",
    ]
    assert [d.spec.label for d in restored] == [
        "Disp Z", "Mz", "Disp Z again",
    ]
    # Constructed against the bound Results, not a stub.
    assert all(d._results is results for d in restored)

    # --- composition membership ---------------------------------------
    assert _membership(geoms) == {
        ("Substrate", "Displacements"): ("contour", "line_force"),
        ("Substrate", "Forces"): ("contour",),
    }

    # --- geometry state ------------------------------------------------
    by_name = {geo.name: geo for geo in geoms.geometries}
    assert set(by_name) == {"Substrate", "Pinned copy"}

    substrate = by_name["Substrate"]
    assert substrate.deform_enabled is True
    assert substrate.deform_field == "displacement"
    assert substrate.deform_scale == pytest.approx(4.0)
    assert tuple(substrate.offset) == (1.0, 2.0, 3.0)
    assert substrate.visible is True
    assert substrate.stage_id is None
    # Active composition restored by NAME (saved uuids are stale).
    assert substrate.compositions.active.name == "Displacements"

    pinned = by_name["Pinned copy"]
    assert pinned.stage_id == _stage_id(results, "grav")
    assert pinned.visible is False

    # Active geometry pointer followed the saved id -> name.
    assert geoms.active.name == "Pinned copy"

    # --- cursor --------------------------------------------------------
    assert director.stage_id == _stage_id(results, "push")
    assert director.step_index == 2

    assert win.status == ["Restored 3 diagram(s)"]


def test_session_apply_lands_survivors_in_their_original_compositions(
    beam_two_stage, tmp_path,
):
    """ADR 0084 D6 / PR 3, end to end.

    PR 3 proved the DESERIALIZER pads a hole so ``layer_indices`` stay
    aligned. This proves the hole survives the full apply: with the
    middle spec retired, "Displacements" must keep only the contour and
    "Forces" must still get the third spec. Under compaction the
    survivor at index 2 would slide into index 1 and be swallowed by
    "Displacements", leaving "Forces" empty.
    """
    results, _node_ids, _line_eids = beam_two_stage
    director = ResultsDirector(results)
    win = _FakeWindow()

    session = deserialize_session(
        _saved_payload(results, tmp_path, retire_middle=True),
    )
    assert session.diagrams[1] is None      # the padded hole

    _controller(director, results).apply(session, win)

    assert [d.kind for d in director.registry.diagrams()] == [
        "contour", "contour",
    ]
    assert _membership(director.geometries) == {
        ("Substrate", "Displacements"): ("contour",),
        ("Substrate", "Forces"): ("contour",),
    }
    # The survivor in "Forces" is the THIRD spec, not the second.
    forces = next(
        c for c in director.geometries.geometries[0].compositions.compositions
        if c.name == "Forces"
    )
    assert forces.layers[0].spec.label == "Disp Z again"

    assert win.status == ["Restored 2 diagram(s); 1 skipped"]


def test_session_apply_v1_payload_bundles_into_one_composition(
    beam_two_stage, tmp_path,
):
    """A v1 session (no ``geometries`` block) still restores — every
    layer lands in one "Restored" composition under the active
    geometry. The legacy fallback the extraction preserved verbatim."""
    results, _node_ids, _line_eids = beam_two_stage
    director = ResultsDirector(results)
    win = _FakeWindow()

    payload = serialize_session(
        specs=[
            DiagramSpec(
                kind="contour",
                selector=SlabSelector(component="displacement_z"),
                style=ContourStyle(topology="nodes"),
            ),
            DiagramSpec(
                kind="line_force",
                selector=SlabSelector(component="bending_moment_z"),
                style=LineForceStyle(),
            ),
        ],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
    )
    payload.pop("geometries", None)

    _controller(director, results).apply(deserialize_session(payload), win)

    assert _membership(director.geometries) == {
        ("Geometry 1", "Restored"): ("contour", "line_force"),
    }
    assert director.geometries.active.compositions.active.name == "Restored"


def test_restored_session_scrubs_on_the_pump_seam(beam_two_stage, backend):
    """The two halves of this file meet: restore a session, then scrub
    the restored graph through the same ``PumpSet``.

    This is the round trip the plan row asks for — a session is only
    restored if the diagrams it rebuilt actually step. The restored
    contour must carry step-2 values after a pump at the restored
    cursor.
    """
    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)

    payload = serialize_session(
        specs=[
            DiagramSpec(
                kind="contour",
                selector=SlabSelector(component="displacement_z"),
                style=ContourStyle(topology="nodes"),
            ),
        ],
        results_path=Path("run.h5"),
        fem_snapshot_id=None,
        active_stage_id=_stage_id(results, "push"),
        active_step=2,
    )
    _controller(director, results).apply(
        deserialize_session(payload), _FakeWindow(),
    )

    restored = list(director.registry.diagrams())
    assert len(restored) == 1
    layer = restored[0]
    layer.attach(backend, results.fem, scene)

    _pumps(director, scene).pump_step(None)

    assert director.step_index == 2
    np.testing.assert_array_equal(
        np.asarray(layer._layer.field_named("displacement_z").values),
        _nodal_disp_z(layer._fem_ids_to_read, 2),
    )


# =====================================================================
# Test 3 — a saved THRESHOLD restored onto a live director (ADR 0084 D1)
# =====================================================================


def _threshold_payload(results, tmp_path: Path, *, snapshot):
    """A two-geometry session where only the SECOND carries a threshold.

    The asymmetry is the point: the restore path re-matches geometries
    by name (saved ids are stale UUIDs), so a spec that reattached by
    id would land on the wrong geometry — or on none.
    """
    return serialize_session(
        specs=[],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        active_stage_id=_stage_id(results, "push"),
        geometries=[
            GeometrySnapshot(id="g0", name="Plain", visible=True),
            GeometrySnapshot(
                id="g1", name="Filtered", visible=True,
                threshold=snapshot,
            ),
        ],
    )


def test_a_saved_threshold_reattaches_to_the_right_geometry(
    beam_two_stage, tmp_path,
):
    """save → deserialize → apply: the threshold comes back ACTIVE, with
    the same component / range / topology, on the geometry that owned
    it — under that geometry's LIVE id, not the stale saved one."""
    from apeGmsh.viewers.core.threshold_controller import ThresholdSettings
    from apeGmsh.viewers.diagrams._session import ThresholdSnapshot

    results, _node_ids, _line_eids = beam_two_stage
    director = ResultsDirector(results)

    session = deserialize_session(_threshold_payload(
        results, tmp_path,
        snapshot=ThresholdSnapshot(
            component="displacement_z", lo=-1.0, hi=2500.0, topology="nodes",
        ),
    ))
    _controller(director, results).apply(session, _FakeWindow())

    by_name = {geo.name: geo for geo in director.geometries.geometries}
    assert set(by_name) == {"Plain", "Filtered"}
    filtered = by_name["Filtered"]
    # The live id is a fresh UUID — nothing like the saved "g1".
    assert filtered.id != "g1"
    assert director.thresholds.settings_for(filtered.id) == ThresholdSettings(
        component="displacement_z", lo=-1.0, hi=2500.0, topology="nodes",
    )
    assert director.thresholds.settings_for(by_name["Plain"].id) is None
    assert director.thresholds.needs_refresh() is True


def test_a_legacy_session_restores_with_no_threshold(
    beam_two_stage, tmp_path,
):
    """Pre-v11 sessions carry no threshold; the restore must leave the
    controller empty (and therefore cost nothing on the STEP path)."""
    results, _node_ids, _line_eids = beam_two_stage
    director = ResultsDirector(results)

    payload = _threshold_payload(results, tmp_path, snapshot=None)
    payload["schema_version"] = 10
    for graw in payload["geometries"]:
        graw.pop("threshold", None)            # what a v10 save looks like

    _controller(director, results).apply(
        deserialize_session(payload), _FakeWindow(),
    )

    for geom in director.geometries.geometries:
        assert director.thresholds.settings_for(geom.id) is None
    assert director.thresholds.needs_refresh() is False


def test_a_restored_threshold_hides_cells_on_the_next_pump(
    beam_two_stage, backend,
):
    """The round trip that matters: restore, pump, and the cells the
    saved range excludes are actually hidden — at the restored cursor.

    ``displacement_z`` is ``node_id + 1000·step``, so a band that keeps
    only the low node ids at step 0 is checkable against the scene's
    own id→row maps rather than against a hard-coded guess about the
    fixture's mesh ordering.
    """
    from apeGmsh.viewers.core.element_visibility import ElementVisibility
    from apeGmsh.viewers.core.threshold_controller import SceneValueReader
    from apeGmsh.viewers.diagrams._session import ThresholdSnapshot

    results, _node_ids, _line_eids = beam_two_stage
    scene = build_fem_scene(results.fem)
    scene.element_visibility = ElementVisibility(scene.grid)
    director = ResultsDirector(results)
    director.thresholds.read_values = SceneValueReader(
        director=director, scene=scene,
    )

    # A band that splits the mesh: the median of each cell's LARGEST
    # node id. (Gmsh numbers corner nodes before interior ones, so ids
    # do not run along the beam — a cut at the median NODE id would
    # hide every cell and make the assertion vacuous.)
    conn = np.asarray(scene.grid.cell_connectivity)
    offsets = np.asarray(scene.grid.offset)
    scene_ids = np.asarray(scene.node_ids, dtype=np.float64)
    cut = float(np.median([
        scene_ids[conn[offsets[c]:offsets[c + 1]]].max()
        for c in range(scene.grid.n_cells)
    ]))
    session = deserialize_session(_threshold_payload(
        results, Path("."),
        snapshot=ThresholdSnapshot(
            component="displacement_z", lo=-1.0, hi=cut, topology="nodes",
        ),
    ))
    _controller(director, results).apply(session, _FakeWindow())
    director.set_step(0)

    filtered = next(
        geo for geo in director.geometries.geometries
        if geo.name == "Filtered"
    )
    _pumps(director, scene).refresh_threshold_for(filtered, scene)

    # Independent oracle: a cell survives iff EVERY one of its nodes has
    # ``node_id <= cut`` (values at step 0 are the node ids themselves).
    in_range = scene_ids <= cut
    expected = {
        c for c in range(scene.grid.n_cells)
        if not bool(in_range[conn[offsets[c]:offsets[c + 1]]].all())
    }
    hidden = set(
        np.flatnonzero(scene.element_visibility.hidden_mask()).tolist(),
    )
    assert hidden == expected
    # …and the band actually splits the mesh, so this is not a vacuous
    # "everything hidden" / "nothing hidden" pass.
    assert 0 < len(hidden) < scene.grid.n_cells
