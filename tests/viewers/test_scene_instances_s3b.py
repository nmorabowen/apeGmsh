"""ADR 0058 S3b — per-geometry stage pin.

A geometry pinned to stage S shows S's state at the ONE global step
cursor clamped into S's range (``director.local_step_for_stage`` —
core ruling 2; per-geometry step cursors were ADR-rejected). The pin
scopes three read paths under one *pinned-or-active* rule:

* substrate deform — ``_read_deform_field(stage_id=)`` /
  ``_compute_deformed_pts`` resolve the pin + clamped step;
* the geometry's diagrams — the registry stamps a
  ``stage_pin_resolver`` (S2b ``bar_prefix_resolver`` mirror) consumed
  by ``Diagram._effective_stage_id`` / ``_scoped_results``; an
  explicit per-diagram ``spec.stage_id`` wins (the two pins compose);
  the STEP pump pushes per-diagram effective steps;
* per-scene stage-activation masks —
  ``StageActivationController.mask_for_stage_id`` (parameterized
  sibling of ``current_mask``).

State lives on ``Geometry.stage_id`` with owner mutator
``GeometryManager.set_stage_pin`` firing the granular
``GEOMETRY_STAGE_PIN_CHANGED`` ({STEP, DEFORM} matrix row); a pin
change also re-attaches that geometry's attached diagrams via the
director's typed GeometryManager observer (GEOMETRY_REMOVED
precedent). Session schema v7 adds ``GeometrySnapshot.stage_id``
(legacy reads None = follow the active stage).

The qt-marked test (local-only) drives a real viewer over a TWO-stage
file: geometry B pinned to stage "grav" keeps grav's field (at the
clamped cursor) while the active stage "push" scrubs; its diagram
receives the clamped step; unpinning re-follows the active stage.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from apeGmsh.viewers.diagrams._dispatch import (
    GEOMETRY_STAGE_PIN_CHANGED,
)
from apeGmsh.viewers.diagrams._geometries import GeometryManager


# =====================================================================
# Geometry.stage_id + GeometryManager.set_stage_pin (owner mutator)
# =====================================================================

def test_geometry_stage_pin_defaults_none():
    gm = GeometryManager()
    assert gm.active.stage_id is None
    other = gm.add("Geometry B", make_active=False)
    assert other.stage_id is None


def test_set_stage_pin_fires_typed_event_with_geom_id_payload():
    gm = GeometryManager()
    geom = gm.active
    typed: list = []
    omnibus: list = []
    gm.subscribe_typed(lambda kind, payload: typed.append((kind, payload)))
    gm.subscribe(lambda: omnibus.append(True))

    assert gm.set_stage_pin(geom.id, "grav") is True
    assert geom.stage_id == "grav"
    assert typed == [(GEOMETRY_STAGE_PIN_CHANGED, geom.id)]
    assert len(omnibus) == 1


def test_set_stage_pin_none_clears_the_pin():
    gm = GeometryManager()
    geom = gm.active
    gm.set_stage_pin(geom.id, "grav")
    typed: list = []
    gm.subscribe_typed(lambda kind, payload: typed.append((kind, payload)))
    assert gm.set_stage_pin(geom.id, None) is True
    assert geom.stage_id is None
    assert typed == [(GEOMETRY_STAGE_PIN_CHANGED, geom.id)]


def test_set_stage_pin_noop_when_unchanged_or_unknown():
    gm = GeometryManager()
    geom = gm.active
    typed: list = []
    gm.subscribe_typed(lambda kind, payload: typed.append((kind, payload)))

    assert gm.set_stage_pin(geom.id, None) is False      # default
    gm.set_stage_pin(geom.id, "grav")
    typed.clear()
    assert gm.set_stage_pin(geom.id, "grav") is False    # equal
    assert gm.set_stage_pin("no-such-id", "push") is False
    assert typed == []
    assert geom.stage_id == "grav"


def test_duplicate_copies_stage_pin():
    gm = GeometryManager()
    geom = gm.active
    gm.set_stage_pin(geom.id, "grav")
    clone = gm.duplicate(geom.id)
    assert clone is not None
    assert clone.stage_id == "grav"


# =====================================================================
# Dispatcher matrix — pin change runs STEP + DEFORM
# =====================================================================

def test_stage_pin_changed_matrix_row_runs_step_and_deform():
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    calls: list[str] = []
    disp = Dispatcher(
        MagicMock(),
        pump_step=lambda layer: calls.append("step"),
        pump_deform=lambda layer: calls.append("deform"),
        pump_gate=lambda: calls.append("gate"),
        render=lambda: calls.append("render"),
        defer_fn=lambda fn: fn(),
    )
    disp.fire(GEOMETRY_STAGE_PIN_CHANGED, payload="g1")
    assert calls == ["step", "deform", "render"]


def test_stage_pin_changed_suppresses_same_tick_omnibus():
    from apeGmsh.viewers.diagrams._dispatch import (
        GEOMETRIES_CHANGED,
        Dispatcher,
    )

    calls: list[str] = []
    disp = Dispatcher(
        MagicMock(),
        pump_step=lambda layer: calls.append("step"),
        pump_deform=lambda layer: calls.append("deform"),
        pump_gate=lambda: calls.append("gate"),
        render=lambda: calls.append("render"),
        defer_fn=lambda fn: fn(),
    )
    disp.fire(GEOMETRY_STAGE_PIN_CHANGED, payload="g1")
    disp.fire(GEOMETRIES_CHANGED)
    # No gate — the omnibus was suppressed by the granular fire.
    assert calls == ["step", "deform", "render"]


def test_stage_pin_render_lane_subscriber_runs_after_pumps():
    """The per-scene LAYER_STAGE mask resync rides the RENDER lane —
    after the STEP/DEFORM pumps, before the closing render."""
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher, Lane

    calls: list = []
    disp = Dispatcher(
        MagicMock(),
        pump_step=lambda layer: calls.append("step"),
        pump_deform=lambda layer: calls.append("deform"),
        render=lambda: calls.append("render"),
        defer_fn=lambda fn: fn(),
    )
    disp.subscribe(
        GEOMETRY_STAGE_PIN_CHANGED,
        lambda kind, payload: calls.append(("handler", payload)),
        lane=Lane.RENDER,
    )
    disp.fire(GEOMETRY_STAGE_PIN_CHANGED, payload="g9")
    assert calls == ["step", "deform", ("handler", "g9"), "render"]


# =====================================================================
# director.local_step_for_stage — clamp the global cursor (ruling 2)
# =====================================================================

def _local_step(ns):
    from apeGmsh.viewers.diagrams._director import ResultsDirector
    return ResultsDirector.local_step_for_stage.__get__(ns)


class _StubStageResults:
    """``results.stage(sid).n_steps`` surface for the clamp helper."""

    def __init__(self, n_steps_by_id):
        self._n = dict(n_steps_by_id)

    def stage(self, sid):
        if sid not in self._n:
            raise KeyError(sid)
        return SimpleNamespace(n_steps=self._n[sid])


def _single_stage_ns(step_index, n_steps_by_id):
    return SimpleNamespace(
        _combined_active=False,
        _real_stages=[],
        _combined_boundaries=np.array([], dtype=np.int64),
        _step_index=int(step_index),
        _results=_StubStageResults(n_steps_by_id),
    )


def test_local_step_single_stage_within_range_is_identity():
    fn = _local_step(_single_stage_ns(1, {"grav": 3}))
    assert fn("grav") == 1


def test_local_step_single_stage_clamps_past_end_of_history():
    # Active stage is longer — the pinned geometry holds at its end.
    fn = _local_step(_single_stage_ns(7, {"grav": 3}))
    assert fn("grav") == 2


def test_local_step_single_stage_unreadable_stage_clamps_to_zero():
    fn = _local_step(_single_stage_ns(5, {"grav": 3}))
    assert fn("no-such-stage") == 0
    # A stage with zero captured steps also clamps to 0.
    fn0 = _local_step(_single_stage_ns(5, {"empty": 0}))
    assert fn0("empty") == 0


def _combined_ns(global_step):
    """Two real stages: s1 = 3 steps [0..2], s2 = 5 steps [3..7]."""
    return SimpleNamespace(
        _combined_active=True,
        _real_stages=[
            SimpleNamespace(id="s1", n_steps=3),
            SimpleNamespace(id="s2", n_steps=5),
        ],
        _combined_boundaries=np.array([0, 3], dtype=np.int64),
        _step_index=int(global_step),
        _results=_StubStageResults({}),
    )


def test_local_step_combined_inside_pinned_segment():
    assert _local_step(_combined_ns(4))("s2") == 1
    assert _local_step(_combined_ns(1))("s1") == 1


def test_local_step_combined_freezes_outside_pinned_segment():
    # Cursor before the pinned segment → holds at its first step.
    assert _local_step(_combined_ns(1))("s2") == 0
    # Cursor past the pinned segment → holds at its last step.
    assert _local_step(_combined_ns(6))("s1") == 2
    assert _local_step(_combined_ns(99))("s2") == 4


# =====================================================================
# Diagram._effective_stage_id / _scoped_results — the two pins compose
# =====================================================================

def _make_spec(stage_id=None):
    from apeGmsh.viewers.diagrams._base import DiagramSpec
    from apeGmsh.viewers.diagrams._selectors import SlabSelector
    from apeGmsh.viewers.diagrams._styles import ContourStyle

    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_x"),
        style=ContourStyle(),
        stage_id=stage_id,
    )


def _make_diagram(spec, results=None):
    from apeGmsh.viewers.diagrams._contour import ContourDiagram
    return ContourDiagram(spec, results if results is not None else MagicMock())


def test_effective_stage_id_none_without_spec_pin_or_resolver():
    d = _make_diagram(_make_spec())
    assert d._effective_stage_id() is None


def test_effective_stage_id_resolves_geometry_pin_via_resolver():
    d = _make_diagram(_make_spec())
    d._stage_pin_resolver = lambda diagram: "grav"
    assert d._effective_stage_id() == "grav"


def test_effective_stage_id_spec_pin_wins_over_geometry_pin():
    d = _make_diagram(_make_spec(stage_id="push"))
    d._stage_pin_resolver = lambda diagram: "grav"
    assert d._effective_stage_id() == "push"


def test_effective_stage_id_survives_raising_resolver():
    d = _make_diagram(_make_spec())

    def _boom(diagram):
        raise RuntimeError("resolver exploded")

    d._stage_pin_resolver = _boom
    assert d._effective_stage_id() is None


def test_scoped_results_follows_geometry_pin():
    results = MagicMock()
    scoped = object()
    results.stage.return_value = scoped
    d = _make_diagram(_make_spec(), results)
    d._stage_pin_resolver = lambda diagram: "grav"
    assert d._scoped_results() is scoped
    results.stage.assert_called_once_with("grav")


def test_scoped_results_unpinned_returns_results_as_is():
    results = MagicMock()
    d = _make_diagram(_make_spec(), results)
    assert d._scoped_results() is results
    results.stage.assert_not_called()


def test_reactions_scoped_results_uses_geometry_pin_defensively():
    """The ReactionsDiagram override goes through the SAME
    ``_effective_stage_id`` helper (so the two can't drift) but
    degrades a bad stage id to None instead of raising."""
    from apeGmsh.viewers.diagrams._reactions import ReactionsDiagram

    d = object.__new__(ReactionsDiagram)
    d.spec = _make_spec()                      # no spec pin
    results = MagicMock()
    scoped = object()
    results.stage.return_value = scoped
    d._results = results
    d._stage_pin_resolver = lambda diagram: "grav"
    assert d._scoped_results() is scoped
    results.stage.assert_called_once_with("grav")

    # Bad stage id → None (defensive), not an exception.
    bad = MagicMock()
    bad.stage.side_effect = KeyError("no-such-stage")
    d._results = bad
    assert d._scoped_results() is None

    # Spec pin still wins over the geometry pin.
    d.spec = _make_spec(stage_id="push")
    d._results = results
    results.stage.reset_mock()
    d._scoped_results()
    results.stage.assert_called_once_with("push")


# =====================================================================
# Registry stamps the stage-pin resolver (S2b bar-prefix mirror)
# =====================================================================

def test_registry_stamps_stage_pin_resolver_on_attach():
    from tests.viewers.conftest import RecordingBackend

    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    class _StubDiagram:
        def __init__(self) -> None:
            self._attached = False
            self.kind = "stub"

        @property
        def is_attached(self) -> bool:
            return self._attached

        def attach(self, backend, view, scene=None) -> None:
            self._attached = True

        def detach(self) -> None:
            self._attached = False

    resolver = lambda d: "grav"    # noqa: E731
    reg = DiagramRegistry()
    # Diagram added before bind picks the resolver up at bind().
    early = _StubDiagram()
    reg.add(early)
    reg.bind(
        RecordingBackend(), view=object(), scene=object(),
        stage_pin_resolver=resolver,
    )
    assert early._stage_pin_resolver is resolver
    # Diagram added after bind is stamped at add().
    late = _StubDiagram()
    reg.add(late)
    assert late._stage_pin_resolver is resolver
    # No resolver bound → nothing stamped.
    reg2 = DiagramRegistry()
    bare = _StubDiagram()
    reg2.add(bare)
    assert not hasattr(bare, "_stage_pin_resolver")


# =====================================================================
# Pin-change reattach — only the pinned geometry's attached diagrams
# cycle (director typed-observer walk, GEOMETRY_REMOVED precedent)
# =====================================================================

class _CountingDiagram:
    def __init__(self, attached=True) -> None:
        self._attached = bool(attached)
        self.kind = "stub"
        self.is_visible = True
        self.detach_count = 0
        self.attach_count = 0

    @property
    def is_attached(self) -> bool:
        return self._attached

    def attach(self, backend, view, scene=None) -> None:
        self._attached = True
        self.attach_count += 1

    def detach(self) -> None:
        self._attached = False
        self.detach_count += 1


def _reattach_ns(diagrams, gm, bound=True):
    from apeGmsh.viewers.diagrams._director import ResultsDirector

    registry = SimpleNamespace(
        is_bound=bound,
        backend=object(),
        diagrams=lambda: list(diagrams),
    )
    ns = SimpleNamespace(
        _registry=registry,
        _geometries=gm,
        view=object(),
        _scene_for_diagram=lambda d: None,
    )
    return ResultsDirector._reattach_for_stage_pin.__get__(ns)


def _two_geometries_with_layers():
    gm = GeometryManager()
    geom_a = gm.active
    geom_b = gm.add("Geometry B", make_active=False)
    d_a = _CountingDiagram()
    d_b = _CountingDiagram()
    for geom, d in ((geom_a, d_a), (geom_b, d_b)):
        comp = geom.compositions.add(name="C", make_active=False)
        geom.compositions.add_layer(comp.id, d)
    return gm, geom_a, geom_b, d_a, d_b


def test_reattach_cycles_only_the_pinned_geometrys_diagrams():
    gm, _geom_a, geom_b, d_a, d_b = _two_geometries_with_layers()
    fn = _reattach_ns([d_a, d_b], gm)
    fn(GEOMETRY_STAGE_PIN_CHANGED, geom_b.id)
    assert (d_b.detach_count, d_b.attach_count) == (1, 1)
    assert (d_a.detach_count, d_a.attach_count) == (0, 0)


def test_reattach_skips_unattached_diagrams_and_other_kinds():
    gm, _geom_a, geom_b, d_a, d_b = _two_geometries_with_layers()
    d_b._attached = False
    fn = _reattach_ns([d_a, d_b], gm)
    fn(GEOMETRY_STAGE_PIN_CHANGED, geom_b.id)
    assert (d_b.detach_count, d_b.attach_count) == (0, 0)

    d_b._attached = True
    fn("geometry_renamed", geom_b.id)            # other kind → no-op
    fn(GEOMETRY_STAGE_PIN_CHANGED, None)         # no payload → no-op
    assert (d_b.detach_count, d_b.attach_count) == (0, 0)


def test_reattach_noop_when_registry_unbound():
    gm, _geom_a, geom_b, _d_a, d_b = _two_geometries_with_layers()
    fn = _reattach_ns([d_b], gm, bound=False)
    fn(GEOMETRY_STAGE_PIN_CHANGED, geom_b.id)
    assert (d_b.detach_count, d_b.attach_count) == (0, 0)


def test_director_registers_the_stage_pin_observer_at_init():
    """The typed observer is wired in ``ResultsDirector.__init__``
    (before the viewer's dispatcher bridge — pumps land on fresh
    attachments)."""
    import inspect

    from apeGmsh.viewers.diagrams._director import ResultsDirector

    src = inspect.getsource(ResultsDirector.__init__)
    assert "_reattach_for_stage_pin" in src


# =====================================================================
# StageActivationController.mask_for_stage_id — per-pin masks
# =====================================================================

def _make_controller(enabled=True):
    from apeGmsh.viewers.data._stage_activation import (
        StageActivationController,
        StageActivationMap,
    )

    mask1 = np.array([True, False, False])
    mask2 = np.array([False, False, True])
    final = np.array([False, True, False])
    amap = StageActivationMap(
        hidden_by_name={"grav": mask1, "push": mask2},
        final_hidden=final,
        n_cells=3,
        unmapped_tags=0,
    )
    name_for_id = {"id-grav": "grav", "id-push": "push"}.get
    ctrl = StageActivationController(
        SimpleNamespace(
            set_layer=lambda *a: None, clear_layer=lambda *a: None,
        ),
        amap,
        stage_name_for_id=name_for_id,
        combined_stage_id="__all__",
    )
    ctrl.set_enabled(enabled)
    return ctrl, mask1, mask2, final


def test_mask_for_stage_id_resolves_explicit_stage():
    ctrl, mask1, mask2, _final = _make_controller()
    np.testing.assert_array_equal(ctrl.mask_for_stage_id("id-grav"), mask1)
    np.testing.assert_array_equal(ctrl.mask_for_stage_id("id-push"), mask2)


def test_mask_for_stage_id_pinned_mask_holds_across_stage_change():
    """The S3b point: the PINNED stage's mask is independent of the
    controller's remembered (active) stage."""
    ctrl, mask1, mask2, _final = _make_controller()
    ctrl.on_stage_changed("id-push")
    np.testing.assert_array_equal(ctrl.current_mask(), mask2)
    # The pinned geometry still resolves ITS stage's mask.
    np.testing.assert_array_equal(ctrl.mask_for_stage_id("id-grav"), mask1)


def test_mask_for_stage_id_combined_maps_to_final_configuration():
    ctrl, _m1, _m2, final = _make_controller()
    np.testing.assert_array_equal(ctrl.mask_for_stage_id("__all__"), final)


def test_mask_for_stage_id_fail_soft_and_disabled():
    ctrl, _m1, _m2, _final = _make_controller()
    assert ctrl.mask_for_stage_id(None) is None
    assert ctrl.mask_for_stage_id("no-such-id") is None    # unmatched
    ctrl.set_enabled(False)
    assert ctrl.mask_for_stage_id("id-grav") is None


def test_current_mask_delegates_to_mask_for_stage_id():
    ctrl, mask1, _m2, _final = _make_controller()
    ctrl.on_stage_changed("id-grav")
    np.testing.assert_array_equal(ctrl.current_mask(), mask1)
    ctrl.on_stage_changed(None)
    assert ctrl.current_mask() is None


# =====================================================================
# Session persistence — schema v7 ``stage_id``
# =====================================================================

def test_new_session_round_trips_stage_pin(tmp_path: Path):
    from apeGmsh.viewers.diagrams._session import (
        GeometrySnapshot,
        load_session,
        save_session,
    )

    geoms = [
        GeometrySnapshot(id="g0", name="A"),
        GeometrySnapshot(id="g1", name="B", stage_id="grav"),
    ]
    saved = save_session(
        specs=[_make_spec()],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        geometries=geoms,
    )
    session = load_session(saved)
    assert session.geometries[0].stage_id is None
    assert session.geometries[1].stage_id == "grav"


def test_legacy_session_without_stage_pin_deserializes_to_none(
    tmp_path: Path,
):
    """Pre-v7 sessions carry no ``stage_id`` key — snapshots read
    None = follow the active stage (additive-field rule, same as
    ``offset`` in v6)."""
    import json

    from apeGmsh.viewers.diagrams._session import (
        load_session,
        serialize_spec,
    )

    payload = {
        "schema_version": 6,
        "results_path": str(tmp_path / "run.h5"),
        "fem_snapshot_id": None,
        "saved_at": "",
        "geometries": [
            {
                "id": "g0",
                "name": "Geometry 1",
                "deform_enabled": False,
                "offset": [1.0, 2.0, 3.0],
                "visible": True,
                "active_composition_id": None,
                "compositions": [],
            },
        ],
        "diagrams": [serialize_spec(_make_spec())],
    }
    target = tmp_path / "v6.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    session = load_session(target)
    assert session.geometries[0].stage_id is None
    # Other fields keep their historical behavior.
    assert session.geometries[0].offset == (1.0, 2.0, 3.0)


# =====================================================================
# Qt — pinned geometry over a real two-stage file (local-only; -m qt)
# =====================================================================

@pytest.fixture
def two_stage_results(g, tmp_path: Path):
    """Tiny native Results with TWO stages of distinguishable
    displacement fields and different lengths: "grav" = 3 steps of
    +1·x̂, "push" = 5 steps of +2·x̂."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "s3b.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_x": np.ones((3, n_nodes))},
        )
        w.end_stage()
        sid = w.begin_stage(
            name="push", kind="static",
            time=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": 2.0 * np.ones((5, n_nodes)),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


class _RecordingLayer:
    """Minimal registry-compatible layer recording steps + cycles."""

    kind = "stub"

    def __init__(self) -> None:
        self._attached = False
        self.is_visible = True
        self.steps: list = []
        self.attach_count = 0
        self.detach_count = 0

    @property
    def is_attached(self) -> bool:
        return self._attached

    def attach(self, backend, view, scene=None) -> None:
        self._attached = True
        self.attach_count += 1

    def detach(self) -> None:
        self._attached = False
        self.detach_count += 1

    def update_to_step(self, step) -> None:
        self.steps.append(int(step))

    def apply_effective_visibility(self, desired) -> None:
        pass

    def sync_substrate_points(self, pts, scene) -> None:
        pass


@pytest.mark.qt
def test_pinned_geometry_holds_its_stage_while_active_scrubs(
    two_stage_results,
):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        two_stage_results, title="s3b-stage-pin",
        restore_session=False, save_session=False,
    )
    seen: dict = {}
    XHAT = np.array([1.0, 0.0, 0.0])

    def _drive_then_close():
        try:
            director = viewer._director
            geoms = director.geometries
            geom_a = geoms.active
            geom_b = geoms.add("Geometry B", make_active=False)
            grav_id = next(
                s.id for s in director.stages() if s.name == "grav"
            )

            # One recording layer per geometry (composition membership
            # FIRST, then registry — the S2a ownership join).
            layer_a, layer_b = _RecordingLayer(), _RecordingLayer()
            for geom, layer in ((geom_a, layer_a), (geom_b, layer_b)):
                comp = geom.compositions.add(name="C", make_active=False)
                geom.compositions.add_layer(comp.id, layer)
                director.registry.add(layer)

            for geom in (geom_a, geom_b):
                geoms.set_deformation(
                    geom.id, enabled=True,
                    field="displacement", scale=1.0,
                )
            scene_a = director.scene_for(geom_a)
            scene_b = director.scene_for(geom_b)

            # Active stage = "push" (5 steps), cursor at its step 4.
            director.set_stage("push")
            director.set_step(4)

            # ── Pin B to "grav": its substrate warps from grav's
            # field at the CLAMPED cursor while A keeps push's. ──
            layer_a.steps.clear()
            layer_b.steps.clear()
            b_cycles_before = layer_b.attach_count
            a_cycles_before = layer_a.attach_count
            geoms.set_stage_pin(geom_b.id, grav_id)

            seen["b_at_pinned_field"] = np.allclose(
                np.asarray(scene_b.grid.points),
                scene_b.reference_points + XHAT,        # grav = +1·x̂
            )
            seen["a_follows_active"] = np.allclose(
                np.asarray(scene_a.grid.points),
                scene_a.reference_points + 2.0 * XHAT,  # push = +2·x̂
            )
            # Only B's diagram cycled (pin-change reattach), and the
            # STEP pump pushed it the CLAMPED step (grav has 3 steps →
            # clamp(4) = 2) while A's got the raw cursor.
            seen["b_reattached"] = (
                layer_b.attach_count == b_cycles_before + 1
            )
            seen["a_not_reattached"] = (
                layer_a.attach_count == a_cycles_before
            )
            seen["b_layer_clamped_step"] = (
                len(layer_b.steps) > 0 and layer_b.steps[-1] == 2
            )
            seen["a_layer_raw_step"] = (
                len(layer_a.steps) > 0 and layer_a.steps[-1] == 4
            )

            # ── Scrub the active stage: B holds, A moves with it. ──
            director.set_step(1)
            seen["b_holds_while_scrubbing"] = np.allclose(
                np.asarray(scene_b.grid.points),
                scene_b.reference_points + XHAT,
            )
            seen["b_layer_step_within_range"] = (
                layer_b.steps[-1] == 1                  # clamp(1) = 1
            )

            # ── Unpin: B re-follows the active stage. ──
            director.set_step(4)
            geoms.set_stage_pin(geom_b.id, None)
            seen["b_refollows_after_unpin"] = np.allclose(
                np.asarray(scene_b.grid.points),
                scene_b.reference_points + 2.0 * XHAT,
            )
            seen["b_layer_raw_after_unpin"] = (
                layer_b.steps[-1] == 4
            )
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen.get("b_at_pinned_field") is True
    assert seen.get("a_follows_active") is True
    assert seen.get("b_reattached") is True
    assert seen.get("a_not_reattached") is True
    assert seen.get("b_layer_clamped_step") is True
    assert seen.get("a_layer_raw_step") is True
    assert seen.get("b_holds_while_scrubbing") is True
    assert seen.get("b_layer_step_within_range") is True
    assert seen.get("b_refollows_after_unpin") is True
    assert seen.get("b_layer_raw_after_unpin") is True
