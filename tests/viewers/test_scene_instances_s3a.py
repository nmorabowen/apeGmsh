"""ADR 0058 S3a — per-geometry spatial offset.

The offset is a pump-time term in the DEFORM primitive —
``reference + offset + scale·field`` — never an actor transform and
never baked into ``FEMSceneData.reference_points``, so the S2c
invariant (world coordinates == grid coordinates) holds and every
pick / overlay / box-projection path stays offset-correct with zero
change. Coverage:

* ``Geometry.offset`` field + owner mutator
  ``GeometryManager.set_offset`` (length-3 validate, float-coerce,
  no-op on equal, fires ``GEOMETRY_OFFSET_CHANGED`` with the geom id
  as payload); ``duplicate`` copies the offset.
* Dispatcher matrix row: offset changes run DEFORM only; the granular
  fire suppresses the same-tick omnibus; the RENDER-lane subscriber
  runs after the pump.
* ``_compose_substrate_points`` — the pump composition rule,
  including the byte-identical ``None`` fast-path at zero offset.
* ``ResultsViewer._on_geometry_offset_changed`` — node-tree
  invalidation + label-overlay rebuild.
* Box pick over an offset geometry's grid finds nodes at the OFFSET
  positions (grid.points projection is offset-correct by
  construction).
* Session persistence: schema v6 round-trips ``offset``; legacy
  sessions (no field) read ``(0, 0, 0)``.

The qt-marked test (local-only) drives a real viewer with two visible
geometries, one offset, and asserts distinct substrate positions, the
diagram fan-out, the fast-path restore at zero offset, deform+offset
composition, and a node pick on the offset actor snapping to the
OFFSET coordinate with the right geometry id.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from apeGmsh.viewers.diagrams._dispatch import (
    GEOMETRY_OFFSET_CHANGED,
)
from apeGmsh.viewers.diagrams._geometries import GeometryManager


# =====================================================================
# Geometry.offset + GeometryManager.set_offset (owner mutator)
# =====================================================================

def test_geometry_offset_defaults_zero():
    gm = GeometryManager()
    assert gm.active.offset == (0.0, 0.0, 0.0)
    other = gm.add("Geometry B", make_active=False)
    assert other.offset == (0.0, 0.0, 0.0)


def test_set_offset_fires_typed_event_with_geom_id_payload():
    gm = GeometryManager()
    geom = gm.active
    typed: list = []
    omnibus: list = []
    gm.subscribe_typed(lambda kind, payload: typed.append((kind, payload)))
    gm.subscribe(lambda: omnibus.append(True))

    # Ints coerce to floats.
    assert gm.set_offset(geom.id, (1, 2, 3)) is True
    assert geom.offset == (1.0, 2.0, 3.0)
    assert all(isinstance(c, float) for c in geom.offset)
    assert typed == [(GEOMETRY_OFFSET_CHANGED, geom.id)]
    assert len(omnibus) == 1


def test_set_offset_noop_when_unchanged_or_unknown():
    gm = GeometryManager()
    geom = gm.active
    typed: list = []
    gm.subscribe_typed(lambda kind, payload: typed.append((kind, payload)))

    assert gm.set_offset(geom.id, (0.0, 0.0, 0.0)) is False   # default
    gm.set_offset(geom.id, (1.0, 2.0, 3.0))
    typed.clear()
    assert gm.set_offset(geom.id, (1, 2, 3)) is False          # equal
    assert gm.set_offset("no-such-id", (9.0, 9.0, 9.0)) is False
    assert typed == []
    assert geom.offset == (1.0, 2.0, 3.0)


def test_set_offset_validates_length_and_content():
    gm = GeometryManager()
    geom = gm.active
    with pytest.raises(ValueError):
        gm.set_offset(geom.id, (1.0, 2.0))
    with pytest.raises(ValueError):
        gm.set_offset(geom.id, (1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError):
        gm.set_offset(geom.id, ("x", "y", "z"))
    with pytest.raises(ValueError):
        gm.set_offset(geom.id, None)
    assert geom.offset == (0.0, 0.0, 0.0)      # untouched on failure


def test_duplicate_copies_offset():
    gm = GeometryManager()
    geom = gm.active
    gm.set_offset(geom.id, (1.5, -2.0, 3.0))
    clone = gm.duplicate(geom.id)
    assert clone is not None
    assert clone.offset == (1.5, -2.0, 3.0)


# =====================================================================
# Dispatcher matrix — offset change runs DEFORM only
# =====================================================================

def test_offset_changed_matrix_row_runs_deform_only():
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
    disp.fire(GEOMETRY_OFFSET_CHANGED, payload="g1")
    assert calls == ["deform", "render"]


def test_offset_changed_suppresses_same_tick_omnibus():
    from apeGmsh.viewers.diagrams._dispatch import (
        GEOMETRIES_CHANGED,
        Dispatcher,
    )

    calls: list[str] = []
    disp = Dispatcher(
        MagicMock(),
        pump_deform=lambda layer: calls.append("deform"),
        pump_gate=lambda: calls.append("gate"),
        render=lambda: calls.append("render"),
        defer_fn=lambda fn: fn(),
    )
    disp.fire(GEOMETRY_OFFSET_CHANGED, payload="g1")
    disp.fire(GEOMETRIES_CHANGED)
    # One deform — the omnibus was suppressed.
    assert calls == ["deform", "render"]


def test_offset_render_lane_subscriber_runs_after_deform_pump():
    """The node-tree invalidation rides the RENDER lane — it must see
    the grid points the pump just moved, before the closing render."""
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher, Lane

    calls: list = []
    disp = Dispatcher(
        MagicMock(),
        pump_deform=lambda layer: calls.append("deform"),
        render=lambda: calls.append("render"),
        defer_fn=lambda fn: fn(),
    )
    disp.subscribe(
        GEOMETRY_OFFSET_CHANGED,
        lambda kind, payload: calls.append(("handler", payload)),
        lane=Lane.RENDER,
    )
    disp.fire(GEOMETRY_OFFSET_CHANGED, payload="g9")
    assert calls == ["deform", ("handler", "g9"), "render"]


# =====================================================================
# _compose_substrate_points — the pump composition rule
# =====================================================================

_REF = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64,
)


def _compose(*args):
    from apeGmsh.viewers.results_viewer import _compose_substrate_points
    return _compose_substrate_points(*args)


def test_compose_zero_offset_no_field_keeps_none_fast_path():
    # The legacy contract, byte-identical at zero offset: nothing to
    # apply → None (pump resets to reference, diagrams told None).
    assert _compose(_REF, (0.0, 0.0, 0.0), None, 1.0) is None
    assert _compose(_REF, None, None, 1.0) is None


def test_compose_offset_only_returns_reference_plus_offset():
    out = _compose(_REF, (1.0, 2.0, 3.0), None, 1.0)
    np.testing.assert_allclose(out, _REF + [1.0, 2.0, 3.0])
    # reference_points stays the pristine model baseline.
    np.testing.assert_allclose(_REF[0], [0.0, 0.0, 0.0])


def test_compose_field_only_matches_legacy_deform():
    field = np.full_like(_REF, 0.5)
    out = _compose(_REF, (0.0, 0.0, 0.0), field, 2.0)
    np.testing.assert_allclose(out, _REF + 1.0)


def test_compose_offset_and_field_compose():
    field = np.zeros_like(_REF)
    field[:, 0] = 1.0
    out = _compose(_REF, (1.0, 2.0, 3.0), field, 2.0)
    np.testing.assert_allclose(out, _REF + [3.0, 2.0, 3.0])


# =====================================================================
# ResultsViewer._on_geometry_offset_changed — node-tree invalidation
# + label rebuild (bound onto a stub namespace, no Qt)
# =====================================================================

class _NS:
    pass


def _offset_handler_ns(scene):
    from apeGmsh.viewers.results_viewer import ResultsViewer

    gm = GeometryManager()
    geom = gm.active
    director = SimpleNamespace(
        geometries=gm,
        scene_for=lambda g: scene if g is not None else None,
    )
    ns = _NS()
    ns._director = director
    handler = ResultsViewer._on_geometry_offset_changed.__get__(ns)
    return ns, gm, geom, handler


def test_offset_change_drops_cached_node_tree():
    scene = SimpleNamespace(node_tree=object())
    _, _, geom, handler = _offset_handler_ns(scene)
    handler(GEOMETRY_OFFSET_CHANGED, geom.id)
    assert scene.node_tree is None


def test_offset_change_rebuilds_visible_label_overlays():
    scene = SimpleNamespace(node_tree=None)
    ns, _, geom, handler = _offset_handler_ns(scene)
    rebuilt: list = []
    ns._node_label_actor = object()
    ns._element_label_actor = object()
    ns._set_node_id_labels = lambda v: rebuilt.append(("node", v))
    ns._set_element_id_labels = lambda v: rebuilt.append(("element", v))
    handler(GEOMETRY_OFFSET_CHANGED, geom.id)
    assert rebuilt == [("node", True), ("element", True)]


def test_offset_handler_tolerates_unknown_geometry_and_no_director():
    scene = SimpleNamespace(node_tree=object())
    ns, _, _, handler = _offset_handler_ns(scene)
    handler(GEOMETRY_OFFSET_CHANGED, "no-such-id")   # find → None
    assert scene.node_tree is not None               # untouched
    ns._director = None
    handler(GEOMETRY_OFFSET_CHANGED, "anything")     # early return


# =====================================================================
# Box pick — offset geometry's nodes found at offset positions
# (S2c stub-backend pattern; grid.points carry the pump-baked offset)
# =====================================================================

class _StubBackend:
    def __init__(self, project=None) -> None:
        self._project = project
        self.on_pick = None
        self.on_box = None

    def install(self, *, on_pick, on_hover=None, on_box=None) -> None:
        self.on_pick = on_pick
        self.on_box = on_box

    def project_points(self, pts):
        if self._project is not None:
            return self._project(pts)
        return np.asarray(pts, dtype=np.float64)[:, :2]

    def uninstall(self) -> None:
        pass

    def fire_box(self, box) -> None:
        from apeGmsh.viewers.scene_ir import BoxGesture
        self.on_box(BoxGesture(box=box, crossing=box[2] < box[0]))


class _Grid:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.cell_data = {}


class _Scene:
    """Minimal FEMSceneData stand-in (box-pick read surface)."""

    def __init__(self, *, node_ids=None, grid=None) -> None:
        self.cell_to_element_id = np.asarray([], dtype=np.int64)
        self.node_ids = np.asarray(
            node_ids if node_ids is not None else [10, 20, 30],
            dtype=np.int64,
        )
        self.grid = grid
        self.cell_dim = np.asarray([], dtype=np.int8)
        self.element_id_to_cell = {}
        self.pick_engine = None


def test_box_pick_finds_offset_geometry_nodes_at_offset_positions():
    from apeGmsh.viewers.core.results_pick import install_results_pick

    proj = lambda pts: np.asarray(pts, dtype=np.float64)[:, :2]  # noqa: E731
    offset = np.array([10.0, 20.0, 0.0])
    reference = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]])
    # The DEFORM pump baked the offset into the grid points.
    scene_b = _Scene(node_ids=[30, 40], grid=_Grid(reference + offset))
    backend = _StubBackend(project=proj)
    boxes: list = []
    install_results_pick(
        None, lambda r: None, scene=_Scene(grid=_Grid(reference)),
        on_box_pick=boxes.append, pick_backend=backend,
        scene_resolver=lambda prop_id: (
            ("gB", scene_b) if prop_id is None else None
        ),
    )
    # Box around node 30's OFFSET position → found.
    backend.fire_box((9.0, 19.0, 11.0, 21.0))
    assert boxes[0].ids.tolist() == [30]
    assert boxes[0].geometry_id == "gB"
    # Box around its REFERENCE position → nothing (it moved).
    backend.fire_box((-1.0, -1.0, 1.0, 1.0))
    assert boxes[1].ids.size == 0


# =====================================================================
# Session persistence — schema v6 ``offset``
# =====================================================================

def _make_contour_spec():
    from apeGmsh.viewers.diagrams._base import DiagramSpec
    from apeGmsh.viewers.diagrams._selectors import SlabSelector
    from apeGmsh.viewers.diagrams._styles import ContourStyle

    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_x"),
        style=ContourStyle(),
    )


def test_new_session_round_trips_offset(tmp_path: Path):
    from apeGmsh.viewers.diagrams._session import (
        GeometrySnapshot,
        load_session,
        save_session,
    )

    geoms = [
        GeometrySnapshot(id="g0", name="A"),
        GeometrySnapshot(id="g1", name="B", offset=(1.5, -2.0, 3.0)),
    ]
    saved = save_session(
        specs=[_make_contour_spec()],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        geometries=geoms,
    )
    session = load_session(saved)
    assert session.geometries[0].offset == (0.0, 0.0, 0.0)
    assert session.geometries[1].offset == (1.5, -2.0, 3.0)


def test_legacy_session_without_offset_deserializes_to_zero(
    tmp_path: Path,
):
    """Pre-v6 sessions carry no ``offset`` key — snapshots read the
    zero offset (additive-field rule, same as ``visible`` in v5)."""
    import json

    from apeGmsh.viewers.diagrams._session import (
        load_session,
        serialize_spec,
    )

    payload = {
        "schema_version": 5,
        "results_path": str(tmp_path / "run.h5"),
        "fem_snapshot_id": None,
        "saved_at": "",
        "geometries": [
            {
                "id": "g0",
                "name": "Geometry 1",
                "deform_enabled": False,
                "visible": True,
                "active_composition_id": None,
                "compositions": [],
            },
        ],
        "diagrams": [serialize_spec(_make_contour_spec())],
    }
    target = tmp_path / "v5.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    session = load_session(target)
    assert session.geometries[0].offset == (0.0, 0.0, 0.0)
    # Other fields keep their historical behavior.
    assert session.geometries[0].visible is True


def test_malformed_offset_degrades_to_zero():
    from apeGmsh.viewers.diagrams._session import _deserialize_geometry

    bad_len = _deserialize_geometry({"name": "G", "offset": [1.0, 2.0]})
    assert bad_len.offset == (0.0, 0.0, 0.0)
    bad_type = _deserialize_geometry({"name": "G", "offset": "nope"})
    assert bad_type.offset == (0.0, 0.0, 0.0)


# =====================================================================
# Qt — offset geometry on a real viewer (local-only; -m qt)
# =====================================================================

@pytest.fixture
def deforming_results(g, tmp_path: Path):
    """Tiny native Results with a non-zero displacement field."""
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

    path = tmp_path / "s3a.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": np.ones((3, n_nodes)),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


class _RecordingLayer:
    """Minimal registry-compatible layer recording substrate syncs."""

    kind = "stub"

    def __init__(self) -> None:
        self._attached = False
        self.is_visible = True
        self.synced: list = []

    @property
    def is_attached(self) -> bool:
        return self._attached

    def attach(self, backend, view, scene=None) -> None:
        self._attached = True

    def detach(self) -> None:
        self._attached = False

    def update_to_step(self, step) -> None:
        pass

    def apply_effective_visibility(self, desired) -> None:
        pass

    def sync_substrate_points(self, pts, scene) -> None:
        self.synced.append(
            None if pts is None else np.asarray(pts).copy()
        )


@pytest.mark.qt
def test_offset_geometry_renders_and_picks_at_offset(deforming_results):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import ResultsViewer
    from apeGmsh.viewers.scene_ir import PickHit, PickModifiers

    viewer = ResultsViewer(
        deforming_results, title="s3a-offset",
        restore_session=False, save_session=False,
    )
    seen: dict = {}
    OFFSET = np.array([1.0, 2.0, 3.0])

    def _drive_then_close():
        try:
            director = viewer._director
            geoms = director.geometries
            geom_a = geoms.active
            scene_a = director.scene_for(geom_a)
            geom_b = geoms.add("Geometry B", make_active=False)
            # A stub layer owned by B records the DEFORM fan-out.
            comp = geom_b.compositions.add(name="C", make_active=False)
            layer = _RecordingLayer()
            director.registry.add(layer)
            geom_b.compositions.add_layer(comp.id, layer)
            scene_b = director.scene_for(geom_b)
            layer.synced.clear()

            # ── Offset B (deform off) — pump-time translation. ──
            geoms.set_offset(geom_b.id, tuple(OFFSET))
            seen["b_at_offset"] = np.allclose(
                np.asarray(scene_b.grid.points),
                scene_b.reference_points + OFFSET,
            )
            seen["a_at_reference"] = np.allclose(
                np.asarray(scene_a.grid.points), scene_a.reference_points,
            )
            # reference_points stays the pristine baseline.
            seen["b_reference_pristine"] = np.allclose(
                scene_b.reference_points, scene_a.reference_points,
            )
            # B's diagram received the offset points through the hook.
            seen["layer_got_offset_pts"] = (
                len(layer.synced) > 0
                and layer.synced[-1] is not None
                and np.allclose(
                    layer.synced[-1],
                    scene_b.reference_points + OFFSET,
                )
            )
            # Both substrate pairs render.
            pair_a = viewer._scene_actors[geom_a.id]
            pair_b = viewer._scene_actors[geom_b.id]
            seen["both_visible"] = all(
                bool(x.GetVisibility()) for x in (*pair_a, *pair_b)
            )

            # ── Node-tree invalidation on offset change. ──
            scene_b.ensure_node_tree()
            had_tree = scene_b.node_tree is not None
            geoms.set_offset(geom_b.id, (4.0, 5.0, 6.0))
            seen["node_tree_dropped"] = (
                had_tree and scene_b.node_tree is None
            )
            geoms.set_offset(geom_b.id, tuple(OFFSET))

            # ── Node pick on B's actor: world == grid holds — the
            # snap lands at the OFFSET coordinate, carrying B's id. ──
            fill_b, _wf_b = viewer._scene_actors[geom_b.id]
            captured: list = []
            viewer._probe_overlay.on_point_result = captured.append
            b_node0 = np.asarray(scene_b.grid.points)[0]
            viewer._pick_controller.set_mode("node")
            backend = viewer._pick_controller._backend
            backend._on_pick(
                PickHit(
                    world=tuple(b_node0 + 0.01),
                    cell_id=0,
                    prop_id=id(fill_b),
                ),
                PickModifiers(),
            )
            result = captured[-1] if captured else None
            seen["pick_geometry_id"] = (
                result is not None
                and result.geometry_id == geom_b.id
            )
            seen["pick_snaps_to_offset_coord"] = (
                result is not None
                and float(
                    np.linalg.norm(
                        np.asarray(result.closest_coord) - b_node0,
                    )
                ) < 1e-9
            )

            # ── Deform + offset compose. ──
            geoms.set_deformation(
                geom_b.id, enabled=True,
                field="displacement", scale=2.0,
            )
            expected = (
                scene_b.reference_points
                + OFFSET
                + 2.0 * np.array([1.0, 0.0, 0.0])
            )
            seen["deform_offset_compose"] = np.allclose(
                np.asarray(scene_b.grid.points), expected,
            )
            geoms.set_deformation(geom_b.id, enabled=False)

            # ── Offset back to zero → None fast-path restored. ──
            layer.synced.clear()
            geoms.set_offset(geom_b.id, (0.0, 0.0, 0.0))
            seen["fast_path_restored"] = (
                len(layer.synced) > 0 and layer.synced[-1] is None
            )
            seen["b_back_at_reference"] = np.allclose(
                np.asarray(scene_b.grid.points), scene_b.reference_points,
            )
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen.get("b_at_offset") is True
    assert seen.get("a_at_reference") is True
    assert seen.get("b_reference_pristine") is True
    assert seen.get("layer_got_offset_pts") is True
    assert seen.get("both_visible") is True
    assert seen.get("node_tree_dropped") is True
    assert seen.get("pick_geometry_id") is True
    assert seen.get("pick_snaps_to_offset_coord") is True
    assert seen.get("deform_offset_compose") is True
    assert seen.get("fast_path_restored") is True
    assert seen.get("b_back_at_reference") is True
