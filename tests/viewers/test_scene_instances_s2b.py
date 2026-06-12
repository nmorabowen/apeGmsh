"""ADR 0058 S2b — concurrent geometry rendering.

After S2b every geometry with ``visible=True`` renders concurrently
(its substrate actors + its diagrams, each at its own deform state);
"active" remains only the editing target. Coverage:

* ``Geometry.visible`` flag + owner mutator
  ``GeometryManager.set_visible`` (fires
  ``GEOMETRY_VISIBILITY_CHANGED`` with the geom id as payload).
* Dispatcher matrix row: visibility flips run DEFORM + GATE.
* Gate truth table — a layer shows iff ``layer.is_visible AND
  composition gate AND owning_geometry.visible``
  (:func:`results_viewer._gate_visible_layer_ids`).
* Scalar-bar title prefix: registry-stamped resolver, "<geometry> —
  <component>" only while more than one geometry is visible.
* Session persistence: new sessions round-trip the flag; legacy
  sessions (no ``visible`` field) deserialize to None and the restore
  path maps None → "visible iff active".
* Outline geometry-row eye drives ``set_visible``.

The qt-marked test (local-only) drives a real viewer with two visible
geometries at different deform scales and asserts both substrate
grids carry their own configuration after the DEFORM pump.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from apeGmsh.viewers.diagrams._dispatch import (
    GEOMETRY_VISIBILITY_CHANGED,
)
from apeGmsh.viewers.diagrams._geometries import Geometry, GeometryManager


# =====================================================================
# Geometry.visible + GeometryManager.set_visible (owner mutator)
# =====================================================================

def test_geometry_visible_defaults_true():
    gm = GeometryManager()
    assert gm.active.visible is True
    other = gm.add("Geometry B", make_active=False)
    assert other.visible is True


def test_set_visible_fires_typed_event_with_geom_id_payload():
    gm = GeometryManager()
    geom = gm.active
    typed: list = []
    omnibus: list = []
    gm.subscribe_typed(lambda kind, payload: typed.append((kind, payload)))
    gm.subscribe(lambda: omnibus.append(True))

    assert gm.set_visible(geom.id, False) is True
    assert geom.visible is False
    assert typed == [(GEOMETRY_VISIBILITY_CHANGED, geom.id)]
    assert len(omnibus) == 1


def test_set_visible_noop_when_unchanged_or_unknown():
    gm = GeometryManager()
    geom = gm.active
    typed: list = []
    gm.subscribe_typed(lambda kind, payload: typed.append((kind, payload)))

    assert gm.set_visible(geom.id, True) is False     # already True
    assert gm.set_visible("no-such-id", False) is False
    assert typed == []
    assert geom.visible is True


def test_duplicate_copies_visible_flag():
    gm = GeometryManager()
    geom = gm.active
    gm.set_visible(geom.id, False)
    clone = gm.duplicate(geom.id)
    assert clone is not None
    assert clone.visible is False


# =====================================================================
# Dispatcher matrix — visibility flip runs DEFORM + GATE
# =====================================================================

def test_visibility_changed_matrix_row_runs_deform_and_gate():
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
    disp.fire(GEOMETRY_VISIBILITY_CHANGED, payload="g1")
    assert calls == ["deform", "gate", "render"]


def test_visibility_changed_suppresses_same_tick_omnibus():
    """``set_visible`` fires the granular kind, then the legacy
    omnibus chain fires GEOMETRIES_CHANGED — the dispatcher's one-tick
    guard must suppress the redundant pump (granular-kind contract)."""
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
    disp.fire(GEOMETRY_VISIBILITY_CHANGED, payload="g1")
    disp.fire(GEOMETRIES_CHANGED)
    # One deform + one gate — the omnibus was suppressed.
    assert calls == ["deform", "gate", "render"]


# =====================================================================
# Gate truth table — layer ∧ composition ∧ geometry visibility
# =====================================================================

class _FakeLayer:
    def __init__(self, name: str = "L", visible: bool = True) -> None:
        self.name = name
        self.is_visible = bool(visible)


def _two_geometry_tree():
    """Two geometries, each with two compositions of one layer."""
    gm = GeometryManager()
    geom_a = gm.active
    geom_b = gm.add("Geometry B", make_active=False)
    layers = {}
    for geom, tag in ((geom_a, "a"), (geom_b, "b")):
        for ci in (1, 2):
            comp = geom.compositions.add(
                name=f"C{tag}{ci}", make_active=False,
            )
            layer = _FakeLayer(f"l{tag}{ci}")
            geom.compositions.add_layer(comp.id, layer)
            layers[f"{tag}{ci}"] = (comp, layer)
    return gm, geom_a, geom_b, layers


def _gated(gm) -> "set[int]":
    from apeGmsh.viewers.results_viewer import _gate_visible_layer_ids
    return _gate_visible_layer_ids(gm)


def test_gate_both_geometries_visible_no_active_comp_shows_all():
    gm, _, _, layers = _two_geometry_tree()
    ids = _gated(gm)
    assert ids == {id(layer) for (_, layer) in layers.values()}


def test_gate_hidden_geometry_drops_its_layers():
    gm, geom_a, geom_b, layers = _two_geometry_tree()
    gm.set_visible(geom_b.id, False)
    ids = _gated(gm)
    assert id(layers["a1"][1]) in ids
    assert id(layers["a2"][1]) in ids
    assert id(layers["b1"][1]) not in ids
    assert id(layers["b2"][1]) not in ids


def test_gate_active_composition_scopes_per_geometry():
    """Per-geometry composition-gate semantics are preserved: an
    active composition in geometry A narrows A's layers without
    touching B's all-compositions fallback."""
    gm, geom_a, _, layers = _two_geometry_tree()
    geom_a.compositions.set_active(layers["a1"][0].id)
    ids = _gated(gm)
    assert id(layers["a1"][1]) in ids
    assert id(layers["a2"][1]) not in ids      # gated out by A's comp
    assert id(layers["b1"][1]) in ids
    assert id(layers["b2"][1]) in ids


def test_gate_all_geometries_hidden_shows_nothing():
    gm, geom_a, geom_b, _ = _two_geometry_tree()
    gm.set_visible(geom_a.id, False)
    gm.set_visible(geom_b.id, False)
    assert _gated(gm) == set()


def test_gate_truth_table_composes_layer_intent():
    """desired = layer.is_visible AND (id(layer) in gate set) — the
    full S2b truth table over (layer, composition, geometry)."""
    gm, geom_a, geom_b, layers = _two_geometry_tree()
    geom_a.compositions.set_active(layers["a1"][0].id)
    gm.set_visible(geom_b.id, False)
    layers["a1"][1].is_visible = False

    ids = _gated(gm)
    desired = {
        key: layer.is_visible and id(layer) in ids
        for key, (_, layer) in layers.items()
    }
    assert desired == {
        "a1": False,   # geometry ✓, composition ✓, layer ✗
        "a2": False,   # geometry ✓, composition ✗, layer ✓
        "b1": False,   # geometry ✗
        "b2": False,   # geometry ✗
    }
    # Flip every term on for a1 → shows.
    layers["a1"][1].is_visible = True
    assert id(layers["a1"][1]) in _gated(gm)


def test_render_geometries_returns_visible_geometries():
    from apeGmsh.viewers.results_viewer import ResultsViewer

    class _NS:
        pass

    class _Director:
        pass

    gm = GeometryManager()
    geom_a = gm.active
    geom_b = gm.add("Geometry B", make_active=False)
    director = _Director()
    director.geometries = gm
    ns = _NS()
    ns._director = director
    render_geometries = ResultsViewer._render_geometries.__get__(ns)

    assert render_geometries() == [geom_a, geom_b]
    gm.set_visible(geom_a.id, False)
    assert render_geometries() == [geom_b]
    ns._director = None
    assert render_geometries() == []


# =====================================================================
# Scalar-bar title prefix (registry-stamped resolver)
# =====================================================================

class _BarHost:
    """Minimal ScalarBarSupport host — spec.selector.component only."""

    def __init__(self, component: str = "Sxx") -> None:
        from apeGmsh.viewers.diagrams._scalar_bar_support import (
            ScalarBarSupport,
        )

        class _Spec:
            pass

        class _Selector:
            pass

        spec = _Spec()
        spec.selector = _Selector()
        spec.selector.component = component
        self.spec = spec
        self._scalar_bar_title = (
            ScalarBarSupport._scalar_bar_title.__get__(self)
        )


def test_scalar_bar_title_unprefixed_without_resolver():
    host = _BarHost("Sxx")
    assert host._scalar_bar_title() == "Sxx"


def test_scalar_bar_title_prefixes_when_resolver_returns_name():
    host = _BarHost("Sxx")
    host._bar_prefix_resolver = lambda d: "Geometry 2"
    assert host._scalar_bar_title() == "Geometry 2 — Sxx"


def test_scalar_bar_title_unprefixed_when_resolver_returns_none():
    host = _BarHost("Sxx")
    host._bar_prefix_resolver = lambda d: None
    assert host._scalar_bar_title() == "Sxx"


def test_scalar_bar_title_survives_raising_resolver():
    host = _BarHost("Sxx")

    def _boom(d):
        raise RuntimeError("resolver exploded")

    host._bar_prefix_resolver = _boom
    assert host._scalar_bar_title() == "Sxx"


def test_registry_stamps_bar_prefix_resolver_on_attach():
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

    resolver = lambda d: "Geometry 2"    # noqa: E731
    reg = DiagramRegistry()
    # Diagram added before bind picks the resolver up at bind().
    early = _StubDiagram()
    reg.add(early)
    reg.bind(
        RecordingBackend(), view=object(), scene=object(),
        bar_prefix_resolver=resolver,
    )
    assert early._bar_prefix_resolver is resolver
    # Diagram added after bind is stamped at add().
    late = _StubDiagram()
    reg.add(late)
    assert late._bar_prefix_resolver is resolver
    # No resolver bound → nothing stamped.
    reg2 = DiagramRegistry()
    bare = _StubDiagram()
    reg2.add(bare)
    assert not hasattr(bare, "_bar_prefix_resolver")


# =====================================================================
# Session persistence — schema v5 ``visible``
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


def test_new_session_round_trips_visible_flag(tmp_path: Path):
    from apeGmsh.viewers.diagrams._session import (
        GeometrySnapshot,
        load_session,
        save_session,
    )

    geoms = [
        GeometrySnapshot(id="g0", name="A", visible=True),
        GeometrySnapshot(id="g1", name="B", visible=False),
    ]
    saved = save_session(
        specs=[_make_contour_spec()],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        geometries=geoms,
    )
    session = load_session(saved)
    assert session.geometries[0].visible is True
    assert session.geometries[1].visible is False


def test_legacy_session_without_visible_deserializes_to_none(
    tmp_path: Path,
):
    """Pre-v5 sessions carry no ``visible`` key — the snapshot must
    keep None (NOT a blanket default) so the restore path can apply
    the ADR 0058 ruling: old sessions load visible = is-active."""
    from apeGmsh.viewers.diagrams._session import (
        load_session,
        serialize_spec,
    )

    payload = {
        "schema_version": 4,
        "results_path": str(tmp_path / "run.h5"),
        "fem_snapshot_id": None,
        "saved_at": "",
        "geometries": [
            {
                "id": "g0",
                "name": "Geometry 1",
                "deform_enabled": False,
                "active_composition_id": None,
                "compositions": [],
            },
        ],
        "diagrams": [serialize_spec(_make_contour_spec())],
    }
    target = tmp_path / "v4.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    session = load_session(target)
    assert session.geometries[0].visible is None
    # Other fields keep their historical defaults.
    assert session.geometries[0].show_mesh is True


# =====================================================================
# Outline geometry-row eye → GeometryManager.set_visible
# =====================================================================

def test_outline_geometry_eye_drives_set_visible():
    from apeGmsh.viewers.ui._outline_tree import (
        _ROLE_GEOMETRY_KEY,
        OutlineTree,
    )

    gm = GeometryManager()
    geom = gm.active

    class _NS:
        pass

    ns = _NS()
    ns._director = MagicMock()
    ns._director.geometries = gm
    ns._refresh_diagrams = lambda: None
    ns._is_geometry_visible = OutlineTree._is_geometry_visible
    on_eye_clicked = OutlineTree._on_eye_clicked.__get__(ns)

    class _FakeItem:
        def data(self, col, role):
            if role == _ROLE_GEOMETRY_KEY:
                return geom.id
            return None

    typed: list = []
    gm.subscribe_typed(lambda kind, payload: typed.append((kind, payload)))

    on_eye_clicked(_FakeItem())
    assert geom.visible is False
    on_eye_clicked(_FakeItem())
    assert geom.visible is True
    assert typed == [
        (GEOMETRY_VISIBILITY_CHANGED, geom.id),
        (GEOMETRY_VISIBILITY_CHANGED, geom.id),
    ]


def test_geometry_dataclass_has_no_saved_visibility():
    """ADR 0058 S2b retired the Plan 03 v2 geometry-level snapshot —
    the eye drives the ``visible`` flag, not a layer cascade."""
    assert "visible" in Geometry.__dataclass_fields__
    assert "saved_visibility" not in Geometry.__dataclass_fields__


# =====================================================================
# Qt — concurrent rendering on a real viewer (local-only; -m qt)
# =====================================================================

@pytest.fixture
def deforming_results(g, tmp_path: Path):
    """Tiny native Results whose displacement field is non-zero, so a
    deform-enabled geometry visibly leaves the reference position."""
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

    path = tmp_path / "s2b.h5"
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


@pytest.mark.qt
def test_two_visible_geometries_render_concurrently(deforming_results):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        deforming_results, title="s2b-concurrent",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geoms = director.geometries
            geom_a = geoms.active
            scene_a = director.scene_for(geom_a)
            geom_b = geoms.add("Geometry B", make_active=False)
            geoms.set_deformation(
                geom_a.id, enabled=True,
                field="displacement", scale=1.0,
            )
            geoms.set_deformation(
                geom_b.id, enabled=True,
                field="displacement", scale=3.0,
            )
            scene_b = director.scene_for(geom_b)
            pair_a = viewer._scene_actors[geom_a.id]
            pair_b = viewer._scene_actors[geom_b.id]
            # Both geometries visible → both pairs render.
            seen["a_visible"] = all(
                bool(x.GetVisibility()) for x in pair_a
            )
            seen["b_visible"] = all(
                bool(x.GetVisibility()) for x in pair_b
            )
            # Each grid carries its OWN deform configuration.
            disp_a = (
                np.asarray(scene_a.grid.points)
                - scene_a.reference_points
            )
            disp_b = (
                np.asarray(scene_b.grid.points)
                - scene_b.reference_points
            )
            seen["a_deformed"] = bool(np.abs(disp_a).max() > 0)
            seen["b_three_x_a"] = np.allclose(disp_b, 3.0 * disp_a)
            # Hide B → its pair drops out; A unaffected; active
            # pointer untouched (visibility ≠ activation).
            geoms.set_visible(geom_b.id, False)
            seen["b_hidden"] = all(
                not bool(x.GetVisibility()) for x in pair_b
            )
            seen["a_still_visible"] = all(
                bool(x.GetVisibility()) for x in pair_a
            )
            seen["active_unchanged"] = geoms.active is geom_a
            # Show B again → pair returns.
            geoms.set_visible(geom_b.id, True)
            seen["b_visible_again"] = all(
                bool(x.GetVisibility()) for x in pair_b
            )
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen.get("a_visible") is True
    assert seen.get("b_visible") is True
    assert seen.get("a_deformed") is True
    assert seen.get("b_three_x_a") is True
    assert seen.get("b_hidden") is True
    assert seen.get("a_still_visible") is True
    assert seen.get("active_unchanged") is True
    assert seen.get("b_visible_again") is True


@pytest.mark.qt
def test_session_restore_maps_legacy_visible_to_is_active(
    deforming_results,
):
    """ADR 0058 ruling: snapshots without ``visible`` restore as
    "visible iff active"; explicit flags restore verbatim."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.diagrams._session import (
        GeometrySnapshot,
        ViewerSession,
    )
    from apeGmsh.viewers.results_viewer import ResultsViewer

    legacy = ViewerSession(
        schema_version=4,
        results_path="",
        fem_snapshot_id=None,
        saved_at="",
        diagrams=(),
        geometries=(
            GeometrySnapshot(id="g0", name="Alpha", visible=None),
            GeometrySnapshot(id="g1", name="Beta", visible=None),
        ),
        active_geometry_id="g1",
    )
    explicit = ViewerSession(
        schema_version=5,
        results_path="",
        fem_snapshot_id=None,
        saved_at="",
        diagrams=(),
        geometries=(
            GeometrySnapshot(id="g0", name="Gamma", visible=True),
            GeometrySnapshot(id="g1", name="Delta", visible=True),
            GeometrySnapshot(id="g2", name="Epsilon", visible=False),
        ),
        active_geometry_id="g0",
    )

    viewer = ResultsViewer(
        deforming_results, title="s2b-session",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            geoms = viewer._director.geometries
            viewer._apply_session(legacy, viewer._win)
            by_name = {g.name: g for g in geoms.geometries}
            seen["legacy_active_visible"] = by_name["Beta"].visible
            seen["legacy_inactive_hidden"] = by_name["Alpha"].visible
            viewer._apply_session(explicit, viewer._win)
            by_name = {g.name: g for g in geoms.geometries}
            seen["explicit"] = (
                by_name["Gamma"].visible,
                by_name["Delta"].visible,
                by_name["Epsilon"].visible,
            )
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen.get("legacy_active_visible") is True
    assert seen.get("legacy_inactive_hidden") is False
    assert seen.get("explicit") == (True, True, False)
