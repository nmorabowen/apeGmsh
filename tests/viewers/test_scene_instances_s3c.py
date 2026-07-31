"""ADR 0058 S3c — reference-ghost preset + duplicate-with-layers.

Two new verbs land the most-common concurrent-geometry asks as one
gesture each (core rulings 3 and 4):

* ``GeometryManager.add_reference_ghost(geom_id)`` — a *preset* verb
  (pure state, headless-testable) that composes the manager's own
  mutators into an EMPTY substrate-only geometry: deform off, nodes
  off, dimmed to ``GHOST_OPACITY`` (0.3), ``offset`` + ``stage_id``
  copied from the source, source stays active, no compositions (a
  dimmed reference must not double the source's contours).
* ``ResultsDirector.duplicate_geometry(geom_id)`` — the director verb
  that composes the manager's state-only ``duplicate`` with diagram
  reconstruction from each layer's ``DiagramSpec`` (the ``_apply_session``
  recipe). Membership is recorded BEFORE ``registry.add`` so attach
  resolves the clone's scene through the existing ``scene_resolver``.
  Runtime overrides / probes / manual hides are explicitly NOT copied —
  *what's in the spec round-trips, what isn't doesn't*.

No session changes — a ghost is an ordinary geometry after creation
(no linkage field, by design). The outline's Duplicate gesture upgrades
to the director verb; a new "Add reference ghost" action calls the
manager verb inside ``gesture_batch``.

The qt-marked test (local-only) drives a real viewer: "Add reference
ghost" yields two substrate pairs (the ghost at reference + source
offset, dimmed); duplicate-with-layers on a contour geometry yields two
contour layers carrying the geometry-name prefix (S2b rule).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.viewers.diagrams._geometries import (
    GHOST_OPACITY,
    GeometryManager,
)


# =====================================================================
# add_reference_ghost — the preset verb (headless, pure state)
# =====================================================================

def test_ghost_is_empty_dimmed_deform_off_nodes_off():
    gm = GeometryManager()
    src = gm.active
    ghost = gm.add_reference_ghost(src.id)
    assert ghost is not None
    assert ghost.name == "Geometry 1 (reference)"
    assert ghost.visible is True
    assert ghost.deform_enabled is False
    assert ghost.show_nodes is False
    assert ghost.display_opacity == GHOST_OPACITY
    # Substrate-only: no compositions / layers doubled on top.
    assert ghost.compositions.compositions == []


def test_ghost_copies_offset_and_stage_pin_from_source():
    gm = GeometryManager()
    src = gm.active
    gm.set_offset(src.id, (1.0, 2.0, 3.0))
    gm.set_stage_pin(src.id, "grav")
    ghost = gm.add_reference_ghost(src.id)
    assert ghost is not None
    assert ghost.offset == (1.0, 2.0, 3.0)
    assert ghost.stage_id == "grav"


def test_ghost_leaves_source_active():
    gm = GeometryManager()
    src = gm.active
    gm.add_reference_ghost(src.id)
    # The preset is decoration — the source stays the editing target,
    # despite the underlying duplicate() flipping active to the clone.
    assert gm.active_id == src.id


def test_ghost_of_unknown_geometry_returns_none():
    gm = GeometryManager()
    assert gm.add_reference_ghost("no-such-id") is None


def test_ghost_of_a_ghost_gets_a_unique_name():
    gm = GeometryManager()
    src = gm.active
    g1 = gm.add_reference_ghost(src.id)
    assert g1 is not None
    # A second ghost of the same source can't collide on the name.
    g2 = gm.add_reference_ghost(src.id)
    assert g2 is not None
    assert g2.name != g1.name
    assert g2.name.startswith("Geometry 1 (reference)")
    # And a ghost OF a ghost is well-named too.
    g3 = gm.add_reference_ghost(g1.id)
    assert g3 is not None
    assert g3.name == "Geometry 1 (reference) (reference)"


def test_ghost_appends_a_new_geometry():
    gm = GeometryManager()
    src = gm.active
    before = len(gm.geometries)
    gm.add_reference_ghost(src.id)
    assert len(gm.geometries) == before + 1


# =====================================================================
# duplicate_geometry — the director verb (real diagrams, headless)
# =====================================================================

def _native_results(g, tmp_path: Path):
    """Tiny native single-stage Results with a nonzero ``displacement_x``
    field so ContourDiagram attaches against real data."""
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

    path = tmp_path / "s3c.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_x": np.ones((3, n_nodes))},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _contour_spec(component="displacement_x", stage_id="grav"):
    from apeGmsh.viewers.diagrams._base import DiagramSpec
    from apeGmsh.viewers.diagrams._selectors import SlabSelector
    from apeGmsh.viewers.diagrams._styles import ContourStyle

    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component=component),
        style=ContourStyle(),
        stage_id=stage_id,
        label="warp",
    )


@pytest.fixture
def director(g, tmp_path: Path):
    from apeGmsh.viewers.diagrams._director import ResultsDirector

    return ResultsDirector(_native_results(g, tmp_path))


def _seed_geometry_with_layers(director, *, n_layers=2):
    """Build one source geometry holding a composition with ``n_layers``
    real contour diagrams (registry unbound — no attach needed)."""
    from apeGmsh.viewers.diagrams._contour import ContourDiagram

    geom = director.geometries.active
    comp = geom.compositions.add(name="Warp", make_active=True)
    layers = []
    for _ in range(n_layers):
        d = ContourDiagram(_contour_spec(), director.results)
        director.registry.add(d)
        geom.compositions.add_layer(comp.id, d)
        layers.append(d)
    return geom, comp, layers


def test_duplicate_geometry_rebuilds_composition_names_and_order(director):
    geom = director.geometries.active
    c1 = geom.compositions.add(name="First", make_active=False)
    c2 = geom.compositions.add(name="Second", make_active=True)
    from apeGmsh.viewers.diagrams._contour import ContourDiagram
    for comp in (c1, c2):
        d = ContourDiagram(_contour_spec(), director.results)
        director.registry.add(d)
        geom.compositions.add_layer(comp.id, d)

    clone = director.duplicate_geometry(geom.id)
    assert clone is not None
    src_names = [c.name for c in geom.compositions.compositions]
    clone_names = [c.name for c in clone.compositions.compositions]
    assert clone_names == src_names


def test_duplicate_geometry_rebuilds_distinct_diagram_instances(director):
    geom, _comp, layers = _seed_geometry_with_layers(director, n_layers=2)
    before = len(director.registry)

    clone = director.duplicate_geometry(geom.id)
    assert clone is not None
    # Registry grew by exactly the rebuilt-layer count.
    assert len(director.registry) == before + 2

    clone_layers = clone.compositions.compositions[0].layers
    assert len(clone_layers) == 2
    for src_d, clone_d in zip(layers, clone_layers):
        # Distinct instances ...
        assert clone_d is not src_d
        # ... with equal (kind, selector, style, stage_id) specs.
        assert clone_d.spec.kind == src_d.spec.kind
        assert clone_d.spec.selector == src_d.spec.selector
        assert clone_d.spec.style == src_d.spec.style
        assert clone_d.spec.stage_id == src_d.spec.stage_id


def test_duplicate_geometry_layers_resolve_clone_scene(director):
    """Membership recorded BEFORE registry.add → the director's
    ``_scene_for_diagram`` (the registry's scene_resolver) resolves the
    clone's layers to the CLONE geometry, not the active-geometry
    fallback."""
    geom, _comp, _layers = _seed_geometry_with_layers(director, n_layers=1)
    clone = director.duplicate_geometry(geom.id)
    assert clone is not None
    clone_d = clone.compositions.compositions[0].layers[0]
    owner = director.geometries.geometry_for_layer(clone_d)
    assert owner is clone


def test_duplicate_geometry_does_not_copy_runtime_overrides(director):
    """A Contour clone is born with the default runtime opacity —
    runtime overrides not reflected into the spec do not survive
    duplication (the session save/restore rule)."""
    from apeGmsh.viewers.diagrams._base import DiagramSpec
    from apeGmsh.viewers.diagrams._contour import ContourDiagram
    from apeGmsh.viewers.diagrams._selectors import SlabSelector
    from apeGmsh.viewers.diagrams._styles import ContourStyle

    geom = director.geometries.active
    comp = geom.compositions.add(name="Contour", make_active=True)
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
        stage_id="grav",
    )
    src_d = ContourDiagram(spec, director.results)
    director.registry.add(src_d)
    geom.compositions.add_layer(comp.id, src_d)
    # Mutate the source's RUNTIME override (not reflected in the spec).
    src_d._runtime_opacity = 0.5

    clone = director.duplicate_geometry(geom.id)
    assert clone is not None
    clone_d = clone.compositions.compositions[0].layers[0]
    assert clone_d is not src_d
    # The clone starts from the default — the override didn't copy.
    assert clone_d._runtime_opacity is None


def test_duplicate_geometry_skips_failing_layers_rest_land(director):
    """A layer whose kind is unknown is skipped; the surviving layers
    still rebuild (fail-soft, same as session restore)."""
    from apeGmsh.viewers.diagrams._base import DiagramSpec
    from apeGmsh.viewers.diagrams._contour import ContourDiagram

    geom = director.geometries.active
    comp = geom.compositions.add(name="Mixed", make_active=True)
    good = ContourDiagram(_contour_spec(), director.results)
    director.registry.add(good)
    geom.compositions.add_layer(comp.id, good)

    # A layer carrying a spec whose kind is not registered. Build a
    # throwaway object with a ``.spec`` whose kind no kind_def matches.
    bad_spec = DiagramSpec(
        kind="__nonexistent_kind__",
        selector=_contour_spec().selector,
        style=_contour_spec().style,
        stage_id="grav",
    )
    bad_layer = type("BadLayer", (), {"spec": bad_spec})()
    geom.compositions.add_layer(comp.id, bad_layer)

    clone = director.duplicate_geometry(geom.id)
    assert clone is not None
    clone_layers = clone.compositions.compositions[0].layers
    # Only the good layer rebuilt; the unknown-kind one was skipped.
    assert len(clone_layers) == 1
    assert clone_layers[0].spec.kind == "contour"


def test_duplicate_geometry_restores_active_composition_by_position(director):
    geom = director.geometries.active
    geom.compositions.add(name="First", make_active=False)
    c2 = geom.compositions.add(name="Second", make_active=True)
    assert geom.compositions.active_id == c2.id

    clone = director.duplicate_geometry(geom.id)
    assert clone is not None
    # The clone's active composition is the one at the SAME position
    # (position 1), not a stale source UUID.
    clone_comps = clone.compositions.compositions
    assert clone.compositions.active_id == clone_comps[1].id
    assert clone.compositions.active.name == "Second"


def test_duplicate_geometry_unknown_id_returns_none(director):
    assert director.duplicate_geometry("no-such-id") is None


def test_duplicate_geometry_copies_offset_and_stage_pin(director):
    geom = director.geometries.active
    director.geometries.set_offset(geom.id, (4.0, 5.0, 6.0))
    director.geometries.set_stage_pin(geom.id, "grav")
    clone = director.duplicate_geometry(geom.id)
    assert clone is not None
    assert clone.offset == (4.0, 5.0, 6.0)
    assert clone.stage_id == "grav"


def test_duplicate_geometry_section_cut_uses_tag_map_path(director):
    """A section_cut layer duplicates through the ``tag_map=`` branch
    (the only kind that takes a third constructor argument)."""
    from apeGmsh.viewers.diagrams._base import DiagramSpec
    from apeGmsh.viewers.diagrams._selectors import SlabSelector
    from apeGmsh.viewers.diagrams._section_cut import SectionCutDiagram

    # Build a section_cut layer directly; constructing it needs a
    # tag_map, so stamp a sentinel on the director and assert the
    # rebuild path forwards it. We bypass the cut-style plumbing by
    # constructing with object.__new__ + a stub spec.
    sentinel_map = object()
    captured: dict = {}

    def _spy_init(self, spec, results, *, tag_map=None):
        captured["tag_map"] = tag_map
        # Minimal init so the instance is usable as a registry layer.
        self.spec = spec
        self._results = results
        self._attached = False
        self._visible = bool(spec.visible)

    geom = director.geometries.active
    comp = geom.compositions.add(name="Cuts", make_active=True)

    # A section_cut spec with a benign style so kind_def resolves; the
    # spied __init__ avoids the real cut machinery.
    from apeGmsh.viewers.diagrams._styles import SectionCutStyle
    spec = DiagramSpec(
        kind="section_cut",
        selector=SlabSelector(component="cut"),
        style=SectionCutStyle.__new__(SectionCutStyle),
        stage_id="grav",
    )
    src = SectionCutDiagram.__new__(SectionCutDiagram)
    src.spec = spec
    src._results = director.results
    src._attached = False
    src._visible = True
    director.registry.add(src)
    geom.compositions.add_layer(comp.id, src)

    # Make the director's tag_map resolve to our sentinel and spy the
    # SectionCutDiagram constructor for the duplicate rebuild.
    import unittest.mock as mock
    with mock.patch.object(
        type(director), "tag_map",
        new_callable=mock.PropertyMock, return_value=sentinel_map,
    ), mock.patch.object(SectionCutDiagram, "__init__", _spy_init):
        clone = director.duplicate_geometry(geom.id)

    assert clone is not None
    assert captured.get("tag_map") is sentinel_map


# =====================================================================
# Outline gesture wiring — Duplicate → director verb; ghost action
# =====================================================================

def test_outline_duplicate_routes_to_director_verb():
    """The geometry-row Duplicate action calls
    ``director.duplicate_geometry`` (not the manager's bare
    duplicate())."""
    import apeGmsh.viewers.ui._outline_tree as outline_mod

    src = inspect.getsource(outline_mod.OutlineTree._on_context_menu)
    assert "self._director.duplicate_geometry(geom_id)" in src
    assert "add_reference_ghost(geom_id)" in src
    assert "gesture_batch()" in src


# =====================================================================
# Qt — real viewer drive (local-only; `pytest -m qt`)
# =====================================================================

@pytest.fixture
def deforming_results(g, tmp_path: Path):
    return _native_results(g, tmp_path)


@pytest.mark.qt
def test_add_reference_ghost_yields_a_dimmed_substrate_pair(
    deforming_results,
):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        deforming_results, title="s3c-ghost",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geoms = director.geometries
            src = geoms.active
            # Offset + deform the source so the ghost (deform off, on
            # the source's frame) sits at reference + source offset.
            geoms.set_offset(src.id, (5.0, 0.0, 0.0))
            geoms.set_deformation(
                src.id, enabled=True, field="displacement", scale=2.0,
            )
            with director.dispatcher.gesture_batch():
                ghost = geoms.add_reference_ghost(src.id)
            seen["source_active"] = geoms.active_id == src.id
            seen["ghost_dimmed"] = ghost.display_opacity == GHOST_OPACITY
            seen["ghost_deform_off"] = ghost.deform_enabled is False
            # Two substrate pairs exist.
            seen["two_pairs"] = (
                src.id in viewer._scene_actors
                and ghost.id in viewer._scene_actors
            )
            ghost_scene = director.scene_for(ghost)
            ref = np.asarray(ghost_scene.reference_points)
            pts = np.asarray(ghost_scene.grid.points)
            # Ghost sits at reference + the SOURCE offset (deform off).
            np.testing.assert_allclose(
                pts, ref + np.array([5.0, 0.0, 0.0]),
            )
            seen["ghost_at_offset_reference"] = True
        finally:
            # ResultsWindow has no .close — the QMainWindow is .window
            # (the s3a idiom). The old `viewer._win.close` raised
            # AttributeError into the Qt loop and hung the test forever.
            viewer._win.window.close()

    QtCore.QTimer.singleShot(0, _drive_then_close)
    viewer.show()

    assert seen.get("source_active") is True
    assert seen.get("ghost_dimmed") is True
    assert seen.get("ghost_deform_off") is True
    assert seen.get("two_pairs") is True
    assert seen.get("ghost_at_offset_reference") is True


@pytest.mark.qt
def test_duplicate_with_layers_yields_two_contour_layers(deforming_results):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.diagrams._contour import ContourDiagram
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        deforming_results, title="s3c-dup",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geom = director.geometries.active
            comp = geom.compositions.add(name="Warp", make_active=True)
            src_d = ContourDiagram(_contour_spec(), director.results)
            director.registry.add(src_d)
            geom.compositions.add_layer(comp.id, src_d)

            clone = director.duplicate_geometry(geom.id)
            seen["clone_made"] = clone is not None
            clone_layers = clone.compositions.compositions[0].layers
            seen["one_clone_layer"] = len(clone_layers) == 1
            seen["distinct"] = clone_layers[0] is not src_d
            # Both contour layers are attached (live registry).
            seen["both_attached"] = (
                src_d.is_attached and clone_layers[0].is_attached
            )
        finally:
            # ResultsWindow has no .close — the QMainWindow is .window
            # (the s3a idiom). The old `viewer._win.close` raised
            # AttributeError into the Qt loop and hung the test forever.
            viewer._win.window.close()

    QtCore.QTimer.singleShot(0, _drive_then_close)
    viewer.show()

    assert seen.get("clone_made") is True
    assert seen.get("one_clone_layer") is True
    assert seen.get("distinct") is True
    assert seen.get("both_attached") is True
