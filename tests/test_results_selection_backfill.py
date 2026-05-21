"""Phase 2 backfill — ``selection=`` resolves through ``fem.mesh_selection``.

The composite reader's selection vocabulary now mirrors
``FEMData.nodes.get()`` exactly: ``pg=`` / ``label=`` / ``selection=`` /
``ids=``. This test file exercises the new ``selection=`` path end to
end through the Results composite layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5


# =====================================================================
# MeshSelectionStore lookup methods (unit level)
# =====================================================================

def test_meshselectionstore_node_ids_by_name(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # selection-unification-v2 P3-R: ``g.mesh_selection.add_nodes`` is
    # removed (SC-11); register the same spatial set via the v2
    # ``g.mesh_selection.select(...).on_plane(...).save_as(name)``.
    g.mesh_selection.select().on_plane(
        (0, 0, 0.0), (0, 0, 1), tol=1e-3).save_as("base")
    fem = g.mesh.queries.get_fem_data(dim=3)

    nids = fem.mesh_selection.node_ids("base")
    assert nids.size > 0
    # All node IDs should be valid (a subset of fem nodes).
    all_node_ids = set(int(n) for n in fem.nodes.ids)
    assert set(int(n) for n in nids).issubset(all_node_ids)


def test_meshselectionstore_element_ids_by_name(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # P3-R: ``g.mesh_selection.add_elements`` removed (SC-11) →
    # register via the v2 ``select(...).in_box(...).save_as(...)``.
    # ``in_box=[0,0,0,1,1,1]`` named "all_tets" = every element in the
    # unit box → the closed (inclusive=True) box is the faithful
    # mapping (a half-open box would drop boundary centroids).
    g.mesh_selection.select(level="element", dim=3).in_box(
        (0, 0, 0), (1, 1, 1), inclusive=True).save_as("all_tets")
    fem = g.mesh.queries.get_fem_data(dim=3)

    eids = fem.mesh_selection.element_ids("all_tets")
    assert eids.size > 0


def test_meshselectionstore_unknown_name_raises(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # Add at least one selection so fem.mesh_selection is non-None.
    g.mesh_selection.select().on_plane(            # P3-R: was add_nodes
        (0, 0, 0.0), (0, 0, 1), tol=1e-3).save_as("dummy")
    fem = g.mesh.queries.get_fem_data(dim=3)
    with pytest.raises(KeyError, match="No mesh selection"):
        fem.mesh_selection.node_ids("not_a_selection")


def test_meshselectionstore_node_only_set_has_no_elements(g) -> None:
    """``dim=0`` selections (nodes only) raise from ``element_ids``."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # selection-unification-v2 P3-R: ``g.mesh_selection.add_nodes`` is
    # removed (SC-11); register the same spatial set via the v2
    # ``g.mesh_selection.select(...).on_plane(...).save_as(name)``.
    g.mesh_selection.select().on_plane(
        (0, 0, 0.0), (0, 0, 1), tol=1e-3).save_as("base")
    fem = g.mesh.queries.get_fem_data(dim=3)

    # 'base' is a node-only set → element_ids() should not find it.
    with pytest.raises(KeyError, match="No element-bearing"):
        fem.mesh_selection.element_ids("base")


# =====================================================================
# Results composite — selection= for nodal reads
# =====================================================================

def test_results_nodes_get_by_selection(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    g.mesh_selection.select().on_plane(            # P3-R: was add_nodes
        (0, 0, 1.0), (0, 0, 1), tol=1e-3).save_as("top")
    fem = g.mesh.queries.get_fem_data(dim=3)

    n_nodes = len(fem.nodes.ids)
    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="g", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
            components={"displacement_x":
                         np.full((1, n_nodes), 0.5, dtype=np.float64)},
        )
        w.end_stage()

    expected_ids = set(int(n) for n in fem.mesh_selection.node_ids("top"))
    assert expected_ids, "fixture invariant: 'top' selection is non-empty"

    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        slab = r.nodes.get(selection="top", component="displacement_x")
        got_ids = set(int(n) for n in slab.node_ids)
        assert got_ids == expected_ids
        np.testing.assert_allclose(slab.values, 0.5)


def test_results_elements_get_by_selection(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # Element selection via spatial filter — top half slab.
    # P3-R: ``g.mesh_selection.add_elements`` removed (SC-11) →
    # the v2 ``select(...).in_box(...).save_as(...)`` (centroid-based
    # in_box, same half-open default the eager add_elements had).
    g.mesh_selection.select(level="element", dim=3).in_box(
        (0, 0, 0.5), (1, 1, 1.001)).save_as("top_slab")
    fem = g.mesh.queries.get_fem_data(dim=3)
    elem_ids = fem.mesh_selection.element_ids("top_slab")
    assert elem_ids.size > 0

    # Synthetic gauss data — write all elements, then filter on read.
    all_eids = np.asarray(fem.elements.ids, dtype=np.int64)
    n_e = all_eids.size
    nat = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    sxx = np.zeros((1, n_e, 1), dtype=np.float64)

    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="g", kind="static",
                             time=np.array([0.0]))
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=all_eids, natural_coords=nat,
            components={"stress_xx": sxx},
        )
        w.end_stage()

    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        slab = r.elements.gauss.get(selection="top_slab", component="stress_xx")
        got_eids = set(int(e) for e in slab.element_index)
        expected = set(int(e) for e in elem_ids)
        assert got_eids == expected


# =====================================================================
# Selector exclusivity & error paths
# =====================================================================

def test_selector_combined_with_ids_raises(g, tmp_path: Path) -> None:
    """``ids=`` + named selector is the only combination that raises.

    ``pg=`` / ``label=`` / ``selection=`` can be combined together —
    the ID lists are unioned (matching the existing FEMData idiom).
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # selection-unification-v2 P3-R: ``g.mesh_selection.add_nodes`` is
    # removed (SC-11); register the same spatial set via the v2
    # ``g.mesh_selection.select(...).on_plane(...).save_as(name)``.
    g.mesh_selection.select().on_plane(
        (0, 0, 0.0), (0, 0, 1), tol=1e-3).save_as("base")
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="g", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
            components={"displacement_x":
                         np.zeros((1, len(fem.nodes.ids)), dtype=np.float64)},
        )
        w.end_stage()

    with Results.from_native(path, fem=fem, model=_open_model_from_h5(path)) as r:
        with pytest.raises(ValueError, match="not multiple"):
            r.nodes.get(ids=[1, 2], selection="base",
                        component="displacement_x")
        with pytest.raises(ValueError, match="not multiple"):
            r.nodes.get(ids=[1, 2], pg="Body",
                        component="displacement_x")
        # pg + selection together is *allowed* — union semantics.
        slab = r.nodes.get(pg="Body", selection="base",
                            component="displacement_x")
        assert slab.node_ids.size > 0


def test_unknown_selection_raises(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # At least one selection so fem.mesh_selection is non-None — the
    # error path we want to test is "name not found", not "no store".
    g.mesh_selection.select().on_plane(            # P3-R: was add_nodes
        (0, 0, 0.0), (0, 0, 1), tol=1e-3).save_as("exists")
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="g", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
            components={"displacement_x":
                         np.zeros((1, len(fem.nodes.ids)), dtype=np.float64)},
        )
        w.end_stage()

    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        with pytest.raises(KeyError, match="No mesh selection"):
            r.nodes.get(selection="nope", component="displacement_x")


# =====================================================================
# Multi-selector union
# =====================================================================

def test_two_selections_union(g, tmp_path: Path) -> None:
    """Passing a list to ``selection=`` unions them."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # P3-R: was add_nodes (removed SC-11) → v2 select(...).save_as.
    g.mesh_selection.select().on_plane(
        (0, 0, 0.0), (0, 0, 1), tol=1e-3).save_as("bottom")
    g.mesh_selection.select().on_plane(
        (0, 0, 1.0), (0, 0, 1), tol=1e-3).save_as("top")
    fem = g.mesh.queries.get_fem_data(dim=3)

    bottom = set(int(n) for n in fem.mesh_selection.node_ids("bottom"))
    top = set(int(n) for n in fem.mesh_selection.node_ids("top"))

    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="g", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
            components={"displacement_x":
                         np.zeros((1, len(fem.nodes.ids)), dtype=np.float64)},
        )
        w.end_stage()

    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        slab = r.nodes.get(
            selection=["bottom", "top"], component="displacement_x",
        )
        got = set(int(n) for n in slab.node_ids)
        assert got == bottom | top
