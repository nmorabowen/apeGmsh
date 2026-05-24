"""Tests for the per-node ``ndf`` broker metadata
(shell-to-solid coupling feature, S1b).

Covers:

- Implicit populator over a uniform 3D solid mesh (ndf=3 everywhere).
- Implicit populator over a uniform 2D surface mesh (ndf=6 everywhere).
- Shell-on-solid interface yielding ndf=6 at the shared nodes (the
  ``max`` rule).  The interface case uses
  ``g.mesh.editing.remove_duplicate_nodes()`` to share nodes between
  the two meshes — until sister PR ``feat/shell-solid-fragment``
  lands the proper conformal-fragment path.
- Explicit override via ``g.model.set_node_ndf(target, ndf=K)`` and
  its precedence over the implicit value.
- ``KeyError`` on lookup of an unknown node tag.
- H5 round-trip: the per-node ``ndf`` survives ``to_h5`` /
  ``from_h5``.
- Backward compatibility: synthetic 2.6.0 file (no ``/nodes/ndf``
  dataset) loads and reports ``ndf_for(...) is None``.
- Schema version bumped to 2.7.0.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION


# =====================================================================
# Helpers
# =====================================================================

def _build_solid_box(g, lc: float = 5.0):
    """Unit 3D box, single tet mesh."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=3)


def _build_surface_plate(g, lc: float = 5.0):
    """Unit 2D plate, single tri mesh."""
    g.model.geometry.add_rectangle(
        0.0, 0.0, 0.0, 10.0, 10.0, label='Plate',
    )
    g.model.sync()
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=2)


def _build_shell_on_solid(g, lc: float = 5.0):
    """3D box with a 2D plate coplanar with its top face.

    After ``remove_duplicate_nodes()`` runs, the plate's nodes and
    the box's top-face nodes share IDs, so the broker observes both
    a 2D group (the plate) and a 3D group (the box) touching the
    same node ids.  The implicit ``max`` rule should then yield
    ``ndf=6`` at those interface nodes (the plate's class implies
    rotational DOFs) and ``ndf=3`` at the box's interior nodes.
    """
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.geometry.add_rectangle(
        0.0, 0.0, 10.0, 10.0, 10.0, label='Cap',
    )
    g.model.sync()
    g.mesh.sizing.set_global_size(lc)
    # Generate both dims; the plate sits on z=10 — coplanar with the
    # box's top face.  Remove duplicate nodes merges the coincident
    # ones so the broker sees a real shared-node interface.
    g.mesh.generation.generate(dim=3)
    g.mesh.editing.remove_duplicate_nodes(verbose=False)


# =====================================================================
# Test 1 — implicit ndf uniform 3D solid
# =====================================================================

def test_implicit_ndf_uniform_3d_solid(g):
    """A pure 3D tet box yields ndf=3 with source='implicit' everywhere."""
    _build_solid_box(g)
    fem = g.mesh.queries.get_fem_data(dim=3)

    assert len(fem.nodes) > 0
    for tag in fem.nodes.ids:
        assert fem.nodes.ndf_for(int(tag)) == 3, (
            f"node {tag} expected ndf=3, got {fem.nodes.ndf_for(int(tag))}"
        )

    records = list(fem.nodes.ndf_records())
    assert len(records) == len(fem.nodes)
    assert {r.source for r in records} == {"implicit"}
    assert {r.ndf for r in records} == {3}


# =====================================================================
# Test 2 — implicit ndf uniform 2D surface
# =====================================================================

def test_implicit_ndf_uniform_2d_surface(g):
    """A pure 2D tri plate yields ndf=6 with source='implicit' everywhere."""
    _build_surface_plate(g)
    fem = g.mesh.queries.get_fem_data(dim=2)

    assert len(fem.nodes) > 0
    for tag in fem.nodes.ids:
        assert fem.nodes.ndf_for(int(tag)) == 6, (
            f"node {tag} expected ndf=6, got {fem.nodes.ndf_for(int(tag))}"
        )

    records = list(fem.nodes.ndf_records())
    assert {r.source for r in records} == {"implicit"}
    assert {r.ndf for r in records} == {6}


# =====================================================================
# Test 3 — shell-on-solid interface gets the max
# =====================================================================

def test_implicit_ndf_shell_on_solid_interface_gets_max(g):
    """A shell coplanar with the top face of a solid box yields ndf=6 at
    shared interface nodes and ndf=3 at interior solid nodes.

    Uses ``remove_duplicate_nodes()`` as the conformal-mesh workaround
    until sister PR ``feat/shell-solid-fragment`` lands the proper
    cross-dim fragmenting path.
    """
    _build_shell_on_solid(g)
    fem = g.mesh.queries.get_fem_data()  # all dims

    # The mesh should now carry both 2D and 3D element groups, with
    # at least one node id shared between them.
    dims_present = {int(g_.dim) for g_ in fem.elements}
    assert {2, 3}.issubset(dims_present), (
        f"expected both 2D and 3D element groups; got dims={dims_present}"
    )

    # Collect node ids per dim.
    surface_nodes: set[int] = set()
    volume_nodes: set[int] = set()
    for grp in fem.elements:
        unique = {int(x) for x in np.asarray(grp.connectivity).reshape(-1)}
        if int(grp.dim) == 2:
            surface_nodes |= unique
        elif int(grp.dim) == 3:
            volume_nodes |= unique
    shared = surface_nodes & volume_nodes
    interior = volume_nodes - surface_nodes

    assert shared, (
        "remove_duplicate_nodes did not yield any shared nodes between "
        "the plate and the box — the test setup is degenerate."
    )

    # Shared nodes: max(IMPLICIT_NDF_BY_DIM[2]=6, [3]=3) == 6
    for tag in shared:
        assert fem.nodes.ndf_for(int(tag)) == 6, (
            f"shared node {tag} expected ndf=6 (max), "
            f"got {fem.nodes.ndf_for(int(tag))}"
        )

    # Interior volume-only nodes: ndf=3
    for tag in interior:
        assert fem.nodes.ndf_for(int(tag)) == 3, (
            f"interior node {tag} expected ndf=3, "
            f"got {fem.nodes.ndf_for(int(tag))}"
        )


# =====================================================================
# Test 4 — explicit override via session API
# =====================================================================

def test_explicit_override_via_session_api(g):
    """``g.model.set_node_ndf('Plate', ndf=2)`` should pin the plate
    nodes to ndf=2 with source='explicit'.

    Uses a 2D plate (default implicit ndf=6 for surfaces); the explicit
    override drops every node down to 2 (plane stress).
    """
    g.model.geometry.add_rectangle(
        0.0, 0.0, 0.0, 10.0, 10.0, label='Plate',
    )
    g.model.sync()
    g.physical.add_surface([1], name='Plate')
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=2)

    # Pin every plate node to ndf=2 (plane stress)
    g.model.set_node_ndf('Plate', ndf=2)

    fem = g.mesh.queries.get_fem_data(dim=2)
    assert len(fem.nodes) > 0
    for tag in fem.nodes.ids:
        assert fem.nodes.ndf_for(int(tag)) == 2, (
            f"node {tag} expected ndf=2 (explicit), "
            f"got {fem.nodes.ndf_for(int(tag))}"
        )

    records = list(fem.nodes.ndf_records())
    assert {r.source for r in records} == {"explicit"}
    assert {r.ndf for r in records} == {2}


# =====================================================================
# Test 5 — explicit override takes precedence over implicit
# =====================================================================

def test_explicit_override_takes_precedence_over_implicit(g):
    """A 3D solid (implicit=3) with an explicit override to ndf=6 on a
    named PG should report ndf=6 for those nodes, source='explicit'."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    # Pick the top face and tag it as a PG so we can resolve by name.
    top_tag = None
    for d, t in g.model.queries.boundary('Body', dim=2):
        com = g.model.queries.center_of_mass(int(t), dim=int(d))
        if abs(com[2] - 10.0) < 1e-6:
            top_tag = int(t)
            break
    assert top_tag is not None
    g.physical.add_surface([top_tag], name='Top')
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)

    g.model.set_node_ndf('Top', ndf=6)

    fem = g.mesh.queries.get_fem_data(dim=3)

    # Find the top-face nodes via gmsh
    import gmsh
    top_node_ids: set[int] = set()
    nt, _, _ = gmsh.model.mesh.getNodes(
        dim=2, tag=top_tag, includeBoundary=True,
        returnParametricCoord=False,
    )
    top_node_ids.update(int(n) for n in nt)
    assert top_node_ids, "top face should have at least one mesh node"

    # Top-face nodes — explicit, ndf=6
    for tag in top_node_ids:
        assert fem.nodes.ndf_for(int(tag)) == 6, (
            f"top node {tag} expected ndf=6 (explicit), "
            f"got {fem.nodes.ndf_for(int(tag))}"
        )
        rec = fem.nodes.ndf_records().get(int(tag))
        assert rec is not None
        assert rec.source == "explicit"

    # An interior node (not on the top face) should still be implicit ndf=3
    interior = set(int(t) for t in fem.nodes.ids) - top_node_ids
    assert interior, "model should have at least one interior node"
    for tag in interior:
        assert fem.nodes.ndf_for(int(tag)) == 3
        rec = fem.nodes.ndf_records().get(int(tag))
        assert rec is not None
        assert rec.source == "implicit"


# =====================================================================
# Test 6 — KeyError on unknown tag
# =====================================================================

def test_ndf_for_unknown_tag_raises_keyerror(g):
    _build_solid_box(g)
    fem = g.mesh.queries.get_fem_data(dim=3)

    bogus = int(np.asarray(fem.nodes.ids).max()) + 10_000
    with pytest.raises(KeyError):
        fem.nodes.ndf_for(bogus)


# =====================================================================
# Test 7 — H5 round-trip
# =====================================================================

def test_ndf_round_trip_through_h5(g, tmp_path: Path):
    """The per-node ``ndf`` vector survives to_h5 / from_h5."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    top_tag = None
    for d, t in g.model.queries.boundary('Body', dim=2):
        com = g.model.queries.center_of_mass(int(t), dim=int(d))
        if abs(com[2] - 10.0) < 1e-6:
            top_tag = int(t)
            break
    assert top_tag is not None
    g.physical.add_surface([top_tag], name='Top')
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.model.set_node_ndf('Top', ndf=6)

    fem = g.mesh.queries.get_fem_data(dim=3)

    # Snapshot ids + per-node ndf BEFORE round-trip
    original = {
        int(t): fem.nodes.ndf_for(int(t)) for t in fem.nodes.ids
    }
    original_src = {
        int(r.node_id): r.source for r in fem.nodes.ndf_records()
    }

    out = tmp_path / "ndf_round_trip.h5"
    fem.to_h5(str(out))

    # Quick smoke test that the datasets were emitted.
    with h5py.File(out, "r") as f:
        assert "nodes/ndf" in f
        assert "nodes/ndf_source" in f
        assert f["nodes/ndf"].dtype == np.int8
        assert f["nodes/ndf_source"].dtype == np.int8

    rebuilt = FEMData.from_h5(str(out))
    rebuilt_map = {
        int(t): rebuilt.nodes.ndf_for(int(t)) for t in rebuilt.nodes.ids
    }
    rebuilt_src = {
        int(r.node_id): r.source for r in rebuilt.nodes.ndf_records()
    }

    assert rebuilt_map == original
    assert rebuilt_src == original_src


# =====================================================================
# Test 8 — backward compatibility: 2.6.0 file without ndf datasets
# =====================================================================

def test_h5_2_6_0_file_loads_without_ndf(tmp_path: Path):
    """A synthetic 2.6.0 file (no /nodes/ndf, no /nodes/ndf_source)
    should load and round-trip with ``ndf_for(...) is None``."""
    from apeGmsh.mesh.FEMData import (
        ElementComposite,
        FEMData,
        MeshInfo,
        NodeComposite,
    )
    from apeGmsh.mesh._element_types import ElementGroup, make_type_info
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet

    # Build a tiny in-memory FEM, then strip the ndf datasets the
    # 2.7.0 writer would normally emit.
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=1,
    )
    tri_group = ElementGroup(
        element_type=tri_info, ids=np.array([10], dtype=np.int64),
        connectivity=np.array([[1, 2, 3]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={2: tri_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=4, n_elems=1, bandwidth=2, types=[tri_info])
    fem = FEMData(nodes=nodes, elements=elements, info=info)

    # Write via the normal path (this WILL write the ndf datasets if
    # the broker carries any; here ``nodes._ndf is None`` so the
    # writer skips them — exactly the 2.6.0 shape).
    out = tmp_path / "legacy_2_6_0.h5"
    fem.to_h5(str(out))

    # Force the per-zone neutral version down to 2.6.0 — the prior
    # minor in our two-version window — and confirm the datasets are
    # absent (the 2.7.0 writer skipped them because ``_ndf is None``).
    with h5py.File(out, "r+") as f:
        f["meta"].attrs["schema_version"] = "2.6.0"
        f["meta"].attrs["neutral_schema_version"] = "2.6.0"
        assert "ndf" not in f["nodes"]
        assert "ndf_source" not in f["nodes"]

    rebuilt = FEMData.from_h5(str(out))
    # Legacy files: ndf metadata absent → ndf_for returns None
    for tag in rebuilt.nodes.ids:
        assert rebuilt.nodes.ndf_for(int(tag)) is None

    # ndf_records() should yield an empty NodeNDFSet
    assert len(rebuilt.nodes.ndf_records()) == 0


# =====================================================================
# Test 9 — schema version bumped to 2.7.0
# =====================================================================

def test_schema_version_bumped_to_2_7_0():
    """NEUTRAL_SCHEMA_VERSION advanced from 2.6.0 to 2.7.0."""
    assert NEUTRAL_SCHEMA_VERSION == "2.7.0"
