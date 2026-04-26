"""Phase 4 — recorder spec resolution against a FEMData snapshot.

These tests use a real session fixture (``g``) to build a small mesh
with PGs, labels, and post-mesh selections, then exercise the
resolve path end to end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.solvers.Recorders import Recorders


# =====================================================================
# Resolution end-to-end
# =====================================================================

def test_resolve_locks_snapshot_id(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.nodes(pg="Body", components=["displacement"])
    spec = r.resolve(fem, ndm=3, ndf=6)

    assert spec.fem_snapshot_id == fem.snapshot_id


def test_resolve_expands_shorthand(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.nodes(pg="Body", components=["displacement"])
    r.gauss(pg="Body", components=["stress"])
    spec = r.resolve(fem, ndm=3, ndf=6)

    assert spec.records[0].components == (
        "displacement_x", "displacement_y", "displacement_z",
    )
    assert spec.records[1].components == (
        "stress_xx", "stress_yy", "stress_zz",
        "stress_xy", "stress_yz", "stress_xz",
    )


def test_resolve_clipped_to_2d(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.nodes(pg="Body", components=["displacement"])
    r.gauss(pg="Body", components=["stress"])
    spec = r.resolve(fem, ndm=2, ndf=2)

    # 2D: 2 displacement components + no rotations.
    assert spec.records[0].components == ("displacement_x", "displacement_y")
    # 2D plane: 3 stress components.
    assert spec.records[1].components == (
        "stress_xx", "stress_yy", "stress_xy",
    )


def test_resolve_node_selectors(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    g.mesh_selection.add_nodes(on_plane=("z", 0.0, 1e-3), name="bottom")
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.nodes(pg="Body", components=["displacement"])
    r.nodes(selection="bottom", components=["reaction_force"])
    r.nodes(ids=[1, 2, 3], components=["acceleration"])
    spec = r.resolve(fem, ndm=3, ndf=6)

    body_ids = set(int(n) for n in spec.records[0].node_ids)
    bottom_ids = set(int(n) for n in spec.records[1].node_ids)
    explicit_ids = set(int(n) for n in spec.records[2].node_ids)

    # PG "Body" covers all volume nodes.
    assert body_ids == set(int(n) for n in fem.nodes.ids)
    # Bottom selection is a subset.
    assert bottom_ids and bottom_ids.issubset(body_ids)
    # Explicit IDs pass through.
    assert explicit_ids == {1, 2, 3}


def test_resolve_element_selectors(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.gauss(pg="Body", components=["stress_xx"])
    r.gauss(components=["stress_xx"])  # no selector — all elements
    spec = r.resolve(fem, ndm=3, ndf=6)

    body_eids = set(int(e) for e in spec.records[0].element_ids)
    all_eids = set(int(e) for e in spec.records[1].element_ids)
    assert body_eids == all_eids


def test_resolve_modal_carries_n_modes(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.modal(n_modes=7)
    spec = r.resolve(fem)
    assert spec.records[0].n_modes == 7
    assert spec.records[0].node_ids is None
    assert spec.records[0].element_ids is None


# =====================================================================
# Component validation per category
# =====================================================================

def test_stress_invalid_for_nodes_raises(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.nodes(components=["stress_xx"])
    with pytest.raises(ValueError, match="not valid for recorder category"):
        r.resolve(fem, ndm=3, ndf=6)


def test_displacement_invalid_for_gauss_raises(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.gauss(components=["displacement_x"])
    with pytest.raises(ValueError, match="not valid for recorder category"):
        r.resolve(fem, ndm=3, ndf=6)


def test_unknown_component_raises(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.nodes(components=["not_a_component"])
    # Unknown name fails in the shorthand expander first.
    with pytest.raises(ValueError, match="Unknown component"):
        r.resolve(fem, ndm=3, ndf=6)


def test_state_variable_pattern_allowed_in_gauss(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.gauss(components=["state_variable_3"])
    # Should not raise — state_variable_<n> is allowed in gauss/fibers/layers
    spec = r.resolve(fem, ndm=3, ndf=6)
    assert "state_variable_3" in spec.records[0].components


# =====================================================================
# Drift detection — re-mesh produces a different snapshot_id
# =====================================================================

def test_resolve_against_different_fem_yields_different_hash(g) -> None:
    """Two FEMData with different connectivity → different snapshot_id."""
    # First mesh
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem1 = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.nodes(pg="Body", components=["displacement"])
    spec1 = r.resolve(fem1)

    # Add a second box adjacent — this changes the mesh fundamentally.
    g.model.geometry.add_box(1, 0, 0, 1, 1, 1, label="box2")
    g.physical.add_volume("box2", name="Body2")
    g.mesh.generation.generate(dim=3)
    fem2 = g.mesh.queries.get_fem_data(dim=3)

    spec2 = r.resolve(fem2)
    assert spec1.fem_snapshot_id != spec2.fem_snapshot_id


# =====================================================================
# Manifest serialization
# =====================================================================

def test_manifest_roundtrip(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    r = Recorders()
    r.nodes(pg="Body", components=["displacement"], dt=0.01)
    r.gauss(pg="Body", components=["stress_xx"], n_steps=5)
    r.modal(n_modes=3)
    spec = r.resolve(fem, ndm=3, ndf=6)

    manifest_path = tmp_path / "manifest.h5"
    spec.to_manifest_h5(manifest_path)

    from apeGmsh.solvers._recorder_specs import ResolvedRecorderSpec
    spec_back = ResolvedRecorderSpec.from_manifest_h5(manifest_path)

    assert spec_back.fem_snapshot_id == spec.fem_snapshot_id
    assert len(spec_back.records) == len(spec.records)
    for orig, back in zip(spec.records, spec_back.records):
        assert orig.category == back.category
        assert orig.name == back.name
        assert orig.components == back.components
        assert orig.dt == back.dt
        assert orig.n_steps == back.n_steps
        assert orig.n_modes == back.n_modes
        if orig.node_ids is not None:
            np.testing.assert_array_equal(orig.node_ids, back.node_ids)
        if orig.element_ids is not None:
            np.testing.assert_array_equal(orig.element_ids, back.element_ids)


# =====================================================================
# Surfacing on g.opensees
# =====================================================================

def test_recorders_attached_to_opensees(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    # Duck-type rather than isinstance — module purges in unrelated
    # tests can produce a different Recorders class identity.
    assert hasattr(g.opensees, "recorders")
    assert type(g.opensees.recorders).__name__ == "Recorders"
    assert hasattr(g.opensees.recorders, "nodes")
    assert hasattr(g.opensees.recorders, "resolve")


def test_recorders_via_opensees_path(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    g.opensees.recorders.nodes(pg="Body", components=["displacement"])
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=6)
    assert len(spec.records) == 1
    assert spec.records[0].category == "nodes"
