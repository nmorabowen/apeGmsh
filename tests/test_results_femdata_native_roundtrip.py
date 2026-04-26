"""Phase 1 — FEMData → embedded /model/ → reconstructed FEMData equality.

The critical contract: the reconstructed FEMData must produce the same
``snapshot_id`` as the original. That's what makes ``Results.bind()``
work for self-contained native files.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from apeGmsh.results.readers import NativeReader
from apeGmsh.results.writers import NativeWriter


def test_embedded_fem_roundtrip_real_session(g, tmp_path: Path) -> None:
    """Real FEMData from a small gmsh session round-trips through HDF5."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    fem = g.mesh.queries.get_fem_data(dim=3)
    original_id = fem.snapshot_id

    # Write a results file with embedded FEMData
    path = tmp_path / "embedded.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem, source_type="domain_capture")

    # Read it back via NativeReader.fem()
    with NativeReader(path) as r:
        fem_back = r.fem()
        assert fem_back is not None
        assert fem_back.snapshot_id == original_id


def test_embedded_fem_preserves_node_data_directly(g, tmp_path: Path) -> None:
    """Spot-check node IDs and coords, not just the hash."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    fem = g.mesh.queries.get_fem_data(dim=3)
    orig_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    orig_coords = np.asarray(fem.nodes.coords, dtype=np.float64)

    path = tmp_path / "embedded.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)

    with NativeReader(path) as r:
        fem_back = r.fem()
        ids_back = np.asarray(fem_back.nodes.ids, dtype=np.int64)
        coords_back = np.asarray(fem_back.nodes.coords, dtype=np.float64)

    # Use sorted equality (the reconstruction may permute internally)
    np.testing.assert_array_equal(np.sort(ids_back), np.sort(orig_ids))
    # Coords match by node ID (sort both by ID first)
    s_orig = np.argsort(orig_ids)
    s_back = np.argsort(ids_back)
    np.testing.assert_allclose(orig_coords[s_orig], coords_back[s_back])


def test_embedded_fem_preserves_physical_groups(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.physical.add_surface(
        g.model.queries.boundary([(3, 1)]),
        name="Boundary",
    )
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "embedded.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)

    with NativeReader(path) as r:
        fem_back = r.fem()

    orig_pgs = sorted(fem.nodes.physical.names())
    back_pgs = sorted(fem_back.nodes.physical.names())
    assert orig_pgs == back_pgs


def test_embedded_fem_carries_snapshot_id_attr(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "embedded.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)

    # Read the attr directly via h5py
    with h5py.File(path, "r") as f:
        assert "/model" in f
        sid = str(f["/model"].attrs["snapshot_id"])
        assert sid == fem.snapshot_id


def test_mesh_selection_roundtrips(g, tmp_path: Path) -> None:
    """Post-mesh selections embed in /model/mesh_selection/ and round-trip."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    g.mesh_selection.add_nodes(on_plane=("z", 0.0, 1e-3), name="bottom")
    g.mesh_selection.add_elements(
        dim=3, in_box=[0, 0, 0.5, 1, 1, 1.001], name="top_slab",
    )
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "with_selection.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)

    with NativeReader(path) as r:
        fem_back = r.fem()
        # Selections survive the round-trip.
        assert "bottom" in fem_back.mesh_selection.names()
        assert "top_slab" in fem_back.mesh_selection.names()
        # Node IDs match the original.
        np.testing.assert_array_equal(
            np.sort(fem.mesh_selection.node_ids("bottom")),
            np.sort(fem_back.mesh_selection.node_ids("bottom")),
        )
        np.testing.assert_array_equal(
            np.sort(fem.mesh_selection.element_ids("top_slab")),
            np.sort(fem_back.mesh_selection.element_ids("top_slab")),
        )


def test_writer_omits_model_when_no_fem(tmp_path: Path) -> None:
    """Writer can be used without a FEMData (raw inspection use case)."""
    path = tmp_path / "no_model.h5"
    with NativeWriter(path) as w:
        w.open()       # no fem= passed

    with h5py.File(path, "r") as f:
        assert "/model" not in f

    with NativeReader(path) as r:
        assert r.fem() is None
