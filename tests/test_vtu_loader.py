"""
Unit tests for :mod:`apeGmshViewer.loaders.vtu_loader`.

Covers:

* ``from_arrays`` — the in-memory ``FEMData → MeshData`` path used by
  ``Results.to_mesh_data``.
* ``load_vtu`` — reading a hand-written .vtu back.
* ``load_pvd`` — reading a hand-written .pvd collection.
* ``create_deformed_mesh`` — the utility used by the deformed-shape
  overlay in the renderer.

These tests do not need Gmsh and do not open any Qt windows — they
exercise only the pure-python data path from numpy arrays through
pyvista.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

from apeGmshViewer.loaders.vtu_loader import (
    MeshData,
    create_deformed_mesh,
    from_arrays,
    load_file,
    load_pvd,
    load_vtu,
)


def _unit_tet_arrays():
    """Return (node_coords, cells_flat, cell_types) for one unit tet."""
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # VTK cell array format: [n_pts_in_cell, idx0, idx1, idx2, idx3, ...]
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.array([pv.CellType.TETRA], dtype=np.uint8)
    return node_coords, cells, cell_types


# =====================================================================
# from_arrays
# =====================================================================

class TestFromArrays(unittest.TestCase):

    def test_static_mesh_with_point_and_cell_data(self):
        coords, cells, ct = _unit_tet_arrays()
        md = from_arrays(
            coords, cells, ct,
            point_data={"Temperature": np.array([1.0, 2.0, 3.0, 4.0])},
            cell_data={"Stress": np.array([42.0])},
            name="unit_tet",
        )
        self.assertEqual(md.name, "unit_tet")
        self.assertEqual(md.n_points, 4)
        self.assertEqual(md.n_cells, 1)
        self.assertEqual(md.point_field_names, ["Temperature"])
        self.assertEqual(md.cell_field_names, ["Stress"])
        self.assertFalse(md.has_time_series)

    def test_time_series_mesh(self):
        coords, cells, ct = _unit_tet_arrays()
        times = [0.0, 1.0, 2.0]
        step_pd = {
            "Disp": [
                np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]),
                np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]]),
                np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]),
            ],
        }
        md = from_arrays(
            coords, cells, ct,
            time_steps=times,
            step_point_data=step_pd,
            name="modal",
        )
        self.assertTrue(md.has_time_series)
        self.assertEqual(md.time_steps, times)
        self.assertEqual(len(md.step_meshes), 3)
        self.assertIn("Disp", md.point_field_names)
        # Each step grid should have the Disp field attached
        for i, step_mesh in enumerate(md.step_meshes):
            self.assertIn("Disp", step_mesh.point_data)
            np.testing.assert_array_equal(
                step_mesh.point_data["Disp"], step_pd["Disp"][i],
            )


# =====================================================================
# load_vtu
# =====================================================================

class TestLoadVtu(unittest.TestCase):

    def test_roundtrip_static_vtu(self):
        """Write a VTU via pyvista, load via load_vtu, check shape."""
        coords, cells, ct = _unit_tet_arrays()
        grid = pv.UnstructuredGrid(cells, ct, coords)
        grid.point_data["Scalar"] = np.array([0.1, 0.2, 0.3, 0.4])
        grid.cell_data["Sig_vm"] = np.array([99.0])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.vtu"
            grid.save(str(path))

            md = load_vtu(path)
            self.assertEqual(md.name, "test")
            self.assertEqual(md.n_points, 4)
            self.assertEqual(md.n_cells, 1)
            self.assertIn("Scalar", md.point_field_names)
            self.assertIn("Sig_vm", md.cell_field_names)

    def test_load_file_dispatches_on_extension(self):
        """``load_file`` should route .vtu to load_vtu and .pvd to load_pvd."""
        coords, cells, ct = _unit_tet_arrays()
        grid = pv.UnstructuredGrid(cells, ct, coords)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auto.vtu"
            grid.save(str(path))
            md = load_file(path)
            self.assertIsInstance(md, MeshData)
            self.assertEqual(md.n_points, 4)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_vtu(Path("/nonexistent/path/does_not_exist.vtu"))


# =====================================================================
# load_pvd
# =====================================================================

class TestLoadPvd(unittest.TestCase):

    def test_roundtrip_pvd_time_series(self):
        """Write a 3-step PVD by hand, load it, check the series."""
        coords, cells, ct = _unit_tet_arrays()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Write 3 individual VTU files
            for i, t in enumerate([0.0, 1.0, 2.0]):
                grid = pv.UnstructuredGrid(cells, ct, coords)
                grid.point_data["T"] = np.full(4, float(i + 1))
                grid.save(str(tmp_path / f"step_{i:03d}.vtu"))

            # Write a tiny PVD collection file pointing at them
            pvd_text = (
                '<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1">\n'
                '  <Collection>\n'
                '    <DataSet timestep="0.0" file="step_000.vtu"/>\n'
                '    <DataSet timestep="1.0" file="step_001.vtu"/>\n'
                '    <DataSet timestep="2.0" file="step_002.vtu"/>\n'
                '  </Collection>\n'
                '</VTKFile>\n'
            )
            pvd_path = tmp_path / "series.pvd"
            pvd_path.write_text(pvd_text, encoding="utf-8")

            md = load_pvd(pvd_path)
            self.assertTrue(md.has_time_series)
            self.assertEqual(len(md.step_meshes), 3)
            self.assertEqual(md.time_steps, [0.0, 1.0, 2.0])

    def test_load_file_dispatches_pvd(self):
        coords, cells, ct = _unit_tet_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            grid = pv.UnstructuredGrid(cells, ct, coords)
            grid.save(str(tmp_path / "s_000.vtu"))
            (tmp_path / "s.pvd").write_text(
                '<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1">\n'
                '  <Collection>\n'
                '    <DataSet timestep="0.0" file="s_000.vtu"/>\n'
                '  </Collection>\n'
                '</VTKFile>\n',
                encoding="utf-8",
            )
            md = load_file(tmp_path / "s.pvd")
            # A 1-step PVD is NOT a time series per the MeshData property
            self.assertEqual(len(md.step_meshes), 1)

    def test_empty_collection_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "empty.pvd"
            bad.write_text(
                '<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1">\n'
                '  <Collection></Collection>\n'
                '</VTKFile>\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_pvd(bad)


# =====================================================================
# create_deformed_mesh
# =====================================================================

class TestCreateDeformedMesh(unittest.TestCase):

    def setUp(self) -> None:
        coords, cells, ct = _unit_tet_arrays()
        self.md = from_arrays(
            coords, cells, ct,
            point_data={
                "Displacement": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0],
                        [0.0, 0.2, 0.0],
                        [0.0, 0.0, 0.3],
                    ],
                ),
            },
            name="tet",
        )

    def test_applies_scale_factor(self):
        deformed = create_deformed_mesh(
            self.md, displacement_field="Displacement", scale_factor=10.0,
        )
        orig_pts = np.asarray(self.md.mesh.points)
        new_pts = np.asarray(deformed.points)
        expected = orig_pts + 10.0 * np.array(
            self.md.mesh.point_data["Displacement"],
        )
        np.testing.assert_allclose(new_pts, expected)

    def test_raises_on_missing_field(self):
        with self.assertRaises(KeyError):
            create_deformed_mesh(
                self.md, displacement_field="nonexistent", scale_factor=1.0,
            )


if __name__ == "__main__":
    unittest.main()
