"""
Round-trip tests for :class:`apeGmsh.results.Results.Results`.

These tests synthesize a small tet-mesh FEMData by hand (no Gmsh
required) and exercise the full Results round-trip pipeline:

1. ``Results.from_fem(...)`` with point / cell / time-series fields
2. ``to_vtu`` → ``from_file`` round-trip (on-disk)
3. ``to_pvd`` → ``from_file`` round-trip (time-series)
4. ``to_mesh_data`` → direct in-memory transfer to the viewer loader
5. ``get_point_field`` / ``get_cell_field`` / ``field_names`` API
6. ``summary`` / ``__repr__`` smoke

They use ``pyvista`` (a transitive dep of the viewer) and the
synthesized ``FEMData`` pretends to have one tet physical group.
The whole suite runs in under a second and needs no live Gmsh
session.

Coverage target: every public entry point on ``Results`` that does
not require an open Qt window, plus the VTU loader path.  The
actual Qt viewer launch is out of scope because it requires a
display.
"""
from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def _install_fake_gmsh() -> None:
    """FEMData imports ``gmsh`` at module top — stub it out."""
    import sys
    if "gmsh" not in sys.modules:
        fake = types.ModuleType("gmsh")
        fake.model = types.SimpleNamespace(mesh=types.SimpleNamespace())
        sys.modules["gmsh"] = fake


_install_fake_gmsh()


from apeGmsh.mesh.FEMData import FEMData, MeshInfo, PhysicalGroupSet
from apeGmsh.results.Results import Results


def _make_unit_tet_fem() -> FEMData:
    """Hand-built FEMData for a single unit tetrahedron.

    Four nodes at the corners, one tet element.  Minimal but
    exercises every Results codepath that depends on mesh topology.
    """
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    element_ids = np.array([1], dtype=np.int64)
    connectivity = np.array([[1, 2, 3, 4]], dtype=np.int64)

    info = MeshInfo(n_nodes=4, n_elems=1, bandwidth=3)

    # One dim=3 physical group covering the single tet.
    groups = {
        (3, 1): {
            "name": "Body",
            "node_ids": node_ids,
            "node_coords": node_coords,
            "element_ids": element_ids,
            "connectivity": connectivity,
        },
    }
    physical = PhysicalGroupSet(groups)

    return FEMData(
        node_ids=node_ids,
        node_coords=node_coords,
        element_ids=element_ids,
        connectivity=connectivity,
        info=info,
        physical=physical,
    )


# =====================================================================
# Static results
# =====================================================================

class TestStaticResults(unittest.TestCase):

    def setUp(self) -> None:
        self.fem = _make_unit_tet_fem()
        self.displacement = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3],
            ],
            dtype=np.float64,
        )
        self.stress = np.array([1.5e6], dtype=np.float64)  # one value per elem

    def test_from_fem_populates_fields_and_mesh(self):
        r = Results.from_fem(
            self.fem,
            point_data={"Displacement": self.displacement},
            cell_data={"sig_vm": self.stress},
            name="unit_tet",
        )
        self.assertEqual(r.name, "unit_tet")
        self.assertEqual(len(r.node_coords), 4)
        self.assertEqual(r.n_primary_cells, 1)
        self.assertEqual(r.n_total_cells, 1)
        self.assertFalse(r.has_time_series)
        self.assertEqual(r.n_steps, 1)
        self.assertIn("Displacement", r.field_names["point"])
        self.assertIn("sig_vm", r.field_names["cell"])

    def test_get_point_field_returns_exact_values(self):
        r = Results.from_fem(
            self.fem,
            point_data={"Displacement": self.displacement},
            name="unit_tet",
        )
        d = r.get_point_field("Displacement")
        np.testing.assert_array_equal(d, self.displacement)

    def test_get_cell_field_returns_exact_values(self):
        r = Results.from_fem(
            self.fem,
            cell_data={"sig_vm": self.stress},
            name="unit_tet",
        )
        s = r.get_cell_field("sig_vm")
        np.testing.assert_array_equal(s, self.stress)

    def test_missing_field_raises_keyerror(self):
        r = Results.from_fem(self.fem, name="unit_tet")
        with self.assertRaises(KeyError):
            r.get_point_field("nope")
        with self.assertRaises(KeyError):
            r.get_cell_field("nope")

    def test_to_vtu_then_from_file_roundtrip(self):
        """Write to .vtu, read back, assert fields survive."""
        r = Results.from_fem(
            self.fem,
            point_data={"Displacement": self.displacement},
            cell_data={"sig_vm": self.stress},
            name="unit_tet",
        )
        with tempfile.TemporaryDirectory() as tmp:
            vtu = r.to_vtu(Path(tmp) / "round.vtu")
            self.assertTrue(vtu.exists())
            r2 = Results.from_file(vtu)
            # Mesh geometry
            self.assertEqual(len(r2.node_coords), 4)
            self.assertEqual(r2.n_total_cells, 1)
            # Fields
            d = r2.get_point_field("Displacement")
            s = r2.get_cell_field("sig_vm")
            np.testing.assert_allclose(d, self.displacement)
            np.testing.assert_allclose(s, self.stress)

    def test_to_mesh_data_populates_mesh_and_fields(self):
        """Direct in-memory transfer to the viewer loader."""
        r = Results.from_fem(
            self.fem,
            point_data={"Displacement": self.displacement},
            cell_data={"sig_vm": self.stress},
            name="unit_tet",
        )
        md = r.to_mesh_data()
        self.assertEqual(md.n_points, 4)
        self.assertEqual(md.n_cells, 1)
        self.assertIn("Displacement", md.point_field_names)
        self.assertIn("sig_vm", md.cell_field_names)
        self.assertFalse(md.has_time_series)

    def test_summary_and_repr_contain_mesh_stats(self):
        r = Results.from_fem(
            self.fem,
            point_data={"Displacement": self.displacement},
            name="unit_tet",
        )
        summary = r.summary()
        self.assertIn("unit_tet", summary)
        self.assertIn("4 nodes", summary)
        self.assertIn("Displacement", summary)

        rep = repr(r)
        self.assertIn("unit_tet", rep)
        self.assertIn("4 nodes", rep)


# =====================================================================
# Time-series results
# =====================================================================

class TestTimeSeriesResults(unittest.TestCase):

    def setUp(self) -> None:
        self.fem = _make_unit_tet_fem()
        # 3 mode shapes at 3 distinct frequencies
        rng = np.random.default_rng(42)
        self.step_shapes = [rng.standard_normal((4, 3)) for _ in range(3)]
        self.frequencies = [1.0, 2.5, 4.7]
        self.steps = [
            {"time": f, "point_data": {"ModeShape": phi}}
            for f, phi in zip(self.frequencies, self.step_shapes)
        ]

    def test_from_fem_with_steps_builds_time_series(self):
        r = Results.from_fem(self.fem, steps=self.steps, name="modal")
        self.assertTrue(r.has_time_series)
        self.assertEqual(r.n_steps, 3)
        self.assertEqual(r.time_steps, self.frequencies)
        self.assertIn("ModeShape", r.field_names["point"])

    def test_get_point_field_requires_step_for_time_series(self):
        r = Results.from_fem(self.fem, steps=self.steps, name="modal")
        with self.assertRaises(ValueError):
            r.get_point_field("ModeShape")  # must pass step=
        phi0 = r.get_point_field("ModeShape", step=0)
        np.testing.assert_array_equal(phi0, self.step_shapes[0])

    def test_get_point_field_step_out_of_range_raises(self):
        r = Results.from_fem(self.fem, steps=self.steps, name="modal")
        with self.assertRaises(IndexError):
            r.get_point_field("ModeShape", step=99)

    def test_to_pvd_writes_collection_and_per_step_vtus(self):
        r = Results.from_fem(self.fem, steps=self.steps, name="modal")
        with tempfile.TemporaryDirectory() as tmp:
            paths = r.to_pvd(Path(tmp) / "modal")
            # 1 pvd + 3 vtu = 4 files
            self.assertEqual(len(paths), 4)
            self.assertTrue(paths[0].suffix == ".pvd")
            for p in paths[1:]:
                self.assertTrue(p.suffix == ".vtu")
                self.assertTrue(p.exists())

    def test_from_file_roundtrip_via_pvd(self):
        r = Results.from_fem(self.fem, steps=self.steps, name="modal")
        with tempfile.TemporaryDirectory() as tmp:
            paths = r.to_pvd(Path(tmp) / "modal")
            pvd = paths[0]
            r2 = Results.from_file(pvd)
            self.assertTrue(r2.has_time_series)
            self.assertEqual(r2.n_steps, 3)
            # Frequencies come back as the timestep values
            np.testing.assert_allclose(r2.time_steps, self.frequencies)
            # Mode shape at step 0 survives the round trip
            phi0 = r2.get_point_field("ModeShape", step=0)
            np.testing.assert_allclose(phi0, self.step_shapes[0])

    def test_to_mesh_data_time_series(self):
        r = Results.from_fem(self.fem, steps=self.steps, name="modal")
        md = r.to_mesh_data()
        self.assertTrue(md.has_time_series)
        self.assertEqual(len(md.step_meshes), 3)
        self.assertEqual(md.time_steps, self.frequencies)
        self.assertIn("ModeShape", md.point_field_names)


# =====================================================================
# Mutual exclusion
# =====================================================================

class TestFromFemValidation(unittest.TestCase):

    def setUp(self) -> None:
        self.fem = _make_unit_tet_fem()

    def test_cannot_mix_static_and_steps(self):
        """``point_data`` and ``steps`` are mutually exclusive."""
        with self.assertRaises(ValueError):
            Results.from_fem(
                self.fem,
                point_data={"u": np.zeros((4, 3))},
                steps=[{"time": 0.0, "point_data": {"u": np.zeros((4, 3))}}],
            )


if __name__ == "__main__":
    unittest.main()
