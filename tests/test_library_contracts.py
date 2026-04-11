"""
High-level contract tests for the :mod:`apeGmsh` package.

These tests stub out the heavy third-party dependencies (``gmsh``,
``pandas``, ``pyvista``) so they run without a real Gmsh installation.
They verify:

* Top-level package imports expose the canonical public names.
* The ``Gmsh2OpenSees`` wrapper calls into ``gmsh2opensees.gmsh2ops``
  when the owning session is active.
* :func:`apeGmsh.mesh._fem_extract.build_fem_data` correctly extracts
  a 3D element set from a faked Gmsh model.
* The surface-tessellation helper
  :func:`apeGmsh.viewers.scene.brep_scene._surface_polydata_from_global_mesh`
  keeps every triangle whose nodes are present in the global mesh.
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np


def _purge_apegmsh_modules() -> None:
    for name in list(sys.modules):
        if name == "apeGmsh" or name.startswith("apeGmsh."):
            del sys.modules[name]


def _install_fake_gmsh() -> None:
    fake_gmsh = types.ModuleType("gmsh")
    fake_gmsh.__version__ = "fake"
    fake_gmsh.initialize = lambda: None
    fake_gmsh.finalize = lambda: None
    fake_gmsh.model = types.SimpleNamespace(mesh=types.SimpleNamespace())
    sys.modules["gmsh"] = fake_gmsh


class LibraryContractTests(unittest.TestCase):

    def setUp(self) -> None:
        _purge_apegmsh_modules()
        self._saved_gmsh = sys.modules.get("gmsh")
        self._saved_g2o = sys.modules.get("gmsh2opensees")
        _install_fake_gmsh()

    def tearDown(self) -> None:
        _purge_apegmsh_modules()
        if self._saved_gmsh is None:
            sys.modules.pop("gmsh", None)
        else:
            sys.modules["gmsh"] = self._saved_gmsh

        if self._saved_g2o is None:
            sys.modules.pop("gmsh2opensees", None)
        else:
            sys.modules["gmsh2opensees"] = self._saved_g2o

    # ------------------------------------------------------------------
    # Package-level API
    # ------------------------------------------------------------------

    def test_top_level_exports_apegmsh_class(self) -> None:
        pkg = importlib.import_module("apeGmsh")
        self.assertTrue(hasattr(pkg, "apeGmsh"))
        self.assertTrue(hasattr(pkg, "Part"))
        # The v1.0 API does not (and must not) expose Assembly — the
        # session IS the assembly.
        self.assertFalse(hasattr(pkg, "Assembly"))

    # ------------------------------------------------------------------
    # Gmsh2OpenSees live-session wrapper
    # ------------------------------------------------------------------

    def test_g2o_transfer_calls_gmsh2ops_when_active(self) -> None:
        calls: list[str] = []
        fake_g2o = types.ModuleType("gmsh2opensees")
        fake_g2o.gmsh2ops = lambda: calls.append("gmsh2ops")
        fake_g2o.msh2ops = lambda path: calls.append(path)
        sys.modules["gmsh2opensees"] = fake_g2o

        mod = importlib.import_module("apeGmsh.solvers.Gmsh2OpenSees")
        wrapper = mod.Gmsh2OpenSees(
            types.SimpleNamespace(is_active=True, _verbose=False),
        )
        wrapper.transfer()

        self.assertEqual(calls, ["gmsh2ops"])

    def test_g2o_transfer_raises_when_session_inactive(self) -> None:
        mod = importlib.import_module("apeGmsh.solvers.Gmsh2OpenSees")
        wrapper = mod.Gmsh2OpenSees(
            types.SimpleNamespace(is_active=False, _verbose=False),
        )
        with self.assertRaises(RuntimeError):
            wrapper.transfer()

    # ------------------------------------------------------------------
    # _fem_extract.build_fem_data (3D path)
    # ------------------------------------------------------------------

    def test_fem_data_dim3_keeps_volume_physical_elements(self) -> None:
        fake_gmsh = sys.modules["gmsh"]

        coords_by_tag = {
            1: [0.0, 0.0, 0.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [0.0, 0.0, 1.0],
        }

        def _flatten(node_tags: list[int]) -> list[float]:
            flat: list[float] = []
            for tag in node_tags:
                flat.extend(coords_by_tag[tag])
            return flat

        class _FakeMesh:
            def getNodes(self, *args, **kwargs):
                tags = [1, 2, 3, 4]
                return tags, _flatten(tags), []

            def getElements(self, dim=-1, tag=-1):
                data = {
                    (1, -1): ([1], [[301]], [[1, 2]]),
                    (2, -1): ([2], [[201]], [[1, 2, 3]]),
                    (3, -1): ([4], [[101]], [[1, 2, 3, 4]]),
                    (1, 30): ([1], [[301]], [[1, 2]]),
                    (2, 20): ([2], [[201]], [[1, 2, 3]]),
                    (3, 10): ([4], [[101]], [[1, 2, 3, 4]]),
                }
                etypes, elem_tags, node_tags = data.get((dim, tag), ([], [], []))
                return (
                    etypes,
                    [np.array(ts, dtype=np.int64) for ts in elem_tags],
                    [np.array(ns, dtype=np.int64) for ns in node_tags],
                )

            def getElementProperties(self, etype):
                return {
                    1: ("Line 2", 1, 1, 2, [], 2),
                    2: ("Triangle 3", 2, 1, 3, [], 3),
                    4: ("Tetrahedron 4", 3, 1, 4, [], 4),
                }[int(etype)]

            def getNodesForPhysicalGroup(self, dim, pg_tag):
                node_tags = {
                    (1, 3): [1, 2],
                    (2, 2): [1, 2, 3],
                    (3, 1): [1, 2, 3, 4],
                }[(dim, pg_tag)]
                return node_tags, _flatten(node_tags)

        class _FakeModel:
            def __init__(self):
                self.mesh = _FakeMesh()

            def getPhysicalGroups(self):
                return [(1, 3), (2, 2), (3, 1)]

            def getPhysicalName(self, dim, pg_tag):
                return {
                    (1, 3): "Edge",
                    (2, 2): "Face",
                    (3, 1): "Body",
                }[(dim, pg_tag)]

            def getEntitiesForPhysicalGroup(self, dim, pg_tag):
                return {
                    (1, 3): [30],
                    (2, 2): [20],
                    (3, 1): [10],
                }[(dim, pg_tag)]

        fake_gmsh.model = _FakeModel()

        mod = importlib.import_module("apeGmsh.mesh._fem_extract")
        fem = mod.build_fem_data(dim=3)

        self.assertEqual(fem.info.n_nodes, 4)
        self.assertEqual(fem.info.n_elems, 1)
        np.testing.assert_array_equal(fem.node_ids, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(
            fem.connectivity,
            np.array([[1, 2, 3, 4]], dtype=int),
        )

        body_tag = fem.physical.get_tag(3, "Body")
        self.assertEqual(body_tag, 1)

        body = fem.physical.get_elements(3, body_tag)
        face = fem.physical.get_elements(2, 2)
        edge = fem.physical.get_elements(1, 3)

        self.assertEqual(body["connectivity"].shape, (1, 4))
        self.assertEqual(face["connectivity"].shape, (1, 3))
        self.assertEqual(edge["connectivity"].shape, (1, 2))
        self.assertEqual(list(map(int, body["element_ids"])), [101])

    # ------------------------------------------------------------------
    # BRep surface tessellation helper
    # ------------------------------------------------------------------

    def test_surface_polydata_keeps_cells_with_embedded_nodes(self) -> None:
        # Fake pyvista so brep_scene imports (restored in finally block
        # so we don't pollute sys.modules for later tests — see
        # test_results_roundtrip, which needs the real pyvista).
        saved_pv = sys.modules.get("pyvista")
        installed_fake = False
        if saved_pv is None:
            sys.modules["pyvista"] = types.ModuleType("pyvista")
            installed_fake = True
        try:
            mod = importlib.import_module("apeGmsh.viewers.scene.brep_scene")
            self._run_brep_scene_assertions(mod)
        finally:
            if installed_fake:
                sys.modules.pop("pyvista", None)

    def _run_brep_scene_assertions(self, mod) -> None:

        node_tags = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        node_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=float)

        tag_to_idx = np.full(51, -1, dtype=np.int64)
        tag_to_idx[node_tags] = np.arange(len(node_tags), dtype=np.int64)

        pts, faces_parts, n_cells = mod._surface_polydata_from_global_mesh(
            node_coords,
            tag_to_idx,
            [2],
            [
                np.array([
                    10, 20, 50,
                    20, 30, 50,
                    30, 40, 50,
                    40, 10, 50,
                ], dtype=np.int64),
            ],
        )

        self.assertEqual(n_cells, 4)
        self.assertEqual(len(pts), 5)
        self.assertEqual(len(faces_parts), 1)
        # 4 triangles × (1 prefix + 3 nodes) = 16 entries
        self.assertEqual(len(faces_parts[0]), 16)
        self.assertEqual(int(faces_parts[0].max()), 4)
        self.assertTrue(
            any(np.allclose(pt, [0.5, 0.5, 0.0]) for pt in pts),
        )


if __name__ == "__main__":
    unittest.main()
