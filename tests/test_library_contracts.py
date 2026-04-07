from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np


def _purge_pygmsh_modules() -> None:
    for name in list(sys.modules):
        if name == "pyGmsh" or name.startswith("pyGmsh."):
            del sys.modules[name]


def _install_fake_gmsh() -> None:
    fake_gmsh = types.ModuleType("gmsh")
    fake_gmsh.__version__ = "fake"
    fake_gmsh.initialize = lambda: None
    fake_gmsh.finalize = lambda: None
    fake_gmsh.model = types.SimpleNamespace(mesh=types.SimpleNamespace())
    sys.modules["gmsh"] = fake_gmsh


def _install_fake_pandas() -> None:
    fake_pandas = types.ModuleType("pandas")

    class _FakeDataFrame:
        def __init__(self, *args, **kwargs) -> None:
            pass

    fake_pandas.DataFrame = _FakeDataFrame
    sys.modules["pandas"] = fake_pandas


class LibraryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        _purge_pygmsh_modules()
        self._saved_gmsh = sys.modules.get("gmsh")
        self._saved_g2o = sys.modules.get("gmsh2opensees")
        self._saved_pandas = sys.modules.get("pandas")
        _install_fake_gmsh()
        _install_fake_pandas()

    def tearDown(self) -> None:
        _purge_pygmsh_modules()
        if self._saved_gmsh is None:
            sys.modules.pop("gmsh", None)
        else:
            sys.modules["gmsh"] = self._saved_gmsh

        if self._saved_g2o is None:
            sys.modules.pop("gmsh2opensees", None)
        else:
            sys.modules["gmsh2opensees"] = self._saved_g2o

        if self._saved_pandas is None:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = self._saved_pandas

    def test_top_level_import_only_requires_core_dependencies(self) -> None:
        pkg = importlib.import_module("pyGmsh")
        self.assertTrue(hasattr(pkg, "pyGmsh"))
        self.assertTrue(hasattr(pkg, "Assembly"))

    def test_g2o_accepts_assembly_style_active_context(self) -> None:
        calls: list[str] = []
        fake_g2o = types.ModuleType("gmsh2opensees")
        fake_g2o.gmsh2ops = lambda: calls.append("gmsh2ops")
        fake_g2o.msh2ops = lambda path: calls.append(path)
        sys.modules["gmsh2opensees"] = fake_g2o

        mod = importlib.import_module("pyGmsh.solvers.Gmsh2OpenSees")
        wrapper = mod.Gmsh2OpenSees(types.SimpleNamespace(is_active=True, _verbose=False))
        wrapper.transfer()

        self.assertEqual(calls, ["gmsh2ops"])

    def test_equal_dof_uses_instance_scope_without_manual_maps(self) -> None:
        mod = importlib.import_module("pyGmsh.core.Assembly")
        asm = mod.Assembly("demo")
        asm.instances = {
            "left": mod.Instance(
                label="left",
                part_name="left",
                file_path=types.SimpleNamespace(),
                bbox=(-1.0, -1.0, -0.1, 1.0, 1.0, 0.0),
            ),
            "right": mod.Instance(
                label="right",
                part_name="right",
                file_path=types.SimpleNamespace(),
                bbox=(-1.0, -1.0, 1e-4, 1.0, 1.0, 0.1),
            ),
        }
        asm.equal_dof("left", "right", tolerance=1e-3)

        node_tags = np.array([1, 2, 3, 4], dtype=int)
        node_coords = np.array(
            [
                [0.0, 0.0, -0.05],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 5e-4],
                [0.0, 0.0, 0.05],
            ],
            dtype=float,
        )

        records = asm.resolve_constraints(node_tags, node_coords)
        pairs = [(rec.master_node, rec.slave_node) for rec in records]
        self.assertEqual(pairs, [(2, 3)])

    def test_rigid_body_master_is_chosen_from_master_instance(self) -> None:
        mod = importlib.import_module("pyGmsh.core.Assembly")
        asm = mod.Assembly("demo")
        asm.instances = {
            "master": mod.Instance(
                label="master",
                part_name="master",
                file_path=types.SimpleNamespace(),
                bbox=(0.1, -1.0, -1.0, 0.3, 1.0, 1.0),
            ),
            "slave": mod.Instance(
                label="slave",
                part_name="slave",
                file_path=types.SimpleNamespace(),
                bbox=(-0.3, -1.0, -1.0, 0.05, 1.0, 1.0),
            ),
        }
        asm.rigid_body("master", "slave", master_point=(0.0, 0.0, 0.0))

        node_tags = np.array([1, 2, 3], dtype=int)
        node_coords = np.array(
            [
                [0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.2, 0.0, 0.0],
            ],
            dtype=float,
        )

        records = asm.resolve_constraints(node_tags, node_coords)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].master_node, 1)

    def test_public_constraint_api_accepts_entity_filters(self) -> None:
        mod = importlib.import_module("pyGmsh.core.Assembly")
        asm = mod.Assembly("demo")
        asm.instances = {
            "solid": mod.Instance(
                label="solid",
                part_name="solid",
                file_path=types.SimpleNamespace(),
            ),
            "frame": mod.Instance(
                label="frame",
                part_name="frame",
                file_path=types.SimpleNamespace(),
            ),
        }

        eq = asm.equal_dof(
            "solid",
            "frame",
            slave_entities=[(0, 10), (0, 11)],
            dofs=[1, 2, 3],
        )
        rl = asm.rigid_link(
            "frame",
            "frame",
            master_point=(1.0, 0.0, 0.0),
            slave_entities=[(0, 10), (0, 11)],
        )

        self.assertEqual(eq.slave_entities, [(0, 10), (0, 11)])
        self.assertEqual(eq.dofs, [1, 2, 3])
        self.assertEqual(rl.slave_entities, [(0, 10), (0, 11)])


if __name__ == "__main__":
    unittest.main()