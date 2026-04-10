"""
Unit test for :meth:`apeGmsh.solvers._opensees_inspect._Inspect.node_table`.

The test stubs out the Gmsh dependency (by installing a fake ``gmsh``
module into ``sys.modules``) and constructs an ``OpenSees`` bridge
directly against a fake parent.  It then populates the internal
tables the way ``build()`` would and asserts the shape of the
DataFrame returned by ``bridge.inspect.node_table()``.
"""
from __future__ import annotations

import sys
import types
import unittest

import pandas as pd


def _purge_apegmsh_modules() -> None:
    for name in list(sys.modules):
        if name == "apeGmsh" or name.startswith("apeGmsh."):
            del sys.modules[name]


class OpenSeesNodeTableTests(unittest.TestCase):

    def setUp(self) -> None:
        _purge_apegmsh_modules()
        self._saved_gmsh = sys.modules.get("gmsh")
        fake_gmsh = types.ModuleType("gmsh")
        fake_gmsh.model = types.SimpleNamespace(
            mesh=types.SimpleNamespace(),
        )
        sys.modules["gmsh"] = fake_gmsh

    def tearDown(self) -> None:
        _purge_apegmsh_modules()
        if self._saved_gmsh is None:
            sys.modules.pop("gmsh", None)
        else:
            sys.modules["gmsh"] = self._saved_gmsh

    def test_node_table_includes_fix_and_load_columns(self) -> None:
        mod = __import__("apeGmsh.solvers.OpenSees", fromlist=["OpenSees"])
        bridge = mod.OpenSees(
            types.SimpleNamespace(_verbose=False, name="demo"),
        )
        bridge._built = True
        bridge._ndf = 3
        bridge._nodes_df = pd.DataFrame({
            "ops_id": [1, 2, 3],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        }).set_index("ops_id")
        bridge._bcs = {"Fixed": {"dofs": [1, 1, 1], "dim": 2}}
        bridge._load_patterns = {
            "Wind": [
                {"type": "nodal", "pg_name": "TopFace",
                 "force": [5.0, 0.0, 0.0], "dim": 2},
            ],
            "Gravity": [
                {"type": "nodal", "pg_name": "TopFace",
                 "force": [0.0, 0.0, -2.0], "dim": 2},
            ],
        }
        bridge._nodes_for_pg = lambda pg_name, dim=None: {
            "Fixed": [1, 2],
            "TopFace": [2, 3],
        }[pg_name]

        df = bridge.inspect.node_table()

        expected_cols = {
            "fix_1", "fix_2", "fix_3",
            "load_1", "load_2", "load_3",
        }
        self.assertTrue(expected_cols.issubset(df.columns))
        self.assertTrue(df.loc[1, "fix_1"])
        self.assertTrue(df.loc[2, "fix_3"])
        self.assertFalse(df.loc[3, "fix_2"])
        self.assertEqual(df.loc[1, "load_1"], 0.0)
        self.assertEqual(df.loc[2, "load_1"], 5.0)
        self.assertEqual(df.loc[3, "load_3"], -2.0)
        self.assertNotIn("fix_1", bridge._nodes_df.columns)


if __name__ == "__main__":
    unittest.main()
