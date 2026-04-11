"""
Phase 11a round-trip tests for tie export.

Verifies that tie entries in ``ops._tie_elements`` are rendered
correctly by both exporters:

1. ``g.opensees.export.tcl`` emits ``element ASDEmbeddedNodeElement``
   lines in a dedicated ``# Tied interfaces`` section.
2. ``g.opensees.export.py`` emits ``ops.element('ASDEmbeddedNodeElement'
   , ...)`` calls in the same section.

The tests use a stub broker with hand-populated state so we don't
need a live Gmsh mesh; the export methods only read from the broker's
internal dicts/dataframes.
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


def _install_fake_gmsh() -> None:
    if "gmsh" not in sys.modules:
        fake = types.ModuleType("gmsh")
        fake.model = types.SimpleNamespace(mesh=types.SimpleNamespace())
        sys.modules["gmsh"] = fake


_install_fake_gmsh()

from apeGmsh.solvers._opensees_constraints import (
    render_tie_py,
    render_tie_tcl,
)


# =====================================================================
# Direct render_tie_tcl / render_tie_py smoke tests
# =====================================================================

class TestRenderTieTcl(unittest.TestCase):

    def test_minimal_tie_no_rot_no_penalty(self):
        entry = {
            "ele_tag": 1001, "cNode": 42, "rNodes": [11, 12, 13],
            "use_rot": False, "penalty": None, "source_kind": "tie",
        }
        line = render_tie_tcl(entry)
        # Exact prefix — this locks the argument order that the
        # OpenSees source expects: $tag $Cnode $Rnode1 $Rnode2 $Rnode3.
        self.assertTrue(
            line.startswith("element ASDEmbeddedNodeElement  1001  42  11  12  13"),
            f"line did not start as expected: {line!r}",
        )
        # No flags
        self.assertNotIn("-rot", line)
        self.assertNotIn("-K", line)

    def test_tie_with_rot(self):
        entry = {
            "ele_tag": 1001, "cNode": 42, "rNodes": [11, 12, 13],
            "use_rot": True, "penalty": None, "source_kind": "tie",
        }
        line = render_tie_tcl(entry)
        self.assertIn("-rot", line)

    def test_tie_with_penalty(self):
        entry = {
            "ele_tag": 1001, "cNode": 42, "rNodes": [11, 12, 13],
            "use_rot": False, "penalty": 1.5e12, "source_kind": "tie",
        }
        line = render_tie_tcl(entry)
        self.assertIn("-K", line)
        # Uses '%.10g' formatting
        self.assertIn("1.5e+12", line.lower())

    def test_tie_with_rot_and_penalty(self):
        entry = {
            "ele_tag": 1001, "cNode": 42, "rNodes": [11, 12, 13],
            "use_rot": True, "penalty": 1e10, "source_kind": "tie",
        }
        line = render_tie_tcl(entry)
        self.assertIn("-rot", line)
        self.assertIn("-K", line)


class TestRenderTiePy(unittest.TestCase):

    def test_minimal_tie_no_rot_no_penalty(self):
        entry = {
            "ele_tag": 1001, "cNode": 42, "rNodes": [11, 12, 13],
            "use_rot": False, "penalty": None, "source_kind": "tie",
        }
        line = render_tie_py(entry)
        self.assertIn("ops.element", line)
        self.assertIn("'ASDEmbeddedNodeElement'", line)
        self.assertIn("1001", line)
        self.assertIn("42", line)
        self.assertIn("11", line)
        self.assertIn("12", line)
        self.assertIn("13", line)
        self.assertNotIn("'-rot'", line)
        self.assertNotIn("'-K'", line)

    def test_tie_with_rot(self):
        entry = {
            "ele_tag": 1001, "cNode": 42, "rNodes": [11, 12, 13],
            "use_rot": True, "penalty": None, "source_kind": "tie",
        }
        line = render_tie_py(entry)
        self.assertIn("'-rot'", line)

    def test_tie_with_penalty(self):
        entry = {
            "ele_tag": 1001, "cNode": 42, "rNodes": [11, 12, 13],
            "use_rot": False, "penalty": 1.5e12, "source_kind": "tie",
        }
        line = render_tie_py(entry)
        self.assertIn("'-K'", line)
        self.assertIn("1.5e+12", line.lower())


# =====================================================================
# End-to-end export round-trip through the broker
# =====================================================================

class _FakeParent:
    """Stand-in for ``apeGmsh._SessionBase`` with just what the
    broker needs: ``name`` and ``_verbose``."""
    def __init__(self, name="tie_test", verbose=False):
        self.name = name
        self._verbose = verbose


class TestTcllExportSection(unittest.TestCase):
    """Build a tiny broker with pre-populated state, export to Tcl,
    verify the ``Tied interfaces`` section appears with the right
    entries."""

    def _make_broker_with_tie(
        self,
        tie_entries: list[dict],
    ):
        from apeGmsh.solvers.OpenSees import OpenSees
        ops = OpenSees(_FakeParent())
        ops.set_model(ndm=3, ndf=6)

        # Pre-populate the minimum broker state that export.tcl
        # reads.  We don't use the real build() because that calls
        # into gmsh; instead we stub the post-build fields directly.
        ops._nodes_df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 0.0],
                "y": [0.0, 0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 0.0, 1.0],
            },
            index=pd.Index([1, 2, 3, 4], name="ops_id"),
        )
        ops._elements_df = pd.DataFrame(
            columns=[
                "gmsh_id", "ops_type", "pg_name", "mat_name", "mat_tag",
                "sec_tag", "transf_tag", "n_nodes", "nodes", "slots",
                "extra",
            ],
        )
        ops._tie_elements = list(tie_entries)
        ops._built = True
        return ops

    def test_tcl_export_contains_tie_section(self):
        entries = [
            {
                "ele_tag": 1_000_000, "cNode": 4,
                "rNodes": [1, 2, 3],
                "use_rot": True, "penalty": 1e12,
                "source_kind": "tie",
            },
        ]
        ops = self._make_broker_with_tie(entries)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.tcl"
            ops.export.tcl(path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("Tied interfaces", text)
        self.assertIn("element ASDEmbeddedNodeElement", text)
        self.assertIn("1000000", text)
        self.assertIn("-rot", text)
        self.assertIn("-K", text)
        # Exactly one ASDEmbeddedNodeElement line
        ase_lines = [
            ln for ln in text.splitlines()
            if "ASDEmbeddedNodeElement" in ln and "element" in ln
        ]
        self.assertEqual(len(ase_lines), 1)

    def test_py_export_contains_tie_section(self):
        entries = [
            {
                "ele_tag": 1_000_000, "cNode": 4,
                "rNodes": [1, 2, 3],
                "use_rot": False, "penalty": None,
                "source_kind": "tie",
            },
            {
                "ele_tag": 1_000_001, "cNode": 4,
                "rNodes": [1, 2, 3],
                "use_rot": True, "penalty": 1e10,
                "source_kind": "tie",
            },
        ]
        ops = self._make_broker_with_tie(entries)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.py"
            ops.export.py(path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("Tied interfaces", text)
        self.assertIn("'ASDEmbeddedNodeElement'", text)

        ase_lines = [
            ln for ln in text.splitlines()
            if "'ASDEmbeddedNodeElement'" in ln
        ]
        self.assertEqual(len(ase_lines), 2)

        # First entry: no -rot, no -K
        self.assertNotIn("'-rot'", ase_lines[0])
        self.assertNotIn("'-K'", ase_lines[0])
        # Second entry: both -rot and -K
        self.assertIn("'-rot'", ase_lines[1])
        self.assertIn("'-K'", ase_lines[1])

    def test_no_tie_section_when_empty(self):
        """If ``_tie_elements`` is empty, no section is emitted."""
        ops = self._make_broker_with_tie([])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.tcl"
            ops.export.tcl(path)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("Tied interfaces", text)
        self.assertNotIn("ASDEmbeddedNodeElement", text)


if __name__ == "__main__":
    unittest.main()
