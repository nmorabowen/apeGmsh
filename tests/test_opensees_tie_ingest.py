"""
Phase 11a regression tests — tie constraint ingest + emission path.

These tests exercise the flow from a hand-built
:class:`apeGmsh.mesh.FEMData.ConstraintSet` through
``g.opensees.ingest.constraints(fem)`` into
``ops._tie_elements``, without involving Gmsh, pyvista, or a real
OpenSees binding.

Coverage goals
--------------

* ``_quad4_split_pick`` — picks the right triangle for each quadrant
  of the Quad4 isoparametric square.
* ``_pick_retained_nodes`` — Tri3, Quad4, Tri6 downgrade, Quad8
  downgrade, and the unsupported-cardinality path.
* ``emit_tie_elements`` — full walk from ``_constraint_records``
  through ``_tie_elements``, with:
    - Tri3 master + translational dofs → no ``-rot``
    - Tri3 master + dofs 1..6 → ``-rot`` set
    - Quad4 master + slave projected into triangle A vs triangle B
    - Tri6 master → downgrade logged, first 3 corners kept
    - 5-node master → skipped, ``tie_skipped`` incremented
    - ``tie_penalty`` propagates from broker state to each entry
* ``ingest.constraints`` — fills ``_constraint_records`` +
  ``_tie_penalty`` on the broker and is a no-op for empty sets.
"""
from __future__ import annotations

import sys
import types
import unittest

import numpy as np


def _install_fake_gmsh() -> None:
    if "gmsh" not in sys.modules:
        fake = types.ModuleType("gmsh")
        fake.model = types.SimpleNamespace(mesh=types.SimpleNamespace())
        sys.modules["gmsh"] = fake


def _purge_apegmsh_modules() -> None:
    """Remove any cached ``apeGmsh.*`` modules so subsequent imports
    get fresh class objects.  Required because other tests in the
    suite (notably ``test_library_contracts``) purge and re-import
    ``apeGmsh.*`` in their tearDown — leaving us holding stale class
    references that fail ``isinstance`` checks inside lazy imports
    in ``ConstraintSet.interpolations()``.
    """
    for name in list(sys.modules):
        if name == "apeGmsh" or name.startswith("apeGmsh."):
            del sys.modules[name]


_install_fake_gmsh()
_purge_apegmsh_modules()


def _fresh_imports():
    """Re-import the symbols we need with a guaranteed-fresh
    ``apeGmsh.solvers.Constraints`` module.  The return tuple is
    unpacked at the top of each TestCase's setUp."""
    _purge_apegmsh_modules()
    from apeGmsh.mesh._record_set import SurfaceConstraintSet as ConstraintSet
    from apeGmsh.solvers.Constraints import InterpolationRecord
    from apeGmsh.solvers._opensees_constraints import (
        _make_tie_tag_allocator,
        _pick_retained_nodes,
        _quad4_split_pick,
        emit_tie_elements,
    )
    return (
        ConstraintSet,
        InterpolationRecord,
        _make_tie_tag_allocator,
        _pick_retained_nodes,
        _quad4_split_pick,
        emit_tie_elements,
    )


def _make_tie_rec_with(
    InterpolationRecord,
    *,
    slave_node: int,
    master_nodes: list[int],
    dofs: list[int],
    xi: float = 0.0,
    eta: float = 0.0,
    kind: str = "tie",
):
    """Hand-build an ``InterpolationRecord`` with the given class.

    Takes ``InterpolationRecord`` as an explicit argument so the
    caller can pass the freshly-imported class from ``setUp`` and
    avoid stale-class identity issues with lazy imports inside
    ``ConstraintSet.interpolations()``.
    """
    return InterpolationRecord(
        kind=kind,
        slave_node=slave_node,
        master_nodes=list(master_nodes),
        weights=np.ones(len(master_nodes)) / len(master_nodes),
        dofs=list(dofs),
        projected_point=np.zeros(3),
        parametric_coords=np.array([xi, eta]),
    )


class _FakeBroker:
    """Minimum broker surface that ``emit_tie_elements`` touches.

    We don't want to instantiate the real ``OpenSees`` class here
    because it pulls in the full composite wiring (materials,
    elements, etc.).  A duck-typed broker with just the fields the
    emitter reads/writes is enough.
    """
    def __init__(
        self,
        constraint_records=None,
        tie_penalty=None,
        elements_df=None,
    ):
        import pandas as pd
        self._constraint_records = constraint_records
        self._tie_penalty = tie_penalty
        self._elements_df = (
            elements_df
            if elements_df is not None
            else pd.DataFrame(columns=["dummy"])
        )
        self._tie_elements: list[dict] = []
        self._log_messages: list[str] = []

    def _log(self, msg: str) -> None:
        self._log_messages.append(msg)


class _TieTestBase(unittest.TestCase):
    """Base that re-imports on every setUp to dodge the stale-class
    identity trap from ``test_library_contracts`` module purge."""

    def setUp(self) -> None:
        (
            self.ConstraintSet,
            self.InterpolationRecord,
            self._make_tie_tag_allocator,
            self._pick_retained_nodes,
            self._quad4_split_pick,
            self.emit_tie_elements,
        ) = _fresh_imports()

    def _mk(self, **kwargs):
        return _make_tie_rec_with(self.InterpolationRecord, **kwargs)


# =====================================================================
# Quad4 split: deterministic triangle choice per quadrant
# =====================================================================

class TestQuad4Split(_TieTestBase):
    """Verify the (0,2) diagonal split picks the right triangle.

    Quad4 nodes in isoparametric frame::

        3 (-1, 1) ─────── 2 (1, 1)
           │                 │
           │     Tri B       │
           │    /            │
           │  /              │
           │/   Tri A        │
        0 (-1,-1) ─────── 1 (1,-1)

    * Tri A = nodes [0, 1, 2] when ``ξ >= η``
    * Tri B = nodes [0, 2, 3] when ``ξ <  η``
    """

    def _rec(self, xi, eta):
        return self._mk(
            slave_node=99,
            master_nodes=[10, 11, 12, 13],
            dofs=[1, 2, 3],
            xi=xi, eta=eta,
        )

    def test_lower_right_picks_triangle_a(self):
        result = self._quad4_split_pick(self._rec(0.5, -0.5))
        self.assertEqual(result, [10, 11, 12])

    def test_upper_left_picks_triangle_b(self):
        result = self._quad4_split_pick(self._rec(-0.5, 0.5))
        self.assertEqual(result, [10, 12, 13])

    def test_on_diagonal_picks_triangle_a(self):
        result = self._quad4_split_pick(self._rec(0.2, 0.2))
        self.assertEqual(result, [10, 11, 12])

    def test_missing_parametric_coords_returns_none(self):
        rec = self._rec(0.0, 0.0)
        rec.parametric_coords = None
        self.assertIsNone(self._quad4_split_pick(rec))


# =====================================================================
# Retained-node selection: all cardinalities
# =====================================================================

class TestPickRetainedNodes(_TieTestBase):

    def setUp(self) -> None:
        super().setUp()
        self.log_lines: list[str] = []

    def _log(self, msg):
        self.log_lines.append(msg)

    def test_tri3_pass_through(self):
        rec = self._mk(
            slave_node=1, master_nodes=[10, 11, 12],
            dofs=[1, 2, 3], xi=0.25, eta=0.25,
        )
        result = self._pick_retained_nodes(rec, self._log, 1)
        self.assertEqual(result, [10, 11, 12])
        self.assertEqual(self.log_lines, [])

    def test_quad4_split_triangle_a(self):
        rec = self._mk(
            slave_node=1, master_nodes=[10, 11, 12, 13],
            dofs=[1, 2, 3], xi=0.3, eta=-0.3,
        )
        result = self._pick_retained_nodes(rec, self._log, 1)
        self.assertEqual(result, [10, 11, 12])

    def test_quad4_split_triangle_b(self):
        rec = self._mk(
            slave_node=1, master_nodes=[10, 11, 12, 13],
            dofs=[1, 2, 3], xi=-0.3, eta=0.3,
        )
        result = self._pick_retained_nodes(rec, self._log, 1)
        self.assertEqual(result, [10, 12, 13])

    def test_tri6_downgraded_to_first_three_corners(self):
        rec = self._mk(
            slave_node=1,
            master_nodes=[10, 11, 12, 13, 14, 15],
            dofs=[1, 2, 3], xi=0.25, eta=0.25,
        )
        result = self._pick_retained_nodes(rec, self._log, 1)
        self.assertEqual(result, [10, 11, 12])
        self.assertTrue(
            any("Tri6" in ln and "downgraded" in ln for ln in self.log_lines),
            f"expected downgrade log, got {self.log_lines}",
        )

    def test_quad8_downgraded_then_split(self):
        rec = self._mk(
            slave_node=1,
            master_nodes=[10, 11, 12, 13, 14, 15, 16, 17],
            dofs=[1, 2, 3], xi=0.5, eta=-0.5,
        )
        result = self._pick_retained_nodes(rec, self._log, 1)
        self.assertEqual(result, [10, 11, 12])
        self.assertTrue(
            any("Quad8" in ln and "downgraded" in ln for ln in self.log_lines),
        )

    def test_unsupported_cardinality_returns_none(self):
        rec = self._mk(
            slave_node=1, master_nodes=[10, 11, 12, 13, 14],
            dofs=[1, 2, 3],
        )
        result = self._pick_retained_nodes(rec, self._log, 1)
        self.assertIsNone(result)
        self.assertTrue(
            any("UNSUPPORTED" in ln for ln in self.log_lines),
        )


# =====================================================================
# Tag allocator
# =====================================================================

class TestTagAllocator(_TieTestBase):

    def test_starts_at_1_000_000_for_empty_broker(self):
        ops = _FakeBroker()
        alloc = self._make_tie_tag_allocator(ops)
        self.assertEqual(alloc(), 1_000_000)
        self.assertEqual(alloc(), 1_000_001)

    def test_starts_at_max_plus_one_for_populated_broker(self):
        import pandas as pd
        df = pd.DataFrame({"x": [0, 0, 0]}, index=[101, 102, 103])
        df.index.name = "ops_id"
        ops = _FakeBroker(elements_df=df)
        alloc = self._make_tie_tag_allocator(ops)
        self.assertEqual(alloc(), 104)
        self.assertEqual(alloc(), 105)


# =====================================================================
# Full emit_tie_elements walk
# =====================================================================

class TestEmitTieElements(_TieTestBase):

    def _cs(self, recs):
        return self.ConstraintSet(recs)

    def test_empty_constraint_set_is_noop(self):
        ops = _FakeBroker(constraint_records=self._cs([]))
        counts = self.emit_tie_elements(ops)
        self.assertEqual(counts, {"tie": 0, "tie_skipped": 0})
        self.assertEqual(ops._tie_elements, [])

    def test_no_constraint_records_is_noop(self):
        ops = _FakeBroker(constraint_records=None)
        counts = self.emit_tie_elements(ops)
        self.assertEqual(counts, {"tie": 0, "tie_skipped": 0})

    def test_tri3_translational_tie(self):
        rec = self._mk(
            slave_node=99, master_nodes=[10, 11, 12],
            dofs=[1, 2, 3], xi=0.25, eta=0.25,
        )
        ops = _FakeBroker(constraint_records=self._cs([rec]))
        counts = self.emit_tie_elements(ops)
        self.assertEqual(counts["tie"], 1)
        self.assertEqual(counts["tie_skipped"], 0)
        self.assertEqual(len(ops._tie_elements), 1)

        entry = ops._tie_elements[0]
        self.assertEqual(entry["cNode"], 99)
        self.assertEqual(entry["rNodes"], [10, 11, 12])
        self.assertFalse(entry["use_rot"])
        self.assertIsNone(entry["penalty"])
        self.assertEqual(entry["source_kind"], "tie")
        self.assertEqual(entry["ele_tag"], 1_000_000)

    def test_tri3_with_rotations_sets_use_rot(self):
        rec = self._mk(
            slave_node=99, master_nodes=[10, 11, 12],
            dofs=[1, 2, 3, 4, 5, 6], xi=0.25, eta=0.25,
        )
        ops = _FakeBroker(constraint_records=self._cs([rec]))
        self.emit_tie_elements(ops)
        self.assertTrue(ops._tie_elements[0]["use_rot"])

    def test_tie_penalty_propagates_to_each_entry(self):
        rec1 = self._mk(
            slave_node=99, master_nodes=[10, 11, 12],
            dofs=[1, 2, 3], xi=0.1, eta=0.1,
        )
        rec2 = self._mk(
            slave_node=100, master_nodes=[10, 11, 12],
            dofs=[1, 2, 3], xi=0.2, eta=0.2,
        )
        ops = _FakeBroker(
            constraint_records=self._cs([rec1, rec2]),
            tie_penalty=1.5e12,
        )
        self.emit_tie_elements(ops)
        self.assertEqual(len(ops._tie_elements), 2)
        for entry in ops._tie_elements:
            self.assertEqual(entry["penalty"], 1.5e12)

    def test_quad4_master_generates_correct_triangle(self):
        rec = self._mk(
            slave_node=99, master_nodes=[10, 11, 12, 13],
            dofs=[1, 2, 3], xi=0.5, eta=-0.5,    # Tri A
        )
        ops = _FakeBroker(constraint_records=self._cs([rec]))
        self.emit_tie_elements(ops)
        self.assertEqual(ops._tie_elements[0]["rNodes"], [10, 11, 12])

    def test_unsupported_master_face_is_skipped(self):
        rec = self._mk(
            slave_node=99, master_nodes=[10, 11, 12, 13, 14],  # 5 nodes
            dofs=[1, 2, 3],
        )
        ops = _FakeBroker(constraint_records=self._cs([rec]))
        counts = self.emit_tie_elements(ops)
        self.assertEqual(counts["tie"], 0)
        self.assertEqual(counts["tie_skipped"], 1)
        self.assertEqual(ops._tie_elements, [])

    def test_distributing_interpolation_records_are_ignored(self):
        """Only ``tie`` and ``embedded`` kinds are emitted; others skipped."""
        tie = self._mk(
            slave_node=99, master_nodes=[10, 11, 12],
            dofs=[1, 2, 3], xi=0.1, eta=0.1,
        )
        distributing = self._mk(
            slave_node=200, master_nodes=[30, 31, 32],
            dofs=[1, 2, 3], xi=0.0, eta=0.0,
            kind="distributing",
        )
        ops = _FakeBroker(constraint_records=self._cs([tie, distributing]))
        counts = self.emit_tie_elements(ops)
        self.assertEqual(counts["tie"], 1)
        self.assertEqual(len(ops._tie_elements), 1)
        self.assertEqual(ops._tie_elements[0]["cNode"], 99)

    def test_embedded_interpolation_records_are_emitted(self):
        """Embedded records (tet4 host: 4 retained) are emitted as-is."""
        tet_embedded = self._mk(
            slave_node=100, master_nodes=[20, 21, 22, 23],
            dofs=[1, 2, 3], xi=0.0, eta=0.0,
            kind="embedded",
        )
        tri_embedded = self._mk(
            slave_node=101, master_nodes=[30, 31, 32],
            dofs=[1, 2, 3], xi=0.0, eta=0.0,
            kind="embedded",
        )
        ops = _FakeBroker(constraint_records=self._cs(
            [tet_embedded, tri_embedded]))
        counts = self.emit_tie_elements(ops)
        self.assertEqual(counts["tie"], 2)
        self.assertEqual(len(ops._tie_elements), 2)
        # 4-node tet passes all 4 retained nodes through (no quad split).
        self.assertEqual(ops._tie_elements[0]["cNode"], 100)
        self.assertEqual(ops._tie_elements[0]["rNodes"], [20, 21, 22, 23])
        self.assertEqual(ops._tie_elements[1]["rNodes"], [30, 31, 32])

    def test_sequential_tie_tags(self):
        recs = [
            self._mk(
                slave_node=100 + i, master_nodes=[10, 11, 12],
                dofs=[1, 2, 3], xi=0.1, eta=0.1,
            )
            for i in range(3)
        ]
        ops = _FakeBroker(constraint_records=self._cs(recs))
        self.emit_tie_elements(ops)
        tags = [e["ele_tag"] for e in ops._tie_elements]
        self.assertEqual(tags, [1_000_000, 1_000_001, 1_000_002])


if __name__ == "__main__":
    unittest.main()
