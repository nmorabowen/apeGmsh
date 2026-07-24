"""PG-based constraints — chain-phase routing for the remaining
face/node verbs: ``tie`` / ``kinematic_coupling`` / ``rigid_body`` /
``distributing_coupling``.

Before this slice the chain-phase router covered EqualDOF / RigidLink /
RigidDiaphragm / Embedded / TiedContact only; the four verbs here fell
through to the bump-counter pattern (def stored but **not** applied — a
silent no-op) in a ``from_h5`` / ``compose`` session.  These tests lock
that they now resolve against the FEMData broker so physical-group
models (no Parts) can constrain post-extraction.

Runs entirely off the FEMData broker — no live gmsh, no openseespy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.defs.constraints import (
    KinematicCouplingDef,
)
from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
)
from apeGmsh._kernel.resolvers._chain_phase_router import route_def_to_fem
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------
# Fixture: one master quad face + two slave nodes on it + a reference
# point above it.  Node PGs: ``m_face`` (quad corners), ``s_nodes``
# (interior slaves), ``ref`` (the reference point).
# ---------------------------------------------------------------------


def _pg_interface_fem(*, empty_ref: bool = False) -> FEMData:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],    # 1  quad corner
            [1.0, 0.0, 0.0],    # 2  quad corner
            [1.0, 1.0, 0.0],    # 3  quad corner
            [0.0, 1.0, 0.0],    # 4  quad corner
            [0.25, 0.25, 0.0],  # 5  slave (on the quad)
            [0.75, 0.75, 0.0],  # 6  slave (on the quad)
            [0.5, 0.5, 1.0],    # 10 reference point
        ],
        dtype=np.float64,
    )
    node_ids = np.array([1, 2, 3, 4, 5, 6, 10], dtype=np.int64)

    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=1,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([500], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
    )

    ref_ids = (np.array([], dtype=np.int64) if empty_ref
               else np.array([10], dtype=np.int64))
    ref_coords = (coords[6:6] if empty_ref else coords[6:7])
    node_pgs = {
        (0, 1): {
            "name": "m_face",
            "node_ids": np.array([1, 2, 3, 4], dtype=np.int64),
            "node_coords": coords[:4],
        },
        (0, 2): {
            "name": "s_nodes",
            "node_ids": np.array([5, 6], dtype=np.int64),
            "node_coords": coords[4:6],
        },
        (0, 3): {
            "name": "ref",
            "node_ids": ref_ids,
            "node_coords": ref_coords,
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={3: quad_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=1, bandwidth=1, types=[quad_info],
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


def _colocated_pg_fem() -> FEMData:
    """Two co-located node PGs (``pm`` / ``ps``) for penalty / equal_dof
    style co-location matching."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],   # 1, 2  master
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],   # 3, 4  slave (co-located)
        ],
        dtype=np.float64,
    )
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([100], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    node_pgs = {
        (0, 1): {"name": "pm",
                 "node_ids": np.array([1, 2], dtype=np.int64),
                 "node_coords": coords[:2]},
        (0, 2): {"name": "ps",
                 "node_ids": np.array([3, 4], dtype=np.int64),
                 "node_coords": coords[2:4]},
    }
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet(node_pgs), labels=LabelSet({}))
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}))
    info = MeshInfo(
        n_nodes=4, n_elems=1, bandwidth=1, types=[line_info])
    return FEMData(
        nodes=nodes, elements=elements, info=info,
        composed_from=ComposeSet(()))


def _from_h5(fem: FEMData, tmp_path: Path):
    path = tmp_path / "h.h5"
    fem.to_h5(str(path))
    return apeGmsh.from_h5(path)


# ---------------------------------------------------------------------


class TestTieChainPhase:
    def test_tie_routes_interpolation_records(self, tmp_path: Path) -> None:
        g = _from_h5(_pg_interface_fem(), tmp_path)
        before = g._fem
        assert len(list(before.elements.constraints.interpolations())) == 0

        g.constraints.tie("m_face", "s_nodes", dofs=[1, 2, 3],
                          tolerance=0.1)

        after = g._fem
        assert after is not before
        recs = list(after.elements.constraints.interpolations())
        assert len(recs) == 2  # one per slave node projected onto the quad
        assert all(isinstance(r, InterpolationRecord) for r in recs)


class TestKinematicCouplingChainPhase:
    def test_routes_node_group_record(self, tmp_path: Path) -> None:
        g = _from_h5(_pg_interface_fem(), tmp_path)
        assert len(list(g._fem.nodes.constraints)) == 0

        g.constraints.kinematic_coupling(
            "ref", "m_face", master_point=(0.5, 0.5, 1.0), dofs=[1, 2, 3])

        recs = list(g._fem.nodes.constraints)
        assert len(recs) == 1
        rec = recs[0]
        assert isinstance(rec, NodeGroupRecord)
        assert rec.master_node == 10
        assert set(rec.slave_nodes) == {1, 2, 3, 4}


class TestRigidBodyChainPhase:
    def test_routes_node_group_record(self, tmp_path: Path) -> None:
        g = _from_h5(_pg_interface_fem(), tmp_path)
        g.constraints.rigid_body(
            "ref", "m_face", master_point=(0.5, 0.5, 1.0))

        recs = list(g._fem.nodes.constraints)
        assert len(recs) == 1
        assert isinstance(recs[0], NodeGroupRecord)
        assert recs[0].master_node == 10
        assert set(recs[0].slave_nodes) == {1, 2, 3, 4}


class TestDistributingCouplingChainPhase:
    def test_uniform_routes_interpolation(self, tmp_path: Path) -> None:
        g = _from_h5(_pg_interface_fem(), tmp_path)
        g.constraints.distributing_coupling(
            "ref", "m_face", master_point=(0.5, 0.5, 1.0),
            weighting="uniform")

        recs = list(g._fem.elements.constraints.interpolations())
        assert len(recs) == 1
        assert isinstance(recs[0], InterpolationRecord)
        assert recs[0].weights is None  # uniform ⇒ no per-node weights

    def test_area_uses_boundary_faces(self, tmp_path: Path) -> None:
        g = _from_h5(_pg_interface_fem(), tmp_path)
        g.constraints.distributing_coupling(
            "ref", "m_face", master_point=(0.5, 0.5, 1.0),
            weighting="area")

        recs = list(g._fem.elements.constraints.interpolations())
        assert len(recs) == 1
        # area weighting pulled the quad face and computed tributary areas
        assert recs[0].weights is not None


class TestPenaltyChainPhase:
    def test_penalty_routes_node_pair_records(self, tmp_path: Path) -> None:
        g = _from_h5(_colocated_pg_fem(), tmp_path)
        assert len(list(g._fem.nodes.constraints)) == 0

        g.constraints.penalty("pm", "ps", dofs=[1, 2, 3],
                              stiffness=1e12, tolerance=1e-6)

        recs = list(g._fem.nodes.constraints)
        assert len(recs) == 2  # (1,3) and (2,4) co-located pairs
        pairs = {(r.master_node, r.slave_node) for r in recs}
        assert pairs == {(1, 3), (2, 4)}


class TestEmptyTargetFailsLoud:
    """F5 regression: an empty node set must NOT silently bind to the
    global closest node — the chain branches guard fail-loud."""

    def test_empty_master_raises_not_silent(self) -> None:
        fem = _pg_interface_fem(empty_ref=True)
        defn = KinematicCouplingDef(
            master_label="ref", slave_label="m_face",
            master_point=(0.5, 0.5, 1.0), dofs=[1, 2, 3])
        with pytest.raises(ValueError, match="zero nodes"):
            route_def_to_fem(fem, defn)
