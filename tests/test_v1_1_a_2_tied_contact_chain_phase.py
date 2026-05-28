"""Compose v1.1-A.2 PR B — TiedContactDef chain-phase routing (ADR 0041).

Tests the new ``_route_tied_contact`` branch in
:mod:`apeGmsh._kernel.resolvers._chain_phase_router`.  Builds in-
memory :class:`FEMData` fixtures with two volume blocks meeting at a
surface interface, plus a slave node mesh that projects onto the
master surface; exercises the chain-phase router end-to-end; asserts
the resulting :class:`SurfaceCouplingRecord` lands on
``fem.elements.constraints``, and locks an H5 round-trip.

No live gmsh session, no openseespy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.defs.constraints import TiedContactDef
from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.records._constraints import (
    ConstraintKind,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.resolvers._chain_phase_router import (
    route_def_to_fem,
    try_chain_phase_route,
)
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------
# Fixture: two-block interface with explicit master + slave surface
# mesh + co-located slave nodes for clean projection.
# ---------------------------------------------------------------------


def _two_block_tied_interface_fem(
    *,
    master_label: str = "master_face",
    slave_label: str = "slave_face",
) -> FEMData:
    """Build a FEMData with:

    * Hex A (ids 1..8) on x in [0, 1].
    * Hex B (uses ids 9..16) on x in [1, 2] — but with **separate
      slave nodes** on its left face at x=1 (the interface).  The
      master face uses hex A's right face (nodes 2, 3, 7, 6 at x=1).
      The slave face uses fresh nodes 9, 10, 11, 12 at the SAME
      physical location, plus hex B's far-right corners 13..16 at x=2.
    * One quad4 master surface element using hex A's right face
      (2, 3, 7, 6).
    * One quad4 slave surface element using nodes 9, 10, 11, 12.
    * Two node-side PGs: ``master_label`` covers {2, 3, 6, 7},
      ``slave_label`` covers {9, 10, 11, 12}.

    Co-located projection: each slave node lies at (1, y, z) — exactly
    on the master quad — so the tied_contact resolver finds an exact
    projection and emits an :class:`InterpolationRecord` per slave
    node (4 total).
    """
    # Master block (hex A) corners, ids 1..8.
    coords_a = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # 1, 2
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # 3, 4
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],  # 5, 6
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],  # 7, 8
        ],
        dtype=np.float64,
    )
    # Slave-face nodes at x=1 (co-located with hex A's 2, 3, 6, 7).
    coords_slave_face = np.array(
        [
            [1.0, 0.0, 0.0],  # 9  ~ 2
            [1.0, 1.0, 0.0],  # 10 ~ 3
            [1.0, 0.0, 1.0],  # 11 ~ 6
            [1.0, 1.0, 1.0],  # 12 ~ 7
        ],
        dtype=np.float64,
    )
    # Hex B far-right corners at x=2, ids 13..16.
    coords_b_right = np.array(
        [
            [2.0, 0.0, 0.0],  # 13
            [2.0, 1.0, 0.0],  # 14
            [2.0, 0.0, 1.0],  # 15
            [2.0, 1.0, 1.0],  # 16
        ],
        dtype=np.float64,
    )
    coords = np.vstack([coords_a, coords_slave_face, coords_b_right])
    node_ids = np.arange(1, 17, dtype=np.int64)

    # Hex A and Hex B connectivity (gmsh hex8 order).
    hex_info = make_type_info(
        code=5, gmsh_name="Hexahedron 8", dim=3, order=1, npe=8, count=2,
    )
    hex_conn = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],         # hex A
            [9, 13, 14, 10, 11, 15, 16, 12],  # hex B
        ],
        dtype=np.int64,
    )
    hex_group = ElementGroup(
        element_type=hex_info,
        ids=np.array([100, 101], dtype=np.int64),
        connectivity=hex_conn,
    )

    # Quad4 master + slave surface elements.
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=2,
    )
    quad_conn = np.array(
        [
            [2, 3, 7, 6],     # master face (hex A right)
            [9, 10, 12, 11],  # slave face
        ],
        dtype=np.int64,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([500, 501], dtype=np.int64),
        connectivity=quad_conn,
    )

    node_pgs = {
        (0, 1): {
            "name": master_label,
            "node_ids": np.array([2, 3, 6, 7], dtype=np.int64),
            "node_coords": coords[[1, 2, 5, 6]],
        },
        (0, 2): {
            "name": slave_label,
            "node_ids": np.array([9, 10, 11, 12], dtype=np.int64),
            "node_coords": coords[8:12],
        },
    }

    elements = ElementComposite(
        groups={5: hex_group, 3: quad_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=4, bandwidth=1,
        types=[hex_info, quad_info],
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


def _hex_only_no_surface_fem() -> FEMData:
    """FEMData with only hex8 (dim=3) groups and node-side master /
    slave PGs.  Used to exercise ADR 0041 §"Decision 5" — no dim=2
    groups in broker → ValueError."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    node_ids = np.arange(1, 9, dtype=np.int64)
    hex_info = make_type_info(
        code=5, gmsh_name="Hexahedron 8", dim=3, order=1, npe=8, count=1,
    )
    hex_group = ElementGroup(
        element_type=hex_info,
        ids=np.array([100], dtype=np.int64),
        connectivity=np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64,
        ),
    )
    node_pgs = {
        (0, 1): {
            "name": "master_face",
            "node_ids": np.array([2, 3, 6, 7], dtype=np.int64),
            "node_coords": coords[[1, 2, 5, 6]],
        },
        (0, 2): {
            "name": "slave_face",
            "node_ids": np.array([1, 4, 5, 8], dtype=np.int64),
            "node_coords": coords[[0, 3, 4, 7]],
        },
    }
    elements = ElementComposite(
        groups={5: hex_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=1, bandwidth=1, types=[hex_info],
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


def _two_block_disjoint_interface_fem() -> FEMData:
    """Variant where master and slave node sets are completely
    disjoint from any quad4 row — the broker has dim=2 groups but
    neither target's PG covers a full face row.  Used to exercise the
    "empty interface returns fem unchanged" branch."""
    fem = _two_block_tied_interface_fem()
    # Replace both PGs with single-node selections that can't form a
    # quad face.
    new_node_pgs = {
        (0, 1): {
            "name": "master_face",
            "node_ids": np.array([1], dtype=np.int64),
            "node_coords": fem.nodes.coords[:1],
        },
        (0, 2): {
            "name": "slave_face",
            "node_ids": np.array([13], dtype=np.int64),
            "node_coords": fem.nodes.coords[12:13],
        },
    }
    new_nodes = NodeComposite(
        node_ids=fem.nodes.ids,
        node_coords=fem.nodes.coords,
        physical=PhysicalGroupSet(new_node_pgs),
        labels=LabelSet({}),
    )
    return FEMData(
        nodes=new_nodes,
        elements=fem.elements,
        info=fem.info,
        composed_from=ComposeSet(()),
    )


def _save(fem: FEMData, tmp_path: Path, name: str = "h.h5") -> Path:
    path = tmp_path / name
    fem.to_h5(str(path))
    return path


# ---------------------------------------------------------------------
# Happy path — tied_contact emits a SurfaceCouplingRecord
# ---------------------------------------------------------------------


class TestTiedContactChainPhase:
    def test_route_def_to_fem_produces_surface_coupling_record(
        self,
    ) -> None:
        """Two-block interface with co-located slave nodes →
        SurfaceCouplingRecord with one InterpolationRecord per slave."""
        fem = _two_block_tied_interface_fem()
        defn = TiedContactDef(
            master_label="master_face",
            slave_label="slave_face",
            tolerance=1e-6,
            name="tie1",
        )

        new_fem = route_def_to_fem(fem, defn)
        assert new_fem is not None
        assert new_fem is not fem

        recs = list(new_fem.elements.constraints)
        assert len(recs) == 1
        rec = recs[0]
        assert isinstance(rec, SurfaceCouplingRecord)
        assert rec.kind == ConstraintKind.TIED_CONTACT
        assert rec.name == "tie1"
        # All 4 slave nodes projected onto the master quad.
        assert len(rec.slave_records) == 4
        slave_ids = {ir.slave_node for ir in rec.slave_records}
        assert slave_ids == {9, 10, 11, 12}
        # Master side reports the 4 master nodes.
        assert sorted(rec.master_nodes) == [2, 3, 6, 7]
        assert sorted(rec.slave_nodes) == [9, 10, 11, 12]

    def test_chain_phase_session_path(self, tmp_path: Path) -> None:
        """End-to-end via ``g.constraints.tied_contact(...)`` after
        ``apeGmsh.from_h5(...)``."""
        path = _save(_two_block_tied_interface_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        fem_before = g._fem
        assert len(list(fem_before.elements.constraints)) == 0

        g.constraints.tied_contact(
            master_label="master_face",
            slave_label="slave_face",
            tolerance=1e-6,
        )

        fem_after = g._fem
        assert fem_after is not fem_before
        recs = list(fem_after.elements.constraints)
        assert len(recs) == 1
        assert isinstance(recs[0], SurfaceCouplingRecord)
        assert recs[0].kind == ConstraintKind.TIED_CONTACT
        assert len(recs[0].slave_records) == 4

    def test_unknown_master_label_falls_back_silently(self) -> None:
        """An unknown ``master_label`` raises KeyError inside the
        router; :func:`try_chain_phase_route` swallows it and returns
        False (matches the v1.1-A.2 KeyError handling for backward-
        compat — labels may be defined later)."""
        fem = _two_block_tied_interface_fem()
        defn = TiedContactDef(
            master_label="not_a_label",
            slave_label="slave_face",
            tolerance=1e-6,
        )

        # Direct route_def_to_fem call propagates KeyError.
        with pytest.raises(KeyError):
            route_def_to_fem(fem, defn)

        # The composite-level entry point swallows the KeyError.
        class Stub:
            _fem = fem

        s = Stub()
        assert try_chain_phase_route(s, defn) is False
        assert s._fem is fem

    def test_dim3_only_broker_raises_value_error(self) -> None:
        """A broker carrying only hex (dim=3) groups + valid node-side
        master / slave PGs raises ValueError per ADR 0041
        §"Decision 5" — chain phase does not synthesize faces from
        volumes."""
        fem = _hex_only_no_surface_fem()
        defn = TiedContactDef(
            master_label="master_face",
            slave_label="slave_face",
            tolerance=1e-6,
        )
        with pytest.raises(ValueError, match="no dim=2 ElementGroups"):
            route_def_to_fem(fem, defn)

    def test_empty_interface_returns_fem_unchanged(self) -> None:
        """When both targets resolve to node sets but neither covers
        any full face row, the router returns the broker unchanged
        (no records appended)."""
        fem = _two_block_disjoint_interface_fem()
        defn = TiedContactDef(
            master_label="master_face",
            slave_label="slave_face",
            tolerance=1e-6,
        )
        new_fem = route_def_to_fem(fem, defn)
        # Empty master + empty slave → fem unchanged.
        assert new_fem is fem
        assert len(list(new_fem.elements.constraints)) == 0


# ---------------------------------------------------------------------
# H5 round-trip — tied-contact record persists
# ---------------------------------------------------------------------


class TestH5RoundTrip:
    def test_tied_contact_record_persists_through_h5(
        self, tmp_path: Path,
    ) -> None:
        """After chain-phase routing, save + reload preserves the
        ``SurfaceCouplingRecord`` on ``elements.constraints``."""
        path = _save(
            _two_block_tied_interface_fem(), tmp_path, "host.h5",
        )
        g = apeGmsh.from_h5(path)
        g.constraints.tied_contact(
            master_label="master_face",
            slave_label="slave_face",
            tolerance=1e-6,
            name="tie_h5",
        )

        out = tmp_path / "after.h5"
        g._fem.to_h5(str(out))

        g2 = apeGmsh.from_h5(out)
        recs = list(g2._fem.elements.constraints)
        assert len(recs) == 1
        rec = recs[0]
        assert isinstance(rec, SurfaceCouplingRecord)
        assert rec.kind == ConstraintKind.TIED_CONTACT
        assert rec.name == "tie_h5"
        assert len(rec.slave_records) == 4


# ---------------------------------------------------------------------
# Build-phase behaviour unchanged (regression)
# ---------------------------------------------------------------------


class TestBuildPhaseUnchanged:
    def test_route_returns_false_when_fem_is_none(self) -> None:
        """When ``session._fem is None``, ``try_chain_phase_route`` is
        a no-op (build phase keeps using the bump-counter pattern)."""

        class Stub:
            _fem = None

        defn = TiedContactDef(
            master_label="a", slave_label="b", tolerance=1e-6,
        )
        assert try_chain_phase_route(Stub(), defn) is False
