"""Compose v1.1-A — chain-phase routing for EqualDOF / RigidLink /
RigidDiaphragm.

Covers the v1.1-A slice of the chain-phase router (PR follow-up to
#366): the three node-only interface-bridging constraints now route
through ``with_constraint(record)`` transforms on the immutable
FEMData chain — same path that BCDef / PointMassDef / PointLoadDef
already take.

Out of scope for v1.1-A: ``EmbeddedDef`` and ``TiedContactDef``.
Those need element-connectivity / face-connectivity queries that
:class:`FEMDataSource` does not yet expose; they continue to fall
back to the bump-counter pattern.  v1.1-A.2 will wire them up.

The tests run entirely off the FEMData broker — no live gmsh session
is required, and openseespy is not imported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.defs.constraints import (
    EmbeddedDef,
    EqualDOFDef,
    RigidDiaphragmDef,
    RigidLinkDef,
    TiedContactDef,
)
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.records._constraints import (
    NodeGroupRecord,
    NodePairRecord,
)
from apeGmsh._kernel.resolvers._chain_phase_router import (
    route_def_to_fem,
    try_chain_phase_route,
)
from apeGmsh._kernel.resolvers._source import FEMDataSource
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_interface_fem(
    *,
    master_coords: np.ndarray,
    slave_coords: np.ndarray,
    master_label: str = "master_set",
    slave_label: str = "slave_set",
) -> FEMData:
    """Build a FEMData with two physical groups (master + slave node sets).

    Nodes are numbered 1..N with master ids first, slave ids after.
    A single Line element is added so MeshInfo is non-trivial — the
    constraint resolvers never consult elements for the three v1.1-A
    paths but FEMData expects at least one element group.
    """
    n_m = master_coords.shape[0]
    n_s = slave_coords.shape[0]
    master_ids = np.arange(1, n_m + 1, dtype=np.int64)
    slave_ids = np.arange(n_m + 1, n_m + 1 + n_s, dtype=np.int64)
    node_ids = np.concatenate([master_ids, slave_ids])
    coords = np.concatenate(
        [
            np.asarray(master_coords, dtype=np.float64),
            np.asarray(slave_coords, dtype=np.float64),
        ],
    )

    # One dummy line element so MeshInfo has something to point at.
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    conn = np.array([[int(node_ids[0]), int(node_ids[1])]], dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([100], dtype=np.int64),
        connectivity=conn,
    )

    node_pgs = {
        (0, 1): {
            "name": master_label,
            "node_ids": master_ids,
            "node_coords": np.asarray(master_coords, dtype=np.float64),
        },
        (0, 2): {
            "name": slave_label,
            "node_ids": slave_ids,
            "node_coords": np.asarray(slave_coords, dtype=np.float64),
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=1, bandwidth=1,
        types=[line_info],
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


def _colocated_fem() -> FEMData:
    """Two pairs of co-located nodes — suitable for equal_dof tests."""
    return _make_interface_fem(
        master_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        slave_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )


def _master_plus_slab_fem() -> FEMData:
    """One master point + three slab nodes on z=5 plane.

    Used for rigid_link (master_point=(0,0,5)) and rigid_diaphragm
    (plane_normal=(0,0,1)).
    """
    return _make_interface_fem(
        master_coords=np.array([[0.0, 0.0, 5.0]]),
        slave_coords=np.array(
            [[1.0, 0.0, 5.0], [1.0, 1.0, 5.0], [0.0, 1.0, 5.0]],
        ),
        master_label="master_pt",
        slave_label="slab",
    )


def _save(fem: FEMData, tmp_path: Path, name: str = "h.h5") -> Path:
    path = tmp_path / name
    fem.to_h5(str(path))
    return path


# ---------------------------------------------------------------------------
# Pass 1.A — EqualDOFDef chain-phase routing
# ---------------------------------------------------------------------------


class TestEqualDOFChainPhase:
    def test_chain_phase_routes_to_node_pair_records(
        self, tmp_path: Path,
    ) -> None:
        """equal_dof in chain phase appends NodePairRecords to _fem."""
        path = _save(_colocated_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        fem_before = g._fem
        assert len(list(fem_before.nodes.constraints)) == 0

        g.constraints.equal_dof(
            "master_set", "slave_set", dofs=[1, 2, 3], tolerance=1e-6,
        )

        fem_after = g._fem
        # FEMData identity changes — transform routed.
        assert fem_after is not fem_before
        recs = list(fem_after.nodes.constraints)
        assert len(recs) == 2  # one pair per co-located master/slave
        for rec in recs:
            assert isinstance(rec, NodePairRecord)
            assert rec.kind == "equal_dof"
            assert rec.dofs == [1, 2, 3]
        master_slave_pairs = {(r.master_node, r.slave_node) for r in recs}
        # Master ids are 1,2; slave ids are 3,4 (co-located).
        assert master_slave_pairs == {(1, 3), (2, 4)}

    def test_route_def_to_fem_returns_new_fem(self) -> None:
        """route_def_to_fem returns a non-None new FEMData."""
        fem = _colocated_fem()
        defn = EqualDOFDef(
            master_label="master_set", slave_label="slave_set",
            dofs=[1, 2, 3], tolerance=1e-6,
        )
        new_fem = route_def_to_fem(fem, defn)
        assert new_fem is not None
        assert new_fem is not fem
        assert len(list(new_fem.nodes.constraints)) == 2

    def test_build_phase_unchanged(self) -> None:
        """When _fem is None (build phase), try_chain_phase_route is a no-op."""

        class Stub:
            _fem = None

        defn = EqualDOFDef(
            master_label="a", slave_label="b", dofs=[1, 2, 3],
        )
        assert try_chain_phase_route(Stub(), defn) is False

    def test_default_dofs_resolve_to_all_six(
        self, tmp_path: Path,
    ) -> None:
        """dofs=None → resolver defaults [1..6]."""
        path = _save(_colocated_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        g.constraints.equal_dof("master_set", "slave_set")
        recs = list(g._fem.nodes.constraints)
        assert all(r.dofs == [1, 2, 3, 4, 5, 6] for r in recs)


# ---------------------------------------------------------------------------
# Pass 1.B — RigidLinkDef chain-phase routing
# ---------------------------------------------------------------------------


class TestRigidLinkChainPhase:
    def test_chain_phase_routes_to_node_pair_records(
        self, tmp_path: Path,
    ) -> None:
        path = _save(_master_plus_slab_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        fem_before = g._fem

        g.constraints.rigid_link(
            "master_pt", "slab", link_type="beam",
            master_point=(0.0, 0.0, 5.0), name="RL1",
        )

        fem_after = g._fem
        assert fem_after is not fem_before
        recs = list(fem_after.nodes.constraints)
        # 3 slave nodes (2,3,4) -> 3 records.
        assert len(recs) == 3
        for rec in recs:
            assert isinstance(rec, NodePairRecord)
            assert rec.kind == "rigid_beam"
            assert rec.master_node == 1
            assert rec.dofs == [1, 2, 3, 4, 5, 6]
            assert rec.offset is not None
            assert rec.name == "RL1"
        slave_nodes = {r.slave_node for r in recs}
        assert slave_nodes == {2, 3, 4}

    def test_rod_link_couples_translations_only(
        self, tmp_path: Path,
    ) -> None:
        path = _save(_master_plus_slab_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        g.constraints.rigid_link(
            "master_pt", "slab", link_type="rod",
            master_point=(0.0, 0.0, 5.0),
        )
        recs = list(g._fem.nodes.constraints)
        assert all(r.kind == "rigid_rod" for r in recs)
        assert all(r.dofs == [1, 2, 3] for r in recs)

    def test_route_def_to_fem_returns_new_fem(self) -> None:
        fem = _master_plus_slab_fem()
        defn = RigidLinkDef(
            master_label="master_pt", slave_label="slab",
            link_type="beam", master_point=(0.0, 0.0, 5.0),
        )
        new_fem = route_def_to_fem(fem, defn)
        assert new_fem is not None
        assert len(list(new_fem.nodes.constraints)) == 3


# ---------------------------------------------------------------------------
# Pass 1.C — RigidDiaphragmDef chain-phase routing
# ---------------------------------------------------------------------------


class TestRigidDiaphragmChainPhase:
    def test_chain_phase_routes_to_node_group_record(
        self, tmp_path: Path,
    ) -> None:
        path = _save(_master_plus_slab_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        fem_before = g._fem

        g.constraints.rigid_diaphragm(
            "master_pt", "slab",
            master_point=(0.0, 0.0, 5.0),
            plane_normal=(0.0, 0.0, 1.0),
            constrained_dofs=[1, 2, 6],
            plane_tolerance=0.5,
            name="D1",
        )

        fem_after = g._fem
        assert fem_after is not fem_before
        recs = list(fem_after.nodes.constraints)
        assert len(recs) == 1
        rec = recs[0]
        assert isinstance(rec, NodeGroupRecord)
        assert rec.kind == "rigid_diaphragm"
        assert rec.name == "D1"
        assert rec.master_node == 1
        assert sorted(rec.slave_nodes) == [2, 3, 4]
        assert rec.dofs == [1, 2, 6]
        # Plane normal stored as unit vector.
        assert rec.plane_normal is not None
        np.testing.assert_allclose(rec.plane_normal, [0.0, 0.0, 1.0])

    def test_diaphragm_with_no_nodes_in_plane_does_not_append(
        self, tmp_path: Path,
    ) -> None:
        """No slabs in plane → empty NodeGroupRecord — skipped to avoid phantom."""
        # Place master at z=5, slabs at z=5 — but require plane at z=0.
        path = _save(_master_plus_slab_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        fem_before = g._fem

        g.constraints.rigid_diaphragm(
            "master_pt", "slab",
            master_point=(0.0, 0.0, 0.0),
            plane_normal=(0.0, 0.0, 1.0),
            constrained_dofs=[1, 2, 6],
            plane_tolerance=0.01,
        )

        # No matching nodes → no record appended.
        assert g._fem is fem_before
        assert len(list(g._fem.nodes.constraints)) == 0

    def test_route_def_to_fem_returns_new_fem(self) -> None:
        fem = _master_plus_slab_fem()
        defn = RigidDiaphragmDef(
            master_label="master_pt", slave_label="slab",
            master_point=(0.0, 0.0, 5.0),
            plane_normal=(0.0, 0.0, 1.0),
            constrained_dofs=[1, 2, 6],
            plane_tolerance=0.5,
        )
        new_fem = route_def_to_fem(fem, defn)
        assert new_fem is not None
        recs = list(new_fem.nodes.constraints)
        assert len(recs) == 1
        assert isinstance(recs[0], NodeGroupRecord)


# ---------------------------------------------------------------------------
# Pass 2 — TiedContact still on bump-counter; Embedded joined in v1.1-A.2
# ---------------------------------------------------------------------------


class TestDeferredDefsFallBackCleanly:
    """v1.1-A handles 3 of 5 defs; v1.1-A.2 added EmbeddedDef and
    TiedContactDef.  Neither def is deferred anymore — both now route
    through the chain-phase router.

    What this class locks now is the *KeyError-swallow* contract that
    survived v1.1-A.2: when an Embedded host label or a TiedContact
    master / slave label has no element-side record in the chain-head
    broker (e.g. the node-only ``_colocated_fem`` fixture), the
    router's ``KeyError`` is caught by :func:`try_chain_phase_route`
    so the def lands on ``constraint_defs`` without applying a
    record — mirrors the v1.1-A behaviour for a missing label that
    might be created later in the session.

    The Embedded-specific chain-phase tests live in
    ``tests/test_v1_1_a_2_embedded_chain_phase.py``; the tied-contact
    chain-phase tests live in
    ``tests/test_v1_1_a_2_tied_contact_chain_phase.py``.
    """

    def test_embedded_against_node_only_fixture_raises_key_error(self) -> None:
        """`route_def_to_fem` for EmbeddedDef now resolves the host
        label element-side via ``FEMDataSource.host_subelements_for``.
        Against a fixture that has only node-side PGs (``_colocated_fem``
        is node-only), the host_label has no element-side record and
        the function raises ``KeyError`` — same propagation path
        ``try_chain_phase_route`` already catches."""
        fem = _colocated_fem()
        defn = EmbeddedDef(
            master_label="master_set", slave_label="slave_set",
            tolerance=1.0,
        )
        with pytest.raises(KeyError):
            route_def_to_fem(fem, defn)

    def test_tied_contact_against_colocated_fixture_raises_key_error(
        self,
    ) -> None:
        """`route_def_to_fem` for TiedContactDef now resolves master /
        slave labels via ``FEMDataSource.boundary_faces_for``, which
        calls :meth:`nodes_for` on the way in.  The colocated fixture's
        node-side PGs resolve cleanly, but the broker carries no dim=2
        ElementGroups — so :meth:`boundary_faces_for` raises
        ``ValueError`` per ADR 0041 §"Decision 5" (no volume→face
        synthesis in chain phase)."""
        fem = _colocated_fem()
        defn = TiedContactDef(
            master_label="master_set", slave_label="slave_set",
            tolerance=1.0,
        )
        with pytest.raises(ValueError, match="no dim=2 ElementGroups"):
            route_def_to_fem(fem, defn)

    def test_embedded_still_callable_in_chain_phase(
        self, tmp_path: Path,
    ) -> None:
        """g.constraints.embedded(...) remains callable post-compose
        per ADR 0038 line 45.  With the node-only-PG fixture the host
        target resolves to no element-side records — the router's
        ``KeyError`` is swallowed by ``try_chain_phase_route`` (same
        as a missing label in v1.1-A); the def still lands on the
        composite's ``constraint_defs``."""
        path = _save(_colocated_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        defn = g.constraints.embedded(
            host_label="master_set", embedded_label="slave_set",
            tolerance=1.0,
        )
        # Def stored on the composite's def list.
        assert defn in g.constraints.constraint_defs
        # No constraint record was applied — host_subelements_for
        # raised KeyError, the router swallowed it, fem unchanged.
        assert len(list(g._fem.nodes.constraints)) == 0
        assert len(list(g._fem.elements.constraints)) == 0

    def test_tied_contact_value_error_surfaces_in_chain_phase(
        self, tmp_path: Path,
    ) -> None:
        """``g.constraints.tied_contact(...)`` against a node-only chain
        head surfaces the router's ``ValueError`` per ADR 0041
        §"Decision 5" — the broker has no dim=2 ElementGroups, so
        :meth:`FEMDataSource.boundary_faces_for` raises with the
        documented remedy.  :func:`try_chain_phase_route` only swallows
        ``KeyError`` / ``TypeError``, so this ValueError propagates out
        of ``_add_def``.  The def is **not** stored (the raise happens
        between the ``constraint_defs.append`` and... wait — no, the
        append runs first, then ``try_chain_phase_route`` runs, so the
        def IS on the list but the broker is unchanged.  Either way
        ``_fem`` is not corrupted."""
        path = _save(_colocated_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        with pytest.raises(ValueError, match="no dim=2 ElementGroups"):
            g.constraints.tied_contact(
                master_label="master_set", slave_label="slave_set",
                tolerance=1.0,
            )
        # Broker untouched.
        assert len(list(g._fem.nodes.constraints)) == 0
        assert len(list(g._fem.elements.constraints)) == 0


# ---------------------------------------------------------------------------
# H5 round-trip
# ---------------------------------------------------------------------------


class TestH5RoundTrip:
    """After chain-phase routing, save + reload preserves the records."""

    def test_equal_dof_persists_through_h5(self, tmp_path: Path) -> None:
        path = _save(_colocated_fem(), tmp_path, "host.h5")
        g = apeGmsh.from_h5(path)
        g.constraints.equal_dof(
            "master_set", "slave_set", dofs=[1, 2, 3], tolerance=1e-6,
        )

        out = tmp_path / "after.h5"
        g._fem.to_h5(str(out))

        # Reload as a fresh chain-phase session.
        g2 = apeGmsh.from_h5(out)
        recs = list(g2._fem.nodes.constraints)
        assert len(recs) == 2
        kinds = {r.kind for r in recs}
        assert kinds == {"equal_dof"}
        pairs = {(r.master_node, r.slave_node) for r in recs}
        assert pairs == {(1, 3), (2, 4)}

    def test_rigid_link_persists_through_h5(self, tmp_path: Path) -> None:
        path = _save(_master_plus_slab_fem(), tmp_path, "host.h5")
        g = apeGmsh.from_h5(path)
        g.constraints.rigid_link(
            "master_pt", "slab",
            link_type="beam", master_point=(0.0, 0.0, 5.0),
            name="RL_h5",
        )

        out = tmp_path / "after.h5"
        g._fem.to_h5(str(out))

        g2 = apeGmsh.from_h5(out)
        recs = list(g2._fem.nodes.constraints)
        assert len(recs) == 3
        assert all(r.kind == "rigid_beam" for r in recs)
        assert all(r.name == "RL_h5" for r in recs)

    def test_rigid_diaphragm_persists_through_h5(
        self, tmp_path: Path,
    ) -> None:
        path = _save(_master_plus_slab_fem(), tmp_path, "host.h5")
        g = apeGmsh.from_h5(path)
        g.constraints.rigid_diaphragm(
            "master_pt", "slab",
            master_point=(0.0, 0.0, 5.0),
            plane_normal=(0.0, 0.0, 1.0),
            constrained_dofs=[1, 2, 6],
            plane_tolerance=0.5,
            name="D_h5",
        )

        out = tmp_path / "after.h5"
        g._fem.to_h5(str(out))

        g2 = apeGmsh.from_h5(out)
        recs = list(g2._fem.nodes.constraints)
        assert len(recs) == 1
        rec = recs[0]
        assert rec.kind == "rigid_diaphragm"
        assert rec.name == "D_h5"
        assert rec.master_node == 1
        assert sorted(rec.slave_nodes) == [2, 3, 4]


# ---------------------------------------------------------------------------
# Validation gate (chain-phase relax)
# ---------------------------------------------------------------------------


class TestChainPhaseValidationGate:
    """In chain phase, ``_add_def`` validates labels against the
    FEMData broker (not g.parts._instances which is empty)."""

    def test_unknown_label_raises_key_error(self, tmp_path: Path) -> None:
        path = _save(_colocated_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        with pytest.raises(KeyError, match="resolves to"):
            g.constraints.equal_dof("does_not_exist", "slave_set")

    def test_known_pg_label_passes_gate(self, tmp_path: Path) -> None:
        path = _save(_colocated_fem(), tmp_path)
        g = apeGmsh.from_h5(path)
        # Both labels are physical groups in the broker — gate passes.
        g.constraints.equal_dof("master_set", "slave_set")

    def test_fem_data_source_has_target_used(
        self, tmp_path: Path,
    ) -> None:
        """The chain-phase gate uses FEMDataSource.has_target."""
        fem = _colocated_fem()
        src = FEMDataSource(fem)
        assert src.has_target("master_set")
        assert src.has_target("slave_set")
        assert not src.has_target("nope")
