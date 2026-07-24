"""Phase 3B.2d tests — chain-phase freeze, resolver sources, verifier,
rank model.

Covers ADR 0038's closing slice of Phase 3B:

* ``ChainPhaseError`` raised on every gated chokepoint when the
  session has produced its first :class:`FEMData`.
* :class:`FEMDataSource` / :class:`GmshSource` Protocol implementations
  resolve targets out of the broker or live gmsh respectively.
* Chain-phase shim routing puts ``g.constraints.bc`` / ``g.masses.point``
  / ``g.loads.point_force`` results onto ``_fem`` via the ``with_*``
  transforms — no get_fem_data() re-extraction needed.
* Phase 2.2 tag-collision verifier fires on real synthetic collision
  fixtures, raising the matching typed exception.
* Rank model — Layer 1 default, Layer 2 hint, Layer 3 METIS override.

These tests run entirely against the FEMData broker; no live gmsh
session is required for any of them.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.resolvers._chain_phase_router import (
    route_def_to_fem,
    try_chain_phase_route,
)
from apeGmsh._kernel.resolvers._source import (
    FEMDataSource,
    GmshSource,
    ResolverSource,
    make_source,
)
from apeGmsh.core._compose_errors import (
    ChainPhaseError,
    ComposeCapacityError,
    ComposeInvariantError,
    PartTagCollisionError,
)
from apeGmsh.core._tag_collision_verifier import (
    ConstraintReference,
    ImportedRecords,
    ReservationRecord,
    tag_collision_verify,
)
from apeGmsh.mesh._compose import (
    _rebuild_partitions_from_modules,
    _run_compose_verifier,
)
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_fem(
    *,
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
    node_pgs: "dict | None" = None,
    node_labels: "dict | None" = None,
    composed_from: "ComposeSet | tuple[ComposeRecord, ...]" = (),
    node_module_labels: "list[str] | None" = None,
    elem_module_labels: "list[str] | None" = None,
) -> FEMData:
    """Build a small FEMData for direct broker tests."""
    if node_ids is None:
        node_ids = np.array([1, 2, 3], dtype=np.int64)
    if elem_ids is None:
        elem_ids = np.array([10, 11], dtype=np.int64)

    n = node_ids.size
    coords = np.array(
        [[float(i), 0.0, 0.0] for i in range(n)], dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    conn = np.array(
        [
            [int(node_ids[i % n]), int(node_ids[(i + 1) % n])]
            for i in range(elem_ids.size)
        ],
        dtype=np.int64,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )

    n_ml = (
        np.array(node_module_labels, dtype=object)
        if node_module_labels is not None
        else None
    )
    e_ml = (
        {1: np.array(elem_module_labels, dtype=object)}
        if elem_module_labels is not None
        else None
    )

    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(node_pgs or {}),
        labels=LabelSet(node_labels or {}),
        module_label=n_ml,
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
        module_label=e_ml,
    )
    info = MeshInfo(
        n_nodes=n, n_elems=elem_ids.size, bandwidth=1,
        types=[line_info],
    )
    return FEMData(
        nodes=nodes, elements=elements, info=info,
        composed_from=composed_from,
    )


def _save_fem(fem: FEMData, path: Path) -> Path:
    fem.to_h5(str(path))
    return path


# =============================================================================
# Step 1 — Chain-phase geometry freeze
# =============================================================================


class TestChainPhaseFreeze:
    """Every gated chokepoint raises ChainPhaseError post-extraction."""

    @pytest.fixture
    def chain_session(self, tmp_path: Path) -> apeGmsh:
        """A chain-phase session — _fem populated via from_h5."""
        host = _save_fem(_make_fem(), tmp_path / "host.h5")
        return apeGmsh.from_h5(host)

    def test_model_register_blocked(self, chain_session: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.model._register(0, 999, None, "test_kind")

    def test_geometry_add_routed_through_register(
        self, chain_session: apeGmsh,
    ) -> None:
        """Adding a point goes through Model._register → ChainPhaseError.

        Tested at the ``_register`` chokepoint directly (rather than
        through ``add_point``) so the test does not depend on gmsh
        state being initialised in the broader test run — the contract
        we lock here is that every geometry primitive flows through
        ``_register`` and that ``_register`` raises ChainPhaseError
        post-extraction.
        """
        with pytest.raises(ChainPhaseError):
            chain_session.model._register(0, 1, "label", "point")

    def test_labels_add_blocked(self, chain_session: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.labels.add(0, [1], "test_label")

    def test_physical_add_blocked(self, chain_session: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.physical.add(0, [1], name="test_pg")

    def test_parts_register_instance_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        from apeGmsh.core._parts_registry import Instance

        inst = Instance(
            label="x", part_name="x", entities={3: [1]}, bbox=None,
        )
        with pytest.raises(ChainPhaseError):
            chain_session.parts._register_instance(inst)

    def test_mesh_generate_blocked(self, chain_session: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.generation.generate(3)

    def test_mesh_refine_blocked(self, chain_session: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.generation.refine()

    def test_mesh_set_order_blocked(self, chain_session: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.generation.set_order(2)

    def test_mesh_optimize_blocked(self, chain_session: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.generation.optimize()

    def test_mesh_set_algorithm_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.generation.set_algorithm(0, "delaunay")

    def test_mesh_editing_clear_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.editing.clear()

    def test_mesh_editing_remove_duplicate_nodes_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.editing.remove_duplicate_nodes()

    def test_mesh_editing_affine_transform_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        identity = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0]
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.editing.affine_transform(identity)

    def test_mesh_sizing_set_global_size_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.sizing.set_global_size(1.0)

    def test_mesh_sizing_set_size_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.sizing.set_size(1, 0.5)

    def test_mesh_structured_recombine_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.structured.recombine()

    def test_mesh_structured_set_transfinite_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        # ``_resolve`` is the central chokepoint; touched via any
        # structured method that takes a tag argument.
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.structured.set_transfinite_curve(1, 5)

    def test_mesh_partitioning_partition_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.partitioning.partition(4)

    def test_mesh_partitioning_unpartition_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.mesh.partitioning.unpartition()

    def test_model_io_heal_shapes_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.model.io.heal_shapes()

    def test_model_io_load_msh_blocked(
        self, chain_session: apeGmsh, tmp_path: Path,
    ) -> None:
        fake = tmp_path / "fake.msh"
        fake.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
        with pytest.raises(ChainPhaseError):
            chain_session.model.io.load_msh(fake)

    def test_model_io_load_geo_blocked(
        self, chain_session: apeGmsh, tmp_path: Path,
    ) -> None:
        fake = tmp_path / "fake.geo"
        fake.write_text("// empty geo\n")
        with pytest.raises(ChainPhaseError):
            chain_session.model.io.load_geo(fake)

    def test_model_boolean_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        with pytest.raises(ChainPhaseError):
            chain_session.model.boolean.fuse(1, 2)

    def test_model_transforms_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        """Transforms route through ``_Transforms._resolve_dt`` which
        guards before any gmsh call."""
        with pytest.raises(ChainPhaseError):
            chain_session.model.transforms._resolve_dt([1], 3)

    def test_interface_bridging_NOT_blocked(
        self, chain_session: apeGmsh,
    ) -> None:
        """ADR 0038 line 45 — interface-bridging primitives must remain
        callable post-compose.

        We only exercise the API surface here — the underlying defs go
        on the def list; the chain-phase router silently leaves them
        unresolved when the target shape needs element-side resolution
        the minimum-viable router does not yet cover.
        """
        # ``g.constraints.bc`` is the simplest case — routes via the
        # chain-phase router and writes SPRecords to ``_fem``.  Use a
        # bare node id list so the resolution path is straightforward.
        defn = chain_session.constraints.bc([1], dofs=[1, 1, 1])
        assert defn is not None
        # ``g.masses.point`` similarly.
        mdefn = chain_session.masses.point([1], mass=10.0)
        assert mdefn is not None

    def test_freeze_message_names_operation(
        self, chain_session: apeGmsh,
    ) -> None:
        """Error message surfaces which API was called."""
        with pytest.raises(ChainPhaseError) as exc_info:
            chain_session.model._register(0, 999, None, "point")
        assert "g.model" in str(exc_info.value)


# =============================================================================
# Step 2 — Resolver source adapter
# =============================================================================


class TestFEMDataSource:
    def test_node_ids_and_coords(self) -> None:
        fem = _make_fem(
            node_ids=np.array([10, 20, 30], dtype=np.int64),
        )
        src = FEMDataSource(fem)
        assert list(src.node_ids()) == [10, 20, 30]
        coords = src.node_coords()
        assert coords.shape == (3, 3)

    def test_nodes_for_resolves_pg(self) -> None:
        fem = _make_fem(
            node_pgs={
                (0, 1): {
                    "name": "support",
                    "node_ids": np.array([1, 2], dtype=np.int64),
                    "node_coords": np.array(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        dtype=np.float64,
                    ),
                },
            },
        )
        src = FEMDataSource(fem)
        result = src.nodes_for("support")
        assert sorted(result) == [1, 2]

    def test_nodes_for_resolves_label(self) -> None:
        # Label dict carries pre-prefixed names.
        fem = _make_fem(
            node_labels={
                (0, 1): {
                    "name": "_label:base",
                    "node_ids": np.array([3], dtype=np.int64),
                    "node_coords": np.array(
                        [[2.0, 0.0, 0.0]], dtype=np.float64,
                    ),
                },
            },
        )
        src = FEMDataSource(fem)
        result = src.nodes_for("base")
        assert list(result) == [3]

    def test_nodes_for_missing_raises_key_error(self) -> None:
        fem = _make_fem()
        src = FEMDataSource(fem)
        with pytest.raises(KeyError):
            src.nodes_for("does_not_exist")

    def test_has_target(self) -> None:
        fem = _make_fem(
            node_pgs={
                (0, 1): {
                    "name": "supports",
                    "node_ids": np.array([1], dtype=np.int64),
                    "node_coords": np.array(
                        [[0.0, 0.0, 0.0]], dtype=np.float64,
                    ),
                },
            },
        )
        src = FEMDataSource(fem)
        assert src.has_target("supports") is True
        assert src.has_target("missing") is False

    def test_protocol_runtime_isinstance_check(self) -> None:
        """FEMDataSource satisfies the runtime Protocol."""
        fem = _make_fem()
        src = FEMDataSource(fem)
        assert isinstance(src, ResolverSource)

    def test_make_source_picks_femdata(self) -> None:
        fem = _make_fem()
        s = make_source(fem)
        assert isinstance(s, FEMDataSource)


# =============================================================================
# Step 3 — Chain-phase shim routing
# =============================================================================


class TestChainPhaseRouting:
    def test_bc_route_yields_new_fem_with_sp(self, tmp_path: Path) -> None:
        """``g.constraints.bc`` in chain phase routes through with_load."""
        host = _save_fem(_make_fem(), tmp_path / "h.h5")
        g = apeGmsh.from_h5(host)
        # Initial _fem has no SP records.
        fem_before = g._fem
        assert len(list(fem_before.nodes.sp)) == 0
        # bc against a bare node id list — works via chain-phase router.
        g.constraints.bc([1], dofs=[1, 1, 1])
        fem_after = g._fem
        # FEM identity changed — transform routing replaced it.
        assert fem_after is not fem_before
        # SP records present (3 DOFs × 1 node = 3).
        sp_recs = list(fem_after.nodes.sp)
        assert len(sp_recs) == 3
        assert {rec.dof for rec in sp_recs} == {1, 2, 3}
        assert all(rec.node_id == 1 for rec in sp_recs)

    def test_point_mass_route_yields_new_fem_with_mass(
        self, tmp_path: Path,
    ) -> None:
        host = _save_fem(_make_fem(), tmp_path / "h.h5")
        g = apeGmsh.from_h5(host)
        fem_before = g._fem
        assert len(list(fem_before.nodes.masses)) == 0
        g.masses.point([2], mass=5.0)
        fem_after = g._fem
        assert fem_after is not fem_before
        mrecs = list(fem_after.nodes.masses)
        assert len(mrecs) == 1
        assert mrecs[0].node_id == 2
        # mass tuple is length 6 (mx, my, mz, Ixx, Iyy, Izz).
        assert mrecs[0].mass[0] == 5.0
        assert mrecs[0].mass[1] == 5.0
        assert mrecs[0].mass[2] == 5.0

    def test_route_def_to_fem_returns_none_on_unsupported(self) -> None:
        """route_def_to_fem returns None for shapes outside the
        chain-phase router's coverage.

        v1.1-A added ``EqualDOFDef`` / ``RigidLinkDef`` /
        ``RigidDiaphragmDef``; v1.1-A.2 added ``EmbeddedDef`` +
        ``TiedContactDef``; the PG-constraints PR added ``TieDef`` /
        ``KinematicCouplingDef`` / ``RigidBodyDef`` /
        ``DistributingCouplingDef`` / ``PenaltyDef``.
        ``NodeToSurfaceDef`` resolves through its own bare-tag gmsh
        path (not the broker) and is the canonical "unsupported in
        chain-phase" shape.
        """
        from apeGmsh._kernel.defs.constraints import NodeToSurfaceDef

        fem = _make_fem()
        defn = NodeToSurfaceDef(master_label="a", slave_label="b")
        result = route_def_to_fem(fem, defn)
        assert result is None

    def test_try_route_returns_false_when_no_fem(self) -> None:
        """try_chain_phase_route is a no-op when session._fem is None."""

        class Stub:
            _fem = None

        from apeGmsh._kernel.defs.constraints import BCDef

        defn = BCDef(target="x", dofs=[1, 1, 1])
        assert try_chain_phase_route(Stub(), defn) is False


# =============================================================================
# Step 4 — Tag-collision verifier wired into compose
# =============================================================================


class TestVerifier:
    """The 5 verifier checks fire on synthetic collision fixtures."""

    def test_check_1_imported_tag_outside_reservation(self) -> None:
        """Imported tag outside reservation → PartTagCollisionError."""
        reservations = [ReservationRecord(label="A", base=1000, size=100)]
        imports = {
            "A": ImportedRecords(
                tags=[1000, 1050, 2000],  # 2000 outside [1000, 1100)
            ),
        }
        with pytest.raises(PartTagCollisionError):
            tag_collision_verify(
                reservations=reservations,
                host_pg_names=(),
                module_imports=imports,
            )

    def test_check_2_overlapping_reservations(self) -> None:
        reservations = [
            ReservationRecord(label="A", base=0, size=200),
            ReservationRecord(label="B", base=100, size=200),  # overlaps
        ]
        with pytest.raises(PartTagCollisionError):
            tag_collision_verify(
                reservations=reservations,
                host_pg_names=(),
                module_imports={},
            )

    def test_check_3_constraint_ref_outside_reservation(self) -> None:
        reservations = [ReservationRecord(label="A", base=1000, size=100)]
        imports = {
            "A": ImportedRecords(
                tags=[1000, 1001],
                constraint_refs=[
                    ConstraintReference(
                        kind="equalDOF", field_name="master_node", tag=999,
                    ),
                ],
            ),
        }
        with pytest.raises(ComposeInvariantError):
            tag_collision_verify(
                reservations=reservations,
                host_pg_names=(),
                module_imports=imports,
            )

    def test_check_4_pg_namespace_collision(self) -> None:
        reservations = [ReservationRecord(label="A", base=1000, size=100)]
        imports = {
            "A": ImportedRecords(
                tags=[1000],
                pg_names=["base"],  # → "A.base" after prefix
            ),
        }
        with pytest.raises(PartTagCollisionError):
            tag_collision_verify(
                reservations=reservations,
                host_pg_names=("A.base",),  # host already has it
                module_imports=imports,
            )

    def test_check_5_source_span_exceeds_cap(self) -> None:
        reservations = [ReservationRecord(label="A", base=1000, size=200)]
        imports = {
            "A": ImportedRecords(
                tags=[1000, 1099],
                source_span=500,  # exceeds the 200-cap
            ),
        }
        with pytest.raises(ComposeCapacityError):
            tag_collision_verify(
                reservations=reservations,
                host_pg_names=(),
                module_imports=imports,
                compose_size_per_module=200,
            )

    def test_verifier_passes_clean_fixture(self) -> None:
        """All five checks pass for a well-formed input."""
        reservations = [
            ReservationRecord(label="A", base=1000, size=200),
            ReservationRecord(label="B", base=2000, size=200),
        ]
        imports = {
            "A": ImportedRecords(
                tags=[1000, 1099],
                pg_names=["base"],
                constraint_refs=[
                    ConstraintReference(
                        kind="equalDOF", field_name="m", tag=1050,
                    ),
                ],
                source_span=100,
            ),
            "B": ImportedRecords(
                tags=[2000, 2099],
                pg_names=["base"],  # different module → namespaced as B.base
                source_span=100,
            ),
        }
        # Should not raise.
        tag_collision_verify(
            reservations=reservations,
            host_pg_names=("host_pg",),
            module_imports=imports,
            compose_size_per_module=200,
        )

    def test_verifier_runs_on_real_compose(
        self, tmp_path: Path,
    ) -> None:
        """The verifier is wired into ``FEMData.compose`` and runs on
        every real merge."""
        host = _save_fem(_make_fem(), tmp_path / "host.h5")
        module = _save_fem(
            _make_fem(
                node_ids=np.array([1, 2, 3], dtype=np.int64),
                elem_ids=np.array([10, 11], dtype=np.int64),
            ),
            tmp_path / "module.h5",
        )
        g = apeGmsh.from_h5(host)
        # This should run the verifier internally and pass.
        g.compose(module, label="A")
        # Sanity check the compose succeeded.
        assert "A" in g._fem.composed_from


# =============================================================================
# Step 5 — Rank model
# =============================================================================


class TestRankModel:
    def test_layer_1_single_module_default(
        self, tmp_path: Path,
    ) -> None:
        """One composed module → host on rank 0, module on rank 1."""
        host = _save_fem(_make_fem(), tmp_path / "host.h5")
        module = _save_fem(
            _make_fem(node_ids=np.array([1, 2, 3], dtype=np.int64)),
            tmp_path / "module.h5",
        )
        g = apeGmsh.from_h5(host)
        g.compose(module, label="A")
        fem = g._fem
        # Partition 0 = host (3 nodes), partition 1 = module A (3 nodes).
        parts = sorted(fem.partitions.keys()) if hasattr(
            fem.partitions, "keys",
        ) else list(fem.partitions._records.keys())
        assert 0 in parts
        assert 1 in parts

    def test_layer_2_partition_rank_hint_honoured(
        self, tmp_path: Path,
    ) -> None:
        """``partition_rank=K`` hint overrides Layer 1 default."""
        host = _save_fem(_make_fem(), tmp_path / "host.h5")
        module = _save_fem(
            _make_fem(node_ids=np.array([1, 2, 3], dtype=np.int64)),
            tmp_path / "module.h5",
        )
        g = apeGmsh.from_h5(host)
        g.compose(module, label="A", partition_rank=5)
        fem = g._fem
        parts = list(fem.partitions._records.keys()) if hasattr(
            fem.partitions, "_records",
        ) else list(fem.partitions.keys())
        assert 5 in parts

    def test_layer_2_hint_collision_raises(self) -> None:
        """Two Layer-2 hints colliding on the same rank → ValueError."""
        rec_a = ComposeRecord(
            label="A", source_path="a.h5", source_fem_hash="h1",
            source_neutral_schema_version="2.9.0",
            translate=(0.0, 0.0, 0.0),
            partition_rank=2,
            composed_at="2026-05-26T12:00:00Z",
        )
        rec_b = ComposeRecord(
            label="B", source_path="b.h5", source_fem_hash="h2",
            source_neutral_schema_version="2.9.0",
            translate=(0.0, 0.0, 0.0),
            partition_rank=2,  # collision!
            composed_at="2026-05-26T12:00:00Z",
        )
        fem = _make_fem(
            composed_from=ComposeSet((rec_a, rec_b)),
            node_module_labels=["A", "B", ""],
            elem_module_labels=["A", "B"],
        )
        with pytest.raises(ValueError, match="rank model"):
            _rebuild_partitions_from_modules(fem)

    def test_layer_2_hint_collision_with_host_rank_raises(self) -> None:
        """Hint partition_rank=0 (host's reserved rank) raises."""
        rec_a = ComposeRecord(
            label="A", source_path="a.h5", source_fem_hash="h1",
            source_neutral_schema_version="2.9.0",
            translate=(0.0, 0.0, 0.0),
            partition_rank=0,
            composed_at="2026-05-26T12:00:00Z",
        )
        fem = _make_fem(
            composed_from=ComposeSet((rec_a,)),
            node_module_labels=["A", "A", "A"],
            elem_module_labels=["A", "A"],
        )
        with pytest.raises(ValueError, match="rank model"):
            _rebuild_partitions_from_modules(fem)

    def test_layer_2_auto_around_hint(self) -> None:
        """Hint=3 on A → B gets next-free rank (1)."""
        rec_a = ComposeRecord(
            label="A", source_path="a.h5", source_fem_hash="h1",
            source_neutral_schema_version="2.9.0",
            translate=(0.0, 0.0, 0.0),
            partition_rank=3,
            composed_at="2026-05-26T12:00:00Z",
        )
        rec_b = ComposeRecord(
            label="B", source_path="b.h5", source_fem_hash="h2",
            source_neutral_schema_version="2.9.0",
            translate=(0.0, 0.0, 0.0),
            partition_rank=None,  # Layer 1 → first free is 1
            composed_at="2026-05-26T12:00:00Z",
        )
        fem = _make_fem(
            composed_from=ComposeSet((rec_a, rec_b)),
            node_module_labels=["A", "B", ""],
            elem_module_labels=["A", "B"],
        )
        new = _rebuild_partitions_from_modules(fem)
        parts = sorted(new.nodes._partitions.keys())
        # Host=0 must be present; A=3 from hint; B picks 1 (lowest unused).
        assert 0 in parts
        assert 1 in parts
        assert 3 in parts
        # Node 1 (label "A") goes to partition 3.
        assert 1 in set(new.nodes._partitions[3]["node_ids"].tolist())
        # Node 2 (label "B") goes to partition 1.
        assert 2 in set(new.nodes._partitions[1]["node_ids"].tolist())
        # Node 3 (label "") goes to host partition 0.
        assert 3 in set(new.nodes._partitions[0]["node_ids"].tolist())

    def test_layer_3_metis_override_warning(self) -> None:
        """Existing METIS partitions + new module → UserWarning."""
        rec_a = ComposeRecord(
            label="A", source_path="a.h5", source_fem_hash="h1",
            source_neutral_schema_version="2.9.0",
            translate=(0.0, 0.0, 0.0),
            composed_at="2026-05-26T12:00:00Z",
        )
        fem = _make_fem(
            composed_from=ComposeSet((rec_a,)),
            node_module_labels=["A", "A", "A"],
            elem_module_labels=["A", "A"],
        )
        # Simulate prior METIS partitioning on the host.
        fem.nodes._partitions = {
            7: {"node_ids": np.array([1, 2, 3], dtype=np.int64),
                "element_ids": np.array([], dtype=np.int64)},
        }
        with pytest.warns(UserWarning, match="rank model"):
            _rebuild_partitions_from_modules(fem)

    def test_uncomposed_fem_passthrough(self) -> None:
        """No composed_from → no partition rebuild needed."""
        fem = _make_fem()
        # Should be a no-op — composed_from is empty.
        result = _rebuild_partitions_from_modules(fem)
        assert result is fem
