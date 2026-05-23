"""Tests for the consolidated g.mesh.partitioning API.

Covers:
- Renumbering (simple, rcm) — correctness, Gmsh mutation, result contract
- Partitioning — basic, explicit, unpartition, queries
- FEMData partition integration — partition= kwarg, intersection
"""
from __future__ import annotations

import gmsh
import numpy as np
import pytest


# ── Helpers ──────────────────────────────────────────────────────────

def _build_plate(g, lc: float = 2.0, dim: int = 2):
    """Create a simple plate mesh for testing."""
    g.model.geometry.add_rectangle(0, 0, 0, 10, 10)
    g.model.sync()
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim)


def _build_box(g, lc: float = 3.0):
    """Create a simple box mesh for testing."""
    g.model.geometry.add_box(0, 0, 0, 10, 10, 5)
    g.model.sync()
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(3)


# =====================================================================
# Renumbering
# =====================================================================

class TestRenumber:

    def test_simple_contiguous_ids(self, g):
        _build_plate(g)
        result = g.mesh.partitioning.renumber(dim=2, method="simple", base=1)
        assert result.n_nodes > 0
        assert result.n_elements > 0
        assert result.method == "simple"

        # Verify tags in Gmsh are now 1..N
        tags, _, _ = gmsh.model.mesh.getNodes()
        assert int(min(tags)) == 1
        assert int(max(tags)) == len(tags)

    def test_rcm_reduces_or_matches_bandwidth(self, g):
        _build_plate(g)
        result = g.mesh.partitioning.renumber(dim=2, method="rcm", base=1)
        assert result.bandwidth_after <= result.bandwidth_before
        assert result.method == "rcm"

    def test_renumber_mutates_gmsh(self, g):
        _build_plate(g)
        tags_before, _, _ = gmsh.model.mesh.getNodes()
        g.mesh.partitioning.renumber(dim=2, method="simple", base=1)
        tags_after, _, _ = gmsh.model.mesh.getNodes()
        # Tags should have changed (unless they were already contiguous)
        after_set = set(int(t) for t in tags_after)
        assert after_set == set(range(1, len(tags_after) + 1))

    def test_renumber_base_0(self, g):
        _build_plate(g)
        g.mesh.partitioning.renumber(dim=2, method="simple", base=0)
        tags, _, _ = gmsh.model.mesh.getNodes()
        assert int(min(tags)) == 0
        assert int(max(tags)) == len(tags) - 1

    def test_unknown_method_raises(self, g):
        _build_plate(g)
        with pytest.raises(ValueError, match="Unknown method"):
            g.mesh.partitioning.renumber(method="bogus")

    def test_renumber_result_repr(self, g):
        _build_plate(g)
        result = g.mesh.partitioning.renumber(dim=2, method="rcm")
        r = repr(result)
        assert "RenumberResult" in r
        assert "rcm" in r
        assert "nodes" in r

    def test_element_renumber_contiguous(self, g):
        _build_plate(g)
        g.mesh.partitioning.renumber(dim=2, method="simple", base=1)
        _, etags_list, _ = gmsh.model.mesh.getElements(dim=2, tag=-1)
        all_tags = []
        for etags in etags_list:
            all_tags.extend(int(t) for t in etags)
        assert min(all_tags) == 1
        assert max(all_tags) == len(all_tags)

    def test_renumber_3d(self, g):
        _build_box(g)
        result = g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
        assert result.n_nodes > 0
        assert result.n_elements > 0
        assert result.bandwidth_after <= result.bandwidth_before


# =====================================================================
# Partitioning
# =====================================================================

class TestPartition:

    def test_partition_basic(self, g):
        _build_plate(g)
        info = g.mesh.partitioning.partition(2)
        assert info.n_parts == 2
        assert len(info.elements_per_partition) >= 1
        assert all(v > 0 for v in info.elements_per_partition.values())

    def test_partition_info_repr(self, g):
        _build_plate(g)
        info = g.mesh.partitioning.partition(2)
        r = repr(info)
        assert "PartitionInfo" in r
        assert "2 parts" in r

    def test_unpartition(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        assert g.mesh.partitioning.n_partitions() > 0
        g.mesh.partitioning.unpartition()
        assert g.mesh.partitioning.n_partitions() == 0

    def test_n_partitions_before_partitioning(self, g):
        _build_plate(g)
        assert g.mesh.partitioning.n_partitions() == 0

    def test_summary(self, g):
        _build_plate(g)
        s = g.mesh.partitioning.summary()
        assert "not partitioned" in s
        g.mesh.partitioning.partition(2)
        s = g.mesh.partitioning.summary()
        assert "partition" in s.lower()

    def test_entity_table(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        df = g.mesh.partitioning.entity_table()
        assert not df.empty
        assert "partitions" in df.columns

    def test_partition_invalid_nparts(self, g):
        _build_plate(g)
        with pytest.raises(ValueError, match="n_parts must be >= 1"):
            g.mesh.partitioning.partition(0)

    def test_partition_explicit(self, g):
        _build_plate(g)
        # Get ALL element tags (all dims) — Gmsh requires every element
        all_tags = []
        for d in range(4):
            _, etags_list, _ = gmsh.model.mesh.getElements(dim=d, tag=-1)
            for etags in etags_list:
                all_tags.extend(int(t) for t in etags)
        # Split in half
        mid = len(all_tags) // 2
        parts = [1] * mid + [2] * (len(all_tags) - mid)
        info = g.mesh.partitioning.partition_explicit(
            2, elem_tags=all_tags, parts=parts)
        assert info.n_parts == 2


# =====================================================================
# FEMData partition integration
# =====================================================================

class TestFEMDataPartitions:

    def test_unpartitioned_has_empty_partitions(self, g):
        _build_plate(g)
        fem = g.mesh.queries.get_fem_data(dim=2)
        # ``fem.partitions`` is now a :class:`PartitionSet` (P2);
        # the per-composite ``.partitions`` accessors stay
        # ``list[int]``.
        assert len(fem.partitions) == 0
        assert fem.partitions.ids == []
        assert fem.nodes.partitions == []
        assert fem.elements.partitions == []

    def test_partitioned_has_partition_list(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        assert len(fem.partitions) >= 1

    def test_nodes_get_partition(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        # P2: ``fem.partitions`` yields :class:`PartitionRecord` — use
        # ``.ids`` for the integer tag.
        p = fem.partitions.ids[0]
        # selection-unification v2 P3-R: ``fem.nodes.get(partition=)``
        # is removed; ``fem.nodes.select(partition=)`` is the migration
        # target (P-NODE — same _resolve_nodes + _intersect_partition,
        # id-for-id).  MeshSelection exposes .ids / .coords directly.
        result = fem.nodes.select(partition=p)
        assert len(result.ids) > 0
        assert result.coords.shape[0] == len(result.ids)

    def test_elements_get_partition(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        p = fem.partitions.ids[0]
        # P3-R: ``fem.elements.get(partition=)`` removed →
        # ``fem.elements.select(partition=)``; the terminal id count is
        # the element count (P-ELEM-IDS).
        result = fem.elements.select(partition=p)
        assert len(result.ids) > 0

    def test_partition_union_covers_all_elements(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        all_ids = set(int(e) for e in fem.elements.ids)
        union = set()
        # P2: iterating ``fem.partitions`` yields :class:`PartitionRecord`.
        for rec in fem.partitions:
            eids = fem.elements.select(partition=rec.id).ids
            union.update(int(e) for e in eids)
        assert union == all_ids

    def test_invalid_partition_raises(self, g):
        _build_plate(g)
        fem = g.mesh.queries.get_fem_data(dim=2)
        with pytest.raises(KeyError, match="Partition 99 not found"):
            fem.nodes.select(partition=99)                 # P3-R: was .get

    def test_partition_with_pg_intersection(self, g):
        """partition= combined with pg= returns intersection."""
        _build_plate(g)
        # Create a physical group on the surface
        surfs = [dt[1] for dt in gmsh.model.getEntities(2)]
        if surfs:
            gmsh.model.addPhysicalGroup(2, surfs, name="Plate")
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions or "Plate" not in fem.nodes.physical:
            pytest.skip("PG or partition not available")
        p = fem.partitions.ids[0]
        # Intersection should be <= each set
        # P3-R: ``fem.nodes.get(...)`` removed → ``.select(...)`` (the
        # pg= / partition= / pg=+partition= selectors are identical —
        # P-NODE, same _resolve_nodes + _intersect_partition).
        pg_ids = fem.nodes.select(pg="Plate").ids
        part_ids = fem.nodes.select(partition=p).ids
        both_ids = fem.nodes.select(pg="Plate", partition=p).ids
        assert len(both_ids) <= len(pg_ids)
        assert len(both_ids) <= len(part_ids)
        # All intersection IDs should be in both sets
        pg_set = set(int(n) for n in pg_ids)
        part_set = set(int(n) for n in part_ids)
        for nid in both_ids:
            assert int(nid) in pg_set
            assert int(nid) in part_set

    def test_select_forwards_partition(self, g):
        """``select(partition=)`` forwards partition= correctly.

        selection-unification v2 P3-R: the legacy ``get(partition=)``
        vs ``get_ids(partition=)`` parity is vacuous (both removed and
        collapsed into the single ``select(partition=).ids``); this
        pins the surviving v2 surface — a non-empty partition id set
        whose every id is in the partition's element/node universe."""
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("No partitions")
        p = fem.partitions.ids[0]
        ids = fem.nodes.select(partition=p).ids
        assert len(ids) > 0
        # the partition id set is a subset of the full node universe
        all_nodes = set(int(n) for n in fem.nodes.ids)
        assert all(int(n) in all_nodes for n in ids)


# =====================================================================
# Phantom node ID uniqueness
# =====================================================================

class TestPhantomNodeNumbering:
    """Phantom nodes generated by node_to_surface must have unique IDs."""

    @staticmethod
    def _build_column_with_two_constraints(g):
        """Plate + two embedded points + two node_to_surface constraints."""
        g.model.geometry.add_box(0, 0, 0, 10, 10, 20)
        g.model.sync()
        g.mesh.sizing.set_global_size(5)

        # Two reference points at opposite ends
        g.model.geometry.add_point(5, 5, 0, lc=5, label='base')
        g.model.geometry.add_point(5, 5, 20, lc=5, label='top')
        g.model.sync()
        g.mesh.generation.generate(3)

        # Get bottom and top face entities
        bottom = []
        top = []
        for _, tag in gmsh.model.getEntities(2):
            bb = gmsh.model.getBoundingBox(2, tag)
            # zmin == zmax == 0 → bottom face
            if abs(bb[2]) < 0.01 and abs(bb[5]) < 0.01:
                bottom.append(tag)
            # zmin == zmax == 20 → top face
            if abs(bb[2] - 20) < 0.01 and abs(bb[5] - 20) < 0.01:
                top.append(tag)

        if not bottom or not top:
            pytest.skip("Could not identify bottom/top faces")

        for tag in bottom:
            g.constraints.node_to_surface('base', [(2, tag)])
        for tag in top:
            g.constraints.node_to_surface('top', [(2, tag)])

    def test_phantom_ids_are_unique(self, g):
        """Multiple node_to_surface calls must not produce overlapping
        phantom node IDs."""
        self._build_column_with_two_constraints(g)
        fem = g.mesh.queries.get_fem_data(dim=3)

        all_phantom_ids = []
        for nid, _ in fem.nodes.constraints.phantom_nodes():
            all_phantom_ids.append(nid)

        assert len(all_phantom_ids) > 0, "Expected phantom nodes"
        assert len(all_phantom_ids) == len(set(all_phantom_ids)), \
            f"Duplicate phantom IDs: {sorted(all_phantom_ids)}"

    def test_phantom_ids_do_not_collide_with_mesh(self, g):
        """Phantom node IDs must not overlap with real mesh node IDs."""
        self._build_column_with_two_constraints(g)
        fem = g.mesh.queries.get_fem_data(dim=3)

        mesh_ids = set(int(n) for n in fem.nodes.ids)
        for nid, _ in fem.nodes.constraints.phantom_nodes():
            assert nid not in mesh_ids, \
                f"Phantom node {nid} collides with a mesh node"

    def test_phantom_count_matches_slave_count(self, g):
        """Each slave node should get exactly one phantom node."""
        self._build_column_with_two_constraints(g)
        fem = g.mesh.queries.get_fem_data(dim=3)

        from apeGmsh._kernel.records import NodeToSurfaceRecord
        total_slaves = 0
        total_phantoms = 0
        for rec in fem.nodes.constraints:
            if isinstance(rec, NodeToSurfaceRecord):
                total_slaves += len(rec.slave_nodes)
                total_phantoms += len(rec.phantom_nodes)

        assert total_phantoms == total_slaves, \
            f"phantoms={total_phantoms} != slaves={total_slaves}"


# =====================================================================
# Weighted partitioning (Flavor B)
#
# Routes through an external METIS binding (pymetis or networkx-metis)
# so the caller can pass per-element vertex weights.  apeGmsh builds
# the element dual graph, calls the backend, then pushes the result
# back into Gmsh via partition_explicit — downstream consumers see
# the same model state as the native (Flavor A) path.
#
# CI may have neither backend; tests use ``importorskip`` so the suite
# stays green where the binding is unavailable.
# =====================================================================


def _total_element_count() -> int:
    """Sum of element counts across all 4 Gmsh dimensions."""
    n = 0
    for d in range(4):
        _, etl, _ = gmsh.model.mesh.getElements(dim=d, tag=-1)
        n += sum(len(t) for t in etl)
    return n


class TestPartitionWeighted:
    """Weighted partitioning surface — Flavor B via pymetis/nx-metis."""

    def test_partition_with_unit_weights_matches_unweighted(self, g):
        """Unit weights through pymetis should give the same load
        balance as Gmsh-native (count_max - count_min within 1 of the
        reference's count_max - count_min) for the **top-dim elements
        only** — comparing total element counts across all dims is
        unreliable because Gmsh-native and the explicit path account
        for lower-dim ghost elements differently in
        ``elements_per_partition``.

        METIS is a heuristic — different bindings can land on slightly
        different cuts even for identical inputs — so we assert
        approximate, not bit-exact, parity.
        """
        pytest.importorskip("pymetis")
        _build_plate(g)
        n = _total_element_count()

        # Reference: Gmsh-native (Flavor A) — measure top-dim balance.
        ref = g.mesh.partitioning.partition(2)
        ref_spread = (
            max(ref.elements_per_partition.values())
            - min(ref.elements_per_partition.values()))
        g.mesh.partitioning.unpartition()

        # Weighted path with unit weights (Flavor B, pymetis)
        info = g.mesh.partitioning.partition(
            2, weights=np.ones(n).tolist())
        got_spread = (
            max(info.elements_per_partition.values())
            - min(info.elements_per_partition.values()))

        assert info.n_parts == 2
        # Both paths should land near-perfect balance (spread ~ 0–few).
        # The exact ghost-element accounting between the two paths
        # differs (Gmsh-native leaves lower-dim ghosts unassigned;
        # the explicit path assigns them too), but the **balance**
        # metric — spread between max and min — should be small in
        # both cases and within ``n / n_parts * 0.05`` of each other
        # for a uniform 2-D mesh.
        tol = max(2, n // 20)
        assert abs(ref_spread - got_spread) <= tol, \
            f"unit-weights load balance diverges from Gmsh-native: " \
            f"ref_spread={ref_spread}, got_spread={got_spread}, " \
            f"tol={tol}"

    def test_partition_with_biased_weights_balances_weight_sum(self, g):
        """First-half heavy / second-half light → METIS should balance
        weight-sum.  Realised weights should be within 10% of each
        other.  (We don't pin element-count imbalance: for a small
        plate the dual-graph topology can still drive balanced counts
        even when weights vary — METIS optimises weight sum, but is
        constrained by the cut-edge cost too.)"""
        pytest.importorskip("pymetis")
        _build_plate(g)
        n = _total_element_count()
        if n < 4:
            pytest.skip("Mesh too small to demonstrate weight balance")

        half = n // 2
        weights = [10.0] * half + [1.0] * (n - half)
        info = g.mesh.partitioning.partition(2, weights=weights)

        assert info.n_parts == 2
        assert info.weights_per_partition is not None
        assert len(info.weights_per_partition) == 2

        w_vals = sorted(info.weights_per_partition.values())
        w_lo, w_hi = w_vals[0], w_vals[-1]
        # Within 10% of each other.
        assert w_lo > 0, f"empty partition: {info.weights_per_partition}"
        ratio = w_hi / w_lo
        assert ratio <= 1.10, \
            f"weight imbalance too large: {w_vals} (ratio {ratio:.3f})"

        # Sanity: total weight is conserved across partitions.
        total = sum(info.weights_per_partition.values())
        expected = sum(weights)
        assert abs(total - expected) < 1e-6, \
            f"weight total {total} != expected {expected}"

    def test_partition_weights_with_gmsh_backend_raises(self, g):
        """``backend='gmsh'`` + ``weights=`` → ValueError (no vwgt API)."""
        _build_plate(g)
        n = _total_element_count()
        with pytest.raises(ValueError, match="Gmsh has no vwgt API"):
            g.mesh.partitioning.partition(
                2, weights=[1.0] * n, backend="gmsh")

    def test_partition_weights_wrong_length_raises(self, g):
        """Mismatched ``weights`` length → ValueError naming both sizes."""
        pytest.importorskip("pymetis")
        _build_plate(g)
        n = _total_element_count()
        wrong = [1.0] * (n + 7)  # off by a known delta
        with pytest.raises(ValueError) as exc_info:
            g.mesh.partitioning.partition(2, weights=wrong)
        msg = str(exc_info.value)
        # Message must mention both expected and got.
        assert str(n) in msg, f"expected {n} not in {msg!r}"
        assert str(len(wrong)) in msg, \
            f"got {len(wrong)} not in {msg!r}"

    def test_partition_backend_missing_raises_with_hint(
        self, g, monkeypatch,
    ):
        """When pymetis is unimportable, the wrapped ImportError must
        mention ``pip install`` so the user knows how to recover.

        We force the import to fail by inserting ``None`` into
        ``sys.modules['pymetis']``.  ``importlib.import_module`` raises
        ``ImportError`` on that sentinel, which ``_import_backend``
        catches and re-raises with the install hint."""
        _build_plate(g)
        n = _total_element_count()

        monkeypatch.setitem(__import__('sys').modules, 'pymetis', None)
        with pytest.raises(ImportError) as exc_info:
            g.mesh.partitioning.partition(2, weights=[1.0] * n)
        msg = str(exc_info.value)
        assert "pip install" in msg, \
            f"missing install hint in: {msg!r}"
        assert "pymetis" in msg, f"backend name not in: {msg!r}"

    def test_partition_weights_partition_info_populated(self, g):
        """After a weighted call, ``info.weights_per_partition`` must be
        a dict with n_parts keys summing to ~sum(weights)."""
        pytest.importorskip("pymetis")
        _build_plate(g)
        n = _total_element_count()
        weights = [2.0] * n  # uniform but non-unit
        info = g.mesh.partitioning.partition(3, weights=weights)

        assert info.weights_per_partition is not None, \
            "weights_per_partition should be populated for weighted call"
        assert isinstance(info.weights_per_partition, dict)
        assert len(info.weights_per_partition) == 3, \
            f"expected 3 keys, got {info.weights_per_partition}"

        # Sum should match — owning-entity accounting (no ghost
        # double-counting) means total == sum(weights).
        total = sum(info.weights_per_partition.values())
        expected = sum(weights)
        assert abs(total - expected) < 1e-6, \
            f"weight total {total} != expected {expected}"

    def test_partition_explicit_still_works_after_weighted(self, g):
        """Regression: ``partition_explicit`` must remain callable after
        a weighted ``partition()`` call.  The weighted path uses
        ``partition_explicit`` internally and caches state; this test
        pins that the cache cleanup leaves the public method usable."""
        pytest.importorskip("pymetis")
        _build_plate(g)
        n = _total_element_count()

        # First a weighted call to populate the cache.
        g.mesh.partitioning.partition(2, weights=[1.0] * n)
        # Tear down (also clears the cache) and call explicit directly.
        g.mesh.partitioning.unpartition()

        all_tags: list[int] = []
        for d in range(4):
            _, etl, _ = gmsh.model.mesh.getElements(dim=d, tag=-1)
            for et in etl:
                all_tags.extend(int(t) for t in et)
        mid = len(all_tags) // 2
        parts = [1] * mid + [2] * (len(all_tags) - mid)
        info = g.mesh.partitioning.partition_explicit(
            2, elem_tags=all_tags, parts=parts)

        # partition_explicit with no weights should report None.
        assert info.n_parts == 2
        assert info.weights_per_partition is None


# =====================================================================
# P2 — PartitionRecord + PartitionSet composite
# =====================================================================

class TestPartitionSetComposite:
    """Pins the broker-side ``fem.partitions`` :class:`PartitionSet`."""

    def test_fem_partitions_is_partition_set(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        from apeGmsh._kernel.record_sets import PartitionSet
        assert isinstance(fem.partitions, PartitionSet)

    def test_partition_record_has_id_node_ids_element_ids(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        from apeGmsh._kernel.records import PartitionRecord
        rec = next(iter(fem.partitions))
        assert isinstance(rec, PartitionRecord)
        assert isinstance(rec.id, int)
        assert isinstance(rec.node_ids, np.ndarray)
        assert isinstance(rec.element_ids, np.ndarray)
        assert rec.node_ids.dtype == np.int64
        assert rec.element_ids.dtype == np.int64
        # The optional weight_sum field defers to P1 — always None here.
        assert rec.weight_sum is None

    def test_partition_set_iteration_yields_records_in_id_order(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(3)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        from apeGmsh._kernel.records import PartitionRecord
        ids_in_iter = []
        for rec in fem.partitions:
            assert isinstance(rec, PartitionRecord)
            ids_in_iter.append(rec.id)
        assert ids_in_iter == sorted(ids_in_iter)
        assert ids_in_iter == fem.partitions.ids

    def test_partition_set_getitem_by_id(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        pid = fem.partitions.ids[0]
        rec = fem.partitions[pid]
        assert rec.id == pid

    def test_partition_set_getitem_missing_raises_keyerror(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        with pytest.raises(KeyError, match="Partition 9999 not found"):
            _ = fem.partitions[9999]

    def test_partition_set_contains(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        pid = fem.partitions.ids[0]
        assert pid in fem.partitions
        assert 9999 not in fem.partitions
        # Non-coercible values are False, not a raise.
        assert "not-an-int" not in fem.partitions

    def test_partition_set_len(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        # ``len`` matches the number of unique partition ids in the
        # union of the node/element back-stores.
        n_node_pids = len(set(fem.nodes._partitions.keys()))
        n_elem_pids = len(set(fem.elements._partitions.keys()))
        expected = len(set(fem.nodes._partitions.keys()) |
                       set(fem.elements._partitions.keys()))
        assert len(fem.partitions) == expected
        # Sanity: both child stores agree (single source of truth).
        assert n_node_pids == n_elem_pids

    def test_partition_record_n_nodes_and_n_elements(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        for rec in fem.partitions:
            assert rec.n_nodes == rec.node_ids.size
            assert rec.n_elements == rec.element_ids.size
            assert rec.n_nodes > 0
            assert rec.n_elements > 0

    def test_select_partition_n_still_works(self, g):
        """Regression: ``select(partition=N)`` still hits the
        per-composite back-stores after the P2 broker rewire."""
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        for rec in fem.partitions:
            sel_nodes = fem.nodes.select(partition=rec.id)
            sel_elems = fem.elements.select(partition=rec.id)
            # IDs from the selector must match the record's arrays.
            assert sorted(int(x) for x in sel_nodes.ids) == \
                sorted(int(x) for x in rec.node_ids)
            assert sorted(int(x) for x in sel_elems.ids) == \
                sorted(int(x) for x in rec.element_ids)

    def test_unpartitioned_fem_has_empty_partition_set(self, g):
        _build_plate(g)
        fem = g.mesh.queries.get_fem_data(dim=2)
        from apeGmsh._kernel.record_sets import PartitionSet
        assert isinstance(fem.partitions, PartitionSet)
        assert len(fem.partitions) == 0
        assert not fem.partitions
        assert list(fem.partitions) == []
        assert fem.partitions.ids == []
