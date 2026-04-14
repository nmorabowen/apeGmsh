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
        assert fem.partitions == []
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
        p = fem.partitions[0]
        ids, coords = fem.nodes.get(partition=p)
        assert len(ids) > 0
        assert coords.shape[0] == len(ids)

    def test_elements_get_partition(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        p = fem.partitions[0]
        result = fem.elements.get(partition=p)
        assert result.n_elements > 0

    def test_partition_union_covers_all_elements(self, g):
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("Partitioning did not produce queryable partitions")
        all_ids = set(int(e) for e in fem.elements.ids)
        union = set()
        for p in fem.partitions:
            eids = fem.elements.get(partition=p).ids
            union.update(int(e) for e in eids)
        assert union == all_ids

    def test_invalid_partition_raises(self, g):
        _build_plate(g)
        fem = g.mesh.queries.get_fem_data(dim=2)
        with pytest.raises(KeyError, match="Partition 99 not found"):
            fem.nodes.get(partition=99)

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
        p = fem.partitions[0]
        # Intersection should be <= each set
        pg_ids, _ = fem.nodes.get(pg="Plate")
        part_ids, _ = fem.nodes.get(partition=p)
        both_ids, _ = fem.nodes.get(pg="Plate", partition=p)
        assert len(both_ids) <= len(pg_ids)
        assert len(both_ids) <= len(part_ids)
        # All intersection IDs should be in both sets
        pg_set = set(int(n) for n in pg_ids)
        part_set = set(int(n) for n in part_ids)
        for nid in both_ids:
            assert int(nid) in pg_set
            assert int(nid) in part_set

    def test_get_ids_with_partition(self, g):
        """get_ids() forwards partition= correctly."""
        _build_plate(g)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
        if not fem.partitions:
            pytest.skip("No partitions")
        p = fem.partitions[0]
        ids_via_get = fem.nodes.get(partition=p).ids
        ids_via_shortcut = fem.nodes.get_ids(partition=p)
        np.testing.assert_array_equal(ids_via_get, ids_via_shortcut)


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
        for nid, _ in fem.nodes.constraints.extra_nodes():
            all_phantom_ids.append(nid)

        assert len(all_phantom_ids) > 0, "Expected phantom nodes"
        assert len(all_phantom_ids) == len(set(all_phantom_ids)), \
            f"Duplicate phantom IDs: {sorted(all_phantom_ids)}"

    def test_phantom_ids_do_not_collide_with_mesh(self, g):
        """Phantom node IDs must not overlap with real mesh node IDs."""
        self._build_column_with_two_constraints(g)
        fem = g.mesh.queries.get_fem_data(dim=3)

        mesh_ids = set(int(n) for n in fem.nodes.ids)
        for nid, _ in fem.nodes.constraints.extra_nodes():
            assert nid not in mesh_ids, \
                f"Phantom node {nid} collides with a mesh node"

    def test_phantom_count_matches_slave_count(self, g):
        """Each slave node should get exactly one phantom node."""
        self._build_column_with_two_constraints(g)
        fem = g.mesh.queries.get_fem_data(dim=3)

        from apeGmsh.solvers.Constraints import NodeToSurfaceRecord
        total_slaves = 0
        total_phantoms = 0
        for rec in fem.nodes.constraints:
            if isinstance(rec, NodeToSurfaceRecord):
                total_slaves += len(rec.slave_nodes)
                total_phantoms += len(rec.phantom_nodes)

        assert total_phantoms == total_slaves, \
            f"phantoms={total_phantoms} != slaves={total_slaves}"
