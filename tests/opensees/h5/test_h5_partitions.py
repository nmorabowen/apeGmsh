"""Tests for :class:`H5Emitter` partition emission (ADR 0027, schema 2.10.0).

Covers the seven acceptance points from the P4 H5 sub-task:

1. Schema-version bumped to 2.10.0.
2. ``/opensees/partitions/`` group is written per
   ``partition_open(rank)`` / ``partition_close()`` bracket.
3. ``partition_ids`` parallel column lands on
   ``/opensees/element_meta/{type}/``.
4. Emitting with no partition brackets writes no partitions group
   AND surfaces the parallel column as all-``-1``.
5. The reader's :meth:`H5Model.partitions` accessor round-trips a
   partitioned model into typed records.
6. Back-compat: a 2.9.0 archive (no partitions group) opens cleanly
   and surfaces an empty partitions list.
7. ``boundary_node_ids`` is symmetric across ranks when the same
   node tag appears under two partition brackets.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._internal.tag_resolution import set_element_nodes
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import H5Emitter, SCHEMA_VERSION
from apeGmsh.opensees.emitter.h5_reader import (
    H5Model,
    PartitionEmittedRecord,
)


# ---------------------------------------------------------------------------
# Test 1 — schema bump
# ---------------------------------------------------------------------------

def test_schema_version_bumped() -> None:
    """Schema bumped from 2.10.0 -> 2.11.0 per the 0-based-rank fix.

    The 0-based-runtime-rank fix flips per-partition ``rank`` attr
    and parallel ``partition_ids`` values from Gmsh's 1-based labels
    to OpenSeesMP's 0-based ``getPID()`` convention; that is a
    breaking-for-readers schema change, hence the minor bump.  The
    bridge's :data:`SCHEMA_VERSION` is the single source for the
    OPENSEES zone (``schema_version.reader_version(OPENSEES)``
    reads it).
    """
    assert SCHEMA_VERSION == "2.11.0"


# ---------------------------------------------------------------------------
# Test 2 — partitions group is written under per-rank brackets
# ---------------------------------------------------------------------------

def test_h5_emitter_writes_partitions_group(tmp_path: Path) -> None:
    """Drive two partition_open/close brackets; assert the resulting
    file shape matches the ADR 0027 layout (parent ``n_partitions``
    attr + per-rank sub-group with ``rank`` / ``n_elements`` /
    ``n_nodes`` attrs + ``element_ids`` / ``node_ids`` datasets).
    """
    e = H5Emitter()
    e.model(ndm=3, ndf=6)

    e.partition_open(0)
    e.node(1, 0.0, 0.0, 0.0)
    e.node(2, 1.0, 0.0, 0.0)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 100, 1, 2, 1, 1)
    e.partition_close()

    e.partition_open(1)
    e.node(3, 2.0, 0.0, 0.0)
    e.node(4, 3.0, 0.0, 0.0)
    set_element_nodes(e, (3, 4))
    e.element("forceBeamColumn", 101, 3, 4, 1, 1)
    e.partition_close()

    out = tmp_path / "two_partitions.h5"
    e.write(str(out))

    with h5py.File(out, "r") as f:
        assert "opensees/partitions" in f
        parent = f["opensees/partitions"]
        assert int(parent.attrs["n_partitions"]) == 2
        assert "partition_00" in parent
        assert "partition_01" in parent

        p0 = parent["partition_00"]
        assert int(p0.attrs["rank"]) == 0
        assert int(p0.attrs["n_elements"]) == 1
        assert int(p0.attrs["n_nodes"]) == 2
        np.testing.assert_array_equal(p0["element_ids"][:], [100])
        np.testing.assert_array_equal(p0["node_ids"][:], [1, 2])

        p1 = parent["partition_01"]
        assert int(p1.attrs["rank"]) == 1
        assert int(p1.attrs["n_elements"]) == 1
        assert int(p1.attrs["n_nodes"]) == 2
        np.testing.assert_array_equal(p1["element_ids"][:], [101])
        np.testing.assert_array_equal(p1["node_ids"][:], [3, 4])


# ---------------------------------------------------------------------------
# Test 3 — partition_ids column on /opensees/element_meta/{type}/
# ---------------------------------------------------------------------------

def test_h5_emitter_partition_ids_column_on_element_meta(
    tmp_path: Path,
) -> None:
    """Each per-type element_meta group gains a ``partition_ids``
    int64 column parallel to ``ids``.  Rows for elements emitted
    under ``partition_open(rank)`` carry that rank's integer; rows
    emitted outside any bracket carry ``-1``.
    """
    e = H5Emitter()
    e.model(ndm=3, ndf=6)

    e.partition_open(0)
    e.node(1, 0.0, 0.0, 0.0)
    e.node(2, 1.0, 0.0, 0.0)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 100, 1, 2, 1, 1)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 101, 1, 2, 1, 1)
    e.partition_close()

    e.partition_open(1)
    e.node(3, 2.0, 0.0, 0.0)
    e.node(4, 3.0, 0.0, 0.0)
    set_element_nodes(e, (3, 4))
    e.element("forceBeamColumn", 200, 3, 4, 1, 1)
    set_element_nodes(e, (3, 4))
    e.element("forceBeamColumn", 201, 3, 4, 1, 1)
    e.partition_close()

    out = tmp_path / "partids.h5"
    e.write(str(out))

    with h5py.File(out, "r") as f:
        g = f["opensees/element_meta/forceBeamColumn"]
        ids = g["ids"][:]
        partition_ids = g["partition_ids"][:]
        assert len(partition_ids) == len(ids)
        # Order is insertion order within the type bin: rank 0 first
        # (tags 100, 101), rank 1 next (tags 200, 201).
        np.testing.assert_array_equal(ids, [100, 101, 200, 201])
        np.testing.assert_array_equal(partition_ids, [0, 0, 1, 1])


# ---------------------------------------------------------------------------
# Test 4 — no brackets => no partitions group, partition_ids all -1
# ---------------------------------------------------------------------------

def test_h5_emitter_no_partition_brackets_writes_no_partitions_group(
    tmp_path: Path,
) -> None:
    """Emitting with no ``partition_open`` / ``partition_close`` calls
    must not create ``/opensees/partitions/`` — the file shape is
    byte-identical to 2.9.x for this case.  The parallel column
    ``partition_ids`` is still emitted but carries the sentinel ``-1``
    in every row.
    """
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.node(1, 0.0, 0.0, 0.0)
    e.node(2, 1.0, 0.0, 0.0)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 10, 1, 2, 1, 1)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 11, 1, 2, 1, 1)

    out = tmp_path / "nopart.h5"
    e.write(str(out))

    with h5py.File(out, "r") as f:
        assert "opensees" in f
        assert "partitions" not in f["opensees"]
        g = f["opensees/element_meta/forceBeamColumn"]
        np.testing.assert_array_equal(g["partition_ids"][:], [-1, -1])


# ---------------------------------------------------------------------------
# Test 5 — H5Model.partitions() reader round-trip
# ---------------------------------------------------------------------------

def test_h5_reader_round_trip_partitions(tmp_path: Path) -> None:
    """Write a partitioned model; round-trip it through the reader
    and assert the typed records carry every field from the file.
    """
    e = H5Emitter()
    e.model(ndm=3, ndf=6)

    e.partition_open(0)
    e.node(1, 0.0, 0.0, 0.0)
    e.node(2, 1.0, 0.0, 0.0)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 100, 1, 2, 1, 1)
    e.partition_close()

    e.partition_open(1)
    e.node(3, 2.0, 0.0, 0.0)
    e.node(4, 3.0, 0.0, 0.0)
    set_element_nodes(e, (3, 4))
    e.element("forceBeamColumn", 101, 3, 4, 1, 1)
    e.partition_close()

    out = tmp_path / "rt.h5"
    e.write(str(out))

    with h5_reader.open(str(out)) as m:
        recs = m.partitions()
        assert isinstance(recs, list)
        assert len(recs) == 2
        assert all(isinstance(r, PartitionEmittedRecord) for r in recs)

        ranks = sorted(r.rank for r in recs)
        assert ranks == [0, 1]

        by_rank = {r.rank: r for r in recs}
        assert by_rank[0].element_ids == (100,)
        assert by_rank[0].node_ids == (1, 2)
        assert by_rank[0].boundary_node_ids == ()
        assert by_rank[1].element_ids == (101,)
        assert by_rank[1].node_ids == (3, 4)
        assert by_rank[1].boundary_node_ids == ()


# ---------------------------------------------------------------------------
# Test 6 — back-compat against a synthetic prior-minor archive
# ---------------------------------------------------------------------------

def test_h5_reader_back_compat_pre_partition_schema(
    tmp_path: Path,
) -> None:
    """A file written under the previous opensees-zone schema (2.10.0,
    pre-0-based-rank-fix) opens cleanly under the current reader
    and ``H5Model.partitions()`` returns ``[]`` when no partition
    brackets were emitted.

    The 2.10.0 → 2.11.0 bump (0-based runtime ranks) is breaking only
    for readers that interpret the ``rank`` attr / ``partition_ids``
    values as 1-based ``PartitionRecord.id`` labels.  For files
    written WITHOUT any ``partition_open`` / ``partition_close``
    brackets the two versions are byte-shape-equivalent (no
    partitions group; ``partition_ids`` column is all ``-1``), so a
    2.10.0 stamp on such a file is honored by the two-version reader
    window from :mod:`apeGmsh.opensees._internal.schema_version`.
    """
    e = H5Emitter(schema_version="2.10.0")
    e.model(ndm=3, ndf=6)
    e.node(1, 0.0, 0.0, 0.0)
    e.node(2, 1.0, 0.0, 0.0)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 10, 1, 2, 1, 1)

    out = tmp_path / "legacy.h5"
    e.write(str(out))

    with h5_reader.open(str(out)) as m:
        assert m.schema_version == "2.10.0"
        recs = m.partitions()
        assert recs == []


# ---------------------------------------------------------------------------
# Test 7 — boundary_node_ids populated when a node appears under two ranks
# ---------------------------------------------------------------------------

def test_boundary_node_ids_populated_when_present(
    tmp_path: Path,
) -> None:
    """When the same node tag is declared under two partition brackets
    (the cross-partition MP-constraint case from ADR 0027 INV-2 — a
    foreign-side rank declares a node native to another rank before
    emitting the constraint line), the boundary set is non-empty on
    both ranks.

    The boundary set is symmetric: node 2 appears in both rank 0's
    and rank 1's ``boundary_node_ids`` because both ranks declared it.
    """
    e = H5Emitter()
    e.model(ndm=3, ndf=6)

    # Rank 0 owns nodes 1, 2 (a single column).
    e.partition_open(0)
    e.node(1, 0.0, 0.0, 0.0)
    e.node(2, 0.0, 0.0, 1.0)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 100, 1, 2, 1, 1)
    e.partition_close()

    # Rank 1 owns nodes 2, 3 (a second column sharing top node 2
    # with rank 0 — declared as foreign before the element line).
    e.partition_open(1)
    e.node(2, 0.0, 0.0, 1.0)
    e.node(3, 1.0, 0.0, 1.0)
    set_element_nodes(e, (2, 3))
    e.element("forceBeamColumn", 101, 2, 3, 1, 1)
    e.partition_close()

    out = tmp_path / "boundary.h5"
    e.write(str(out))

    with h5_reader.open(str(out)) as m:
        recs = m.partitions()
        by_rank = {r.rank: r for r in recs}
        # Node 2 is shared — appears on both ranks' boundary sets.
        assert 2 in by_rank[0].boundary_node_ids
        assert 2 in by_rank[1].boundary_node_ids
        # Native-only nodes do not appear in the boundary set.
        assert 1 not in by_rank[0].boundary_node_ids
        assert 3 not in by_rank[1].boundary_node_ids
