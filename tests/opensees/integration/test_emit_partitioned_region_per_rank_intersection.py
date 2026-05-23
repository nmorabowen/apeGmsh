"""Integration test for ADR 0027 INV-4 — per-rank region intersection.

Per ADR 0027 §"Regions interaction":

* A region's ``element_ids`` / ``node_ids`` are intersected with
  per-rank ownership before emission on each rank.
* Empty intersection ⇒ no ``region`` line emitted on that rank
  (no bare ``region $tag`` line).
* The region ``tag`` is the **same** scalar across every rank that
  emits, so MPCO post-processing can stitch by tag identity.
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
)


def _per_rank_calls(rec: RecordingEmitter) -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = {}
    cur: int | None = None
    for name, args, kwargs in rec.calls:
        if name == "partition_open":
            cur = int(args[0])
            out.setdefault(cur, [])
        elif name == "partition_close":
            cur = None
        elif cur is not None:
            out[cur].append((name, args, kwargs))
    return out


def test_region_intersects_per_rank_same_tag() -> None:
    """A region spanning nodes across two ranks emits on each rank
    with the rank's owned subset and the SAME tag (INV-4).
    """
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    # Region spanning ALL four nodes (2 on rank 0, 2 on rank 1).
    ops.region(name="all_nodes", nodes=[1, 2, 3, 4])

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)

    # Each rank must emit one region line; both with the SAME tag but
    # disjoint -node subsets.
    region_calls_rank0 = [a for (n, a, _k) in per_rank[0] if n == "region"]
    region_calls_rank1 = [a for (n, a, _k) in per_rank[1] if n == "region"]
    assert len(region_calls_rank0) == 1
    assert len(region_calls_rank1) == 1

    # INV-4: same tag across ranks.
    tag0 = int(region_calls_rank0[0][0])
    tag1 = int(region_calls_rank1[0][0])
    assert tag0 == tag1, (
        f"INV-4: region tag must be the same scalar across ranks; "
        f"rank 0={tag0}, rank 1={tag1}"
    )

    # Rank 0 owns nodes {1, 2}; rank 1 owns {3, 4}. The region call's
    # args are (tag, '-node', n1, n2, ...).
    nodes0 = sorted(int(x) for x in region_calls_rank0[0][2:])
    nodes1 = sorted(int(x) for x in region_calls_rank1[0][2:])
    assert nodes0 == [1, 2], f"rank 0 region must carry only owned nodes; got {nodes0}"
    assert nodes1 == [3, 4], f"rank 1 region must carry only owned nodes; got {nodes1}"


def test_region_empty_intersection_emits_nothing_on_that_rank() -> None:
    """A region whose nodes all belong to rank 0 emits NO region line
    on rank 1 (INV-4: empty intersection ⇒ no emit on that rank).
    """
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    # Region only on rank 0's nodes.
    ops.region(name="left_only", nodes=[1, 2])

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)

    region_calls_rank0 = [a for (n, a, _k) in per_rank[0] if n == "region"]
    region_calls_rank1 = [a for (n, a, _k) in per_rank[1] if n == "region"]
    assert len(region_calls_rank0) == 1, (
        "rank 0 owns all region members; must emit one region line"
    )
    assert len(region_calls_rank1) == 0, (
        f"INV-4: rank 1 has empty intersection; must emit NO region line. "
        f"Got: {region_calls_rank1!r}"
    )
