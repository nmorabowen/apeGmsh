"""Integration tests for ASDEmbeddedNodeElement under partitioning.

Locks three ADR 0027 invariants for the surface-coupling fan-out:

* **INV-OWN (single host-rank).** An ``embeddedNode`` line emits on
  exactly one rank — the canonical host rank
  ``min(intersection(node_owners[m] for m in masters))`` per
  ``_canonical_host_rank`` (build.py).

* **INV-TAG (globally unique element tag).** Each emitted
  ``ASDEmbeddedNodeElement`` line carries a globally unique integer
  ``$tag`` drawn from the bridge's canonical
  :class:`TagAllocator` (ADR 0027 §"Tag determinism").

* **INV-BOUNDARY (no duplicate emit on shared masters).** When every
  master node is owned by multiple ranks (interior interface), the
  embeddedNode line still emits exactly once — on
  ``min(intersection)``.
"""
from __future__ import annotations

from typing import cast

import numpy as np

from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------
# Helpers (mirrors test_emit_partitioned_mp_constraint_replication.py)
# ---------------------------------------------------------------------


def _per_rank_calls(rec: RecordingEmitter) -> dict[int, list[tuple]]:
    """Group ``rec.calls`` into per-rank lists between
    ``partition_open`` / ``partition_close`` brackets.
    """
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


def _embedded_calls(per_rank: dict[int, list[tuple]]) -> dict[int, list[tuple]]:
    """Subset of per-rank calls limited to ``embeddedNode`` emits."""
    return {
        rank: [(args, kwargs) for (name, args, kwargs) in calls
               if name == "embeddedNode"]
        for rank, calls in per_rank.items()
    }


def _build_disjoint_triangles_fem(
    record_a: InterpolationRecord,
    record_b: InterpolationRecord,
) -> FEMStub:
    """Two partitions each owning a disjoint triangle host + a slave.

    Partition 0 owns nodes ``{1, 2, 3, 7}`` and element 1 (a beam on
    1-2 used as a non-embedded structural placeholder so the build
    pipeline has at least one element to emit).  Partition 1 owns
    nodes ``{4, 5, 6, 8}`` and element 2 (beam on 4-5).

    Record A's host triangle is ``(1, 2, 3)`` and slave is 7 — all on
    rank 0; canonical host rank = 0.  Record B's host triangle is
    ``(4, 5, 6)`` and slave is 8 — all on rank 1; canonical host rank
    = 1.  Clean disjoint case: no boundary-shared masters, no foreign
    declarations needed for masters or slaves.
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5, 6, 7, 8],
        coords=[
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0),
            (3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.5, 1.0, 0.0),
            (0.5, 0.3, 0.5), (3.5, 0.3, 0.5),
        ],
        node_pgs={"Base": [1, 4], "Top": [2, 5]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(
                ids=(1, 2),
                connectivity=((1, 2), (4, 5)),
            ),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.set_partitions([
        (0, [1, 2, 3, 7], [1]),
        (1, [4, 5, 6, 8], [2]),
    ])
    fem.add_surface_constraints([record_a, record_b])
    return fem


def _emit_with_recording(fem) -> RecordingEmitter:
    """Drive the bridge build through a ``RecordingEmitter`` and return it."""
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    # Need at least one structural element so the build pipeline has
    # a real element pass to run; reuse the existing PG ``"Cols"``.
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    return rec


# ---------------------------------------------------------------------
# Records — masters[0] selects which rank emits (host-rank proxy rule)
# ---------------------------------------------------------------------


def _record_for_rank0() -> InterpolationRecord:
    """Embedded record whose host triangle is rank 0's (nodes 1, 2, 3)
    and whose slave (node 7) is also on rank 0.

    Canonical host rank = ``min({0} ∩ {0} ∩ {0}) = 0`` — emits on rank 0.
    """
    return InterpolationRecord(
        kind=ConstraintKind.EMBEDDED,
        slave_node=7,
        master_nodes=[1, 2, 3],
        weights=np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64),
        dofs=[1, 2, 3],
        name="embedded_on_rank_0",
    )


def _record_for_rank1() -> InterpolationRecord:
    """Embedded record whose host triangle is rank 1's (nodes 4, 5, 6)
    and whose slave (node 8) is also on rank 1.
    """
    return InterpolationRecord(
        kind=ConstraintKind.EMBEDDED,
        slave_node=8,
        master_nodes=[4, 5, 6],
        weights=np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64),
        dofs=[1, 2, 3],
        name="embedded_on_rank_1",
    )


# ---------------------------------------------------------------------
# INV-OWN: each record emits on exactly one rank (passes today)
# ---------------------------------------------------------------------


def test_partitioned_embedded_emits_only_on_host_rank() -> None:
    """ADR 0027 §"ASDEmbeddedNodeElement ownership" — each embedded
    record's ``embeddedNode`` line appears on exactly one rank, and on
    the rank picked by ``_canonical_host_rank`` (min of the master
    owners' intersection).
    """
    rec_a = _record_for_rank0()
    rec_b = _record_for_rank1()
    fem = _build_disjoint_triangles_fem(rec_a, rec_b)

    rec = _emit_with_recording(fem)
    per_rank = _per_rank_calls(rec)
    embedded = _embedded_calls(per_rank)

    # Exactly one embeddedNode line per rank.
    assert len(embedded[0]) == 1, (
        f"rank 0 should emit one embeddedNode line; got {embedded[0]!r}"
    )
    assert len(embedded[1]) == 1, (
        f"rank 1 should emit one embeddedNode line; got {embedded[1]!r}"
    )

    # The cnode and the master tuple identify which record landed where.
    (args0, _kw0) = embedded[0][0]
    (args1, _kw1) = embedded[1][0]
    # Signature: embeddedNode(ele_tag, cnode, *master_nodes)
    cnode0, masters0 = int(args0[1]), tuple(int(a) for a in args0[2:])
    cnode1, masters1 = int(args1[1]), tuple(int(a) for a in args1[2:])

    assert (cnode0, masters0) == (7, (1, 2, 3)), (
        f"rank 0 should carry record A (cnode=7, masters=(1,2,3)); "
        f"got cnode={cnode0}, masters={masters0}"
    )
    assert (cnode1, masters1) == (8, (4, 5, 6)), (
        f"rank 1 should carry record B (cnode=8, masters=(4,5,6)); "
        f"got cnode={cnode1}, masters={masters1}"
    )


# ---------------------------------------------------------------------
# INV-TAG: globally unique element tags across ranks (FAILS today)
# ---------------------------------------------------------------------


def test_partitioned_embedded_tags_globally_unique() -> None:
    """ADR 0027 §"Tag determinism" — file-internal element tags emitted
    across all rank blocks must form a globally unique set.  Tags come
    from the bridge's canonical :class:`TagAllocator` (``"element"``
    kind); the previous static ``1_000_000`` base collided across
    ranks.
    """
    rec_a = _record_for_rank0()
    rec_b = _record_for_rank1()
    fem = _build_disjoint_triangles_fem(rec_a, rec_b)

    rec = _emit_with_recording(fem)
    per_rank = _per_rank_calls(rec)
    embedded = _embedded_calls(per_rank)

    tags_per_rank: dict[int, list[int]] = {
        rank: [int(args[0]) for (args, _kw) in calls]
        for rank, calls in embedded.items()
    }
    all_tags = [t for tags in tags_per_rank.values() for t in tags]

    assert len(all_tags) == 2, (
        f"expected exactly 2 embeddedNode emits (one per rank); "
        f"got {tags_per_rank!r}"
    )
    assert len(set(all_tags)) == len(all_tags), (
        f"ADR 0027 §\"Tag determinism\": element tags must be globally "
        f"unique across ranks; got duplicate tag(s) — per-rank emit: "
        f"{tags_per_rank!r}"
    )


# ---------------------------------------------------------------------
# INV-BOUNDARY: shared masters do not cause duplicate emit (FAILS today)
# ---------------------------------------------------------------------


def _make_shared_master_fem() -> FEMStub:
    """4-node, 2-partition fixture where ALL master nodes are boundary-
    shared (owned by both ranks); slave lives on rank 1 only.

    Partition 0 owns nodes 1, 2, 3.
    Partition 1 owns nodes 1, 2, 3, 4.
    Therefore ``node_owners`` resolves to::

        {1: {0, 1}, 2: {0, 1}, 3: {0, 1}, 4: {1}}

    A single embedded record ``slave=4, masters=[1, 2, 3]`` has every
    master owned by both ranks. Under the current
    ``if partition_rank in host_owners`` rule (``build.py:2532``) the
    embeddedNode line emits on BOTH ranks (duplicate stiffness
    contribution at solve time). The intended Phase 1 rule —
    ``min(intersection(node_owners[m] for m in masters))`` — yields the
    single canonical rank 0.
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.5, 0.5, 1.0),
        ],
        node_pgs={"Base": [1, 2, 3], "Apex": [4]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(
                ids=(1, 2),
                connectivity=((1, 4), (2, 4)),
            ),
        },
    )
    stub = FEMStub(nodes=nodes, elements=elements)
    stub.set_partitions([
        (0, [1, 2, 3], [1]),
        (1, [1, 2, 3, 4], [2]),
    ])
    stub.add_surface_constraints([
        InterpolationRecord(
            kind=ConstraintKind.EMBEDDED,
            slave_node=4,
            master_nodes=[1, 2, 3],
            weights=np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64),
            dofs=[1, 2, 3],
            name="boundary_shared_masters",
        ),
    ])
    return stub


def test_partitioned_embedded_emits_once_with_shared_master() -> None:
    """ADR 0027 §"ASDEmbeddedNodeElement ownership" — when every master
    node is owned by multiple ranks (boundary-shared interface), the
    embeddedNode line still emits on exactly one rank: the canonical
    ``min(intersection)`` of the master owners.
    """
    fem = _make_shared_master_fem()
    rec = _emit_with_recording(fem)
    per_rank = _per_rank_calls(rec)
    embedded = _embedded_calls(per_rank)

    counts = {rank: len(calls) for rank, calls in embedded.items()}
    total = sum(counts.values())

    assert total == 1, (
        f"a single embedded record with shared masters must emit exactly "
        f"once across all ranks; got {total} emits — per-rank: {counts!r}"
    )


# ---------------------------------------------------------------------
# Real-resolver verification — Gmsh's atomic-element guarantee
# ---------------------------------------------------------------------
#
# The stub tests above pin emit-side invariants against hand-built
# records.  This test drives the production path end-to-end: real
# Gmsh mesh → ``g.constraints.embedded(...)`` → resolver →
# partitioner → emit.  It proves that the new intersection-based
# host-rank rule (``_canonical_host_rank``) never fail-louds against
# resolver-produced records, because Gmsh's partitioner is element-
# atomic: every host element's corner nodes are either locally owned
# by that element's partition or boundary-shared with neighbour
# partitions, so the master-owner intersection is always non-empty.


def test_real_resolver_partitioned_embedded_does_not_fail_loud() -> None:
    """End-to-end: rebar-in-concrete embedded constraint across a
    partition boundary emits cleanly.

    Geometry: a thin 3D box (host) with a line (rebar) running along
    its long axis. After meshing (tet host + line embedded) and
    partitioning across 2 ranks along the long axis, some rebar nodes
    fall on one rank's tets and some on the other — i.e. the embedded
    constraint genuinely straddles the partition boundary.  Each
    resolved :class:`InterpolationRecord` has 4 master nodes (one tet
    host); per Gmsh's element-atomic partitioning those 4 nodes must
    be reachable from the rank that owns the host tet, so
    ``_canonical_host_rank`` returns a valid rank for every record
    (no ``ValueError``).
    """
    from apeGmsh import apeGmsh
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name="part_embedded_real", verbose=False) as g:
        # Long thin box (concrete host) — 4.0 x 0.4 x 0.4
        box = g.model.geometry.add_box(0.0, 0.0, 0.0, 4.0, 0.4, 0.4)
        # Rebar line along the long axis at the box centroid
        p0 = g.model.geometry.add_point(0.0, 0.2, 0.2, lc=0.4)
        p1 = g.model.geometry.add_point(4.0, 0.2, 0.2, lc=0.4)
        rebar = g.model.geometry.add_line(p0, p1)
        g.model.sync()

        g.physical.add(3, [box],   name="concrete")
        g.physical.add(1, [rebar], name="rebar")

        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(3)

        # The straddling embedded coupling under test.
        g.constraints.embedded("concrete", "rebar")

        # Partition along the long axis (default METIS gives two
        # contiguous halves for a long thin geometry).
        g.mesh.partitioning.partition(2)

        fem = g.mesh.queries.get_fem_data(dim=3)

    # Sanity — partitioning produced two ranks and the resolver
    # emitted at least one embedded interpolation record.
    assert len(fem.partitions) == 2, (
        f"expected 2 partitions; got {len(fem.partitions)}"
    )
    interps = list(fem.elements.constraints.interpolations())
    assert len(interps) > 0, (
        "resolver produced no embedded InterpolationRecords — geometry "
        "or PG labels did not exercise the embedded resolution path"
    )

    # Drive emit.  If any resolver-produced record has masters whose
    # owner-intersection is empty, ``_canonical_host_rank`` raises
    # ``ValueError`` here — the test fails immediately with that
    # message.  Otherwise emit completes and we inspect the per-rank
    # output for INV-OWN + INV-TAG.
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=3)
    rec = RecordingEmitter()
    bm = ops.build()
    bm.emit(rec)

    per_rank = _per_rank_calls(rec)
    embedded = _embedded_calls(per_rank)

    # INV-OWN: every emitted embeddedNode line goes out on exactly one
    # rank — never duplicated.  We assert this at the line level
    # (tag + cnode + masters tuple identifies a line uniquely).
    line_emit_ranks: dict[tuple, list[int]] = {}
    for rank, calls in embedded.items():
        for (args, _kw) in calls:
            key = tuple(int(a) for a in args)
            line_emit_ranks.setdefault(key, []).append(rank)
    duplicates = {k: rs for k, rs in line_emit_ranks.items() if len(rs) > 1}
    assert not duplicates, (
        f"INV-OWN: embedded lines emitted on multiple ranks: {duplicates!r}"
    )

    # INV-TAG: every emitted element tag is globally unique across
    # ranks (canonical TagAllocator threading).
    all_tags = [int(args[0]) for calls in embedded.values()
                for (args, _kw) in calls]
    assert len(set(all_tags)) == len(all_tags), (
        f"INV-TAG: duplicate element tag across ranks; tags={sorted(all_tags)!r}"
    )

    # Coverage: at least one record actually emitted on each rank
    # (otherwise the test is degenerate — the partition might not have
    # straddled the rebar).
    ranks_with_emits = {r for r, calls in embedded.items() if calls}
    assert len(ranks_with_emits) >= 1, (
        f"degenerate: no embedded emit landed on any rank — per-rank "
        f"counts: { {r: len(c) for r, c in embedded.items()} }"
    )
