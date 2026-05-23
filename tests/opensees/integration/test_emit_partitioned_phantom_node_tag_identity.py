"""Integration test for ADR 0027 INV-3 — phantom-node tag identity.

Phantom-node tags are broker-derived (one canonical numbering at build
time). Under partitioning the same phantom tag and the same coordinates
must appear on EVERY rank that hosts a constraint referencing the
phantom, with the same ``ndf=6`` override.
"""
from __future__ import annotations

from typing import cast

import numpy as np

from apeGmsh._kernel.records._constraints import (
    NodePairRecord,
    NodeToSurfaceRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
    FEMStub,
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


def _make_node_to_surface_fixture() -> FEMStub:
    """Two ranks; one node_to_surface record straddling both.

    Geometry: 4 nodes on a horizontal plane (a tiny tile) with a
    floating master node above. The master is on rank 0; the surface
    slaves span ranks 0 and 1; the phantom node sits at the master.
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5],
        coords=[
            (0.0, 0.0, 0.0),   # node 1 — corner of surface, rank 0
            (1.0, 0.0, 0.0),   # node 2 — corner of surface, rank 0
            (1.0, 1.0, 0.0),   # node 3 — corner of surface, rank 1
            (0.0, 1.0, 0.0),   # node 4 — corner of surface, rank 1
            (0.5, 0.5, 0.5),   # node 5 — master, rank 0
        ],
        node_pgs={
            "Master": [5],
            "Surface": [1, 2, 3, 4],
        },
    )
    elements = _ElementsStub(
        elem_pgs={
            "Plate": _ElementGroupView(
                ids=(1,),
                connectivity=((1, 2, 3, 4),),
            ),
        },
    )
    stub = FEMStub(nodes=nodes, elements=elements)
    stub.set_partitions([
        (0, [1, 2, 5], [1]),
        (1, [3, 4], []),
    ])
    # Synthesize a NodeToSurfaceRecord whose phantom tag is 999 — well
    # outside the bridge's element/node tag range so collisions are
    # impossible. We hand-build the compound; real broker would compute
    # rigid_link_records / equal_dof_records from the resolver chain.
    phantom_tag = 999
    phantom_coord = np.array([[0.5, 0.5, 0.5]])
    n2s = NodeToSurfaceRecord(
        master_node=5,
        slave_nodes=[1, 2, 3, 4],
        phantom_nodes=[phantom_tag],
        phantom_coords=phantom_coord,
        rigid_link_records=[
            NodePairRecord(
                kind=ConstraintKind.RIGID_BEAM,
                master_node=5,
                slave_node=phantom_tag,
                dofs=[1, 2, 3, 4, 5, 6],
                name="n2s_link",
            ),
        ],
        equal_dof_records=[
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=phantom_tag,
                slave_node=1,
                dofs=[1, 2, 3],
                name="n2s_ed_1",
            ),
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=phantom_tag,
                slave_node=2,
                dofs=[1, 2, 3],
                name="n2s_ed_2",
            ),
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=phantom_tag,
                slave_node=3,
                dofs=[1, 2, 3],
                name="n2s_ed_3",
            ),
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=phantom_tag,
                slave_node=4,
                dofs=[1, 2, 3],
                name="n2s_ed_4",
            ),
        ],
        kind=ConstraintKind.NODE_TO_SURFACE,
        dofs=[1, 2, 3],
        name="bind",
    )
    stub.add_node_constraints([n2s])
    return stub


def test_phantom_tag_identical_across_ranks() -> None:
    """Phantom node 999 emits with the SAME tag and SAME coords on
    every rank that references it (INV-3).
    """
    fem = _make_node_to_surface_fixture()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=3)

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)

    # Both ranks should declare phantom 999 with the same coords + ndf=6.
    def _phantom_decls(rank: int) -> list[tuple]:
        out = []
        for name, args, kwargs in per_rank[rank]:
            if (
                name == "node"
                and len(args) >= 1
                and int(args[0]) == 999
                and kwargs.get("ndf") == 6
            ):
                out.append((args, kwargs))
        return out

    p0 = _phantom_decls(0)
    p1 = _phantom_decls(1)
    assert len(p0) == 1, f"rank 0 must declare phantom 999 exactly once: {p0!r}"
    assert len(p1) == 1, f"rank 1 must declare phantom 999 exactly once: {p1!r}"
    # INV-3: tags and coords identical.
    assert p0 == p1, (
        f"INV-3: phantom-node declaration must be byte-identical across ranks; "
        f"rank 0={p0!r}, rank 1={p1!r}"
    )
