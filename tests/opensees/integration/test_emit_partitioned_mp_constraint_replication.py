"""Integration test for ADR 0027 cross-partition MP-constraint replication.

Per ADR 0027 §"Decision":

* ``equalDOF(master, slave, *dofs)`` and ``rigidLink(...)`` straddling
  ranks emit on BOTH owning ranks, with the foreign node declared
  via ``node(tag, *xyz, ndf=...)`` BEFORE the constraint line on the
  non-owning side (INV-1 + INV-2).
* ``rigidDiaphragm(perp_dir, master, *slaves)`` emits on every rank
  that owns any slave; the master is foreign on all-but-one rank.
"""
from __future__ import annotations

from typing import cast

from apeGmsh._kernel.records._constraints import NodeGroupRecord, NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees._helpers.partition_diff import (
    assert_partition_blocks_equivalent,
)
from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
)


def _per_rank_calls(rec: RecordingEmitter) -> dict[int, list[tuple]]:
    """Group ``rec.calls`` into per-rank lists between partition_open/close.

    Returns ``{rank: [(method_name, args, kwargs), ...]}`` covering
    every emit call inside a ``partition_open(K)`` /
    ``partition_close()`` bracket. Calls emitted outside any bracket
    are NOT included.
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


def test_rigid_link_replicates_on_both_ranks_with_foreign_node() -> None:
    """A rigid_beam between nodes 2 (rank 0) and 4 (rank 1) emits on
    BOTH ranks; the foreign node is declared before the rigidLink line
    on each rank (ADR 0027 INV-1, INV-2).
    """
    fem = make_two_column_frame_partitioned()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.RIGID_BEAM,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3, 4, 5, 6],
            name="cross_link",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)

    # Both ranks see exactly one rigidLink emit with the same args.
    rl0 = [(a, k) for (n, a, k) in per_rank[0] if n == "rigidLink"]
    rl1 = [(a, k) for (n, a, k) in per_rank[1] if n == "rigidLink"]
    assert len(rl0) == 1, f"rank 0 must emit rigidLink once; got {rl0!r}"
    assert len(rl1) == 1, f"rank 1 must emit rigidLink once; got {rl1!r}"
    assert rl0 == rl1, "INV-1: cross-partition constraint text must match across ranks"

    # The foreign node is declared via node(tag, *xyz) BEFORE the
    # rigidLink line on each rank (INV-2). Find indices.
    def _first_node_decl_index(rank: int, tag: int) -> int:
        """First ``node(tag, ...)`` emit at the given tag, or -1."""
        for idx, (name, args, _kw) in enumerate(per_rank[rank]):
            if name == "node" and len(args) >= 1 and int(args[0]) == int(tag):
                return idx
        return -1

    def _first_method_index(rank: int, method: str) -> int:
        for idx, (name, _a, _kw) in enumerate(per_rank[rank]):
            if name == method:
                return idx
        return -1

    # Rank 0 natively owns node 2; node 4 is foreign and must be declared.
    foreign_decl_rank0 = _first_node_decl_index(0, 4)
    rigid_link_rank0 = _first_method_index(0, "rigidLink")
    assert foreign_decl_rank0 != -1, "rank 0 must declare foreign node 4"
    assert foreign_decl_rank0 < rigid_link_rank0, (
        f"INV-2: foreign node 4 must be declared BEFORE the rigidLink on rank 0; "
        f"node-decl at index {foreign_decl_rank0}, rigidLink at index {rigid_link_rank0}"
    )

    # Rank 1 natively owns node 4; node 2 is foreign.
    foreign_decl_rank1 = _first_node_decl_index(1, 2)
    rigid_link_rank1 = _first_method_index(1, "rigidLink")
    assert foreign_decl_rank1 != -1, "rank 1 must declare foreign node 2"
    assert foreign_decl_rank1 < rigid_link_rank1, (
        f"INV-2: foreign node 2 must be declared BEFORE the rigidLink on rank 1"
    )


def test_rigid_diaphragm_replicates_on_every_slave_owning_rank() -> None:
    """A rigid_diaphragm with master on rank 0 and one slave on rank 1
    emits the FULL rigidDiaphragm line (master + all slaves) on BOTH
    ranks (ADR 0027 §"Decision" bullet 3).
    """
    fem = make_two_column_frame_partitioned()
    fem.add_node_constraints([
        NodeGroupRecord(
            kind=ConstraintKind.RIGID_DIAPHRAGM,
            master_node=2,
            slave_nodes=[4],
            plane_normal=(0.0, 0.0, 1.0),
            dofs=None,
            name="floor",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)

    # Both ranks emit the rigidDiaphragm line; the args are identical.
    rd0 = [(a, k) for (n, a, k) in per_rank[0] if n == "rigidDiaphragm"]
    rd1 = [(a, k) for (n, a, k) in per_rank[1] if n == "rigidDiaphragm"]
    assert len(rd0) == 1
    assert len(rd1) == 1
    assert rd0 == rd1, (
        "INV-1: rigidDiaphragm emits byte-identical text on every owning rank"
    )


def test_tcl_text_byte_identical_across_ranks_for_cross_partition() -> None:
    """The Tcl text for a cross-partition constraint is identical between
    the two rank blocks (INV-1).
    """
    fem = make_two_column_frame_partitioned()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="ed",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    emitter = TclEmitter()
    bm.emit(emitter)
    text = "\n".join(emitter.lines())

    # The equalDOF line should appear twice — once per rank block.
    equal_dof_lines = [
        line.strip() for line in text.split("\n") if "equalDOF" in line
    ]
    assert len(equal_dof_lines) == 2, (
        f"equalDOF must appear once per rank under cross-partition replication; "
        f"got {equal_dof_lines!r}"
    )
    # INV-1: cross-partition constraint text must be byte-identical
    # modulo foreign-node decl prefixes.  The extracted equalDOF lines
    # carry no node decls themselves, so the helper degenerates to a
    # straight byte-compare; routing through it keeps the assertion
    # uniform across all INV-1 tests.
    assert_partition_blocks_equivalent(
        equal_dof_lines[0], equal_dof_lines[1],
        label="equalDOF cross-partition",
    )
