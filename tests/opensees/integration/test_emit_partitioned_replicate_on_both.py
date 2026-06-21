"""E2E partitioned MP-constraint replication tests (ADR 0027).

Locks the **emit-and-parse-the-deck** shape of the four replicate-on-
both rules from ADR 0027 §"Decision", plus the phantom-node ordering
rule from §"Phantom-node policy" / INV-3.

The companion file ``test_emit_partitioned_mp_constraint_replication.py``
uses ``RecordingEmitter`` to capture the bridge's method calls — that
is a unit-style trace test.  This file goes one step further: it
drives the full Tcl + Py emit pipelines, writes the deck to a temp
file, splits the file by ``if {[getPID] == K}`` (Tcl) /
``if getPID() == K:`` (Py) brackets, and asserts the per-rank text
against the replication rule.

Each test follows the same E2E shape:

    1. Build a partitioned FEM (stub, since METIS partitioning of real
       Gmsh meshes is brittle for tiny cross-rank topologies; the stub
       is the same fixture pattern used by the existing partitioned
       integration tests under this directory).
    2. Declare the cross-rank MP constraint(s) under test.
    3. Drive ``apeSees(fem).tcl(path)`` + ``.py(path)`` into a
       ``tmp_path``.
    4. Parse the deck by rank and assert ADR 0027's replication rule.

Tests use ``assert_partition_blocks_equivalent`` from
``tests.opensees._helpers.partition_diff`` where ADR 0027 INV-1
mandates byte-equivalence of the constraint line across ranks
(modulo foreign-node decl prefixes — the helper's job).  Within-rank
ordering assertions (foreign-node decl precedes the constraint line)
use direct string-position checks instead, since the helper is for
cross-rank comparison only.
"""
from __future__ import annotations

import re
import warnings
from typing import cast

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees

from tests.opensees._helpers.partition_diff import (
    assert_partition_blocks_equivalent,
)
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
    make_two_column_frame_partitioned,
)


# Silence ADR 0027 INV-5 auto-emit warnings (constraint handler /
# numberer / system) — they fire because none of these tests declare
# an MP-friendly chain.  The warnings are themselves contracted
# behavior locked by other tests; here they would mask the actual
# assertions if left to escalate under ``pytest -W error::UserWarning``.
_MP_AUTO_EMIT_FILTERS = (
    "ignore:MP constraints are present in the model:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


# ---------------------------------------------------------------------------
# Deck parsers — comment-preserving variants of the helpers from
# test_emit_partitioned_embedded.py.  The existing helpers strip
# ``#``-prefixed lines; we need the comments retained so the
# ``mp_constraint_comment`` test can find them.
# ---------------------------------------------------------------------------


_TCL_RANK_OPEN_RE = re.compile(r"^\s*if\s*\{\[getPID\]\s*==\s*(\d+)\}\s*\{\s*$")
_PY_RANK_OPEN_RE = re.compile(r"^if\s+getPID\(\)\s*==\s*(\d+):\s*$")


def _split_tcl_deck_lines(deck_text: str) -> dict[int, list[str]]:
    """Split a partitioned Tcl deck into ``{rank: [raw_line, ...]}``.

    Preserves all non-blank lines inside each ``if {[getPID] == K} {
    ... }`` block, including ``#``-comments.  Each returned line is
    ``.strip()``-ed so cross-rank comparisons aren't disturbed by the
    rank-indent (the block contents are indented 4 spaces inside each
    bracket — that's emit-formatting, not semantic content per ADR
    0027 INV-1).
    """
    out: dict[int, list[str]] = {}
    cur_rank: int | None = None
    cur_depth = 0
    for raw in deck_text.splitlines():
        if cur_rank is None:
            m = _TCL_RANK_OPEN_RE.match(raw)
            if m:
                cur_rank = int(m.group(1))
                cur_depth = 1
                out.setdefault(cur_rank, [])
            continue
        opens = raw.count("{")
        closes = raw.count("}")
        cur_depth += opens - closes
        if cur_depth <= 0:
            cur_rank = None
            cur_depth = 0
            continue
        body = raw.strip()
        if not body:
            continue
        out[cur_rank].append(body)
    return out


def _split_py_deck_lines(deck_text: str) -> dict[int, list[str]]:
    """Split a partitioned Py deck into ``{rank: [raw_line, ...]}``.

    Same comment-preserving guarantee as :func:`_split_tcl_deck_lines`.
    Python uses indentation, so the block runs until a non-blank line
    returns to column 0.  Each returned line is ``.strip()``-ed.
    """
    out: dict[int, list[str]] = {}
    cur_rank: int | None = None
    for raw in deck_text.splitlines():
        if cur_rank is None:
            m = _PY_RANK_OPEN_RE.match(raw)
            if m:
                cur_rank = int(m.group(1))
                out.setdefault(cur_rank, [])
            continue
        if raw.strip() == "":
            continue
        if not (raw.startswith(" ") or raw.startswith("\t")):
            cur_rank = None
            m = _PY_RANK_OPEN_RE.match(raw)
            if m:
                cur_rank = int(m.group(1))
                out.setdefault(cur_rank, [])
            continue
        out[cur_rank].append(raw.strip())
    return out


def _emit_decks(fem: FEMStub, tmp_path, *, ndf: int = 6) -> tuple[str, str]:
    """Drive ``apeSees(fem)`` through both ``.tcl`` and ``.py``;
    return ``(tcl_text, py_text)``.
    """
    transf_pg = "Cols" if "Cols" in fem.elements._pgs else None

    def _build(side: str) -> None:
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=ndf)
        if transf_pg is not None:
            transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
            ops.element.elasticBeamColumn(
                pg=transf_pg, transf=transf,
                A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
            )
        if side == "tcl":
            ops.tcl(str(tmp_path / "deck.tcl"))
        else:
            ops.py(str(tmp_path / "deck.py"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _build("tcl")
        _build("py")

    tcl_text = (tmp_path / "deck.tcl").read_text(encoding="utf-8")
    py_text = (tmp_path / "deck.py").read_text(encoding="utf-8")
    return tcl_text, py_text


# ---------------------------------------------------------------------------
# Test 1 — equalDOF replicates on both owning ranks (E2E)
# ---------------------------------------------------------------------------


def test_cross_rank_equalDOF_replicates_on_both_owner_ranks(tmp_path) -> None:
    """ADR 0027 §"Decision" bullet 1 — ``equalDOF(master, slave, *dofs)``
    straddling ranks emits on BOTH owning ranks.  Each rank declares
    the foreign node via ``node(tag, *xyz, ndf=K)`` BEFORE the
    ``equalDOF`` line; the native-side rank does NOT redeclare its
    own node; the two ``equalDOF`` lines are byte-identical (INV-1).

    Fixture: ``make_two_column_frame_partitioned`` — nodes 1,2 on
    rank 0, nodes 3,4 on rank 1; one ``equalDOF(2, 4, 1, 2, 3)``
    crosses the partition.
    """
    fem = make_two_column_frame_partitioned()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="cross_equal_dof",
        ),
    ])

    tcl_text, py_text = _emit_decks(fem, tmp_path, ndf=6)
    tcl_by_rank = _split_tcl_deck_lines(tcl_text)
    py_by_rank = _split_py_deck_lines(py_text)

    assert set(tcl_by_rank.keys()) == {0, 1}, (
        f"Tcl deck should expose both ranks; got {sorted(tcl_by_rank)}"
    )
    assert set(py_by_rank.keys()) == {0, 1}, (
        f"Py deck should expose both ranks; got {sorted(py_by_rank)}"
    )

    # ----- Tcl side --------------------------------------------------------
    # equalDOF emits exactly once per rank.
    tcl_ed_lines_per_rank = {
        rank: [ln for ln in lines if ln.startswith("equalDOF")]
        for rank, lines in tcl_by_rank.items()
    }
    assert len(tcl_ed_lines_per_rank[0]) == 1, tcl_ed_lines_per_rank
    assert len(tcl_ed_lines_per_rank[1]) == 1, tcl_ed_lines_per_rank

    # INV-1: the two equalDOF lines are byte-identical
    # (the line is bare — no foreign-node decl prefix inline).
    assert_partition_blocks_equivalent(
        tcl_ed_lines_per_rank[0][0], tcl_ed_lines_per_rank[1][0],
        label="Tcl equalDOF cross-rank",
    )

    # INV-2: foreign-node decl precedes the equalDOF line on each rank.
    # Rank 0 owns node 2 (native); node 4 is foreign. Both take the envelope
    # ndf, so the -ndf token is elided (ADR 0048 inference + elide-on-equal).
    # Rank 1 owns node 4 (native); node 2 is foreign.
    def _tcl_node_decl_index(lines: list[str], tag: int) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith(f"node {tag} "):
                return i
        return -1

    def _tcl_ed_index(lines: list[str]) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith("equalDOF"):
                return i
        return -1

    # Rank 0: foreign decl for node 4, native node 2 unchanged.
    r0 = tcl_by_rank[0]
    foreign_idx_r0 = _tcl_node_decl_index(r0, 4)
    ed_idx_r0 = _tcl_ed_index(r0)
    assert foreign_idx_r0 != -1, (
        f"rank 0 must declare foreign node 4 before equalDOF; "
        f"lines={r0!r}"
    )
    assert foreign_idx_r0 < ed_idx_r0, (
        f"INV-2: foreign node 4 must precede equalDOF on rank 0 "
        f"(decl at {foreign_idx_r0}, ed at {ed_idx_r0})"
    )
    # Uniform ndf-6 model: foreign + native both elide -ndf (envelope wins).
    assert "-ndf" not in r0[foreign_idx_r0]
    native_idx_r0 = _tcl_node_decl_index(r0, 2)
    assert native_idx_r0 != -1, (
        f"rank 0 must declare its own node 2; lines={r0!r}"
    )
    assert "-ndf" not in r0[native_idx_r0], (
        f"native node 2 on rank 0 must NOT carry -ndf "
        f"(envelope ndf applies); got {r0[native_idx_r0]!r}"
    )

    # Rank 1: foreign decl for node 2, native node 4 unchanged.
    r1 = tcl_by_rank[1]
    foreign_idx_r1 = _tcl_node_decl_index(r1, 2)
    ed_idx_r1 = _tcl_ed_index(r1)
    assert foreign_idx_r1 != -1, (
        f"rank 1 must declare foreign node 2 before equalDOF; lines={r1!r}"
    )
    assert foreign_idx_r1 < ed_idx_r1
    assert "-ndf" not in r1[foreign_idx_r1]
    native_idx_r1 = _tcl_node_decl_index(r1, 4)
    assert native_idx_r1 != -1
    assert "-ndf" not in r1[native_idx_r1]

    # ----- Py side ---------------------------------------------------------
    py_ed_lines_per_rank = {
        rank: [ln for ln in lines if ln.startswith("ops.equalDOF(")]
        for rank, lines in py_by_rank.items()
    }
    assert len(py_ed_lines_per_rank[0]) == 1, py_ed_lines_per_rank
    assert len(py_ed_lines_per_rank[1]) == 1, py_ed_lines_per_rank
    assert_partition_blocks_equivalent(
        py_ed_lines_per_rank[0][0], py_ed_lines_per_rank[1][0],
        label="Py equalDOF cross-rank",
    )

    def _py_node_decl_index(lines: list[str], tag: int) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith(f"ops.node({tag},"):
                return i
        return -1

    def _py_ed_index(lines: list[str]) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith("ops.equalDOF("):
                return i
        return -1

    pr0 = py_by_rank[0]
    pr1 = py_by_rank[1]
    f0 = _py_node_decl_index(pr0, 4)
    e0 = _py_ed_index(pr0)
    assert 0 <= f0 < e0
    assert "'-ndf'" not in pr0[f0]
    n0 = _py_node_decl_index(pr0, 2)
    assert n0 != -1 and "'-ndf'" not in pr0[n0]

    f1 = _py_node_decl_index(pr1, 2)
    e1 = _py_ed_index(pr1)
    assert 0 <= f1 < e1
    assert "'-ndf'" not in pr1[f1]
    n1 = _py_node_decl_index(pr1, 4)
    assert n1 != -1 and "'-ndf'" not in pr1[n1]


# ---------------------------------------------------------------------------
# Test 2 — rigidLink replicates on both owning ranks (E2E)
# ---------------------------------------------------------------------------


def test_cross_rank_rigidLink_replicates_on_both_owner_ranks(tmp_path) -> None:
    """ADR 0027 §"Decision" bullet 2 — ``rigidLink(kind, master, slave)``
    straddling ranks emits on BOTH owning ranks; same replication rule
    as equalDOF.  Foreign node declared before the rigidLink line.

    The api-level entry is ``g.constraints.rigid_link(...)`` (composite
    API) which resolves to a :class:`NodePairRecord` with kind
    ``RIGID_BEAM``.  We construct the record directly so the test
    skips the resolver and locks the emit side.
    """
    fem = make_two_column_frame_partitioned()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.RIGID_BEAM,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3, 4, 5, 6],
            name="cross_rigid_link",
        ),
    ])

    tcl_text, py_text = _emit_decks(fem, tmp_path, ndf=6)
    tcl_by_rank = _split_tcl_deck_lines(tcl_text)
    py_by_rank = _split_py_deck_lines(py_text)

    # ----- Tcl side --------------------------------------------------------
    tcl_rl_per_rank = {
        rank: [ln for ln in lines if ln.startswith("rigidLink ")]
        for rank, lines in tcl_by_rank.items()
    }
    assert len(tcl_rl_per_rank[0]) == 1, tcl_rl_per_rank
    assert len(tcl_rl_per_rank[1]) == 1, tcl_rl_per_rank
    assert_partition_blocks_equivalent(
        tcl_rl_per_rank[0][0], tcl_rl_per_rank[1][0],
        label="Tcl rigidLink cross-rank",
    )

    # Foreign node precedes rigidLink on each rank.
    def _idx_line_startswith(lines: list[str], prefix: str) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith(prefix):
                return i
        return -1

    r0 = tcl_by_rank[0]
    f0 = _idx_line_startswith(r0, "node 4 ")
    rl0 = _idx_line_startswith(r0, "rigidLink ")
    assert 0 <= f0 < rl0, f"INV-2: rank 0 foreign decl/rigidLink: {r0!r}"
    assert "-ndf" not in r0[f0]

    r1 = tcl_by_rank[1]
    f1 = _idx_line_startswith(r1, "node 2 ")
    rl1 = _idx_line_startswith(r1, "rigidLink ")
    assert 0 <= f1 < rl1, f"INV-2: rank 1 foreign decl/rigidLink: {r1!r}"
    assert "-ndf" not in r1[f1]

    # ----- Py side ---------------------------------------------------------
    py_rl_per_rank = {
        rank: [ln for ln in lines if ln.startswith("ops.rigidLink(")]
        for rank, lines in py_by_rank.items()
    }
    assert len(py_rl_per_rank[0]) == 1, py_rl_per_rank
    assert len(py_rl_per_rank[1]) == 1, py_rl_per_rank
    assert_partition_blocks_equivalent(
        py_rl_per_rank[0][0], py_rl_per_rank[1][0],
        label="Py rigidLink cross-rank",
    )

    pr0 = py_by_rank[0]
    pf0 = _idx_line_startswith(pr0, "ops.node(4,")
    prl0 = _idx_line_startswith(pr0, "ops.rigidLink(")
    assert 0 <= pf0 < prl0
    assert "'-ndf'" not in pr0[pf0]

    pr1 = py_by_rank[1]
    pf1 = _idx_line_startswith(pr1, "ops.node(2,")
    prl1 = _idx_line_startswith(pr1, "ops.rigidLink(")
    assert 0 <= pf1 < prl1
    assert "'-ndf'" not in pr1[pf1]


# ---------------------------------------------------------------------------
# Test 3 — rigidDiaphragm replicates on every slave-owning rank (E2E)
# ---------------------------------------------------------------------------


def _make_three_rank_diaphragm_fem() -> FEMStub:
    """3-rank floor-diaphragm fixture.

    Geometry:
        * nodes 1-2: column 1 at x=0 (rank 0)
        * nodes 3-4: column 2 at x=3 (rank 1)
        * nodes 5-6: column 3 at x=6 (rank 2)
        * node 7   : master at (3, 3, 3) (rank 0)
        * node 8   : ancillary base node (rank 0)

    Slaves [2, 4, 6] live on ranks 0, 1, 2 respectively — every rank
    owns at least one slave, so the diaphragm must emit on every rank.
    The master (node 7) is rank 0 only; on ranks 1 and 2 it's foreign.

    Manual ``PartitionRecord`` assignment is used here so the cross-
    rank topology is deterministic — METIS would also produce a
    column-stripe partition here, but the manual assignment removes
    any flakiness.  Per ADR 0027 §"Tag determinism" this is purely a
    labeling pass — tag values are unaffected.
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5, 6, 7, 8],
        coords=[
            (0.0, 0.0, 0.0), (0.0, 0.0, 3.0),
            (3.0, 0.0, 0.0), (3.0, 0.0, 3.0),
            (6.0, 0.0, 0.0), (6.0, 0.0, 3.0),
            (3.0, 3.0, 3.0), (3.0, 3.0, 0.0),
        ],
        node_pgs={
            "Base": [1, 3, 5, 8],
            "Top": [2, 4, 6],
            "Master": [7],
        },
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(
                ids=(1, 2, 3),
                connectivity=((1, 2), (3, 4), (5, 6)),
            ),
        },
    )
    stub = FEMStub(nodes=nodes, elements=elements)
    stub.set_partitions([
        (0, [1, 2, 7, 8], [1]),
        (1, [3, 4], [2]),
        (2, [5, 6], [3]),
    ])
    return stub


def test_cross_rank_rigidDiaphragm_replicates_on_all_slave_owner_ranks(
    tmp_path,
) -> None:
    """ADR 0027 §"Decision" bullet 3 — ``rigidDiaphragm(perp_dir,
    master, *slaves)`` emits on EVERY rank that owns at least one
    slave.  The full slave list is replicated identically (not
    sharded).  The master is foreign on all-but-one rank.

    Three ranks; slaves on every rank — the full ``rigidDiaphragm 3 7
    2 4 6`` line emits identically on all three.
    """
    fem = _make_three_rank_diaphragm_fem()
    fem.add_node_constraints([
        NodeGroupRecord(
            kind=ConstraintKind.RIGID_DIAPHRAGM,
            master_node=7,
            slave_nodes=[2, 4, 6],
            plane_normal=(0.0, 0.0, 1.0),
            dofs=None,
            name="floor_3",
        ),
    ])

    tcl_text, py_text = _emit_decks(fem, tmp_path, ndf=6)
    tcl_by_rank = _split_tcl_deck_lines(tcl_text)
    py_by_rank = _split_py_deck_lines(py_text)

    assert set(tcl_by_rank.keys()) == {0, 1, 2}, (
        f"Tcl deck should expose 3 ranks; got {sorted(tcl_by_rank)}"
    )
    assert set(py_by_rank.keys()) == {0, 1, 2}, (
        f"Py deck should expose 3 ranks; got {sorted(py_by_rank)}"
    )

    # ----- Tcl side --------------------------------------------------------
    tcl_rd_per_rank = {
        rank: [ln for ln in lines if ln.startswith("rigidDiaphragm ")]
        for rank, lines in tcl_by_rank.items()
    }
    for rank in (0, 1, 2):
        assert len(tcl_rd_per_rank[rank]) == 1, (
            f"rank {rank} must emit rigidDiaphragm exactly once; "
            f"got {tcl_rd_per_rank[rank]!r}"
        )

    # INV-1: full slave list byte-identical across all 3 ranks.
    # Pairwise comparison gives clearer diff messages than 3-way.
    assert_partition_blocks_equivalent(
        tcl_rd_per_rank[0][0], tcl_rd_per_rank[1][0],
        label="Tcl rigidDiaphragm floor_3 rank0-vs-rank1",
    )
    assert_partition_blocks_equivalent(
        tcl_rd_per_rank[1][0], tcl_rd_per_rank[2][0],
        label="Tcl rigidDiaphragm floor_3 rank1-vs-rank2",
    )

    # The slave list is NOT sharded: every rank's rigidDiaphragm line
    # carries all three slaves 2, 4, 6.
    for rank, lines in tcl_rd_per_rank.items():
        ln = lines[0]
        # rigidDiaphragm <perp> <master> <s1> <s2> <s3>
        toks = ln.split()
        assert toks[0] == "rigidDiaphragm"
        assert int(toks[2]) == 7, f"rank {rank}: master tag != 7: {ln!r}"
        slaves_in_line = sorted(int(t) for t in toks[3:])
        assert slaves_in_line == [2, 4, 6], (
            f"rank {rank}: slave list must be the full [2,4,6] "
            f"(no sharding); got {slaves_in_line!r}"
        )

    # INV-2: foreign master 7 declared before rigidDiaphragm on ranks 1 and 2;
    # rank 0 owns master natively (no -ndf prefix on its node-7 decl).
    def _idx_line_startswith(lines: list[str], prefix: str) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith(prefix):
                return i
        return -1

    # Rank 0: native master, no -ndf.
    r0 = tcl_by_rank[0]
    n7_r0 = _idx_line_startswith(r0, "node 7 ")
    rd_r0 = _idx_line_startswith(r0, "rigidDiaphragm ")
    assert n7_r0 != -1
    assert "-ndf" not in r0[n7_r0], (
        f"rank 0 owns master node 7; must not carry -ndf: {r0[n7_r0]!r}"
    )
    # Foreign slaves 4 and 6 declared before rigidDiaphragm on rank 0.
    n4_r0 = _idx_line_startswith(r0, "node 4 ")
    n6_r0 = _idx_line_startswith(r0, "node 6 ")
    assert n4_r0 != -1 and "-ndf" not in r0[n4_r0]
    assert n6_r0 != -1 and "-ndf" not in r0[n6_r0]
    assert max(n4_r0, n6_r0) < rd_r0

    # Rank 1: foreign master + foreign slave 6.
    r1 = tcl_by_rank[1]
    n7_r1 = _idx_line_startswith(r1, "node 7 ")
    rd_r1 = _idx_line_startswith(r1, "rigidDiaphragm ")
    assert n7_r1 != -1 and "-ndf" not in r1[n7_r1], (
        f"rank 1: foreign master 7 takes the envelope ndf (=6), so its "
        f"-ndf token is elided (ADR 0048 inference + elide-on-equal): {r1!r}"
    )
    n2_r1 = _idx_line_startswith(r1, "node 2 ")
    n6_r1 = _idx_line_startswith(r1, "node 6 ")
    assert n2_r1 != -1 and "-ndf" not in r1[n2_r1]
    assert n6_r1 != -1 and "-ndf" not in r1[n6_r1]
    assert max(n7_r1, n2_r1, n6_r1) < rd_r1

    # Rank 2: foreign master + foreign slaves 2 and 4.
    r2 = tcl_by_rank[2]
    n7_r2 = _idx_line_startswith(r2, "node 7 ")
    rd_r2 = _idx_line_startswith(r2, "rigidDiaphragm ")
    assert n7_r2 != -1 and "-ndf" not in r2[n7_r2]
    n2_r2 = _idx_line_startswith(r2, "node 2 ")
    n4_r2 = _idx_line_startswith(r2, "node 4 ")
    assert n2_r2 != -1 and "-ndf" not in r2[n2_r2]
    assert n4_r2 != -1 and "-ndf" not in r2[n4_r2]
    assert max(n7_r2, n2_r2, n4_r2) < rd_r2

    # ----- Py side ---------------------------------------------------------
    py_rd_per_rank = {
        rank: [ln for ln in lines if ln.startswith("ops.rigidDiaphragm(")]
        for rank, lines in py_by_rank.items()
    }
    for rank in (0, 1, 2):
        assert len(py_rd_per_rank[rank]) == 1, py_rd_per_rank[rank]
    assert_partition_blocks_equivalent(
        py_rd_per_rank[0][0], py_rd_per_rank[1][0],
        label="Py rigidDiaphragm floor_3 rank0-vs-rank1",
    )
    assert_partition_blocks_equivalent(
        py_rd_per_rank[1][0], py_rd_per_rank[2][0],
        label="Py rigidDiaphragm floor_3 rank1-vs-rank2",
    )
    # Same no-sharding check on the Py side.
    py_rd_re = re.compile(r"^ops\.rigidDiaphragm\(([^)]*)\)")
    for rank, lines in py_rd_per_rank.items():
        m = py_rd_re.match(lines[0])
        assert m is not None, f"py rd line not parseable: {lines[0]!r}"
        toks = [t.strip() for t in m.group(1).split(",")]
        # toks = [perp, master, s1, s2, s3]
        assert int(toks[1]) == 7, f"py rank {rank}: master != 7"
        py_slaves = sorted(int(t) for t in toks[2:])
        assert py_slaves == [2, 4, 6], (
            f"py rank {rank}: slave list must be full [2,4,6]; "
            f"got {py_slaves!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — mp_constraint_comment replicates with its associated constraint
# ---------------------------------------------------------------------------


def test_cross_rank_mp_constraint_comment_replicates_with_associated_constraint(
    tmp_path,
) -> None:
    """ADR 0027 §"Decision" bullet 5 — ``mp_constraint_comment(name)``
    emits on every rank where the associated constraint emits, BEFORE
    the constraint line.  Comment text is identical across ranks.

    Reuses the Test-1 equalDOF fixture but with a distinctive ``name=``
    that surfaces as a ``# cross_rank_named_constraint`` comment line.
    """
    fem = make_two_column_frame_partitioned()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="cross_rank_named_constraint",
        ),
    ])

    tcl_text, py_text = _emit_decks(fem, tmp_path, ndf=6)
    tcl_by_rank = _split_tcl_deck_lines(tcl_text)
    py_by_rank = _split_py_deck_lines(py_text)

    expected_comment = "# cross_rank_named_constraint"

    # ----- Tcl side --------------------------------------------------------
    # Every rank that emits equalDOF must also carry the comment, BEFORE
    # the equalDOF line.  Comment text byte-identical across ranks.
    tcl_comments_per_rank = {
        rank: [ln for ln in lines if ln == expected_comment]
        for rank, lines in tcl_by_rank.items()
    }
    tcl_ed_per_rank = {
        rank: [ln for ln in lines if ln.startswith("equalDOF")]
        for rank, lines in tcl_by_rank.items()
    }

    for rank in (0, 1):
        assert len(tcl_ed_per_rank[rank]) == 1
        assert len(tcl_comments_per_rank[rank]) == 1, (
            f"rank {rank}: must carry the mp_constraint_comment once "
            f"(found {tcl_comments_per_rank[rank]!r}); ADR 0027 bullet 5"
        )

    # INV: comment text identical across ranks.
    assert (
        tcl_comments_per_rank[0][0] == tcl_comments_per_rank[1][0]
    ), tcl_comments_per_rank

    # INV: comment precedes constraint line on each rank.
    for rank in (0, 1):
        lines = tcl_by_rank[rank]
        comment_idx = lines.index(expected_comment)
        ed_idx = next(
            i for i, ln in enumerate(lines) if ln.startswith("equalDOF")
        )
        assert comment_idx < ed_idx, (
            f"rank {rank}: mp_constraint_comment must precede the "
            f"constraint line; comment at {comment_idx}, eq at {ed_idx}"
        )

    # Cross-rank byte-equivalence on the (comment + constraint) pair.
    # We slice the (comment, eq) sub-block per rank and compare via
    # the helper — the helper strips foreign-node decls but the slice
    # we pass already excludes them, so this degenerates to a straight
    # byte-equality check (which is the contract for ADR 0027 INV-1
    # over the constraint sub-block).
    def _comment_plus_constraint_block(
        lines: list[str], constraint_prefix: str,
    ) -> str:
        comment_idx = lines.index(expected_comment)
        ed_idx = next(
            i for i, ln in enumerate(lines)
            if ln.startswith(constraint_prefix)
        )
        return "\n".join(lines[comment_idx:ed_idx + 1])

    assert_partition_blocks_equivalent(
        _comment_plus_constraint_block(tcl_by_rank[0], "equalDOF"),
        _comment_plus_constraint_block(tcl_by_rank[1], "equalDOF"),
        label="Tcl mp_constraint_comment + equalDOF",
    )

    # ----- Py side ---------------------------------------------------------
    # PyEmitter uses the same ``# name`` syntax (matching Tcl) per
    # ``PyEmitter.mp_constraint_comment``.
    py_comments_per_rank = {
        rank: [ln for ln in lines if ln == expected_comment]
        for rank, lines in py_by_rank.items()
    }
    py_ed_per_rank = {
        rank: [ln for ln in lines if ln.startswith("ops.equalDOF(")]
        for rank, lines in py_by_rank.items()
    }
    for rank in (0, 1):
        assert len(py_ed_per_rank[rank]) == 1
        assert len(py_comments_per_rank[rank]) == 1, py_comments_per_rank[rank]
    assert py_comments_per_rank[0][0] == py_comments_per_rank[1][0]

    for rank in (0, 1):
        lines = py_by_rank[rank]
        c = lines.index(expected_comment)
        e = next(i for i, ln in enumerate(lines) if ln.startswith("ops.equalDOF("))
        assert c < e

    assert_partition_blocks_equivalent(
        _comment_plus_constraint_block(py_by_rank[0], "ops.equalDOF("),
        _comment_plus_constraint_block(py_by_rank[1], "ops.equalDOF("),
        label="Py mp_constraint_comment + equalDOF",
    )


# ---------------------------------------------------------------------------
# Test 5 — equationConstraint (enforce="equation") replicates on every rank
# that owns the slave or any master (ADR 0068 Open item 2)
# ---------------------------------------------------------------------------


def test_cross_rank_equationConstraint_replicates_on_owning_ranks(
    tmp_path,
) -> None:
    """ADR 0068 Open item 2 — an ``enforce="equation"`` surface tie that
    straddles a partition emits its ``equationConstraint`` rows on EVERY
    rank that owns the slave OR any master (the rigidDiaphragm rule, not
    the single-host-rank element rule that the penalty tie uses).  Each
    rank ghost-declares the foreign nodes it doesn't own BEFORE the
    constraint rows; the per-DOF rows are byte-identical across ranks.

    Fixture: ``make_two_column_frame_partitioned`` — nodes 1,2 on rank 0,
    nodes 3,4 on rank 1.  The tie ties slave node 4 (rank 1) to masters
    1 & 2 (rank 0), so it touches BOTH ranks.  Previously this fail-louded
    (``NotImplementedError``) under partitioned emit.
    """
    fem = make_two_column_frame_partitioned()
    fem.add_surface_constraints([
        InterpolationRecord(
            kind=ConstraintKind.TIE,
            slave_node=4,
            master_nodes=[1, 2],
            weights=np.array([0.5, 0.5]),
            dofs=[1, 2, 3],
            enforce="equation",
            name="cross_rank_eq_tie",
        ),
    ])

    # Must NOT raise (the old behaviour was NotImplementedError here).
    tcl_text, py_text = _emit_decks(fem, tmp_path, ndf=6)
    tcl_by_rank = _split_tcl_deck_lines(tcl_text)
    py_by_rank = _split_py_deck_lines(py_text)

    assert set(tcl_by_rank.keys()) == {0, 1}, sorted(tcl_by_rank)
    assert set(py_by_rank.keys()) == {0, 1}, sorted(py_by_rank)

    # ----- Tcl side --------------------------------------------------------
    # 3 equationConstraint rows (one per tied DOF) on EACH owning rank.
    tcl_eq_per_rank = {
        rank: [ln for ln in lines if ln.startswith("equationConstraint ")]
        for rank, lines in tcl_by_rank.items()
    }
    assert len(tcl_eq_per_rank[0]) == 3, tcl_eq_per_rank
    assert len(tcl_eq_per_rank[1]) == 3, tcl_eq_per_rank

    # INV-1: the per-DOF rows are byte-identical across the two ranks
    # (the equation row text depends only on the record, not the rank).
    for row0, row1 in zip(tcl_eq_per_rank[0], tcl_eq_per_rank[1]):
        assert_partition_blocks_equivalent(
            row0, row1, label="Tcl equationConstraint cross-rank",
        )

    # The sum-to-zero form: 1.0 on the slave, -w on each master. For dof 1:
    #   equationConstraint 4 1 1.0 1 1 -0.5 2 1 -0.5
    dof1 = next(ln for ln in tcl_eq_per_rank[0] if ln.split()[2] == "1")
    toks = dof1.split()
    assert toks[:4] == ["equationConstraint", "4", "1", "1.0"], dof1
    assert int(toks[4]) == 1 and float(toks[6]) == -0.5
    assert int(toks[7]) == 2 and float(toks[9]) == -0.5

    # INV-2: foreign-node declarations precede the equationConstraint rows.
    def _idx(lines: list[str], prefix: str) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith(prefix):
                return i
        return -1

    # Rank 0 owns masters 1,2 (native); slave 4 is foreign → declared first.
    r0 = tcl_by_rank[0]
    eq0 = _idx(r0, "equationConstraint ")
    f4 = _idx(r0, "node 4 ")
    assert f4 != -1 and f4 < eq0, f"rank0 must declare foreign node 4 first: {r0!r}"
    # Rank 1 owns slave 4 (native); masters 1,2 are foreign → declared first.
    r1 = tcl_by_rank[1]
    eq1 = _idx(r1, "equationConstraint ")
    f1 = _idx(r1, "node 1 ")
    f2 = _idx(r1, "node 2 ")
    assert f1 != -1 and f2 != -1, f"rank1 must declare foreign masters: {r1!r}"
    assert max(f1, f2) < eq1, "foreign masters must precede equationConstraint"

    # ----- Py side ---------------------------------------------------------
    py_eq_per_rank = {
        rank: [ln for ln in lines if ln.startswith("ops.equationConstraint(")]
        for rank, lines in py_by_rank.items()
    }
    assert len(py_eq_per_rank[0]) == 3, py_eq_per_rank
    assert len(py_eq_per_rank[1]) == 3, py_eq_per_rank
    for row0, row1 in zip(py_eq_per_rank[0], py_eq_per_rank[1]):
        assert_partition_blocks_equivalent(
            row0, row1, label="Py equationConstraint cross-rank",
        )


def test_equationConstraint_single_owning_rank_no_spurious_replication(
    tmp_path,
) -> None:
    """When the slave and all masters live on ONE rank, the equation tie
    emits only on that rank — replicate-on-owning must not over-emit it
    onto ranks that touch neither the slave nor any master.

    Tie ties slave 2 → masters [1] (both rank 0); rank 1 (nodes 3,4)
    touches nothing and must carry no equationConstraint row.
    """
    fem = make_two_column_frame_partitioned()
    fem.add_surface_constraints([
        InterpolationRecord(
            kind=ConstraintKind.TIE,
            slave_node=2,
            master_nodes=[1],
            weights=np.array([1.0]),
            dofs=[1, 2, 3],
            enforce="equation",
            name="local_eq_tie",
        ),
    ])

    tcl_text, _py = _emit_decks(fem, tmp_path, ndf=6)
    tcl_by_rank = _split_tcl_deck_lines(tcl_text)

    eq_r0 = [ln for ln in tcl_by_rank.get(0, []) if ln.startswith("equationConstraint ")]
    eq_r1 = [ln for ln in tcl_by_rank.get(1, []) if ln.startswith("equationConstraint ")]
    assert len(eq_r0) == 3, f"rank0 (owns 1,2) must emit the tie: {eq_r0!r}"
    assert eq_r1 == [], f"rank1 (owns neither slave nor master) must not: {eq_r1!r}"


# ---------------------------------------------------------------------------
# Test 6 — phantom-node tag identity + within-rank ordering (E2E)
# ---------------------------------------------------------------------------


def _make_cross_rank_phantom_fem() -> FEMStub:
    """NodeToSurface fixture where the phantom is referenced on BOTH
    ranks (master is on rank 0 with two slaves; the other two slaves
    are on rank 1) — so the phantom emit fires per rank.

    Mirrors the fixture from
    ``test_emit_partitioned_phantom_node_tag_identity.py`` but the
    surface here parses the actual emitted deck rather than the
    ``RecordingEmitter`` call list — closing the gap between
    "bridge called node(999, ...) twice" and "the deck file actually
    contains the phantom-decl line on each rank's block, before the
    referencing constraint".
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
                slave_node=s,
                dofs=[1, 2, 3],
                name=f"n2s_ed_{s}",
            )
            for s in (1, 2, 3, 4)
        ],
        kind=ConstraintKind.NODE_TO_SURFACE,
        dofs=[1, 2, 3],
        name="bind",
    )
    stub.add_node_constraints([n2s])
    return stub


def test_cross_rank_phantom_node_tag_identity(tmp_path) -> None:
    """ADR 0027 §"Phantom-node policy" / INV-3 — the phantom-node tag
    and ``*xyz`` coordinates are broker-derived (one canonical value
    pair) and emitted identically on every rank that hosts a
    referencing constraint.  The phantom decl precedes its first
    referencing constraint on each rank's block.

    Distinct from ``test_emit_partitioned_phantom_node_tag_identity``:
    that test asserts the RecordingEmitter calls; this test parses
    the actual ``.tcl`` / ``.py`` files to lock the within-block
    ordering rule.
    """
    fem = _make_cross_rank_phantom_fem()

    # Empty Cols PG → bridge emits no elements; we need to avoid the
    # geomTransf/element path or the build pipeline tries to expand
    # ``"Cols"`` (which doesn't exist on this stub).  The fixture above
    # carries a single ``Plate`` PG but no element-pass; the bridge
    # tolerates a constraint-only emit.  We bypass the helper's
    # geomTransf+element auto-step by inlining the emit here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops_tcl = apeSees(cast("object", fem))
        ops_tcl.model(ndm=3, ndf=3)
        ops_tcl.tcl(str(tmp_path / "deck.tcl"))
        ops_py = apeSees(cast("object", fem))
        ops_py.model(ndm=3, ndf=3)
        ops_py.py(str(tmp_path / "deck.py"))

    tcl_text = (tmp_path / "deck.tcl").read_text(encoding="utf-8")
    py_text = (tmp_path / "deck.py").read_text(encoding="utf-8")
    tcl_by_rank = _split_tcl_deck_lines(tcl_text)
    py_by_rank = _split_py_deck_lines(py_text)

    assert set(tcl_by_rank.keys()) == {0, 1}, (
        f"Tcl deck should expose both ranks; got {sorted(tcl_by_rank)}"
    )
    assert set(py_by_rank.keys()) == {0, 1}

    phantom_tag = 999
    expected_xyz = (0.5, 0.5, 0.5)

    # ----- Tcl side --------------------------------------------------------
    # Pull the phantom decl line on each rank; assert tag + xyz identical.
    tcl_phantom_re = re.compile(
        r"^node\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+-ndf\s+(\d+)\s*$"
    )

    def _find_phantom_line(lines: list[str], tag: int) -> tuple[int, str]:
        for i, ln in enumerate(lines):
            m = tcl_phantom_re.match(ln)
            if m and int(m.group(1)) == tag:
                return i, ln
        return -1, ""

    idx0, line0 = _find_phantom_line(tcl_by_rank[0], phantom_tag)
    idx1, line1 = _find_phantom_line(tcl_by_rank[1], phantom_tag)
    assert idx0 != -1, (
        f"Tcl rank 0 must declare phantom {phantom_tag}; got {tcl_by_rank[0]!r}"
    )
    assert idx1 != -1, (
        f"Tcl rank 1 must declare phantom {phantom_tag}; got {tcl_by_rank[1]!r}"
    )
    # INV-3: tag + coords + ndf identical across ranks.
    assert line0 == line1, (
        f"INV-3: phantom decl must be byte-identical across ranks:\n"
        f"  rank 0: {line0!r}\n  rank 1: {line1!r}"
    )

    # Sanity: parse-out and confirm xyz + ndf=6.
    for ln in (line0, line1):
        m = tcl_phantom_re.match(ln)
        assert m is not None
        got_xyz = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        for got, want in zip(got_xyz, expected_xyz):
            assert abs(got - want) < 1e-12, (
                f"phantom xyz {got_xyz} != broker xyz {expected_xyz}"
            )
        assert int(m.group(5)) == 6, (
            f"phantom must carry -ndf 6; got {m.group(5)} in {ln!r}"
        )

    # Phantom decl precedes the first referencing constraint on each rank.
    # Referencing constraints carry the phantom tag in any position
    # (rigidLink to phantom on rank 0 via "rigidLink beam 5 999", or
    # equalDOF FROM phantom via "equalDOF 999 <slave>").
    def _first_reference_to_phantom(lines: list[str], tag: int) -> int:
        tok = str(tag)
        for i, ln in enumerate(lines):
            if ln.startswith("node "):
                continue
            if tok in ln.split():
                return i
        return -1

    ref0 = _first_reference_to_phantom(tcl_by_rank[0], phantom_tag)
    ref1 = _first_reference_to_phantom(tcl_by_rank[1], phantom_tag)
    assert ref0 != -1 and ref1 != -1, (
        f"both ranks must reference phantom {phantom_tag} in some constraint"
    )
    assert idx0 < ref0, (
        f"Tcl rank 0: phantom decl at {idx0} must precede first ref at {ref0}"
    )
    assert idx1 < ref1, (
        f"Tcl rank 1: phantom decl at {idx1} must precede first ref at {ref1}"
    )

    # ----- Py side ---------------------------------------------------------
    py_phantom_re = re.compile(
        r"^ops\.node\((\d+),\s*(\S+?),\s*(\S+?),\s*(\S+?),\s*'-ndf',\s*(\d+)\)\s*$"
    )

    def _find_py_phantom_line(lines: list[str], tag: int) -> tuple[int, str]:
        for i, ln in enumerate(lines):
            m = py_phantom_re.match(ln)
            if m and int(m.group(1)) == tag:
                return i, ln
        return -1, ""

    pidx0, pline0 = _find_py_phantom_line(py_by_rank[0], phantom_tag)
    pidx1, pline1 = _find_py_phantom_line(py_by_rank[1], phantom_tag)
    assert pidx0 != -1 and pidx1 != -1, (
        f"Py decks must declare phantom {phantom_tag} on both ranks"
    )
    assert pline0 == pline1, (
        f"INV-3 (Py): phantom decl must be byte-identical across ranks:\n"
        f"  rank 0: {pline0!r}\n  rank 1: {pline1!r}"
    )
    for ln in (pline0, pline1):
        m = py_phantom_re.match(ln)
        assert m is not None
        got_xyz = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        for got, want in zip(got_xyz, expected_xyz):
            assert abs(got - want) < 1e-12
        assert int(m.group(5)) == 6

    # Within-rank ordering on the Py side: phantom decl precedes first
    # constraint that mentions the phantom tag.  Constraints in the
    # Py deck appear as e.g. ``ops.rigidLink('beam', 5, 999)`` /
    # ``ops.equalDOF(999, 1, 1, 2, 3)``; we treat any non-node line
    # carrying the tag (as a comma-separated token) as a reference.
    def _first_py_reference(lines: list[str], tag: int) -> int:
        tag_token = str(tag)
        for i, ln in enumerate(lines):
            if ln.startswith("ops.node("):
                continue
            # Tokenise on commas + parens; strip whitespace.
            toks = [
                t.strip()
                for t in ln.replace("(", ",").replace(")", ",").split(",")
            ]
            if tag_token in toks:
                return i
        return -1

    pref0 = _first_py_reference(py_by_rank[0], phantom_tag)
    pref1 = _first_py_reference(py_by_rank[1], phantom_tag)
    assert pref0 != -1 and pref1 != -1
    assert pidx0 < pref0
    assert pidx1 < pref1
