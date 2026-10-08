"""ADR 0117 INV-9 (AS4-b): a two-instance, two-rank assembly deck.

Two instances of the 2x2x2 block, ``pier_1`` on rank 0 and ``pier_2`` on
rank 1, stacked and joined across the ranks by a ``tie``
(``enforce="equation"``), an ``equal_dof``, and from the rank-0 reference
nodes a ``rigid_diaphragm`` and the AS4-a couplings ``rigid_link`` and
``couple(kind="kinematic")``.

Oracles, each naming the right answer independently of the code under test:

* ownership is closed form: instance ``k`` occupies the FEM-id window
  ``k * 1_000_000`` (ADR 0038), so a node below ``2_000_000`` is rank 0's
  (``pier_1`` and the reference nodes) and the rest are rank 1's;
* each MP line of the flat deck (``equalDOF``, ``equationConstraint``,
  ``rigidLink``, ``rigidDiaphragm``) is written once, identically, on
  exactly the ranks that own one of its nodes (INV-9; ADR 0027), and the
  closed-form counts of cross-rank lines are reached: 9 face-node pairs,
  9 x 3 equation rows, 9 links, 1 diaphragm;
* so the set of MP lines over both ranks equals the flat deck's (the tags
  are the node ids a line names, the same in both decks);
* the RBE2 coupling element is written once, with the flat deck's tag, on
  the rank that owns its target; a rank-0 reference node a rank-1 line
  needs is declared on rank 1 before that line (a ghost).

Line order is not asserted: K1-3d S7 (#1459) reorders partitioned decks.
"""
from __future__ import annotations

import re
import warnings

import pytest

from tests.assembly.test_two_instances_one_tie import (
    GRANULE,
    H,
    SIDE,
    block_fem,
    declare_block,
    write_instance,
)

#: Nodes on one face of the 2x2x2 block: (2 + 1) ** 2.
FACE = 9
MP = ("equalDOF", "equationConstraint", "rigidLink", "rigidDiaphragm")
PID_BLOCK = re.compile(r"^if \{\[getPID\] == (\d+)\} \{$")
#: The reference node above the top of the stack: FEM id 1.
REF = (SIDE / 2, SIDE / 2, 2 * H + 5.0)
#: The diaphragm reference node, in the interface plane beside the stack
#: (no block node shares its point): FEM id 2.
CM = (SIDE + 5.0, SIDE / 2, H)


@pytest.fixture(scope="module")
def decks(tmp_path_factory) -> tuple[dict[int, list[str]], list[str]]:
    from apeGmsh.assembly import Assembly

    d = tmp_path_factory.mktemp("inv9")
    block = write_instance(d / "block.h5", block_fem(d), declare_block)
    asm = (Assembly("inv9")
           .instance("pier_1", block, partition_rank=0)
           .instance("pier_2", block, translate=(0.0, 0.0, H), partition_rank=1)
           .node("ref", REF)
           .node("cm", CM)
           .tie("pier_1.top", "pier_2.bot", enforce="equation", dofs=[1, 2, 3])
           .equal_dof("pier_1.top", "pier_2.bot", dofs=[1, 2, 3])
           .rigid_diaphragm("cm", "pier_2.bot", constrained_dofs=(1, 2, 6))
           .rigid_link("ref", "pier_2.top", link_type="rod")
           .couple("pier_2.top", kind="kinematic", reference="ref"))
    ops = asm.bridge(ndm=3, ndf=3)
    assert ops.fem.nodes.partitions == [0, 1]
    ops.ndf(1, ndf=6)
    ops.ndf(2, ndf=6)
    with warnings.catch_warnings():
        # The emit-time advisories (auto handler, parallel numberer and
        # system, a detached diaphragm master) do not bear on INV-9.
        warnings.simplefilter("ignore")
        ops.tcl(str(d / "ranked.tcl"))
        ops.tcl(str(d / "flat.tcl"), flat=True)
    ranked = (d / "ranked.tcl").read_text(encoding="utf-8").splitlines()
    flat = (d / "flat.tcl").read_text(encoding="utf-8").splitlines()
    return _blocks(ranked), [ln.strip() for ln in flat]


def _blocks(deck: list[str]) -> dict[int, list[str]]:
    """``{rank: stripped lines}`` of each ``if {[getPID] == k} {...}`` block."""
    out: dict[int, list[str]] = {}
    cur: "int | None" = None
    for ln in deck:
        m = PID_BLOCK.match(ln)
        if m:
            cur = int(m.group(1))
            out[cur] = []
        elif ln == "}" and cur is not None:
            cur = None
        elif cur is not None:
            out[cur].append(ln.strip())
    return out


def _owner(node: int) -> int:
    """Closed form: rank 0 owns the reference nodes and ``pier_1``'s window."""
    return 0 if node < 2 * GRANULE else 1


def _nodes_of(line: str) -> set[int]:
    """The node ids an MP line names (not its DOFs or coefficients)."""
    tok = line.split()
    if tok[0] == "equalDOF":
        return {int(tok[1]), int(tok[2])}
    if tok[0] == "equationConstraint":        # cnode cdof c (rnode rdof c)*
        return {int(t) for t in tok[1::3]}
    if tok[0] == "rigidLink":                  # rigidLink bar|beam m s
        return {int(tok[2]), int(tok[3])}
    if tok[0] == "rigidDiaphragm":             # rigidDiaphragm perp m s...
        return {int(t) for t in tok[2:]}
    raise AssertionError(f"not an MP line: {line!r}")


def _mp(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.split(" ", 1)[0] in MP]


def test_two_blocks_and_no_other(decks):
    blocks, _ = decks
    assert sorted(blocks) == [0, 1]
    assert all(blocks.values())


@pytest.mark.parametrize("kind, n_cross", [
    ("equalDOF", FACE),
    ("equationConstraint", 3 * FACE),
    ("rigidLink", FACE),
    ("rigidDiaphragm", 1),
])
def test_each_mp_line_is_written_on_its_owning_ranks(decks, kind, n_cross):
    blocks, flat = decks
    lines = [ln for ln in _mp(flat) if ln.startswith(kind + " ")]
    owners = {ln: {_owner(n) for n in _nodes_of(ln)} for ln in lines}
    assert sum(o == {0, 1} for o in owners.values()) == n_cross, owners
    for ln in lines:
        for rank in (0, 1):
            assert blocks[rank].count(ln) == (rank in owners[ln]), (rank, ln)


def test_the_mp_tag_set_equals_the_flat_decks(decks):
    blocks, flat = decks
    assert set(_mp(blocks[0])) | set(_mp(blocks[1])) == set(_mp(flat))
    assert set(_mp(blocks[0])) & set(_mp(blocks[1]))   # some lines are shared


def test_the_coupling_is_written_once_with_its_reference_node_ghosted(decks):
    blocks, flat = decks
    (line,) = [ln for ln in flat if ln.startswith("element LadrunoKinematicCoupling")]
    on = [r for r, lines in blocks.items() if line in lines]
    assert on == [1]                            # the rank that owns the target
    (diaphragm,) = [ln for ln in flat if ln.startswith("rigidDiaphragm")]
    for tag, xyz, use in ((1, REF, line), (2, CM, diaphragm)):
        decl = f"node {tag} {xyz[0]} {xyz[1]} {xyz[2]} -ndf 6"
        assert decl in blocks[0]                # owned by rank 0
        lines = blocks[1]
        assert decl in lines                    # ghosted on rank 1 ...
        assert lines.index(decl) < lines.index(use)    # ... before its use
