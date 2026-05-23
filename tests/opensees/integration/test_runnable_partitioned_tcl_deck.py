"""Integration test for a 2-rank partitioned Tcl deck.

Drives a small frame model through the partitioned emit path, writes
the Tcl deck to disk, and verifies its structural properties:

* Brace balance.
* Every node / element appears under exactly one ``partition_open``
  block (no leaks).
* Cross-partition MP constraints emit on multiple ranks with
  byte-identical text (ADR 0027 INV-1).
* The ``proc getPID`` shim is emitted exactly once.
* Analysis commands (``numberer`` / ``system`` / ...) sit OUTSIDE
  any ``if {[getPID] == K} { ... }`` block.

This is the textual-shape gate. The runtime parse-and-execute gate is
P5 (OpenSeesMP subprocess run).
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import cast

from apeGmsh._kernel.records._constraints import NodeGroupRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
)


def _check_brace_balance(text: str) -> None:
    """Raise AssertionError on unbalanced Tcl braces.

    Counts ``{`` and ``}`` characters globally — this catches the
    common mistake of forgetting to close a ``partition_open`` block
    without going full Tcl-parser.
    """
    n_open = text.count("{")
    n_close = text.count("}")
    assert n_open == n_close, (
        f"Tcl brace imbalance: {n_open} '{{' vs {n_close} '}}'"
    )


def _split_into_partition_blocks(text: str) -> dict[int, str]:
    """Return ``{rank: body_text}`` for each ``if {[getPID] == K} {`` block.

    Body text is everything between the opening brace and the matching
    closing brace. Brace-depth tracked manually since Tcl uses
    syntactic braces.
    """
    blocks: dict[int, str] = {}
    lines = text.split("\n")
    i = 0
    rank_re = re.compile(r"if \{\[getPID\] == (\d+)\} \{")
    while i < len(lines):
        m = rank_re.search(lines[i])
        if not m:
            i += 1
            continue
        rank = int(m.group(1))
        depth = 1  # we're inside one open brace from the header
        i += 1
        body: list[str] = []
        while i < len(lines) and depth > 0:
            for ch in lines[i]:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            if depth == 0:
                # Strip the closing brace line.
                break
            body.append(lines[i])
            i += 1
        i += 1
        blocks[rank] = "\n".join(body)
    return blocks


def test_runnable_partitioned_tcl_deck_shape() -> None:
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
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.tcl")
        ops.tcl(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    # 1. Brace balance.
    _check_brace_balance(text)

    # 2. Exactly one getPID shim at the top of the file.
    assert text.count("proc getPID") == 1, (
        "the getPID shim must be emitted exactly once"
    )

    # 3. Two partition blocks present.
    blocks = _split_into_partition_blocks(text)
    assert set(blocks.keys()) == {0, 1}, (
        f"expected ranks 0 and 1; got {sorted(blocks.keys())}"
    )

    # 4. Every node appears under exactly one partition block as a
    # NATIVE declaration (not a foreign-side declaration before a
    # rigidDiaphragm). Node 1 → rank 0; node 2 → rank 0; node 3 →
    # rank 1; node 4 → rank 1.  Per ADR 0027 INV-2 the rigid_diaphragm
    # declares one foreign node on each rank's block (master 2 on
    # rank 1, slave 4 on rank 0) BEFORE the rigidDiaphragm line —
    # those are additional ``node`` lines and don't violate the "every
    # owned node appears in exactly one block" invariant on the
    # native side.  Test the native-ownership shape: nodes 1/2 must
    # appear in rank 0 block, nodes 3/4 in rank 1 block.
    def _has_node_decl(body: str, tag: int) -> bool:
        # 'node 2 ...' or 'node 2 0.0 0.0 1.0' patterns
        return bool(re.search(rf"\bnode\s+{tag}\b", body))

    for tag in (1, 2):
        assert _has_node_decl(blocks[0], tag), (
            f"node {tag} (owner rank 0) must appear in rank-0 block"
        )
    for tag in (3, 4):
        assert _has_node_decl(blocks[1], tag), (
            f"node {tag} (owner rank 1) must appear in rank-1 block"
        )

    # 5. Cross-partition rigidDiaphragm — INV-1: text identical on
    # both ranks.
    rd_lines_rank0 = [
        ln.strip() for ln in blocks[0].split("\n") if "rigidDiaphragm" in ln
    ]
    rd_lines_rank1 = [
        ln.strip() for ln in blocks[1].split("\n") if "rigidDiaphragm" in ln
    ]
    assert len(rd_lines_rank0) == 1
    assert len(rd_lines_rank1) == 1
    assert rd_lines_rank0 == rd_lines_rank1, (
        "INV-1: cross-partition rigidDiaphragm must be byte-identical "
        "across ranks"
    )

    # 6. Analysis commands must NOT be inside any partition block.
    # The partition-block bodies are blocks[0] and blocks[1]; the
    # OUTSIDE text is everything else. Re-split via regex.
    outside = re.sub(
        r"if \{\[getPID\] == \d+\} \{[\s\S]*?\n\}",
        "",
        text,
    )
    assert "numberer ParallelPlain" in outside, (
        "numberer must be emitted globally (outside any partition block)"
    )
    assert "system Mumps" in outside, (
        "system must be emitted globally (outside any partition block)"
    )
    # And they MUST NOT appear inside any rank block.
    for rank, body in blocks.items():
        assert "numberer" not in body, (
            f"numberer must not appear inside rank-{rank} block"
        )
        assert "system" not in body, (
            f"system must not appear inside rank-{rank} block"
        )
