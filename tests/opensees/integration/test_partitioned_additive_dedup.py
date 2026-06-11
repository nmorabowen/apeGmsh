"""Additive nodal quantities emit on ONE rank under partitioned emit.

Under OpenSeesMP, shared (interface) nodes exist on every rank that
defines them and the parallel assembly **SUMS** each domain's nodal
contributions at the merged equations. Two line families therefore need
opposite fan-out policies:

* **idempotent** lines — ``node`` / ``fix`` / ``sp`` — replicate on
  every owning rank (each domain needs the node and its constraints);
* **additive** lines — ``mass`` / pattern ``load`` — must emit on
  exactly ONE rank (the node's *primary* rank,
  :func:`primary_owner_map`), or interface nodes carry the quantity
  once per owning rank.

Before this lock, an 8-partition plane-wave model emitted 177 ``mass``
lines for 81 massed nodes and the partitioned transient diverged from
the byte-identical sequential run by ~100 % of peak velocity;
deduplicated, the two runs agree to machine precision (~5e-15 of peak).
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import cast

from apeGmsh.opensees import apeSees

from tests.opensees.fixtures.fem_stub import make_two_column_frame_partitioned


def _emit_deck_with_shared_node() -> str:
    """Two-rank frame whose partitions BOTH list node 2 (a halo/interface
    node), with mass / fix / load / sp all targeting that shared node."""
    fem = make_two_column_frame_partitioned()
    # Make node 2 a shared (interface) node: both ranks define it.
    fem.set_partitions([
        (0, [1, 2], [1]),
        (1, [2, 3, 4], [2]),
    ])

    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(nodes=[2], dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(nodes=[2], values=(1.5, 1.5, 1.5, 0.0, 0.0, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=2, forces=(0.0, 0.0, -5e3, 0.0, 0.0, 0.0))
        p.sp(node=2, dof=1, value=0.01)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.tcl")
        ops.tcl(path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def _count_lines(text: str, pattern: str) -> int:
    rx = re.compile(pattern)
    return sum(1 for ln in text.splitlines() if rx.match(ln.strip()))


def test_shared_node_mass_emits_on_exactly_one_rank() -> None:
    """ADDITIVE: the shared node's ``mass`` line appears ONCE in the
    whole deck — OpenSeesMP sums nodal mass across ranks at shared
    equations, so per-owner replication double-counts interface mass."""
    text = _emit_deck_with_shared_node()
    assert _count_lines(text, r"mass 2 ") == 1, text


def test_shared_node_load_emits_on_exactly_one_rank() -> None:
    """ADDITIVE: the shared node's pattern ``load`` line appears ONCE —
    same MP-assembly summation as nodal mass."""
    text = _emit_deck_with_shared_node()
    assert _count_lines(text, r"load 2 ") == 1, text


def test_shared_node_fix_and_sp_replicate_on_every_owning_rank() -> None:
    """IDEMPOTENT: ``fix`` / ``sp`` lines replicate on BOTH owning ranks
    (each rank's domain holds its own copy of the node and needs the
    constraint locally) — locks that the dedup did NOT leak into the
    constraint fan-out."""
    text = _emit_deck_with_shared_node()
    assert _count_lines(text, r"fix 2 ") == 2, text
    assert _count_lines(text, r"sp 2 ") == 2, text


def test_shared_node_declaration_replicates_on_every_owning_rank() -> None:
    """IDEMPOTENT: the shared node's ``node`` declaration stays on both
    ranks — only additive quantities were narrowed to the primary rank."""
    text = _emit_deck_with_shared_node()
    assert _count_lines(text, r"node 2 ") == 2, text
