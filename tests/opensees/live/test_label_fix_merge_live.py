"""ADR 0115 D6 live check (R2-a, #1651), on stock openseespy.

A staged footing whose shared corner is fixed by two overlapping
``fix`` calls (the left edge and the bottom edge both hold the corner
at the origin) builds and runs. On ``main`` before R2-a, V2 refused this
staged build ("Duplicate fix / mass"); without the merge, OpenSees
itself refuses the second SP on the corner (``Domain::addSP_Constraint``).

The oracle is equilibrium: the stage pushes ``-P`` per load step at the
top-right node for two LoadControl steps, so the vertical reactions of
the supports sum to ``+2P``.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from tests.opensees.unit.test_label_selector_fix_merge import (
    build_footing_fem,
    chain,
    corner_of,
    footing,
)

P = 10.0


@pytest.mark.live
def test_staged_footing_with_shared_corner_fixes_builds_and_runs(
    tmp_path: Path,
) -> None:
    # The module the emitted py deck imports by name.
    osp = pytest.importorskip("openseespy.opensees")
    fem = build_footing_fem()
    ops = footing(fem)
    ops.fix(label="left", dofs=(1, 1))
    ops.fix(label="bottom", dofs=(1, 1))
    load_node = next(
        int(n) for n, xyz in zip(fem.nodes.ids, fem.nodes.coords)
        if abs(xyz[0] - 2.0) < 1e-9 and abs(xyz[1] - 1.0) < 1e-9
    )
    with ops.stage(name="load") as s:
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts) as p:
            p.load(node=load_node, forces=(0.0, -P))
        s.analysis(**chain(ops))
        s.run(n_increments=2)
    deck = tmp_path / "footing.py"
    ops.py(str(deck))
    text = deck.read_text()
    corner = corner_of(fem)
    assert text.count(f"ops.fix({corner}, ") == 1

    runpy.run_path(str(deck), run_name="__main__")

    supports = {int(n) for n in fem.nodes.select(label="left").ids} | {
        int(n) for n in fem.nodes.select(label="bottom").ids}
    osp.reactions()
    ry = sum(osp.nodeReaction(n, 2) for n in supports)
    assert ry == pytest.approx(2.0 * P, rel=1e-9)
    assert osp.nodeDisp(load_node, 2) < 0.0
    assert osp.nodeDisp(corner, 1) == 0.0 and osp.nodeDisp(corner, 2) == 0.0
    osp.wipe()
