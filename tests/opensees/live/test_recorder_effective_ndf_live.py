"""B5-e (#1759, fixes #1745) live oracle: a decoupled master raised to
ndf 6 by ``ops.ndf`` in an ndf=3 solid model records its rotation.

A tet box (ndf 3, every node held) and an element-less decoupled master
raised to ndf 6. The master's translations and two rotations are held, and
its ``rotation_x`` (DOF 4) is imposed to ``ROT`` by a pattern ``sp``. The
penalty handler puts every DOF in the system, so the solve has equations,
and the master's DOF 4 carries only the imposed value: the closed form is
``rotation_x = ROT`` exactly.

The deck route (``recorder.declare`` + a deck run + ``ops.results``) and the
live ``.read`` both return ``ROT``. On origin/main the deck route fails:
#1738's emit refused ``rotation_x`` on the raised master.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

from apeGmsh.opensees.apesees import apeSees

from tests.opensees.unit.test_node_pair_zerolength import _box_with_ground

pytestmark = pytest.mark.live

ROT = 0.01


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


@pytest.fixture
def box() -> Iterator[tuple[Any, Any]]:
    g, fem, h = _box_with_ground("b5e_live_master")
    try:
        yield fem, h
    finally:
        g.end()


def _bridge(fem: Any, h: Any) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="Body", material=mat)
    ops.ndf(h, ndf=6)
    tag = int(h.tag)
    ops.fix(pg="Body", dofs=(1, 1, 1))
    ops.fix(nodes=[tag], dofs=(1, 1, 1, 0, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(node=tag, dof=4, value=ROT)
    ops.constraints.Penalty(alpha_sp=1e12, alpha_mp=1e12)
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def test_raised_master_rotation_through_deck_and_read(
    box: tuple[Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    fem, h = box
    tag = int(h.tag)

    ops = _bridge(fem, h)
    ops.recorder.declare(nodes=("rotation_x",), ids=(tag,),
                         file_root=str(tmp_path), name="m")
    deck = tmp_path / "deck.py"
    ops.py(str(deck), run=True, analyze_steps=1, python=sys.executable)
    assert f"'-node', {tag}, '-dof', 4, 'disp'" in deck.read_text()
    got = ops.results(str(deck)).nodes.get(
        ids=[tag], component="rotation_x").item()
    assert got == pytest.approx(ROT, rel=1e-12)

    live = _bridge(fem, h)
    assert live.analyze(steps=1) == 0
    assert live.nodes[tag].read("rotation_x") == pytest.approx(ROT, rel=1e-12)
