"""B5-b (#1726, fixes #1723) live oracle: ``recorder.declare`` maps node
components through the per-node DOF layout.

The 2-D cantilever at ``ndf=3`` (the R2-b fixture), tip load ``P``, run as
an openseespy deck in a subprocess of this interpreter. The declared tip
``rotation_z`` is DOF 3 of a planar beam node; the fixed 6-DOF table sent
it to ``-dof 6``, which OpenSees answers with 0 and no warning. The
recorder file must read ``PL^2/2EI`` (and ``PL^3/3EI`` for the
deflection beside it).

The agreement test: a cantilever on a zero-length rotational spring at
``ndm=2, ndf=3`` and ``ndm=3, ndf=6``. The tip ``rotation_z`` is
``-(PL^2/2EI + PL/k)`` and the spring's fixed ground node, which only the
zero-length touches, reacts ``reaction_moment_z = PL``. The deck
(``recorder.declare`` + ``ops.results``) and the live ``.read`` give both,
from the one rotation-role table. On origin/main both fail: at ``ndf=3`` the
declare deck records DOF 6 (0) and ``ops.results`` refuses it; at ``ndf=6``
``ops.results`` and ``.read`` refuse the zero-length ground node (#1703).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.opensees.live.test_readback_live import (
    E,
    I_BEAM,
    L_BEAM,
    P_TIP,
    build_cantilever_fem,
    cantilever_bridge,
)

pytestmark = pytest.mark.live

UY = -P_TIP * L_BEAM ** 3 / (3.0 * E * I_BEAM)
RZ = -P_TIP * L_BEAM ** 2 / (2.0 * E * I_BEAM)


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


def test_declared_tip_rotation_matches_pl2_over_2ei(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = cantilever_bridge(cantilever)
    ops.recorder.declare(
        nodes=("displacement_y", "rotation_z"), label="tip",
        file_root=str(tmp_path), name="tip",
    )
    deck = tmp_path / "deck.py"
    ops.py(str(deck), run=True, analyze_steps=1, python=sys.executable)

    out = tmp_path / "tip__default__disp.out"
    assert out.is_file(), sorted(p.name for p in tmp_path.iterdir())
    rows = np.atleast_2d(np.loadtxt(out))
    # [time, uy, rz] for the one tip node, -dof 2 3.
    assert rows.shape[1] == 3, rows
    _t, uy, rz = rows[-1]
    assert uy == pytest.approx(UY, rel=1e-10)
    assert rz == pytest.approx(RZ, rel=1e-10)


# ---------------------------------------------------------------------------
# Agreement: declare deck + ops.results, and the live .read
# ---------------------------------------------------------------------------

K_ROT = 2.0e6      # rotational spring at the base (N m / rad)
K_STIFF = 1.0e14   # the spring's other directions: effectively rigid


def build_spring_cantilever_fem(ndm: int) -> tuple[Any, int]:
    """The R2-b cantilever line plus a decoupled ground node at its base;
    returns the FEM and the ground node's tag."""
    import gmsh

    from apeGmsh import apeGmsh

    with apeGmsh(model_name=f"b5_spring_{ndm}", verbose=False) as g:
        geo = g.model.geometry
        base = geo.add_point(0.0, 0.0, 0.0)
        tip = geo.add_point(L_BEAM, 0.0, 0.0)
        line = geo.add_line(base, tip)
        gmsh.model.mesh.setTransfiniteCurve(line, 5)
        g.physical.add(1, [line], name="Beam")
        g.labels.add(0, [base], "base")
        g.labels.add(0, [tip], "tip")
        h = g.decouple_node(coords=(0.0, 0.0, 0.0), label="gnd")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1), int(h.tag)


def spring_cantilever_bridge(fem: Any, gnd: int, ndm: int) -> Any:
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

    ndf = 3 if ndm == 2 else 6
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=ndm, ndf=ndf)
    if ndm == 2:
        t = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(pg="Beam", transf=t, A=1.0e-2, E=E,
                                      Iz=I_BEAM)
    else:
        t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops.element.elasticBeamColumn(
            pg="Beam", transf=t, A=1.0e-2, E=E, Iz=I_BEAM, Iy=I_BEAM,
            G=E / 2.6, J=2.0 * I_BEAM)
    stiff = ops.uniaxialMaterial.ElasticMaterial(E=K_STIFF)
    spring = ops.uniaxialMaterial.ElasticMaterial(E=K_ROT)
    rz = 3 if ndm == 2 else 6
    ops.element.ZeroLength(
        nodes=(gnd, ops.nodes.one(label="base").tag),
        mat_dirs=tuple(
            ZeroLengthMatDir(material=spring if d == rz else stiff, dof=d)
            for d in range(1, ndf + 1)),
    )
    ops.fix(nodes=[gnd], dofs=(1,) * ndf)
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=ops.nodes.one(label="tip"),
               forces=(0.0, -P_TIP) + (0.0,) * (ndf - 2))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


RZ_SPRING = -(P_TIP * L_BEAM ** 2 / (2.0 * E * I_BEAM) + P_TIP * L_BEAM / K_ROT)
M_GND = P_TIP * L_BEAM


@pytest.mark.parametrize("ndm", [2, 3])
def test_declare_results_and_read_agree_on_a_zero_length_node(
    ndm: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    fem, gnd = build_spring_cantilever_fem(ndm)

    # The deck route: recorder.declare + a deck run + ops.results.
    ops = spring_cantilever_bridge(fem, gnd, ndm)
    ops.recorder.declare(nodes=("rotation_z",), label="tip",
                         file_root=str(tmp_path), name="tip")
    ops.recorder.declare(nodes=("reaction_moment_z",), ids=(gnd,),
                         file_root=str(tmp_path), name="gnd")
    deck = tmp_path / "deck.py"
    ops.py(str(deck), run=True, analyze_steps=1, python=sys.executable)
    r = ops.results(str(deck))
    deck_rz = r.nodes.get(label="tip", component="rotation_z").item()
    deck_m = r.nodes.get(ids=[gnd], component="reaction_moment_z").item()
    assert deck_rz == pytest.approx(RZ_SPRING, rel=1e-9)
    assert deck_m == pytest.approx(M_GND, rel=1e-9)

    # The live route: analyze + .read, on a fresh bridge.
    live = spring_cantilever_bridge(fem, gnd, ndm)
    assert live.analyze(steps=1) == 0
    read_rz = live.nodes.one(label="tip").read("rotation_z")
    read_m = live.nodes[gnd].read("reaction_moment_z")
    assert read_rz == pytest.approx(deck_rz, rel=1e-9)
    assert read_m == pytest.approx(deck_m, rel=1e-9)
