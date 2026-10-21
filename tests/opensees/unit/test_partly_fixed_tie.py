"""An ``equalDOF`` whose retained node is partly fixed (#2396).

OpenSees' Transformation handler (the bridge default for ``equalDOF``)
resets the constrained node's free tied DOF at every step when the
retained node carries an SP on another tied DOF
(``TransformationDOF_Group::enforceSPs``). The bridge rewrites such a tie
as ``equalDOF(m, s, free DOFs)`` + ``fix(s, fixed DOFs)``, the same
constraint set under every handler.

The model: two unit squares meshed apart, so their shared edge carries
coincident node pairs tied by ``g.constraints.equal_dof``; only square
A's base is fixed vertically, so the bottom pair's retained node is fixed
on DOF 2 and free on DOF 1.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.constraints import PartlyFixedTieWarning


@pytest.fixture(scope="module")
def squares() -> Any:
    g = apeGmsh(model_name="t2396", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A")
        g.model.geometry.add_rectangle(1, 0, 0, 1, 1, label="B")
        g.physical.add_surface("A", name="SA")
        g.physical.add_surface("B", name="SB")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.structured.set_transfinite_automatic()
        g.constraints.equal_dof("SA", "SB", dofs=[1, 2], name="iface")
        g.mesh.generation.generate(dim=2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()
    xyz = {int(n): c for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    masters = {int(r.master_node): int(r.slave_node)
               for r in fem.nodes.constraints}
    base_a = sorted(n for n, c in xyz.items()
                    if abs(c[1]) < 1e-9 and (c[0] < 1 - 1e-9 or n in masters))
    bottom_master = next(n for n in masters if abs(xyz[n][1]) < 1e-9)
    corner = min(base_a, key=lambda n: xyz[n][0])
    top_right = max(xyz, key=lambda n: (xyz[n][1], xyz[n][0]))
    return fem, base_a, bottom_master, masters[bottom_master], corner, top_right


def _ops(squares, *, staged_release: bool = False) -> apeSees:
    fem, base_a, master, _s, corner, top_right = squares
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3)
    ops.element.FourNodeQuad(pg="SA", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="SB", thickness=1.0, material=mat)
    ops.fix(nodes=base_a, dofs=(0, 1))
    ops.fix(nodes=[corner], dofs=(1, 0))
    if not staged_release:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=top_right, forces=(1.0, 0.0))
    else:
        chain = dict(
            test=ops.test.NormDispIncr(tol=1e-10, max_iter=5),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Lagrange(),
            numberer=ops.numberer.RCM(), system=ops.system.UmfPack(),
            analysis=ops.analysis.Static())
        with ops.stage(name="release") as s:
            s.remove_sp(nodes=[master], dofs=(2,))
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.load(node=top_right, forces=(1.0, 0.0))
            s.analysis(**chain)
            s.run(n_increments=1)
    return ops


def _lines(ops: apeSees, path: Path) -> list[str]:
    ops.tcl(str(path))
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip().startswith(("equalDOF", "fix "))]


def test_partly_fixed_master_is_rewritten(squares, tmp_path: Path) -> None:
    _fem, _b, m, s, _c, _t = squares
    lines = _lines(_ops(squares), tmp_path / "m.tcl")
    assert f"equalDOF {m} {s} 1" in lines          # DOF 2 left out ...
    assert f"fix {s} 0 1" in lines                 # ... and fixed on the slave
    others = [ln for ln in lines if ln.startswith("equalDOF")
              and not ln.startswith(f"equalDOF {m} ")]
    assert others and all(ln.endswith(" 1 2") for ln in others)


@pytest.mark.live
def test_transformation_now_matches_lagrange(squares) -> None:
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import get_ops

    u = {}
    for handler in ("Transformation", "Lagrange"):
        ops = _ops(squares)
        getattr(ops.constraints, handler)()
        ops.algorithm.Linear()
        ops.integrator.LoadControl(dlam=1.0)
        ops.numberer.RCM()
        ops.system.UmfPack()
        ops.analysis.Static()
        assert ops.analyze(steps=3) == 0
        osp = get_ops()
        u[handler] = np.array([[osp.nodeDisp(int(n), k) for k in (1, 2)]
                               for n in squares[0].nodes.ids])
    assert np.abs(u["Transformation"]).max() > 0
    assert np.allclose(u["Transformation"], u["Lagrange"],
                       atol=1e-9 * np.abs(u["Lagrange"]).max())


def test_a_staged_release_keeps_the_tie_and_warns(squares, tmp_path) -> None:
    _fem, _b, m, s, _c, _t = squares
    with pytest.warns(PartlyFixedTieWarning, match="#2396"):
        lines = _lines(_ops(squares, staged_release=True), tmp_path / "m.tcl")
    assert f"equalDOF {m} {s} 1 2" in lines


def test_the_archive_writes_the_rewritten_fix(squares, tmp_path) -> None:
    ops = _ops(squares)
    from apeGmsh.opensees._internal.artifact_write import BridgeArtifactWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", BridgeArtifactWarning)
        ops.h5(str(tmp_path / "arch.h5"))
