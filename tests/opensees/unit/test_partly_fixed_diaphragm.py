"""A ``rigidDiaphragm`` whose master is partly fixed in plane (#2427).

Under OpenSees' Transformation handler a diaphragm master held on some
of its in-plane DOFs (ux, uy, rz for ``perpDirn`` 3) and free on others
accumulates its total in-plane displacement at every step: the slaves
are right, the master is not. The bridge refuses that deck; a master
free or fully held in plane, or fixed only out of plane, is exact and
emits silently.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import NodeGroupRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from tests.opensees.fixtures.fem_stub import make_two_column_frame


def _ops(master_fix, handler: "str | None", *, load: bool = True) -> apeSees:
    fem = make_two_column_frame()
    fem.add_node_constraints([NodeGroupRecord(
        kind=ConstraintKind.RIGID_DIAPHRAGM, master_node=2, slave_nodes=[4],
        dofs=[1, 2, 6], plane_normal=np.array([0.0, 0.0, 1.0]),
        name="floor")])
    ops = apeSees(cast("Any", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(pg="Cols", transf=transf, A=0.01, E=200e9,
                                  Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    if any(master_fix):
        ops.fix(nodes=[2], dofs=master_fix)
    if load:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=4, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    if handler is not None:
        getattr(ops.constraints, handler)(
            **({"alpha_sp": 1e12, "alpha_mp": 1e12}
               if handler == "Penalty" else {}))
    return ops


def _tcl(ops: apeSees, path: Path) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # auto-emit notices
        ops.tcl(str(path))
    return path.read_text()


@pytest.mark.parametrize("handler", ["Transformation", None])
@pytest.mark.parametrize("master_fix", [(0, 1, 0, 0, 0, 0),
                                        (1, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 1),
                                        (1, 1, 1, 1, 1, 0)])
def test_partly_fixed_in_plane_is_refused(master_fix, handler, tmp_path):
    with pytest.raises(BridgeError, match="#2427") as err:
        _tcl(_ops(master_fix, handler), tmp_path / "m.tcl")
    assert "Lagrange" in str(err.value) and "Penalty" in str(err.value)


@pytest.mark.parametrize("master_fix", [(0, 0, 1, 1, 1, 0),   # out of plane
                                        (1, 1, 0, 0, 0, 1),   # fully in plane
                                        (1, 1, 1, 1, 1, 1),
                                        (0, 0, 0, 0, 0, 0)])
def test_exact_fixities_emit_silently(master_fix, tmp_path) -> None:
    text = _tcl(_ops(master_fix, "Transformation"), tmp_path / "m.tcl")
    assert "rigidDiaphragm 3 2 4" in text


@pytest.mark.parametrize("handler", ["Lagrange", "Penalty"])
def test_other_handlers_emit(handler, tmp_path) -> None:
    text = _tcl(_ops((0, 1, 0, 0, 0, 0), handler), tmp_path / "m.tcl")
    assert "rigidDiaphragm 3 2 4" in text


@pytest.mark.live
def test_the_refused_case_diverges_upstream() -> None:
    """Raw openseespy: the refused fixity is wrong under Transformation,
    and the out-of-plane control is not (keeps the refusal honest)."""
    ops = pytest.importorskip("openseespy.opensees")

    def run(handler: str, master_fix) -> list:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0.0, 0.0, 1.0)
        for i, (x, y) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            ops.node(2 + i, x, y, 1.0)
            ops.node(12 + i, x, y, 0.0)
            ops.fix(12 + i, 1, 1, 1, 1, 1, 1)
        ops.fix(1, *master_fix)
        ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
        for i in range(4):
            ops.element("elasticBeamColumn", 1 + i, 12 + i, 2 + i,
                        1.0, 1.0e3, 400.0, 1.0, 1.0, 1.0, 1)
        ops.rigidDiaphragm(3, 1, 2, 3, 4, 5)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(3, 1.0, 0.5, 0, 0, 0, 0)
        ops.constraints(handler)
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        for _ in range(4):
            assert ops.analyze(1) == 0
        out = [ops.nodeDisp(1, d) for d in (1, 2, 6)]
        ops.wipe()
        return out

    refused, control = (0, 1, 1, 1, 1, 0), (0, 0, 1, 1, 1, 0)
    t, lag = run("Transformation", refused), run("Lagrange", refused)
    assert abs(lag[0]) > 0
    assert not np.allclose(t, lag, rtol=1e-3)            # 10x vs 4x
    assert np.allclose(run("Transformation", control),
                       run("Lagrange", control), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("handler, refused", [("Transformation", True),
                                              ("Lagrange", False)])
@pytest.mark.parametrize("claim", [False, True])
def test_staged_decks_check_every_stage_handler(
        handler, refused, claim, tmp_path) -> None:
    ops = _ops((0, 1, 0, 0, 0, 0), None, load=False)
    with ops.stage(name="s") as st:
        if claim:
            st.rigid_diaphragm(name="floor")
        with st.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=4, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        st.analysis(
            test=ops.test.NormDispIncr(tol=1e-9, max_iter=5),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=getattr(ops.constraints, handler)(),
            numberer=ops.numberer.RCM(), system=ops.system.UmfPack(),
            analysis=ops.analysis.Static())
        st.run(n_increments=1)
    if refused:
        with pytest.raises(BridgeError, match="#2427"):
            _tcl(ops, tmp_path / "m.tcl")
    else:
        assert "rigidDiaphragm 3 2 4" in _tcl(ops, tmp_path / "m.tcl")
