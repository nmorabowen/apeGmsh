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


def _ops(master_fix, handler: "str | None", *, load: bool = True,
         ndm: int = 3, normal=(0.0, 0.0, 1.0)) -> apeSees:
    ndf = 6 if ndm == 3 else 3
    fem = make_two_column_frame(ndm=ndm)
    fem.add_node_constraints([NodeGroupRecord(
        kind=ConstraintKind.RIGID_DIAPHRAGM, master_node=2, slave_nodes=[4],
        dofs=[1, 2, 6], plane_normal=np.array(normal), name="floor")])
    ops = apeSees(cast("Any", fem))
    ops.model(ndm=ndm, ndf=ndf)
    if ndm == 3:
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        ops.element.elasticBeamColumn(pg="Cols", transf=transf, A=0.01,
                                      E=200e9, Iz=1e-4, Iy=1e-4, G=80e9,
                                      J=1e-4)
    else:
        transf = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(pg="Cols", transf=transf, A=0.01,
                                      E=200e9, Iz=1e-4)
    ops.fix(pg="Base", dofs=(1,) * ndf)
    if any(master_fix):
        ops.fix(nodes=[2], dofs=master_fix)
    if load:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=4, forces=(1000.0,) + (0.0,) * (ndf - 1))
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


@pytest.mark.parametrize("ndm, normal, perp, master_fix, refused", [
    # 2-D: perp 3 ties ux, uy, rz = DOFs 1, 2, 3 (RigidDiaphragm.cpp)
    (2, (0, 0, 1), 3, (0, 0, 1), True),
    (2, (0, 0, 1), 3, (1, 0, 0), True),
    (2, (0, 0, 1), 3, (1, 1, 1), False),
    # 3-D, perp 1 ties uy, uz, rx; perp 2 ties ux, uz, ry
    (3, (1, 0, 0), 1, (0, 1, 0, 0, 0, 0), True),
    (3, (1, 0, 0), 1, (1, 0, 0, 0, 1, 1), False),     # out of plane
    (3, (1, 0, 0), 1, (0, 1, 1, 1, 0, 0), False),     # fully in plane
    (3, (0, 1, 0), 2, (0, 0, 0, 0, 1, 0), True),
    (3, (0, 1, 0), 2, (0, 1, 0, 1, 0, 1), False),     # out of plane
])
def test_the_in_plane_dofs_follow_ndm_and_perp(
        ndm, normal, perp, master_fix, refused, tmp_path) -> None:
    ops = _ops(master_fix, "Transformation", ndm=ndm, normal=normal)
    if refused:
        with pytest.raises(BridgeError, match="#2427"):
            _tcl(ops, tmp_path / "m.tcl")
    else:
        assert f"rigidDiaphragm {perp} 2 4" in _tcl(ops, tmp_path / "m.tcl")


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


@pytest.mark.parametrize("handler, refused", [
    ("Transformation", True), ("Lagrange", False),
    (None, True),     # runs under the auto-emitted global Transformation
])
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
            constraints=(getattr(ops.constraints, handler)()
                         if handler else None),
            numberer=ops.numberer.RCM(), system=ops.system.UmfPack(),
            analysis=ops.analysis.Static())
        st.run(n_increments=1)
    if refused:
        with pytest.raises(BridgeError, match="#2427"):
            _tcl(ops, tmp_path / "m.tcl")
    else:
        assert "rigidDiaphragm 3 2 4" in _tcl(ops, tmp_path / "m.tcl")


@pytest.mark.live
def test_a_2d_master_held_on_rz_only_diverges_upstream() -> None:
    """2-D perp 3 ties DOFs 1, 2, 3: a master held on rz alone is a
    refused case there (the 3-D table would call DOF 3 out of plane)."""
    ops = pytest.importorskip("openseespy.opensees")

    def run(handler: str, master_fix) -> list:
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.node(1, 0.3, 1.0)
        for i, x in enumerate((-1.0, 1.0)):
            ops.node(2 + i, x, 1.0)
            ops.node(12 + i, x, 0.0)
            ops.fix(12 + i, 1, 1, 1)
        ops.fix(1, *master_fix)
        ops.geomTransf("Linear", 1)
        for i in range(2):
            ops.element("elasticBeamColumn", 1 + i, 12 + i, 2 + i,
                        1.0, 1.0e3, 1.0 + i, 1)
        ops.rigidDiaphragm(3, 1, 2, 3)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(3, 1.0, 0.0, 0.0)
        ops.constraints(handler)
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        for _ in range(4):
            assert ops.analyze(1) == 0
        out = [ops.nodeDisp(1, d) for d in (1, 2, 3)]
        ops.wipe()
        return out

    t, lag = run("Transformation", (0, 0, 1)), run("Lagrange", (0, 0, 1))
    assert abs(lag[0]) > 0
    assert not np.allclose(t, lag, rtol=1e-3)
    free = (0, 0, 0)
    assert np.allclose(run("Transformation", free), run("Lagrange", free),
                       rtol=1e-6, atol=1e-12)
