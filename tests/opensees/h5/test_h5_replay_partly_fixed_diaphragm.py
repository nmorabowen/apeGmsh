"""#2427 on the replay path (#2435).

An archive written before the emit-time refusal (#2431) can carry a
``rigidDiaphragm`` whose master is fixed on some in-plane DOFs only,
under Transformation. ``OpenSeesModel.from_h5(...).build`` must refuse
it before emitting anything, as the bridge now does. The legacy archive
is written here with the emit-time check patched out.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.build import constraints as _c
from apeGmsh.opensees._internal.build.errors import BridgeError
from tests.opensees.h5.test_h5_mp_constraints_replay import (
    _KINDS, _bridge, _fem,
)

PARTLY, OUT_OF_PLANE = (0, 1, 1, 1, 1, 0), (0, 0, 1, 1, 1, 0)


def _chain(o, handler: str) -> dict:
    return dict(
        test=o.test.NormDispIncr(tol=1e-9, max_iter=5),
        algorithm=o.algorithm.Linear(),
        integrator=o.integrator.LoadControl(dlam=1.0),
        constraints=getattr(o.constraints, handler)(),
        numberer=o.numberer.RCM(), system=o.system.UmfPack(),
        analysis=o.analysis.Static())


def _legacy_archive(path: Path, monkeypatch, *, master_fix, handler: str,
                    staged: bool) -> Path:
    o = _bridge(_fem(partitioned=False, **_KINDS["rigid_diaphragm"]))
    o.fix(nodes=[2], dofs=master_fix)
    if staged:
        with o.stage(name="s") as st:
            with st.pattern(series=o.timeSeries.Linear()) as p:
                p.load(node=4, forces=(1.0, 0, 0, 0, 0, 0))
            st.analysis(**_chain(o, handler))
            st.run(n_increments=1)
    else:
        getattr(o.constraints, handler)()
        o.numberer.RCM()
        o.system.UmfPack()
        o.test.NormDispIncr(tol=1e-9, max_iter=5)
        o.algorithm.Linear()
        o.integrator.LoadControl(dlam=1.0)
        o.analysis.Static()
    with monkeypatch.context() as m:       # what a pre-#2431 bridge wrote
        m.setattr(_c, "_refuse_partly_fixed_diaphragm",
                  lambda *a, **k: None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o.h5(str(path))
    return path


def _replay(path: Path) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OpenSeesModel.from_h5(str(path)).build("tcl", flat=True)


@pytest.mark.parametrize("staged", [False, True])
def test_a_legacy_partly_fixed_diaphragm_is_refused_on_replay(
        staged, monkeypatch, tmp_path) -> None:
    path = _legacy_archive(tmp_path / "legacy.h5", monkeypatch,
                           master_fix=PARTLY, handler="Transformation",
                           staged=staged)
    with pytest.raises(BridgeError, match="#2427"):
        _replay(path)


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("master_fix, handler", [
    (OUT_OF_PLANE, "Transformation"), (PARTLY, "Lagrange")])
def test_exact_archives_still_replay(
        staged, master_fix, handler, monkeypatch, tmp_path) -> None:
    path = _legacy_archive(tmp_path / "ok.h5", monkeypatch,
                           master_fix=master_fix, handler=handler,
                           staged=staged)
    assert "rigidDiaphragm 3 2 4" in _replay(path)


def _transformation_chain(o) -> None:
    o.constraints.Transformation()
    o.numberer.RCM()
    o.system.UmfPack()
    o.test.NormDispIncr(tol=1e-9, max_iter=5)
    o.algorithm.Linear()
    o.integrator.LoadControl(dlam=1.0)
    o.analysis.Static()


@pytest.mark.parametrize("staged", [False, True])
def test_a_pattern_sp_completing_the_plane_replays(staged, tmp_path) -> None:
    """Emit counts a pattern ``p.sp`` on the master as varying, so ux, uy
    fixed plus an SP on rz is not refused; the archive must replay."""
    o = _bridge(_fem(partitioned=False, **_KINDS["rigid_diaphragm"]))
    o.fix(nodes=[2], dofs=(1, 1, 1, 1, 1, 0))
    if staged:
        with o.stage(name="s") as st:
            with st.pattern(series=o.timeSeries.Linear()) as p:
                p.sp(node=2, dof=6, value=1e-3)
            st.analysis(**_chain(o, "Transformation"))
            st.run(n_increments=1)
    else:
        with o.pattern.Plain(series=o.timeSeries.Linear()) as p:
            p.sp(node=2, dof=6, value=1e-3)
        _transformation_chain(o)
    path = tmp_path / "sp.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o.h5(str(path))                       # no patch: a current archive
    assert "rigidDiaphragm 3 2 4" in _replay(path)


@pytest.mark.live
def test_a_live_build_refuses_before_wiping_the_domain(
        monkeypatch, tmp_path) -> None:
    osp = pytest.importorskip("openseespy.opensees")
    path = _legacy_archive(tmp_path / "legacy.h5", monkeypatch,
                           master_fix=PARTLY, handler="Transformation",
                           staged=False)
    osp.wipe()
    osp.model("basic", "-ndm", 2, "-ndf", 2)
    osp.node(77, 0.0, 0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(BridgeError, match="#2427"):
                OpenSeesModel.from_h5(str(path)).build("live")
        assert list(osp.getNodeTags()) == [77]
    finally:
        osp.wipe()


def test_a_tie_into_a_diaphragm_master_is_a_chain(tmp_path) -> None:
    """#2396 would fix the diaphragm master to rewrite an equalDOF into
    it, making the master partly fixed under the diaphragm. A diaphragm
    node is a chain node: the tie keeps its form and warns, and the deck
    writes, then replays."""
    from apeGmsh._kernel.records._constraints import (
        NodeGroupRecord, NodePairRecord,
    )
    from apeGmsh.opensees._internal.build.constraints import (
        PartlyFixedTieWarning,
    )
    import numpy as np

    o = _bridge(_fem(partitioned=False, node_constraints=[
        NodeGroupRecord(kind="rigid_diaphragm", master_node=2,
                        slave_nodes=[4], plane_normal=np.array([0., 0., 1.])),
        NodePairRecord(kind="equal_dof", master_node=4, slave_node=2,
                       dofs=[1, 2])]))
    o.fix(nodes=[4], dofs=(1, 0, 0, 0, 0, 0))
    _transformation_chain(o)
    path = tmp_path / "chain.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("always", PartlyFixedTieWarning)
        with pytest.warns(PartlyFixedTieWarning, match="chain"):
            o.h5(str(path))
    text = _replay(path)
    assert "equalDOF 4 2 1 2" in text and "fix 2 " not in text
