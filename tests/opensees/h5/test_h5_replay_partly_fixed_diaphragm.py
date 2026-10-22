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
