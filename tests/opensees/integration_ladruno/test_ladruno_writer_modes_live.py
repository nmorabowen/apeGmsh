"""#2178 (X slice 9a): the Python modes writer equals the fork recorder.

One bridge model is emitted into the fork with a ``recorder ladruno -N
modesOfVibration modesOfVibrationRotational`` and solved by
``ops.eigen(3)``; the recorder records the solve once and closes, and
``EigenResult.to_ladruno`` then writes the same modes from Python. The
two files must agree, information for information:

The models are a 2-D cantilever, a 3-D frame and a 3-D truss at ndf 3
(whose rotational group the recorder fills with zeros).

1. both pass the fork's own ``ladruno_format.validate()`` at ``FORK_PIN``;
2. ``INFO`` (``GENERATOR``, the format versions, ``SPATIAL_DIM``, the
   precision and the partition manifest) and the one stage's ``KIND``;
3. ``MODEL/NODES`` per node id;
4. the modal groups: names, attributes (``DIMENSION`` absent in both),
   row ids, mode keys, eigenvalues
   and ``OMEGA`` / ``FREQUENCY`` / ``PERIOD`` (rtol 1e-10), and every
   mode shape up to sign;
5. ``Results.from_ladruno(..., model_h5=)`` reads the same modes from
   both.

What the Python file leaves out is declared: no ``MODEL/ELEMENTS``
buckets, and ``PROVENANCE@unsupported == ["elements"]``.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

import h5py
import numpy as np
import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.live import get_ops
from apeGmsh.results import Results

from tests.opensees.integration_ladruno import _fork_spec
from tests.opensees.integration_ladruno.test_ladruno_modes_live import (
    MODAL_TOKENS,
    modal_bridge,
)
from tests.opensees.live.test_readback_live import build_cantilever_fem
from tests.test_eigen_to_native import _build_fem, _frame

pytestmark = pytest.mark.ladruno_fork

N_MODES = 3
_RTOL = 1e-10
_INFO = ("GENERATOR", "FORMAT_VERSION", "FORMAT_MINOR_VERSION", "SPATIAL_DIM",
         "STORED_PRECISION", "PARTITIONED", "PARTITION_ID", "NUM_PARTITIONS")
#: ``DIMENSION`` is compared for presence too: the recorder skips an empty
#: string attribute, so neither file carries it.
_GROUP_ATTRS = ("COMPONENTS", "DATA_TYPE", "DESCRIPTION", "DIMENSION",
                "DISPLAY_NAME", "TYPE")
_MODE_SCALARS = ("LAMBDA", "OMEGA", "FREQUENCY", "PERIOD")


def _cantilever_2d() -> apeSees:
    return modal_bridge(build_cantilever_fem())


def _frame_3d() -> apeSees:
    return _frame(_build_fem(3), ndm=3)


def _truss_3d() -> apeSees:
    """ndm 3, ndf 3: no rotations, so the recorder's MODES_OF_VIBRATION(R)
    holds zeros. Axial modes only (transverse DOFs fixed): 3 free DOFs, so
    ARPACK finds 2 modes."""
    ops = apeSees(_build_fem(3), default_orientation=None)
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=2.0e11, name="steel")
    ops.element.Truss(pg="Beam", A=1e-2, material=mat)
    ops.fix(pg="Beam", dofs=(0, 1, 1))
    ops.fix(label="base", dofs=(1, 0, 0))
    ops.mass(pg="Beam", values=(10.0, 10.0, 10.0))
    return ops


@pytest.fixture(scope="module")
def spec(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ModuleType]:
    where = _fork_spec.fetch(tmp_path_factory.mktemp("ladruno_spec"))
    with _fork_spec.ladruno_format(where) as module:
        yield module


def _attr(obj: Any, name: str) -> Any:
    return np.asarray(obj.attrs[name]).tolist()


def _sole_stage(f: h5py.File) -> h5py.Group:
    stages = [k for k in f if k.startswith("MODEL_STAGE[")]
    assert len(stages) == 1, stages
    return f[stages[0]]


def _rows(grp: h5py.Group) -> dict[int, int]:
    ids = np.asarray(grp["ID"][()]).ravel()
    return {int(t): i for i, t in enumerate(ids)}


def _compare(ours: Path, fork: Path, n_modes: int) -> None:
    with h5py.File(ours, "r") as fo, h5py.File(fork, "r") as ff:
        for name in _INFO:
            if name in ff["INFO"].attrs or name in fo["INFO"].attrs:
                assert _attr(fo["INFO"], name) == _attr(ff["INFO"], name), name
        so, sf = _sole_stage(fo), _sole_stage(ff)
        assert _attr(so, "KIND") == _attr(sf, "KIND") == [b"eigen"]

        no, nf = so["MODEL/NODES"], sf["MODEL/NODES"]
        ro, rf = _rows(no), _rows(nf)
        assert sorted(ro) == sorted(rf)
        for nid in ro:
            np.testing.assert_allclose(no["COORDINATES"][ro[nid]],
                                       nf["COORDINATES"][rf[nid]], rtol=0,
                                       atol=1e-12)
        assert list(so["MODEL/ELEMENTS"]) == []   # declared: unsupported

        go, gf = so["RESULTS/ON_NODES"], sf["RESULTS/ON_NODES"]
        assert sorted(go) == sorted(gf)
        for gname in gf:
            a, b = go[gname], gf[gname]
            for attr in _GROUP_ATTRS:
                assert (attr in a.attrs) == (attr in b.attrs), (gname, attr)
                if attr in b.attrs:
                    assert _attr(a, attr) == _attr(b, attr), (gname, attr)
            ra, rb = _rows(a), _rows(b)
            assert sorted(ra) == sorted(rb)
            # One modal set each. Its STEP_<s> key is the recorder's commit
            # counter, which an earlier run in the same process moves (the
            # second case here reads STEP_1); the Python file has no
            # commits and writes STEP_0.
            assert len(a["DATA"]) == len(b["DATA"]) == 1
            for ma, mb in zip(a["DATA"].values(), b["DATA"].values()):
                assert sorted(ma) == sorted(mb)
                assert len(mb) == n_modes
                for mode in mb:
                    assert _attr(ma[mode], "MODE") == _attr(mb[mode], "MODE")
                    for s in _MODE_SCALARS:
                        assert float(ma[mode].attrs[s][0]) == pytest.approx(
                            float(mb[mode].attrs[s][0]), rel=_RTOL), (mode, s)
                    va = np.asarray(ma[mode][()])
                    vb = np.asarray(mb[mode][()])
                    order = [ra[nid] for nid in sorted(rb)]
                    va = va[order]
                    vb = vb[[rb[nid] for nid in sorted(rb)]]
                    sign = 1.0 if float(np.sum(va * vb)) >= 0 else -1.0
                    scale = max(float(np.max(np.abs(vb))), 1e-300)
                    np.testing.assert_allclose(sign * va, vb, rtol=0,
                                               atol=_RTOL * scale)


def _modes(path: Path, archive: Path) -> list[tuple[float, float, float]]:
    with Results.from_ladruno(path, model_h5=archive) as r:
        return [(m.eigenvalue, m.frequency_hz, m.period_s)
                for m in sorted(r.modes, key=lambda m: m.mode_index)]


@pytest.mark.parametrize(
    "bridge,n_modes",
    [(_cantilever_2d, N_MODES), (_frame_3d, N_MODES), (_truss_3d, 2)],
    ids=["cantilever_2d", "frame_3d", "truss_3d_ndf3"],
)
def test_python_writer_equals_the_fork_recorder(
    bridge: Callable[[], apeSees], n_modes: int, spec: ModuleType,
    tmp_path: Path,
) -> None:
    fork = tmp_path / "fork.ladruno"
    ops = bridge()
    ops.recorder.Ladruno(file=str(fork), nodal_responses=MODAL_TOKENS)
    res = ops.eigen(n_modes)
    live = get_ops()
    live.record()
    live.remove("recorders")      # closes the recorder's file

    ours = res.to_ladruno(tmp_path / "ours.ladruno")
    archive = ours.with_suffix(".h5")

    assert spec.validate(str(fork)) == []
    assert spec.validate(str(ours)) == []
    _compare(ours, fork, n_modes)
    with h5py.File(ours, "r") as f:
        prov = dict(f["INFO/PROVENANCE"].attrs)
    assert json.loads(prov["unsupported"]) == ["elements"]
    assert str(prov["writer"]).startswith("apeGmsh eigen->ladruno ")

    got, want = _modes(ours, archive), _modes(fork, archive)
    assert len(got) == len(want) == n_modes
    for g, w in zip(got, want):
        assert g == pytest.approx(w, rel=_RTOL)
