"""Fork lane: per-material damage and Tri31 strain through the spec MPCO emit.

Gap G5 of the DomainCapture parity table (#1767, #1945; slice #1952).
DomainCapture reads two responses per Gauss point through
``eleResponse(e, "material", gp, <resp>)``: ``damage`` on a catalogued host,
and ``strain`` on Tri31, which has no element-level ``strains`` branch. This
file proves the spec MPCO emit (``mpco_ops_args``) carries both, on the
module the bridge drives (``get_ops()``, the fork on the ``live-fork`` lane;
pip ``openseespy`` 3.8.0 ships no MPCO recorder):

* **Damage:** a two-``Brick`` column of ``ASDConcrete3D`` pulled past
  cracking. ``damage_tension`` / ``damage_compression`` read through
  ``Results.from_mpco`` equal ``eleResponse(e, "material", gp, "damage")``
  at every step and GP, and the ``material.damage`` bucket is 2 wide per GP
  (the card's stop condition).
* **Tri31 strain:** ``strain_xx/yy/xy`` of two ``Tri31`` equal
  ``eleResponse(e, "material", "1", "strain")``. The plate also holds a
  ``FourNodeQuad``: MPCO has no per-token element filter, so
  ``material.strain`` records the quad too. Its read-back must still equal
  ``eleResponse(e, "strains")``, with and without its own ``strains`` record.

Authority: ``eleResponse`` on the live domain after each step, independent of
the emit and the reader under test. Tags are not 1..n.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees.emitter.live import get_ops
from apeGmsh.results import Results
from apeGmsh.results.spec._emit import mpco_ops_args
from apeGmsh.results.spec._resolved import ResolvedRecorderRecord

from tests.conftest import _stub_model_h5_path

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10

E, FT = 30e9, 2.0e6
N_PULL = 10
U_TOP = 3.0 * FT / E                   # 1 m tall: 3x the strain at f_t

BRICK_TAGS = (7, 9)
#: ``eleResponse(e, "material", gp, "strain")`` order on Tri31 (plane):
#: spelled out, not read from RESPONSE_CATALOG (the reader's layout source).
STRAIN_2D = ("strain_xx", "strain_yy", "strain_xy")
TRI_TAGS = (11, 12)
QUAD_TAG = 13
N_PLATE = 2


def _gauss(name: str, comps: tuple[str, ...], cls: str,
           ids: tuple[int, ...]) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="gauss", name=name, components=comps, dt=None, n_steps=None,
        element_ids=np.array(ids), element_class_name=cls,
    )


def _read(path: Path, comp: str) -> tuple[np.ndarray, np.ndarray]:
    with Results.from_mpco(str(path), model_h5=_stub_model_h5_path()) as r:
        slab = r.stage(r.stages[0].id).elements.gauss.get(component=comp)
        return (np.asarray(slab.element_index, dtype=int),
                np.asarray(slab.values, dtype=float))


def _close(want: np.ndarray, got: np.ndarray, msg: str) -> None:
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=RTOL * float(np.abs(want).max()),
        err_msg=msg,
    )


# ---------------------------------------------------------------------------
# Damage: Brick + ASDConcrete3D
# ---------------------------------------------------------------------------


def test_mpco_damage_matches_eleresponse(tmp_path: Path) -> None:
    rec = _gauss("dmg", ("damage_tension", "damage_compression"), "Brick",
                 BRICK_TAGS)
    args = mpco_ops_args([rec], output_dir=str(tmp_path), filename="b.mpco")
    assert args == ("mpco", f"{tmp_path}/b.mpco", "-E", "material.damage")
    path = tmp_path / "b.mpco"

    ops = get_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)

    def nid(ix: int, iy: int, iz: int) -> int:
        return 100 * iz + 10 * ix + iy + 1

    for ix in range(3):
        for iy in range(2):
            for iz in range(2):
                ops.node(nid(ix, iy, iz), float(ix), float(iy), float(iz))
            # Bottom on rollers, x held on x=0 and y on y=0: free Poisson.
            ops.fix(nid(ix, iy, 0), int(ix == 0), int(iy == 0), 1)
    ops.nDMaterial("ASDConcrete3D", 1, E, 0.2, "-fc", 30e6, "-ft", FT)
    # Side by side, not stacked: both see the same strain, no localisation.
    for ix, tag in enumerate(BRICK_TAGS):
        face = ((ix, 0), (ix + 1, 0), (ix + 1, 1), (ix, 1))
        ops.element("stdBrick", tag, *(nid(x, y, 0) for x, y in face),
                    *(nid(x, y, 1) for x, y in face), 1)
    ops.recorder(*args)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for ix in range(3):
        for iy in range(2):
            ops.sp(nid(ix, iy, 1), 3, U_TOP)
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-10, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / N_PULL)
    ops.analysis("Static")
    truth = []                                # (step, elem, gp, [t, c])
    for _ in range(N_PULL):
        assert ops.analyze(1) == 0
        truth.append([[list(ops.eleResponse(e, "material", str(gp), "damage"))
                       for gp in range(1, 9)] for e in BRICK_TAGS])
    ops.remove("recorders")
    ops.wipe()
    want = np.asarray(truth, dtype=float)
    assert want.shape == (N_PULL, 2, 8, 2)
    assert want[-1, :, :, 0].max() > 0.0     # cracked in tension
    assert np.all(want[0, :, :, 0] == 0.0)   # first step is elastic

    # Measured layout (the card's stop condition): Brick bucket, 2 per GP.
    with h5py.File(path, "r") as h:
        grp = h["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/material.damage"]
        (key,) = [k for k in grp if "-Brick[" in k]
        assert grp[key]["META/NUM_COMPONENTS"][()].ravel().tolist() == [2] * 8

    for k, comp in enumerate(("damage_tension", "damage_compression")):
        eidx, got = _read(path, comp)
        assert eidx.tolist() == [t for t in BRICK_TAGS for _ in range(8)]
        assert got.shape == (N_PULL, 16), (comp, got.shape)
        _close(want[..., k].reshape(N_PULL, 16), got, comp)


# ---------------------------------------------------------------------------
# Tri31 strain, in a plate that also holds a FourNodeQuad
# ---------------------------------------------------------------------------


def _solve_plate(path: Path, records: list[ResolvedRecorderRecord]):
    ops = get_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for nid, xy in {21: (0.0, 0.0), 22: (1.0, 0.1), 23: (1.1, 1.0),
                    24: (0.0, 0.9), 25: (2.0, 0.0), 26: (2.1, 1.2)}.items():
        ops.node(nid, *xy)
    ops.fix(21, 1, 1)
    ops.fix(24, 1, 0)
    ops.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3)
    ops.element("tri31", TRI_TAGS[0], 21, 22, 23, 1.0, "PlaneStress", 1)
    ops.element("tri31", TRI_TAGS[1], 21, 23, 24, 1.0, "PlaneStress", 1)
    ops.element("quad", QUAD_TAG, 22, 25, 26, 23, 1.0, "PlaneStress", 1)
    ops.recorder(*mpco_ops_args(records, output_dir=str(path.parent),
                                filename=path.name))
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(25, 1.0e6, 2.0e5)
    ops.load(26, 3.0e5, -4.0e5)
    ops.load(23, -1.0e5, 2.0e5)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("UmfPack")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0 / N_PLATE)
    ops.analysis("Static")
    tri, quad = [], []
    for _ in range(N_PLATE):
        assert ops.analyze(1) == 0
        tri.append([list(ops.eleResponse(e, "material", "1", "strain"))
                    for e in TRI_TAGS])
        quad.append(np.asarray(ops.eleResponse(QUAD_TAG, "strains"),
                               dtype=float).reshape(4, 3))
    assert list(ops.eleResponse(TRI_TAGS[0], "strains")) == []
    ops.remove("recorders")
    ops.wipe()
    tri_a = np.asarray(tri, dtype=float)          # (step, tri, 3)
    assert tri_a.shape == (N_PLATE, 2, 3)
    assert np.all(tri_a[-1] != 0.0)
    return tri_a, np.asarray(quad)                # quad: (step, 4 gp, 3)


TRI_REC = _gauss("tri", STRAIN_2D, "Tri31", TRI_TAGS)
QUAD_REC = _gauss("quad", STRAIN_2D, "FourNodeQuad", (QUAD_TAG,))


@pytest.mark.parametrize("with_quad_record", [False, True],
                         ids=["tri-only", "tri+quad"])
def test_mpco_tri31_strain_matches_eleresponse(
    tmp_path: Path, with_quad_record: bool,
) -> None:
    records = [TRI_REC, QUAD_REC] if with_quad_record else [TRI_REC]
    toks = mpco_ops_args(records)[3:]
    assert toks == (("material.strain", "strains") if with_quad_record
                    else ("material.strain",))
    path = tmp_path / "plate.mpco"
    tri, quad = _solve_plate(path, records)

    with h5py.File(path, "r") as h:
        grp = h["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/material.strain"]
        (key,) = [k for k in grp if "-Tri31[" in k]
        assert grp[key]["META/NUM_COMPONENTS"][()].ravel().tolist() == [3]
        # No per-token filter in MPCO: the quad is in this bucket too.
        assert any("-FourNodeQuad[" in k for k in grp)

    for k, comp in enumerate(STRAIN_2D):
        eidx, got = _read(path, comp)
        assert got.shape == (N_PLATE, len(eidx)), (comp, got.shape)
        on_tri = np.isin(eidx, TRI_TAGS)
        assert eidx[on_tri].tolist() == list(TRI_TAGS)
        _close(tri[..., k], got[:, on_tri], f"{comp} on Tri31")
        on_quad = eidx == QUAD_TAG
        assert int(on_quad.sum()) == 4, eidx.tolist()
        _close(quad[..., k], got[:, on_quad], f"{comp} on FourNodeQuad")
