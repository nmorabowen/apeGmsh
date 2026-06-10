"""H5 round-trip of flat ASDAbsorbingBoundary3D/2D declarations (ADR 0054).

The ADR deferred an explicit check that absorbing elements survive the
``ops.h5`` -> ``OpenSeesModel.from_h5`` -> ``to_h5`` / ``build('tcl')``
cycle like any other element fan-out (raw G/v/rho[/thickness] + btype
string + optional ``-fx`` time-series ref in the args).  The staged flip
(``s.activate_absorbing``) is deliberately OUT of scope — staged-bucket
archival is ADR 0055 Phase 2; these models are flat.

Real plane-wave sessions (not stubs) so the broker zone is present and
the fan-out runs over a genuine skin.
"""
from __future__ import annotations

import collections
from pathlib import Path

import h5py
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.time_series.time_series import Path as PathSeries

from tests.opensees.h5.test_opensees_model import _assert_h5_equal

# Small grids; closed-form skin counts.
SKIN_3D = 108   # x=(40,4) y=(50,5) z=(30,3)
BOTTOM_3D = 42
SKIN_2D = 12    # x=(40,4) y=(30,3) -> L=R=3 B=4 BL=BR=1
BOTTOM_2D = 6


@pytest.fixture(scope="module")
def fem_3d():
    """(fem, skin) for a small 3D plane-wave box."""
    g = apeGmsh(model_name="abs_h5_3d", verbose=False)
    g.begin()
    try:
        skin = g.parts.add_plane_wave_box(
            x=(40.0, 4), y=(50.0, 5), z=(30.0, 3),
        )
        g.mesh.generation.generate(dim=3)
        yield g.mesh.queries.get_fem_data(), skin
    finally:
        g.end()


@pytest.fixture(scope="module")
def fem_2d():
    """(fem, skin) for a small 2D plane-wave box."""
    g = apeGmsh(model_name="abs_h5_2d", verbose=False)
    g.begin()
    try:
        skin = g.parts.add_plane_wave_box_2d(x=(40.0, 4), y=(30.0, 3))
        g.mesh.generation.generate(dim=2)
        yield g.mesh.queries.get_fem_data(), skin
    finally:
        g.end()


def _author_3d(fem_and_skin) -> apeSees:
    fem, skin = fem_and_skin
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    soil = ops.register(ElasticIsotropic(E=3410.0, nu=0.262, rho=2.4e-9))
    ts = ops.register(PathSeries(values=(0.0, 1.0, 0.0), dt=0.1))
    ops.element.stdBrick(pg=skin.soil_pg, material=soil)
    ops.element.absorbing_boundary(
        skin=skin, material=soil, base_series=ts, base_dirs=("x",),
    )
    return ops


def _author_2d(fem_and_skin) -> apeSees:
    fem, skin = fem_and_skin
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    soil = ops.register(ElasticIsotropic(E=3410.0, nu=0.262, rho=2.4e-9))
    ts = ops.register(PathSeries(values=(0.0, 1.0, 0.0), dt=0.1))
    ops.element.FourNodeQuad(pg=skin.soil_pg, thickness=1.0, material=soil)
    ops.element.absorbing_boundary(
        skin=skin, material=soil, base_series=ts, base_dirs=("x",),
        thickness=1.0,
    )
    return ops


def _absorbing_lines(tcl_text: str, token: str) -> list[str]:
    return [ln.strip() for ln in tcl_text.splitlines() if token in ln]


def _btype_tally(lines: list[str]) -> dict[str, int]:
    return dict(collections.Counter(
        ln.split()[-3] if "-fx" in ln else ln.split()[-1] for ln in lines
    ))


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def test_h5_writes_and_validates_3d(tmp_path: Path, fem_3d) -> None:
    ops = _author_3d(fem_3d)
    out = tmp_path / "abs3d.h5"
    ops.h5(str(out))
    with h5_reader.open(str(out)) as model:
        assert model.validate() == []
    om = OpenSeesModel.from_h5(out)
    recs = [r for r in om.elements()
            if r.type_token == "ASDAbsorbingBoundary3D"]
    assert len(recs) == SKIN_3D


def test_from_h5_to_h5_byte_equivalent_3d(tmp_path: Path, fem_3d) -> None:
    ops = _author_3d(fem_3d)
    src = tmp_path / "abs3d.h5"
    ops.h5(str(src))
    om = OpenSeesModel.from_h5(src)
    out = tmp_path / "abs3d_rt.h5"
    om.to_h5(out)
    with h5py.File(src, "r") as a, h5py.File(out, "r") as b:
        _assert_h5_equal(a, b)


def test_build_tcl_preserves_absorbing_payload_3d(
    tmp_path: Path, fem_3d,
) -> None:
    """The rehydrated deck carries the same absorbing fan-out: count,
    btype tally, -fx coverage, and the raw G/v/rho payload."""
    ops = _author_3d(fem_3d)
    src = tmp_path / "abs3d.h5"
    ops.h5(str(src))
    bridge_path = tmp_path / "bridge3d.tcl"
    ops.tcl(str(bridge_path))
    bridge_lines = _absorbing_lines(
        bridge_path.read_text(encoding="utf-8"), "ASDAbsorbingBoundary3D",
    )

    om = OpenSeesModel.from_h5(src)
    om_lines = _absorbing_lines(om.build("tcl"), "ASDAbsorbingBoundary3D")

    assert len(om_lines) == len(bridge_lines) == SKIN_3D
    assert _btype_tally(om_lines) == _btype_tally(bridge_lines)
    assert (sum("-fx" in ln for ln in om_lines)
            == sum("-fx" in ln for ln in bridge_lines)
            == BOTTOM_3D)
    # Raw payload (derived G, v, rho) survives verbatim on some L panel.
    sample = next(ln for ln in om_lines if ln.split()[-1] == "L")
    assert " 1351.030" in sample          # G = E/(2(1+v))
    assert " 0.262 " in sample
    # Skin emits raw floats, never a matTag: exactly one nDMaterial.
    assert om.build("tcl").count("nDMaterial ElasticIsotropic") == 1


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

def test_h5_writes_and_validates_2d(tmp_path: Path, fem_2d) -> None:
    ops = _author_2d(fem_2d)
    out = tmp_path / "abs2d.h5"
    ops.h5(str(out))
    with h5_reader.open(str(out)) as model:
        assert model.validate() == []
    om = OpenSeesModel.from_h5(out)
    recs = [r for r in om.elements()
            if r.type_token == "ASDAbsorbingBoundary2D"]
    assert len(recs) == SKIN_2D


def test_from_h5_to_h5_byte_equivalent_2d(tmp_path: Path, fem_2d) -> None:
    ops = _author_2d(fem_2d)
    src = tmp_path / "abs2d.h5"
    ops.h5(str(src))
    om = OpenSeesModel.from_h5(src)
    out = tmp_path / "abs2d_rt.h5"
    om.to_h5(out)
    with h5py.File(src, "r") as a, h5py.File(out, "r") as b:
        _assert_h5_equal(a, b)


def test_build_tcl_preserves_absorbing_payload_2d(
    tmp_path: Path, fem_2d,
) -> None:
    """2D adds the out-of-plane thickness before btype — it must survive."""
    ops = _author_2d(fem_2d)
    src = tmp_path / "abs2d.h5"
    ops.h5(str(src))
    bridge_path = tmp_path / "bridge2d.tcl"
    ops.tcl(str(bridge_path))
    bridge_lines = _absorbing_lines(
        bridge_path.read_text(encoding="utf-8"), "ASDAbsorbingBoundary2D",
    )

    om = OpenSeesModel.from_h5(src)
    om_lines = _absorbing_lines(om.build("tcl"), "ASDAbsorbingBoundary2D")

    assert len(om_lines) == len(bridge_lines) == SKIN_2D
    assert _btype_tally(om_lines) == _btype_tally(bridge_lines)
    assert (sum("-fx" in ln for ln in om_lines)
            == sum("-fx" in ln for ln in bridge_lines)
            == BOTTOM_2D)
    # thickness sits immediately before btype on a plain row.
    sample = next(ln for ln in om_lines if ln.split()[-1] == "L")
    assert float(sample.split()[-2]) == 1.0
