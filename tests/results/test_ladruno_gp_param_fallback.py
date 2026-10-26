"""The per-gid ``GP_PARAM`` fallback of ``.ladruno`` line stations warns (#2474).

A section-level (``section.force`` / ``section.deformation``) station takes
its natural coordinate from the class's ``QUADRATURE/GP_PARAM`` row keyed by
its ``GAUSS_ID``. When that row is missing, the reader places the station
evenly over ``[-1, +1]``; it used to do so silently. It now warns
:class:`LineStationGpParamFallbackWarning`, naming the bucket, the element
class and the element count, and stays silent when every station maps.

Warn-as-contract: run with
``-W error::apeGmsh.results.readers._ladruno_element_io.LineStationGpParamFallbackWarning``.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_element_io import (
    LineStationGpParamFallbackWarning,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
_FIBERBEAM = _FIXTURES / "fiberbeam.ladruno"
_ELEM = "MODEL_STAGE[1]/MODEL/ELEMENTS/73-ForceBeamColumn2d[1000:1]"
_BUCKET = "73-ForceBeamColumn2d[1000:1:0]"
_COMPONENTS = ("axial_force", "bending_moment_z")


def _read(path: Path, component: str) -> np.ndarray:
    r = LadrunoReader(path)
    slab = r.read_line_stations(r.stages()[0].id, component)
    return np.asarray(slab.station_natural_coord)


def _copy(tmp_path: Path) -> Path:
    dst = tmp_path / "fiberbeam.ladruno"
    shutil.copyfile(_FIBERBEAM, dst)
    return dst


@pytest.mark.parametrize("component", _COMPONENTS)
def test_committed_fiberbeam_maps_every_station_silently(component: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", LineStationGpParamFallbackWarning)
        xi = _read(_FIBERBEAM, component)
    np.testing.assert_allclose(xi, [-1.0, 0.0, 1.0])


def test_uneven_gp_param_is_read_not_replaced(tmp_path: Path) -> None:
    """A mapped GP_PARAM wins over even spacing, with no warning."""
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as f:
        f[f"{_ELEM}/QUADRATURE/GP_PARAM"][...] = [[-0.9], [0.2], [0.8]]
    with warnings.catch_warnings():
        warnings.simplefilter("error", LineStationGpParamFallbackWarning)
        xi = _read(path, "axial_force")
    np.testing.assert_allclose(xi, [-0.9, 0.2, 0.8])


def test_missing_gp_param_warns_by_name(tmp_path: Path) -> None:
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as f:
        del f[f"{_ELEM}/QUADRATURE/GP_PARAM"]
    with pytest.warns(LineStationGpParamFallbackWarning) as rec:
        xi = _read(path, "axial_force")
    np.testing.assert_allclose(xi, [-1.0, 0.0, 1.0])
    msg = str(rec[0].message)
    assert _BUCKET in msg
    assert "ForceBeamColumn2d" in msg
    assert "1 element(s)" in msg
    assert "no QUADRATURE/GP_PARAM" in msg
    assert "3 of 3 station(s)" in msg


def test_gauss_id_outside_gp_param_warns_by_name(tmp_path: Path) -> None:
    """GAUSS_ID 2 has no row in a two-row GP_PARAM: only it falls back."""
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as f:
        quad = f[f"{_ELEM}/QUADRATURE"]
        del quad["GP_PARAM"]
        quad.create_dataset("GP_PARAM", data=np.array([[-0.9], [0.2]]))
    with pytest.warns(LineStationGpParamFallbackWarning) as rec:
        xi = _read(path, "bending_moment_z")
    np.testing.assert_allclose(xi, [-0.9, 0.2, 1.0])
    msg = str(rec[0].message)
    assert _BUCKET in msg
    assert "ForceBeamColumn2d" in msg
    assert "1 element(s)" in msg
    assert "GAUSS_ID(s) [2]" in msg
    assert "1 of 3 station(s)" in msg
