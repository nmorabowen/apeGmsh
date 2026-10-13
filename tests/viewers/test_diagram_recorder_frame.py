"""Viewer diagrams orient from the recorder frame (.ladruno MODEL/LOCAL_AXES).

The matplotlib ``results.plot.line_force`` already prefers the recorder's
true beam frame over the geometric guess; these tests lock the same
contract for the *interactive* diagrams:

* ``LineForceDiagram`` — the per-row ``LineStationSlab.local_axes_quaternion``
  overlays the element's vecxz with the recorder z-axis, so the fill
  direction carries the true cross-section roll (and survives the
  deformed-substrate resync, which re-derives from the same vecxz).
* ``FiberSectionDiagram`` — recorded frames from
  ``results.elements.local_axes()`` place the fiber cloud in the rolled
  section plane (previously the diagram used the bare geometric default).

The fixtures' recorded frames coincide with the geometric default, so each
test rewrites ``MODEL/LOCAL_AXES/<cls>/FRAME`` in a tmp copy with a frame
rolled about the beam axis — a roll node geometry cannot recover — and
asserts the diagram renders in the rolled frame, not the default one.
GL-free: diagrams attach to the shared ``RecordingBackend`` stub.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results._slabs import LocalAxes
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    FiberSectionDiagram,
    FiberSectionStyle,
    LineForceDiagram,
    LineForceStyle,
    SlabSelector,
)
from apeGmsh.viewers.diagrams._beam_geometry import (
    compute_local_axes,
    station_position,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
BEAM3D = FIXTURES / "beam3d.ladruno"
FIBERBEAM = FIXTURES / "fiberbeam.ladruno"


# ---------------------------------------------------------------------------
# Helpers — build a rolled frame and write it into a fixture copy
# ---------------------------------------------------------------------------

def _quat_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion whose ``LocalAxes.matrices`` rows are x/y/z."""
    m = np.vstack([x, y, z]).astype(np.float64)
    w = float(np.sqrt(max(1.0 + np.trace(m), 0.0))) / 2.0
    assert w > 0.1, "test frames must keep a positive-trace rotation"
    q = np.array([
        w,
        (m[2, 1] - m[1, 2]) / (4.0 * w),
        (m[0, 2] - m[2, 0]) / (4.0 * w),
        (m[1, 0] - m[0, 1]) / (4.0 * w),
    ])
    # Sanity: the reader-side decode must reproduce the frame exactly.
    np.testing.assert_allclose(
        LocalAxes(np.array([0]), q.reshape(1, 4)).matrices[0], m, atol=1e-12,
    )
    return q


def _rolled_frame(
    ci: np.ndarray, cj: np.ndarray, theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric default frame rolled by ``theta`` about the beam axis."""
    x, y, z, _ = compute_local_axes(np.asarray(ci), np.asarray(cj))
    yr = np.cos(theta) * y + np.sin(theta) * z
    zr = np.cross(x, yr)
    return x, yr, zr


def _rolled_copy(
    src: Path, tmp_path: Path, cls_key: str, quaternion: np.ndarray,
) -> Path:
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        frame = f[f"MODEL_STAGE[1]/MODEL/LOCAL_AXES/{cls_key}/FRAME"]
        frame[...] = quaternion.reshape(1, 4)
    return dst


# ---------------------------------------------------------------------------
# LineForceDiagram — fill direction = recorder y_local, not geometric default
# ---------------------------------------------------------------------------

_BEAM3D_CI = np.array([0.0, 0.0, 0.0])
_BEAM3D_CJ = np.array([3.0, 1.0, 2.0])


def _line_force_spec(component: str = "bending_moment_z") -> DiagramSpec:
    return DiagramSpec(
        kind="line_force",
        selector=SlabSelector(component=component),
        style=LineForceStyle(scale=1.0),
    )


@pytest.fixture
def rolled_beam3d(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    theta = 0.6
    x, yr, zr = _rolled_frame(_BEAM3D_CI, _BEAM3D_CJ, theta)
    path = _rolled_copy(
        BEAM3D, tmp_path, "5-ElasticBeam3d", _quat_from_axes(x, yr, zr),
    )
    return path, yr, zr


def test_line_force_fill_uses_recorder_roll(rolled_beam3d, backend) -> None:
    path, yr, zr = rolled_beam3d
    r = Results.from_ladruno(path)
    scene = build_fem_scene(r.fem)
    diagram = LineForceDiagram(_line_force_spec(), r)
    diagram.attach(backend, r.fem, scene)

    # bending_moment_z fills along y_local — the ROLLED y, which the
    # geometric default (no model vecxz on a from_ladruno open) cannot
    # produce.
    _, y_default, _, _ = compute_local_axes(_BEAM3D_CI, _BEAM3D_CJ)
    assert not np.allclose(yr, y_default, atol=1e-3)
    np.testing.assert_allclose(
        diagram._fill_directions,
        np.tile(yr, (diagram._n_stations, 1)),
        atol=1e-6,
    )
    # The overlay parked the recorder z as the element's vecxz.
    np.testing.assert_allclose(diagram._element_vecxz[1], zr, atol=1e-9)


def test_line_force_recorder_roll_survives_substrate_sync(
    rolled_beam3d, backend,
) -> None:
    path, yr, _ = rolled_beam3d
    r = Results.from_ladruno(path)
    scene = build_fem_scene(r.fem)
    diagram = LineForceDiagram(_line_force_spec(), r)
    diagram.attach(backend, r.fem, scene)

    # The resync re-derives frames from (deformed) endpoints + the cached
    # vecxz; with the recorder z parked there the roll must not snap back
    # to the geometric default.
    diagram.sync_substrate_points(None, scene)
    np.testing.assert_allclose(
        diagram._fill_directions,
        np.tile(yr, (diagram._n_stations, 1)),
        atol=1e-6,
    )


def test_line_force_pristine_fixture_matches_recorder_frame(backend) -> None:
    # The unmodified beam3d's recorded frame coincides with the geometric
    # default — the rewire must not change the rendered picture there.
    r = Results.from_ladruno(BEAM3D)
    scene = build_fem_scene(r.fem)
    diagram = LineForceDiagram(_line_force_spec(), r)
    diagram.attach(backend, r.fem, scene)
    _, y_default, _, _ = compute_local_axes(_BEAM3D_CI, _BEAM3D_CJ)
    np.testing.assert_allclose(
        diagram._fill_directions,
        np.tile(y_default, (diagram._n_stations, 1)),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# FiberSectionDiagram — fiber cloud lands in the rolled section plane
# ---------------------------------------------------------------------------

def test_fiber_cloud_uses_recorder_roll(tmp_path: Path, backend) -> None:
    # fiberbeam runs along +X with an identity recorded frame
    # (y=(0,1,0), z=(0,0,1)). Roll 90 deg about x: y->(0,0,1), z->(0,-1,0).
    ci = np.array([0.0, 0.0, 0.0])
    cj = np.array([1.0, 0.0, 0.0])
    x, yr, zr = _rolled_frame(ci, cj, np.pi / 2.0)
    path = _rolled_copy(
        FIBERBEAM, tmp_path, "73-ForceBeamColumn2d",
        _quat_from_axes(x, yr, zr),
    )

    r = Results.from_ladruno(path)
    scene = build_fem_scene(r.fem)
    spec = DiagramSpec(
        kind="fiber_section",
        selector=SlabSelector(component="fiber_stress"),
        style=FiberSectionStyle(),
    )
    diagram = FiberSectionDiagram(spec, r)
    diagram.attach(backend, r.fem, scene)
    assert diagram._points is not None

    slab = r.elements.fibers.get(component="fiber_stress", time=[0])
    xi = np.asarray(slab.station_natural_coord, dtype=np.float64)
    expected = np.array([
        station_position(ci, cj, float(xi[k]))
        + float(slab.y[k]) * yr + float(slab.z[k]) * zr
        for k in range(slab.y.size)
    ])
    np.testing.assert_allclose(
        np.asarray(diagram._points.coords, dtype=np.float64),
        expected, atol=1e-5,
    )

    # Negative control: the default-frame placement is a different cloud.
    _, y_def, z_def, _ = compute_local_axes(ci, cj)
    default = np.array([
        station_position(ci, cj, float(xi[k]))
        + float(slab.y[k]) * y_def + float(slab.z[k]) * z_def
        for k in range(slab.y.size)
    ])
    assert not np.allclose(expected, default, atol=1e-6)


def test_fiber_recorder_z_axes_empty_for_non_ladruno() -> None:
    # Non-Ladruno readers raise TypeError from elements.local_axes();
    # the helper must swallow it into "no recorded frames" so every
    # element falls back to vecxz / geometry.
    demo = Results.demo(n_steps=2)
    out = FiberSectionDiagram._recorder_z_axes(demo, np.array([1, 2]))
    assert out == {}
