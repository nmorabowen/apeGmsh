"""The turntable orbit — world-Z yaw, clamped pitch, level horizon.

``_orbit_around`` is pure math against the VTK camera API, so a fake
renderer + camera exercise the whole truth table headless:

* yaw rotates about WORLD +Z — camera height and elevation are
  invariant under pure yaw, no matter how the camera is pitched;
* elevation is clamped short of the poles — the model can never flip,
  and yaw never reverses;
* the horizon is level by construction — the camera's right vector
  stays horizontal (up has no roll component) after ANY gesture
  sequence, including long combined drags that used to drift it;
* the pivot fallback is the focal point (stale scene-centre retired).
"""
from __future__ import annotations

import math

from apeGmsh.viewers.core.navigation import (
    _ELEVATION_LIMIT,
    _orbit_around,
)


class _FakeCamera:
    def __init__(self, pos, fp, up) -> None:
        self._pos = list(pos)
        self._fp = list(fp)
        self._up = list(up)

    def GetPosition(self):
        return tuple(self._pos)

    def GetFocalPoint(self):
        return tuple(self._fp)

    def GetViewUp(self):
        return tuple(self._up)

    def SetPosition(self, *p):
        self._pos = list(p)

    def SetFocalPoint(self, *p):
        self._fp = list(p)

    def SetViewUp(self, *u):
        self._up = list(u)

    def OrthogonalizeViewUp(self):    # legacy API — must not be needed
        raise AssertionError("turntable must set view-up explicitly")


class _FakeRenderer:
    def __init__(self, cam) -> None:
        self._cam = cam
        self.clip_resets = 0

    def GetActiveCamera(self):
        return self._cam

    def ResetCameraClippingRange(self, *args):
        self.clip_resets += 1


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _view_dir(cam):
    p, f = cam.GetPosition(), cam.GetFocalPoint()
    return _norm([f[i] - p[i] for i in range(3)])


def _elevation(cam):
    return math.asin(max(-1.0, min(1.0, _view_dir(cam)[2])))


def _right(cam):
    vd, up = _view_dir(cam), cam.GetViewUp()
    return [
        vd[1] * up[2] - vd[2] * up[1],
        vd[2] * up[0] - vd[0] * up[2],
        vd[0] * up[1] - vd[1] * up[0],
    ]


def _rig(pos=(10.0, 0.0, 3.0), fp=(0.0, 0.0, 3.0)):
    """Camera looking at a column mid-height, level horizon."""
    cam = _FakeCamera(pos, fp, (0.0, 0.0, 1.0))
    return _FakeRenderer(cam), cam


PIVOT = (0.0, 0.0, 3.0)


def test_pure_yaw_preserves_height_and_elevation():
    renderer, cam = _rig()
    z0, e0 = cam.GetPosition()[2], _elevation(cam)
    for _ in range(200):
        _orbit_around(renderer, PIVOT, dx_px=17, dy_px=0)
    assert abs(cam.GetPosition()[2] - z0) < 1e-9
    assert abs(_elevation(cam) - e0) < 1e-9


def test_yaw_after_pitch_still_spins_about_world_z():
    """The old scheme yawed about the (tilted) view-up after a pitch —
    the model precessed. World-Z yaw keeps height/elevation invariant
    regardless of pitch state."""
    renderer, cam = _rig()
    _orbit_around(renderer, PIVOT, dx_px=0, dy_px=80)   # pitch first
    z0, e0 = cam.GetPosition()[2], _elevation(cam)
    assert abs(e0) > 0.1, "pitch must have moved the elevation"
    for _ in range(300):
        _orbit_around(renderer, PIVOT, dx_px=11, dy_px=0)
    assert abs(cam.GetPosition()[2] - z0) < 1e-6
    assert abs(_elevation(cam) - e0) < 1e-6


def test_elevation_clamps_short_of_the_poles():
    renderer, cam = _rig()
    # Pitch hard downward for many ticks — far beyond 90°.
    for _ in range(500):
        _orbit_around(renderer, PIVOT, dx_px=0, dy_px=-40)
    assert _elevation(cam) >= -_ELEVATION_LIMIT - 1e-9
    # And upward.
    for _ in range(1000):
        _orbit_around(renderer, PIVOT, dx_px=0, dy_px=40)
    assert _elevation(cam) <= _ELEVATION_LIMIT + 1e-9


def test_yaw_never_reverses_after_hitting_the_clamp():
    """Crossing the pole used to invert view-up, reversing yaw. With
    the clamp the same dx must keep rotating the same way."""
    renderer, cam = _rig()
    for _ in range(500):
        _orbit_around(renderer, PIVOT, dx_px=0, dy_px=-40)   # at clamp

    def _azimuth():
        p = cam.GetPosition()
        return math.atan2(p[1] - PIVOT[1], p[0] - PIVOT[0])

    deltas = []
    for _ in range(8):
        a0 = _azimuth()
        _orbit_around(renderer, PIVOT, dx_px=25, dy_px=0)
        d = _azimuth() - a0
        d = (d + math.pi) % (2 * math.pi) - math.pi
        deltas.append(d)
    assert all(d < 0 for d in deltas) or all(d > 0 for d in deltas), (
        f"yaw direction flipped at the clamp: {deltas}"
    )


def test_horizon_is_level_after_a_long_combined_drag():
    """Roll drift killer: after an arbitrary yaw+pitch scribble the
    camera's right vector must still be horizontal (z ≈ 0) and view-up
    must point skyward."""
    renderer, cam = _rig()
    moves = [(13, 7), (-9, 21), (31, -17), (-25, -5), (8, 30)] * 40
    for dx, dy in moves:
        _orbit_around(renderer, PIVOT, dx_px=dx, dy_px=dy)
    assert abs(_right(cam)[2]) < 1e-9, "horizon rolled"
    assert cam.GetViewUp()[2] > 0.0, "up vector flipped below the horizon"


def test_up_is_orthogonal_to_view_direction():
    renderer, cam = _rig()
    for dx, dy in [(40, 25), (-60, -10), (15, 45)]:
        _orbit_around(renderer, PIVOT, dx_px=dx, dy_px=dy)
    vd, up = _view_dir(cam), cam.GetViewUp()
    dot = sum(vd[i] * up[i] for i in range(3))
    assert abs(dot) < 1e-9


def test_pivot_stays_fixed_in_world():
    """Orbiting must not translate the pivot: camera and focal point
    keep their distances to it."""
    renderer, cam = _rig()
    p0 = cam.GetPosition()
    d_pos = math.dist(p0, PIVOT)
    for dx, dy in [(20, 10), (-35, 25), (50, -40)]:
        _orbit_around(renderer, PIVOT, dx_px=dx, dy_px=dy)
    assert abs(math.dist(cam.GetPosition(), PIVOT) - d_pos) < 1e-9


def test_top_view_pole_arrival_can_pitch_away():
    """A preset top view (view direction ≈ −Z) degenerates the world-up
    cross product; the fallback right axis must still let a pitch walk
    the camera off the pole instead of freezing."""
    cam = _FakeCamera((0.0, 0.0, 20.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    renderer = _FakeRenderer(cam)
    e0 = _elevation(cam)
    for _ in range(30):
        _orbit_around(renderer, (0.0, 0.0, 0.0), dx_px=0, dy_px=40)
    assert _elevation(cam) > e0 + 0.05, "camera stuck at the pole"


def test_navigation_handle_invalidates_bounds():
    from apeGmsh.viewers.core.navigation import NavigationHandle

    hits = []
    handle = NavigationHandle(invalidate_bounds=lambda: hits.append(1))
    handle.invalidate_bounds()
    assert hits == [1]
