"""Dim-filter opacity must follow the live Display slider, not the default."""
from __future__ import annotations

from types import SimpleNamespace

from apeGmsh.viewers.overlays.pref_helpers import (
    GHOST_DIM_OPACITY,
    filter_dim_opacity,
)


def _reg(kwargs_by_dim=None):
    return SimpleNamespace(_add_mesh_kwargs=dict(kwargs_by_dim or {}))


def test_inactive_dim_ghosts():
    reg = _reg({2: {"opacity": 0.8}})
    assert filter_dim_opacity(reg, 2, active=False, fallback=0.35) == GHOST_DIM_OPACITY


def test_active_surface_uses_live_slider_not_fallback():
    reg = _reg({2: {"opacity": 0.8}, 3: {"opacity": 0.8}})
    assert filter_dim_opacity(reg, 2, active=True, fallback=0.35) == 0.8
    assert filter_dim_opacity(reg, 3, active=True, fallback=0.35) == 0.8


def test_active_surface_falls_back_when_kwargs_missing():
    reg = _reg()
    assert filter_dim_opacity(reg, 2, active=True, fallback=0.35) == 0.35


def test_active_points_and_curves_stay_opaque():
    reg = _reg({1: {"opacity": 0.2}})
    assert filter_dim_opacity(reg, 0, active=True, fallback=0.35) == 1.0
    assert filter_dim_opacity(reg, 1, active=True, fallback=0.35) == 1.0


def test_toggle_off_then_on_round_trip_keeps_slider():
    """The ghost write must not be stored as the new intended opacity."""
    reg = _reg({2: {"opacity": 0.75}})
    ghost = filter_dim_opacity(reg, 2, active=False, fallback=0.35)
    assert ghost == GHOST_DIM_OPACITY
    # kwargs still hold the slider — restoring the dim recovers 0.75
    assert filter_dim_opacity(reg, 2, active=True, fallback=0.35) == 0.75
