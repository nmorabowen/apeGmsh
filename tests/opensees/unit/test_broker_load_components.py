"""Unit tests for ``broker_load_components`` — the bridge chokepoint that
maps a DOF-agnostic ``NodalLoadRecord`` (pure 3-D force/moment) onto a
model's ``(ndm, ndf)`` DOF layout (ADR 0051).

Covers the two correctness guarantees added alongside 2-D gravity:

* ``ndf == 3`` is disambiguated by ``ndm`` — a 2-D planar frame is
  ``(Fx, Fy, Mz)`` but a 3-D solid is ``(Fx, Fy, Fz)``.
* a non-zero component the model cannot carry fails loud rather than
  being silently dropped.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._internal.build import (
    BridgeError,
    broker_load_components,
)


class _Rec:
    def __init__(self, force=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 0.0),
                 node_id=7):
        self.force_xyz = force
        self.moment_xyz = moment
        self.node_id = node_id


# ---------------------------------------------------------------------
# ndm-aware DOF layout
# ---------------------------------------------------------------------

def test_2d_plane_ndf2():
    # ndm=2, ndf=2 (plane stress/strain): in-plane forces only.
    assert broker_load_components(_Rec((3.0, 4.0, 0.0)), 2, 2) == (3.0, 4.0)


def test_2d_frame_ndf3_keeps_moment():
    # ndm=2, ndf=3 planar frame: (ux, uy, rz) -> (Fx, Fy, Mz).
    out = broker_load_components(_Rec((1.0, 2.0, 0.0), (0.0, 0.0, 9.0)), 3, 2)
    assert out == (1.0, 2.0, 9.0)


def test_3d_solid_ndf3_keeps_fz():
    # ndm=3, ndf=3 solid: (ux, uy, uz) -> (Fx, Fy, Fz). This is the bug
    # that previously dropped 3-D self-weight (mapped ndf==3 -> Mz).
    out = broker_load_components(_Rec((0.0, 0.0, -10.0)), 3, 3)
    assert out == (0.0, 0.0, -10.0)


def test_3d_full_ndf6():
    out = broker_load_components(
        _Rec((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), 6, 3)
    assert out == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


# ---------------------------------------------------------------------
# Fail-loud on a dropped non-zero component
# ---------------------------------------------------------------------

def test_out_of_plane_force_on_2d_raises():
    with pytest.raises(BridgeError, match=r"Fz=-10.*ndm=2, ndf=2"):
        broker_load_components(_Rec((0.0, 0.0, -10.0)), 2, 2)


def test_moment_on_3d_solid_raises():
    with pytest.raises(BridgeError, match=r"Mz=5.*ndm=3, ndf=3"):
        broker_load_components(_Rec((0.0, 0.0, -1.0), (0.0, 0.0, 5.0)), 3, 3)


def test_zero_dropped_component_is_fine():
    # A z-force of exactly 0 on a 2-D model must NOT raise.
    assert broker_load_components(_Rec((5.0, 6.0, 0.0)), 2, 2) == (5.0, 6.0)


def test_error_names_the_node():
    with pytest.raises(BridgeError, match=r"node 42"):
        broker_load_components(_Rec((0.0, 0.0, -1.0), node_id=42), 2, 2)
