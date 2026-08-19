"""ADR 0073 — rigid analytical-plane contact (`contactPlane`).

The `g.constraints.contact_plane` generator emits one `contactSurface -slave`
(the slave node set) + the fork `contactPlane tag slaveSurfTag nx ny nz px py pz
kn [-visc μ] [-soft S]` verb. These tests lock the grammar builder
(`contact_plane_args`), the `ContactPlaneDef` validation, and the
`emit_contact_planes` build pass — fork-free (no gmsh, no openseespy).
"""
from __future__ import annotations

import pytest

from apeGmsh._kernel.defs.constraints import ContactPlaneDef
from apeGmsh._kernel.records._constraints import ContactPlaneRecord
from apeGmsh.opensees._internal.build import emit_contact_planes
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.element.contact import contact_plane_args
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# --------------------------------------------------------------------------
# contact_plane_args grammar
# --------------------------------------------------------------------------
def test_minimal_args_are_slave_normal_point_kn():
    a = contact_plane_args(7, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0e7)
    assert a == [7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e7]


def test_visc_and_soft_modifiers():
    a = contact_plane_args(7, (0, 0, 1), (1, 2, 3), 1e7, visc=2.5, soft=0.1)
    assert a == [7, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 1e7, "-visc", 2.5, "-soft", 0.1]


def test_soft_true_emits_bare_flag():
    a = contact_plane_args(7, (0, 0, 1), (0, 0, 0), 1e7, soft=True)
    assert a == [7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e7, "-soft"]


def test_soft_false_and_none_emit_nothing():
    base = [7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e7]
    assert contact_plane_args(7, (0, 0, 1), (0, 0, 0), 1e7, soft=False) == base
    assert contact_plane_args(7, (0, 0, 1), (0, 0, 0), 1e7, soft=None) == base


def test_args_reject_bad_arity():
    with pytest.raises(ValueError, match="3-vector"):
        contact_plane_args(7, (1.0,), (0, 0, 0), 1e7)
    with pytest.raises(ValueError, match="3-vector"):
        contact_plane_args(7, (0, 0, 1), (0, 0, 0, 0), 1e7)


# --------------------------------------------------------------------------
# 2D: z-pad into the permanently-valid 9-arg form (fork ADR-85, slice S1)
# --------------------------------------------------------------------------
def test_2d_normal_and_point_z_pad_to_the_9_arg_form():
    """The fork also takes a shorter 5-number 2D layout, but the zero-padded
    9-argument form stays valid on a 2D slave surface (fork T0 gated exactly
    that as a back-compat row), so apeGmsh keeps ONE emitted grammar."""
    a = contact_plane_args(7, (0.0, 1.0), (0.0, 0.5), 1.0e7)
    assert a == [7, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0e7]


def test_2d_z_pad_carries_the_modifiers():
    a = contact_plane_args(7, (0, 1), (0, 0), 1e7, visc=2.5, soft=True)
    assert a == [7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1e7, "-visc", 2.5, "-soft"]


# --------------------------------------------------------------------------
# ContactPlaneDef validation
# --------------------------------------------------------------------------
def _def(**over):
    base = dict(slave_label="floor", normal=(0, 0, 1), point=(0, 0, 0), kn=1e7)
    base.update(over)
    return ContactPlaneDef(**base)


def test_def_valid():
    d = _def(visc=2.0, soft=True, name="floor")
    assert d.kind == "contact_plane" and d.kn == 1e7 and d.soft is True


def test_def_requires_slave_label():
    with pytest.raises(ValueError, match="slave_label"):
        _def(slave_label="")


@pytest.mark.parametrize("kn", [None, 0.0, -1.0])
def test_def_kn_required_and_positive(kn):
    with pytest.raises(ValueError, match="kn"):
        _def(kn=kn)


def test_def_normal_must_be_nonzero_3vec():
    with pytest.raises(ValueError, match="non-zero"):
        _def(normal=(0, 0, 0))
    with pytest.raises(ValueError, match="3-vector"):
        _def(point=(0,))
    with pytest.raises(ValueError, match="3-vector"):
        _def(normal=(0, 0, 1, 0))


def test_def_z_pads_a_2d_normal_and_point():
    """Z-PAD, never reshape — the ContactDef.outward decision applied to the
    plane's geometry, so the record, the (3,) H5 slots and compose's rotation
    all keep one shape."""
    d = _def(normal=(0, 1), point=(0, 0.5))
    assert d.normal == (0.0, 1.0, 0.0)
    assert d.point == (0.0, 0.5, 0.0)


def test_def_still_refuses_an_all_zero_2d_normal():
    with pytest.raises(ValueError, match="non-zero"):
        _def(normal=(0, 0))


def test_def_visc_and_soft_reject_negative():
    with pytest.raises(ValueError, match="visc"):
        _def(visc=-1.0)
    with pytest.raises(ValueError, match="soft"):
        _def(soft=-0.1)


# --------------------------------------------------------------------------
# emit_contact_planes build pass
# --------------------------------------------------------------------------
class _Elems:
    def __init__(self, planes):
        self.contact_planes = planes


class _Fem:
    def __init__(self, planes):
        self.elements = _Elems(planes)


def _emit(rec):
    e = RecordingEmitter()
    emit_contact_planes(e, _Fem([rec]), TagAllocator())
    return e.calls


def test_emit_surface_then_plane():
    rec = ContactPlaneRecord(
        kind="contact_plane", name="floor", slave_nodes=[1, 2, 3, 4],
        normal=(0, 0, 1), point=(0, 0, 0), kn=1e7, visc=None, soft=True)
    calls = _emit(rec)
    kinds = [c[0] for c in calls]
    # name comment, then the slave contactSurface, then the contactPlane verb
    assert kinds == ["mp_constraint_comment", "contact_surface", "contact_plane"]
    cs = next(c for c in calls if c[0] == "contact_surface")
    assert cs[1][1] == "-slave" and list(cs[1][2:]) == [1, 2, 3, 4]
    cp = next(c for c in calls if c[0] == "contact_plane")
    # tag, slaveSurfTag, nx ny nz, px py pz, kn, -soft
    assert cp[1][1:] == (1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e7, "-soft")


def test_emit_2d_record_is_the_9_arg_form_with_zero_z_slots():
    """A 2D record reaches emit already z-padded (the def pads on the way
    in), so ``emit_contact_planes`` needs no dimension branch at all — which
    is why the 3D deck is byte-identical."""
    rec = ContactPlaneRecord(
        kind="contact_plane", name=None, slave_nodes=[1, 2, 3],
        normal=(0.0, 1.0, 0.0), point=(0.0, 0.5, 0.0), kn=1e6)
    cp = next(c for c in _emit(rec) if c[0] == "contact_plane")
    assert cp[1][1:] == (1, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1e6)
    assert cp[1][4] == 0.0 and cp[1][7] == 0.0        # nz, pz


def test_emit_no_planes_is_noop():
    assert emit_contact_planes(RecordingEmitter(), _Fem([]), TagAllocator()) is None
