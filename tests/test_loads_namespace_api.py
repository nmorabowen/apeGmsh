"""ADR 0050 P1 — dimension-indexed g.loads authoring surface.

These tests exercise the *authoring* layer only (def construction), not
resolution: string targets resolve lazily, so no mesh is needed. They
lock the new namespace verbs onto the same underlying LoadDef kinds the
old flat methods produced, and confirm the old names are gone.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------
# point namespace
# ---------------------------------------------------------------------

def test_point_force_builds_point_def(g):
    d = g.loads.point.force("Top", (1.0, 2.0, 3.0))
    assert d.kind == "point"
    assert d.force_xyz == (1.0, 2.0, 3.0)
    assert d.moment_xyz is None
    assert g.loads.load_defs[-1] is d


def test_point_moment_builds_point_def(g):
    d = g.loads.point.moment("Top", (0.0, 0.0, 5.0))
    assert d.kind == "point"
    assert d.moment_xyz == (0.0, 0.0, 5.0)
    assert d.force_xyz is None


def test_point_force_closest_builds_closest_def(g):
    d = g.loads.point.force_closest((1.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert d.kind == "point_closest"
    assert d.force_xyz == (10.0, 0.0, 0.0)
    assert d.xyz_request == (1.0, 0.0, 0.0)


def test_point_moment_closest_builds_closest_def(g):
    d = g.loads.point.moment_closest((1.0, 0.0, 0.0), (0.0, 0.0, 7.0))
    assert d.kind == "point_closest"
    assert d.moment_xyz == (0.0, 0.0, 7.0)


def test_point_closest_requires_a_vector(g):
    with pytest.raises(ValueError, match="force or moment"):
        g.loads.point.force_closest((0.0, 0.0, 0.0))


# ---------------------------------------------------------------------
# surface namespace
# ---------------------------------------------------------------------

def test_surface_pressure_is_normal_load(g):
    d = g.loads.surface.pressure("Face", 1.2e3)
    assert d.kind == "surface"
    assert d.normal is True
    assert d.magnitude == 1.2e3


def test_surface_traction_is_vector_load(g):
    d = g.loads.surface.traction("Face", (0.0, 0.0, -2.5e3))
    assert d.kind == "surface"
    assert d.normal is False
    # magnitude is the vector norm; direction carries the (non-unit) vector
    assert d.magnitude == pytest.approx(2.5e3)
    np.testing.assert_allclose(d.direction, (0.0, 0.0, -2.5e3))


def test_surface_force_resultant_is_face_load(g):
    d = g.loads.surface.force_resultant_center_mass("Face", force=(0, 0, -10))
    assert d.kind == "face_load"
    assert d.force_xyz == (0, 0, -10)


def test_surface_pressure_carries_reduction_knobs(g):
    d = g.loads.surface.pressure(
        "Face", 1.0, reduction="consistent", target_form="element")
    assert d.reduction == "consistent"
    assert d.target_form == "element"


# ---------------------------------------------------------------------
# plain-callable dimensions
# ---------------------------------------------------------------------

def test_volume_builds_body_def(g):
    d = g.loads.volume("Vol", force_per_volume=(0.0, 0.0, -1.0))
    assert d.kind == "body"
    assert d.force_per_volume == (0.0, 0.0, -1.0)


def test_line_unchanged(g):
    d = g.loads.line("Edge", magnitude=-15e3, direction=(0, 0, -1))
    assert d.kind == "line"


# ---------------------------------------------------------------------
# old flat names are gone (hard rename)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["body", "face_load", "point_closest"])
def test_old_flat_methods_removed(g, name):
    assert not callable(getattr(g.loads, name, None))


def test_point_and_surface_are_namespaces_not_methods(g):
    # g.loads.point / .surface are now verb namespaces, not directly callable
    assert not callable(g.loads.point)
    assert not callable(g.loads.surface)
    assert hasattr(g.loads.point, "force")
    assert hasattr(g.loads.surface, "pressure")
