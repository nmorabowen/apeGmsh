"""Unit + integration tests for the orientation-based frame rule."""
import math

import numpy as np
import pytest

from apeGmsh.opensees import AlongBeam, Cartesian, Cylindrical, Spherical
from apeGmsh.opensees._orientation import resolve_vecxz


def _close(a, b, tol=1e-9):
    return all(abs(float(x) - float(y)) < tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Cartesian.triad_at — constant triad
# ---------------------------------------------------------------------------

class TestCartesianTriad:
    def test_default_is_global_xyz(self):
        cs = Cartesian()
        e1, e2, e3 = cs.triad_at((1.23, 4.56, 7.89))
        assert _close(e3, (0, 0, 1))
        # e1 perpendicular to e3, e2 = e3 × e1
        assert abs(float(np.dot(e1, e3))) < 1e-12
        assert _close(np.cross(e3, e1), e2)

    def test_y_up_gives_e3_y(self):
        cs = Cartesian(reference_axis=(0, 1, 0))
        _, _, e3 = cs.triad_at((0, 0, 0))
        assert _close(e3, (0, 1, 0))

    def test_normalises_non_unit_reference(self):
        cs = Cartesian(reference_axis=(0, 0, 7))
        _, _, e3 = cs.triad_at((0, 0, 0))
        assert _close(e3, (0, 0, 1))

    def test_oblique_reference_picks_perpendicular_e1(self):
        cs = Cartesian(reference_axis=(1, 1, 1))
        e1, e2, e3 = cs.triad_at((0, 0, 0))
        # Orthonormal triad
        assert abs(float(np.dot(e1, e3))) < 1e-12
        assert abs(float(np.dot(e2, e3))) < 1e-12
        assert abs(float(np.dot(e1, e2))) < 1e-12
        assert _close(np.cross(e3, e1), e2)


# ---------------------------------------------------------------------------
# Cylindrical.triad_at
# ---------------------------------------------------------------------------

class TestCylindricalTriad:
    def test_axial_is_e3(self):
        cs = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        _, _, e3 = cs.triad_at((1, 0, 5))
        assert _close(e3, (0, 0, 1))

    def test_radial_outward_at_xpos(self):
        cs = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        e_r, _, _ = cs.triad_at((3, 0, 0))
        assert _close(e_r, (1, 0, 0))

    def test_radial_outward_at_45deg(self):
        cs = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        s = math.sqrt(0.5)
        e_r, _, _ = cs.triad_at((s, s, 1.7))
        # Radial component lives in the X-Y plane (axial removed)
        assert _close(e_r, (s, s, 0))

    def test_circumferential_perpendicular_to_radial_and_axial(self):
        cs = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        e_r, e_t, e_z = cs.triad_at((2.0, 0.5, 0.3))
        assert abs(float(np.dot(e_r, e_t))) < 1e-12
        assert abs(float(np.dot(e_t, e_z))) < 1e-12
        assert abs(float(np.dot(e_r, e_z))) < 1e-12

    def test_on_axis_does_not_crash(self):
        cs = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        e_r, e_t, e_z = cs.triad_at((0, 0, 5))   # on the axis
        # Should still return an orthonormal triad
        assert abs(float(np.linalg.norm(e_r) - 1.0)) < 1e-12
        assert abs(float(np.dot(e_r, e_z))) < 1e-12

    def test_origin_translation(self):
        cs = Cylindrical(origin=(10, 20, 30), axis=(0, 0, 1))
        e_r, _, _ = cs.triad_at((13, 20, 30))
        assert _close(e_r, (1, 0, 0))


# ---------------------------------------------------------------------------
# Spherical.triad_at
# ---------------------------------------------------------------------------

class TestSphericalTriad:
    def test_radial_outward_along_x(self):
        cs = Spherical(origin=(0, 0, 0))
        _, _, e_r = cs.triad_at((5, 0, 0))
        assert _close(e_r, (1, 0, 0))

    def test_radial_outward_arbitrary_point(self):
        cs = Spherical(origin=(0, 0, 0))
        p = np.array([1.0, 2.0, 3.0])
        _, _, e_r = cs.triad_at(p)
        assert _close(e_r, p / np.linalg.norm(p))

    def test_orthonormal(self):
        cs = Spherical(origin=(0, 0, 0))
        e1, e2, e3 = cs.triad_at((1.0, 2.0, 3.0))
        for v in (e1, e2, e3):
            assert abs(float(np.linalg.norm(v) - 1.0)) < 1e-12
        assert abs(float(np.dot(e1, e2))) < 1e-12
        assert abs(float(np.dot(e1, e3))) < 1e-12
        assert abs(float(np.dot(e2, e3))) < 1e-12

    def test_at_origin_does_not_crash(self):
        cs = Spherical(origin=(0, 0, 0))
        e1, e2, e3 = cs.triad_at((0, 0, 0))
        assert abs(float(np.linalg.norm(e3) - 1.0)) < 1e-12


# ---------------------------------------------------------------------------
# resolve_vecxz — the orientation rule
# ---------------------------------------------------------------------------

class TestResolveRule:
    def test_horizontal_beam_default_z_gives_plus_z_vecxz(self):
        cs = Cartesian()
        e1, e2, e3 = cs.triad_at((0, 0, 0))
        v = resolve_vecxz((1, 0, 0), e1, e2, e3)
        assert _close(v, (0, 0, 1))

    def test_vertical_beam_falls_back_to_minus_x(self):
        # Tangent +Z, default Cartesian → degenerate branch
        cs = Cartesian()
        e1, e2, e3 = cs.triad_at((0, 0, 0))
        v = resolve_vecxz((0, 0, 1), e1, e2, e3)
        # ly = e2 = +Y, lz = (+Z) × (+Y) = -X
        assert _close(v, (-1, 0, 0))

    def test_vertical_beam_minus_z_gives_plus_x(self):
        # Tangent -Z, default Cartesian (continuous arch traversal case)
        cs = Cartesian()
        e1, e2, e3 = cs.triad_at((0, 0, 0))
        v = resolve_vecxz((0, 0, -1), e1, e2, e3)
        # lz = (-Z) × (+Y) = +X
        assert _close(v, (1, 0, 0))

    def test_quarter_arch_in_xz_plane(self):
        # Tangent at quarter-arc point: 45° between +X and +Z
        cs = Cartesian()
        e1, e2, e3 = cs.triad_at((0, 0, 0))
        s = math.sqrt(0.5)
        v = resolve_vecxz((s, 0, s), e1, e2, e3)
        # ly = Z × t = (0,1,0)·s -> (0,1,0); not unit yet — actually
        # Z × (s,0,s) = (-s, 0, 0)... let me compute carefully.
        # Z × t where Z=(0,0,1), t=(s,0,s):
        #   = (0·s - 1·0, 1·s - 0·s, 0·0 - 0·s) = (0, s, 0)
        # ly = (0, 1, 0)
        # lz = t × ly = (s,0,s) × (0,1,0) = (0·0 - s·1, s·0 - s·0, s·1 - 0·0)
        #             = (-s, 0, s)
        assert _close(v, (-s, 0, s))

    def test_y_up_horizontal_x_beam_gives_plus_y_vecxz(self):
        cs = Cartesian(reference_axis=(0, 1, 0))
        e1, e2, e3 = cs.triad_at((0, 0, 0))
        v = resolve_vecxz((1, 0, 0), e1, e2, e3)
        # ly = Y × X = -Z, lz = X × (-Z) = +Y
        assert _close(v, (0, 1, 0))

    def test_roll_90_about_horizontal_beam(self):
        cs = Cartesian()
        e1, e2, e3 = cs.triad_at((0, 0, 0))
        v = resolve_vecxz((1, 0, 0), e1, e2, e3, roll_deg=90)
        # vecxz starts as +Z; roll +90 about +X takes +Z to -Y
        assert _close(v, (0, -1, 0))


# ---------------------------------------------------------------------------
# Cylindrical applied at the rule layer
# ---------------------------------------------------------------------------

class TestCylindricalRule:
    def test_ring_beam_segment_radial_strong_axis(self):
        # Vertical-axis cylinder, beam tangent circumferential
        cs = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        # At point (1, 0, 0): tangent direction along +Y (circumferential)
        e1, e2, e3 = cs.triad_at((1, 0, 0))   # e_r=+X, e_t=+Y, e_z=+Z
        v = resolve_vecxz((0, 1, 0), e1, e2, e3)
        # ly = Z × Y = -X
        # lz = Y × (-X) = (1·0 - 0·(-1), 0·0 - 0·0, 0·(-1) - 1·0) = (0, 0, 1)... let me redo
        # Y × (-X) = (Yy·(-X)z - Yz·(-X)y, Yz·(-X)x - Yx·(-X)z, Yx·(-X)y - Yy·(-X)x)
        # Y = (0,1,0); -X = (-1,0,0)
        # cross = (1·0 - 0·0, 0·(-1) - 0·0, 0·0 - 1·(-1)) = (0, 0, 1)
        assert _close(v, (0, 0, 1))


# ---------------------------------------------------------------------------
# AlongBeam — tangent-derived orientation
# ---------------------------------------------------------------------------

def _straight_x_axis_fem(n_segments: int = 5):
    """A FEM stub whose 'MainBar' PG is a chain of segments along +X.

    Nodes 1..n_segments+1 at (0,0,0), (1,0,0), (2,0,0), ...
    """
    from tests.opensees.fixtures.fem_stub import (
        FEMStub, _ElementGroupView, _ElementsStub, _NodesStub,
    )
    ids = list(range(1, n_segments + 2))
    coords = [(float(i), 0.0, 0.0) for i in range(n_segments + 1)]
    nodes = _NodesStub(ids=ids, coords=coords, node_pgs={})
    elem_ids = tuple(range(1, n_segments + 1))
    connectivity = tuple((i, i + 1) for i in range(1, n_segments + 1))
    elements = _ElementsStub(
        elem_pgs={
            "MainBar": _ElementGroupView(ids=elem_ids, connectivity=connectivity),
        }
    )
    return FEMStub(nodes=nodes, elements=elements)


def _quarter_arc_fem(n_segments: int = 4):
    """A FEM stub whose 'MainBar' PG is a quarter-arc in the X-Y plane.

    Nodes evenly placed on the unit circle from (1,0,0) to (0,1,0).
    """
    from tests.opensees.fixtures.fem_stub import (
        FEMStub, _ElementGroupView, _ElementsStub, _NodesStub,
    )
    angles = [math.pi * 0.5 * i / n_segments for i in range(n_segments + 1)]
    ids = list(range(1, n_segments + 2))
    coords = [(math.cos(a), math.sin(a), 0.0) for a in angles]
    nodes = _NodesStub(ids=ids, coords=coords, node_pgs={})
    elem_ids = tuple(range(1, n_segments + 1))
    connectivity = tuple((i, i + 1) for i in range(1, n_segments + 1))
    elements = _ElementsStub(
        elem_pgs={
            "MainBar": _ElementGroupView(ids=elem_ids, connectivity=connectivity),
        }
    )
    return FEMStub(nodes=nodes, elements=elements)


class TestAlongBeamConstruction:
    def test_repr(self):
        ab = AlongBeam(reference_pg="MainBar")
        assert "MainBar" in repr(ab)
        assert "AlongBeam" in repr(ab)

    def test_reference_pg_attribute(self):
        ab = AlongBeam(reference_pg="MainBar")
        assert ab.reference_pg == "MainBar"

    def test_triad_at_before_bind_raises(self):
        ab = AlongBeam(reference_pg="MainBar")
        with pytest.raises(RuntimeError, match="bind_fem"):
            ab.triad_at((0.0, 0.0, 0.0))


class TestAlongBeamBindFem:
    def test_bind_fem_with_straight_reference(self):
        fem = _straight_x_axis_fem(n_segments=3)
        ab = AlongBeam(reference_pg="MainBar")
        ab.bind_fem(fem)
        # After bind, triad_at should work.
        _, _, e3 = ab.triad_at((1.5, 0.0, 0.0))
        assert _close(e3, (1, 0, 0))

    def test_bind_fem_missing_pg_raises(self):
        fem = _straight_x_axis_fem()
        ab = AlongBeam(reference_pg="DoesNotExist")
        with pytest.raises(ValueError, match="reference PG not found"):
            ab.bind_fem(fem)

    def test_bind_fem_is_idempotent(self):
        fem = _straight_x_axis_fem()
        ab = AlongBeam(reference_pg="MainBar")
        ab.bind_fem(fem)
        triad_1 = ab.triad_at((0.5, 0.0, 0.0))
        ab.bind_fem(fem)   # rebind — should give the same answer
        triad_2 = ab.triad_at((0.5, 0.0, 0.0))
        for a, b in zip(triad_1, triad_2):
            assert _close(a, b)


class TestAlongBeamProjection:
    def test_straight_bar_tangent_is_uniform(self):
        """Any probe near a straight +X bar gets e3 = +X."""
        fem = _straight_x_axis_fem(n_segments=5)
        ab = AlongBeam(reference_pg="MainBar")
        ab.bind_fem(fem)
        for probe in [(0.5, 0.0, 0.0), (1.5, 0.0, 0.0), (4.5, 0.0, 0.0)]:
            _, _, e3 = ab.triad_at(probe)
            assert _close(e3, (1, 0, 0)), f"probe={probe} → e3={e3}"

    def test_offset_probe_uses_projection_to_segment(self):
        """A probe offset from the bar in +Y still gets the bar's tangent.

        The projection algorithm finds the nearest point on the
        segment, not the nearest segment midpoint, so axial offset
        from a midpoint doesn't confuse it.
        """
        fem = _straight_x_axis_fem(n_segments=5)
        ab = AlongBeam(reference_pg="MainBar")
        ab.bind_fem(fem)
        # Stirrup centered at x=2.5, offset 0.3 in +Y.
        _, _, e3 = ab.triad_at((2.5, 0.3, 0.0))
        assert _close(e3, (1, 0, 0))

    def test_quarter_arc_tangent_varies_with_position(self):
        """A curved reference bar produces different tangents at different
        probe positions — the whole point of AlongBeam."""
        fem = _quarter_arc_fem(n_segments=8)
        ab = AlongBeam(reference_pg="MainBar")
        ab.bind_fem(fem)

        # Near (1, 0, 0) the tangent is along ~+Y (start of arc).
        _, _, e3_start = ab.triad_at((1.0, 0.05, 0.0))
        # Near (0, 1, 0) the tangent is along ~-X (end of arc).
        _, _, e3_end = ab.triad_at((0.05, 1.0, 0.0))

        # Sanity: tangents differ substantially between the two ends.
        assert float(np.dot(e3_start, e3_end)) < 0.5

    def test_probe_off_endpoint_clipped_to_segment(self):
        """A probe beyond the bar's end clips to the last segment
        (no extrapolation)."""
        fem = _straight_x_axis_fem(n_segments=3)  # bar from x=0 to x=3
        ab = AlongBeam(reference_pg="MainBar")
        ab.bind_fem(fem)
        # Probe well beyond the bar's end.
        _, _, e3 = ab.triad_at((100.0, 0.0, 0.0))
        # Tangent of the last segment is still +X — no NaN, no extrapolation.
        assert _close(e3, (1, 0, 0))

    def test_triad_is_orthonormal(self):
        """The derived (e1, e2, e3) is always an orthonormal triad."""
        fem = _quarter_arc_fem(n_segments=6)
        ab = AlongBeam(reference_pg="MainBar")
        ab.bind_fem(fem)
        e1, e2, e3 = ab.triad_at((0.7, 0.7, 0.0))
        # All unit-length
        assert abs(float(np.linalg.norm(e1)) - 1.0) < 1e-9
        assert abs(float(np.linalg.norm(e2)) - 1.0) < 1e-9
        assert abs(float(np.linalg.norm(e3)) - 1.0) < 1e-9
        # Mutually orthogonal
        assert abs(float(np.dot(e1, e2))) < 1e-9
        assert abs(float(np.dot(e2, e3))) < 1e-9
        assert abs(float(np.dot(e3, e1))) < 1e-9


# The legacy ``TestAddGeomTransfApi`` and ``TestBuildWithCsys`` classes
# that used to live here exercised the deprecated
# ``g.opensees.elements.add_geom_transf(csys=…)`` flow on the legacy
# ``apeGmsh.solvers.OpenSees`` bridge.  Equivalent coverage on the
# new typed ``Linear`` / ``PDelta`` / ``Corotational`` primitives
# (which accept ``orientation=`` directly at construction) lives in
# :mod:`tests.opensees.unit.primitives.test_geom_transf` and
# :mod:`tests.opensees.contract.test_geom_transf_contract`.
