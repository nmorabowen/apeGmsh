"""Unit + integration tests for the CS-based frame orientation rule."""
import math

import numpy as np
import pytest

from apeGmsh.solvers import Cartesian, Cylindrical, Spherical
from apeGmsh.solvers._opensees_csys import resolve_vecxz


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
# add_geom_transf API guards
# ---------------------------------------------------------------------------

class TestAddGeomTransfApi:
    def test_vecxz_and_csys_mutually_exclusive(self, g):
        g.opensees.set_model(ndm=3, ndf=6)
        with pytest.raises(ValueError, match="not both"):
            g.opensees.elements.add_geom_transf(
                "T", "Linear",
                vecxz=[0, 0, 1],
                csys=Cartesian(),
            )

    def test_roll_deg_without_csys_raises(self, g):
        g.opensees.set_model(ndm=3, ndf=6)
        with pytest.raises(ValueError, match="roll_deg"):
            g.opensees.elements.add_geom_transf(
                "T", "Linear", vecxz=[0, 0, 1], roll_deg=30,
            )

    def test_csys_stored_on_geom_transf(self, g):
        g.opensees.set_model(ndm=3, ndf=6)
        cs = Cartesian()
        g.opensees.elements.add_geom_transf("T", "Linear", csys=cs)
        stored = g.opensees._geom_transfs["T"]
        assert stored["csys"] is cs
        assert stored["vecxz"] is None
        assert stored["roll_deg"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end: build + export with csys
# ---------------------------------------------------------------------------

class TestBuildWithCsys:
    """
    Build a tiny 3-D frame with one horizontal beam and one column,
    declared under a csys-based transf.  Verify the build produces
    one geomTransf per distinct vecxz and that elements point at the
    right tags.
    """

    def _build_two_segment_model(self, g):
        g.opensees.set_model(ndm=3, ndf=6)
        # Geometry: horizontal beam (0,0,0)-(1,0,0); column (1,0,0)-(1,0,1).
        m = g.model
        p0 = m.geometry.add_point(0.0, 0.0, 0.0, lc=2.0)
        p1 = m.geometry.add_point(1.0, 0.0, 0.0, lc=2.0)
        p2 = m.geometry.add_point(1.0, 0.0, 1.0, lc=2.0)
        l_beam = m.geometry.add_line(p0, p1)
        l_col  = m.geometry.add_line(p1, p2)
        m.sync()
        g.physical.add(1, [l_beam, l_col], name="Beams")
        g.mesh.generation.generate(dim=1)

        g.opensees.elements.add_geom_transf(
            "Frame", "Linear", csys=Cartesian(),
        )
        g.opensees.elements.assign(
            "Beams", "elasticBeamColumn",
            geom_transf="Frame",
            A=1.0, E=1.0, G=1.0, Jx=1.0, Iy=1.0, Iz=1.0,
        )
        g.opensees.build()

    def test_two_segments_yield_two_distinct_vecxz(self, g):
        self._build_two_segment_model(g)
        ops = g.opensees
        emit = ops._geom_transfs_emit
        # Should have exactly 2 emitted transfs, both sourced from "Frame"
        assert len(emit) == 2
        sources = {v["source"] for v in emit.values()}
        assert sources == {"Frame"}
        vecxz_vals = sorted(tuple(v["vecxz"]) for v in emit.values())
        # Horizontal beam → (0,0,1); column → (-1,0,0)
        assert (0.0, 0.0, 1.0) in vecxz_vals
        assert (-1.0, 0.0, 0.0) in vecxz_vals

    def test_element_transf_tags_match_orientation(self, g):
        self._build_two_segment_model(g)
        ops = g.opensees
        # Tag → vecxz lookup
        tag_to_vecxz = {
            ops._transf_tags[name]: tuple(t["vecxz"])
            for name, t in ops._geom_transfs_emit.items()
        }
        # Element rows: each beam segment gets the right vecxz
        for ops_id, row in ops._elements_df.iterrows():
            n0, n1 = row.nodes
            p0 = ops._nodes_df.loc[n0]
            p1 = ops._nodes_df.loc[n1]
            edge = np.array([p1.x - p0.x, p1.y - p0.y, p1.z - p0.z])
            tangent = edge / np.linalg.norm(edge)
            v = tag_to_vecxz[row.transf_tag]
            if abs(tangent[2]) > 0.99:   # vertical
                assert v in {(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)}
            else:                          # horizontal
                assert v == (0.0, 0.0, 1.0)

    def test_export_tcl_emits_two_geom_transf_lines(self, g, tmp_path):
        self._build_two_segment_model(g)
        path = tmp_path / "m.tcl"
        g.opensees.export.tcl(path)
        text = path.read_text(encoding="utf-8")
        transf_lines = [
            ln for ln in text.splitlines()
            if ln.startswith("geomTransf")
        ]
        assert len(transf_lines) == 2

    def test_csys_in_2d_raises(self, g):
        g.opensees.set_model(ndm=2, ndf=3)
        m = g.model
        p0 = m.geometry.add_point(0, 0, 0, lc=2.0)
        p1 = m.geometry.add_point(1, 0, 0, lc=2.0)
        l = m.geometry.add_line(p0, p1)
        m.sync()
        g.physical.add(1, [l], name="Beam")
        g.mesh.generation.generate(dim=1)

        g.opensees.elements.add_geom_transf(
            "T", "Linear", csys=Cartesian(),
        )
        g.opensees.elements.assign(
            "Beam", "elasticBeamColumn",
            geom_transf="T", A=1.0, E=1.0, Iz=1.0,
        )
        with pytest.raises(ValueError, match="ndm=3"):
            g.opensees.build()
