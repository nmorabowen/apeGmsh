"""End-to-end tests for ``g.parts.add_plane_wave_box`` (ADR 0054, AB-1a).

Pure geometry/mesh — not OpenSees-aware.  Exercises the live-session offset-shell
build: soil box + one-element absorbing skin, per-btype volume PGs, and the
transfinite cascade, against a live OCC kernel.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.parts.plane_wave_box import AbsorbingSkinResult, _btype_for

# Small mesh-friendly box.  nx=4, ny=5, nz=3 -> distinct counts per axis so a
# transposed btype mapping would be caught.
BOX = dict(x=(40.0, 4), y=(50.0, 5), z=(30.0, 3))
NX, NY, NZ = 4, 5, 3

# Closed-form skin element counts (product grid; top excluded).
EXPECTED = {
    "L": NY * NZ, "R": NY * NZ,            # min-X / max-X face panels
    "F": NX * NZ, "K": NX * NZ,            # min-Y / max-Y face panels
    "B": NX * NY,                          # bottom panel
    "LF": NZ, "LK": NZ, "RF": NZ, "RK": NZ,   # vertical edges
    "BL": NY, "BR": NY, "BF": NX, "BK": NX,   # bottom edges
    "BLF": 1, "BLK": 1, "BRF": 1, "BRK": 1,   # bottom corners
}
SOIL_HEX = NX * NY * NZ                    # 60
SKIN_HEX = sum(EXPECTED.values())          # 108
TOTAL_HEX = (NX + 2) * (NY + 2) * (NZ + 1)  # 168 = soil + skin


def _pg_element_count(name: str, dim: int = 3) -> int:
    for d, tag in gmsh.model.getPhysicalGroups(dim):
        if gmsh.model.getPhysicalName(d, tag) == name:
            total = 0
            for ent in gmsh.model.getEntitiesForPhysicalGroup(d, tag):
                _types, etags, _ = gmsh.model.mesh.getElements(d, ent)
                total += sum(len(t) for t in etags)
            return total
    return 0


def _element_counts(dim: int) -> dict[str, int]:
    etypes, etags, _ = gmsh.model.mesh.getElements(dim=dim)
    out: dict[str, int] = {}
    for et, tags in zip(etypes, etags):
        name, *_ = gmsh.model.mesh.getElementProperties(et)
        out[name] = out.get(name, 0) + len(tags)
    return out


class TestBtypeHelper:
    def test_soil_is_empty(self):
        assert _btype_for("soil", "soil", "soil") == ""

    def test_faces_edges_corners_canonical_order(self):
        assert _btype_for("L", "soil", "soil") == "L"
        assert _btype_for("L", "F", "soil") == "LF"
        assert _btype_for("soil", "soil", "B") == "B"
        assert _btype_for("L", "soil", "B") == "BL"   # canonical BLRFK order
        assert _btype_for("R", "K", "B") == "BRK"
        assert _btype_for("L", "F", "B") == "BLF"


class TestHappyPath:
    def test_result_shape_and_pgs(self):
        g = apeGmsh(model_name="pwb_smoke", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(**BOX)
            assert isinstance(res, AbsorbingSkinResult)
            assert res.soil_pg == "soil"
            assert res.skin_all_pg == "absorbing"
            assert set(res.axes) == {"x", "y", "z"}
            assert res.center == (0.0, 0.0, 0.0)
            assert res.rotation_z == 0.0
            # Exactly the 17 btype regions, no more, no fewer.
            assert set(res.skin_pgs) == set(EXPECTED)
            assert res.skin_pgs["L"] == "absorbing_L"
            assert res.skin_pgs["BLF"] == "absorbing_BLF"
            # bottom PGs = every btype containing B.
            assert set(res.bottom_pgs) == {
                res.skin_pgs[b] for b in EXPECTED if "B" in b
            }
            assert res.free_surface_pg == "free_surface"
            # 18 sub-volumes: 1 soil + 17 skin regions.
            assert len([t for _d, t in gmsh.model.getEntities(3)]) == 18
        finally:
            g.end()

    def test_btype_distribution_matches_closed_form(self):
        g = apeGmsh(model_name="pwb_dist", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(**BOX)
            g.mesh.generation.generate(dim=3)
            counts = _element_counts(3)
            assert counts.get("Hexahedron 8") == TOTAL_HEX
            assert counts.get("Tetrahedron 4", 0) == 0
            # Soil interior.
            assert _pg_element_count(res.soil_pg) == SOIL_HEX
            # Every skin btype PG matches the closed-form count.
            for btype, expected in EXPECTED.items():
                assert _pg_element_count(res.skin_pgs[btype]) == expected, btype
            # Roll-up = sum of all skin regions.
            assert _pg_element_count(res.skin_all_pg) == SKIN_HEX
            # Free surface = nx*ny quads on the soil top.
            assert _pg_element_count(res.free_surface_pg, dim=2) == NX * NY
        finally:
            g.end()

    def test_no_illegal_combo(self):
        g = apeGmsh(model_name="pwb_illegal", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(**BOX)
            for btype in res.skin_pgs:
                assert not ("L" in btype and "R" in btype), btype
                assert not ("F" in btype and "K" in btype), btype
                assert all(c in "BLRFK" for c in btype), btype
                assert "T" not in btype and "U" not in btype  # no top letter
        finally:
            g.end()

    def test_scalar_thickness(self):
        g = apeGmsh(model_name="pwb_scalar", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(**BOX, skin_thickness=7.5)
            g.mesh.generation.generate(dim=3)
            # Distribution is thickness-independent.
            assert _pg_element_count(res.skin_pgs["B"]) == NX * NY
            assert _pg_element_count(res.soil_pg) == SOIL_HEX
        finally:
            g.end()

    def test_name_prefix_and_center(self):
        g = apeGmsh(model_name="pwb_named", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(
                **BOX, name="site", center=(100.0, 200.0, -5.0),
            )
            assert res.soil_pg == "site_soil"
            assert res.skin_pgs["L"] == "site_absorbing_L"
            assert res.center == (100.0, 200.0, -5.0)
            # Geometry shifted: max local z=0 -> world z = cz = -5.0 (free surface).
            zmax = max(
                gmsh.model.occ.getCenterOfMass(3, t)[2]
                for _d, t in gmsh.model.getEntities(3)
            )
            assert zmax < -5.0  # all COMs are below the free surface
        finally:
            g.end()


class TestGuards:
    def test_rotation_rejected(self):
        g = apeGmsh(model_name="pwb_rot", verbose=False)
        g.begin()
        try:
            with pytest.raises(NotImplementedError, match="rotation"):
                g.parts.add_plane_wave_box(**BOX, rotation_z_deg=15.0)
        finally:
            g.end()

    def test_layered_z_rejected(self):
        g = apeGmsh(model_name="pwb_layer", verbose=False)
        g.begin()
        try:
            with pytest.raises(NotImplementedError, match="layered Z"):
                g.parts.add_plane_wave_box(
                    x=(40.0, 4), y=(50.0, 5), z=[(15.0, 2), (15.0, 1)],
                )
        finally:
            g.end()

    def test_bad_thickness_rejected(self):
        g = apeGmsh(model_name="pwb_thick", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="skin_thickness"):
                g.parts.add_plane_wave_box(**BOX, skin_thickness=-1.0)
        finally:
            g.end()

    def test_bad_axis_rejected(self):
        g = apeGmsh(model_name="pwb_axis", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="n_elements"):
                g.parts.add_plane_wave_box(x=(40.0, 0), y=(50.0, 5), z=(30.0, 3))
        finally:
            g.end()
