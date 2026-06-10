"""End-to-end tests for ``g.parts.add_plane_wave_box`` (ADR 0054, AB-1a).

Pure geometry/mesh — not OpenSees-aware.  Exercises the live-session offset-shell
build: soil box + one-element absorbing skin, per-btype volume PGs, and the
transfinite cascade, against a live OCC kernel.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.parts.plane_wave_box import (
    AbsorbingSkinResult,
    WarnAbsorbingSkinAspect,
    _btype_for,
)

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


class TestBridgeEmit:
    """End-to-end AB-1a -> AB-2: box -> mesh -> FEMData -> emitted deck."""

    def test_absorbing_boundary_deck(self):
        import collections
        import os
        import tempfile

        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.material.nd import ElasticIsotropic
        from apeGmsh.opensees.time_series.time_series import Path

        g = apeGmsh(model_name="pwb_emit", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(**BOX)
            g.mesh.generation.generate(dim=3)
            fem = g.mesh.queries.get_fem_data()
            ops = apeSees(fem)
            ops.model(ndm=3, ndf=3)
            soil = ops.register(ElasticIsotropic(E=3410.0, nu=0.262, rho=2.4e-9))
            ts = ops.register(Path(values=(0.0, 1.0, 0.0), dt=0.1))
            ops.element.stdBrick(pg=res.soil_pg, material=soil)
            ops.element.absorbing_boundary(
                skin=res, material=soil, base_series=ts, base_dirs=("x",),
            )
            path = os.path.join(tempfile.gettempdir(), "pwb_emit_test.tcl")
            ops.tcl(path)
            txt = open(path).read()
        finally:
            g.end()

        lines = [l for l in txt.splitlines() if "ASDAbsorbingBoundary3D" in l]
        assert len(lines) == SKIN_HEX
        # btype = the field before "-fx" on bottom rows, else the last field.
        tally = collections.Counter(
            l.split()[-3] if "-fx" in l else l.split()[-1] for l in lines
        )
        assert dict(tally) == EXPECTED
        # -fx attached to every B-containing cell, and only those.
        n_bottom = sum(v for b, v in EXPECTED.items() if "B" in b)
        assert sum("-fx" in l for l in lines) == n_bottom
        # Skin emits raw G/v/rho (NOT a matTag): only the soil nDMaterial is emitted.
        assert txt.count("nDMaterial ElasticIsotropic") == 1
        # G = E/(2(1+v)) derived from the soil material.
        assert " 1351.030" in next(l for l in lines if l.split()[-1] == "L")


class TestStagedFlip:
    """AB-3: the s.activate_absorbing() stage flip over a plane-wave skin."""

    def test_flip_deck(self):
        import os
        import tempfile

        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.material.nd import ElasticIsotropic

        g = apeGmsh(model_name="pwb_flip", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(**BOX)
            g.mesh.generation.generate(dim=3)
            fem = g.mesh.queries.get_fem_data()
            ops = apeSees(fem)
            ops.model(ndm=3, ndf=3)
            soil = ops.register(ElasticIsotropic(E=3410.0, nu=0.262, rho=2.4e-9))
            ops.element.stdBrick(pg=res.soil_pg, material=soil)
            ops.element.absorbing_boundary(skin=res, material=soil)
            with ops.stage(name="dyn") as s:
                s.activate_absorbing(pg=res.skin_all_pg)
                s.analysis(
                    test=ops.test.NormDispIncr(tol=1e-6, max_iter=20),
                    algorithm=ops.algorithm.Newton(),
                    integrator=ops.integrator.LoadControl(dlam=0.1),
                    constraints=ops.constraints.Plain(),
                    numberer=ops.numberer.RCM(),
                    system=ops.system.UmfPack(),
                    analysis=ops.analysis.Static(),
                )
                s.run(n_increments=1, dt=0.01)
            path = os.path.join(tempfile.gettempdir(), "pwb_flip_test.tcl")
            ops.tcl(path)
            lines = open(path).read().splitlines()
        finally:
            g.end()

        # One parameter block: N addToParameter ... stage (one per skin element).
        assert sum(1 for l in lines if l.strip().startswith("parameter ")) == 1
        flips = [l for l in lines if "addToParameter" in l and l.strip().endswith("stage")]
        assert len(flips) == SKIN_HEX
        assert sum(1 for l in lines if l.strip().startswith("updateParameter ")) == 1
        assert sum(1 for l in lines if l.strip().startswith("remove parameter")) == 1
        # The flip must precede the transient analyze.  (Since ADR 0055 P2.2
        # the analyze is emitted as a per-increment loop — ``[analyze 1 ...]``
        # inside a ``for`` — so match the bracketed call, not a bare prefix.)
        i_param = next(i for i, l in enumerate(lines) if l.strip().startswith("parameter "))
        i_analyze = next(i for i, l in enumerate(lines) if "[analyze " in l)
        assert i_param < i_analyze


class TestGuards:
    def test_rotation_rejected(self):
        # The ASDAbsorbingBoundary3D element requires X/Y face normals
        # (ASDAbsorbingBoundary3D.cpp:2135) — a rotated box is unsupported.
        g = apeGmsh(model_name="pwb_rot", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="X or Y"):
                g.parts.add_plane_wave_box(**BOX, rotation_z_deg=15.0)
        finally:
            g.end()

    def test_center_offset_meshes(self):
        # center != 0 translates the block; the OCC translate + sync renumbers
        # entities, which used to strand _metadata keys and trip the pre-mesh
        # validator.  Build + mesh at an offset must stay clean.
        g = apeGmsh(model_name="pwb_center", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(**BOX, center=(100.0, 200.0, -5.0))
            g.mesh.generation.generate(dim=3)   # would raise on stale metadata
            assert _pg_element_count(res.soil_pg) == SOIL_HEX
            assert _pg_element_count(res.skin_all_pg) == SKIN_HEX
            assert res.free_surface_pg  # top face still resolved after the shift
        finally:
            g.end()

    def test_thick_skin_warns(self):
        g = apeGmsh(model_name="pwb_aspect", verbose=False)
        g.begin()
        try:
            # soil element = 40/4 = 10; skin 60 -> 6:1 -> warns.
            with pytest.warns(WarnAbsorbingSkinAspect):
                g.parts.add_plane_wave_box(**BOX, skin_thickness=60.0)
        finally:
            g.end()

    def test_matched_skin_silent(self):
        import warnings as _w
        g = apeGmsh(model_name="pwb_noaspect", verbose=False)
        g.begin()
        try:
            with _w.catch_warnings():
                _w.simplefilter("error", WarnAbsorbingSkinAspect)
                g.parts.add_plane_wave_box(**BOX)  # matched default -> silent
        finally:
            g.end()

    def test_empty_layer_list_rejected(self):
        g = apeGmsh(model_name="pwb_emptylayer", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="layer list cannot be empty"):
                g.parts.add_plane_wave_box(x=(40.0, 4), y=(50.0, 5), z=[])
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


# ── AB-1c: layered stratigraphy via the turnkey z-list ───────────────────
class TestLayeredPlaneWaveBox:
    def test_turnkey_layered_structure(self):
        g = apeGmsh(model_name="pwb_layered", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box(
                x=(40.0, 4), y=(50.0, 5), z=[(15.0, 3), (25.0, 5)],
            )
            g.mesh.generation.generate(dim=3)

            assert res.n_layers == 2
            assert len(res.soil_pgs) == 2
            assert _pg_element_count(res.soil_pgs[0]) == 4 * 5 * 3   # top
            assert _pg_element_count(res.soil_pgs[1]) == 4 * 5 * 5   # bottom
            # base skin only in the bottom layer
            assert "B" not in res.skin_pgs_by_layer[0]
            assert "B" in res.skin_pgs_by_layer[1]
            # lateral skin splits per layer: L = NY * nz_k
            assert _pg_element_count(res.skin_pgs_by_layer[0]["L"]) == 5 * 3
            assert _pg_element_count(res.skin_pgs_by_layer[1]["L"]) == 5 * 5
            # per-btype roll-up spans both layers
            assert _pg_element_count(res.skin_pgs["L"]) == 5 * 3 + 5 * 5
            # free surface = top of the top layer (NX*NY quads)
            assert _pg_element_count(res.free_surface_pg, dim=2) == 4 * 5

            import numpy as np
            tags, coords, _ = gmsh.model.mesh.getNodes()
            xyz = np.asarray(coords).reshape(-1, 3)
            assert len(np.unique(np.round(xyz, 5), axis=0)) == len(tags)
        finally:
            g.end()
