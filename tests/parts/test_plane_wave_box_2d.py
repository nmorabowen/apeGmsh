"""End-to-end tests for ``g.parts.add_plane_wave_box_2d`` (ADR 0054, AB-5).

The 2D plane-strain sibling of ``test_plane_wave_box.py``: structured soil
rectangle in the X–Y plane + one-element absorbing skin on the L/R/B
truncation faces, per-btype surface PGs, the transfinite cascade, the
``ASDAbsorbingBoundary2D`` bridge fan-out (with the out-of-plane thickness),
the staged flip, and the build-time distortion guard.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.parts.plane_wave_box import (
    AbsorbingSkinResult,
    _btype_for_2d,
)

# nx != ny so a transposed btype mapping would be caught.
BOX = dict(x=(40.0, 4), y=(30.0, 3))
NX, NY = 4, 3

# Closed-form skin element counts (product grid; top excluded).
EXPECTED = {
    "L": NY, "R": NY,        # min-X / max-X side panels
    "B": NX,                 # bottom panel
    "BL": 1, "BR": 1,        # bottom corners
}
SOIL_QUAD = NX * NY                       # 12
SKIN_QUAD = sum(EXPECTED.values())        # 12
TOTAL_QUAD = (NX + 2) * (NY + 1)          # 24 = soil + skin


def _pg_element_count(name: str, dim: int = 2) -> int:
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


class TestBtypeHelper2D:
    def test_soil_is_empty(self):
        assert _btype_for_2d("soil", "soil") == ""
        assert _btype_for_2d("soil", "soil_1") == ""

    def test_faces_and_corners_canonical_order(self):
        assert _btype_for_2d("L", "soil") == "L"
        assert _btype_for_2d("R", "soil") == "R"
        assert _btype_for_2d("soil", "B") == "B"
        assert _btype_for_2d("L", "B") == "BL"   # canonical B-first order
        assert _btype_for_2d("R", "B") == "BR"


class TestHappyPath:
    def test_result_shape_and_pgs(self):
        g = apeGmsh(model_name="pwb2d_smoke", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box_2d(**BOX)
            assert isinstance(res, AbsorbingSkinResult)
            assert res.ndm == 2
            assert res.soil_pg == "soil"
            assert res.skin_all_pg == "absorbing"
            assert set(res.axes) == {"x", "y"}
            assert res.center == (0.0, 0.0, 0.0)
            # Exactly the 5 2D btype regions.
            assert set(res.skin_pgs) == set(EXPECTED)
            assert res.skin_pgs["L"] == "absorbing_L"
            assert res.skin_pgs["BL"] == "absorbing_BL"
            assert set(res.bottom_pgs) == {
                res.skin_pgs[b] for b in EXPECTED if "B" in b
            }
            assert res.free_surface_pg == "free_surface"
            # 6 sub-surfaces: 1 soil + 5 skin regions.
            assert len([t for _d, t in gmsh.model.getEntities(2)]) == 6
        finally:
            g.end()

    def test_btype_distribution_matches_closed_form(self):
        g = apeGmsh(model_name="pwb2d_dist", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box_2d(**BOX)
            g.mesh.generation.generate(dim=2)
            counts = _element_counts(2)
            assert counts.get("Quadrilateral 4") == TOTAL_QUAD
            assert counts.get("Triangle 3", 0) == 0
            assert _pg_element_count(res.soil_pg) == SOIL_QUAD
            for btype, expected in EXPECTED.items():
                assert _pg_element_count(res.skin_pgs[btype]) == expected, btype
            assert _pg_element_count(res.skin_all_pg) == SKIN_QUAD
            # Free surface: NX top edges.
            assert _pg_element_count(res.free_surface_pg, dim=1) == NX
        finally:
            g.end()

    def test_no_illegal_combo(self):
        g = apeGmsh(model_name="pwb2d_combo", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box_2d(**BOX)
            for bt in res.skin_pgs:
                assert not ("L" in bt and "R" in bt)
                assert set(bt) <= {"B", "L", "R"}
        finally:
            g.end()

    def test_name_prefix_and_center(self):
        g = apeGmsh(model_name="pwb2d_named", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box_2d(
                **BOX, name="site", center=(100.0, -5.0),
            )
            assert res.soil_pg == "site_soil"
            assert res.skin_pgs["L"] == "site_absorbing_L"
            assert res.center == (100.0, -5.0, 0.0)
            # Geometry shifted: all COMs below the free surface y = -5.
            ymax = max(
                gmsh.model.occ.getCenterOfMass(2, t)[1]
                for _d, t in gmsh.model.getEntities(2)
            )
            assert ymax < -5.0
        finally:
            g.end()

    def test_layered_y(self):
        g = apeGmsh(model_name="pwb2d_layers", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box_2d(
                x=(40.0, 4), y=[(10.0, 1), (20.0, 2)],
            )
            assert res.n_layers == 2
            assert res.soil_pgs == ("soil_layer0", "soil_layer1")
            # Lateral skin split per layer; base skin only on the bottom layer.
            assert set(res.skin_pgs_by_layer[0]) == {"L", "R"}
            assert set(res.skin_pgs_by_layer[1]) == {"L", "R", "B", "BL", "BR"}
            g.mesh.generation.generate(dim=2)
            # Layer 0: 4x1 soil; layer 1: 4x2 soil.
            assert _pg_element_count("soil_layer0") == 4
            assert _pg_element_count("soil_layer1") == 8
            # Per-btype roll-up spans both layers: L = 1 + 2.
            assert _pg_element_count(res.skin_pgs["L"]) == 3
        finally:
            g.end()


class TestBridgeEmit:
    """End-to-end AB-5: 2D box -> mesh -> FEMData -> emitted deck."""

    def test_absorbing_boundary_deck(self):
        import collections
        import os
        import tempfile

        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.material.nd import ElasticIsotropic
        from apeGmsh.opensees.time_series.time_series import Path

        g = apeGmsh(model_name="pwb2d_emit", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box_2d(**BOX)
            g.mesh.generation.generate(dim=2)
            fem = g.mesh.queries.get_fem_data()
            ops = apeSees(fem)
            ops.model(ndm=2, ndf=2)
            soil = ops.register(ElasticIsotropic(E=3410.0, nu=0.262, rho=2.4e-9))
            ts = ops.register(Path(values=(0.0, 1.0, 0.0), dt=0.1))
            ops.element.FourNodeQuad(pg=res.soil_pg, thickness=1.0, material=soil)
            ops.element.absorbing_boundary(
                skin=res, material=soil, base_series=ts, base_dirs=("x",),
                thickness=1.0,
            )
            path = os.path.join(tempfile.gettempdir(), "pwb2d_emit_test.tcl")
            ops.tcl(path)
            txt = open(path).read()
        finally:
            g.end()

        lines = [l for l in txt.splitlines() if "ASDAbsorbingBoundary2D" in l]
        assert len(lines) == SKIN_QUAD
        # btype = the field before "-fx" on bottom rows, else the last field.
        tally = collections.Counter(
            l.split()[-3] if "-fx" in l else l.split()[-1] for l in lines
        )
        assert dict(tally) == EXPECTED
        # -fx attached to every B-containing cell, and only those.
        n_bottom = sum(v for b, v in EXPECTED.items() if "B" in b)
        assert sum("-fx" in l for l in lines) == n_bottom
        # Skin emits raw G/v/rho + thickness (NOT a matTag).
        assert txt.count("nDMaterial ElasticIsotropic") == 1
        # Field layout: "element" keyword, token, tag, 4 nodes, G v rho
        # thickness btype = 12 fields.
        plain = next(l for l in lines if l.split()[-1] == "L")
        fields = plain.split()
        assert len(fields) == 12
        assert float(fields[-2]) == 1.0          # thickness sits before btype
        assert " 1351.030" in plain              # G = E/(2(1+v))

    def test_staged_flip_deck(self):
        import os
        import tempfile

        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.material.nd import ElasticIsotropic

        g = apeGmsh(model_name="pwb2d_flip", verbose=False)
        g.begin()
        try:
            res = g.parts.add_plane_wave_box_2d(**BOX)
            g.mesh.generation.generate(dim=2)
            fem = g.mesh.queries.get_fem_data()
            ops = apeSees(fem)
            ops.model(ndm=2, ndf=2)
            soil = ops.register(ElasticIsotropic(E=3410.0, nu=0.262, rho=2.4e-9))
            ops.element.FourNodeQuad(pg=res.soil_pg, thickness=1.0, material=soil)
            ops.element.absorbing_boundary(skin=res, material=soil, thickness=1.0)
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
            path = os.path.join(tempfile.gettempdir(), "pwb2d_flip_test.tcl")
            ops.tcl(path)
            lines = open(path).read().splitlines()
        finally:
            g.end()

        flips = [l for l in lines
                 if "addToParameter" in l and l.strip().endswith("stage")]
        assert len(flips) == SKIN_QUAD
        # Since ADR 0055 P2.2 the analyze emits as a per-increment loop
        # (``[analyze 1 ...]`` inside a ``for``) — match the bracketed call.
        i_param = next(i for i, l in enumerate(lines)
                       if l.strip().startswith("parameter "))
        i_analyze = next(i for i, l in enumerate(lines)
                         if "[analyze " in l)
        assert i_param < i_analyze


class TestGuards:
    def test_rotation_rejected(self):
        g = apeGmsh(model_name="pwb2d_rot", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="distortion"):
                g.parts.add_plane_wave_box_2d(**BOX, rotation_z_deg=15.0)
        finally:
            g.end()

    def test_empty_layer_list_rejected(self):
        g = apeGmsh(model_name="pwb2d_empty", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="cannot be empty"):
                g.parts.add_plane_wave_box_2d(x=(40.0, 4), y=[])
        finally:
            g.end()

    def test_bad_thickness_rejected(self):
        g = apeGmsh(model_name="pwb2d_thick", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="skin_thickness"):
                g.parts.add_plane_wave_box_2d(**BOX, skin_thickness=-1.0)
        finally:
            g.end()

    def test_distortion_guard_rejects_skewed_quads(self):
        # The 2D element has NO source-side distortion handling — the
        # bridge build gate must reject a non-axis-aligned absorbing quad.
        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees._internal.build import BridgeError

        g = apeGmsh(model_name="pwb2d_guard", verbose=False)
        g.begin()
        try:
            g.model.geometry.add_rectangle(
                0.0, 0.0, 0.0, 10.0, 5.0, angles_deg=(0.0, 0.0, 30.0),
                label="rot",
            )
            g.physical.add(2, "rot", name="rot")
            g.mesh.structured.set_transfinite("rot", n=3)
            g.mesh.generation.generate(dim=2)
            fem = g.mesh.queries.get_fem_data()
            ops = apeSees(fem)
            ops.model(ndm=2, ndf=2)
            ops.element.ASDAbsorbingBoundary2D(
                pg="rot", btype="L", thickness=1.0, G=100.0, v=0.25, rho=1.0,
            )
            import os
            import tempfile
            with pytest.raises(BridgeError, match="not axis-aligned"):
                ops.tcl(os.path.join(tempfile.gettempdir(), "pwb2d_guard.tcl"))
        finally:
            g.end()
