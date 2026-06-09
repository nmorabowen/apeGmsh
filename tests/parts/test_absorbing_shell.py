"""End-to-end tests for ``g.parts.add_absorbing_shell`` (ADR 0054, AB-1b).

Bring-your-own-box entry: the user builds a single axis-aligned rectangular soil
volume, and this welds a one-element absorbing skin onto its five truncation
faces (size-based, structured-by-construction).  Returns the same
``AbsorbingSkinResult`` as ``add_plane_wave_box`` so it is drop-in for AB-2/AB-3.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh

# Box 40 x 50 x 30 meshed at element_size 10 -> nx=4, ny=5, nz=3, so the skin
# tally is identical to the AB-1a golden box (reused below).
BOX_DX, BOX_DY, BOX_DZ = 40.0, 50.0, 30.0
ESIZE = 10.0
NX, NY, NZ = 4, 5, 3

EXPECTED = {
    "L": NY * NZ, "R": NY * NZ,
    "F": NX * NZ, "K": NX * NZ,
    "B": NX * NY,
    "LF": NZ, "LK": NZ, "RF": NZ, "RK": NZ,
    "BL": NY, "BR": NY, "BF": NX, "BK": NX,
    "BLF": 1, "BLK": 1, "BRF": 1, "BRK": 1,
}
SOIL_HEX = NX * NY * NZ                 # 60
SKIN_HEX = sum(EXPECTED.values())        # 108
TOTAL_HEX = SOIL_HEX + SKIN_HEX          # 168


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


def _pg_names(dim: int = 3) -> set[str]:
    return {
        gmsh.model.getPhysicalName(d, tag)
        for d, tag in gmsh.model.getPhysicalGroups(dim)
    }


def _make_soil_box(g, *, name: str = "soil"):
    """A single axis-aligned rectangular soil box, top face at z=0, in a PG."""
    vtag = g.model.geometry.add_box(0.0, 0.0, -BOX_DZ, BOX_DX, BOX_DY, BOX_DZ)
    g.physical.add_volume([vtag], name=name)
    return vtag


class TestSkinDistribution:
    def test_btype_distribution_matches_closed_form(self):
        g = apeGmsh(model_name="shell_dist", verbose=False)
        g.begin()
        try:
            _make_soil_box(g)
            res = g.parts.add_absorbing_shell(box="soil", element_size=ESIZE)
            g.mesh.generation.generate(dim=3)

            assert set(res.skin_pgs) == set(EXPECTED)
            for btype, pg in res.skin_pgs.items():
                assert _pg_element_count(pg) == EXPECTED[btype], btype
            assert _pg_element_count(res.skin_all_pg) == SKIN_HEX
            assert _pg_element_count("soil") == SOIL_HEX
        finally:
            g.end()

    def test_soil_intact_and_all_hex(self):
        g = apeGmsh(model_name="shell_hex", verbose=False)
        g.begin()
        try:
            _make_soil_box(g)
            g.parts.add_absorbing_shell(box="soil", element_size=ESIZE)
            # one soil interior + 17 skin cells
            assert len(gmsh.model.getEntities(3)) == 18
            g.mesh.generation.generate(dim=3)
            counts = _element_counts(3)
            assert set(counts) == {"Hexahedron 8"}
            assert counts["Hexahedron 8"] == TOTAL_HEX

            # Conformal weld: no duplicate coincident nodes at the soil/skin
            # interfaces (a non-conformal weld would disconnect the skin from
            # the soil and produce a singular model).
            import numpy as np
            tags, coords, _ = gmsh.model.mesh.getNodes()
            xyz = np.asarray(coords).reshape(-1, 3)
            unique = len(np.unique(np.round(xyz, 5), axis=0))
            assert unique == len(tags), (
                f"non-conformal weld: {len(tags)} nodes but {unique} unique "
                "coordinates (duplicate interface nodes)"
            )
        finally:
            g.end()

    def test_no_illegal_combo(self):
        g = apeGmsh(model_name="shell_illegal", verbose=False)
        g.begin()
        try:
            res = (
                _make_soil_box(g),
                g.parts.add_absorbing_shell(box="soil", element_size=ESIZE),
            )[1]
            for btype in res.skin_pgs:
                assert not ("L" in btype and "R" in btype)
                assert not ("F" in btype and "K" in btype)
                assert "T" not in btype
        finally:
            g.end()


class TestFacesRestriction:
    def test_drop_one_face(self):
        g = apeGmsh(model_name="shell_faces", verbose=False)
        g.begin()
        try:
            _make_soil_box(g)
            res = g.parts.add_absorbing_shell(
                box="soil", element_size=ESIZE, faces=("L", "R", "F", "B"),
            )
            # K and every combo containing K are gone.
            dropped = {"K", "LK", "RK", "BK", "BLK", "BRK"}
            assert dropped.isdisjoint(res.skin_pgs)
            # The faces that remain are still there.
            for kept in ("L", "R", "F", "B", "LF", "RF", "BL", "BR", "BF",
                         "BLF", "BRF"):
                assert kept in res.skin_pgs
        finally:
            g.end()


class TestSoilPgHandling:
    def test_named_box_reuses_its_pg(self):
        g = apeGmsh(model_name="shell_named", verbose=False)
        g.begin()
        try:
            _make_soil_box(g, name="my_soil")
            res = g.parts.add_absorbing_shell(box="my_soil", element_size=ESIZE)
            assert res.soil_pg == "my_soil"
            # No duplicate "soil" PG was synthesised.
            assert "soil" not in _pg_names(3) or "soil" == "my_soil"
        finally:
            g.end()

    def test_handle_box_creates_soil_pg(self):
        g = apeGmsh(model_name="shell_handle", verbose=False)
        g.begin()
        try:
            vtag = g.model.geometry.add_box(
                0.0, 0.0, -BOX_DZ, BOX_DX, BOX_DY, BOX_DZ,
            )
            res = g.parts.add_absorbing_shell(box=vtag, element_size=ESIZE)
            assert res.soil_pg == "soil"
            g.mesh.generation.generate(dim=3)
            assert _pg_element_count("soil") == SOIL_HEX
        finally:
            g.end()


class TestGuards:
    def test_multi_volume_box_rejected(self):
        g = apeGmsh(model_name="shell_multi", verbose=False)
        g.begin()
        try:
            v1 = g.model.geometry.add_box(0.0, 0.0, -30.0, 40.0, 50.0, 30.0)
            v2 = g.model.geometry.add_box(100.0, 0.0, -30.0, 40.0, 50.0, 30.0)
            g.physical.add_volume([v1, v2], name="two")
            with pytest.raises(ValueError, match="exactly one"):
                g.parts.add_absorbing_shell(box="two", element_size=ESIZE)
        finally:
            g.end()

    def test_non_rectangular_box_rejected(self):
        g = apeGmsh(model_name="shell_sphere", verbose=False)
        g.begin()
        try:
            ball = g.model.geometry.add_sphere(0.0, 0.0, 0.0, 10.0)
            g.physical.add_volume([ball], name="ball")
            with pytest.raises(ValueError, match="rectangular"):
                g.parts.add_absorbing_shell(box="ball", element_size=ESIZE)
        finally:
            g.end()

    def test_bad_element_size_rejected(self):
        g = apeGmsh(model_name="shell_size", verbose=False)
        g.begin()
        try:
            _make_soil_box(g)
            with pytest.raises(ValueError, match="element_size"):
                g.parts.add_absorbing_shell(box="soil", element_size=0.0)
        finally:
            g.end()

    def test_bad_face_letter_rejected(self):
        g = apeGmsh(model_name="shell_face", verbose=False)
        g.begin()
        try:
            _make_soil_box(g)
            with pytest.raises(ValueError, match="faces"):
                g.parts.add_absorbing_shell(
                    box="soil", element_size=ESIZE, faces=("L", "T"),
                )
        finally:
            g.end()


class TestBridgePlugIn:
    """AB-1b output is drop-in for the AB-2 bridge element."""

    def test_absorbing_boundary_deck(self):
        import collections
        import os
        import tempfile

        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.material.nd import ElasticIsotropic
        from apeGmsh.opensees.time_series.time_series import Path

        g = apeGmsh(model_name="shell_emit", verbose=False)
        g.begin()
        try:
            _make_soil_box(g)
            res = g.parts.add_absorbing_shell(box="soil", element_size=ESIZE)
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
            path = os.path.join(tempfile.gettempdir(), "shell_emit_test.tcl")
            ops.tcl(path)
            txt = open(path).read()
        finally:
            g.end()

        lines = [l for l in txt.splitlines() if "ASDAbsorbingBoundary3D" in l]
        assert len(lines) == SKIN_HEX
        tally = collections.Counter(
            l.split()[-3] if "-fx" in l else l.split()[-1] for l in lines
        )
        assert dict(tally) == EXPECTED
        n_bottom = sum(v for b, v in EXPECTED.items() if "B" in b)
        assert sum("-fx" in l for l in lines) == n_bottom
        assert txt.count("nDMaterial ElasticIsotropic") == 1
