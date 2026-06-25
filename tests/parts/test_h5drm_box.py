"""End-to-end tests for ``g.parts.add_DRM_box_from_h5drm`` (ADR 0066, D-2).

Pure geometry/mesh — not OpenSees-aware.  Synthesizes a tiny ``.h5drm`` (no
ShakerMaker dependency), builds the matched structured box in a live session,
and checks the frame contract, the soil + boundary-face PGs, and — the key
acceptance — that every dataset station has a coincident mesh node (the
fork study's "98/98 nodes" property, in miniature).
"""
from __future__ import annotations

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.parts.h5drm_box import DRMBoxFromH5Result, WarnDRMGridIrregular

CRD = 1000.0          # km -> m
H_KM = 0.05           # 50 m grid
# Distinct counts per axis so a transposed axis mapping would be caught.
NX, NY, NZ = 3, 4, 3
CENTER_KM = (6.0, 8.0, 0.0)   # drmbox_x0: lateral mid, z = surface (min)


def _write_synthetic_h5drm(path, *, nx=NX, ny=NY, nz=NZ, h=H_KM,
                           center=CENTER_KM, drmbox_x0=True):
    """A complete regular grid (km) centred at ``center`` laterally, z-down from
    the surface.  ``internal`` flags the interior (non-shell) stations."""
    import h5py

    cx, cy, cz = center
    xs = cx + (np.arange(nx) - (nx - 1) / 2.0) * h
    ys = cy + (np.arange(ny) - (ny - 1) / 2.0) * h
    zs = cz + np.arange(nz) * h                         # surface (cz) downward
    xyz = np.array([(x, y, z) for x in xs for y in ys for z in zs], dtype=float)
    internal = np.array([
        (xs[0] < x < xs[-1]) and (ys[0] < y < ys[-1]) and (zs[0] < z < zs[-1])
        for (x, y, z) in xyz
    ])
    with h5py.File(path, "w") as f:
        d = f.create_group("DRM_Data")
        d["xyz"] = xyz
        d["internal"] = internal
        m = f.create_group("DRM_Metadata")
        m["dt"] = 0.01
        if drmbox_x0:
            m["drmbox_x0"] = np.array([cx, cy, cz])
    return xyz, internal


def _pg_element_count(name: str, dim: int) -> int:
    for d, tag in gmsh.model.getPhysicalGroups(dim):
        if gmsh.model.getPhysicalName(d, tag) == name:
            total = 0
            for ent in gmsh.model.getEntitiesForPhysicalGroup(d, tag):
                _t, etags, _ = gmsh.model.mesh.getElements(d, ent)
                total += sum(len(t) for t in etags)
            return total
    return 0


class TestFrameContractAndShape:
    def test_result_shape(self, tmp_path):
        f = str(tmp_path / "m.h5drm")
        _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_shape", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box_from_h5drm(h5drm=f)
            assert isinstance(res, DRMBoxFromH5Result)
            # frame contract — defaults reproduce the station coords.
            assert res.crd_scale == CRD
            assert res.transform is None              # identity
            assert res.x0 == (0.0, 0.0, 0.0)
            assert res.center == CENTER_KM            # = drmbox_x0
            assert res.counts == (NX, NY, NZ)
            assert res.spacing == pytest.approx(H_KM * CRD)   # 50 m
            # box centred laterally, surface at z=0 (z-down).
            assert res.origin == pytest.approx(
                (-(NX - 1) / 2 * H_KM * CRD, -(NY - 1) / 2 * H_KM * CRD, 0.0))
            # PGs.
            assert res.soil_pg == "drm_soil"
            assert set(res.boundary_pgs) == {
                "xmin", "xmax", "ymin", "ymax", "top", "bottom"}
            assert res.free_surface_pg == res.boundary_pgs["top"]
            # exterior = sides + bottom, NEVER the free surface.
            assert res.free_surface_pg not in res.exterior_pgs
            assert len(res.exterior_pgs) == 5
            assert res.boundary_all_pg == "drm_boundary"
        finally:
            g.end()

    def test_name_prefix(self, tmp_path):
        f = str(tmp_path / "m.h5drm")
        _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_named", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box_from_h5drm(h5drm=f, name="site")
            assert res.soil_pg == "site_drm_soil"
            assert res.boundary_pgs["bottom"] == "site_drm_face_bottom"
        finally:
            g.end()


class TestNodeCoincidence:
    def test_every_station_has_a_coincident_node(self, tmp_path):
        # The key acceptance: H5DRM node-matching is trivial because the mesh
        # nodes land exactly on (station - drmbox_x0) * crd_scale.
        f = str(tmp_path / "m.h5drm")
        xyz, _internal = _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_nodes", verbose=False)
        g.begin()
        try:
            g.parts.add_DRM_box_from_h5drm(h5drm=f)
            g.mesh.generation.generate(dim=3)
            _tags, coords, _ = gmsh.model.mesh.getNodes()
            pts = coords.reshape(-1, 3)
            expected = (xyz - np.array(CENTER_KM)) * CRD
            for e in expected:
                dmin = float(np.min(np.linalg.norm(pts - e, axis=1)))
                assert dmin < 1e-4, f"station {e} unmatched (min dist {dmin})"
        finally:
            g.end()

    def test_pg_element_counts(self, tmp_path):
        f = str(tmp_path / "m.h5drm")
        _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_counts", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box_from_h5drm(h5drm=f)
            g.mesh.generation.generate(dim=3)
            # soil hexes = (nx-1)(ny-1)(nz-1)
            assert _pg_element_count(res.soil_pg, 3) == (NX - 1) * (NY - 1) * (NZ - 1)
            # top/bottom faces: (nx-1)(ny-1) quads; xmin/xmax: (ny-1)(nz-1)
            assert _pg_element_count(res.boundary_pgs["top"], 2) == (NX - 1) * (NY - 1)
            assert _pg_element_count(res.boundary_pgs["xmin"], 2) == (NY - 1) * (NZ - 1)
            assert _pg_element_count(res.boundary_pgs["ymin"], 2) == (NX - 1) * (NZ - 1)
            # roll-up over the whole shell.
            shell = (2 * (NX - 1) * (NY - 1) + 2 * (NY - 1) * (NZ - 1)
                     + 2 * (NX - 1) * (NZ - 1))
            assert _pg_element_count(res.boundary_all_pg, 2) == shell
        finally:
            g.end()


class TestBuffer:
    BUF = 2

    def test_buffered_shape(self, tmp_path):
        f = str(tmp_path / "m.h5drm")
        _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_buf_shape", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box_from_h5drm(h5drm=f, buffer=self.BUF)
            assert res.layers == self.BUF
            assert res.soil_pg == "drm_soil"
            assert res.buffer_pg == "drm_buffer"
            assert res.domain_pg == "drm_domain"
            # model boundary = outer (buffer) faces: sides+bottom, never the top.
            assert set(res.boundary_pgs) == {
                "xmin", "xmax", "ymin", "ymax", "top", "bottom"}
            assert res.free_surface_pg == res.boundary_pgs["top"]
            assert res.free_surface_pg not in res.exterior_pgs
            assert len(res.exterior_pgs) == 5
        finally:
            g.end()

    def test_buffered_is_conformal_and_keeps_stations(self, tmp_path):
        # The crux: the inner stations still have coincident nodes, AND the
        # inner/buffer interface is conformal (unique node count == the full
        # extended structured grid — a non-conformal weld would inflate it).
        f = str(tmp_path / "m.h5drm")
        xyz, _ = _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_buf_conformal", verbose=False)
        g.begin()
        try:
            g.parts.add_DRM_box_from_h5drm(h5drm=f, buffer=self.BUF)
            g.mesh.generation.generate(dim=3)
            _t, coords, _ = gmsh.model.mesh.getNodes()
            pts = coords.reshape(-1, 3)
            # every inner station still matched
            expected = (xyz - np.array(CENTER_KM)) * CRD
            for e in expected:
                assert float(np.min(np.linalg.norm(pts - e, axis=1))) < 1e-4
            # conformal node count: (nx+2b)(ny+2b)(nz+b)  [buffer below only]
            b = self.BUF
            exp_nodes = (NX + 2 * b) * (NY + 2 * b) * (NZ + b)
            assert pts.shape[0] == exp_nodes
        finally:
            g.end()

    def test_buffered_element_counts(self, tmp_path):
        f = str(tmp_path / "m.h5drm")
        _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_buf_counts", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box_from_h5drm(h5drm=f, buffer=self.BUF)
            g.mesh.generation.generate(dim=3)
            b = self.BUF
            inner = (NX - 1) * (NY - 1) * (NZ - 1)
            domain = (NX - 1 + 2 * b) * (NY - 1 + 2 * b) * (NZ - 1 + b)
            assert _pg_element_count(res.soil_pg, 3) == inner
            assert _pg_element_count(res.buffer_pg, 3) == domain - inner
            assert _pg_element_count(res.domain_pg, 3) == domain
            # free surface + bottom span the FULL lateral extent (side buffers
            # run full height); xmin spans the full y/z extent.
            assert _pg_element_count(res.free_surface_pg, 2) == \
                (NX - 1 + 2 * b) * (NY - 1 + 2 * b)
            assert _pg_element_count(res.boundary_pgs["bottom"], 2) == \
                (NX - 1 + 2 * b) * (NY - 1 + 2 * b)
            assert _pg_element_count(res.boundary_pgs["xmin"], 2) == \
                (NY - 1 + 2 * b) * (NZ - 1 + b)
        finally:
            g.end()


class TestAbsorbing:
    BUF = 2
    # 17 btypes for a 5-face skin ring (no top), mirroring plane_wave_box.
    EXPECTED_BTYPES = {
        "L", "R", "F", "K", "B",
        "LF", "LK", "RF", "RK",
        "BL", "BR", "BF", "BK",
        "BLF", "BLK", "BRF", "BRK",
    }

    def test_absorbing_requires_buffer(self, tmp_path):
        f = str(tmp_path / "m.h5drm")
        _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_asd_nobuf", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="buffer >= 1"):
                g.parts.add_DRM_box_from_h5drm(h5drm=f, absorbing=True)  # buffer=0
        finally:
            g.end()

    def test_absorbing_skin_shape(self, tmp_path):
        from apeGmsh.parts.plane_wave_box import AbsorbingSkinResult

        f = str(tmp_path / "m.h5drm")
        _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_asd_shape", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box_from_h5drm(
                h5drm=f, buffer=self.BUF, absorbing=True)
            assert isinstance(res.skin, AbsorbingSkinResult)
            assert res.skin.ndm == 3
            assert set(res.skin.skin_pgs) == self.EXPECTED_BTYPES
            assert res.skin.skin_pgs["L"] == "drm_absorbing_L"
            assert res.skin.skin_all_pg == "drm_absorbing"
            # bottom skin PGs = every btype containing B (9 of the 17).
            assert set(res.skin.bottom_pgs) == {
                res.skin.skin_pgs[b] for b in self.EXPECTED_BTYPES if "B" in b}
            assert res.skin.free_surface_pg == res.free_surface_pg
            assert res.domain_pg == res.soil_pg     # stdBrick target = inner+buffer
            assert res.layers == self.BUF
        finally:
            g.end()

    def test_absorbing_conformal_keeps_stations_and_counts(self, tmp_path):
        f = str(tmp_path / "m.h5drm")
        xyz, _ = _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_asd_mesh", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box_from_h5drm(
                h5drm=f, buffer=self.BUF, absorbing=True)
            g.mesh.generation.generate(dim=3)
            _t, coords, _ = gmsh.model.mesh.getNodes()
            pts = coords.reshape(-1, 3)
            # inner stations still matched
            for e in (xyz - np.array(CENTER_KM)) * CRD:
                assert float(np.min(np.linalg.norm(pts - e, axis=1))) < 1e-4
            # conformal node count: skin adds 1 ring outside the buffer on
            # sides+bottom -> (nx+2b+2)(ny+2b+2)(nz+b+1)
            b = self.BUF
            assert pts.shape[0] == (NX + 2 * b + 2) * (NY + 2 * b + 2) * (NZ + b + 1)
            # element split: soil (inner+buffer) vs skin
            ex, ey, ez = NX - 1 + 2 * b, NY - 1 + 2 * b, NZ - 1 + b   # non-skin elems
            soil = ex * ey * ez
            total = (ex + 2) * (ey + 2) * (ez + 1)                    # + skin ring
            assert _pg_element_count(res.soil_pg, 3) == soil
            assert _pg_element_count(res.skin.skin_all_pg, 3) == total - soil
        finally:
            g.end()

    def test_absorbing_skin_feeds_the_bridge(self, tmp_path):
        # Composability: the skin drops straight into the ADR 0054 facade.
        from apeGmsh.opensees import apeSees

        f = str(tmp_path / "m.h5drm")
        _write_synthetic_h5drm(f)
        g = apeGmsh(model_name="drm_asd_bridge", verbose=False)
        g.begin()
        try:
            res = g.parts.add_DRM_box_from_h5drm(
                h5drm=f, buffer=self.BUF, absorbing=True)
            g.mesh.generation.generate(dim=3)
            fem = g.mesh.queries.get_fem_data(dim=3)
        finally:
            g.end()
        ops = apeSees(fem)
        soil = ops.nDMaterial.ElasticIsotropic(E=2.77e10, nu=0.3333, rho=2600.0)
        ops.element.stdBrick(pg=res.domain_pg, material=soil)
        specs = ops.element.absorbing_boundary(skin=res.skin, material=soil)
        # one ASDAbsorbingBoundary3D declaration per btype.
        assert len(specs) == len(self.EXPECTED_BTYPES)


class TestValidation:
    def test_incomplete_grid_raises(self, tmp_path):
        import h5py
        f = str(tmp_path / "bad.h5drm")
        _write_synthetic_h5drm(f)
        with h5py.File(f, "r+") as h:          # drop one station -> not a full grid
            xyz = h["DRM_Data/xyz"][:]
            internal = h["DRM_Data/internal"][:]
            del h["DRM_Data/xyz"], h["DRM_Data/internal"]
            h["DRM_Data/xyz"] = xyz[:-1]
            h["DRM_Data/internal"] = internal[:-1]
        g = apeGmsh(model_name="drm_incomplete", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="complete regular grid"):
                g.parts.add_DRM_box_from_h5drm(h5drm=f)
        finally:
            g.end()

    def test_nonuniform_spacing_raises(self, tmp_path):
        import h5py
        f = str(tmp_path / "aniso.h5drm")
        _write_synthetic_h5drm(f)
        with h5py.File(f, "r+") as h:          # stretch z so hz != hx
            xyz = h["DRM_Data/xyz"][:]
            xyz[:, 2] *= 3.0
            del h["DRM_Data/xyz"]
            h["DRM_Data/xyz"] = xyz
        g = apeGmsh(model_name="drm_aniso", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="not uniform"):
                g.parts.add_DRM_box_from_h5drm(h5drm=f)
        finally:
            g.end()

    def test_missing_drmbox_x0_warns_and_uses_geometric_center(self, tmp_path):
        f = str(tmp_path / "nox0.h5drm")
        _write_synthetic_h5drm(f, drmbox_x0=False)
        g = apeGmsh(model_name="drm_nox0", verbose=False)
        g.begin()
        try:
            with pytest.warns(WarnDRMGridIrregular, match="drmbox_x0 missing"):
                res = g.parts.add_DRM_box_from_h5drm(h5drm=f)
            # geometric centre: lateral mid, z = surface (min) — equals CENTER_KM here.
            assert res.center == pytest.approx(CENTER_KM)
        finally:
            g.end()

    def test_not_an_h5drm_raises(self, tmp_path):
        import h5py
        f = str(tmp_path / "empty.h5")
        with h5py.File(f, "w") as h:
            h.create_group("nope")
        g = apeGmsh(model_name="drm_notdrm", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="not an .h5drm"):
                g.parts.add_DRM_box_from_h5drm(h5drm=f)
        finally:
            g.end()
