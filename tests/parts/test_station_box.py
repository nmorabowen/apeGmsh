"""Tests for ``g.parts.add_station_soil_box`` (ADR 0118).

A synthetic two-shell ``.h5drm`` (the ShakerMaker DRMBox layout: an outer
shell on the sides and bottom of a lattice, the inner shell one line inward),
read through the San Ramón frame (km → mm, x/y swapped, z flipped up, offset
origin). The DRM box must put a node on every station and no other node
within the fork's matching tolerance; the absorbing (Tier-3) configuration
must reproduce the same interior node for node.

The fork rule is replicated here on purpose instead of calling the module's
helper: ``H5DRMLoadPattern::node_matching_BruteForce`` visits every domain
node, takes its nearest station, and matches it when ``d < tolerance``
(strict).

The real-file check runs when ``APEGMSH_H5DRM_REAL`` names a two-shell
``.h5drm`` (5 GB class, not in CI).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.parts import Exterior, NearField, Pit, SoilLattice

H_KM = 0.0025                       # 2.5 m grid
CENTER = (12.5, 7.25, 0.0)          # drmbox_x0 (km)
T_SR = ((0, 1, 0), (1, 0, 0), (0, 0, -1))   # San Ramón: x<->y, z up
X0_SR = (22000.0, 15500.0, 0.0)              # mm
Z_UP = ((1, 0, 0), (0, 1, 0), (0, 0, -1))   # metres-style z flip only
NX, NY, NZ = 16, 14, 8              # distinct per axis: a swapped axis shows


def write_two_shell(path, *, xs=None, ys=None, zs=None, center=CENTER,
                    complete=False, drop=0, flip_internal=0):
    """Two-shell station set (km, z down from 0) of the lattice xs × ys × zs."""
    import h5py

    cx, cy, cz = center
    if xs is None:
        xs = cx + (np.arange(NX) - (NX - 1) / 2) * H_KM
    if ys is None:
        ys = cy + (np.arange(NY) - (NY - 1) / 2) * H_KM
    if zs is None:
        zs = cz + np.arange(NZ) * H_KM
    nx, ny, nz = len(xs), len(ys), len(zs)
    pts, internal = [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                outer = i in (0, nx - 1) or j in (0, ny - 1) or k == nz - 1
                inner = (not outer) and (
                    i in (1, nx - 2) or j in (1, ny - 2) or k == nz - 2)
                if outer or inner or complete:
                    pts.append((x, y, z))
                    internal.append(not outer)
    pts = np.array(pts)[drop:]
    internal = np.array(internal)[drop:]
    if flip_internal:
        internal[:flip_internal] = ~internal[:flip_internal]
    with h5py.File(path, "w") as f:
        f["DRM_Data/xyz"] = pts
        f["DRM_Data/internal"] = internal
        f["DRM_Metadata/drmbox_x0"] = np.array(center)
    return pts


def fork_match(xyz, stations, tol):
    """Replica of the fork's rule: nearest station per node, ``d < tol``."""
    d = np.empty(len(xyz))
    idx = np.empty(len(xyz), dtype=int)
    for a in range(0, len(xyz), 1024):
        blk = xyz[a:a + 1024]
        d2 = ((blk[:, None, :] - stations[None, :, :]) ** 2).sum(-1)
        idx[a:a + 1024] = d2.argmin(1)
        d[a:a + 1024] = np.sqrt(d2[np.arange(len(blk)), idx[a:a + 1024]])
    return d < tol, idx, d


def to_model(pts, *, crd_scale=1e6, T=T_SR, x0=X0_SR, center=CENTER):
    """The fork's transform, written out independently of the module."""
    v = (np.asarray(pts) - np.asarray(center)) * crd_scale
    return v @ np.asarray(T, dtype=float).T + np.asarray(x0)


def build(lat, **kw):
    g = apeGmsh(model_name="station_box", verbose=False)
    g.begin()
    try:
        res = g.parts.add_station_soil_box(lat, **kw)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return res, fem


def node_ids(fem, pg):
    return {int(i) for i in np.asarray(fem.nodes.select(pg=pg).ids)}


def coords_of(fem, ids):
    all_ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    pos = {int(t): k for k, t in enumerate(all_ids)}
    return xyz[[pos[i] for i in sorted(ids)]]


def sr_specs(lat):
    """Near field one lattice cell inside the inner shell (A6) and a pit."""
    nf = NearField(lo=(lat.x[2], lat.y[2], lat.z[1]), hi=(lat.x[-3], lat.y[-3], 0.0),
                   size=(2500.0, 2500.0, 1000.0))
    pit = Pit(lo=(lat.x[4] + 300.0, lat.y[4] + 200.0, -3000.0),
              hi=(lat.x[-5] - 600.0, lat.y[-5] - 400.0, 0.0))
    return nf, pit


@pytest.fixture(scope="module")
def sr_file(tmp_path_factory):
    f = str(tmp_path_factory.mktemp("station_box") / "two_shell.h5drm")
    pts = write_two_shell(f)
    return f, pts


@pytest.fixture(scope="module")
def sr_lattice(sr_file):
    f, _ = sr_file
    return SoilLattice.from_h5drm(f, crd_scale=1e6, transform=T_SR, x0=X0_SR)


# ─────────────────────────────────────────────────────────────────────
# Lattice
# ─────────────────────────────────────────────────────────────────────

class TestLattice:
    def test_frame_and_lines(self, sr_lattice):
        lat = sr_lattice
        # model x <- station y (NY lines), model y <- station x (NX lines)
        assert len(lat.x) == NY - 2 and len(lat.y) == NX - 2 and len(lat.z) == NZ - 1
        assert lat.surface == "max" and lat.z_surface == pytest.approx(0.0)
        assert lat.z_far == pytest.approx(-(NZ - 2) * 2500.0)
        xl, xh, _yl, _yh, zf = lat.layer
        assert lat.x[0] - xl == pytest.approx(2500.0)
        assert xh - lat.x[-1] == pytest.approx(2500.0)
        assert zf == pytest.approx(-(NZ - 1) * 2500.0)
        # centred on x0 (the fork moves drmbox_x0 to x0)
        assert 0.5 * (lat.x[0] + lat.x[-1]) == pytest.approx(X0_SR[0])
        assert lat.frame.distance_tolerance == 1.0          # 4e-4 x 2.5 m, in mm
        kw = lat.frame.pattern_kwargs()
        assert kw["crd_scale"] == 1e6 and kw["x0"] == X0_SR
        assert kw["transform"] == ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))

    def test_stations_in_model_frame(self, sr_file, sr_lattice):
        _, pts = sr_file
        np.testing.assert_allclose(sr_lattice.stations, to_model(pts), atol=1e-9)

    def test_z_down_identity(self, tmp_path):
        f = str(tmp_path / "zdown.h5drm")
        write_two_shell(f)
        lat = SoilLattice.from_h5drm(f, crd_scale=1000.0)
        assert lat.surface == "min" and lat.z_surface == pytest.approx(0.0)
        assert lat.frame.distance_tolerance == pytest.approx(1e-3)  # metres

    def test_regular(self):
        lat = SoilLattice.regular(lo=(0, 0, -30), hi=(105, 105, 0), spacing=2.5)
        assert len(lat.x) == 43 and len(lat.z) == 13 and lat.layer is None


class TestRefusals:
    def test_complete_grid(self, tmp_path):
        f = str(tmp_path / "full.h5drm")
        write_two_shell(f, complete=True)
        with pytest.raises(ValueError, match="add_DRM_box_from_h5drm"):
            SoilLattice.from_h5drm(f, crd_scale=1e6, transform=T_SR)

    def test_missing_station(self, tmp_path):
        f = str(tmp_path / "partial.h5drm")
        write_two_shell(f, drop=3)
        with pytest.raises(ValueError, match="3 missing"):
            SoilLattice.from_h5drm(f, crd_scale=1e6, transform=T_SR)

    def test_internal_flags(self, tmp_path):
        f = str(tmp_path / "flags.h5drm")
        write_two_shell(f, flip_internal=2)
        with pytest.raises(ValueError, match="internal"):
            SoilLattice.from_h5drm(f, crd_scale=1e6, transform=T_SR)

    def test_rotation(self, sr_file):
        c, s = np.cos(0.3), np.sin(0.3)
        with pytest.raises(ValueError, match="signed permutation"):
            SoilLattice.from_h5drm(sr_file[0], transform=((c, -s, 0), (s, c, 0), (0, 0, 1)))

    def test_depth_axis_off_z(self, sr_file):
        with pytest.raises(ValueError, match="depth axis"):
            SoilLattice.from_h5drm(sr_file[0], transform=((0, 0, 1), (0, 1, 0), (1, 0, 0)))

    def test_drm_needs_exterior(self, sr_lattice):
        g = apeGmsh(model_name="refuse", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="needs an Exterior"):
                g.parts.add_station_soil_box(sr_lattice)
            with pytest.raises(ValueError, match="from_h5drm"):
                g.parts.add_station_soil_box(
                    SoilLattice.regular(lo=(0, 0, -10), hi=(10, 10, 0), spacing=2.5),
                    exterior=Exterior(5.0, 2.5))
        finally:
            g.end()

    def test_nearfield_on_the_shell(self, sr_lattice):
        lat = sr_lattice
        g = apeGmsh(model_name="refuse_nf", verbose=False)
        g.begin()
        try:
            nf = NearField(lo=(lat.x[0], lat.y[2], lat.z[1]),
                           hi=(lat.x[-3], lat.y[-3], 0.0), size=2500.0)
            with pytest.raises(ValueError, match="A6"):
                g.parts.add_station_soil_box(lat, exterior=Exterior(5000.0, 2500.0),
                                             nearfield=nf)
            nf = NearField(lo=(lat.x[2] + 100.0, lat.y[2], lat.z[1]),
                           hi=(lat.x[-3], lat.y[-3], 0.0), size=2500.0)
            with pytest.raises(ValueError, match="not on an interior lattice line"):
                g.parts.add_station_soil_box(lat, exterior=Exterior(5000.0, 2500.0),
                                             nearfield=nf)
        finally:
            g.end()

    def test_nearfield_lines_outside_the_block(self, sr_lattice):
        lat = sr_lattice
        g = apeGmsh(model_name="refuse_nf_lines", verbose=False)
        g.begin()
        try:
            for lines in (((lat.x[0] - 1.0,), (), ()), ((), (), (lat.z[0] - 10.0,)),
                          ((), (lat.y[-1] + 5.0,), ())):
                nf = NearField(lo=(lat.x[2], lat.y[2], lat.z[1]),
                               hi=(lat.x[-3], lat.y[-3], 0.0), size=2500.0,
                               lines=lines)
                with pytest.raises(ValueError, match="outside the block"):
                    g.parts.add_station_soil_box(
                        lat, exterior=Exterior(5000.0, 2500.0), nearfield=nf)
        finally:
            g.end()

    def test_pit_off_lines_without_nearfield(self, sr_lattice):
        lat = sr_lattice
        g = apeGmsh(model_name="refuse_pit", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="NearField"):
                g.parts.add_station_soil_box(
                    lat, exterior=Exterior(5000.0, 2500.0),
                    pit=Pit(lo=(lat.x[3] + 10.0, lat.y[3], -2500.0),
                            hi=(lat.x[5], lat.y[5], 0.0)))
        finally:
            g.end()


# ─────────────────────────────────────────────────────────────────────
# Tier-4 configuration: DRM box on the stations
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def t4(sr_lattice):
    nf, pit = sr_specs(sr_lattice)
    res, fem = build(sr_lattice, exterior=Exterior((5000.0, 5000.0), 2500.0),
                     boundary="fixed", nearfield=nf, pit=pit)
    return sr_lattice, nf, pit, res, fem


class TestDRMBox:
    def test_counts_are_the_axes(self, t4):
        _lat, _nf, _pit, res, fem = t4
        assert len(np.asarray(fem.nodes.ids)) == res.expected_nodes
        for key, pg in (("domain", res.domain_pg), ("interior", res.interior_pg),
                        ("lattice", res.lattice_pg), ("nearfield", res.nearfield_pg),
                        ("drm", res.drm_pg), ("exterior", res.exterior_pg)):
            assert len(np.asarray(fem.elements.select(pg=pg).ids)) == res.expected_hex[key], key
        assert res.expected_hex["drm"] > 0 and res.expected_hex["skin"] == 0
        # transfinite everywhere: only hex8
        assert [t.name for t in fem.info.types] == ["hex8"]

    def test_every_drm_node_is_a_station(self, sr_file, t4):
        _, pts = sr_file
        lat, _nf, _pit, res, fem = t4
        stations = to_model(pts)
        drm = node_ids(fem, res.drm_pg)
        xyz_drm = coords_of(fem, drm)
        m, idx, d = fork_match(xyz_drm, stations, lat.frame.distance_tolerance)
        assert m.all() and d.max() < 1e-6
        assert len(drm) == len(stations) == len(set(idx.tolist()))
        chk = res.station_check(fem)                   # on the mesh nodes
        assert chk.n_drm_nodes == chk.n_matched == chk.n_stations == len(stations)
        assert chk.max_distance < 1e-6

    def test_station_check_reads_the_mesh(self, t4):
        import dataclasses

        _lat, _nf, _pit, res, fem = t4
        moved = dataclasses.replace(res, stations=res.stations + 100.0)
        with pytest.raises(RuntimeError, match="does not put one node"):
            moved.station_check(fem)

    def test_no_off_grid_match(self, sr_file, t4):
        _, pts = sr_file
        lat, _nf, _pit, res, fem = t4
        ids = np.asarray(fem.nodes.ids)
        xyz = np.asarray(fem.nodes.coords)
        m, _idx, d = fork_match(xyz, to_model(pts), lat.frame.distance_tolerance)
        drm = node_ids(fem, res.drm_pg)
        matched = {int(i) for i in ids[m]}
        assert matched == drm                     # 0 off-grid matches
        assert m.sum() == len(pts)                # one node per station
        assert d[~m].min() > 1000.0               # nothing near the tolerance

    def test_pit_void_and_pgs(self, t4):
        _lat, _nf, pit, res, fem = t4
        xyz = np.asarray(fem.nodes.coords)
        lo, hi = np.asarray(pit.lo), np.asarray(pit.hi)
        inside = np.all((xyz > lo + 1.0) & (xyz < hi - 1.0), axis=1)
        assert not inside.any()
        assert res.pit_pgs["bottom"] and res.pit_pgs["walls"] and res.pit_pgs["all"]
        bot = coords_of(fem, node_ids(fem, res.pit_pgs["bottom"]))
        np.testing.assert_allclose(bot[:, 2], pit.lo[2])
        walls = coords_of(fem, node_ids(fem, res.pit_pgs["walls"]))
        on_wall = (np.isclose(walls[:, 0], pit.lo[0]) | np.isclose(walls[:, 0], pit.hi[0])
                   | np.isclose(walls[:, 1], pit.lo[1]) | np.isclose(walls[:, 1], pit.hi[1]))
        assert on_wall.all()
        free = coords_of(fem, node_ids(fem, res.free_surface_pg))
        np.testing.assert_allclose(free[:, 2], 0.0, atol=1e-9)
        bnd = coords_of(fem, node_ids(fem, res.boundary_pg))
        ax, ay, az = res.axes["x"], res.axes["y"], res.axes["z"]
        on_outer = (np.isclose(bnd[:, 0], ax.lo) | np.isclose(bnd[:, 0], ax.hi)
                    | np.isclose(bnd[:, 1], ay.lo) | np.isclose(bnd[:, 1], ay.hi)
                    | np.isclose(bnd[:, 2], az.lo))
        assert on_outer.all()

    def test_nearfield_is_tied(self, t4):
        _lat, _nf, _pit, res, fem = t4
        host_pg, emb_pg = res.interface_pgs
        emb = node_ids(fem, emb_pg)
        assert not (emb & node_ids(fem, res.lattice_pg))       # non-conforming
        recs = [r for r in fem.elements.constraints if r.kind == "embedded"]
        assert {int(r.slave_node) for r in recs} == emb
        host = node_ids(fem, host_pg)
        assert all({int(n) for n in r.master_nodes} <= host for r in recs)

    def test_emitted_pattern_line(self, t4, tmp_path):
        lat, _nf, _pit, res, fem = t4
        from apeGmsh.opensees import apeSees

        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        soil = ops.nDMaterial.ElasticIsotropic(E=3400.0, nu=0.262, rho=2.4e-9)
        ops.element.stdBrick(pg=res.domain_pg, material=soil)
        ops.fix(pg=res.boundary_pg, dofs=(1, 1, 1))
        ops.constraints.Penalty(alpha_sp=1e14, alpha_mp=1e14)
        with ops.pattern.H5DRM(factor=1000.0, **res.frame.pattern_kwargs()):
            pass
        deck = tmp_path / "deck.tcl"
        ops.tcl(str(deck))
        lines = [ln for ln in deck.read_text().splitlines() if "pattern H5DRM" in ln]
        assert len(lines) == 1
        tok = lines[0].split()
        assert tok[3].strip("{}") == lat.frame.h5drm            # Tcl-braced path
        assert [float(t) for t in tok[4:8]] == [1000.0, 1e6, 1.0, 1.0]
        assert [float(t) for t in tok[8:17]] == [0, 1, 0, 1, 0, 0, 0, 0, -1]
        assert [float(t) for t in tok[17:20]] == list(X0_SR)


# ─────────────────────────────────────────────────────────────────────
# Tier-3 configuration: same interior + absorbing skin
# ─────────────────────────────────────────────────────────────────────

class TestAbsorbingBox:
    def test_t3_counts_skin_pgs(self, sr_lattice):
        nf, pit = sr_specs(sr_lattice)
        res, fem = build(sr_lattice, drm=False, boundary="absorbing",
                         nearfield=nf, pit=pit)
        assert res.drm_pg == "" and res.exterior_pg == "" and res.frame is None
        assert res.boundary_pg == ""                          # the skin is the boundary
        skin = res.skin
        assert set(skin.skin_pgs) == {
            "L", "R", "F", "K", "B", "LF", "LK", "RF", "RK",
            "BL", "BR", "BF", "BK", "BLF", "BLK", "BRF", "BRK"}
        assert all("B" in skin_pg.rsplit("_", 1)[-1] for skin_pg in skin.bottom_pgs)
        nx, ny = len(sr_lattice.x) - 1, len(sr_lattice.y) - 1
        nz = len(sr_lattice.z) - 1
        n_skin = len(np.asarray(fem.elements.select(pg=skin.skin_all_pg).ids))
        assert n_skin == res.expected_hex["skin"] == (
            (nx + 2) * (ny + 2) * (nz + 1) - nx * ny * nz)
        n_base = sum(len(np.asarray(fem.elements.select(pg=p).ids)) for p in skin.bottom_pgs)
        assert n_base == (nx + 2) * (ny + 2)
        assert len(np.asarray(fem.nodes.ids)) == res.expected_nodes
        assert len(np.asarray(fem.elements.select(pg=res.domain_pg).ids)) == \
            res.expected_hex["domain"]
        # the skin sits on the interior faces, one cell thick
        assert res.axes["x"].lo == pytest.approx(sr_lattice.x[0] - 2500.0)
        assert res.axes["z"].lo == pytest.approx(sr_lattice.z_far - 2500.0)

    def test_t3_interior_equals_t4_interior(self, sr_lattice, t4):
        _lat, nf, pit, res4, fem4 = t4
        res3, fem3 = build(sr_lattice, drm=False, boundary="absorbing",
                           nearfield=nf, pit=pit)

        def key(fem, pg):
            xyz = coords_of(fem, node_ids(fem, pg))
            return np.unique(np.round(xyz, 6), axis=0)

        a, b = key(fem3, res3.interior_pg), key(fem4, res4.interior_pg)
        assert a.shape == b.shape
        np.testing.assert_array_equal(a, b)
        assert res3.expected_hex["interior"] == res4.expected_hex["interior"]

    def test_t3_without_file(self):
        lat = SoilLattice.regular(lo=(-30.5, -37.0, -30.0), hi=(74.5, 68.0, 0.0),
                                  spacing=(2.5, 2.5, 2.5))
        res, fem = build(lat, drm=False, boundary="absorbing", skin_thickness=5.25)
        assert res.axes["x"].lo == pytest.approx(-35.75)
        assert len(np.asarray(fem.elements.select(pg=res.domain_pg).ids)) == 42 * 42 * 12


# ─────────────────────────────────────────────────────────────────────
# Non-uniform / anisotropic station grids and the z-down frame
# ─────────────────────────────────────────────────────────────────────

class TestGrids:
    def test_non_uniform_anisotropic(self, tmp_path):
        f = str(tmp_path / "graded.h5drm")
        gx = np.r_[0.0, np.cumsum([3.0, 2.5, 2.5, 2.5, 2.0, 2.0, 2.5, 3.0])] / 1000
        xs = CENTER[0] - gx.mean() + gx
        zs = np.arange(6) * 0.002                          # 2 m vertical
        pts = write_two_shell(f, xs=xs, zs=zs)
        lat = SoilLattice.from_h5drm(f, crd_scale=1e6, transform=T_SR, x0=X0_SR)
        res, fem = build(lat, exterior=Exterior(4000.0, 2000.0))
        ids = np.asarray(fem.nodes.ids)
        m, _idx, d = fork_match(np.asarray(fem.nodes.coords), to_model(pts),
                                lat.frame.distance_tolerance)
        assert {int(i) for i in ids[m]} == node_ids(fem, res.drm_pg)
        assert m.sum() == len(pts) and d[m].max() < 1e-6
        assert len(ids) == res.expected_nodes

    def test_z_down_frame_drm(self, tmp_path):
        f = str(tmp_path / "zdown.h5drm")
        pts = write_two_shell(f)
        lat = SoilLattice.from_h5drm(f, crd_scale=1000.0)
        res, fem = build(lat, exterior=Exterior(5.0, 2.5), boundary="fixed")
        ids = np.asarray(fem.nodes.ids)
        m, _idx, _d = fork_match(np.asarray(fem.nodes.coords),
                                to_model(pts, crd_scale=1000.0, T=np.eye(3),
                                         x0=(0, 0, 0)),
                                lat.frame.distance_tolerance)
        assert {int(i) for i in ids[m]} == node_ids(fem, res.drm_pg)
        assert m.sum() == len(pts)
        assert res.station_check(fem).n_matched == len(pts)
        bnd = coords_of(fem, node_ids(fem, res.boundary_pg))
        assert bnd[:, 2].max() == pytest.approx(res.axes["z"].hi)   # bottom at max z

    @pytest.mark.parametrize("drm", [True, False])
    def test_z_down_absorbing_refused(self, tmp_path, drm):
        # ASDAbsorbingBoundary3D takes the min-z face of a "B" element as its
        # base; in a z-down model that face would sit on the soil side.
        f = str(tmp_path / "zdown.h5drm")
        write_two_shell(f)
        lat = SoilLattice.from_h5drm(f, crd_scale=1000.0)
        assert lat.surface == "min"
        g = apeGmsh(model_name="refuse_zdown", verbose=False)
        g.begin()
        try:
            with pytest.raises(ValueError, match="z-up"):
                g.parts.add_station_soil_box(
                    lat, drm=drm, exterior=Exterior(5.0, 2.5) if drm else None,
                    boundary="absorbing")
        finally:
            g.end()

    def test_z_up_absorbing_base_below_the_soil(self, tmp_path):
        # the same file read z-up: the "B" skin is the min-z layer
        f = str(tmp_path / "zup.h5drm")
        write_two_shell(f)
        lat = SoilLattice.from_h5drm(f, crd_scale=1000.0, transform=Z_UP)
        res, fem = build(lat, drm=False, boundary="absorbing")
        base = coords_of(fem, set().union(
            *(node_ids(fem, pg) for pg in res.skin.bottom_pgs)))
        soil = coords_of(fem, node_ids(fem, res.domain_pg))
        assert base[:, 2].min() < soil[:, 2].min()
        assert base[:, 2].max() == pytest.approx(soil[:, 2].min())


# ─────────────────────────────────────────────────────────────────────
# Boxes far from the global origin (any x0)
# ─────────────────────────────────────────────────────────────────────

class TestFarFromOrigin:
    @pytest.mark.parametrize("crd_scale, T, x0", [
        (1000.0, Z_UP, (0.0, 500.0, 0.0)),                 # metres, 500 m off
        (1000.0, Z_UP, (22000.0, 15500.0, 0.0)),           # metres, 27 km off
        (1e6, T_SR, (-5.0e6, -3.2e6, -2.0e4)),             # mm, large negative
        (1000.0, Z_UP, (0.0, -50.0, 0.0)),                 # partial miss
    ], ids=["m-500", "m-27km", "mm-negative", "m-partial"])
    def test_drm_box_on_the_stations(self, sr_file, crd_scale, T, x0):
        f, pts = sr_file
        lat = SoilLattice.from_h5drm(f, crd_scale=crd_scale, transform=T, x0=x0)
        h = 2.5 * crd_scale / 1000.0
        res, fem = build(lat, exterior=Exterior(2 * h, h))
        ids = np.asarray(fem.nodes.ids)
        assert len(ids) == res.expected_nodes
        assert len(np.asarray(fem.elements.select(pg=res.domain_pg).ids)) ==             res.expected_hex["domain"]
        stations = to_model(pts, crd_scale=crd_scale, T=T, x0=x0)
        m, _idx, _d = fork_match(np.asarray(fem.nodes.coords), stations,
                                 lat.frame.distance_tolerance)
        assert m.sum() == len(pts)
        assert {int(i) for i in ids[m]} == node_ids(fem, res.drm_pg)
        assert res.station_check(fem).n_matched == len(pts)

    @pytest.mark.parametrize("lo, hi", [
        ((-16.25, -63.75, -15.0), (16.25, -36.25, 0.0)),
        ((9983.75, -20013.75, -315.0), (10016.25, -19986.25, -300.0)),
    ], ids=["partial", "far"])
    def test_absorbing_regular(self, lo, hi):
        lat = SoilLattice.regular(lo=lo, hi=hi, spacing=2.5)
        res, fem = build(lat, drm=False, boundary="absorbing")
        assert len(np.asarray(fem.nodes.ids)) == res.expected_nodes
        assert len(np.asarray(fem.elements.select(pg=res.domain_pg).ids)) == 13 * 11 * 6
        assert len(np.asarray(fem.elements.select(pg=res.skin.skin_all_pg).ids)) ==             res.expected_hex["skin"]


# ─────────────────────────────────────────────────────────────────────
# Real file (opt-in)
# ─────────────────────────────────────────────────────────────────────

REAL = os.environ.get("APEGMSH_H5DRM_REAL", "")


@pytest.mark.skipif(not (REAL and os.path.exists(REAL)),
                    reason="set APEGMSH_H5DRM_REAL to a two-shell .h5drm")
def test_real_file_tier4():
    lat = SoilLattice.from_h5drm(REAL, crd_scale=1e6, transform=T_SR, x0=X0_SR)
    nf = NearField(lo=(-13000.0, -19500.0, -27500.0), hi=(57000.0, 50500.0, 0.0),
                   size=(2500.0, 2500.0, 1000.0))
    pit = Pit(lo=(0.0, 0.0, -6000.0), hi=(44000.0, 31000.0, 0.0))
    res, fem = build(lat, exterior=Exterior((22500.0, 22500.0), 4500.0),
                     nearfield=nf, pit=pit)
    ids = np.asarray(fem.nodes.ids)
    m, idx, d = fork_match(np.asarray(fem.nodes.coords), lat.stations,
                           lat.frame.distance_tolerance)
    n_st = len(lat.stations)
    print(f"\nreal: {len(ids)} nodes, {res.expected_hex}, matched {int(m.sum())} "
          f"of {n_st}, max d {d[m].max():.3e}, nearest unmatched {d[~m].min():.1f}")
    assert m.sum() == n_st == len(set(idx[m].tolist()))
    assert {int(i) for i in ids[m]} == node_ids(fem, res.drm_pg)
    assert d[m].max() < 1e-3 * lat.frame.distance_tolerance
    assert res.station_check(fem).n_matched == n_st
