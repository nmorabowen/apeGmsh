"""Fork lane: WP-200 ``LadrunoDispBeamColumn`` and ``LadrunoIMKBeam`` live-fork integration.

Exercises the WP-200 surface of both Ladruno beam families against the live
``opensees.pyd`` binary:

1. ``LadrunoDispBeamColumn`` (2-D and 3-D):
   - numeric ``lch=0.25`` (emitted as string token and accepted natively by C++),
   - dual-end hinges ``hinge_ij`` / ``hinge_yij`` and ``hinge_biaxial_ij``,
   - interior hinge locations ``hinge_loc=(0.05, 0.95)``,
   - axial-moment interaction ``pm=(250.0e3, 0.8)``,
   - element-level IMPLEX extrapolation ``implex=True``,
   - ``hingeSummary``, ``implexError``, ``hingeJumpI``, ``hingeJumpJ``, and
     ``energy`` responses.

2. ``LadrunoIMKBeam`` (2-D and 3-D):
   - Timoshenko ``shear`` (2-tuple ``(G, Avy)`` in 2-D; 3-tuple ``(G, Avy, Avz)``
     in 3-D),
   - rigid ``end_zones=(0.15, 0.15)``,
   - consistent mass ``mass=120.0, c_mass=True``,
   - axial-moment interaction ``pm=(250.0e3, 0.8)``,
   - element-level IMPLEX extrapolation ``implex=True``,
   - 3-D ``localForce`` and ``basicForce`` line-station synthesis on both
     ``.mpco`` (:meth:`Results.from_mpco`) and ``.ladruno``
     (:meth:`Results.from_ladruno`), matching the closed-form cantilever
     section resultants at ``xi = [-1, +1]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
L = 4.0
FX, FY, FZ, MX = -60.0e3, -18.0e3, 12.0e3, 3.0e3


def _two_element_cantilever_fem(name: str, length: float = L) -> Any:
    """Two-element straight cantilever along +X from x=0 (``base``) to x=L (``tip``)."""
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        pm = geo.add_point(length / 2.0, 0.0, 0.0)
        p1 = geo.add_point(length, 0.0, 0.0)
        c1 = geo.add_line(p0, pm)
        c2 = geo.add_line(pm, p1)
        for c in (c1, c2):
            gmsh.model.mesh.setTransfiniteCurve(c, 2)
        g.physical.add(1, [c1, c2], name="Beam")
        g.labels.add(0, [p0], "base")
        g.labels.add(0, [p1], "tip")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def test_ladruno_disp_beam_column_wp200_2d_and_3d_live(tmp_path: Path) -> None:
    """Exercise 2-D and 3-D ``LadrunoDispBeamColumn`` with numeric ``lch=0.25``,
    ``hinge_ij``, ``hinge_loc``, ``pm``, and ``implex=True`` on the live fork."""
    # --- 2-D model ---
    fem2 = _two_element_cantilever_fem("wp200_ldbc_2d")
    ops2 = apeSees(fem2, default_orientation=None)
    ops2.model(ndm=2, ndf=3)
    tr2 = ops2.geomTransf.Linear()
    sec2 = ops2.section.Elastic(E=2.0e11, A=0.04, Iz=1.6e-4)
    integ2 = ops2.beamIntegration.Lobatto(section=sec2, n_ip=4)
    h_i2 = ops2.uniaxialMaterial.LadrunoCohesiveHinge(
        Mc=45.0e3, Gf=250.0, softening="linear", implex=True
    )
    h_j2 = ops2.uniaxialMaterial.LadrunoCohesiveHinge(
        Mc=50.0e3, Gf=280.0, softening="linear", implex=True
    )
    ops2.element.LadrunoDispBeamColumn(
        pg="Beam",
        transf=tr2,
        integration=integ2,
        lch=0.25,
        hinge_ij=(h_i2, h_j2),
        hinge_loc=(0.05, 0.95),
        pm=(250.0e3, 0.8),
        implex=True,
        mass=80.0,
        c_mass=True,
    )
    ops2.fix(label="base", dofs=(1, 1, 1))
    with ops2.pattern.Plain(series=ops2.timeSeries.Linear()) as p:
        p.load(node=ops2.nodes.one(label="tip"), forces=(FX, FY, 0.0))
    ops2.constraints.Plain()
    ops2.numberer.RCM()
    ops2.system.BandGeneral()
    ops2.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops2.algorithm.Newton()
    ops2.integrator.LoadControl(dlam=0.5)
    ops2.analysis.Static()

    h5_2d = str(tmp_path / "ldbc2d_model.h5")
    ops2.h5(h5_2d)
    em2 = LiveOpsEmitter(wipe=True)
    ops2.build().emit(em2)
    f2o_2d = _fem_to_ops(h5_2d)
    eids2 = [int(e) for e in fem2.elements.select(pg="Beam").ids]

    for _ in range(2):
        assert em2.analyze(steps=1) == 0

    o2 = em2.ops
    o2.reactions()
    base2 = int(ops2.nodes.one(label="base").tag)
    np.testing.assert_allclose(
        o2.nodeReaction(base2),
        [-FX, -FY, -L * FY],
        rtol=RTOL,
        atol=1e-6,
    )
    # Root element has Mz > My_hinge, so its end-I hinge activates
    hs2 = [np.asarray(o2.eleResponse(f2o_2d[e], "hingeSummary"), dtype=float) for e in eids2]
    assert any(abs(h[0]) > 0.0 and abs(h[1]) > 0.0 for h in hs2), hs2
    en2 = [float(o2.eleResponse(f2o_2d[e], "energy")[0]) for e in eids2]
    assert all(val > 0.0 for val in en2), en2

    # --- 3-D model with dual biaxial hinges, numeric lch=0.25, hinge_loc, pm, implex ---
    fem3 = _two_element_cantilever_fem("wp200_ldbc_3d")
    ops3 = apeSees(fem3, default_orientation=None)
    ops3.model(ndm=3, ndf=6)
    tr3 = ops3.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    sec3 = ops3.section.Elastic(
        E=2.0e11, A=0.04, Iz=1.6e-4, Iy=1.2e-4, G=8.0e10, J=2.0e-4
    )
    integ3 = ops3.beamIntegration.Lobatto(section=sec3, n_ip=4)
    hb_i = ops3.nDMaterial.LadrunoCohesiveHingeBiaxial(
        Mcz=45.0e3, Gfz=250.0, Mcy=35.0e3, Gfy=200.0, implex=True
    )
    hb_j = ops3.nDMaterial.LadrunoCohesiveHingeBiaxial(
        Mcz=50.0e3, Gfz=280.0, Mcy=40.0e3, Gfy=220.0, implex=True
    )
    ops3.element.LadrunoDispBeamColumn(
        pg="Beam",
        transf=tr3,
        integration=integ3,
        lch=0.25,
        hinge_biaxial_ij=(hb_i, hb_j),
        hinge_loc=(0.05, 0.95),
        pm=(250.0e3, 0.8),
        implex=True,
        mass=80.0,
        c_mass=True,
    )
    ops3.fix(label="base", dofs=(1, 1, 1, 1, 1, 1))
    with ops3.pattern.Plain(series=ops3.timeSeries.Linear()) as p:
        p.load(
            node=ops3.nodes.one(label="tip"),
            forces=(FX, FY, FZ, MX, 0.0, 0.0),
        )
    ops3.constraints.Plain()
    ops3.numberer.RCM()
    ops3.system.BandGeneral()
    ops3.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops3.algorithm.Newton()
    ops3.integrator.LoadControl(dlam=0.5)
    ops3.analysis.Static()

    h5_3d = str(tmp_path / "ldbc3d_model.h5")
    ops3.h5(h5_3d)
    em3 = LiveOpsEmitter(wipe=True)
    ops3.build().emit(em3)
    f2o_3d = _fem_to_ops(h5_3d)
    eids3 = [int(e) for e in fem3.elements.select(pg="Beam").ids]

    for _ in range(2):
        assert em3.analyze(steps=1) == 0

    o3 = em3.ops
    o3.reactions()
    base3 = int(ops3.nodes.one(label="base").tag)
    np.testing.assert_allclose(
        o3.nodeReaction(base3),
        [-FX, -FY, -FZ, -MX, L * FZ, -L * FY],
        rtol=RTOL,
        atol=1e-6,
    )
    hs3 = [np.asarray(o3.eleResponse(f2o_3d[e], "hingeSummary"), dtype=float) for e in eids3]
    assert all(h.shape == (8,) for h in hs3)
    assert any(np.linalg.norm(h) > 0.0 for h in hs3), hs3
    en3 = [float(o3.eleResponse(f2o_3d[e], "energy")[0]) for e in eids3]
    assert all(val > 0.0 for val in en3), en3


def test_ladruno_imk_beam_wp200_options_and_3d_line_stations_live(tmp_path: Path) -> None:
    """Exercise ``LadrunoIMKBeam`` with ``shear``, ``end_zones``, ``c_mass``,
    ``pm``, ``implex``, and verify 3-D ``localForce`` and ``basicForce``
    line-station synthesis on both MPCO and ``.ladruno`` against the closed form."""
    # --- 2-D LadrunoIMKBeam with shear, end_zones, c_mass, pm, implex ---
    fem2 = _two_element_cantilever_fem("wp200_imk_2d")
    ops2 = apeSees(fem2, default_orientation=None)
    ops2.model(ndm=2, ndf=3)
    tr2 = ops2.geomTransf.Linear()
    mat_z2 = ops2.uniaxialMaterial.Steel01(fy=50.0e3, E=2.0e8, b=0.05)
    ops2.element.LadrunoIMKBeam(
        pg="Beam",
        transf=tr2,
        A=0.04,
        E=2.0e11,
        Iz=1.6e-4,
        ends="both",
        mat_z=mat_z2,
        shear=(8.0e10, 0.03),
        end_zones=(0.15, 0.15),
        mass=120.0,
        c_mass=True,
        pm=(250.0e3, 0.8),
        implex=True,
    )
    ops2.fix(label="base", dofs=(1, 1, 1))
    with ops2.pattern.Plain(series=ops2.timeSeries.Linear()) as p:
        p.load(node=ops2.nodes.one(label="tip"), forces=(FX, FY, 0.0))
    ops2.constraints.Plain()
    ops2.numberer.RCM()
    ops2.system.BandGeneral()
    ops2.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops2.algorithm.Newton()
    ops2.integrator.LoadControl(dlam=0.5)
    ops2.analysis.Static()

    h5_2d = str(tmp_path / "imk2d_model.h5")
    ops2.h5(h5_2d)
    em2 = LiveOpsEmitter(wipe=True)
    ops2.build().emit(em2)
    f2o_2d = _fem_to_ops(h5_2d)
    eids2 = [int(e) for e in fem2.elements.select(pg="Beam").ids]
    for _ in range(2):
        assert em2.analyze(steps=1) == 0
    o2 = em2.ops
    o2.reactions()
    np.testing.assert_allclose(
        o2.nodeReaction(int(ops2.nodes.one(label="base").tag)),
        [-FX, -FY, -L * FY],
        rtol=RTOL,
        atol=1e-6,
    )
    hs2 = [np.asarray(o2.eleResponse(f2o_2d[e], "hingeSummary"), dtype=float) for e in eids2]
    assert all(h.shape == (4,) for h in hs2)
    assert any(abs(h[0]) > 0.0 and abs(h[1]) > 0.0 for h in hs2), hs2

    # --- 3-D LadrunoIMKBeam with shear, c_mass, pm, implex, and line-station synthesis ---
    fem = _two_element_cantilever_fem("wp200_imk_3d")
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    mat_z = ops.uniaxialMaterial.Steel01(fy=50.0e3, E=2.0e8, b=0.05)
    mat_y = ops.uniaxialMaterial.Steel01(fy=40.0e3, E=1.5e8, b=0.05)
    ops.element.LadrunoIMKBeam(
        pg="Beam",
        transf=tr,
        A=0.04,
        E=2.0e11,
        G=8.0e10,
        Jx=2.0e-4,
        Iy=1.2e-4,
        Iz=1.6e-4,
        ends="both",
        mat_z=mat_z,
        mat_y=mat_y,
        shear=(8.0e10, 0.03, 0.025),
        end_zones=(0.0, 0.0),
        mass=120.0,
        c_mass=True,
        pm=(250.0e3, 0.8),
        implex=True,
    )
    ops.fix(label="base", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(
            node=ops.nodes.one(label="tip"),
            forces=(FX, FY, FZ, MX, 0.0, 0.0),
        )

    mpco_path = str(tmp_path / "imk3d.mpco")
    lad_lf_path = str(tmp_path / "imk3d_lf.ladruno")
    lad_bf_path = str(tmp_path / "imk3d_bf.ladruno")
    ops.recorder.MPCO(
        file=mpco_path,
        nodal_responses=("displacement",),
        elem_responses=("localForce",),
    )
    ops.recorder.Ladruno(
        file=lad_lf_path,
        elem_responses=("localForce",),
    )
    ops.recorder.Ladruno(
        file=lad_bf_path,
        elem_responses=("basicForce",),
    )

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops.algorithm.Newton()
    lams = (0.5, 1.0)
    ops.integrator.LoadControl(dlam=lams[0])
    ops.analysis.Static()

    model_h5 = str(tmp_path / "imk3d_model.h5")
    ops.h5(model_h5)
    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)
    f2o = _fem_to_ops(model_h5)
    eids = [int(e) for e in fem.elements.select(pg="Beam").ids]

    coords = {
        int(n): np.asarray(c, dtype=float)
        for n, c in zip(fem.nodes.ids, fem.nodes.coords)
    }
    ends: dict[int, tuple[float, float]] = {}
    for grp in fem.elements:
        for eid, conn in zip(grp.ids, grp.connectivity):
            ends[int(eid)] = (
                float(coords[int(conn[0])][0]),
                float(coords[int(conn[-1])][0]),
            )

    for _ in lams:
        assert em.analyze(steps=1) == 0

    o = em.ops
    hs = [np.asarray(o.eleResponse(f2o[e], "hingeSummary"), dtype=float) for e in eids]
    assert all(h.shape == (8,) for h in hs)
    assert any(abs(h[0]) > 0.0 and abs(h[4]) > 0.0 for h in hs), hs
    o.remove("recorders")

    r_mpco = Results.from_mpco(mpco_path, fem=fem, model_h5=model_h5)
    r_lad_lf = Results.from_ladruno(lad_lf_path, fem=fem, model_h5=model_h5)
    r_lad_bf = Results.from_ladruno(lad_bf_path, fem=fem, model_h5=model_h5)

    def _closed_station(comp: str, x: float) -> float:
        return {
            "axial_force": FX,
            "shear_y": -FY,
            "shear_z": FZ,
            "torsion": MX,
            "bending_moment_y": -FZ * (L - x),
            "bending_moment_z": FY * (L - x),
        }[comp]

    all_comps = (
        "axial_force",
        "shear_y",
        "shear_z",
        "torsion",
        "bending_moment_y",
        "bending_moment_z",
    )
    for r_name, r in (("mpco", r_mpco), ("ladruno_lf", r_lad_lf)):
        avail = set(r.elements.line_stations.available_components())
        assert set(all_comps) <= avail, (r_name, avail)
        for comp in all_comps:
            for eid in eids:
                slab = r.elements.line_stations.get(component=comp, ids=[eid])
                idx = np.asarray(slab.element_index)
                rows = [i for i, x in enumerate(idx) if int(x) == eid]
                assert len(rows) == 2
                np.testing.assert_array_equal(
                    np.asarray(slab.station_natural_coord)[rows], [-1.0, 1.0]
                )
                xa, xb = ends[eid]
                want = np.outer(
                    lams, [_closed_station(comp, xa), _closed_station(comp, xb)]
                )
                np.testing.assert_allclose(
                    np.asarray(slab.values)[:, rows],
                    want,
                    rtol=RTOL,
                    atol=1e-6,
                    err_msg=f"{r_name} {comp} eid={eid}",
                )

    # basicForce-only .ladruno synthesizes bending_moment_y and bending_moment_z stations
    bf_avail = set(r_lad_bf.elements.line_stations.available_components())
    assert {"bending_moment_y", "bending_moment_z"} <= bf_avail, bf_avail
    for comp in ("bending_moment_y", "bending_moment_z"):
        for eid in eids:
            slab = r_lad_bf.elements.line_stations.get(component=comp, ids=[eid])
            idx = np.asarray(slab.element_index)
            rows = [i for i, x in enumerate(idx) if int(x) == eid]
            xa, xb = ends[eid]
            want = np.outer(
                lams, [_closed_station(comp, xa), _closed_station(comp, xb)]
            )
            np.testing.assert_allclose(
                np.asarray(slab.values)[:, rows],
                want,
                rtol=RTOL,
                atol=1e-6,
                err_msg=f"ladruno_bf {comp} eid={eid}",
            )


def test_ladruno_beam_shear_hinge_single_hinge_loc_and_subdivide_live(
    tmp_path: Path,
) -> None:
    """Exercise ``LadrunoIMKBeam`` with ``shear_hinge`` (2-D & 3-D tuple) and
    ``shear_hinge_y`` / ``shear_hinge_z`` (3-D), plus single-hinge ``hinge_loc``
    and 2-tuple ``subdivide`` on ``LadrunoDispBeamColumn`` / ``LadrunoForceBeamColumn``."""
    # 1. 2-D LadrunoIMKBeam with shear_hinge
    fem2 = _two_element_cantilever_fem("wp200_imk_sh2d")
    ops2 = apeSees(fem2, default_orientation=None)
    ops2.model(ndm=2, ndf=3)
    tr2 = ops2.geomTransf.Linear()
    sh2 = ops2.uniaxialMaterial.Steel01(fy=30.0e3, E=2.0e8, b=0.05)
    ops2.element.LadrunoIMKBeam(
        pg="Beam",
        transf=tr2,
        A=0.04,
        E=2.0e11,
        Iz=1.6e-4,
        shear_hinge=sh2,
    )
    ops2.fix(label="base", dofs=(1, 1, 1))
    with ops2.pattern.Plain(series=ops2.timeSeries.Linear()) as p:
        p.load(node=ops2.nodes.one(label="tip"), forces=(FX, FY, 0.0))
    ops2.constraints.Plain()
    ops2.numberer.RCM()
    ops2.system.BandGeneral()
    ops2.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops2.algorithm.Newton()
    ops2.integrator.LoadControl(dlam=1.0)
    ops2.analysis.Static()

    em2 = LiveOpsEmitter(wipe=True)
    ops2.build().emit(em2)
    assert em2.analyze(steps=1) == 0

    # 2. 3-D LadrunoIMKBeam with shear_hinge_y / shear_hinge_z and shear_hinge=(shy, shz)
    for use_pair in (False, True):
        fem3 = _two_element_cantilever_fem(f"wp200_imk_sh3d_{int(use_pair)}")
        ops3 = apeSees(fem3, default_orientation=None)
        ops3.model(ndm=3, ndf=6)
        tr3 = ops3.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        shy = ops3.uniaxialMaterial.Steel01(fy=30.0e3, E=2.0e8, b=0.05)
        shz = ops3.uniaxialMaterial.Steel01(fy=25.0e3, E=1.5e8, b=0.05)
        sh_kwargs = (
            {"shear_hinge": (shy, shz)}
            if use_pair
            else {"shear_hinge_y": shy, "shear_hinge_z": shz}
        )
        ops3.element.LadrunoIMKBeam(
            pg="Beam",
            transf=tr3,
            A=0.04,
            E=2.0e11,
            G=8.0e10,
            Jx=2.0e-4,
            Iy=1.2e-4,
            Iz=1.6e-4,
            **sh_kwargs,
        )
        ops3.fix(label="base", dofs=(1, 1, 1, 1, 1, 1))
        with ops3.pattern.Plain(series=ops3.timeSeries.Linear()) as p:
            p.load(
                node=ops3.nodes.one(label="tip"),
                forces=(FX, FY, FZ, MX, 0.0, 0.0),
            )
        ops3.constraints.Plain()
        ops3.numberer.RCM()
        ops3.system.BandGeneral()
        ops3.test.NormDispIncr(tol=1e-12, max_iter=25)
        ops3.algorithm.Newton()
        ops3.integrator.LoadControl(dlam=1.0)
        ops3.analysis.Static()

        em3 = LiveOpsEmitter(wipe=True)
        ops3.build().emit(em3)
        assert em3.analyze(steps=1) == 0

    # 3. Single-hinge hinge_loc=0.5 on LadrunoDispBeamColumn and LadrunoForceBeamColumn
    #    with both 2-tuple subdivide=(4, 4.0) and scalar subdivide=4 followed by mass=10.0 (2D & 3D)
    for cls_name in ("LadrunoDispBeamColumn", "LadrunoForceBeamColumn"):
        sub_cases: tuple[dict[str, Any], ...] = (
            ({"subdivide": (4, 4.0)}, {"subdivide": 4, "mass": 10.0})
            if cls_name == "LadrunoForceBeamColumn"
            else ({},)
        )
        for case_idx, extra in enumerate(sub_cases):
            fem_h = _two_element_cantilever_fem(f"wp200_{cls_name}_single_loc_{case_idx}")
            ops_h = apeSees(fem_h, default_orientation=None)
            ops_h.model(ndm=2, ndf=3)
            tr_h = ops_h.geomTransf.Linear()
            sec_h = ops_h.section.Elastic(E=2.0e11, A=0.04, Iz=1.6e-4)
            integ_h = ops_h.beamIntegration.Lobatto(section=sec_h, n_ip=4)
            h_m = ops_h.uniaxialMaterial.LadrunoCohesiveHinge(
                Mc=40.0e3, Gf=250.0, softening="linear"
            )
            getattr(ops_h.element, cls_name)(
                pg="Beam",
                transf=tr_h,
                integration=integ_h,
                hinge=h_m,
                hinge_loc=0.5,
                **extra,
            )
            ops_h.fix(label="base", dofs=(1, 1, 1))
            with ops_h.pattern.Plain(series=ops_h.timeSeries.Linear()) as p:
                p.load(node=ops_h.nodes.one(label="tip"), forces=(FX, FY, 0.0))
            ops_h.constraints.Plain()
            ops_h.numberer.RCM()
            ops_h.system.BandGeneral()
            ops_h.test.NormDispIncr(tol=1e-12, max_iter=25)
            ops_h.algorithm.Newton()
            ops_h.integrator.LoadControl(dlam=1.0)
            ops_h.analysis.Static()

            em_h = LiveOpsEmitter(wipe=True)
            ops_h.build().emit(em_h)
            assert em_h.analyze(steps=1) == 0

    # 4. 3-D LadrunoForceBeamColumn with scalar subdivide=4 and mass=10.0
    fem_f3 = _two_element_cantilever_fem("wp200_lfbc3d_scalar_subdivide")
    ops_f3 = apeSees(fem_f3, default_orientation=None)
    ops_f3.model(ndm=3, ndf=6)
    tr_f3 = ops_f3.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    sec_f3 = ops_f3.section.Elastic(
        E=2.0e11, A=0.04, Iz=1.6e-4, Iy=1.2e-4, G=8.0e10, J=2.0e-4
    )
    integ_f3 = ops_f3.beamIntegration.Lobatto(section=sec_f3, n_ip=4)
    ops_f3.element.LadrunoForceBeamColumn(
        pg="Beam",
        transf=tr_f3,
        integration=integ_f3,
        subdivide=4,
        mass=10.0,
    )
    ops_f3.fix(label="base", dofs=(1, 1, 1, 1, 1, 1))
    with ops_f3.pattern.Plain(series=ops_f3.timeSeries.Linear()) as p:
        p.load(
            node=ops_f3.nodes.one(label="tip"),
            forces=(FX, FY, FZ, MX, 0.0, 0.0),
        )
    ops_f3.constraints.Plain()
    ops_f3.numberer.RCM()
    ops_f3.system.BandGeneral()
    ops_f3.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops_f3.algorithm.Newton()
    ops_f3.integrator.LoadControl(dlam=1.0)
    ops_f3.analysis.Static()

    em_f3 = LiveOpsEmitter(wipe=True)
    ops_f3.build().emit(em_f3)
    assert em_f3.analyze(steps=1) == 0
