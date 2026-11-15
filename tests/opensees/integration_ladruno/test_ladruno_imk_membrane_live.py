"""Fork lane: ``LadrunoIMKBeam`` as ``LadrunoMembrane`` frames, and its
``plasticRotation`` read back from ``.ladruno``.

YOO's nonlinear model hangs IMK beams under ``LadrunoMembrane`` slabs. Two
contracts, both against the live fork binary:

1. **Membrane load reaches an IMK frame.** A 6 m x 6 m panel on four beams
   with fully fixed corners, ``method="tributary"``, pressure q = 5 kPa.
   Each beam takes the 45-degree triangle, w = q L / 2 = 15 kN/m peak, so
   its end shears are q L^2 / 8 = 22.5 kN and its fixed-end moments
   5 w L^2 / 96 = 28.125 kN m. ``LadrunoIMKBeam`` beams with stiff elastic
   hinges must give ``elasticBeamColumn``'s end forces to 1e-6. Builds
   before fork WP-200 (34df942c7) refused element loads on the IMK beam and
   dropped this load in silence.
2. **``plasticRotation`` records and reads back.** ``recorder ladruno -E
   plasticRotation`` on a yielding IMK cantilever: ``Results.from_ladruno``
   serves the raw token (columns ``thetaZP_1, thetaZP_2, thetaYP_1,
   thetaYP_2``) equal to the live ``eleResponse`` at every step. Despite its
   name the fork returns the TOTAL hinge-spring rotation theta_h (elastic
   spring part included, ``LadrunoIMKBeam::getResponse`` case 4), so it is
   M / k_h before yield and grows by the hardening branch after.
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

pytestmark = [pytest.mark.live, pytest.mark.ladruno_fork]

L = 6.0                      # m, panel side and beam length
Q = 5.0                      # kPa, panel pressure (downward)
E, G, A, J, IY, IZ = 30.0e6, 12.5e6, 0.18, 0.002, 0.003, 0.003
V_END = Q * L * L / 8.0                    # 22.5 kN
M_END = 5.0 * (Q * L / 2.0) * L * L / 96.0  # 28.125 kN m


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


# --------------------------------------------------------------------------
# 1. membrane panel on IMK frames
# --------------------------------------------------------------------------
def _panel_fem() -> Any:
    with apeGmsh(model_name="imk_membrane_panel", verbose=False) as g:
        geo = g.model.geometry
        pts = [geo.add_point(x, y, 0.0)
               for x, y in ((0, 0), (L, 0), (L, L), (0, L))]
        edges = [geo.add_line(pts[k], pts[(k + 1) % 4]) for k in range(4)]
        for c in edges:
            gmsh.model.mesh.setTransfiniteCurve(c, 2)   # one element per edge
        g.physical.add(1, edges, name="Beams")
        g.labels.add(0, pts, "corners")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _corner_loop(fem: Any) -> tuple[int, ...]:
    """The four corner node ids, counter-clockwise from (0, 0)."""
    order = {(0.0, 0.0): 0, (L, 0.0): 1, (L, L): 2, (0.0, L): 3}
    found: dict[int, int] = {}
    for n, c in zip(fem.nodes.ids, fem.nodes.coords):
        key = (round(float(c[0]), 9), round(float(c[1]), 9))
        if key in order:
            found[order[key]] = int(n)
    assert len(found) == 4, found
    return tuple(found[k] for k in range(4))


def _solve_panel(tmp: Path, beam: str) -> dict[int, np.ndarray]:
    fem = _panel_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    if beam == "elastic":
        beams = ops.element.elasticBeamColumn(
            pg="Beams", transf=tr, A=A, E=E, G=G, J=J, Iy=IY, Iz=IZ)
    else:
        stiff = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6 * 6.0 * E * IZ / L)
        beams = ops.element.LadrunoIMKBeam(
            pg="Beams", transf=tr, A=A, E=E, G=G, Jx=J, Iy=IY, Iz=IZ,
            mat_z=stiff, mat_y=stiff)
    mem = ops.element.LadrunoMembrane(
        nodes=_corner_loop(fem), frames=(beams,), method="tributary")
    ops.fix(label="corners", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.surface_load(ele=mem, pressure=-Q)
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    model_h5 = str(tmp / f"panel_{beam}.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    o = emitter.ops
    f2o = _fem_to_ops(model_h5)
    eids = [int(e) for e in fem.elements.select(pg="Beams").ids]
    want = "LadrunoIMKBeam" if beam == "imk" else "ElasticBeam3d"
    assert {o.eleType(f2o[e]) for e in eids} == {want}
    return {e: np.asarray(o.eleResponse(f2o[e], "localForce"), dtype=float)
            for e in eids}


@pytest.fixture(scope="module")
def panel(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("imk_panel")
    return {beam: _solve_panel(tmp, beam) for beam in ("elastic", "imk")}


def test_membrane_load_reaches_imk_frames(panel: dict[str, Any]) -> None:
    """IMK frames carry the membrane's tributary load exactly as elastic ones."""
    for e, f_el in panel["elastic"].items():
        np.testing.assert_allclose(panel["imk"][e], f_el, rtol=1e-6,
                                   atol=1e-6 * M_END)


def test_membrane_panel_hand_values(panel: dict[str, Any]) -> None:
    """The whole panel load q L^2 reaches the beams exactly; each beam's end
    shear q L^2 / 8 and fixed-end moment 5 w L^2 / 96 hold to the kernel's
    meshed tributary partition (about 6e-4 here, elastic beams alike)."""
    total = 0.0
    for f in panel["imk"].values():
        # localForce: [N, Vy, Vz, T, My, Mz] at i, then at j
        v_i = np.hypot(f[1], f[2])
        v_j = np.hypot(f[7], f[8])
        m_i = np.hypot(f[4], f[5])
        m_j = np.hypot(f[10], f[11])
        total += v_i + v_j
        np.testing.assert_allclose([v_i, v_j], V_END, rtol=2e-3)
        np.testing.assert_allclose([m_i, m_j], M_END, rtol=2e-3)
    np.testing.assert_allclose(total, Q * L * L, rtol=1e-9)


# --------------------------------------------------------------------------
# 2. plasticRotation through recorder ladruno
# --------------------------------------------------------------------------
LC = 3.0                     # m, cantilever length
MY = 40.0                    # kN m, hinge yield moment
TIP = (0.0, -1.0, 0.0)       # kN per unit load factor, local -y
LAMS = (5.0, 10.0, 15.0, 20.0)  # yield at P = MY / LC = 13.3 kN
B_HARD = 0.01                # hinge post-yield / initial stiffness


def _cantilever_fem() -> Any:
    with apeGmsh(model_name="imk_plastic_rotation", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(LC, 0.0, 0.0)
        c = geo.add_line(p0, p1)
        gmsh.model.mesh.setTransfiniteCurve(c, 2)
        g.physical.add(1, [c], name="Beam")
        g.labels.add(0, [p0], "base")
        g.labels.add(0, [p1], "tip")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


@pytest.fixture(scope="module")
def cantilever(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("imk_prot")
    fem = _cantilever_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    kh = 1.0e3 * 4.0 * E * IZ / LC
    hinge = ops.uniaxialMaterial.Steel01(fy=MY, E=kh, b=B_HARD)
    ops.element.LadrunoIMKBeam(
        pg="Beam", transf=tr, A=A, E=E, G=G, Jx=J, Iy=IY, Iz=IZ,
        mat_zi=hinge)
    ops.fix(label="base", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tip"),
               forces=(*TIP, 0.0, 0.0, 0.0))
    path = str(tmp / "imk.ladruno")
    ops.recorder.Ladruno(file=path, elem_responses=("plasticRotation",))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=30)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=LAMS[0])
    ops.analysis.Static()
    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    o = emitter.ops
    tag = _fem_to_ops(model_h5)[int(fem.elements.select(pg="Beam").ids[0])]
    live = []
    for _ in LAMS:
        assert emitter.analyze(steps=1) == 0
        live.append(np.asarray(o.eleResponse(tag, "plasticRotation"),
                               dtype=float))
    o.remove("recorders")    # flush and close the file
    return {"live": np.stack(live),
            "results": Results.from_ladruno(path, fem=fem, model_h5=model_h5)}


def test_plastic_rotation_is_available(cantilever: dict[str, Any]) -> None:
    comps = cantilever["results"].elements.available_components()
    assert "plasticRotation" in comps, comps


def test_plastic_rotation_equals_ele_response(
        cantilever: dict[str, Any]) -> None:
    """Every step and column of the ``.ladruno`` read equals the live value."""
    slab = cantilever["results"].elements.get(component="plasticRotation")
    vals = np.asarray(slab.values)               # (T, E, 4)
    assert vals.shape == (len(LAMS), 1, 4), vals.shape
    np.testing.assert_allclose(vals[:, 0, :], cantilever["live"],
                               rtol=1e-12, atol=1e-15)


def test_plastic_rotation_is_hinge_rotation(cantilever: dict[str, Any]) -> None:
    """theta_h = M / k_h before yield; past yield it follows the b*k_h branch."""
    kh = 1.0e3 * 4.0 * E * IZ / LC
    m = np.asarray(LAMS) * LC              # base moment magnitude
    want = np.where(m <= MY, m / kh, MY / kh + (m - MY) / (B_HARD * kh))
    np.testing.assert_allclose(np.abs(cantilever["live"][:, 0]), want,
                               rtol=1e-6)
    assert np.all(cantilever["live"][:, 1:] == 0.0)   # no other hinge
