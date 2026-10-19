"""Live: fibres and per-Gauss-point material states round-trip through ``.out``
(#1982, link X; the route the #1976 probe selected).

``spec.emit_recorders`` records, ``Results.from_recorders`` reads back, and
every value is held to a closed form, never to the engine's own
``eleResponse`` alone (which shares ``setResponse`` with the recorder):

* **Fibres.** A two-element cantilever along ``+x`` of ``dispBeamColumn`` or
  ``forceBeamColumn``, an elastic fibre section of four fibres, three Lobatto
  points, 2-D and 3-D. Tip loads ``N``, ``Mz`` (and ``My`` in 3-D), plus a tip
  shear ``Fy`` on the force beam, which is exact for it. At the station
  ``x``: ``strain = N/EA - y Mz(x)/EIz + z My/EIy`` with
  ``Mz(x) = Mz + Fy (L - x)``, and ``stress = E strain``. The rows carry
  the FEM ids (the bridge numbers elements sequentially, so the translator
  is exercised), the Lobatto stations, and the section's ``y, z, area`` and
  material tag.
* **Continuum stress and strain.** One ``stdBrick`` of ``ElasticIsotropic``
  under the prescribed trilinear field ``u_x = a x (1 + y + 2 z)``, which the
  element reproduces exactly: a distinct state at every Gauss point, located
  from the element's own nodes, so a point read at the wrong index fails.
  A second, detached brick keeps free DOFs (stock openseespy aborts on an
  empty system).
* **Damage.** One ``stdBrick`` of ``ASDConcrete3D`` (``Te = (0, et, 10 et)``,
  ``Td = (0, 0, 0.9)``) pulled in uniaxial stress to ``4 et`` over 20 steps:
  ``d+ = max(0, 0.1 (eps/et - 1))``, ``d- = 0``. It is recorded per Gauss
  point (``material <gp> damage``); the oracle holds to ``DAMAGE_ATOL`` (the
  material's pre-crack secant is ``0.999991 E``), and the read-back equals
  the live ``eleResponse`` at ``rtol 1e-12``.

The component names are the MPCO route's (``fiber_stress`` / ``fiber_strain``,
``stress_xx`` ..., ``damage_tension`` / ``damage_compression``). Stock
openseespy and the fork alike: no ``ladruno_fork`` marker, so this runs on
CI's ``live-stock`` lane.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.live

pytest.importorskip("openseespy.opensees")

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh.opensees import OpenSeesModel, apeSees  # noqa: E402
from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402
from apeGmsh.opensees.material.nd import (  # noqa: E402
    ASDConcrete3D,
    ElasticIsotropic,
)
from apeGmsh.opensees.section.fiber import FiberPoint  # noqa: E402
from apeGmsh.results import Results  # noqa: E402
from apeGmsh.results.spec._resolved import (  # noqa: E402
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

RTOL = 1e-10

# --- fibres -------------------------------------------------------------
E_FIB = 2.0e5
FIB_A = 1.0e-2
L_BEAM = 2.0
N_TIP = 10.0 / 3.0          # no short decimal form: -precision is load-bearing
MZ_TIP = 1.5 / 7.0
MY_TIP = 0.4 / 11.0
FY_TIP = 0.3 / 13.0         # force beam only: exact there, not for a disp beam
FIB_2D = tuple((y, 0.0) for y in (-0.3, -0.1, 0.1, 0.3))
FIB_3D = ((-0.3, -0.1), (-0.3, 0.1), (0.3, -0.1), (0.3, 0.1))
LOBATTO_3 = (-1.0, 0.0, 1.0)
LAMS = (0.5, 1.0)

# --- cubes --------------------------------------------------------------
E_C = 30_000.0
NU_C = 0.2
FT = 3.0
FC = 30.0
ET = FT / E_C
A_FIELD = 1.0e-5 / 3.0
N_STEPS = 20
U_MAX = 4.0 * ET
DAMAGE_ATOL = 1.0e-6
#: The OpenSees / gmsh hex8 corner order in natural coordinates.
HEX8 = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                 (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=float)


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {int(e): int(t) for tok in meta
                for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])}


def _run(ops: apeSees, fem: Any, tmp: Path, records: tuple, n_steps: int,
         per_step: Callable[[Any, dict[int, int]], None] | None = None,
         ) -> tuple[Results, dict[int, int]]:
    """Build ``ops`` live, record ``records`` on ``.out`` for ``n_steps``,
    and read the stage back with ``Results.from_recorders``."""
    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    f2o = _fem_to_ops(model_h5)
    model = OpenSeesModel.from_h5(model_h5)
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=records)
    out = tmp / "out"
    with spec.emit_recorders(str(out), ops=emitter.ops, model=model) as live:
        live.begin_stage("s", kind="static")
        for _ in range(n_steps):
            assert emitter.analyze(steps=1) == 0
            if per_step is not None:
                per_step(emitter.ops, f2o)
        live.end_stage()
    r = Results.from_recorders(spec, out, fem=fem, stage_id="s", model=model,
                               cache_root=tmp / "cache")
    return r, f2o


def _close(got: Any, want: Any, what: str, *, rtol: float = RTOL) -> None:
    want = np.asarray(want, dtype=float)
    scale = float(np.abs(want).max())
    assert scale > 0.0, what
    np.testing.assert_allclose(np.asarray(got, dtype=float), want, rtol=rtol,
                               atol=rtol * scale, err_msg=what)


# =====================================================================
# Fibres
# =====================================================================

def _beam_fem(name: str) -> Any:
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        a = geo.add_point(0.0, 0.0, 0.0)
        b = geo.add_point(L_BEAM, 0.0, 0.0)
        line = geo.add_line(a, b)
        g.model.sync()
        g.physical.add(1, [line], name="Beam")
        g.physical.add(0, [a], name="Base")
        g.physical.add(0, [b], name="Tip")
        g.mesh.structured.set_transfinite_curve(line, 3)     # two elements
        g.mesh.generation.generate(1)
        return g.mesh.queries.get_fem_data(dim=1)


def _element_x0(fem: Any) -> dict[int, tuple[float, float]]:
    """``{fem_eid: (x_start, length)}``; each element runs ``+x``."""
    coords = {int(n): np.asarray(c, dtype=float)
              for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    out = {}
    for grp in fem.elements:
        for eid, conn in zip(grp.ids, grp.connectivity):
            x0, x1 = coords[int(conn[0])][0], coords[int(conn[-1])][0]
            assert x1 > x0
            out[int(eid)] = (float(x0), float(x1 - x0))
    return out


@pytest.mark.parametrize("element", ["dispBeamColumn", "forceBeamColumn"])
@pytest.mark.parametrize("ndm", [2, 3])
def test_fibres_round_trip_through_out(ndm: int, element: str,
                                       tmp_path: Path) -> None:
    fem = _beam_fem(f"fib_{ndm}_{element}")
    fibres = FIB_2D if ndm == 2 else FIB_3D
    fy = FY_TIP if element == "forceBeamColumn" else 0.0
    ops = apeSees(fem, default_orientation=None)          # sequential tags
    ops.model(ndm=ndm, ndf=3 if ndm == 2 else 6)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=E_FIB)
    sec = ops.section.Fiber(
        fibers=tuple(FiberPoint(material=mat, y=y, z=z, area=FIB_A)
                     for y, z in fibres),
        GJ=None if ndm == 2 else 1.0e6)
    tr = (ops.geomTransf.Linear() if ndm == 2
          else ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)))
    getattr(ops.element, element)(
        pg="Beam", transf=tr,
        integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3))
    ops.fix(pg="Base", dofs=(1,) * (3 if ndm == 2 else 6))
    tip = ((N_TIP, fy, MZ_TIP) if ndm == 2
           else (N_TIP, fy, 0.0, 0.0, MY_TIP, MZ_TIP))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Tip", forces=tip)
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=LAMS[0])
    ops.analysis.Static()

    span = _element_x0(fem)
    cls = f"{'Force' if element == 'forceBeamColumn' else 'Disp'}BeamColumn{ndm}d"
    rec = ResolvedRecorderRecord(
        category="fibers", name="fib",
        components=("fiber_stress", "fiber_strain"), dt=None, n_steps=None,
        element_ids=np.asarray(sorted(span)), element_class_name=cls)
    r, f2o = _run(ops, fem, tmp_path, (rec,), len(LAMS))
    assert any(f2o[e] != e for e in span), "the translation is not exercised"

    assert sorted(r.elements.fibers.available_components()) == [
        "fiber_stress", "fiber_strain"]
    ea = E_FIB * FIB_A * len(fibres)
    eiz = E_FIB * sum(FIB_A * y * y for y, _z in fibres)
    eiy = E_FIB * sum(FIB_A * z * z for _y, z in fibres)
    my = MY_TIP if ndm == 3 else 0.0
    for comp in ("fiber_stress", "fiber_strain"):
        slab = r.elements.fibers.get(component=comp)
        rows = list(zip(np.asarray(slab.element_index).tolist(),
                        np.asarray(slab.gp_index).tolist()))
        assert rows == [(e, g) for e in sorted(span) for g in range(3)
                        for _f in fibres], comp
        np.testing.assert_array_equal(slab.time, LAMS)
        np.testing.assert_array_equal(
            np.column_stack([slab.y, slab.z]), np.tile(fibres, (2 * 3, 1)))
        np.testing.assert_array_equal(slab.area, FIB_A)
        np.testing.assert_array_equal(slab.material_tag, ops.tag_for(mat))
        assert slab.station_natural_coord is not None
        np.testing.assert_allclose(
            slab.station_natural_coord, [LOBATTO_3[g] for _e, g in rows],
            atol=1e-12)
        strain = []
        for (e, g), y, z in zip(rows, slab.y, slab.z):
            x0, length = span[e]
            x = x0 + 0.5 * (LOBATTO_3[g] + 1.0) * length
            mz = MZ_TIP + fy * (L_BEAM - x)
            strain.append(N_TIP / ea - y * mz / eiz + (z * my / eiy if my else 0.0))
        want = np.outer(LAMS, strain) * (E_FIB if comp == "fiber_stress" else 1.0)
        _close(slab.values, want, f"{ndm}-D {element} {comp}")


# =====================================================================
# Continuum stress / strain and per-Gauss-point damage on stdBrick
# =====================================================================

def _cube_fem(name: str, *, spare: bool) -> Any:
    """A unit hex ``cube``; ``spare`` adds a detached hex at ``x = 3..4``."""
    with apeGmsh(model_name=name, verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="cube")
        g.physical.add(3, "cube", name="cube")
        g.mesh.structured.set_transfinite("cube", n=2)
        if spare:
            g.model.geometry.add_box(3.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="spare")
            g.physical.add(3, "spare", name="spare")
            g.mesh.structured.set_transfinite("spare", n=2)
        g.mesh.generation.generate(dim=3)
        return g.mesh.queries.get_fem_data()


def _cube(fem: Any, pg: str) -> tuple[int, np.ndarray, list[int]]:
    """``(fem_eid, (8, 3) corner coords in connectivity order, node ids)``."""
    coords = {int(n): np.asarray(c, dtype=float)
              for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    (eid,) = [int(e) for e in fem.elements.select(pg=pg).ids]
    for grp in fem.elements:
        for e, conn in zip(grp.ids, grp.connectivity):
            if int(e) == eid:
                nodes = [int(n) for n in conn]
                return eid, np.stack([coords[n] for n in nodes]), nodes
    raise AssertionError(pg)


def _gp_xyz(corners: np.ndarray, natural: np.ndarray) -> np.ndarray:
    """Trilinear map of natural ``(n, 3)`` points through the hex corners."""
    shape = np.prod(1.0 + natural[:, None, :] * HEX8[None, :, :], axis=2) / 8.0
    return shape @ corners


def _trilinear(resp: str, xyz: np.ndarray) -> dict[str, np.ndarray]:
    x, y, z = xyz.T
    exx, gxy, gxz = A_FIELD * (1.0 + y + 2.0 * z), A_FIELD * x, 2.0 * A_FIELD * x
    zero = np.zeros_like(x)
    if resp == "strain":
        return {"strain_xx": exx, "strain_yy": zero, "strain_zz": zero,
                "strain_xy": gxy, "strain_yz": zero, "strain_xz": gxz}
    lam = E_C * NU_C / ((1.0 + NU_C) * (1.0 - 2.0 * NU_C))
    mu = E_C / (2.0 * (1.0 + NU_C))
    return {"stress_xx": (lam + 2.0 * mu) * exx, "stress_yy": lam * exx,
            "stress_zz": lam * exx, "stress_xy": mu * gxy,
            "stress_yz": zero, "stress_xz": mu * gxz}


@pytest.mark.parametrize("resp", ["stress", "strain"])
def test_brick_gauss_points_round_trip_through_out(resp: str,
                                                   tmp_path: Path) -> None:
    fem = _cube_fem(f"tri_{resp}", spare=True)
    eid, corners, nodes = _cube(fem, "cube")
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=E_C, nu=NU_C))
    ops.element.stdBrick(pg="cube", material=mat)
    ops.element.stdBrick(pg="spare", material=mat)
    _sid, spare_xyz, spare_nodes = _cube(fem, "spare")
    ops.fix(nodes=[n for n, c in zip(spare_nodes, spare_xyz) if c[2] == 0.0],
            dofs=(1, 1, 1))
    ops.fix(nodes=nodes, dofs=(0, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for n, (x, y, z) in zip(nodes, corners):
            if x == 0.0:
                ops.fix(nodes=[n], dofs=(1, 0, 0))
            else:
                p.sp(node=n, dof=1, value=A_FIELD * x * (1.0 + y + 2.0 * z))
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-14, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    comps = tuple(_trilinear(resp, np.zeros((1, 3))))
    rec = ResolvedRecorderRecord(
        category="gauss", name="cube", components=comps, dt=None,
        n_steps=None, element_ids=np.asarray([eid]), element_class_name="Brick")
    r, _f2o = _run(ops, fem, tmp_path, (rec,), 1)
    for comp in comps:
        slab = r.elements.gauss.get(component=comp)
        np.testing.assert_array_equal(slab.element_index, eid)
        natural = np.asarray(slab.natural_coords, dtype=float)
        want = _trilinear(resp, _gp_xyz(corners, natural))[comp]
        assert np.asarray(slab.values).shape == (1, 8), comp
        if not want.any():
            np.testing.assert_allclose(slab.values[0], 0.0, atol=1e-12 * E_C * A_FIELD)
            continue
        _close(slab.values[0], want, f"stdBrick {comp}")
    # The field is not uniform: no two Gauss points share a state.
    xyz = _gp_xyz(corners, np.asarray(slab.natural_coords, dtype=float))
    assert len({tuple(np.round(v, 9)) for v in xyz}) == 8


def test_brick_damage_round_trips_per_gauss_point(tmp_path: Path) -> None:
    fem = _cube_fem("dmg", spare=False)
    eid, corners, nodes = _cube(fem, "cube")
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=3)
    conc = ops.register(ASDConcrete3D(
        E=E_C, v=NU_C,
        Te=(0.0, ET, 10.0 * ET), Ts=(0.0, FT, 0.1 * FT), Td=(0.0, 0.0, 0.9),
        Ce=(0.0, FC / E_C, 10.0 * FC / E_C), Cs=(0.0, FC, FC),
        Cd=(0.0, 0.0, 0.0), lch_ref=1.0, auto_regularize=False))
    ops.element.stdBrick(pg="cube", material=conc)
    for axis in range(3):                     # symmetry rollers
        mask = tuple(int(k == axis) for k in range(3))
        ops.fix(nodes=[n for n, c in zip(nodes, corners) if c[axis] == 0.0],
                dofs=mask)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for n, c in zip(nodes, corners):
            if c[0] == 1.0:
                p.sp(node=n, dof=1, value=U_MAX)
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-14, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0 / N_STEPS)
    ops.analysis.Static()

    live: list[np.ndarray] = []

    def per_step(o: Any, f2o: dict[int, int]) -> None:
        live.append(np.stack([
            np.asarray(o.eleResponse(f2o[eid], "material", str(gp), "damage"),
                       dtype=float) for gp in range(1, 9)]))

    rec = ResolvedRecorderRecord(
        category="gauss", name="dmg",
        components=("damage_tension", "damage_compression"), dt=None,
        n_steps=None, element_ids=np.asarray([eid]), element_class_name="Brick")
    r, _f2o = _run(ops, fem, tmp_path, (rec,), N_STEPS, per_step)
    truth = np.stack(live)                                  # (T, 8, 2)
    eps = U_MAX * np.arange(1, N_STEPS + 1) / N_STEPS
    d_plus = np.maximum(0.0, 0.1 * (eps / ET - 1.0))
    assert d_plus[-1] > 0.25                  # cracked: far from zero
    assert {"damage_tension", "damage_compression"} <= set(
        r.elements.gauss.available_components())
    for k, comp in enumerate(("damage_tension", "damage_compression")):
        slab = r.elements.gauss.get(component=comp)
        values = np.asarray(slab.values)
        assert values.shape == (N_STEPS, 8), comp
        np.testing.assert_array_equal(slab.element_index, eid)
        want = np.repeat((d_plus if k == 0 else 0.0 * d_plus)[:, None], 8, axis=1)
        np.testing.assert_allclose(values, want, rtol=0.0, atol=DAMAGE_ATOL,
                                   err_msg=f"{comp} vs d+ = 0.1 (e/et - 1)")
        np.testing.assert_allclose(values, truth[:, :, k], rtol=1e-12,
                                   atol=1e-17, err_msg=f"{comp} vs eleResponse")
    assert math.isclose(float(slab.time[-1]), 1.0)
