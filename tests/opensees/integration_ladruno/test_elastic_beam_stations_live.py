"""Fork lane: elastic-beam line stations against the solver and the closed form.

Gap G17 of the DomainCapture parity table (#1767, #1945). An elastic beam has
no sections, so its line stations are a two-station slab (xi = -1, +1)
synthesised from ``localForce``. Until this file, no test compared those
stations with a real solve on MPCO or ``.ladruno``. Here two models are built
through the bridge with sequential tags (FEM ids are not OpenSees tags):

* ``cantilever3d``: an ``elasticBeamColumn`` cantilever along +x, two
  elements, fixed at x = 0, tip load ``(Fx, Fy, Fz, Mx)``;
* ``simply2d``: an ``elasticBeamColumn`` beam, pinned at x = 0 and on a
  roller at x = L, two elements, a point load ``-P`` at midspan and an axial
  pull ``H`` at the roller.

Each is solved in two load steps and recorded with ``localForce`` to MPCO
and to ``.ladruno`` in the same run. A test that reads 0 steps cannot pass:
every comparison is a ``(2, 2)`` array per element.

**The convention: the force-beam section convention.** A station must equal
what an adjacent ``forceBeamColumn`` reports through ``section force`` at
the same point, so a force/elastic junction is continuous. Statics on the
free body ``x..L`` gives the stress resultants on the section's +x face;
OpenSees's section convention reports N, Vz, T, My and Mz as those
resultants and Vy as ``dMz/dx``, its negative. In terms of ``localForce``
(end i forces then end j forces, element local axes) that is, per
component, station 1 = ``s * end_i`` and station 2 = ``-s * end_j``, with
``s`` in :data:`SIGN`. ``s`` was measured on fork 79e0623670 at a
forceBeam / elasticBeam / forceBeam junction (aggregated 6-component
section), and :func:`test_section_convention_is_statics` re-proves it at
every run against the closed form below.

The closed forms (load factor ``lam``):

* cantilever at x: ``N = Fx``, ``Vy = -Fy``, ``Vz = Fz``, ``T = Mx``,
  ``My = -Fz (L - x)``, ``Mz = Fy (L - x)``; at the root ``|M| = P L``;
* simply supported at x <= L/2: ``N = H``, ``Vy = P/2``, ``Mz = P x / 2``
  (``P L / 4`` at the midspan node); mirrored for x >= L/2, ``Vy = -P/2``.

Every route today takes station 1 as recorded and negates station 2 for
every component (``s = +1``), which agrees only where ``s = +1``. The other
components are pinned ``xfail(strict=True)`` on the defects that own them,
asserting the right value, so the fix flips them to a strict XPASS:

* #2009: N and M (and, in 3-D, Vz and T) have the opposite sign to the
  force-beam section convention on every route;
* #2010: MPCO drops a 2-D ``localForce`` bucket's ``V``/``M``;
* #2011: a ``basicForce``-only ``.ladruno`` file reads basic-system end
  moments as stations.
"""
from __future__ import annotations

from typing import Any, Callable

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
KINDS = ("mpco", "ladruno")
LAMS = (0.5, 1.0)                     # LoadControl dlam = 0.5, two steps

# Cantilever3d: length, tip load (Fx, Fy, Fz, Mx).
L3 = 4.0
FX, FY, FZ, MX = 1.0e3, -2.0e3, 3.0e3, 4.0e2
# Simply2d: span, midspan load magnitude (downward), axial pull at the roller.
L2 = 6.0
P2, H2 = 5.0e3, 2.0e3

COMPS = {
    "cantilever3d": ("axial_force", "shear_y", "shear_z", "torsion",
                     "bending_moment_y", "bending_moment_z"),
    "simply2d": ("axial_force", "shear_y", "bending_moment_z"),
}

#: Station = (s * end_i, -s * end_j) of ``localForce``, per component.
SIGN = {
    "cantilever3d": {"axial_force": -1, "shear_y": +1, "shear_z": -1,
                     "torsion": -1, "bending_moment_y": -1,
                     "bending_moment_z": -1},
    "simply2d": {"axial_force": -1, "shear_y": +1, "bending_moment_z": -1},
}


def _cantilever_closed(comp: str, x: float) -> float:
    return {
        "axial_force": FX, "shear_y": -FY, "shear_z": FZ, "torsion": MX,
        "bending_moment_y": -FZ * (L3 - x), "bending_moment_z": FY * (L3 - x),
    }[comp]


def _simply_closed(comp: str, x: float) -> float:
    """Shear is discontinuous at midspan: call it inside a half."""
    return {
        "axial_force": H2,
        "shear_y": P2 / 2.0 if x < L2 / 2.0 else -P2 / 2.0,
        "bending_moment_z": P2 * min(x, L2 - x) / 2.0,
    }[comp]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _two_element_line(name: str, x_end: float) -> Any:
    """A straight line along +x of two 1-element curves, labelled
    ``a`` (x = 0), ``m`` (x = x_end / 2) and ``b`` (x = x_end)."""
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        pa = geo.add_point(0.0, 0.0, 0.0)
        pm = geo.add_point(x_end / 2.0, 0.0, 0.0)
        pb = geo.add_point(x_end, 0.0, 0.0)
        c1 = geo.add_line(pa, pm)
        c2 = geo.add_line(pm, pb)
        for c in (c1, c2):
            gmsh.model.mesh.setTransfiniteCurve(c, 2)
        g.physical.add(1, [c1, c2], name="Beam")
        g.labels.add(0, [pa], "a")
        g.labels.add(0, [pm], "m")
        g.labels.add(0, [pb], "b")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _cantilever3d(ops: apeSees) -> None:
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))  # local y = Y, z = Z
    ops.element.elasticBeamColumn(pg="Beam", transf=tr, A=0.03, E=2.0e11,
                                  Iz=5.0e-4, Iy=4.0e-4, G=8.0e10, J=3.0e-4)
    ops.fix(label="a", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="b"),
               forces=(FX, FY, FZ, MX, 0.0, 0.0))


def _simply2d(ops: apeSees) -> None:
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beam", transf=tr, A=0.02, E=2.0e11,
                                  Iz=3.0e-4)
    ops.fix(label="a", dofs=(1, 1, 0))
    ops.fix(label="b", dofs=(0, 1, 0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="m"), forces=(0.0, -P2, 0.0))
        p.load(node=ops.nodes.one(label="b"), forces=(H2, 0.0, 0.0))


CASES: dict[str, tuple[Callable[[apeSees], None], float,
                       Callable[[str, float], float]]] = {
    "cantilever3d": (_cantilever3d, L3, _cantilever_closed),
    "simply2d": (_simply2d, L2, _simply_closed),
}


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def _solve(tmp: Any, case: str, elem_responses: tuple[str, ...],
           kinds: tuple[str, ...]) -> dict[str, Any]:
    build, length, _closed = CASES[case]
    fem = _two_element_line(f"g17_{case}", length)
    ops = apeSees(fem, default_orientation=None)       # sequential tags
    build(ops)
    paths = {k: str(tmp / f"{case}.{k}") for k in kinds}
    if "mpco" in paths:
        # A node channel too: an element-only MPCO file reads 0 steps
        # (#1970, pinned in test_element_families_ladruno_live.py).
        ops.recorder.MPCO(file=paths["mpco"],
                          nodal_responses=("displacement",),
                          elem_responses=elem_responses)
    if "ladruno" in paths:
        ops.recorder.Ladruno(file=paths["ladruno"],
                             elem_responses=elem_responses)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=LAMS[0])
    ops.analysis.Static()

    model_h5 = str(tmp / f"{case}_model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    fem_to_ops = _fem_to_ops(model_h5)
    eids = [int(e) for e in fem.elements.select(pg="Beam").ids]
    assert any(fem_to_ops[e] != e for e in eids)       # not vacuous

    # Element end x-coordinates, read from the FEM (closed-form stations).
    coords = {int(n): np.asarray(c, dtype=float)
              for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    ends: dict[int, tuple[float, float]] = {}
    for grp in fem.elements:
        for eid, conn in zip(grp.ids, grp.connectivity):
            xa, xb = coords[int(conn[0])][0], coords[int(conn[-1])][0]
            ends[int(eid)] = (float(xa), float(xb))
    assert all(ends[e][0] < ends[e][1] for e in eids)   # local x = +X

    local: dict[int, list[np.ndarray]] = {e: [] for e in eids}
    for _ in LAMS:
        assert emitter.analyze(steps=1) == 0
        for e in eids:
            local[e].append(np.asarray(
                emitter.ops.eleResponse(fem_to_ops[e], "localForce"),
                dtype=float))
    emitter.ops.remove("recorders")
    readers = {"mpco": Results.from_mpco, "ladruno": Results.from_ladruno}
    return {
        "results": {k: readers[k](p, fem=fem, model_h5=model_h5)
                    for k, p in paths.items()},
        "eids": eids,
        "ends": ends,
        "local": {e: np.stack(v) for e, v in local.items()},  # (T, 2*n)
    }


@pytest.fixture(scope="module")
def solved(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return {
        case: _solve(tmp_path_factory.mktemp(f"g17_{case}"), case,
                     ("localForce",), KINDS)
        for case in CASES
    }


@pytest.fixture(scope="module")
def basic_only(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _solve(tmp_path_factory.mktemp("g17_basic_force"), "cantilever3d",
                  ("basicForce",), ("ladruno",))


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


def _section_from_local(case: str, comp: str, lf: np.ndarray) -> np.ndarray:
    """``(T, 2)`` stations of one element in the force-beam section
    convention, from its ``(T, 2 * n)`` ``localForce`` history."""
    comps = COMPS[case]
    n, k, s = len(comps), comps.index(comp), SIGN[case][comp]
    return np.stack([s * lf[:, k], -s * lf[:, n + k]], axis=1)


def _closed(case: str, comp: str, eid: int, ends: dict) -> np.ndarray:
    """``(T, 2)`` closed-form stations of one element."""
    f = CASES[case][2]
    xa, xb = ends[eid]
    if case == "simply2d":
        # Shear is constant over each half; evaluate it inside the element.
        xa_s = xa if comp != "shear_y" else 0.5 * (xa + xb)
        xb_s = xb if comp != "shear_y" else 0.5 * (xa + xb)
    else:
        xa_s, xb_s = xa, xb
    per_unit = np.array([f(comp, xa_s), f(comp, xb_s)])
    return np.outer(LAMS, per_unit)


def _close(got: np.ndarray, want: np.ndarray, msg: str) -> None:
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=RTOL * float(np.abs(want).max()),
        err_msg=msg,
    )


def _station_values(r: Results, comp: str, eid: int) -> np.ndarray:
    """``(T, 2)`` stations of ``eid`` read back, asserting xi = -1, +1."""
    slab = r.elements.line_stations.get(component=comp, ids=[eid])
    idx = np.asarray(slab.element_index)
    rows = [i for i, x in enumerate(idx) if int(x) == eid]
    assert len(rows) == 2, f"{comp} of element {eid}: rows {rows}"
    np.testing.assert_array_equal(
        np.asarray(slab.station_natural_coord)[rows], [-1.0, 1.0])
    return np.asarray(slab.values)[:, rows]


def _known(kind: str, case: str, comp: str) -> list[str]:
    why = []
    if SIGN[case][comp] < 0:
        why.append("#2009: elastic-beam station sign against the force-beam "
                   "section convention")
    if kind == "mpco" and case == "simply2d" and comp != "axial_force":
        why.append("#2010: MPCO drops 2-D localForce V/M")
    return why


def _params() -> list[Any]:
    out = []
    for kind in KINDS:
        for case, comps in COMPS.items():
            for comp in comps:
                why = _known(kind, case, comp)
                marks = ([pytest.mark.xfail(strict=True, raises=AssertionError,
                                            reason="; ".join(why))]
                         if why else [])
                out.append(pytest.param(kind, case, comp, marks=marks,
                                        id=f"{kind}-{case}-{comp}"))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(CASES))
def test_section_convention_is_statics(
    solved: dict[str, Any], case: str,
) -> None:
    """The oracle itself: ``localForce`` mapped by :data:`SIGN` equals the
    closed form, for every component, element and step. No reader runs."""
    sol = solved[case]
    for comp in COMPS[case]:
        for eid in sol["eids"]:
            _close(_section_from_local(case, comp, sol["local"][eid]),
                   _closed(case, comp, eid, sol["ends"]),
                   f"{case} {comp} of element {eid}")


def test_closed_form_reaches_pl_and_pl_over_4(solved: dict[str, Any]) -> None:
    """The closed forms hit the card's anchors: ``P L`` at the cantilever
    root, ``P L / 4`` at the simply supported midspan node."""
    c = solved["cantilever3d"]
    root = next(e for e in c["eids"] if c["ends"][e][0] == 0.0)
    _close(_closed("cantilever3d", "bending_moment_z", root, c["ends"])[-1, 0],
           np.float64(FY * L3), "Mz at the root")
    _close(_closed("cantilever3d", "bending_moment_y", root, c["ends"])[-1, 0],
           np.float64(-FZ * L3), "My at the root")
    s = solved["simply2d"]
    for eid in s["eids"]:
        mid = 1 if s["ends"][eid][1] == L2 / 2.0 else 0
        _close(_closed("simply2d", "bending_moment_z", eid, s["ends"])[-1, mid],
               np.float64(P2 * L2 / 4.0), f"Mz at midspan, element {eid}")


@pytest.mark.parametrize("kind, case, comp", _params())
def test_stations_equal_local_force_and_closed_form(
    solved: dict[str, Any], kind: str, case: str, comp: str,
) -> None:
    """Every station of every element, read back from ``kind``, equals the
    matching ``localForce`` end in the section convention and the closed
    form, at ``rtol=1e-10``."""
    sol = solved[case]
    r = sol["results"][kind]
    for eid in sol["eids"]:
        got = _station_values(r, comp, eid)
        _close(got, _section_from_local(case, comp, sol["local"][eid]),
               f"{kind} {case} {comp} of element {eid} vs localForce")
        _close(got, _closed(case, comp, eid, sol["ends"]),
               f"{kind} {case} {comp} of element {eid} vs closed form")


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2011: basicForce-only .ladruno reads basic-system "
                          "end moments as stations (and #2009's sign)")
@pytest.mark.parametrize("comp", ["bending_moment_y", "bending_moment_z"])
def test_basic_force_only_moments_are_absent_or_section_convention(
    basic_only: dict[str, Any], comp: str,
) -> None:
    """With only ``basicForce`` recorded, a moment station is either absent
    (dropped with a named warning) or in the section convention."""
    ls = basic_only["results"]["ladruno"].elements.line_stations
    if comp not in ls.available_components():
        return
    for eid in basic_only["eids"]:
        _close(_station_values(basic_only["results"]["ladruno"], comp, eid),
               _closed("cantilever3d", comp, eid, basic_only["ends"]),
               f"basicForce-only {comp} of element {eid}")
