"""#2290 (C) live oracles: catalogued element classes the bridge builds no
element of, recorded from a raw openseespy deck plus a FEMData archive on
the classic-recorder route (``spec.emit_default``).

Each class is paired by fact (``LiveDomainFacts.pair_elements``) and typed
by its family's token (``_TOKEN_FOR_LIVE_CLASS``); the read decodes it by
the catalog entry of the class its record names. The build is wrapped in
:class:`~tests.test_results_live_default_format.NoMPCO`, so the cases run
the classic-recorder route on pip openseespy and on a build that has MPCO.

Two oracles per class, at rtol 1e-12 against the live domain:

* **values**: every component at every Gauss point (or line station) read
  back equals the live ``eleResponse``;
* **position**: each read-back Gauss point sits where the element sampled.
  Planes and solids: ``D B(xi) u`` from the live nodal displacements at the
  read-back natural coordinates is the live stress there (with the
  mean-dilatation term for ``BbarBrick``; ``Twenty_Node_Brick`` in its own
  serendipity node order). Shells: under an imposed ``w = a x^3`` the
  curvature grows with ``x``, so the read-back points sort by ``x`` as the
  live moments do. ``ForceBeamColumnCBDI2d``: a cantilever's section
  moments are the closed form ``V (L - x)`` at the read-back stations.

The classes stock openseespy cannot record stay unmapped and refuse by
name. A ``Twenty_Node_Brick`` built in Gmsh node order is not tested here:
upstream exits the process at ``setDomain`` on its non-positive Jacobian,
so its refusal by id is the fake-domain case in
``tests/test_results_live_unmapped_classes.py``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from apeGmsh._vocabulary import expand_shorthand
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.results.live import DefaultRouteWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.test_results_live_default_format import NoMPCO

pytestmark = pytest.mark.live

_RTOL = dict(rtol=1e-12, atol=0)
E, NU = 200e9, 0.3
EID = 7

#: Element source order of ``eleResponse(e, "stresses")``, spelled out
#: (not read from the catalog the reader decodes with).
PLANE = ("stress_xx", "stress_yy", "stress_xy")
SOLID = ("stress_xx", "stress_yy", "stress_zz",
         "stress_xy", "stress_yz", "stress_xz")
SHELL = expand_shorthand("shell_resultant", ndm=3, ndf=6)


@pytest.fixture
def opspy() -> Any:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    yield ops
    ops.wipe()


# ---------------------------------------------------------------------------
# FEMData and the route
# ---------------------------------------------------------------------------


def _fem(code: int, name: str, dim: int, order: int, coords: np.ndarray,
         conn: "list[list[int]]", ids: "list[int]") -> FEMData:
    """One element group of Gmsh type ``code``; node ``k + 1`` at
    ``coords[k]``, each row of ``conn`` in Gmsh order."""
    xyz = np.zeros((coords.shape[0], 3))
    xyz[:, :coords.shape[1]] = coords
    info = make_type_info(code=code, gmsh_name=name, dim=dim, order=order,
                          npe=len(conn[0]), count=len(conn))
    group = ElementGroup(element_type=info,
                         ids=np.asarray(ids, dtype=np.int64),
                         connectivity=np.asarray(conn, dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, len(xyz) + 1, dtype=np.int64),
                            node_coords=xyz, physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={code: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=len(xyz), n_elems=len(conn),
                      bandwidth=len(xyz), types=[info]),
        model_name=f"unmapped_{name.replace(' ', '_')}",
    )


def _record(category: str, comps: "tuple[str, ...]", cls: str,
            ids: "list[int]") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(category=category, name="rec",
                                  components=comps, dt=None, n_steps=None,
                                  element_ids=np.asarray(ids, dtype=np.int64),
                                  element_class_name=cls)


def _analysis(ops: Any, *, penalty: bool = False) -> None:
    ops.constraints(*(("Penalty", 1.0e18, 1.0e18) if penalty else ("Plain",)))
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1.0e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def _emit(fem: FEMData, rec: ResolvedRecorderRecord, tmp: Path, ops: Any) -> Any:
    model_h5 = tmp / "model.h5"
    fem.to_h5(str(model_h5))
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(rec,))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_default(tmp / "run", model_h5=model_h5,
                               ops=NoMPCO(ops)) as run:
            assert ops.analyze(1) == 0
    assert run.route == "recorders"
    assert [w.category for w in caught] == [DefaultRouteWarning], [
        str(w.message) for w in caught]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return run.results(cache_root=tmp / "cache")


def _refuses(fem: FEMData, rec: ResolvedRecorderRecord, tmp: Path, ops: Any,
             match: str) -> None:
    model_h5 = tmp / "model.h5"
    fem.to_h5(str(model_h5))
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(rec,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DefaultRouteWarning)
        with pytest.raises(ValueError, match=match):
            with spec.emit_default(tmp / "run", model_h5=model_h5,
                                   ops=NoMPCO(ops)):
                pass


# ---------------------------------------------------------------------------
# Shape functions in each element's own node order, and D B(xi) u
# ---------------------------------------------------------------------------

_HEX_CORNERS = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
#: Twenty_Node_Brick's node order (upstream ``shap3dv``): corners, lower
#: ring, upper ring, verticals.
_HEX20_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)]
_HEX20_NAT = np.vstack([_HEX_CORNERS, [0.5 * (_HEX_CORNERS[a] + _HEX_CORNERS[b])
                                       for a, b in _HEX20_EDGES]])
_QUAD_CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], float)
_QUAD8_NAT = np.vstack([_QUAD_CORNERS, [0.5 * (_QUAD_CORNERS[a] + _QUAD_CORNERS[(a + 1) % 4])
                                        for a in range(4)]])


def _serendipity(nat: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Serendipity (or, for corners only, multilinear) shape functions at
    ``xi`` for nodes at natural coords ``nat``."""
    n = []
    quadratic = nat.shape[0] > 2 ** nat.shape[1]
    for c in nat:
        if not quadratic:
            n.append(np.prod([(1 + x * a) / 2 for x, a in zip(xi, c)]))
        elif np.all(c != 0):
            n.append(np.prod([(1 + x * a) / 2 for x, a in zip(xi, c)])
                     * (float(np.dot(xi, c)) - len(c) + 1))
        else:
            f = 1.0
            for x, a in zip(xi, c):
                f *= (1 - x * x) if a == 0 else (1 + x * a) / 2
            n.append(f)
    return np.asarray(n)


def _d_matrix(ndm: int) -> np.ndarray:
    if ndm == 2:      # plane stress
        return E / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0],
                                           [0, 0, (1 - NU) / 2]])
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[:3, :3] += 2 * mu * np.eye(3)
    D[3:, 3:] = mu * np.eye(3)
    return D


def _strain(nat: np.ndarray, coords: np.ndarray, disp: np.ndarray,
            xi: np.ndarray) -> "tuple[np.ndarray, float]":
    """Engineering strain and ``det J`` at ``xi``."""
    h = 1e-4
    dn = np.array([(_serendipity(nat, xi + h * e) - _serendipity(nat, xi - h * e))
                   / (2 * h) for e in np.eye(len(xi))])
    jac = dn @ coords
    g = np.linalg.solve(jac, dn) @ disp            # du_j/dx_i
    if coords.shape[1] == 2:
        eps = np.array([g[0, 0], g[1, 1], g[1, 0] + g[0, 1]])
    else:
        eps = np.array([g[0, 0], g[1, 1], g[2, 2], g[1, 0] + g[0, 1],
                        g[2, 1] + g[1, 2], g[2, 0] + g[0, 2]])
    return eps, float(np.linalg.det(jac))


def _stress_anchor(nat: np.ndarray, coords: np.ndarray, disp: np.ndarray,
                   xis: np.ndarray, *, bbar: bool) -> np.ndarray:
    """``D B(xi) u`` at each of ``xis``; with ``bbar`` the dilatation is the
    element's volume mean over the same points (equal weights: the
    tensor-product rule)."""
    D = _d_matrix(coords.shape[1])
    pairs = [_strain(nat, coords, disp, xi) for xi in xis]
    eps = np.array([p[0] for p in pairs])
    if bbar:
        det = np.array([p[1] for p in pairs])
        theta = eps[:, :3].sum(axis=1)
        mean = float((det * theta).sum() / det.sum())
        eps[:, :3] += ((mean - theta) / 3.0)[:, None]
    return eps @ D.T


# ---------------------------------------------------------------------------
# Plane and solid Gauss classes
# ---------------------------------------------------------------------------

_SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.2, 1.0], [-0.1, 1.1]])
_CUBE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                  [1.1, 1.0, 1.2], [0.0, 1.0, 1.0]])
_UNIT_CUBE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                       [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
#: Gmsh's own hex20 edge order (slots 8-19), so a FEM hex20 is Gmsh's.
_GMSH_HEX20_EDGES = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6),
                     (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)]


@dataclass(frozen=True)
class Continuum:
    """One plane or solid class: its deck, FEMData and anchor."""

    cls: str
    tcl: str
    fem_code: int
    fem_name: str
    order: int
    coords: np.ndarray        # FEM node coords, Gmsh order
    deck_order: "tuple[int, ...]"   # FEM slot of each eleNodes entry
    nat: np.ndarray           # natural coords of each eleNodes entry
    n_gp: int
    tail: "Callable[[], tuple[Any, ...]]"
    bbar: bool = False


def _quad8_coords() -> np.ndarray:
    c = _SQUARE
    return np.vstack([c, [0.5 * (c[a] + c[(a + 1) % 4]) for a in range(4)]])


def _hex20_gmsh_coords() -> np.ndarray:
    return np.vstack([_CUBE, [0.5 * (_CUBE[a] + _CUBE[b])
                              for a, b in _GMSH_HEX20_EDGES]])


def _hex20_deck_order() -> "tuple[int, ...]":
    """The Gmsh slot of each Twenty_Node_Brick node, read off the edges
    (not from the bridge's table under test)."""
    corners = list(range(8))
    return tuple(corners + [_GMSH_HEX20_EDGES.index(tuple(sorted(e))) + 8
                            for e in _HEX20_EDGES])


_PSTRESS = (1.0, "PlaneStress", 1)
CONTINUA: "dict[str, Continuum]" = {c.cls: c for c in (
    Continuum("SSPquad", "SSPquad", 3, "Quadrilateral 4", 1, _SQUARE,
              (0, 1, 2, 3), _QUAD_CORNERS, 1, lambda: (1, "PlaneStress", 1.0)),
    Continuum("EightNodeQuad", "quad8n", 16, "Quadrilateral 8", 2,
              _quad8_coords(), tuple(range(8)), _QUAD8_NAT, 9,
              lambda: _PSTRESS),
    # A parallelepiped: SSPbrick's one point carries the volume-mean
    # strain, which is B(0) u only there.
    Continuum("SSPbrick", "SSPbrick", 5, "Hexahedron 8", 1, _UNIT_CUBE,
              tuple(range(8)), _HEX_CORNERS, 1, lambda: (1,)),
    Continuum("BbarBrick", "bbarBrick", 5, "Hexahedron 8", 1, _CUBE,
              tuple(range(8)), _HEX_CORNERS, 8, lambda: (1,), bbar=True),
    Continuum("Twenty_Node_Brick", "20NodeBrick", 17, "Hexahedron 20", 2,
              _hex20_gmsh_coords(), _hex20_deck_order(), _HEX20_NAT, 27,
              lambda: (1, 0.0, 0.0, 0.0)),
)}


def _build_continuum(ops: Any, case: Continuum) -> "list[int]":
    """The one element, tagged by its FEM id, with its nodes in its class's
    own order. Its bottom is fixed, its top pulled sideways and down, so
    every component varies over the GPs."""
    ndm = case.coords.shape[1]
    ops.model("basic", "-ndm", ndm, "-ndf", ndm)
    ops.nDMaterial("ElasticIsotropic", 1, E, NU)
    for k, xyz in enumerate(case.coords, 1):
        ops.node(k, *(float(v) for v in xyz))
    up = case.coords[:, ndm - 1]
    for k in np.flatnonzero(up < 1e-9) + 1:
        ops.fix(int(k), *([1] * ndm))
    nodes = [int(i) + 1 for i in case.deck_order]
    ops.element(case.tcl, EID, *nodes, *case.tail())
    top = np.flatnonzero(up > up.max() - 1e-9) + 1
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    load = (3.0e5, -1.0e6) if ndm == 2 else (2.0e5, -1.0e5, -1.0e6)
    for k in top:
        ops.load(int(k), *(v / len(top) for v in load))
    _analysis(ops)
    return nodes


@pytest.mark.parametrize("cls", list(CONTINUA))
def test_a_continuum_class_records_its_live_gauss_stress(cls, opspy, tmp_path) -> None:
    case = CONTINUA[cls]
    comps = PLANE if case.coords.shape[1] == 2 else SOLID
    fem = _fem(case.fem_code, case.fem_name, case.coords.shape[1], case.order,
               case.coords, [list(range(1, len(case.coords) + 1))], [EID])
    nodes = _build_continuum(opspy, case)
    assert opspy.eleType(EID) == cls
    res = _emit(fem, _record("gauss", comps, cls, [EID]), tmp_path, opspy)
    try:
        truth = np.asarray(opspy.eleResponse(EID, "stresses"), dtype=float)
        assert truth.size == case.n_gp * len(comps)
        truth = truth.reshape(case.n_gp, len(comps))
        assert np.abs(truth).max() > 0
        slabs = [res.elements.gauss.get(component=c, ids=np.array([EID]))
                 for c in comps]
        for k, (comp, slab) in enumerate(zip(comps, slabs)):
            assert [int(i) for i in slab.element_index] == [EID] * case.n_gp
            np.testing.assert_allclose(np.asarray(slab.values)[-1], truth[:, k],
                                       **_RTOL, err_msg=f"{cls} {comp}")
        # Position: D B(xi) u at each read-back xi is the live stress there.
        xis = np.asarray(slabs[0].natural_coords, dtype=float)
        assert xis.shape == (case.n_gp, case.coords.shape[1])
        coords = np.array([opspy.nodeCoord(n) for n in nodes], dtype=float)
        disp = np.array([opspy.nodeDisp(n) for n in nodes], dtype=float)
        anchor = _stress_anchor(case.nat, coords, disp, xis, bbar=case.bbar)
        np.testing.assert_allclose(anchor, truth, rtol=1e-7,
                                   atol=1e-7 * np.abs(truth).max(),
                                   err_msg=f"{cls}: D B(xi) u at the read-back "
                                           "Gauss points is not the live stress")
    finally:
        res.close()


# ---------------------------------------------------------------------------
# Shells: an imposed w = a x^3, whose curvature grows with x
# ---------------------------------------------------------------------------

_A = 1.0e-3
_TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.6, 0.8, 0.0]])
_QUAD = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.9, 0.0],
                  [-0.1, 0.8, 0.0]])
SHELLS = {"ShellDKGT": (_TRI, 2, "Triangle 3", 4),
          "ShellNLDKGT": (_TRI, 2, "Triangle 3", 4),
          "ShellNLDKGQ": (_QUAD, 3, "Quadrilateral 4", 4)}


def _shell_x(xyz: np.ndarray, xi: np.ndarray) -> float:
    """Physical x at read-back ``xi``: Gmsh's frame (node 0 carries
    ``1 - u - v`` on a triangle, #2228)."""
    if len(xyz) == 3:
        u, v = xi
        return float(np.array([1 - u - v, u, v]) @ xyz[:, 0])
    return float(_serendipity(_QUAD_CORNERS, xi) @ xyz[:, 0])


@pytest.mark.parametrize("cls", list(SHELLS))
def test_a_shell_class_records_its_live_resultants(cls, opspy, tmp_path) -> None:
    xyz, code, name, n_gp = SHELLS[cls]
    fem = _fem(code, name, 2, 1, xyz, [list(range(1, len(xyz) + 1))], [EID])
    opspy.model("basic", "-ndm", 3, "-ndf", 6)
    opspy.section("ElasticMembranePlateSection", 1, E, NU, 0.1, 0.0)
    for k, c in enumerate(xyz, 1):
        opspy.node(k, *(float(v) for v in c))
    opspy.element(cls, EID, *range(1, len(xyz) + 1), 1)
    opspy.timeSeries("Linear", 1)
    opspy.pattern("Plain", 1, 1)
    for k, (x, _y, _z) in enumerate(xyz, 1):
        for dof, val in enumerate((1.0e-4 * x, 0.0, _A * x**3, 0.0,
                                   -3 * _A * x * x, 0.0), 1):
            opspy.sp(k, dof, val)
    _analysis(opspy, penalty=True)
    assert opspy.eleType(EID) == cls
    res = _emit(fem, _record("gauss", SHELL, cls, [EID]), tmp_path, opspy)
    try:
        truth = np.asarray(opspy.eleResponse(EID, "stresses"), dtype=float)
        assert truth.size == n_gp * len(SHELL)
        truth = truth.reshape(n_gp, len(SHELL))
        sec = np.array([opspy.eleResponse(EID, "section", g, "forces")
                        for g in range(1, n_gp + 1)], dtype=float)
        np.testing.assert_allclose(truth, sec, **_RTOL)  # source order = section's
        slabs = [res.elements.gauss.get(component=c, ids=np.array([EID]))
                 for c in SHELL]
        for k, (comp, slab) in enumerate(zip(SHELL, slabs)):
            assert [int(i) for i in slab.element_index] == [EID] * n_gp
            np.testing.assert_allclose(np.asarray(slab.values)[-1], truth[:, k],
                                       **_RTOL, err_msg=f"{cls} {comp}")
        # Position: the bending moment grows with x (M = -D k, k = 6 a x),
        # so the read-back points sort by x as the moments do. The DKG
        # interpolation is not exact for a cubic w, so the order is the
        # oracle, not the magnitude.
        xis = np.asarray(slabs[0].natural_coords, dtype=float)
        assert xis.shape == (n_gp, 2)
        x = np.array([_shell_x(xyz, xi) for xi in xis])
        m = -np.asarray(slabs[SHELL.index("bending_moment_xx")].values)[-1]
        assert len(set(np.round(x, 6))) == n_gp
        assert np.argsort(x).tolist() == np.argsort(m).tolist(), (x, m)
    finally:
        res.close()


# ---------------------------------------------------------------------------
# Beams: a cantilever's section forces
# ---------------------------------------------------------------------------

L, A_, IZ = 3.0, 1.0e-2, 8.0e-6
N_IP = 4
VY = -2.0e3


def test_a_cbdi_force_beam_records_its_live_section_forces(opspy, tmp_path) -> None:
    """``ForceBeamColumnCBDI2d`` under a transverse tip load only: its
    curvature-based interpolation adds P-delta to the moments under an axial
    load, so the closed form holds with none."""
    cls = "ForceBeamColumnCBDI2d"
    xyz = np.array([[0.0, 0.0], [L, 0.0]])
    fem = _fem(1, "Line 2", 1, 1, xyz, [[1, 2]], [EID])
    opspy.model("basic", "-ndm", 2, "-ndf", 3)
    for k, c in enumerate(xyz, 1):
        opspy.node(k, *(float(v) for v in c))
    opspy.fix(1, 1, 1, 1)
    opspy.geomTransf("Linear", 1)
    opspy.section("Elastic", 1, E, A_, IZ)
    opspy.beamIntegration("Lobatto", 1, 1, N_IP)
    opspy.element("forceBeamColumnCBDI", EID, 1, 2, 1, 1)
    opspy.timeSeries("Linear", 1)
    opspy.pattern("Plain", 1, 1)
    opspy.load(2, 0.0, VY, 0.0)
    _analysis(opspy)
    assert opspy.eleType(EID) == cls
    comps = ("axial_force", "bending_moment_z")
    res = _emit(fem, _record("line_stations", comps, cls, [EID]), tmp_path, opspy)
    try:
        truth = np.array([opspy.eleResponse(EID, "section", k, "force")
                          for k in range(1, N_IP + 1)], dtype=float)
        assert truth.shape == (N_IP, len(comps)) and np.abs(truth).max() > 0
        ls = res.elements.line_stations
        slabs = [ls.get(component=c, ids=np.array([EID])) for c in comps]
        for k, (comp, slab) in enumerate(zip(comps, slabs)):
            np.testing.assert_allclose(np.asarray(slab.values)[-1], truth[:, k],
                                       **_RTOL, err_msg=f"{cls} {comp}")
        # Position: the cantilever's statics, M = V (L - x), at each
        # read-back station.
        xi = np.asarray(slabs[0].station_natural_coord, dtype=float)
        assert xi.shape == (N_IP,) and np.all(np.diff(xi) > 0)
        x = (1.0 + xi) * L / 2.0
        np.testing.assert_allclose(np.asarray(slabs[1].values)[-1],
                                   VY * (L - x), rtol=1e-9,
                                   atol=1e-9 * abs(VY * L))
    finally:
        res.close()


# ---------------------------------------------------------------------------
# What stock cannot record stays unmapped, refused by name
# ---------------------------------------------------------------------------


def _shell_mitc9(ops: Any) -> "tuple[FEMData, ResolvedRecorderRecord]":
    xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0, 0],
                    [1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0], [0.5, 0.5, 0]], float)
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.section("ElasticMembranePlateSection", 1, E, NU, 0.1, 0.0)
    for k, c in enumerate(xyz, 1):
        ops.node(k, *c)
    ops.element("ShellMITC9", EID, *range(1, 10), 1)
    return (_fem(10, "Quadrilateral 9", 2, 2, xyz, [list(range(1, 10))], [EID]),
            _record("gauss", SHELL, "ShellMITC9", [EID]))


def _warping(ops: Any) -> "tuple[FEMData, ResolvedRecorderRecord]":
    xyz = np.array([[0.0, 0.0], [L, 0.0]])
    ops.model("basic", "-ndm", 2, "-ndf", 4)
    for k, c in enumerate(xyz, 1):
        ops.node(k, *c)
    ops.geomTransf("Corotational", 1)
    ops.section("Elastic", 1, E, A_, IZ)
    ops.beamIntegration("Lobatto", 1, 1, N_IP)
    ops.element("forceBeamColumnWarping", EID, 1, 2, 1, 1)
    return (_fem(1, "Line 2", 1, 1, xyz, [[1, 2]], [EID]),
            _record("line_stations", ("axial_force", "bending_moment_z"),
                    "ForceBeamColumnWarping2d", [EID]))


def _elastic_force_beam(ndm: int) -> "Callable[[Any], tuple[FEMData, ResolvedRecorderRecord]]":
    def build(ops: Any) -> "tuple[FEMData, ResolvedRecorderRecord]":
        xyz = np.array([[0.0] * ndm, [L] + [0.0] * (ndm - 1)])
        ops.model("basic", "-ndm", ndm, "-ndf", 3 if ndm == 2 else 6)
        for k, c in enumerate(xyz, 1):
            ops.node(k, *c)
        if ndm == 2:
            ops.geomTransf("Linear", 1)
            ops.section("Elastic", 1, E, A_, IZ)
        else:
            ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
            ops.section("Elastic", 1, E, A_, IZ, IZ, E / 2.6, IZ)
        ops.beamIntegration("Lobatto", 1, 1, N_IP)
        ops.element("elasticForceBeamColumn", EID, 1, 2, 1, 1)
        return (_fem(1, "Line 2", 1, 1, xyz, [[1, 2]], [EID]),
                _record("line_stations", ("axial_force", "bending_moment_z"),
                        ops.eleType(EID), [EID]))
    return build


def _truss2(tcl: str) -> "Callable[[Any], tuple[FEMData, ResolvedRecorderRecord]]":
    def build(ops: Any) -> "tuple[FEMData, ResolvedRecorderRecord]":
        xyz = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        ops.uniaxialMaterial("Elastic", 1, E)
        for k, c in enumerate(xyz, 1):
            ops.node(k, *c)
        ops.element(tcl, EID, 1, 2, 3, 4, A_, 1)
        return (_fem(1, "Line 2", 1, 1, xyz, [[1, 2]], [EID]),
                _record("elements", ("nodal_resisting_force_x",),
                        ops.eleType(EID), [EID]))
    return build


def _mod_elastic(ops: Any) -> "tuple[FEMData, ResolvedRecorderRecord]":
    xyz = np.array([[0.0, 0.0], [L, 0.0]])
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for k, c in enumerate(xyz, 1):
        ops.node(k, *c)
    ops.geomTransf("Linear", 1)
    ops.element("ModElasticBeam2d", EID, 1, 2, A_, E, IZ, 1.0, 1.0, 1.0, 1)
    return (_fem(1, "Line 2", 1, 1, xyz, [[1, 2]], [EID]),
            _record("elements", ("nodal_resisting_force_x",),
                    "ModElasticBeam2d", [EID]))


#: Built by stock, but not recordable on its classic route yet: ShellMITC9
#: answers 84 values for its 72 (ShellMITC9.cpp:518);
#: ElasticForceBeamColumn2d/3d write an empty ``section force`` file (no
#: nodal-force layout either); ForceBeamColumnWarping2d has no stock
#: section it can invert ("could not invert flexibility"); Truss2 and
#: CorotTruss2 record only their axial force, whose classic route is #2290
#: (D); ModElasticBeam2d is held by the #2288 refusal case.
UNMAPPED = {"ShellMITC9": _shell_mitc9,
            "ElasticForceBeamColumn2d": _elastic_force_beam(2),
            "ElasticForceBeamColumn3d": _elastic_force_beam(3),
            "ForceBeamColumnWarping2d": _warping,
            "Truss2": _truss2("Truss2"),
            "CorotTruss2": _truss2("CorotTruss2"),
            "ModElasticBeam2d": _mod_elastic}


@pytest.mark.parametrize("cls", list(UNMAPPED))
def test_a_class_stock_cannot_record_refuses_by_name(cls, opspy, tmp_path) -> None:
    fem, rec = UNMAPPED[cls](opspy)
    assert opspy.eleType(EID) == cls
    _refuses(fem, rec, tmp_path, opspy,
             rf"no bridge element token \('{cls}': FEM elements \[{EID}\]\)")


# ---------------------------------------------------------------------------
# The duplicate-connectivity narrowing, on a real domain
# ---------------------------------------------------------------------------


def _twins(ops: Any, *, truss_class: str = "Truss",
           build_truss: bool = True) -> Any:
    """Beam 5 and truss 6 on nodes [1, 2]; the pairing's facts over them."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from apeGmsh.opensees._recorder_translate import LiveDomainFacts

    xyz = np.array([[0.0, 0.0], [L, 0.0]])
    fem = _fem(1, "Line 2", 1, 1, xyz, [[1, 2], [1, 2]], [5, 6])
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for k, c in enumerate(xyz, 1):
        ops.node(k, *c)
    ops.geomTransf("Linear", 1)
    ops.uniaxialMaterial("Elastic", 1, E)
    ops.element("elasticBeamColumn", 5, 1, 2, A_, E, IZ, 1)
    if build_truss and truss_class == "Truss":
        ops.element("Truss", 6, 1, 2, A_, 1)
    elif build_truss:
        ops.element("elasticBeamColumn", 6, 1, 2, A_, E, IZ, 1)
    return fem, OpenSeesModel, LiveDomainFacts


def _facts(ops: Any, fem: FEMData, tmp: Path, model_cls: Any, facts_cls: Any) -> Any:
    path = tmp / "model.h5"
    fem.to_h5(str(path))
    return facts_cls(ops, model_cls.from_h5(path))


def test_twins_of_other_live_classes_pair_on_a_named_class(opspy, tmp_path) -> None:
    fem, model_cls, facts_cls = _twins(opspy)
    facts = _facts(opspy, fem, tmp_path, model_cls, facts_cls)
    facts.pair_elements([5], where="t", named_classes={5: {"ElasticBeam2d"}})
    assert [r.type_token for r in facts.elements()] == ["elasticBeamColumn"]
    assert facts.live_class(5) == "ElasticBeam2d"


@pytest.mark.parametrize("variant, named", [
    ("equal-class", {5: {"ElasticBeam2d"}}),
    ("no-class", {5: {None}}),
    ("no-record-classes", {}),
    ("twin-not-live", {5: {"ElasticBeam2d"}}),
])
def test_twins_still_refuse_without_a_class_apart(
    variant, named, opspy, tmp_path,
) -> None:
    fem, model_cls, facts_cls = _twins(
        opspy, truss_class="Beam" if variant == "equal-class" else "Truss",
        build_truss=variant != "twin-not-live")
    facts = _facts(opspy, fem, tmp_path, model_cls, facts_cls)
    with pytest.raises(ValueError, match=r"FEM element 5 shares nodes \[1, 2\] "
                                         r"with \[6\]"):
        facts.pair_elements([5], where="t", named_classes=named)
    assert facts.elements() == ()
