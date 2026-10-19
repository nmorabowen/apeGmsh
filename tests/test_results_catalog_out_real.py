"""RESPONSE_CATALOG against a real solve, read back from ``.out`` recorders.

Gap G6 of the DomainCapture parity table (#1767, #1945, slice #1953). Five
DomainCapture tests were the only real-solver check of the catalog's plane,
shell, truss and solid layouts (GP count, component order, natural
coordinates). This file ports their class lists onto the classic recorder
route a stock user gets: ``ResolvedRecorderSpec.emit_recorders`` into the live
domain, then ``Results.from_recorders``. The fork-lane twin, MPCO and
``.ladruno``, is
``tests/opensees/integration_ladruno/test_results_catalog_hdf5_live.py``; it
imports the cases from here.

Each case builds the DomainCapture test's one-element model in the live
domain, solves it, and checks every catalogued component at every GP of the
read-back slab against two oracles:

* **the live vector** at ``rtol=1e-10`` (``.out`` writes 17 digits, #1974).
  It is ``ops.eleResponse(eid, token)`` split by the element's own source
  order, spelled out below rather than read from the catalog: the reader
  under test takes its layout from the catalog, so deriving the order from
  it would assert nothing.
* **an anchor that does not share the element's output path.** Planes and
  solids: ``D B(xi) u`` from the live nodal displacements, evaluated at the
  natural coordinates the reader reports, so a wrong GP order or a wrong
  coordinate table fails as well as a component swap. Shells: the section's
  own response per GP (``eleResponse(e, "section", g, "forces")``) and, for
  ``ElasticMembranePlateSection``, its constitutive law applied to the
  section deformation. Trusses: the applied load of the determinate bar.

The upstream shells in ``ZERO_GAUSS_PROBE_CLASSES`` are included. Under
``algorithm Linear``, which never re-forms the residual after the solve,
their sections lag one step behind the converged strain: zero on the first
step, stale after it. Under
Newton (used here for every case) the element, the section probe and the
recorder all agree; that finding is chain B issue #1992. ShellMITC9 is left
out: its ``eleResponse(e, "stresses")`` corrupted the heap of this process on
the fork build (0xc0000374, two runs in four), and a process-killing call
does not belong in the shared pytest process (AGENTS.md).

The models are raw ``ops`` decks, as in the tests they supersede, because
the bridge has no primitive for SSPquad, EightNodeQuad, SSPbrick or the
NL/triangular DKG shells. The emit side gets its ``(fem_eid, tag)`` pairing from
``tests.fixtures.recorder_models.identity_model``; the read side embeds the
session stub ``model.h5``, as the superseded tests did.

Marked ``live``: the ``live-stock`` lane runs it on pip ``openseespy``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from apeGmsh._vocabulary import expand_shorthand
from apeGmsh.opensees._response_catalog import IntRule, lookup

pytestmark = pytest.mark.live

#: ``.out`` holds 17 significant digits since #1974.
RTOL = 1e-10
#: The anchors differentiate shape functions by central differences, which
#: are exact for these (per-variable quadratic) polynomials up to round-off.
RTOL_ANCHOR = 1e-8

E, NU = 200e9, 0.3
SHELL_H = 0.1

#: Element source order, spelled out (not read from RESPONSE_CATALOG).
PLANE = ("stress_xx", "stress_yy", "stress_xy")
SOLID = ("stress_xx", "stress_yy", "stress_zz",
         "stress_xy", "stress_yz", "stress_xz")
SHELL = ("membrane_force_xx", "membrane_force_yy", "membrane_force_xy",
         "bending_moment_xx", "bending_moment_yy", "bending_moment_xy",
         "transverse_shear_xz", "transverse_shear_yz")


def get_live_ops() -> Any:
    """The module the bridge drives; skip when there is no backend."""
    from apeGmsh.opensees.emitter.live import get_ops

    try:
        return get_ops()
    except Exception as e:  # no OpenSees backend importable at all
        pytest.skip(f"no live OpenSees backend: {e}")


# ---------------------------------------------------------------------------
# One-element models (ported from the DomainCapture tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Built:
    """What a builder put in the domain."""

    node_ids: np.ndarray
    coords: np.ndarray       # (n, ndm)
    ndf: int
    eid: int = 1


class MinimalFem:
    """The node table and snapshot hash the recorder routes read."""

    def __init__(self, built: Built) -> None:
        coords3d = np.zeros((built.coords.shape[0], 3), dtype=np.float64)
        coords3d[:, :built.coords.shape[1]] = built.coords
        self.nodes = SimpleNamespace(ids=built.node_ids, coords=coords3d)
        self.elements: list[Any] = []
        self._ndf = built.ndf

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group: Any) -> None:
        from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group
        write_neutral_zone_into_group(self, group, ndf=self._ndf)


def _nodes(ops: Any, coords: np.ndarray) -> np.ndarray:
    ids = np.arange(1, coords.shape[0] + 1, dtype=np.int64)
    for nid, xyz in zip(ids, coords):
        ops.node(int(nid), *(float(v) for v in xyz))
    return ids


def _loads(ops: Any, nids: list[int], per_node: tuple[float, ...]) -> None:
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid in nids:
        ops.load(int(nid), *(v / len(nids) for v in per_node))


_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])


def _mid(c: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array([0.5 * (c[a] + c[b]) for a, b in pairs])


def _plane(tcl: str, coords: np.ndarray, fixed: list[int], top: list[int],
           tail: tuple[Any, ...]) -> Callable[[Any], Built]:
    def build(ops: Any) -> Built:
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        ops.nDMaterial("ElasticIsotropic", 1, E, NU)
        ids = _nodes(ops, coords)
        for nid in fixed:
            ops.fix(nid, 1, 1)
        ops.element(tcl, 1, *(int(n) for n in ids), *tail)
        # Vertical compression with a lateral share, so every component is
        # non-trivial at every GP.
        _loads(ops, top, (3.0e5, -1.0e6))
        return Built(ids, coords, ndf=2)
    return build


_Q8 = np.vstack([_SQUARE, _mid(_SQUARE, [(0, 1), (1, 2), (2, 3), (3, 0)])])
_T6 = np.vstack([_TRIANGLE, _mid(_TRIANGLE, [(0, 1), (1, 2), (2, 0)])])
_PSTRESS = (1.0, "PlaneStress", 1)

_CUBE = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
])
_TET = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                 [0.5, 0.5, 1.0]])


def _solid(tcl: str, coords: np.ndarray, fixed: list[int],
           top: list[int]) -> Callable[[Any], Built]:
    def build(ops: Any) -> Built:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 3)
        ops.nDMaterial("ElasticIsotropic", 1, E, NU)
        ids = _nodes(ops, coords)
        for nid in fixed:
            ops.fix(nid, 1, 1, 1)
        ops.element(tcl, 1, *(int(n) for n in ids), 1)
        _loads(ops, top, (2.0e5, -1.0e5, -1.0e6))
        return Built(ids, coords, ndf=3)
    return build


_SHELL_SQUARE = np.hstack([_SQUARE, np.zeros((4, 1))])
_SHELL_TRIANGLE = np.hstack([_TRIANGLE, np.zeros((3, 1))])


def _shell(tcl: str, coords: np.ndarray, fixed: list[int], top: list[int],
           *, layered: bool = False) -> Callable[[Any], Built]:
    def build(ops: Any) -> Built:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        if layered:
            ops.nDMaterial("ElasticIsotropic", 1, E, NU)
            ops.nDMaterial("PlateFiber", 2, 1)
            ops.section("LayeredShell", 1, 3, 2, 0.0333, 2, 0.0334, 2, 0.0333)
        else:
            ops.section("ElasticMembranePlateSection", 1, E, NU, SHELL_H, 0.0)
        ids = _nodes(ops, coords)
        for nid in fixed:
            ops.fix(nid, 1, 1, 1, 1, 1, 1)
        ops.element(tcl, 1, *(int(n) for n in ids), 1)
        # In-plane pull plus a transverse tip load: membrane, bending and
        # transverse shear all respond.
        _loads(ops, top, (1.0e4, 3.0e3, -1.0e3, 0.0, 0.0, 0.0))
        return Built(ids, coords, ndf=6)
    return build


def _truss(tcl: str) -> Callable[[Any], Built]:
    def build(ops: Any) -> Built:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 3)
        ops.uniaxialMaterial("Elastic", 1, E)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ids = _nodes(ops, coords)
        ops.fix(1, 1, 1, 1)
        ops.fix(2, 0, 1, 1)
        ops.element(tcl, 1, 1, 2, 0.001, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, TRUSS_P, 0.0, 0.0)
        return Built(ids, coords, ndf=3)
    return build


TRUSS_P = 1.0e5


# ---------------------------------------------------------------------------
# The case table: one row per class the DomainCapture files covered
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One catalogued class and the test of it this file supersedes."""

    id: str
    cpp_class: str
    int_rule: int
    build: Callable[[Any], Built]
    comps: tuple[str, ...]
    n_gp: int
    anchor: str              # "strain" | "section" | "truss"
    topology: str = ""       # shape functions for the "strain" anchor
    token: str = "stresses"  # eleResponse / recorder token
    catalog_token: str = "stress"
    plane_section: bool = False    # anchor with the membrane-plate law
    supersedes: str = ""


_PLANES = "tests/test_results_catalog_planes_real.py::test_real_2d_capture_for_class"
_SHELLS = "tests/test_results_catalog_shells_real.py::test_shell_capture_for_class"
_LAYERED = ("tests/test_results_catalog_shells_real.py::"
            "test_layered_shell_uses_same_catalog_entry")
_VOCAB = ("tests/test_results_shell_resultant_vocabulary.py::"
          "test_declared_shell_record_captures_real_resultants")
_TRUSSES = "tests/test_results_catalog_trusses_real.py::test_truss_capture_for_class"
_SOLIDS = "tests/test_results_catalog_solids_real.py::test_real_capture_for_class"
_NEW = "(none: DomainCapture skipped it)"

CASES: dict[str, Case] = {c.id: c for c in (
    Case("FourNodeQuad", "FourNodeQuad", IntRule.Quad_GL_2,
         _plane("quad", _SQUARE, [1, 2], [3, 4], _PSTRESS),
         PLANE, 4, "strain", "quad4", supersedes=_PLANES),
    Case("Tri31", "Tri31", IntRule.Triangle_GL_1,
         _plane("tri31", _TRIANGLE, [1, 2], [3], _PSTRESS),
         PLANE, 1, "strain", "tri3", supersedes=_PLANES),
    Case("SixNodeTri", "SixNodeTri", IntRule.Triangle_GL_2,
         _plane("tri6n", _T6, [1, 2, 4], [3], _PSTRESS),
         PLANE, 3, "strain", "tri6", supersedes=_PLANES),
    Case("SSPquad", "SSPquad", IntRule.Quad_GL_1,
         _plane("SSPquad", _SQUARE, [1, 2], [3, 4], (1, "PlaneStress", 1.0)),
         PLANE, 1, "strain", "quad4", supersedes=_PLANES),
    Case("EightNodeQuad", "EightNodeQuad", IntRule.Quad_GL_3,
         _plane("quad8n", _Q8, [1, 2, 5], [3, 4, 7], _PSTRESS),
         PLANE, 9, "strain", "quad8", supersedes=_PLANES),
    Case("ASDShellQ4", "ASDShellQ4", IntRule.Quad_GL_2,
         _shell("ASDShellQ4", _SHELL_SQUARE, [1, 2], [3, 4]),
         SHELL, 4, "section", plane_section=True,
         supersedes=f"{_SHELLS}; {_VOCAB}"),
    Case("ASDShellT3", "ASDShellT3", IntRule.Triangle_GL_2B,
         _shell("ASDShellT3", _SHELL_TRIANGLE, [1, 2], [3]),
         SHELL, 3, "section", plane_section=True, supersedes=_SHELLS),
    Case("ASDShellQ4-layered", "ASDShellQ4", IntRule.Quad_GL_2,
         _shell("ASDShellQ4", _SHELL_SQUARE, [1, 2], [3, 4], layered=True),
         SHELL, 4, "section", supersedes=_LAYERED),
    Case("ShellMITC4", "ShellMITC4", IntRule.Quad_GL_2,
         _shell("ShellMITC4", _SHELL_SQUARE, [1, 2], [3, 4]),
         SHELL, 4, "section", plane_section=True, supersedes=_NEW),
    Case("ShellDKGQ", "ShellDKGQ", IntRule.Quad_GL_2,
         _shell("ShellDKGQ", _SHELL_SQUARE, [1, 2], [3, 4]),
         SHELL, 4, "section", plane_section=True, supersedes=_NEW),
    Case("ShellNLDKGQ", "ShellNLDKGQ", IntRule.Quad_GL_2,
         _shell("ShellNLDKGQ", _SHELL_SQUARE, [1, 2], [3, 4]),
         SHELL, 4, "section", supersedes=_NEW),
    Case("ShellDKGT", "ShellDKGT", IntRule.Triangle_GL_2C,
         _shell("ShellDKGT", _SHELL_TRIANGLE, [1, 2], [3]),
         SHELL, 4, "section", plane_section=True, supersedes=_NEW),
    Case("ShellNLDKGT", "ShellNLDKGT", IntRule.Triangle_GL_2C,
         _shell("ShellNLDKGT", _SHELL_TRIANGLE, [1, 2], [3]),
         SHELL, 4, "section", supersedes=_NEW),
    Case("Truss", "Truss", IntRule.Line_GL_1, _truss("Truss"),
         ("axial_force",), 1, "truss", token="axialForce",
         catalog_token="axial_force", supersedes=_TRUSSES),
    Case("CorotTruss", "CorotTruss", IntRule.Line_GL_1, _truss("corotTruss"),
         ("axial_force",), 1, "truss", token="axialForce",
         catalog_token="axial_force", supersedes=_TRUSSES),
    Case("FourNodeTetrahedron", "FourNodeTetrahedron", IntRule.Tet_GL_1,
         _solid("FourNodeTetrahedron", _TET, [1, 2, 3], [4]),
         SOLID, 1, "strain", "tet4", supersedes=_SOLIDS),
    Case("Brick", "Brick", IntRule.Hex_GL_2,
         _solid("stdBrick", _CUBE, [1, 2, 3, 4], [5, 6, 7, 8]),
         SOLID, 8, "strain", "hex8", supersedes=_SOLIDS),
    Case("SSPbrick", "SSPbrick", IntRule.Hex_GL_1,
         _solid("SSPbrick", _CUBE, [1, 2, 3, 4], [5, 6, 7, 8]),
         SOLID, 1, "strain", "hex8", supersedes=_SOLIDS),
)}


def solve(ops: Any) -> None:
    """Newton, not Linear: under Linear the DKG / MITC4 shells' sections
    lag the converged strain by one step (#1992)."""
    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


@dataclass
class Truth:
    """The live domain's answer, captured right after the solve."""

    live: np.ndarray                 # (n_gp, n_comp), element source order
    anchor: Callable[[np.ndarray], np.ndarray]   # natural coords -> values


def _shape(topology: str, xi: np.ndarray) -> np.ndarray:
    """Shape functions in the element's node order at natural coords ``xi``."""
    if topology == "quad4":
        s, t = xi
        return 0.25 * np.array([(1 - s) * (1 - t), (1 + s) * (1 - t),
                                (1 + s) * (1 + t), (1 - s) * (1 + t)])
    if topology == "quad8":
        s, t = xi
        corner = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        n = [0.25 * (1 + s * a) * (1 + t * b) * (s * a + t * b - 1)
             for a, b in corner]
        n += [0.5 * (1 - s * s) * (1 - t), 0.5 * (1 + s) * (1 - t * t),
              0.5 * (1 - s * s) * (1 + t), 0.5 * (1 - s) * (1 - t * t)]
        return np.array(n)
    if topology in ("tri3", "tri6"):
        # The reader's frame is gmsh (u, v): node 0 carries 1 - u - v
        # (#2228). The element's own (node-0-first) frame used to sit
        # here, which pinned the rotated coordinates as correct.
        u, v = xi
        l1, l2, l3 = 1.0 - u - v, u, v
        if topology == "tri3":
            return np.array([l1, l2, l3])
        return np.array([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
                         l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3,
                         4 * l3 * l1])
    if topology == "hex8":
        corner = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                  (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        return np.array([0.125 * np.prod([1 + x * c for x, c in zip(xi, cc)])
                         for cc in corner])
    if topology == "tet4":
        return np.array([1.0 - xi.sum(), *xi])
    raise ValueError(f"no shape functions for topology {topology!r}")


def _strain_anchor(topology: str, coords: np.ndarray,
                   disp: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """``xi -> D B(xi) u`` for a linear-elastic isotropic element."""
    ndm = coords.shape[1]
    if ndm == 2:     # plane stress
        D = E / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0],
                                        [0, 0, (1 - NU) / 2]])
    else:
        lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        mu = E / (2 * (1 + NU))
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[:3, :3] += 2 * mu * np.eye(3)
        D[3:, 3:] = mu * np.eye(3)

    def at(xi: np.ndarray) -> np.ndarray:
        h = 1e-3
        dn = np.array([(_shape(topology, xi + h * e) - _shape(topology, xi - h * e))
                       / (2 * h) for e in np.eye(len(xi))])      # (nxi, nn)
        jac = dn @ coords                                        # (ndm, ndm)
        grad = np.linalg.solve(jac, dn)                          # dN/dx (ndm, nn)
        g = grad @ disp                                          # du_j/dx_i
        if ndm == 2:
            eps = np.array([g[0, 0], g[1, 1], g[1, 0] + g[0, 1]])
        else:
            eps = np.array([g[0, 0], g[1, 1], g[2, 2], g[1, 0] + g[0, 1],
                            g[2, 1] + g[1, 2], g[2, 0] + g[0, 2]])
        return D @ eps
    return at


def _membrane_plate_law() -> np.ndarray:
    """ElasticMembranePlateSection: resultants = D e (moments carry -D)."""
    dm = E * SHELL_H / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0],
                                              [0, 0, (1 - NU) / 2]])
    D = np.zeros((8, 8))
    D[:3, :3] = dm
    D[3:6, 3:6] = -dm * SHELL_H**2 / 12
    D[6:, 6:] = 5.0 / 6.0 * E / (2 * (1 + NU)) * SHELL_H * np.eye(2)
    return D


def _by_gp_only(_xi: np.ndarray) -> np.ndarray:
    raise AssertionError("a section-anchored case is checked per GP index")


def capture_truth(ops: Any, case: Case, built: Built) -> Truth:
    """Read the oracles off the live domain (call after the last step)."""
    eid = built.eid
    flat = np.asarray(ops.eleResponse(eid, case.token), dtype=np.float64)
    n = case.n_gp * len(case.comps)
    if case.anchor == "section":
        sec = np.array([ops.eleResponse(eid, "section", g, "forces")
                        for g in range(1, case.n_gp + 1)], dtype=np.float64)
        assert sec.shape == (case.n_gp, 8), case.cpp_class
        assert np.abs(sec).max() > 0.0, f"{case.cpp_class}: section reads zero"
        if case.plane_section:
            dfm = np.array([ops.eleResponse(eid, "section", g, "deformation")
                            for g in range(1, case.n_gp + 1)], dtype=np.float64)
            np.testing.assert_allclose(
                dfm @ _membrane_plate_law().T, sec, rtol=1e-9,
                atol=1e-9 * np.abs(sec).max(),
                err_msg=f"{case.cpp_class}: section order is not "
                        "(membrane, bending, shear)",
            )
        # The element's own vector is the section's, GP by GP.
        assert flat.size == n, case.cpp_class
        np.testing.assert_allclose(flat, sec.ravel(), rtol=1e-12, atol=0,
                                   err_msg=case.cpp_class)
        # The section response is itself the per-GP anchor.
        return Truth(live=sec, anchor=_by_gp_only)
    assert flat.size == n, (
        f"{case.cpp_class}: eleResponse {case.token!r} has {flat.size} "
        f"values, the source layout {n}"
    )
    live = flat.reshape(case.n_gp, len(case.comps))
    if case.anchor == "truss":
        np.testing.assert_allclose(live, [[TRUSS_P]], rtol=1e-10)
        return Truth(live=live, anchor=lambda _xi: np.array([TRUSS_P]))
    disp = np.array([ops.nodeDisp(int(n_))[:built.coords.shape[1]]
                     for n_ in built.node_ids], dtype=np.float64)
    return Truth(live=live,
                 anchor=_strain_anchor(case.topology, built.coords, disp))


def assert_values_match(case: Case, built: Built, truth: Truth,
                        get: Callable[[str], Any], *, where: str) -> None:
    """Every component at every GP, read back, equals the live vector."""
    scale = np.abs(truth.live).max()
    for k, comp in enumerate(case.comps):
        slab = get(comp)
        values = np.asarray(slab.values)
        assert values.shape == (1, case.n_gp), (
            f"{case.cpp_class} {comp} on {where}: slab {values.shape}, "
            f"expected (1, {case.n_gp})"
        )
        assert [int(e) for e in slab.element_index] == [built.eid] * case.n_gp
        np.testing.assert_allclose(
            values[0], truth.live[:, k], rtol=RTOL, atol=RTOL * scale,
            err_msg=f"{case.cpp_class} {comp} on {where}",
        )


def assert_coords_match(case: Case, built: Built, truth: Truth,
                        get: Callable[[str], Any], *, where: str) -> None:
    """The GP natural coordinates read back are where the element sampled.

    Planes and solids: ``D B(xi) u`` at each read-back ``xi`` reproduces the
    live stress of that GP. Trusses: the midpoint. Shells: count and width
    only (no independent anchor for a shell's GP position).
    """
    coords = [np.asarray(get(c).natural_coords, dtype=np.float64)
              for c in case.comps]
    xi_all = coords[0]
    for comp, xi in zip(case.comps, coords):
        np.testing.assert_array_equal(xi, xi_all, err_msg=f"{comp} on {where}")
    if case.anchor == "truss":
        np.testing.assert_array_equal(xi_all, [[0.0]])
        return
    width = 2 if case.anchor == "section" else built.coords.shape[1]
    assert xi_all.shape == (case.n_gp, width), (
        f"{case.cpp_class} on {where}: natural coords {xi_all.shape}, "
        f"expected ({case.n_gp}, {width})"
    )
    if case.anchor == "section":
        return
    scale = np.abs(truth.live).max()
    for g in range(case.n_gp):
        np.testing.assert_allclose(
            truth.anchor(xi_all[g]), truth.live[g], rtol=RTOL_ANCHOR,
            atol=RTOL_ANCHOR * scale,
            err_msg=f"{case.cpp_class} GP {g} at xi={xi_all[g]} on {where}: "
                    "D B(xi) u is not the stress the element reports there",
        )


def assert_catalog_entry(case: Case) -> None:
    """The catalog entry the reader decodes with is the source layout."""
    layout = lookup(case.cpp_class, case.int_rule, case.catalog_token)
    assert layout.n_gauss_points == case.n_gp, case.cpp_class
    assert tuple(layout.component_layout) == case.comps, case.cpp_class


def record_components(case: Case) -> tuple[str, ...]:
    """What the record declares; shells go through the shorthand."""
    if case.comps == SHELL:
        comps = expand_shorthand("shell_resultant", ndm=3, ndf=6)
        assert comps == SHELL
        return comps
    return case.comps


# ---------------------------------------------------------------------------
# The .out route
# ---------------------------------------------------------------------------


def _run_out(case: Case, tmp_path: Path, stub_model_h5: Path) -> tuple[Built, Truth, Any]:
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results import Results
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    from tests.fixtures.recorder_models import identity_model

    ops = get_live_ops()
    built = case.build(ops)
    assert ops.eleType(built.eid) == case.cpp_class
    fem = MinimalFem(built)
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        ResolvedRecorderRecord(
            category="gauss", name="rec", components=record_components(case),
            dt=None, n_steps=None, element_ids=np.array([built.eid]),
            element_class_name=case.cpp_class,
        ),
    ))
    solve(ops)
    out = tmp_path / "out"
    with spec.emit_recorders(
        str(out), ops=ops, model=identity_model([built.eid], fem.snapshot_id),
    ) as rec:
        rec.begin_stage("static", kind="static")
        assert ops.analyze(1) == 0
        truth = capture_truth(ops, case, built)
        rec.end_stage()
    r = Results.from_recorders(
        spec, out, fem=fem, stage_id="static",
        model=OpenSeesModel.from_h5(stub_model_h5),
        cache_root=tmp_path / "cache",
    )
    return built, truth, r


@pytest.mark.parametrize("key", list(CASES))
def test_catalog_layout_reads_back_from_out(
    key: str, tmp_path: Path, stub_model_h5: Path,
) -> None:
    case = CASES[key]
    assert_catalog_entry(case)
    built, truth, r = _run_out(case, tmp_path, stub_model_h5)
    def get(c: str) -> Any:
        return r.elements.gauss.get(component=c)

    assert_values_match(case, built, truth, get, where=".out")
    assert_coords_match(case, built, truth, get, where=".out")
