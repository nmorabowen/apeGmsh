"""Write the line-diagram oracle fixtures: OpenSees-solved ``model.h5`` + ``.ladruno`` + golden JSON (#2471).

    $env:PYTHONPATH = "<worktree>\\src"
    $env:LADRUNO_OPENSEES_QUIET = "1"
    python tests/fixtures/ladruno/make_line_diagram_oracle.py            # every fixture
    python tests/fixtures/ladruno/make_line_diagram_oracle.py --check    # re-solve, compare to the committed JSON

run from the repository root with a Ladruno fork build of OpenSees behind
``openseespy`` (``import opensees`` works) and apeGmsh importable from
``src`` (the editable install may point at another checkout: check the
printed path). It lives here, not under ``apeGmshViewer/``, because it builds
the models with apeGmsh and the app never imports apeGmsh (ADR 0112 D4;
``scripts/check_app_wall.py`` rule W1). It writes, into
``apeGmshViewer/fixtures/``, for every fixture stem ``line_oracle_<name>``:

* ``<stem>.h5``: the model.h5 apeSees wrote for the model it solved;
* ``<stem>.ladruno``: what the fork's ``recorder ladruno`` wrote while
  solving it (two ``LoadControl 0.5`` steps, ``TIME`` = 0.5 and 1.0);
* ``<stem>.golden.json``: the expected values, read back from the
  ``.ladruno`` with h5py.

Where every golden number comes from, and what it is checked against:

* **OpenSees output.** Each recorded bucket's ``DATA`` is copied from the
  ``.ladruno`` (``raw``), and equals the live domain's ``eleResponse`` at
  every step to ``rtol=1e-12`` (the run asserts it). Station positions are
  OpenSees' own ``sectionLocation`` for ``section.force`` (checked against
  the textbook Legendre / Lobatto abscissae) and ``xi = -1, +1`` for end
  forces. The file's ``QUADRATURE/GP_PARAM`` is recorded beside them
  (``gp_param_by_gauss_id``), never used: for a Legendre rule the fork
  writes it wrong (``gp_param_defect`` in that JSON says how).
* **Statics.** Every station value is also computed in closed form from a
  free body of the (statically determinate) structure: the resultant the
  part on the node-j side of the cut exerts on the node-i side, in the
  element's local frame. The run asserts OpenSees and statics agree to
  ``rtol=1e-9`` (``ATOL`` near a zero). A disagreement is a finding: the run
  stops and names it, it is never tuned away.
* **Frames.** Each element's local axes are computed from its end
  coordinates and ``vecxz`` (OpenSees ``LinearCrdTransf``), and checked
  against the file's ``LOCAL_AXES`` quaternion.

The sign convention is the force-beam ``section force`` one every route
shares (#2009), recorded in each JSON under ``sign_convention``: station 1
(``xi = -1``) is ``s * localForce(end i)`` and station 2 (``xi = +1``) is
``-s * localForce(end j)``, with ``s`` from
``_response_catalog._LOCAL_FORCE_END_I_SIGN``. Physically: ``N``, ``Vz``,
``T``, ``My``, ``Mz`` are the resultants on the section's +x face (toward
node j), and ``Vy`` is their ``dMz/dx`` (the negative of the +x-face
y-resultant).

The orientation probe (#2471): apeGmsh never emits a beam whose ``iNode`` is
not its line cell's first corner (``apesees.py`` sets each element's nodes
from the FEM connectivity row unchanged), so no fixture has i/j reversed
against its cell. A gmsh line cell follows its curve's direction, so
``line_oracle_ss_udl`` draws one half of the span against +x: those cells,
and the elements, run from right to left.

``--check`` re-solves every fixture into a temporary directory and compares
its golden JSON to the committed one (the generator is deterministic).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gmsh
import h5py
import numpy as np

import apeGmsh as apegmsh_pkg
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._response_catalog import _LOCAL_FORCE_END_I_SIGN
from apeGmsh.opensees.emitter.live import LiveOpsEmitter, get_backend_build
from apeGmsh.results.writers import stamp_provenance

FIXTURES = Path(__file__).resolve().parents[3] / "apeGmshViewer" / "fixtures"
GENERATOR = "tests/fixtures/ladruno/make_line_diagram_oracle.py"
STEPS = 2
DLAM = 0.5
RTOL_LIVE = 1e-12
RTOL = 1e-9
ATOL = 1e-9
#: The fork's component spellings, by bucket, to the canonical line-station names.
CANON = {
    "N": "axial_force", "P": "axial_force",
    "V": "shear_y", "Vy": "shear_y", "Vz": "shear_z",
    "T": "torsion", "My": "bending_moment_y",
    "Mz": "bending_moment_z", "M": "bending_moment_z",
}
SECTION_2D = {"E": 2.0e8, "A": 0.01, "Iz": 1.0e-4}
SECTION_3D = {"E": 2.0e8, "A": 0.01, "Iz": 1.0e-4, "Iy": 2.0e-4, "G": 8.0e7, "J": 1.5e-4}


# ---------------------------------------------------------------------------
# Statics: a free body of a straight, statically determinate line structure
# ---------------------------------------------------------------------------

@dataclass
class PointLoad:
    """A force and moment (global, at lambda = 1) at a point: a nodal load or a reaction."""

    at: tuple[float, float, float]
    force: tuple[float, float, float]
    moment: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class LineLoad:
    """A uniform load per length (global vector, at lambda = 1) on the straight segment a-b."""

    a: tuple[float, float, float]
    b: tuple[float, float, float]
    q: tuple[float, float, float]


@dataclass
class Statics:
    points: list[PointLoad] = field(default_factory=list)
    lines: list[LineLoad] = field(default_factory=list)

    def cut(self, p: np.ndarray, ex: np.ndarray, at_j: bool) -> tuple[np.ndarray, np.ndarray]:
        """Force and moment (about ``p``) the node-j side exerts on the node-i side.

        The node-j side is every point ``r`` with ``(r - p) . ex > 0``; a
        point load at ``p`` itself is on it only when the cut is at the
        element's node-j end (``at_j``), never at node i.
        """
        f = np.zeros(3)
        m = np.zeros(3)
        tol = 1e-12
        for pl in self.points:
            r = np.asarray(pl.at, dtype=float)
            t = float((r - p) @ ex)
            if t > tol or (at_j and abs(t) <= tol):
                fk = np.asarray(pl.force, dtype=float)
                f += fk
                m += np.cross(r - p, fk) + np.asarray(pl.moment, dtype=float)
        for ll in self.lines:
            a = np.asarray(ll.a, dtype=float)
            b = np.asarray(ll.b, dtype=float)
            ta, tb = float((a - p) @ ex), float((b - p) @ ex)
            if ta == tb:
                raise ValueError("a line load perpendicular to the cut direction")
            # The part of a-b on the node-j side: parameters u in [0, 1] with t(u) > 0.
            u0 = -ta / (tb - ta)
            lo, hi = (max(0.0, u0), 1.0) if tb > ta else (0.0, min(1.0, u0))
            if hi <= lo:
                continue
            ra, rb = a + lo * (b - a), a + hi * (b - a)
            fk = np.asarray(ll.q, dtype=float) * float(np.linalg.norm(rb - ra))
            f += fk
            m += np.cross(0.5 * (ra + rb) - p, fk)
        return f, m


def section_resultants(f: np.ndarray, m: np.ndarray, frame: np.ndarray) -> dict[str, float]:
    """The six section resultants in the #2009 convention from a cut's force and moment."""
    ex, ey, ez = frame
    return {
        "axial_force": float(f @ ex),
        "shear_y": -float(f @ ey),        # Vy = dMz/dx: minus the +x-face y-resultant
        "shear_z": float(f @ ez),
        "torsion": float(m @ ex),
        "bending_moment_y": float(m @ ey),
        "bending_moment_z": float(m @ ez),
    }


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def linear_frame(xi: np.ndarray, xj: np.ndarray, ndm: int, vecxz: tuple[float, ...] | None) -> np.ndarray:
    """Rows ``(ex, ey, ez)`` as OpenSees ``LinearCrdTransf2d/3d`` builds them."""
    ex = (xj - xi) / np.linalg.norm(xj - xi)
    if ndm == 2:
        ez = np.array([0.0, 0.0, 1.0])
        ey = np.cross(ez, ex)
        return np.stack([ex, ey, ez])
    assert vecxz is not None
    ey = np.cross(np.asarray(vecxz, dtype=float), ex)
    n = float(np.linalg.norm(ey))
    if n < 1e-12:
        raise ValueError(f"vecxz {vecxz} is parallel to the element axis {ex}")
    ey /= n
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """A unit quaternion ``(w, x, y, z)`` as a rotation matrix whose columns are the rotated axes."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Fixture cases
# ---------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    title: str
    ndm: int
    build: Callable[[Case], Any]                   # () -> FEMData
    model: Callable[[apeSees], None]               # elements, fixes, pattern
    responses: tuple[str, ...]
    statics: Statics
    vecxz: tuple[float, ...] | None = None
    closed_form_notes: list[str] = field(default_factory=list)


SESSION_DIR = Path(tempfile.mkdtemp(prefix="line_oracle_session_"))
atexit.register(shutil.rmtree, SESSION_DIR, ignore_errors=True)


def _mesh(name: str, points: dict[str, tuple[float, float, float]],
          curves: list[tuple[str, str, str, int]]) -> Any:
    """A 1-D gmsh model: named points, curves ``(pg, from, to, n_elements)``, a label per point."""
    with apeGmsh(model_name=name, verbose=False, units="kN_m_s",
                 save_to=SESSION_DIR / f"{name}.h5") as g:
        geo = g.model.geometry
        tags = {k: geo.add_point(*xyz) for k, xyz in points.items()}
        by_pg: dict[str, list[int]] = {}
        for pg, a, b, n in curves:
            c = geo.add_line(tags[a], tags[b])
            gmsh.model.mesh.setTransfiniteCurve(c, n + 1)
            by_pg.setdefault(pg, []).append(c)
        for pg, cs in by_pg.items():
            g.physical.add(1, cs, name=pg)
        for k, t in tags.items():
            g.labels.add(0, [t], k)
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


# 1. Simply supported beam, uniform load, elastic (2-D). -------------------
SS_L, SS_W = 6.0, 10.0


def _ss_build(case: Case) -> Any:
    # The right half is drawn from b to m: its cells (and elements) run against +x.
    return _mesh("line_oracle_ss_udl",
                 {"a": (0.0, 0.0, 0.0), "m": (SS_L / 2, 0.0, 0.0), "b": (SS_L, 0.0, 0.0)},
                 [("Fwd", "a", "m", 2), ("Rev", "b", "m", 2)])


def _ss_model(ops: apeSees) -> None:
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Fwd", transf=tr, **SECTION_2D)
    ops.element.elasticBeamColumn(pg="Rev", transf=tr, **SECTION_2D)
    ops.fix(label="a", dofs=(1, 1, 0))
    ops.fix(label="b", dofs=(0, 1, 0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        # Gravity is -Y. Local y is +Y on Fwd and -Y on Rev (2-D: y = z x x).
        p.beam_uniform(pg="Fwd", wy=-SS_W)
        p.beam_uniform(pg="Rev", wy=+SS_W)


SS_CASE = Case(
    name="ss_udl",
    title="Simply supported beam, L = 6, uniform load w = 10 down, elasticBeamColumn (2-D). "
          "Left half a->m along +x (Fwd), right half drawn b->m against +x (Rev).",
    ndm=2, build=_ss_build, model=_ss_model, responses=("localForce",),
    statics=Statics(
        points=[PointLoad((0.0, 0.0, 0.0), (0.0, SS_W * SS_L / 2, 0.0)),
                PointLoad((SS_L, 0.0, 0.0), (0.0, SS_W * SS_L / 2, 0.0))],
        lines=[LineLoad((0.0, 0.0, 0.0), (SS_L, 0.0, 0.0), (0.0, -SS_W, 0.0))],
    ),
    closed_form_notes=[
        "Reactions wL/2 up at a and b; sagging moment M(x) = w x (L - x) / 2, M(L/2) = wL^2/8 = 45.",
        ("Fwd (local y = +Y): Mz = +M(x). Rev (local y = -Y, local x = -X): Mz = -M(x). "
         "Both put tension at the bottom: Mz > 0 is tension on the -y side."),
    ],
)


# 2. 3-D cantilever, tip loads in local y and z plus a torque. -------------
CT_L = 4.0
CT_TIP = (1.0, 2.0, 3.0, 0.4, 0.0, 0.0)


def _ct_build(case: Case) -> Any:
    return _mesh("line_oracle_cantilever3d",
                 {"a": (0.0, 0.0, 0.0), "m": (CT_L / 2, 0.0, 0.0), "b": (CT_L, 0.0, 0.0)},
                 [("Beam", "a", "m", 1), ("Beam", "m", "b", 1)])


def _ct_model(ops: apeSees) -> None:
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(pg="Beam", transf=tr, **SECTION_3D)
    ops.fix(label="a", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="b"), forces=CT_TIP)


CT_CASE = Case(
    name="cantilever3d",
    title="3-D cantilever along +x, L = 4, fixed at a, tip load (Fx, Fy, Fz, Mx) = (1, 2, 3, 0.4) at b, "
          "elasticBeamColumn, vecxz = (0, 0, 1) so local y = +Y, z = +Z.",
    ndm=3, build=_ct_build, model=_ct_model, responses=("localForce",),
    vecxz=(0.0, 0.0, 1.0),
    statics=Statics(points=[PointLoad((CT_L, 0.0, 0.0), CT_TIP[:3], CT_TIP[3:])]),
    closed_form_notes=[
        "Section at x: N = Fx, Vy = -Fy, Vz = +Fz, T = Mx, My = -Fz (L - x), Mz = +Fy (L - x).",
        "Fy bends in the x-y plane (Mz); Fz bends in the x-z plane (My).",
    ],
)


# 3. Force-based beam, Legendre and Lobatto (2-D cantilever). --------------
FB_L = 3.0
FB_TIP = (1.0, -5.0, 0.0)
FB_W = -2.0


def _fb_build(case: Case) -> Any:
    return _mesh(f"line_oracle_{case.name}",
                 {"a": (0.0, 0.0, 0.0), "b": (FB_L, 0.0, 0.0)}, [("Beam", "a", "b", 1)])


def _fb_model(rule: str, n_ip: int) -> Callable[[apeSees], None]:
    def model(ops: apeSees) -> None:
        ops.model(ndm=2, ndf=3)
        tr = ops.geomTransf.Linear()
        sec = ops.section.Elastic(**SECTION_2D)
        integ = getattr(ops.beamIntegration, rule)(section=sec, n_ip=n_ip)
        ops.element.forceBeamColumn(pg="Beam", transf=tr, integration=integ)
        ops.fix(label="a", dofs=(1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=ops.nodes.one(label="b"), forces=FB_TIP)
            p.beam_uniform(pg="Beam", wy=FB_W)
    return model


def _fb_case(rule: str, n_ip: int) -> Case:
    return Case(
        name=f"fbc_{rule.lower()}",
        title=f"2-D cantilever along +x, L = 3, fixed at a, tip load (Fx, Fy) = (1, -5) at b and "
              f"uniform wy = -2, one forceBeamColumn, {rule} with {n_ip} points, elastic section.",
        ndm=2, build=_fb_build, model=_fb_model(rule, n_ip),
        responses=("section.force", "localForce"),
        statics=Statics(
            points=[PointLoad((FB_L, 0.0, 0.0), (FB_TIP[0], FB_TIP[1], 0.0), (0.0, 0.0, FB_TIP[2]))],
            lines=[LineLoad((0.0, 0.0, 0.0), (FB_L, 0.0, 0.0), (0.0, FB_W, 0.0))],
        ),
        closed_form_notes=[
            "Section at x: N = Fx, Mz = Fy (L - x) + w (L - x)^2 / 2, Vy = dMz/dx = -Fy - w (L - x).",
            (f"Stations at the {rule} points: xi from the file's QUADRATURE/GP_PARAM, "
             "checked against the textbook abscissae."),
        ],
    )


# 4. Vertical 3-D column with an explicit vecxz. ---------------------------
CO_H = 3.0
CO_TIP = (2.0, 3.0, -10.0, 0.0, 0.0, 0.5)
CO_VECXZ = (1.0, 0.0, 0.0)
CO_DEGENERATE_VECXZ = (0.0, 0.0, 1.0)


def _co_build(case: Case) -> Any:
    return _mesh("line_oracle_column3d",
                 {"a": (0.0, 0.0, 0.0), "m": (0.0, 0.0, CO_H / 2), "b": (0.0, 0.0, CO_H)},
                 [("Col", "a", "m", 1), ("Col", "m", "b", 1)])


def _co_model(vecxz: tuple[float, float, float]) -> Callable[[apeSees], None]:
    def model(ops: apeSees) -> None:
        ops.model(ndm=3, ndf=6)
        tr = ops.geomTransf.Linear(vecxz=vecxz)
        ops.element.elasticBeamColumn(pg="Col", transf=tr, **SECTION_3D)
        ops.fix(label="a", dofs=(1, 1, 1, 1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=ops.nodes.one(label="b"), forces=CO_TIP)
    return model


CO_CASE = Case(
    name="column3d",
    title="Vertical 3-D column, base a at the origin fixed, top b at z = 3, top load "
          "(FX, FY, FZ, MZ) = (2, 3, -10, 0.5), elasticBeamColumn, vecxz = (1, 0, 0): "
          "local x = +Z, y = -Y, z = +X.",
    ndm=3, build=_co_build, model=_co_model(CO_VECXZ), responses=("localForce",),
    vecxz=CO_VECXZ,
    statics=Statics(points=[PointLoad((0.0, 0.0, CO_H), CO_TIP[:3], CO_TIP[3:])]),
    closed_form_notes=[
        "Local tip load (fx, fy, fz) = (FZ, -FY, FX) = (-10, -3, 2), local torque mx = MZ = 0.5.",
        "Section at s from the base: N = fx, Vy = -fy, Vz = fz, T = mx, My = -fz (H - s), Mz = fy (H - s).",
    ],
)


# 5. basicForce-only element (3-D cantilever). -----------------------------
BO_TIP = (1.0, 2.0, 3.0, 0.4, 0.5, 0.6)


def _bo_build(case: Case) -> Any:
    return _mesh("line_oracle_basic_only",
                 {"a": (0.0, 0.0, 0.0), "m": (CT_L / 2, 0.0, 0.0), "b": (CT_L, 0.0, 0.0)},
                 [("Beam", "a", "m", 1), ("Beam", "m", "b", 1)])


def _bo_model(ops: apeSees) -> None:
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(pg="Beam", transf=tr, **SECTION_3D)
    ops.fix(label="a", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="b"), forces=BO_TIP)


BO_CASE = Case(
    name="basic_only",
    title="3-D cantilever along +x, L = 4, tip load (1, 2, 3, 0.4, 0.5, 0.6), elasticBeamColumn, "
          "vecxz = (0, 0, 1), recording basicForce only (N, Mz_1, Mz_2, My_1, My_2, T).",
    ndm=3, build=_bo_build, model=_bo_model, responses=("basicForce",),
    vecxz=(0.0, 0.0, 1.0),
    statics=Statics(points=[PointLoad((CT_L, 0.0, 0.0), BO_TIP[:3], BO_TIP[3:])]),
    closed_form_notes=[
        "Section at x: N = Fx, T = Mx, My = My_tip - Fz (L - x), Mz = Mz_tip + Fy (L - x).",
        ("basicForce has no shear. Its end moments read as stations [-M_1, +M_2] at xi = -1, +1 "
         "(#2011); N and T are one station at xi = 0."),
    ],
)

CASES = [SS_CASE, CT_CASE, _fb_case("Legendre", 3), _fb_case("Lobatto", 4), CO_CASE, BO_CASE]


# ---------------------------------------------------------------------------
# Solve, read back, check, write
# ---------------------------------------------------------------------------

def _text(attr: Any) -> str:
    v = np.asarray(attr).reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def _fem_to_ops(model_h5: Path) -> dict[int, int]:
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {int(e): int(t) for tok in meta
                for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])}


def _solve(case: Case, out: Path) -> dict[str, Any]:
    """Build, emit, solve ``case`` into ``out``; return the live responses per step."""
    stem = f"line_oracle_{case.name}"
    model_h5, ladruno = out / f"{stem}.h5", out / f"{stem}.ladruno"
    for p in (model_h5, ladruno):
        p.unlink(missing_ok=True)
    fem = case.build(case)
    ops = apeSees(fem, default_orientation=None)
    case.model(ops)
    ops.recorder.Ladruno(
        file=str(ladruno),
        nodal_responses=("displacement",),
        elem_responses=case.responses,
    )
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    # Newton, not Linear: Linear takes one step without iterating, and a
    # forceBeamColumn carrying a beamUniform load is left out of equilibrium
    # by it (a free tip reported a 0.75 end moment, proportional to dlam).
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=DLAM)
    ops.analysis.Static()
    ops.h5(str(model_h5))
    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)
    fem_to_ops = _fem_to_ops(model_h5)
    o = em.ops
    live: dict[str, dict[int, list[np.ndarray]]] = {r: {e: [] for e in fem_to_ops} for r in case.responses}
    n_ip: dict[int, int] = {}
    for _ in range(STEPS):
        assert em.analyze(steps=1) == 0, f"{stem}: analyze failed"
        for e, t in fem_to_ops.items():
            for r in case.responses:
                if r == "section.force":
                    if e not in n_ip:
                        n_ip[e] = len(o.sectionLocation(t))
                    live[r][e].append(np.concatenate(
                        [np.asarray(o.eleResponse(t, "section", ip, "force")) for ip in range(1, n_ip[e] + 1)]))
                else:
                    live[r][e].append(np.asarray(o.eleResponse(t, r)))
    nodes = {int(n): [int(c) for c in o.eleNodes(t)] for n, t in fem_to_ops.items()}
    sec_loc = {e: [float(x) for x in o.sectionLocation(fem_to_ops[e])] for e in n_ip}
    build = get_backend_build() or "unknown"
    o.remove("recorders")
    o.wipe()
    # Pair the closed file with its model.h5 the way the bridge does (ADR 0122 D3).
    stamp_provenance(ladruno, model_h5, opensees_build_sha=build)
    return {"model_h5": model_h5, "ladruno": ladruno, "fem_to_ops": fem_to_ops,
            "live": live, "ele_nodes": nodes, "section_location": sec_loc, "build": build}


def _read_bucket(f: h5py.File, response: str) -> list[dict[str, Any]]:
    """Every block of ``response``: ids, DATA, TIME, component names per GAUSS row, GP_PARAM."""
    st = f["MODEL_STAGE[1]"]
    blocks = []
    for bname, blk in st[f"RESULTS/ON_ELEMENTS/{response}"].items():
        model_block = bname.rsplit(":", 1)[0] + "]"
        quad = st["MODEL/ELEMENTS"].get(f"{model_block}/QUADRATURE")
        elem_attrs = st["MODEL/ELEMENTS"][model_block].attrs
        blocks.append({
            "block": bname,
            "ids": blk["ID"][()].reshape(-1).astype(int).tolist(),
            "data": blk["DATA"][()],
            "time": blk["TIME"][()].tolist(),
            "step": blk["STEP"][()].astype(int).tolist(),
            "comp_rows": [row.split(",") for row in _text(blk["COLUMN_MAP"].attrs["COMP_NAMES"]).split("\n")],
            "gauss_id": blk["COLUMN_MAP/GAUSS_ID"][()].reshape(-1).astype(int).tolist(),
            "gp_param": None if quad is None else quad["GP_PARAM"][()].reshape(-1).tolist(),
            "param_domain": _text(elem_attrs["PARAM_DOMAIN"]) if "PARAM_DOMAIN" in elem_attrs else None,
        })
    return blocks


def _local_axes(f: h5py.File) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    la = f["MODEL_STAGE[1]/MODEL"].get("LOCAL_AXES")
    if la is None:
        return out
    for grp in la.values():
        for i, q in zip(grp["ID"][()].reshape(-1), grp["FRAME"][()]):
            out[int(i)] = [float(v) for v in q]
    return out


def _close(a: float, b: float, rtol: float = RTOL, atol: float = ATOL) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def _textbook_xi(rule: str, n: int) -> list[float]:
    if rule == "fbc_legendre":
        return [float(x) for x in np.polynomial.legendre.leggauss(n)[0]]
    if rule == "fbc_lobatto":
        # Gauss-Lobatto: the ends plus the roots of P'_{n-1}.
        inner = np.polynomial.legendre.Legendre.basis(n - 1).deriv().roots()
        return [-1.0, *sorted(float(np.real(x)) for x in inner), 1.0]
    raise KeyError(rule)


def make(case: Case, out: Path) -> dict[str, Any]:
    stem = f"line_oracle_{case.name}"
    sol = _solve(case, out)
    findings: list[str] = []
    with h5py.File(sol["model_h5"], "r") as f:
        coords = {int(n): np.asarray(c, dtype=float) for n, c in zip(f["nodes/ids"][()], f["nodes/coords"][()])}
        cells = {int(e): [int(c) for c in conn]
                 for grp in f["elements"].values()
                 for e, conn in zip(grp["ids"][()], grp["connectivity"][()])}
    with h5py.File(sol["ladruno"], "r") as f:
        buckets = {r: _read_bucket(f, r) for r in case.responses}
        quats = _local_axes(f)
        prov = {k: _text(v) for k, v in f["INFO/PROVENANCE"].attrs.items()} if "INFO/PROVENANCE" in f else {}

    elements = []
    for e in sorted(cells):
        t = sol["fem_to_ops"][e]
        i_node, j_node = cells[e]
        # Probe, live: OpenSees' (iNode, jNode) is the cell's first and second corner.
        assert sol["ele_nodes"][e] == cells[e], (stem, e, sol["ele_nodes"][e], cells[e])
        xi_c, xj_c = coords[i_node], coords[j_node]
        length = float(np.linalg.norm(xj_c - xi_c))
        frame = linear_frame(xi_c, xj_c, case.ndm, case.vecxz)
        el: dict[str, Any] = {
            "fem_id": e, "ops_tag": t, "nodes": [i_node, j_node],
            "coords_i": xi_c.tolist(), "coords_j": xj_c.tolist(), "length": length,
            "frame": {"x": frame[0].tolist(), "y": frame[1].tolist(), "z": frame[2].tolist()},
            "buckets": {},
        }
        if t in quats:
            r = quat_to_matrix(np.asarray(quats[t]))
            el["frame_quaternion_wxyz"] = quats[t]
            if np.allclose(r, frame.T, atol=1e-12):
                el["frame_quaternion_columns"] = "local x, y, z axes (R columns, global components)"
            elif np.allclose(r, frame, atol=1e-12):
                el["frame_quaternion_columns"] = "transposed: R rows are the local axes"
            else:
                findings.append(f"element {e}: LOCAL_AXES quaternion {quats[t]} does not match "
                                f"the LinearCrdTransf frame {frame.tolist()}")

        def closed(comp: str, xi: float,
                   _xi_c: np.ndarray = xi_c, _xj_c: np.ndarray = xj_c,
                   _frame: np.ndarray = frame) -> float:
            p = _xi_c + 0.5 * (xi + 1.0) * (_xj_c - _xi_c)
            fcut, mcut = case.statics.cut(p, _frame[0], at_j=xi > 0.999999)
            return section_resultants(fcut, mcut, _frame)[comp]

        for response in case.responses:
            blk = next(b for b in buckets[response] if t in b["ids"])
            row = blk["ids"].index(t)
            raw = blk["data"][:, row, :]                       # (T, ncols)
            live = np.stack(sol["live"][response][e])
            if live.shape != raw.shape or not np.allclose(raw, live, rtol=RTOL_LIVE, atol=1e-12):
                findings.append(f"element {e} {response}: file DATA != live eleResponse\n{raw}\n{live}")
            columns = [c for r_ in blk["comp_rows"] for c in r_]
            entry: dict[str, Any] = {
                "block": blk["block"], "columns": columns, "raw": raw.tolist(),
                "time": blk["time"], "step": blk["step"], "stations": {},
            }
            stations: dict[str, dict[str, Any]] = {}
            if response == "localForce":
                names = [c.rsplit("_", 1)[0] for c in columns]
                half = len(names) // 2
                for k in range(half):
                    comp = CANON[names[k]]
                    s = _LOCAL_FORCE_END_I_SIGN[comp]
                    stations[comp] = {
                        "xi": [-1.0, 1.0],
                        "values": [[s * float(v[k]), -s * float(v[half + k])] for v in raw],
                        "from": f"[{s:+g} * {columns[k]}, {-s:+g} * {columns[half + k]}]",
                    }
            elif response == "section.force":
                gp = blk["gp_param"]
                entry["param_domain"] = blk["param_domain"]
                if blk["param_domain"] != "[-1,1]":
                    findings.append(f"element {e}: PARAM_DOMAIN {blk['param_domain']!r}, expected '[-1,1]'")
                # The stations' true positions: OpenSees' own sectionLocation
                # (distance from node i), checked against the textbook abscissae.
                at = np.asarray(sol["section_location"][e])
                xi_st = [float(2.0 * x / length - 1.0) for x in at]
                entry["section_location"] = at.tolist()
                ref = _textbook_xi(case.name, len(xi_st))
                if not np.allclose(xi_st, ref, atol=1e-12):
                    findings.append(f"element {e}: sectionLocation xi {xi_st} != textbook {case.name} {ref}")
                # What the file says, per GAUSS_ID row. A mismatch is a fork
                # finding: it is recorded beside the true xi, never used.
                file_xi = [float(gp[g]) for g in blk["gauss_id"]]
                entry["gp_param_by_gauss_id"] = file_xi
                entry["gp_param_matches_true_xi"] = bool(np.allclose(file_xi, xi_st, atol=1e-12))
                if not entry["gp_param_matches_true_xi"]:
                    entry["gp_param_defect"] = (
                        "The fork's QUADRATURE/GP_PARAM is not where OpenSees put the sections: "
                        "LadrunoElementResults getCustomGaussPointLocations min/max-normalizes the "
                        "'integrationPoints' positions, 2 (x - x_min) / (x_max - x_min) - 1, instead of "
                        "2 x / L - 1, which is exact only for rules with both end points (Lobatto). "
                        "The station values are at the true xi (stations.xi); a reader that takes xi "
                        "from GP_PARAM draws them at gp_param_by_gauss_id.")
                    print(f"FINDING {stem} element {e}: GP_PARAM {file_xi} != true xi {xi_st}")
                ncomp = len(blk["comp_rows"][0])
                for k, name in enumerate(blk["comp_rows"][0]):
                    comp = CANON[name]
                    stations[comp] = {
                        "xi": xi_st,
                        "values": [[float(v[g * ncomp + k]) for g in range(len(xi_st))] for v in raw],
                        "from": f"{name} at each GAUSS_ID",
                    }
            elif response == "basicForce":
                idx = {c: i for i, c in enumerate(columns)}
                for name, comp in (("Mz", "bending_moment_z"), ("My", "bending_moment_y")):
                    i1, i2 = idx[f"{name}_1"], idx[f"{name}_2"]
                    stations[comp] = {"xi": [-1.0, 1.0],
                                      "values": [[-float(v[i1]), float(v[i2])] for v in raw],
                                      "from": f"[-{name}_1, +{name}_2]"}
                for name, comp in (("N", "axial_force"), ("T", "torsion")):
                    stations[comp] = {"xi": [0.0], "values": [[float(v[idx[name]])] for v in raw],
                                      "from": name}
            # Statics, at lambda = TIME.
            for comp, st_ in stations.items():
                cf = [[lam * closed(comp, x) for x in st_["xi"]] for lam in blk["time"]]
                st_["closed_form"] = cf
                for k_t, (got_row, want_row) in enumerate(zip(st_["values"], cf)):
                    for x, got, want in zip(st_["xi"], got_row, want_row):
                        if not _close(got, want):
                            findings.append(f"element {e} {response} {comp} xi={x} step {k_t}: "
                                            f"OpenSees {got!r} != statics {want!r}")
            entry["stations"] = stations
            el["buckets"][response] = entry
        elements.append(el)

    sign = {
        "convention": "force-beam section force (#2009): station 1 (xi=-1) = s * localForce(end i), "
                      "station 2 (xi=+1) = -s * localForce(end j)",
        "end_i_sign": dict(sorted(_LOCAL_FORCE_END_I_SIGN.items())),
        "end_i_sign_source": "src/apeGmsh/opensees/_response_catalog.py _LOCAL_FORCE_END_I_SIGN",
        "physical": "N, Vz, T, My, Mz: resultants on the section's +x face (toward node j); "
                    "Vy = dMz/dx (minus the +x-face y-resultant).",
        "basicForce": "end moments read as stations [-M_1, +M_2] (#2011); N and T one station at xi = 0",
        "moment_tension_side": {"bending_moment_z": "Mz > 0 puts tension on the local -y side",
                                "bending_moment_y": "My > 0 puts tension on the local +z side"},
        "xi": "natural coordinate along the element from node i (-1) to node j (+1); "
              "node i is the line cell's first corner",
    }
    golden = {
        "fixture": stem,
        "title": case.title,
        "model_h5": f"{stem}.h5",
        "ladruno": f"{stem}.ladruno",
        "generator": GENERATOR,
        "ndm": case.ndm,
        "analysis": f"static, LoadControl {DLAM} x {STEPS} steps, Newton to NormDispIncr 1e-12; lambda = TIME",
        "responses": list(case.responses),
        "vecxz": None if case.vecxz is None else list(case.vecxz),
        "sign_convention": sign,
        "closed_form": case.closed_form_notes,
        "elements": elements,
        "ladruno_provenance_keys": sorted(prov),
        "opensees_build_sha": sol["build"],
    }
    if findings:
        raise SystemExit(f"{stem}: OpenSees and the closed form disagree (a finding, do not tune):\n  "
                         + "\n  ".join(findings))
    return golden


def degenerate_vecxz_probe(out: Path) -> dict[str, Any]:
    """Does OpenSees accept a vecxz parallel to a vertical column? Record what happens.

    It runs in a child process: OpenSees' refusal (``LinearCrdTransf3d``,
    at element ``setDomain``) ends the process it runs in.
    """
    child = subprocess.run(
        [sys.executable, __file__, "--degenerate-child", str(out)],
        capture_output=True, text=True, check=False,
        env={**os.environ, "LADRUNO_OPENSEES_QUIET": "1"},
    )
    said = [ln.strip() for ln in (child.stdout + child.stderr).splitlines()
            if "LinearCrdTransf3d" in ln or "parallel" in ln or "setDomain" in ln or "SOLVED" in ln]
    accepted = child.returncode == 0 and "SOLVED" in child.stdout
    if accepted:
        raise SystemExit("OpenSees now accepts a vecxz parallel to the column: add the degenerate "
                         "variant as a fixture (#2471 item 4) instead of documenting the refusal.")
    return {
        "vecxz": list(CO_DEGENERATE_VECXZ),
        "apesees_emits_it": True,
        "opensees_accepts_it": False,
        "opensees_said": [ln for ln in said if "SOLVED" not in ln],
        "note": "No fixture: OpenSees refuses the element and ends the process, so there is no "
                "solved file to pin. A viewer meeting such a transform must refuse the beam "
                "itself (plan P0-4).",
    }


def _degenerate_child(out: Path) -> None:
    case = Case(name="column3d_degenerate", title="", ndm=3, build=_co_build,
                model=_co_model(CO_DEGENERATE_VECXZ), responses=("localForce",),
                vecxz=CO_DEGENERATE_VECXZ, statics=Statics())
    _solve(case, out)
    print("SOLVED")


def _dump(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=1, sort_keys=False) + "\n"


def _case_child(case_name: str, out: Path, tmp: Path) -> None:
    case = next(c for c in CASES if c.name == case_name)
    golden = make(case, out)
    if case.name == "column3d":
        golden["degenerate_vecxz"] = degenerate_vecxz_probe(tmp)
    (tmp / f"line_oracle_{case.name}.golden.json").write_text(_dump(golden))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="re-solve into a temporary directory and compare to the committed golden JSON")
    ap.add_argument("--only", nargs="*", help="fixture names to (re)write")
    ap.add_argument("--degenerate-child", help=argparse.SUPPRESS)
    ap.add_argument("--case-child", nargs=3, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.degenerate_child:
        _degenerate_child(Path(args.degenerate_child))
        return
    if args.case_child:
        _case_child(args.case_child[0], Path(args.case_child[1]), Path(args.case_child[2]))
        return
    print("apeGmsh from", apegmsh_pkg.__file__)
    with tempfile.TemporaryDirectory(prefix="line_oracle_") as tmp_str:
        tmp = Path(tmp_str)
        out = tmp if args.check else FIXTURES
        bad = []
        for case in CASES:
            if args.only and case.name not in args.only:
                continue
            subprocess.run(
                [sys.executable, __file__, "--case-child", case.name, str(out), str(tmp)],
                check=True,
                env={**os.environ, "LADRUNO_OPENSEES_QUIET": "1"},
            )
            path = FIXTURES / f"line_oracle_{case.name}.golden.json"
            text = (tmp / path.name).read_text()
            if args.check:
                same = path.exists() and path.read_text() == text
                print(("same     " if same else "DIFFERENT"), path.name)
                bad += [] if same else [path.name]
            else:
                path.write_text(text)
                for p in (path, out / f"line_oracle_{case.name}.h5", out / f"line_oracle_{case.name}.ladruno"):
                    print(f"wrote {p.name} ({p.stat().st_size} bytes)")
        if bad:
            sys.exit(f"golden JSON not reproduced: {bad}")


if __name__ == "__main__":
    main()
