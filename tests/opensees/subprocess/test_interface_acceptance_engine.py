"""ADR 0093 S10 — the acceptance battery, engine half.

The checks no amount of deck reading can answer: they need a solver.
Each one is the permanent form of a probe that already measured the
behaviour on the fork engine during S5/S6/S7, with the probes'
hard-won fixture lessons kept intact:

* reactions are read at **fixed supports**, never at pattern-``sp``
  DOFs (those do not populate reliably);
* in the slip case the driven edge's orthogonal DOF is pinned — an
  ``ENT`` at exactly zero strain has exactly zero tangent
  (``ENTMaterial.cpp:123-126``), so that direction would otherwise be
  unrestrained;
* ``puts`` pads its numbers, so every parse allows ``\\s+``.

What lives here, against the ADR's S10 list:

* **bonded limit** — a bilateral interface with stiff ``elastic`` laws
  converging on an exact bonded tie, at a rate derived from the
  stiffness ratio (see the test's own derivation);
* **unilateral zero-tension, both signs, curved master, both normal
  laws** — the quarter-annulus tunnel (concave master, normals into the
  opening), driven into contact and into separation, for ``ent`` *and*
  ``epp_gap`` (INV-1's sign rule is **per law kind**);
* **slip saturation** — Σ tangential reaction pinned to
  ``τ_b × L × t``;
* **compose invariance (INV-2)** — the same module composed with a 90°
  rotation must reproduce the unrotated run's reactions;
* **MPCO springs channels (D6)** — ``Results.from_mpco(...)
  .elements.springs`` read back per pair against the engine's own
  ``eleResponse`` — values to 1e-12, and one distinct element identity
  per pair. The identity half started as a strict ``xfail``: this
  battery found interface rows being persisted with a duplicated
  ``fem_eid``, fixed on main by #951, and the gate is now a plain
  assertion.

Not here, deliberately:

* the deck-provable half of the battery (per-pair slip closed form,
  INV-3 closure on an uneven mesh, h5-round-trip byte identity, the
  recorder declaration) — ``tests/opensees/integration/
  test_interface_acceptance_battery.py``, which needs no binary;
* **1-vs-N rank identity** — already owned end to end by
  ``test_interface_partitioned_numeric_twin.py`` (serial vs 2-rank
  OpenSeesMP at ``REL_TOL = 1e-10``, worst measured 3.6e-15) and, at
  the byte level, by ``tests/opensees/integration/
  test_interface_partitioned_emit.py``. Duplicating it here would add
  an MPI dependency to this file and no coverage.

Environment: ``APEGMSH_OPENSEES_BIN`` must point at a dist ``bin``
holding ``OpenSees.exe``; the whole module skips loudly otherwise. No
MPI needed.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import apeSees

from tests.opensees._helpers.interface_fixtures import tunnel_geometry

RUN_TIMEOUT_S = 300
THICKNESS = 0.5
E_ROCK, NU_ROCK = 30.0e9, 0.2

# The S5/S6 probe laws — the strip cases keep them verbatim so the
# measured slip cap stays the same number the probe measured.
K_N, K_T, TAU_B = 1.0e9, 1.0e8, 2.5e5
NORMAL_ENT = NormalLaw(kind="ent", k_per_area=K_N)
TANGENTIAL_EPP = TangentialLaw(kind="epp", k_per_area=K_T, tau_b=TAU_B)

SLIP_CAP = TAU_B * 1.0 * THICKNESS          # 1.25e5 N on the unit face
DELTA = 0.05                                # prescribed drive, >> yield slip


# ---------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------

def _dist_bin() -> "Path | None":
    d = os.environ.get("APEGMSH_OPENSEES_BIN")
    if not d:
        return None
    p = Path(d)
    return p if (p / "OpenSees.exe").is_file() else None


pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.slow,
    pytest.mark.skipif(
        _dist_bin() is None,
        reason=(
            "APEGMSH_OPENSEES_BIN unset or does not hold OpenSees.exe — "
            "point it at a Ladruno-fork dist\\bin to run the ADR 0093 "
            "S10 acceptance battery"
        ),
    ),
]


def _run_env() -> "dict[str, str]":
    dist = _dist_bin()
    assert dist is not None
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(dist), env.get("PATH", "")])
    tcl = dist.parent / "lib" / "tcl8.6"
    if tcl.is_dir():
        env["TCL_LIBRARY"] = str(tcl)
    return env


def _run(deck: Path) -> str:
    dist = _dist_bin()
    assert dist is not None
    r = subprocess.run(
        [str(dist / "OpenSees.exe"), str(deck)], cwd=str(deck.parent),
        env=_run_env(), capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    out = r.stdout + r.stderr
    assert "S10_DONE" in out, (
        f"deck {deck.name} did not complete:\n{out[-3000:]}")
    return out


def _floats(out: str, tag: str) -> "list[float]":
    """Every value printed under ``tag``, in deck order.

    ``puts`` pads its numbers, so the split is on ``\\s+``. Every
    ``S10_*`` line therefore carries **values only** — no node or
    element ids — and ordering is what identifies them; a printed tag
    would land in this list as if it were a measurement.
    """
    vals: "list[float]" = []
    for line in re.findall(rf"{tag}\s+(.+)", out):
        vals.extend(float(v) for v in line.split())
    return vals


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------
# Shared model helpers
# ---------------------------------------------------------------------

def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _two_squares(g, n: int) -> None:
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
    g.mesh.generation.generate(2)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
    g.physical.add(1, [_curve_at_x(left, 0.0)], name="base")
    g.physical.add(1, [_curve_at_x(right, 2.0)], name="anchor")


def _quads(ops: apeSees, *pgs: str) -> None:
    mat = ops.nDMaterial.ElasticIsotropic(E=E_ROCK, nu=NU_ROCK, rho=2400)
    for pg in pgs:
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")


def _nodes_at_x(fem, x: float, tol: float = 1e-6) -> "list[int]":
    return sorted(
        int(t) for t, p in zip(fem.nodes.ids, fem.nodes.coords)
        if abs(float(p[0]) - x) < tol)


_STATIC_CHAIN = """
constraints Transformation
numberer RCM
system UmfPack
test NormDispIncr 1.0e-11 200
algorithm Newton
"""


def _zerolength_tags(text: str) -> "dict[tuple[int, int], int]":
    """``(iNode, jNode) -> element tag`` for every emitted interface
    ``zeroLength``."""
    out: "dict[tuple[int, int], int]" = {}
    for m in re.finditer(r"^element zeroLength (\d+) (\d+) (\d+) ",
                         text, re.M):
        out[(int(m.group(2)), int(m.group(3)))] = int(m.group(1))
    return out


# =====================================================================
# 1 — the bonded limit reproduces an exact bilateral tie
# =====================================================================
#
# ``g.constraints.tie()`` is NOT the reference here, and cannot be: it
# is a *surface* verb — it refuses a 2D line master outright
# (``TieDef master: label 'face' contains non-surface entities [(1, 2)]
# — a face constraint requires dim=2 surfaces``) — while
# ``g.constraints.interface()`` refuses 3D surface masters (ADR 0093
# D2). Their domains are disjoint, so "the same mesh, once with each"
# is not constructible. Two further reasons the substitution is the
# honest one even if it were: ``tie``'s default enforcement is a
# *penalty* ``ASDEmbeddedNodeElement``, so the comparison would be
# penalty-against-penalty and would green a bonded interface that is
# merely as wrong as the penalty tie; and on a node-for-node coincident
# interface — the only topology this verb accepts — shape-function
# interpolation degenerates to identity anyway.
#
# The exact bonded twin for coincident pairs is therefore
# ``g.constraints.equal_dof`` (an ``equalDOF`` MP constraint enforced
# exactly by the Transformation handler, the "equation route" of the
# S10 brief).

BOND_FX, BOND_FY = 1.0e6, 2.5e5             # per anchor node
BOND_N = 4                                  # transfinite division


def _bonded_fem(k: "float | None"):
    """``k=None`` ⇒ the exact ``equalDOF`` twin; otherwise a bilateral
    ``elastic``/``elastic`` interface at ``k_per_area = k``."""
    with apeGmsh(model_name="s10_bond", verbose=False) as g:
        _two_squares(g, n=BOND_N)
        if k is None:
            g.constraints.equal_dof("face", "wire", dofs=[1, 2])
        else:
            g.constraints.interface(
                "face", "wire",
                normal=NormalLaw(kind="elastic", k_per_area=k),
                tangential=TangentialLaw(kind="elastic", k_per_area=k),
                thickness=THICKNESS, name="Bonded")
        return g.mesh.queries.get_fem_data()


def _bonded_deck(fem, path: Path) -> str:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    _quads(ops, "rock", "liner")
    ops.fix(pg="base", dofs=(1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="anchor", forces=(BOND_FX, BOND_FY))
    ops.tcl(str(path))
    probes = _nodes_at_x(fem, 2.0) + _nodes_at_x(fem, 1.0)
    lines = [path.read_text(encoding="utf-8"), _STATIC_CHAIN, """
integrator LoadControl 1.0
analysis Static
if {[analyze 1] != 0} { puts "S10_FAIL"; exit 1 }
"""]
    for n in probes:
        lines.append(f'puts "S10_U [nodeDisp {n} 1] [nodeDisp {n} 2]"')
    lines.append('puts "S10_DONE"')
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.name


@pytest.fixture(scope="module")
def bonded(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    d = tmp_path_factory.mktemp("s10_bonded")
    out: "dict[str, object]" = {}
    for tag, k in (("tie", None), ("k14", 1.0e14), ("k16", 1.0e16)):
        fem = _bonded_fem(k)
        deck = d / f"bond_{tag}.tcl"
        _bonded_deck(fem, deck)
        out[tag] = np.asarray(_floats(_run(deck), "S10_U"))
        if k is not None:
            out[f"{tag}_pairs"] = len(fem.elements.interfaces)
    return out


def test_bonded_limit_converges_on_the_exact_tie(bonded) -> None:
    """A bilateral interface with stiff ``elastic`` laws must reproduce
    the ``equalDOF``-bonded solution, to a tolerance *derived* from the
    stiffness — not guessed.

    **The bound.** The interface is a bed of springs in series with the
    continuum, and a bed has three compliances, all ∝ 1/k:

    * axial — total normal stiffness ``K_N = Σ k·A_trib = k·L·t``
      (INV-3's closure is what makes that exact instead of
      mesh-dependent), so a normal resultant ``N`` buys ``N / (k·L·t)``;
    * shear — the tangential bed has the same total stiffness, so a
      shear resultant ``V`` buys ``V / (k·L·t)``;
    * rotation — the NORMAL springs at offset ``y`` resist the moment
      the eccentric shear delivers to the face, with
      ``K_θ = k·t·L³/12``, giving a relative rotation ``θ = M / K_θ``.
      (The tangential springs all sit on the face line, so they add
      nothing to ``K_θ``.)

    Downstream of the interface the liner rides on all three, so a probe
    at ``(x, y)`` measured from the face centroid sees at most
    ``Δ_N + θ·|y|`` in x and ``Δ_V + θ·|x|`` in y.

    Here the anchor edge carries ``N = 4·1e6`` and ``V = 4·2.5e5`` one
    metre outboard of the face, so ``M = V·1``, and the probes reach
    ``|y| = L/2 = 0.5`` on the face and ``|x| = 1`` at the anchor. At
    ``k = 1e14``: ``Δ_N = 8.0e-8``, ``Δ_V = 2.0e-8``,
    ``θ = 2.4e-7 rad`` ⇒ bounds of 2.0e-7 m in x and 2.6e-7 m in y —
    both dominated by the rotation, which is the honest answer for a
    spring bed under an eccentric load and not something the axial term
    alone would have predicted. At ``k = 1e16``, a hundredth of each.
    Nothing else differs between the two decks — same mesh, material,
    supports, loads.

    The test asserts the measured gap sits inside that bound (1.5x
    margin for the second-order coupling between the modes) and — the
    stronger statement, and the one a merely-small-but-wrong answer
    cannot fake — that a 100x stiffness step shrinks the gap ~100x.
    That is what "limit" means.
    """
    tie = bonded["tie"]
    n_pairs = bonded["k14_pairs"]
    assert n_pairs == BOND_N
    n_tot = BOND_FX * BOND_N
    v_tot = BOND_FY * BOND_N
    m_tot = v_tot * 1.0                     # anchor edge is 1 m outboard
    face_l = 1.0

    def _bound(k: float, dof: int) -> float:
        delta_n = n_tot / (k * face_l * THICKNESS)
        delta_v = v_tot / (k * face_l * THICKNESS)
        theta = 12.0 * m_tot / (k * THICKNESS * face_l ** 3)
        if dof == 0:                        # x: axial + theta * |y|max
            return 1.5 * (delta_n + theta * 0.5 * face_l)
        return 1.5 * (delta_v + theta * 1.0)  # y: shear + theta * |x|max

    gaps = {}
    for tag, k in (("k14", 1.0e14), ("k16", 1.0e16)):
        got = bonded[tag]
        assert got.shape == tie.shape
        # values come out as [ux, uy] per probe node, in deck order
        d = np.abs(got - tie).reshape(-1, 2)
        gaps[tag] = d.max(axis=0)
        for dof in (0, 1):
            bound = _bound(k, dof)
            assert gaps[tag][dof] <= bound, (
                f"{tag} dof{dof + 1}: gap {gaps[tag][dof]:.3e} exceeds the "
                f"spring-bed bound {bound:.3e}")

    print(f"\nS10 bonded limit: max |u_interface - u_tie| "
          f"k=1e14 {gaps['k14']} / k=1e16 {gaps['k16']}; "
          f"|u_tie|max = {np.abs(tie).max():.6e}")

    # The limit itself: 100x stiffer ⇒ ~100x closer.
    for dof in (0, 1):
        ratio = gaps["k14"][dof] / max(gaps["k16"][dof], 1e-300)
        assert 50.0 < ratio < 200.0, (
            f"dof{dof + 1}: the bonded gap does not scale as 1/k "
            f"(ratio {ratio:.1f} over a 100x stiffness step) — the "
            f"residual is not the interface's series compliance")


# =====================================================================
# 2 — unilateral zero-tension: both signs, curved master, both laws
# =====================================================================

TUNNEL_DRIVE = 2.0e-3           # radial displacement imposed on the rock rim
EPP_GAP_LAW = NormalLaw(
    kind="epp_gap", k_per_area=K_N, tau_b_n=1.0e9, gap=0.0)


def _tunnel_fem(normal_law):
    with apeGmsh(model_name="s10_tunnel", verbose=False) as g:
        tunnel_geometry(g)
        g.constraints.interface(
            "face", "wire", normal=normal_law, tangential=TANGENTIAL_EPP,
            thickness=THICKNESS, name="RockLiner")
        return g.mesh.queries.get_fem_data()


def _tunnel_deck(fem, path: Path, *, sign: float) -> None:
    """Fix the liner's intrados, then drive the rock's outer rim
    radially by ``sign * TUNNEL_DRIVE``.

    ``sign < 0`` is ground convergence — the rock squeezes onto the
    liner and the pairs must close; ``sign > 0`` pulls the ground away
    and every pair must go free. Both cases are statically determinate
    on each side of the interface, so the OPEN case is a legitimate
    analysis rather than a singular one.
    """
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    _quads(ops, "rock", "liner")
    ops.fix(pg="intrados", dofs=(1, 1))
    ops.tcl(str(path))
    text = path.read_text(encoding="utf-8")
    zl = _zerolength_tags(text)
    by_master = {i: t for (i, _j), t in zl.items()}
    coords = {int(t): tuple(map(float, p))
              for t, p in zip(fem.nodes.ids, fem.nodes.coords)}

    lines = [text, "pattern Plain 1 Linear {"]
    for n in sorted(int(t) for t in fem.nodes.select(pg="ground").ids):
        x, y = coords[n][0], coords[n][1]
        r = math.hypot(x, y)
        lines.append(f"    sp {n} 1 {sign * TUNNEL_DRIVE * x / r!r}")
        lines.append(f"    sp {n} 2 {sign * TUNNEL_DRIVE * y / r!r}")
    lines.append("}")
    lines.append(_STATIC_CHAIN)
    lines.append("""
integrator LoadControl 0.1
analysis Static
if {[analyze 10] != 0} { puts "S10_FAIL"; exit 1 }
reactions""")
    for n in sorted(int(t) for t in fem.nodes.select(pg="intrados").ids):
        lines.append(f'puts "S10_R [nodeReaction {n} 1] '
                     f'[nodeReaction {n} 2]"')
    for master in sorted(by_master):
        lines.append(
            f'puts "S10_F [eleResponse {by_master[master]} basicForce]"')
    lines.append('puts "S10_DONE"')
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(scope="module")
def tunnel(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    d = tmp_path_factory.mktemp("s10_tunnel")
    out: "dict[str, object]" = {}
    for law_id, law in (("ent", NORMAL_ENT), ("epp_gap", EPP_GAP_LAW)):
        fem = _tunnel_fem(law)
        out[f"{law_id}_pairs"] = len(fem.elements.interfaces)
        for case, sign in (("close", -1.0), ("open", +1.0)):
            deck = d / f"tunnel_{law_id}_{case}.tcl"
            _tunnel_deck(fem, deck, sign=sign)
            text = _run(deck)
            forces = _floats(text, "S10_F")
            out[f"{law_id}_{case}_normal"] = forces[0::2]
            out[f"{law_id}_{case}_tangential"] = forces[1::2]
            # The RESULTANT the liner's support has to carry (vector
            # sum over the intrados, not a sum of magnitudes) — the
            # quantity the interface's transmitted force must equal.
            r = np.asarray(_floats(text, "S10_R")).reshape(-1, 2).sum(axis=0)
            out[f"{law_id}_{case}_reaction"] = float(np.hypot(*r))
        out[f"{law_id}_slip_capacity"] = TAU_B * sum(
            float(rec.a_trib) for rec in fem.elements.interfaces)
    return out


@pytest.mark.parametrize("law_id", ["ent", "epp_gap"])
def test_curved_master_transmits_compression_on_every_pair(
        tunnel, law_id) -> None:
    """Ground convergence: every pair on the 90° arc closes.

    ``basicForce[0]`` is the NORMAL spring's force, in the pair's own
    local frame. Under ``strain = x̂·(u_j − u_i)`` with local-x the
    master's outward normal (INV-1), a closing interface reads
    *negative*. Every one of the seven pairs — from the springline at
    θ=0° to the crown at θ=90°, where the normal has swung a full
    quarter turn — must read negative and non-trivial. A frame flipped
    anywhere (resolver normal, record node order, emit i/jNode,
    material translation) turns this into a tension-only interface that
    still converges: exactly the silent class the verb exists to kill.
    """
    n = np.asarray(tunnel[f"{law_id}_close_normal"])
    assert len(n) == tunnel[f"{law_id}_pairs"] == 7
    print(f"\nS10 tunnel[{law_id}] CLOSE: normal forces {n}; "
          f"|R_intrados| = {tunnel[f'{law_id}_close_reaction']:.6e}")
    assert (n < 0).all(), f"{law_id}: a pair failed to carry compression: {n}"
    assert np.abs(n).min() > 1.0e2
    assert tunnel[f"{law_id}_close_reaction"] > 1.0e5


@pytest.mark.parametrize("law_id", ["ent", "epp_gap"])
def test_curved_master_carries_nothing_in_separation(tunnel, law_id) -> None:
    """The other sign, the same fixture: the ground is pulled away and
    every pair must go **exactly** free.

    ``ent`` and ``epp_gap`` reach that zero by different code paths —
    ``ENTMaterial`` returns zero above zero strain
    (``ENTMaterial.cpp:112-118``); ``EPPGapMaterial`` branches on
    ``sign(fy)`` and only *warns* on a mismatched ``(Fy, gap)`` pair
    (``EPPGapMaterial.cpp:109-168``), which is why the emit-time
    translation forces ``Fy < 0`` and ``gap ≤ 0`` rather than trusting
    the caller. Running both is INV-1's per-law sign rule; the zero is
    asserted as an exact ``0.0`` because that is what a genuinely
    unloaded unilateral spring returns — anything else is leakage.

    The liner is **not** left at exactly zero reaction, and pretending
    otherwise would be a fixture lie: the tangential law is bilateral
    (``epp``, elastic until it slips), so the retreating rock still
    drags the liner along the face. That residual has a derived
    ceiling — with the normal path carrying exactly nothing, everything
    the liner's support feels arrives through the tangential bed, which
    cannot deliver more than its own slip capacity
    ``Σ τ_b·A_trib = τ_b·L_polyline·t``. The measured resultant is
    checked against that ceiling, and against the closed case, rather
    than against a made-up "≈ 0".
    """
    n = np.asarray(tunnel[f"{law_id}_open_normal"])
    t = np.asarray(tunnel[f"{law_id}_open_tangential"])
    close = np.abs(np.asarray(tunnel[f"{law_id}_close_normal"])).max()
    r_open = tunnel[f"{law_id}_open_reaction"]
    cap = tunnel[f"{law_id}_slip_capacity"]
    print(f"\nS10 tunnel[{law_id}] OPEN: normal forces {n}; "
          f"tangential {t}; |R_intrados| = {r_open:.6e} vs slip capacity "
          f"{cap:.6e} (closed case: |F_n|max {close:.6e}, |R| "
          f"{tunnel[f'{law_id}_close_reaction']:.6e})")
    assert (n == 0.0).all(), (
        f"{law_id}: separation carried normal force {n} — the interface "
        f"is not zero-tension")
    assert np.abs(t).max() > 0.0, (
        f"{law_id}: the open pairs carry nothing at all — a dead element "
        f"would also print zeros for the normal channel")
    assert r_open <= cap, (
        f"{law_id}: the separated liner carries {r_open!r}, above the "
        f"tangential bed's whole slip capacity {cap!r} — force is "
        f"reaching it through a path that should be open")
    assert r_open < 0.1 * tunnel[f"{law_id}_close_reaction"]


# =====================================================================
# 3 — slip saturation: Σ tangential = τ_b × L × t
# =====================================================================

def _strip_fem(n: int = 2):
    with apeGmsh(model_name="s10_strip", verbose=False) as g:
        _two_squares(g, n=n)
        g.constraints.interface(
            "face", "wire", normal=NORMAL_ENT, tangential=TANGENTIAL_EPP,
            thickness=THICKNESS, name="RockLiner")
        return g.mesh.queries.get_fem_data()


def _drive_deck(fem, path: Path, model_text: str, *, fixed, driven,
                dof: int, delta: float, pin_dof: "int | None" = None) -> None:
    lines = [model_text]
    for n in fixed:
        lines.append(f"fix {n} 1 1")
    if pin_dof is not None:
        flags = " ".join("1" if d == pin_dof else "0" for d in (1, 2))
        for n in driven:
            lines.append(f"fix {n} {flags}")
    lines.append("pattern Plain 1 Linear {")
    for n in driven:
        lines.append(f"    sp {n} {dof} {delta!r}")
    lines.append("}")
    lines.append(_STATIC_CHAIN)
    lines.append("""
integrator LoadControl 0.1
analysis Static
if {[analyze 10] != 0} { puts "S10_FAIL"; exit 1 }
reactions""")
    for n in fixed:
        lines.append(f'puts "S10_R [nodeReaction {n} {dof}]"')
    lines.append('puts "S10_DONE"')
    path.write_text("\n".join(lines), encoding="utf-8")


def _strip_reactions(fem, tmp: Path, stem: str) -> "dict[str, float]":
    """PUSH / PULL / SLIP on the two-square strip — the S5 probe's three
    cases, reactions summed at the FIXED rock supports."""
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    _quads(ops, "rock", "liner")
    base = tmp / f"{stem}_model.tcl"
    ops.tcl(str(base))
    model_text = base.read_text(encoding="utf-8")
    fixed = _nodes_at_x(fem, 0.0)
    driven = _nodes_at_x(fem, 2.0)
    out = {}
    for case, dof, delta, pin in (("push", 1, -DELTA, None),
                                  ("pull", 1, +DELTA, None),
                                  ("slip", 2, +DELTA, 1)):
        deck = tmp / f"{stem}_{case}.tcl"
        _drive_deck(fem, deck, model_text, fixed=fixed, driven=driven,
                    dof=dof, delta=delta, pin_dof=pin)
        out[case] = float(sum(_floats(_run(deck), "S10_R")))
    return out


@pytest.fixture(scope="module")
def strip(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, float]":
    d = tmp_path_factory.mktemp("s10_strip")
    return _strip_reactions(_strip_fem(), d, "strip")


def test_slip_saturates_at_tau_b_times_length_times_thickness(strip) -> None:
    """The tangential closed form on the engine: drive the liner along
    the face far past yield and the interface can transmit exactly
    ``τ_b × L × t`` — 2.5e5 × 1.0 × 0.5 = 1.25e5 N — no matter how the
    tributary shares are distributed among the pairs, because they sum
    to ``L × t`` (INV-3).

    The driven edge's x-DOF is pinned: at the yield plateau the
    tangential tangent is zero and the ``ENT`` normal at zero strain has
    zero tangent too (``ENTMaterial.cpp:123-126``), so without the pin
    that direction is unrestrained — a fixture necessity, not a code
    concern.
    """
    print(f"\nS10 strip reactions: {strip}")
    assert abs(strip["slip"]) == pytest.approx(SLIP_CAP, rel=1e-9), (
        f"slip cap {strip['slip']!r} != tau_b*L*t {SLIP_CAP!r}")


def test_strip_push_transmits_and_pull_separates(strip) -> None:
    """The unilateral pair on the flat strip — the straight-master
    sibling of the tunnel cases, kept because it is the fixture the
    slip cap is measured on and a sign flip there would otherwise only
    show up as a strange cap."""
    assert abs(strip["push"]) > 1.0e6
    assert abs(strip["pull"]) < 1.0e-3 * abs(strip["push"])
    assert abs(strip["pull"]) < 1.0


# =====================================================================
# 4 — compose invariance (INV-2): the 90°-rotated frame
# =====================================================================

def _module_h5(path: Path) -> None:
    with apeGmsh(model_name="s10_module", verbose=False) as g:
        _two_squares(g, n=2)
        g.constraints.interface(
            "face", "wire", normal=NORMAL_ENT, tangential=TANGENTIAL_EPP,
            thickness=THICKNESS, name="RockLiner")
        g.mesh.queries.get_fem_data().to_h5(str(path))


def _host_h5(path: Path) -> None:
    with apeGmsh(model_name="s10_host", verbose=False) as g:
        surf = g.model.geometry.add_rectangle(9.0, 9.0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, surf)], n=2)
        g.mesh.generation.generate(2)
        g.physical.add(2, [surf], name="host")
        g.mesh.queries.get_fem_data(dim=2).to_h5(str(path))


@pytest.fixture(scope="module")
def rotated(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    """The module composed onto a host with a 90° rotation about z, so
    the interface line lands at ``y'=1`` with outward normal ``+y'``,
    then RUN in that rotated frame."""
    d = tmp_path_factory.mktemp("s10_rot")
    mod, host, merged = d / "mod.h5", d / "host.h5", d / "merged.h5"
    _module_h5(mod)
    _host_h5(host)
    g = apeGmsh.from_h5(str(host))
    g.compose(str(mod), label="A",
              rotate=(0.0, 0.0, 1.0, math.pi / 2.0),
              translate=(2.0, 0.0, 0.0))
    g.save(str(merged))
    fem = FEMData.from_h5(str(merged))

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    _quads(ops, "A.rock", "A.liner", "host")
    base = d / "rot_model.tcl"
    ops.tcl(str(base), flat=True)
    model_text = base.read_text(encoding="utf-8")

    coords = {int(t): tuple(map(float, p))
              for t, p in zip(fem.nodes.ids, fem.nodes.coords)}

    def where(fn):
        return sorted(n for n, p in coords.items() if fn(p))

    # Rotated geometry: rock y'∈[0,1], liner y'∈[1,2], both x'∈[1,2];
    # the untouched host square sits out at (9, 9).
    fixed = where(lambda p: abs(p[1]) < 1e-6 and 0.5 < p[0] < 2.5)
    driven = where(lambda p: abs(p[1] - 2.0) < 1e-6 and 0.5 < p[0] < 2.5)
    host_nodes = where(lambda p: p[0] > 8.0)
    assert len(fixed) == 2 and len(driven) == 2 and host_nodes

    out: "dict[str, object]" = {"orients": [r.orient for r in
                                            fem.elements.interfaces]}
    # In the rotated frame the interface acts along dof 2 and slips
    # along dof 1 — the mirror of the unrotated strip.
    for case, dof, delta, pin in (("push", 2, -DELTA, None),
                                  ("pull", 2, +DELTA, None),
                                  ("slip", 1, +DELTA, 2)):
        deck = d / f"rot_{case}.tcl"
        _drive_deck(fem, deck, model_text, fixed=fixed + host_nodes,
                    driven=driven, dof=dof, delta=delta, pin_dof=pin)
        # Reactions are summed over the rock edge only (the host square
        # is fully fixed scenery and carries nothing).
        text = _run(deck)
        vals = _floats(text, "S10_R")
        by_node = dict(zip(fixed + host_nodes, vals))   # deck print order
        out[case] = float(sum(by_node[n] for n in fixed))
    return out


def test_compose_rotates_the_per_pair_frames(rotated) -> None:
    """INV-2's numeric gate: every record's orient 6-tuple came out of
    the rotation, not out of the original frame. A compose that rotated
    node coordinates but skipped (or mis-rotated) the orient vectors
    passes every field test written against the same rotation matrix —
    this and the run below are what catch it."""
    for orient in rotated["orients"]:
        np.testing.assert_allclose(
            orient, (0, 1, 0, -1, 0, 0), atol=1e-9)


def test_rotated_compose_reproduces_the_unrotated_mechanics(
        rotated, strip) -> None:
    """The mechanical half of INV-2, and the reason the numeric gate
    above is not enough: run the composed model and the reactions must
    be the unrotated strip's, digit for digit.

    Same module mesh, same laws, same drive — only the frame differs
    (and a fully fixed host square off at (9,9) that carries nothing),
    so PUSH / PULL / SLIP must land on the same three numbers, including
    the ``τ_b × L × t`` cap.
    """
    deltas = {c: _rel(rotated[c], strip[c]) for c in ("push", "slip")}
    print(f"\nS10 compose invariance: rotated={{"
          f"push: {rotated['push']:.16e}, pull: {rotated['pull']:.16e}, "
          f"slip: {rotated['slip']:.16e}}} vs unrotated {strip}; "
          f"relative deltas {deltas}")
    for case, rel in deltas.items():
        assert rel < 1e-9, (
            f"rotated {case} reaction {rotated[case]!r} differs from the "
            f"unrotated {strip[case]!r} (rel {rel:.3e})")
    # PULL is a near-zero against a near-zero; compare it absolutely
    # against the scale the closed case sets.
    assert abs(rotated["pull"] - strip["pull"]) < 1e-9 * abs(strip["push"])
    assert abs(rotated["slip"]) == pytest.approx(SLIP_CAP, rel=1e-9)


# =====================================================================
# 5 — MPCO springs channels, per pair (ADR 0093 D6)
# =====================================================================

@pytest.fixture(scope="module")
def springs(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    """Push the strip into contact with an MPCO recorder asking for the
    two spring channels, then read the file back through ``Results``."""
    d = tmp_path_factory.mktemp("s10_springs")
    fem = _strip_fem(n=3)
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    _quads(ops, "rock", "liner")
    ops.fix(pg="base", dofs=(1, 1))
    ops.fix(pg="anchor", dofs=(1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="face", forces=(1.0e6, 2.0e5))
    ops.recorder.MPCO(
        file="springs.mpco",
        nodal_responses=("displacement",),
        elem_responses=("basicForce", "deformation"))
    deck = d / "springs.tcl"
    ops.tcl(str(deck))
    model_h5 = d / "springs_model.h5"
    ops.h5(str(model_h5))

    text = deck.read_text(encoding="utf-8")
    zl = _zerolength_tags(text)
    tags = [zl[k] for k in sorted(zl)]
    lines = [text, _STATIC_CHAIN, """
integrator LoadControl 1.0
analysis Static
if {[analyze 1] != 0} { puts "S10_FAIL"; exit 1 }
"""]
    for t in tags:
        lines.append(f'puts "S10_EF [eleResponse {t} basicForce]"')
        lines.append(f'puts "S10_ED [eleResponse {t} deformation]"')
    lines.append("remove recorders")
    lines.append('puts "S10_DONE"')
    deck.write_text("\n".join(lines), encoding="utf-8")
    out = _run(deck)

    forces = _floats(out, "S10_EF")
    deforms = _floats(out, "S10_ED")
    return {
        "dir": d, "tags": tags, "mpco": d / "springs.mpco",
        "model_h5": model_h5,
        "force_0": forces[0::2], "force_1": forces[1::2],
        "def_0": deforms[0::2], "def_1": deforms[1::2],
    }


def test_mpco_springs_channels_match_the_engine(springs) -> None:
    """ADR 0093 D6, end to end: the per-pair recorder story.

    The interface deliberately emits a plain ``zeroLength`` + uniaxials
    rather than a ``zeroLengthContact*`` variant precisely so this read
    works — the contact variants are written by MPCO as
    ``UnknownMovableObject`` and are invisible to ``Results`` discovery
    (``_response_catalog.py:1508-1512``). This test is what keeps that
    promise honest: run the model, then read
    ``results.elements.springs.get(component="spring_force_0")`` back
    and match it against the engine's own ``eleResponse`` for the same
    elements, spring by spring, direction 0 (normal) and 1 (tangential),
    force and deformation.

    The read is made with an explicit ``ids=`` filter of the pairs'
    OpenSees element tags — the filtered path, kept distinct from the
    unfiltered one that
    ``test_mpco_springs_unfiltered_read_keeps_ops_tags_per_pair``
    covers, because the two travel different branches of
    ``ElementTagTranslator.read_translation`` and only the unfiltered
    one depends on how the model h5 stamped these rows.
    """
    from apeGmsh.results import Results

    assert springs["mpco"].is_file(), "the run wrote no .mpco file"
    r = Results.from_mpco(springs["mpco"], model_h5=springs["model_h5"])
    try:
        stage = r.stage(r.stages[0].name)
        available = stage.elements.springs.available_components()
        print(f"\nS10 MPCO springs components: {sorted(available)}")
        for comp in ("spring_force_0", "spring_force_1",
                     "spring_deformation_0", "spring_deformation_1"):
            assert comp in available, (
                f"{comp} missing from the springs topology: {available}")

        tags = springs["tags"]
        expected = {
            "spring_force_0": springs["force_0"],
            "spring_force_1": springs["force_1"],
            "spring_deformation_0": springs["def_0"],
            "spring_deformation_1": springs["def_1"],
        }
        for comp, engine in expected.items():
            slab = stage.elements.springs.get(component=comp, ids=tags)
            got = dict(zip((int(e) for e in slab.element_index),
                           (float(v) for v in np.asarray(slab.values)[-1])))
            want = dict(zip(tags, engine))
            assert set(got) == set(want), (
                f"{comp}: recorded elements {sorted(got)} do not match "
                f"the interface pairs {sorted(want)}")
            for tag, ref in want.items():
                assert got[tag] == pytest.approx(ref, rel=1e-12, abs=1e-12), (
                    f"{comp}[{tag}]: MPCO {got[tag]!r} != eleResponse "
                    f"{ref!r}")
            print(f"  {comp}: {[want[t] for t in tags]}")
    finally:
        r.close()


def test_mpco_springs_unfiltered_read_keeps_ops_tags_per_pair(
        springs) -> None:
    """The identity half of D6: an UNFILTERED springs read labels every
    column with its own pair.

    This gate started life as a strict ``xfail``. The battery found
    that interface ``zeroLength`` rows were persisted into the model h5
    with a duplicated ``fem_eid`` — measured ``[32, 32, 32]`` for ops
    tags ``[11, 12, 13]``, 32 being the *last quad's* id — because
    ``H5Emitter.element()`` inherited a stale per-element side channel
    instead of defaulting to the ADR 0049 sentinel. The values were
    right; only the identity was lost, which is the quiet kind of wrong
    (a user plotting per-pair spring force would get three curves under
    one label).

    Fixed on main by #951: the emitter clears both side channels per
    call, and element-minting sites install the sentinel explicitly, so
    ``ElementTagTranslator.from_model`` skips these rows and the raw ops
    tags survive into ``element_index``. This test now asserts that
    directly — one distinct ops tag per pair, values still matching the
    engine — so a regression re-labels itself loudly instead of going
    back to being "expected".
    """
    from apeGmsh.results import Results

    tags = springs["tags"]
    r = Results.from_mpco(springs["mpco"], model_h5=springs["model_h5"])
    try:
        stage = r.stage(r.stages[0].name)
        slab = stage.elements.springs.get(component="spring_force_0")
        index = [int(e) for e in slab.element_index]
        print(f"\nS10 MPCO springs unfiltered element_index: {index} "
              f"(pairs are ops tags {tags})")
        assert index == tags, (
            f"unfiltered springs read labelled the columns {index}, not "
            f"the pairs' ops tags {tags}")
        got = dict(zip(index, (float(v)
                               for v in np.asarray(slab.values)[-1])))
        for tag, ref in zip(tags, springs["force_0"]):
            assert got[tag] == pytest.approx(ref, rel=1e-12, abs=1e-12)
    finally:
        r.close()
