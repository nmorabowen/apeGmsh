"""TIMs A10 S4 — verification of ``g.constraints.interface()`` in 3-D.

S1 built the kernel, S2 lifted the gates, S3 made the deck emit. None of
those says the 3-D interface is *right*: a deck with the correct shape
and a wrong frame, a wrong tributary or a spring acting on the wrong DOF
runs to a plausible answer on the fork and exits 0. This file is the
answer, and it is built out of comparisons rather than eyeballed
numbers, because only a comparison can fail for the right reason.

Four of them:

1. **The 2-D case, rotated into 3-D.** The ADR 0093 acceptance geometry
   — a strip footing bearing on a slab, ``ENT`` normal + ``epp``
   Coulomb tangential, the same laws and the same face — solved once as
   a plane-strain 2-D model of thickness ``t`` and once as its 3-D twin
   extruded ONE element deep to the same ``t``, with every node's
   out-of-plane DOF fixed. The twin is exact by construction, not
   approximately so: each 2-D pair splits into the two z-layer pairs
   above and below it, each carrying half its ``A_trib`` (measured: the
   3-D shares are exactly half, and both models close on
   ``L x t = 0.5``), and each 2-D nodal load splits evenly over the same
   two nodes. So the settlement and the interface's normal-force sum
   must agree to ROUND-OFF, and the tolerance below (1e-12 relative) is
   three orders looser than what was measured (1.4e-16 / 1.6e-16) only
   to leave room for a different solver's summation order.

2. **The three springs, read back.** The S3 slice claimed a 3-D pair
   comes back as ``spring_force_0..2`` with no catalog change, because
   ``n_springs`` is read from ``META/NUM_COMPONENTS`` — claimed, never
   measured (adversarial review row 16). Measured here through
   ``Results.from_mpco``, the same route the 2-D acceptance battery
   uses: normal compressive under the vertical load, in-plane tangent
   negligible, OUT-of-plane tangent exactly zero (its DOF is fixed by
   the plane-strain fixities, which is what makes ``0.0`` a real
   prediction rather than a coincidence), and the normal channel summing
   to the applied load.

3. **The u-p passenger DOF, at model scale.** The ``(4, 3)`` pair the
   campaign actually has, with a real ``LadrunoUP`` soil, a pressure
   datum declared the way the ADR 0074 / A2 gate requires (a ``fix`` on
   DOF 4 of a carrier node — the deck PASSES the gate, it does not
   sidestep it) and a non-zero pore pressure imposed on an interface
   node. Two things must then hold, and both are measured against a
   twin rather than against a hand number: the pore-pressure field must
   be the ``equalDOF 1 2 3`` twin's, digit for digit, and the
   interface element's own force vector must carry an exact ``0.0`` in
   the master's DOF-4 slot. The second is the fork's own G2 assertion
   (``contact_3d_passenger_dof_adoption.md``) reproduced through
   apeGmsh; the first is the statement a user cares about.

4. **A master that wraps a corner.** The S1 kernel pins the box-corner
   and reentrant-fold rules against synthetic facet arrays; the S3
   review could not reach either through the verb (row 4) and left it
   to S4. It IS reachable — with a slave that is ONE conformal mesh
   across the corner, which a fragment of three boxes gives — and the
   averaged corner normal, the tributary closure and the reentrant
   refusal are pinned in ``tests/mesh/test_interface_verb.py``. What
   is pinned HERE is the half that needs an engine: the corner-wrapping
   deck runs on the fork with zero refusals and every pair in
   compression.

Environment: ``pytest.mark.ladruno_fork`` (root ``conftest.py``
auto-skips unless the live backend resolves to the fork) plus
``APEGMSH_OPENSEES_BIN`` pointing at a dist ``bin`` holding the Tcl
binary; the module skips loudly without it. Measured 2026-09-08 against
fork build ``1652f945c`` (ADR 96).

Why the decks are run by hand here rather than through
``ops.tcl(run=True, bin=...)``: the measurements need ``eleResponse``
and ``nodeDisp`` values the recorders do not carry (MPCO's nodal
``displacement`` is 3-wide even on an ndf-4 node, and its ``PRESSURE``
channel is ``nodePressure``, not the u-p DOF). Every deck therefore
ends in a ``A10_DONE`` marker and :func:`_run` asserts it — strictly
stronger than reading the exit code, since ``OpenSees.exe`` returns 0
on a Tcl error. The fork's three ADR 96 refusal lines are scanned for
on the same output.
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
from apeGmsh.opensees import apeSees

RUN_TIMEOUT_S = 300
NDIV = 3
THICKNESS = 0.5                     # the 2-D out-of-plane depth == the 3-D one
E_SOIL, NU_SOIL = 30.0e9, 0.2
P_TOTAL = -3.0e6                    # total vertical load on the footing cap
P_IMPOSED = 1.0e6                   # imposed pore pressure, u-p case

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)

#: The twin is exact by construction; this leaves room only for a
#: different summation order (measured 1.4e-16 / 1.6e-16).
TWIN_REL_TOL = 1.0e-12

#: What the fork prints when it refuses a pair or disables an element —
#: ADR 96's own wording, per ``contact_3d_passenger_dof_adoption.md``.
_FORK_REFUSALS = (
    re.compile(r"differing\s+dof", re.I),
    re.compile(r"passenger\s+mode", re.I),
    re.compile(r"element\s+disabled", re.I),
)

_CHAIN = """
constraints Transformation
numberer RCM
system UmfPack
test NormDispIncr 1.0e-6 100
algorithm Newton
integrator LoadControl 1.0
analysis Static
if {[analyze 1] != 0} { puts "A10_FAIL"; exit 1 }
reactions
"""


# ---------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------

def _dist_bin() -> "Path | None":
    d = os.environ.get("APEGMSH_OPENSEES_BIN")
    if not d:
        return None
    exe = "OpenSees.exe" if os.name == "nt" else "OpenSees"
    p = Path(d)
    return p if (p / exe).is_file() else None


pytestmark = [
    pytest.mark.ladruno_fork,
    pytest.mark.slow,
    pytest.mark.skipif(
        _dist_bin() is None,
        reason=(
            "APEGMSH_OPENSEES_BIN unset or does not hold the OpenSees Tcl "
            "binary -- point it at a Ladruno-fork dist/bin to run the "
            "TIMs A10 S4 verification"
        ),
    ),
]


def _exe() -> Path:
    dist = _dist_bin()
    assert dist is not None
    return dist / ("OpenSees.exe" if os.name == "nt" else "OpenSees")


def _run(deck: Path) -> str:
    """Run a deck and return its output, asserting it reached the end.

    ``OpenSees.exe`` exits 0 on a Tcl error, so the trailing marker is
    the only proof the deck ran to completion. The fork's ADR 96
    refusals are a warning plus an inert element, so those are scanned
    for too: without this a wrong 3-D deck would pass every value check
    by having no interface at all.
    """
    dist = _dist_bin()
    assert dist is not None
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(dist), env.get("PATH", "")])
    tcl = dist.parent / "lib" / "tcl8.6"
    if tcl.is_dir():
        env["TCL_LIBRARY"] = str(tcl)
    r = subprocess.run(
        [str(_exe()), str(deck)], cwd=str(deck.parent), env=env,
        capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    out = r.stdout + r.stderr
    assert "A10_DONE" in out, (
        f"deck {deck.name} did not complete:\n{out[-3000:]}")
    offenders = [ln for ln in out.splitlines()
                 if any(p.search(ln) for p in _FORK_REFUSALS)]
    assert not offenders, (
        "the fork refused part of the interface -- ADR 96 makes these a "
        "warning plus an inert element, so the run would otherwise pass "
        "with the springs silently absent:\n" + "\n".join(offenders))
    return out


def _floats(out: str, tag: str) -> "list[float]":
    vals: "list[float]" = []
    for line in re.findall(rf"{tag}\s+(.+)", out):
        vals.extend(float(v) for v in line.split())
    return vals


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _zl_tags(text: str) -> "list[int]":
    return [int(m.group(1)) for m in
            re.finditer(r"^element zeroLength (\d+) ", text, re.M)]


def _nodes(fem, pg: str) -> "list[int]":
    return sorted(int(t) for t in fem.nodes.select(pg=pg).ids)


# =====================================================================
# Geometry — the ADR 0093 acceptance case and its one-element-deep twin
# =====================================================================

def _curve_at_y(surface: int, y: float, tol: float = 1e-6) -> int:
    for _dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[1] - y) < tol and abs(bb[4] - y) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at y={y}")


def _surface_at_y(volume: int, y: float, tol: float = 1e-6) -> int:
    for _dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False):
        bb = gmsh.model.getBoundingBox(2, abs(tag))
        if abs(bb[1] - y) < tol and abs(bb[4] - y) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary surface of volume {volume} at y={y}")


def _fem_2d():
    """Slab ``[0,1]^2`` under a strip footing ``[0,1] x [1,1.5]``,
    un-fragmented, so the two curves at ``y = 1`` carry coincident but
    distinct node sets. Plane strain of depth ``THICKNESS``."""
    with apeGmsh(model_name="a10s4_2d", verbose=False) as g:
        soil = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        foot = g.model.geometry.add_rectangle(0, 1, 0, 1, 0.5)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, soil), (2, foot)], n=NDIV)
        g.mesh.generation.generate(2)
        g.physical.add(2, [soil], name="soil")
        g.physical.add(2, [foot], name="footing")
        g.physical.add(1, [_curve_at_y(soil, 1.0)], name="face")
        g.physical.add(1, [_curve_at_y(foot, 1.0)], name="skin")
        g.physical.add(1, [_curve_at_y(soil, 0.0)], name="base")
        g.physical.add(1, [_curve_at_y(foot, 1.5)], name="cap")
        g.constraints.interface(
            "face", "skin", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, name="SoilFooting")
        return g.mesh.queries.get_fem_data()


def _fem_3d(*, slave_ndf: "int | None" = None, equal_dof: bool = False):
    """The same model extruded ONE element deep, to exactly the 2-D
    thickness: ``n = 2`` along z, ``NDIV`` in the plane. The out-of-plane
    fixities are applied at deck time (every node), which with a single
    element layer makes the hex field z-independent — plane strain, not
    an approximation of it."""
    with apeGmsh(model_name="a10s4_3d", verbose=False) as g:
        soil = g.model.geometry.add_box(0, 0, 0, 1, 1, THICKNESS)
        foot = g.model.geometry.add_box(0, 1, 0, 1, 0.5, THICKNESS)
        g.model.sync()
        g.mesh.structured.set_transfinite(
            [(3, soil), (3, foot)], n={"x": NDIV, "y": NDIV, "z": 2})
        g.mesh.generation.generate(3)
        g.physical.add(3, [soil], name="soil")
        g.physical.add(3, [foot], name="footing")
        g.physical.add(2, [_surface_at_y(soil, 1.0)], name="face")
        g.physical.add(2, [_surface_at_y(foot, 1.0)], name="skin")
        g.physical.add(2, [_surface_at_y(soil, 0.0)], name="base")
        g.physical.add(2, [_surface_at_y(foot, 1.5)], name="cap")
        if equal_dof:
            # The fork's own G3 twin: the same coincident pairs joined
            # by an exact kinematic constraint on DOFs 1-3 instead of a
            # spring bed. Neither touches DOF 4.
            g.constraints.equal_dof("face", "skin", dofs=[1, 2, 3])
        else:
            g.constraints.interface(
                "face", "skin", normal=NORMAL, tangential=TANGENTIAL,
                slave_ndf=slave_ndf, name="SoilFooting")
        return g.mesh.queries.get_fem_data(dim=3)


# =====================================================================
# Decks
# =====================================================================

def _deck_2d(fem, deck: Path) -> str:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=E_SOIL, nu=NU_SOIL, rho=0.0)
    for pg in ("soil", "footing"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")
    ops.fix(pg="base", dofs=(1, 1))
    cap = _nodes(fem, "cap")
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for n in cap:
            p.load(node=n, forces=(0.0, P_TOTAL / len(cap)))
    ops.tcl(str(deck))
    text = deck.read_text(encoding="utf-8")
    probes = [f'puts "A10_UY [nodeDisp {n} 2]"' for n in _nodes(fem, "cap")]
    probes += [f'puts "A10_F [eleResponse {t} basicForce]"'
               for t in _zl_tags(text)]
    probes.append("remove recorders")
    deck.write_text("\n".join([text, _CHAIN, *probes, 'puts "A10_DONE"']),
                    encoding="utf-8")
    return _run(deck)


def _deck_3d(fem, deck: Path, *, mpco: "Path | None" = None,
             model_h5: "Path | None" = None,
             up: bool = False, impose_p: bool = False) -> str:
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E_SOIL, nu=NU_SOIL, rho=0.0)
    if up:
        ops.element.LadrunoUP(
            pg="soil", material=mat,
            Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4,) * 3)
    else:
        ops.element.stdBrick(pg="soil", material=mat)
    ops.element.stdBrick(pg="footing", material=mat)

    base = set(_nodes(fem, "base"))
    # The base carries the u-p pressure datum the ADR 0074 / A2 gate
    # demands (slot ndm+1 of a carrier node); every other node is pinned
    # out of plane only. Two disjoint fix records, so no DOF is fixed
    # twice.
    ops.fix(pg="base", dofs=(1, 1, 1, 1) if up else (1, 1, 1))
    ops.fix(nodes=[int(t) for t in fem.nodes.ids if int(t) not in base],
            dofs=(0, 0, 1))

    cap = _nodes(fem, "cap")
    face = _nodes(fem, "face")
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for n in cap:
            p.load(node=n, forces=(0.0, P_TOTAL / len(cap), 0.0))
        if impose_p:
            p.sp(node=face[0], dof=4, value=P_IMPOSED)
    if mpco is not None:
        ops.recorder.MPCO(
            file=str(mpco), nodal_responses=("displacement",),
            elem_responses=("basicForce",))
    ops.tcl(str(deck))
    if model_h5 is not None:
        # Same apeSees instance, so the element tags the reader
        # translates through are the deck's own.
        ops.h5(str(model_h5))
    text = deck.read_text(encoding="utf-8")
    probes = [f'puts "A10_UY [nodeDisp {n} 2]"' for n in cap]
    if up:
        probes += [f'puts "A10_P [nodeDisp {n} 4]"' for n in face]
    probes += [f'puts "A10_F [eleResponse {t} basicForce]"'
               for t in _zl_tags(text)]
    probes += [f'puts "A10_EF [eleResponse {t} force]"'
               for t in _zl_tags(text)]
    probes.append("remove recorders")
    deck.write_text("\n".join([text, _CHAIN, *probes, 'puts "A10_DONE"']),
                    encoding="utf-8")
    return _run(deck)


# =====================================================================
# 1 — the 2-D case and its 3-D twin
# =====================================================================

@pytest.fixture(scope="module")
def twin(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    d = tmp_path_factory.mktemp("a10s4_twin")
    fem2 = _fem_2d()
    fem3 = _fem_3d()
    out: "dict[str, object]" = {
        "pairs_2d": len(fem2.elements.interfaces),
        "pairs_3d": len(fem3.elements.interfaces),
        "a_trib_2d": sorted(float(r.a_trib) for r in fem2.elements.interfaces),
        "a_trib_3d": sorted(float(r.a_trib) for r in fem3.elements.interfaces),
        "dir": d,
        "mpco": d / "twin3d.mpco",
    }
    o2 = _deck_2d(fem2, d / "twin2d.tcl")
    out["uy_2d"] = _floats(o2, "A10_UY")
    f2 = _floats(o2, "A10_F")
    out["fn_2d"] = f2[0::2]                      # two springs per 2-D pair
    out["ft_2d"] = f2[1::2]

    out["model_h5_3d"] = d / "twin3d_model.h5"
    o3 = _deck_3d(fem3, d / "twin3d.tcl", mpco=out["mpco"],
                  model_h5=out["model_h5_3d"])
    out["uy_3d"] = _floats(o3, "A10_UY")
    f3 = _floats(o3, "A10_F")
    out["fn_3d"] = f3[0::3]                      # three springs per 3-D pair
    out["ft1_3d"] = f3[1::3]
    out["ft2_3d"] = f3[2::3]
    out["tags_3d"] = _zl_tags((d / "twin3d.tcl").read_text(encoding="utf-8"))
    return out


def test_3d_twin_splits_every_2d_pair_in_half(twin) -> None:
    """The precondition the whole comparison rests on, asserted before
    any solver number is read: one element deep means each 2-D pair
    becomes exactly two 3-D pairs, each with half its tributary, and
    both models close on ``L x t``."""
    assert twin["pairs_2d"] == NDIV
    assert twin["pairs_3d"] == 2 * NDIV
    assert sum(twin["a_trib_3d"]) == pytest.approx(
        sum(twin["a_trib_2d"]), rel=1e-15)
    assert sum(twin["a_trib_2d"]) == pytest.approx(1.0 * THICKNESS, rel=1e-15)
    halves = sorted(a / 2.0 for a in twin["a_trib_2d"] for _ in (0, 1))
    np.testing.assert_allclose(
        twin["a_trib_3d"], halves, rtol=1e-15, atol=0.0)


def test_3d_twin_reproduces_the_2d_settlement(twin) -> None:
    """The plan's S4 acceptance, half one: the footing settles by the
    same amount in both models.

    Every cap node is compared, not just the extreme one — a frame or
    tributary error that is symmetric would move the whole cap and
    could hide behind a single min().
    """
    u2, u3 = twin["uy_2d"], twin["uy_3d"]
    assert len(u3) == 2 * len(u2)
    # The 3-D cap has two z-layers over each 2-D cap node; the field is
    # z-independent, so each 2-D value must appear twice.
    for got, want in zip(sorted(u3), sorted(u2 + u2)):
        assert _rel(got, want) < TWIN_REL_TOL, (
            f"3-D cap displacement {got!r} != 2-D {want!r} "
            f"(rel {_rel(got, want):.3e} > {TWIN_REL_TOL:g})")
    print(f"\nA10 S4 settlement: 2-D {min(u2):.17e}  3-D {min(u3):.17e}  "
          f"rel {_rel(min(u3), min(u2)):.3e}")


def test_3d_twin_reproduces_the_2d_normal_spring_sum(twin) -> None:
    """The plan's S4 acceptance, half two — and the one that would catch
    a wrong ``A_trib``: the interface transmits the same total normal
    force, and that force IS the applied load (equilibrium across the
    interface, which no amount of frame error can fake)."""
    s2 = sum(twin["fn_2d"])
    s3 = sum(twin["fn_3d"])
    print(f"A10 S4 normal spring sum: 2-D {s2:.17e}  3-D {s3:.17e}  "
          f"rel {_rel(s3, s2):.3e}  (applied {P_TOTAL:.6e})")
    assert _rel(s3, s2) < TWIN_REL_TOL
    assert s3 == pytest.approx(P_TOTAL, rel=1e-9)
    assert s2 == pytest.approx(P_TOTAL, rel=1e-9)
    # Pair by pair, too: each 2-D spring's force is the sum of its two
    # z-layer twins'. Sorting is enough to pair them — the values are
    # distinct per station and the halves are identical.
    doubled = sorted(f / 2.0 for f in twin["fn_2d"] for _ in (0, 1))
    for got, want in zip(sorted(twin["fn_3d"]), doubled):
        assert _rel(got, want) < TWIN_REL_TOL


def test_3d_twin_normal_springs_are_compressive_and_tangents_are_not(
    twin,
) -> None:
    """INV-1's sign rule in 3-D: under a purely vertical load every
    normal spring reads NEGATIVE (``ENT`` under closing strain), the
    in-plane tangent carries only the Poisson mismatch between the two
    bodies, and the OUT-of-plane tangent — whose DOF the plane-strain
    fixities hold — is exactly zero."""
    fn = twin["fn_3d"]
    assert all(f < 0.0 for f in fn), fn
    scale = max(abs(f) for f in fn)
    assert max(abs(f) for f in twin["ft1_3d"]) < 1.0e-3 * scale
    assert twin["ft2_3d"] == [0.0] * len(fn), twin["ft2_3d"]


# =====================================================================
# 2 — the springs, read back per pair (adversarial review row 16)
# =====================================================================

def test_three_springs_per_pair_read_back_from_mpco(twin) -> None:
    """S3 claimed three channels come back with no catalog change
    because ``n_springs`` is per-element metadata; measured here.

    Matched against the engine's own ``eleResponse basicForce`` for the
    same elements — the 2-D acceptance battery's idiom
    (``test_mpco_springs_channels_match_the_engine``), extended by the
    third direction that only exists in 3-D.
    """
    from apeGmsh.results import Results

    mpco = twin["mpco"]
    assert Path(mpco).is_file(), "the run wrote no .mpco file"
    r = Results.from_mpco(str(mpco), model_h5=str(twin["model_h5_3d"]))
    try:
        stage = r.stage(r.stages[0].name)
        available = stage.elements.springs.available_components()
        print(f"\nA10 S4 springs components: {sorted(available)}")
        for i in range(3):
            assert f"spring_force_{i}" in available, sorted(available)
        assert "spring_force_3" not in available, (
            "a 3-D pair carries THREE springs, not more — a fourth "
            "channel means -dir grew a slot the frame cannot name")

        tags = twin["tags_3d"]
        engine = {"spring_force_0": twin["fn_3d"],
                  "spring_force_1": twin["ft1_3d"],
                  "spring_force_2": twin["ft2_3d"]}
        for comp, ref in engine.items():
            slab = stage.elements.springs.get(component=comp, ids=tags)
            got = dict(zip((int(e) for e in slab.element_index),
                           (float(v) for v in np.asarray(slab.values)[-1])))
            assert set(got) == set(tags), (
                f"{comp}: recorded elements {sorted(got)} are not the "
                f"interface pairs {sorted(tags)}")
            for tag, want in zip(tags, ref):
                assert got[tag] == pytest.approx(want, rel=1e-12, abs=1e-12)
            print(f"  {comp}: {[got[t] for t in tags]}")

        # Equilibrium, off the RECORDER rather than off the engine
        # print: the normal channel sums to the applied load.
        slab = stage.elements.springs.get(component="spring_force_0", ids=tags)
        total = float(np.asarray(slab.values)[-1].sum())
        assert total == pytest.approx(P_TOTAL, rel=1e-9), total
    finally:
        r.close()


# =====================================================================
# 3 — the u-p soil: pressure datum, imposed p, passenger DOF
# =====================================================================

@pytest.fixture(scope="module")
def up_case(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    d = tmp_path_factory.mktemp("a10s4_up")
    out: "dict[str, object]" = {}

    fem_i = _fem_3d(slave_ndf=3)
    o = _deck_3d(fem_i, d / "up_iface.tcl", up=True, impose_p=True)
    out["face"] = _nodes(fem_i, "face")
    out["p_iface"] = _floats(o, "A10_P")
    out["ef_iface"] = _floats(o, "A10_EF")
    out["n_pairs"] = len(fem_i.elements.interfaces)

    fem_e = _fem_3d(equal_dof=True)
    o = _deck_3d(fem_e, d / "up_eqdof.tcl", up=True, impose_p=True)
    out["p_eqdof"] = _floats(o, "A10_P")

    # p == 0 everywhere (the base datum is the only pressure BC and
    # there is no source), against the (3, 3) all-brick twin.
    fem_0 = _fem_3d(slave_ndf=3)
    o = _deck_3d(fem_0, d / "up_p0.tcl", up=True)
    out["uy_up_p0"] = _floats(o, "A10_UY")
    out["ef_up_p0"] = _floats(o, "A10_EF")
    fem_b = _fem_3d()
    o = _deck_3d(fem_b, d / "brick.tcl")
    out["uy_brick"] = _floats(o, "A10_UY")
    out["ef_brick"] = _floats(o, "A10_EF")
    return out


def test_up_deck_declares_the_pressure_datum_the_gate_wants(
    tmp_path: Path,
) -> None:
    """The datum is the A2 gate's own mechanism, not a way around it:
    drop the DOF-4 flag from the base ``fix`` and the SAME deck is
    refused by name at build time. (``validate_up_pressure_datum``: a
    sealed static u-p region is singular in p and every serial solver
    factorises it through round-off, returning rc = 0 with an arbitrary
    pressure level.)"""
    from apeGmsh.opensees._internal.build import BridgeError

    fem = _fem_3d(slave_ndf=3)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E_SOIL, nu=NU_SOIL, rho=0.0)
    ops.element.LadrunoUP(
        pg="soil", material=mat,
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4,) * 3)
    ops.element.stdBrick(pg="footing", material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))              # no DOF-4 flag
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-6, max_iter=100)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    with pytest.raises(BridgeError, match=r"LadrunoUP node \d+"):
        ops.tcl(str(tmp_path / "sealed.tcl"))


def test_up_pore_pressure_is_the_equal_dof_twins(up_case) -> None:
    """The statement S4 owes: the interface does not touch the pressure
    DOF.

    The same mesh, the same u-p soil, the same datum and the same
    imposed pore pressure, joined once by ``interface()`` and once by
    ``equalDOF 1 2 3`` — the fork's own G3 twin. The two ties have very
    different mechanics (a spring bed against an exact kinematic
    constraint), so the DISPLACEMENTS differ; the pressure field must
    not, because neither tie reaches DOF 4.
    """
    a, b = up_case["p_iface"], up_case["p_eqdof"]
    assert len(a) == len(b) == len(up_case["face"])
    assert max(abs(v) for v in a) > 0.0, "the imposed pressure never took"
    scale = max(abs(v) for v in a)
    worst = max(abs(x - y) for x, y in zip(a, b))
    print(f"\nA10 S4 pore pressure at the interface: {a}\n"
          f"  equalDOF twin: {b}\n  max |delta| {worst:.3e} on {scale:.3e} "
          f"(rel {worst / scale:.3e})")
    assert worst <= 1.0e-12 * scale, (
        f"the interface moved the pore pressure by {worst!r} -- it must "
        f"be a passenger on DOF 4 (fork #808 / ADR 96)")


def test_up_interface_element_carries_zero_on_dof_four(up_case) -> None:
    """The fork's G2 assertion, reproduced at model scale: on a
    ``(4, 3)`` pair the element force vector is ``ndf1 + ndf2 = 7``
    wide and the master's DOF-4 slot is EXACTLY ``0.0``.

    Not approximately: the element never assembles into that row, so
    any non-zero would mean the spring bundle grew a fourth direction.
    """
    n = up_case["n_pairs"]
    ef = up_case["ef_iface"]
    assert len(ef) == 7 * n, (
        f"eleResponse force is {len(ef) / n} wide, not 7 -- the pair is "
        "not the (4, 3) one this test is about")
    passengers = [ef[i * 7 + 3] for i in range(n)]
    print(f"\nA10 S4 DOF-4 slots: {passengers}")
    assert passengers == [0.0] * n
    # The equal-ndf twin has no passenger slot at all: 3 + 3 = 6.
    assert len(up_case["ef_brick"]) == 6 * n


def test_up_soil_with_zero_pressure_reproduces_the_equal_ndf_twin(
    up_case,
) -> None:
    """With ``p == 0`` everywhere (the base datum is the only pressure
    BC and there is no source) the ndf-4 soil must behave exactly like
    the ndf-3 brick: the extra DOF is inert, so the ``(4, 3)`` deck and
    the ``(3, 3)`` deck are the same mechanical problem.

    This is what makes the twin comparison above meaningful — it says
    the mixed-ndf join costs nothing, and it is the only place a
    silently-coupled pressure DOF would show up as displacement.
    """
    a, b = up_case["uy_up_p0"], up_case["uy_brick"]
    assert len(a) == len(b)
    worst = max(_rel(x, y) for x, y in zip(a, b))
    print(f"\nA10 S4 (4,3) vs (3,3) cap settlement: max rel {worst:.3e}")
    assert worst < TWIN_REL_TOL


# =====================================================================
# 4 — a master that wraps a corner (adversarial review row 4)
# =====================================================================

CORNER_TH = 0.2


def _faces_at(volume: int, *, plane: str, value: float,
              tol: float = 1e-6) -> "list[int]":
    lo = {"x": 0, "y": 1, "z": 2}[plane]
    out = []
    for _d, tag in gmsh.model.getBoundary([(3, volume)], oriented=False):
        bb = gmsh.model.getBoundingBox(2, abs(tag))
        if abs(bb[lo] - value) < tol and abs(bb[lo + 3] - value) < tol:
            out.append(abs(tag))
    return out


def build_corner_model(g, *, reentrant: bool):
    """Block ``[0,1]^3`` with an L-shaped body wrapping its ``(x=1, z=1)``
    edge — a master spanning TWO adjacent faces.

    The review could not reach this through the verb because a slave
    built as two separate bodies puts TWO nodes at the corner and the
    resolver's ambiguity refusal fires first. The fix is to make the L
    ONE conformal mesh: three boxes (side plate, top plate, corner
    block) FRAGMENTED together, so the corner line's nodes are shared,
    not duplicated. Every soil-facing edge is ``NDIV`` nodes long and
    every plate is one element thick, so the two bodies meet node for
    node on both faces.

    ``reentrant=True`` swaps the roles: the L becomes the master and its
    notch is a 270-degree reentrant fold, which S1 refuses.
    """
    block = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    side = g.model.geometry.add_box(1, 0, 0, CORNER_TH, 1, 1)
    top = g.model.geometry.add_box(0, 0, 1, 1, 1, CORNER_TH)
    corner = g.model.geometry.add_box(1, 0, 1, CORNER_TH, 1, CORNER_TH)
    g.model.sync()
    ell = [int(t) for t in g.model.boolean.fragment(
        [side, top, corner], [], dim=3)]
    g.model.sync()

    st = g.mesh.structured
    st.set_transfinite([(3, block)], n={"x": NDIV, "y": NDIV, "z": NDIV})
    for v in ell:
        bb = gmsh.model.getBoundingBox(3, v)
        st.set_transfinite([(3, v)], n={
            ax: (2 if abs(bb[hi] - bb[lo]) < 0.5 else NDIV)
            for ax, lo, hi in (("x", 0, 3), ("y", 1, 4), ("z", 2, 5))})
    g.mesh.generation.generate(3)

    v_side = next(v for v in ell
                  if gmsh.model.occ.getCenterOfMass(3, v)[0] > 1.0
                  and gmsh.model.occ.getCenterOfMass(3, v)[2] < 1.0)
    v_top = next(v for v in ell
                 if gmsh.model.occ.getCenterOfMass(3, v)[0] < 1.0)
    block_faces = (_faces_at(block, plane="z", value=1.0)
                   + _faces_at(block, plane="x", value=1.0))
    ell_faces = (_faces_at(v_top, plane="z", value=1.0)
                 + _faces_at(v_side, plane="x", value=1.0))

    g.physical.add(3, [block], name="block")
    g.physical.add(3, ell, name="ell")
    g.physical.add(2, ell_faces if reentrant else block_faces, name="face")
    g.physical.add(2, block_faces if reentrant else ell_faces, name="skin")
    g.physical.add(2, _faces_at(block, plane="z", value=0.0), name="base")
    g.physical.add(2, _faces_at(block, plane="x", value=0.0), name="west")


def test_corner_wrapping_master_runs_on_the_fork(tmp_path: Path) -> None:
    """The engine half of the corner-wrap case: 15 pairs across two
    faces, the three corner ones carrying an averaged ``(1,0,1)/sqrt(2)``
    normal, pushed diagonally into the corner so BOTH faces close — and
    the fork runs it with zero refusals and no pair in tension.

    (The record-level rules — the averaged normal, the tributary
    closure, the reentrant refusal — need no engine and live in
    ``tests/mesh/test_interface_verb.py``.)
    """
    with apeGmsh(model_name="a10s4_corner", verbose=False) as g:
        build_corner_model(g, reentrant=False)
        g.constraints.interface(
            "face", "skin", normal=NORMAL, tangential=TANGENTIAL,
            name="Wrap")
        fem = g.mesh.queries.get_fem_data(dim=3)

    n_pairs = len(fem.elements.interfaces)
    assert n_pairs == 15, n_pairs
    corner = [r for r in fem.elements.interfaces
              if abs(float(r.orient[0]) - 1.0 / math.sqrt(2.0)) < 1e-9]
    assert len(corner) == NDIV, (
        f"{len(corner)} pairs carry the averaged corner normal, expected "
        f"{NDIV} — the master did not wrap the corner")

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E_SOIL, nu=NU_SOIL, rho=0.0)
    for pg in ("block", "ell"):
        ops.element.stdBrick(pg=pg, material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    ops.fix(pg="west", dofs=(1, 0, 0))
    # A body-diagonal push seats the L on BOTH faces at once; an ENT
    # bed left open on either face would show up as a tension-free
    # zero, which the assertion below would let through, so the load
    # direction is the test's own precondition.
    f = P_TOTAL / len(_nodes(fem, "ell")) / math.sqrt(2.0)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for n in _nodes(fem, "ell"):
            p.load(node=n, forces=(f, 0.0, f))

    deck = tmp_path / "corner.tcl"
    ops.tcl(str(deck))
    text = deck.read_text(encoding="utf-8")
    tags = _zl_tags(text)
    assert len(tags) == n_pairs
    probes = [f'puts "A10_F [eleResponse {t} basicForce]"' for t in tags]
    deck.write_text("\n".join([text, _CHAIN, *probes, 'puts "A10_DONE"']),
                    encoding="utf-8")
    out = _run(deck)

    normal = _floats(out, "A10_F")[0::3]
    assert len(normal) == n_pairs
    print(f"\nA10 S4 corner-wrap normal spring forces: {normal}")
    scale = max(abs(v) for v in normal)
    assert all(v <= 1e-9 * scale for v in normal), (
        "an ENT normal spring went into TENSION on the corner-wrapping "
        f"master: {normal}")
    assert all(v < 0.0 for v in normal), (
        "some pair never closed under the diagonal push, so this run "
        f"does not exercise both faces: {normal}")
