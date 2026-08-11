"""TIMs acceptance gate — the strip-footing discriminator sweep (ADR 0091).

The incident deck itself, run both ways. On the TIMs reference mesh
(3775 BezierTet10, uniform far-field surcharge q=10 over the 30 m² top,
non-associated DruckerPrager phi=20°):

* with the **Lagrange**-consistent load vector (the bundle's own ``trib``
  array — all weight on midsides, vertices exactly zero) every leg
  DIVERGES at first yield;
* with the **Bernstein**-consistent vector this branch produces
  (``q·A_face/6`` on all six control points) every leg converges the FULL
  surcharge in ONE step at ``sum Rz = 300.0000``.

That contrast is the bug and its fix, measured end to end.

The mesh bundle is not in the repo (it is TIMs' reproducer, ~550 kB of
Dropbox-synced mesh); the test skips when absent. Point
``APEGMSH_TIMS_BUNDLE`` at the folder holding ``r3_strip_far3.npz`` to
run it elsewhere.

**Engine prerequisite:** a post-#709 fork build. Before #709 BezierTet10
assembled only the upper triangle of BᵀDB and mirrored it, which is
silently wrong for the unsymmetric non-associated plastic tangent and
makes Newton locally divergent at every plastic step — the Bernstein
legs would then fail for that unrelated reason. The assertion message
says so.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from apeGmsh._kernel.defs.loads import SurfaceLoadDef
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

pytestmark = pytest.mark.ladruno_fork

_DEFAULT_BUNDLE = (
    r"C:\Users\nmb\Dropbox\obsidian\Ladruño\TIMs\apeGmsh_TIM\fork_bundle"
)
MESH = "r3_strip_far3.npz"

# Deck constants — identical to the bundle's REPRODUCE_item1_BezierTet10.py.
PHI, Q0, K_EL, NU = 20.0, 10.0, 1.5e5, 0.45
_SPHI = math.sin(math.radians(PHI))
ALPHA = 2.0 * _SPHI / (math.sqrt(3.0) * (3.0 - _SPHI))
RHO = math.sqrt(2.0) * ALPHA
G_EL = 3.0 * K_EL * (1.0 - 2.0 * NU) / (2.0 * (1.0 + NU))
TOTAL_RZ = 300.0

# gmsh tet10 local order: 0-3 vertices; 4=(0,1) 5=(1,2) 6=(0,2) 7=(0,3)
# 8=(2,3) 9=(1,3).  A tri6 face is [a, b, c, mid(ab), mid(bc), mid(ca)].
_MID = {frozenset((0, 1)): 4, frozenset((1, 2)): 5, frozenset((0, 2)): 6,
        frozenset((0, 3)): 7, frozenset((2, 3)): 8, frozenset((1, 3)): 9}
_FACE_VERTS = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))


def _bundle_dir() -> Path:
    return Path(os.environ.get("APEGMSH_TIMS_BUNDLE", _DEFAULT_BUNDLE))


@pytest.fixture(scope="module")
def bundle():
    path = _bundle_dir() / MESH
    if not path.is_file():
        pytest.skip(
            f"TIMs reference mesh not found at {path}. Set "
            f"APEGMSH_TIMS_BUNDLE to the folder holding {MESH}."
        )
    m = np.load(str(path))
    return {k: m[k] for k in m.files}


def _top_faces(cells, top_set):
    out = []
    for c in cells:
        for (a, b, d) in _FACE_VERTS:
            loc = [a, b, d, _MID[frozenset((a, b))],
                   _MID[frozenset((b, d))], _MID[frozenset((d, a))]]
            gl = [int(c[i]) for i in loc]
            if all(g in top_set for g in gl):
                out.append([g + 1 for g in gl])
    return out


def _bernstein_loads(nodes, faces):
    """The branch's load vector, through the real resolver path."""
    res = LoadResolver(np.arange(1, len(nodes) + 1), nodes.astype(float))
    defn = SurfaceLoadDef(
        target="top", magnitude=Q0, mode="traction",
        direction=(0.0, 0.0, -Q0), reduction="consistent",
        basis="bernstein",
    )
    return {
        rec.node_id: float(rec.force_xyz[2])
        for rec in res.resolve_surface_consistent(defn, faces)
    }


def _surcharge_step(ops, b, sy, loads, bbar):
    nodes, cells = b["nodes"], b["cells"]
    s_bot, s_xf = b["set_bottom"], b["set_xface"]
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for i, (x, y, z) in enumerate(nodes, 1):
        ops.node(i, float(x), float(y), float(z))
    ops.nDMaterial("DruckerPrager", 1, K_EL, G_EL, sy, RHO,
                   0.0, 0, 0, 0, 0, 0, 0, 0)
    extra = ["-bbar"] if bbar else []
    for e, c in enumerate(cells, 1):
        ops.element("BezierTet10", e, *[int(v) + 1 for v in c], 1, *extra)
    bt, xf = set(s_bot.tolist()), set(s_xf.tolist())
    for n in range(len(nodes)):
        if n in bt:
            ops.fix(n + 1, 1, 1, 1)
        else:
            ops.fix(n + 1, 1 if n in xf else 0, 1, 0)
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for nid, fz in loads.items():
        ops.load(int(nid), 0.0, 0.0, fz)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormUnbalance", 3.0e-3, 40, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        return None
    ops.reactions()
    return sum(ops.nodeReaction(int(n) + 1, 3) for n in s_bot)


@pytest.fixture(scope="module")
def loads(bundle):
    nodes, cells, trib = bundle["nodes"], bundle["cells"], bundle["trib"]
    top_set = {int(v) for v in bundle["set_top"]}
    faces = _top_faces(cells, top_set)
    assert faces, "no tri6 face found with all six nodes on the top set"
    bern = _bernstein_loads(nodes, faces)
    lagr = {int(n) + 1: -Q0 * float(trib[n])
            for n in bundle["set_top"] if trib[n] > 0}
    return {"bernstein": bern, "lagrange": lagr, "faces": faces}


# ---------------------------------------------------------------------
# Gate 2 on the reference mesh — both vectors carry the same resultant
# ---------------------------------------------------------------------

def test_both_bases_carry_the_full_surcharge(loads):
    """Why this was silent: the resultant is right in either basis."""
    for basis in ("lagrange", "bernstein"):
        total = sum(loads[basis].values())
        assert total == pytest.approx(-TOTAL_RZ, rel=1e-12), basis
    # The Bernstein vector loads every control point; the Lagrange one
    # leaves the vertices at exactly zero — the distribution difference.
    assert len(loads["bernstein"]) > len(loads["lagrange"])


# ---------------------------------------------------------------------
# Gate 3 — the discriminator sweep
# ---------------------------------------------------------------------

@pytest.mark.parametrize("bbar", [False, True], ids=["std", "bbar"])
@pytest.mark.parametrize("sy", [5.0, 0.2])
def test_bernstein_loads_converge_full_surcharge(bundle, loads, sy, bbar):
    """All four legs: one step, full surcharge, sum Rz = 300.0000."""
    ops = LiveOpsEmitter(wipe=True).ops
    rz = _surcharge_step(ops, bundle, sy, loads["bernstein"], bbar)
    assert rz is not None, (
        f"BezierTet10{' -bbar' if bbar else ''} sigma_y={sy} did NOT "
        f"converge under the Bernstein-consistent load. Either the load "
        f"branch regressed, or the engine predates fork PR #709 (the "
        f"mirrored BᵀDB tangent, which is locally divergent at every "
        f"plastic step for a non-associated material)."
    )
    assert rz == pytest.approx(TOTAL_RZ, abs=5e-4)


@pytest.mark.slow
@pytest.mark.parametrize("bbar", [False, True], ids=["std", "bbar"])
def test_lagrange_loads_diverge_at_first_yield(bundle, loads, bbar):
    """The control: the same deck under the Lagrange vector fails.

    This is the defect ADR 0091 fixes — if it ever starts passing, the
    discriminator has gone blunt (regime no longer fragile) and the
    Bernstein legs above stop proving anything.

    Marked ``slow``: a diverging leg burns all 40 Newton iterations
    before giving up (~170 s for ``std``, ~27 s for ``bbar``), where a
    converging Bernstein leg finishes in ~6-9 s. Deselect with
    ``-m 'ladruno_fork and not slow'`` to keep the fork harness quick;
    the Bernstein legs (the gate proper) still run.
    """
    ops = LiveOpsEmitter(wipe=True).ops
    rz = _surcharge_step(ops, bundle, 0.2, loads["lagrange"], bbar)
    assert rz is None, (
        f"Lagrange-consistent loads on BezierTet10"
        f"{' -bbar' if bbar else ''} converged (sum Rz = {rz}) — the "
        f"discriminator no longer discriminates."
    )
