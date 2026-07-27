"""Fork-only — ``system Pardiso -krylov`` on a genuinely nonlinear tangent.

The cantilever cases in ``tests/opensees/live/test_systems_live.py`` are
*linear*: the factorization is built once, so CGS reuse never engages and
they cannot catch a ``-krylov`` that returns a different equilibrium
point.  This one yields.

``-krylov L`` reuses the previous factorization as a CGS preconditioner
and accepts a correction whose residual has fallen to ``10**-L`` (fork
ADR-75 P1e).  That is an **inexact** solve, so the question the fork's
recipe raises — does it move the answer? — is only meaningful under full
Newton on a tangent that keeps changing.  Here an elastoplastic block is
loaded ~57× past first yield under ``Newton``, and every PARDISO mode has
to reproduce the ``UmfPack`` answer to full double precision.

(The recipe's trap 5 — ``-krylov`` picking a different post-peak branch —
needs a *limit point*.  This block hardens, so it has none; that trap is
documented on the primitive rather than tested here.)
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

pytestmark = pytest.mark.ladruno_fork


def _build_block():
    with apeGmsh(model_name="krylov_block", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="Body")
        return g.mesh.queries.get_fem_data(dim=3)


def _plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def _solve(fem, make_system) -> float:
    """Mean tip settlement after 12 full-Newton load steps."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    # sig0 low relative to the load: the block yields early and the
    # tangent changes on every iteration, so full Newton really does
    # refactorize and -krylov really does have something to reuse.
    mat = ops.nDMaterial.LadrunoJ2(
        K=1.6667e8, G=7.6923e7, sig0=2.0e5, Hiso=1.0e6,
    )
    ops.element.LadrunoBrick(pg="Body", material=mat)
    ops.fix(nodes=_plane(fem, 2, 0.0), dofs=(1, 1, 1))
    top = _plane(fem, 2, 3.0)
    ts = ops.timeSeries.Linear()
    # Spread a fixed TOTAL force over the top face, not a fixed per-node
    # one — otherwise the stress (and so whether the block yields at all)
    # silently tracks the mesh density.
    per_node = 3.0e5 / len(top)
    with ops.pattern.Plain(series=ts) as p:
        for nid in top:
            p.load(node=nid, forces=(0.0, 0.0, -per_node))
    ops.constraints.Plain()
    ops.numberer.RCM()
    make_system(ops)
    ops.test.NormDispIncr(tol=1e-10, max_iter=40)
    ops.algorithm.Newton()  # FULL Newton — the only regime -krylov pays in
    ops.integrator.LoadControl(dlam=1.0 / 12)
    ops.analysis.Static()

    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)
    assert em.analyze(steps=12) == 0, "the nonlinear reference run diverged"
    return float(np.mean([em.ops.nodeDisp(int(n), 3) for n in top]))


@pytest.mark.live
def test_krylov_reuse_does_not_move_the_nonlinear_answer() -> None:
    fem = _build_block()
    reference = _solve(fem, lambda o: o.system.UmfPack())

    # Sanity: this has to be the plastic branch, or the whole premise
    # (a tangent that keeps changing) is untested. The elastic settlement
    # at this load is ~2e-3; yielding takes it two orders further.
    assert abs(reference) > 1e-2, (
        f"block stayed elastic (uz={reference:.3e}) — -krylov would never "
        f"engage and this test would prove nothing"
    )

    for label, make in (
        ("direct", lambda o: o.system.Pardiso()),
        ("krylov=6", lambda o: o.system.Pardiso(krylov=6)),
        ("krylov=12", lambda o: o.system.Pardiso(krylov=12)),
        ("spd+krylov=6", lambda o: o.system.Pardiso(
            matrix_type="spd", krylov=6)),
    ):
        got = _solve(fem, make)
        assert got == pytest.approx(reference, rel=1e-12), (
            f"Pardiso {label} disagreed with UmfPack on a plastic tangent"
        )
