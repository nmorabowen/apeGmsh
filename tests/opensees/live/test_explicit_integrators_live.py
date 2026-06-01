"""Live, fork-gated runs of the Ladruno explicit integrators.

Gated by the ``live`` marker AND the OpenSees *Ladruno fork* build:
``ExplicitBathe`` / ``ExplicitBatheLNVD`` / ``CentralDifferenceLadruno``
are fork-only integrators. On stock ``openseespy`` the
``integrator <Type>`` call raises at run time — those tests skip.

Each test builds a tiny 2-node beam with tip mass, fixes the base, and
drives a short explicit transient. The assertion is a smoke check
(``analyze`` returns 0 and advances the clock) — numerical accuracy of
the explicit schemes is verified on the fork side.

⚠ The apeGmsh editable install resolves to the *main* ``src/`` tree, not
a worktree. To exercise worktree code, run with the worktree ``src`` on
``PYTHONPATH``.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees

# Gate 1: openseespy present (collect-time skip otherwise).
openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_node_beam,
)


def _fork_has(integrator: str, *args: object) -> bool:
    """True if the live OpenSees build accepts ``integrator <type> ...``.

    Probes a throwaway domain so a stock (non-fork) build skips cleanly
    rather than failing.
    """
    ops = openseespy
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.node(2, 1.0)
    try:
        ops.integrator(integrator, *args)
        return True
    except Exception:
        return False
    finally:
        ops.wipe()


def _build_tip_mass_beam(ops: apeSees) -> None:
    """Fixed-base 2-node beam with lumped tip mass — a minimal model
    with invertible mass for explicit time integration."""
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    # Lumped mass on the free tip (node 2) — translational + rotational.
    ops.mass(nodes=[2], values=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.Diagonal()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()


@pytest.mark.live
def test_explicit_bathe_runs() -> None:
    if not _fork_has("ExplicitBathe", 0.54):
        pytest.skip("OpenSees build lacks the Ladruno ExplicitBathe integrator")
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    _build_tip_mass_beam(ops)
    ops.integrator.ExplicitBathe(p=0.54, cfl=True)
    ops.analysis.Transient()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=5, dt=1e-4) == 0
    assert emitter.ops.getTime() == pytest.approx(5e-4, rel=1e-6)


@pytest.mark.live
def test_explicit_bathe_lnvd_runs() -> None:
    if not _fork_has("ExplicitBatheLNVD", 0.54, 0.8):
        pytest.skip("OpenSees build lacks the Ladruno ExplicitBatheLNVD integrator")
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    _build_tip_mass_beam(ops)
    ops.integrator.ExplicitBatheLNVD(p=0.54, alpha=0.8)
    ops.analysis.Transient()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=5, dt=1e-4) == 0


@pytest.mark.live
def test_central_difference_ladruno_runs() -> None:
    if not _fork_has("CentralDifferenceLadruno"):
        pytest.skip(
            "OpenSees build lacks the Ladruno CentralDifferenceLadruno integrator"
        )
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    _build_tip_mass_beam(ops)
    ops.integrator.CentralDifferenceLadruno(cfl=True)
    ops.analysis.Transient()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=5, dt=1e-4) == 0
