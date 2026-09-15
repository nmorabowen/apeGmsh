"""Live, fork-gated acceptance for the F9 augmentation sweep (PR #839).

Runs the §3.6 recipe end-to-end on the smallest RBE2 model apeGmsh can
build: an elastic cantilever whose tip is tied to an **offset** reference
node by a ``LadrunoKinematicCoupling`` with ``enforce="al"``. The push is
applied at the reference node, so the whole load path runs through the tie
and its constraint gap is the thing under test.

Two cadences, one recipe:

* the default (``al_update=None`` ⇒ the fork's ``commit``), where a single
  step leaves the **penalty** gap standing and the held-load sweep is what
  closes it — this is the measurement that justifies the helper;
* ``al_update="iter"``, the expert opt-in, which is legal here because the
  analysis is full Newton + ``LoadControl`` (anything else and the fork
  refuses it at the first ``update()``).

Fork-only: ``ladrunoBeginAugment`` / ``ladrunoEndAugment`` and the
coupling's ``constraintViolation`` response do not exist on stock.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees

openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh._kernel._coupling_control import CouplingControl  # noqa: E402
from apeGmsh._kernel.records._constraints import (  # noqa: E402
    NodeGroupRecord,
)
from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _has_fork() -> bool:
    from apeGmsh.opensees._target import probe_live_capabilities

    return probe_live_capabilities().has_fork


pytestmark = pytest.mark.skipif(
    not _has_fork(),
    reason="OpenSees build is not the Ladruno fork (no ladrunoBeginAugment)",
)

#: Translational penalty. k_host (tip lateral stiffness 3EI/L^3) is 6e7,
#: so this is ~1.7e2x — the low end of the fork's measured 1e2..1e4 band,
#: the regime where AL + a sweep is the right answer and cranking k is not.
_KT = 1.0e10
_P = 1.0e3          # tip push [N]


def _tied_cantilever(al_update: str | None) -> tuple[apeSees, int]:
    """Cantilever (nodes 1-2) whose tip is RBE2-tied to an offset
    reference node 3; the push is applied at node 3."""
    nodes = _NodesStub(
        ids=[1, 2, 3],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.0, 1.0)],
        node_pgs={"Base": [1], "Top": [2], "Ref": [3]},
    )
    elements = _ElementsStub(
        elem_pgs={"Cols": _ElementGroupView(ids=(1,), connectivity=((1, 2),))},
    )
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.add_node_constraints([NodeGroupRecord(
        kind="kinematic_coupling",
        master_node=3, slave_nodes=[2], dofs=[1, 2, 3],
        # exactly what g.constraints.kinematic_coupling(
        #     "Ref", "Top", dofs=[1,2,3], k=_KT,
        #     enforce="al", al_update=al_update) builds.
        control=CouplingControl(k=_KT, enforce="al", al_update=al_update),
        name="tip_tie",
    )])

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    # The reference node's rotations carry no stiffness with a single
    # slave (a rotation about the offset direction moves nothing), so
    # restrain them: the tie is then a pure offset rigid link.
    ops.fix(pg="Ref", dofs=(0, 0, 0, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=3, forces=(_P, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=50)
    # -alUpdate iter is refused by the fork outside full Newton +
    # LoadControl; both cadences run under exactly that here.
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops, 3


def _run(al_update: str | None) -> tuple[LiveOpsEmitter, int]:
    ops, _ref = _tied_cantilever(al_update)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0, "the driving step failed"
    # The coupling is the last element the tag allocator handed out.
    return emitter, max(int(t) for t in openseespy.getEleTags())


def _violation(tag: int) -> float:
    return float(openseespy.eleResponse(tag, "constraintViolation")[0])


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_augment_sweep_closes_the_constraint_default_cadence() -> None:
    """The default ``commit`` cadence leaves the penalty gap after one
    step; the held-load sweep drives it to the solver floor."""
    emitter, tag = _run(None)
    before = _violation(tag)
    assert before > 0.0, "a penalty tie cannot have a zero gap"

    tol = 1e-14
    with emitter.augment(element=tag, tol=tol, max_passes=10) as gaps:
        pass

    assert gaps, "the sweep ran no passes"
    assert gaps[-1] < tol, f"sweep stalled at {gaps[-1]:.3e} (from {before:.3e})"
    assert len(gaps) <= 10
    # Monotone contraction is the Uzawa property the sweep relies on.
    assert gaps[-1] < before
    # The run continues afterwards: the flag is cleared and the caller's
    # integrator is back, so an ordinary step still converges.
    assert emitter.analyze(steps=1) == 0


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_augment_sweep_under_al_update_iter() -> None:
    """``al_update="iter"`` is legal under full Newton + LoadControl, and
    the sweep still converges and restores the run."""
    emitter, tag = _run("iter")

    tol = 1e-14
    with emitter.augment(element=tag, tol=tol, max_passes=10) as gaps:
        pass

    assert gaps and gaps[-1] < tol, f"sweep stalled at {gaps}"
    assert emitter.analyze(steps=1) == 0
