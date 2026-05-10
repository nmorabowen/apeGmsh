"""Live in-process tests via :class:`LiveOpsEmitter`.

Gated by the ``live`` marker — only runs when ``openseespy`` is
installed. The default pytest invocation (``-m "not live"``) skips
these.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees

# Importing openseespy at module load is the gate — if it's not
# installed, all tests in this module are skipped (one collect-time
# import error vs. per-test ImportError).
openseespy = pytest.importorskip("openseespy.opensees")

# Defer additional imports until openseespy is confirmed present.
from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_node_beam,
)


@pytest.mark.live
def test_cantilever_static_displacement_matches_pl3_over_3ei() -> None:
    """End-to-end live run: 1-element elastic cantilever under tip
    point load. The tip displacement matches ``P L^3 / (3 E I)``
    within rel=1e-3 (small geometric / numerical noise tolerance).

    Setup:
      * 2 nodes (base at origin, tip at (0, 0, 1)).
      * 1 elastic beam-column with E = 200e9, A = 0.01, I = 1e-4.
      * Base fully fixed.
      * Tip carries a 1 kN axial-(X)-direction load.
      * Static analysis with LoadControl, 1 step at 1.0.
    """
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    E = 200e9
    A = 0.01
    Iz = 1e-4
    Iy = 1e-4
    G = 80e9
    J = 1e-4
    L = 1.0
    P = 1000.0
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=A, E=E, Iz=Iz, Iy=Iy, G=G, J=J,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        # Apply a transverse force at the tip — direction X (DOF 1).
        # This should cause bending about the local-y axis if the
        # element is oriented along Z.
        p.load(node=2, forces=(P, 0.0, 0.0, 0.0, 0.0, 0.0))

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    # Drive the live emitter and run.
    emitter = LiveOpsEmitter(wipe=True)
    bm = ops.build()
    bm.emit(emitter)
    ret = emitter.analyze(steps=1)
    assert ret == 0, "openseespy.analyze returned non-zero"

    # Tip displacement in X direction (DOF 1).
    tip_disp_x = emitter.ops.nodeDisp(2, 1)
    expected = P * L**3 / (3.0 * E * Iz)
    assert tip_disp_x == pytest.approx(expected, rel=1e-3)


@pytest.mark.live
def test_apesees_run_emits_live_without_analyze() -> None:
    """``ops.run()`` drives the live emitter without analyzing — useful
    for declaring a model and then running custom analysis flows."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    # No analysis chain — just emit the model. Nothing should raise.
    ops.run()


@pytest.mark.live
def test_apesees_analyze_drives_full_chain() -> None:
    """``ops.analyze`` builds, emits, and analyzes through the live
    emitter; returns the openseespy ``analyze`` return value."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    ret = ops.analyze(steps=1)
    assert ret == 0


@pytest.mark.live
def test_apesees_analyze_raises_when_chain_incomplete() -> None:
    """Calling ``analyze`` without a complete analysis chain raises
    BridgeError naming the missing pieces."""
    from apeGmsh.opensees._internal.build import BridgeError

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    # No analysis chain registered — should fail.
    with pytest.raises(BridgeError, match="analysis chain is incomplete"):
        ops.analyze(steps=1)


@pytest.mark.live
def test_force_beam_with_hinge_radau_drives_openseespy() -> None:
    """Concentrated-plasticity beam-column driven through openseespy.

    Exercises the HingeRadau integration rule (i-end plastic hinge,
    j-end plastic hinge, elastic interior) — a capability the previous
    ``-section secTag n_ip`` form could not express."""
    from apeGmsh.opensees.section.fiber import FiberPoint

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    # Plastic-hinge section (steel) at each end; elastic section in
    # the interior. In practice these would be fiber sections capturing
    # the inelastic constitutive response; for a quick smoke test the
    # interior gets a 1-D elastic section.
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    plastic_sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
        GJ=1e9,
    )
    elastic_sec = ops.section.Elastic(E=200e9, A=0.01, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))

    hinge = ops.beamIntegration.HingeRadau(
        section_i=plastic_sec, lp_i=0.1,
        section_j=plastic_sec, lp_j=0.1,
        section_interior=elastic_sec,
    )
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=hinge)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    # ops.run() (no analyze) -- just drives the deck through live.
    ops.run(wipe=True)
