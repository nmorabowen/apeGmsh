"""Live warrant tests for ``LadrunoForceBeamColumn`` (roster D11).

Stock OpenSees does not ship ``LadrunoForceBeamColumn`` (``ELE_TAG = 33024 / 33025``);
``LiveOpsEmitter`` refuses it on stock before the domain call with an error
naming the Ladruno fork. On the fork, an elastic 2-D cantilever of length
``L = 3.0`` m (``E = 30e9`` Pa, ``A = 0.09`` m^2, ``Iz = 6.75e-4`` m^4) under a
tip shear ``Fx = 10e3`` N reproduces the Euler-Bernoulli closed form
``u_x = Fx * L^3 / (3 * E * Iz)`` and base reaction ``M_z = Fx * L``.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import live
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

pytestmark = pytest.mark.live


def _ops():
    try:
        return live._get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")


def _build_cantilever_ops(*, fx: float = 10e3) -> apeSees:
    nodes = _NodesStub(
        ids=[1, 2],
        coords=[
            (0.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
        ],
        node_pgs={"Base": [1], "Tip": [2]},
    )
    elems = _ElementsStub(
        elem_pgs={
            "Col": _ElementGroupView(
                ids=(1,),
                connectivity=((1, 2),),
            )
        }
    )
    fem = FEMStub(nodes=nodes, elements=elems)
    ops = apeSees(cast("object", fem))
    ops.model(ndm=2, ndf=3)
    sec = ops.section.Elastic(E=30e9, A=0.09, Iz=6.75e-4)
    transf = ops.geomTransf.Linear()
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.LadrunoForceBeamColumn(
        pg="Col",
        transf=transf,
        integration=integ,
        lch=0.30,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Tip", forces=(fx, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _build_knows_ladruno_force_beam_column() -> bool:
    """True iff the loaded OpenSees binary recognises ``element LadrunoForceBeamColumn``."""
    ops_mod = _ops()
    emitter = live.LiveOpsEmitter(wipe=True)
    emitter.model(ndm=2, ndf=3)
    emitter.node(1, 0.0, 0.0)
    emitter.node(2, 0.0, 3.0)
    try:
        ops_mod.section("Elastic", 1, 30e9, 0.09, 6.75e-4)
        ops_mod.geomTransf("Linear", 1)
        ops_mod.beamIntegration("Lobatto", 1, 1, 3)
        ops_mod.element("LadrunoForceBeamColumn", 1, 1, 2, 1, 1)
        tags = ops_mod.getEleTags()
        if isinstance(tags, int):
            tags = [tags]
        known = 1 in (tags or [])
    except Exception:  # noqa: BLE001
        known = False
    finally:
        live.LiveOpsEmitter(wipe=True)
    return known


def test_live_emitter_refuses_ladruno_force_beam_column_on_stock() -> None:
    """Stock side of LadrunoForceBeamColumn's warrant row (roster D11)."""
    if _build_knows_ladruno_force_beam_column():
        pytest.skip("stock-only: fork build supports LadrunoForceBeamColumn")
    ops = _build_cantilever_ops()
    emitter = live.LiveOpsEmitter(wipe=True)
    with pytest.raises(RuntimeError, match=r"(?s)LadrunoForceBeamColumn.*Ladruno fork"):
        ops.build().emit(emitter)
    live.LiveOpsEmitter(wipe=True)


@pytest.mark.ladruno_fork
def test_ladruno_force_beam_column_cantilever_matches_closed_form_on_fork() -> None:
    """Fork side of LadrunoForceBeamColumn's warrant row (roster D11)."""
    if not _build_knows_ladruno_force_beam_column():
        pytest.skip("fork build predates LadrunoForceBeamColumn (ELE_TAG = 33024)")
    fx = 10e3
    L = 3.0
    E = 30e9
    Iz = 6.75e-4
    ops = _build_cantilever_ops(fx=fx)
    emitter = live.LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    lv = emitter.ops
    ux_tip = lv.nodeDisp(2, 1)
    lv.reactions()
    rx_base = lv.nodeReaction(1, 1)
    mz_base = lv.nodeReaction(1, 3)
    lv.wipe()
    expected_ux = fx * (L**3) / (3.0 * E * Iz)
    assert ux_tip == pytest.approx(expected_ux, rel=1e-9)
    assert rx_base == pytest.approx(-fx, rel=1e-9)
    assert mz_base == pytest.approx(fx * L, rel=1e-9)
