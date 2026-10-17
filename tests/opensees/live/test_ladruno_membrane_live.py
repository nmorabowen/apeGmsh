"""Live warrant tests for ``LadrunoMembrane`` (roster D10).

Stock OpenSees does not ship ``LadrunoMembrane`` (``ELE_TAG = 33022``);
``LiveOpsEmitter`` refuses it on stock before the domain call with an error
naming the Ladruno fork. On the fork, a 4 m x 4 m square panel under uniform
pressure ``q = -10`` kPa supported on its four corner nodes produces symmetric
vertical reactions ``R_z = 40`` kN per corner (``sum(R_z) = -q * A = 160`` kN).
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import live
from tests.opensees.fixtures.fem_stub import FEMStub, _ElementsStub, _NodesStub

pytestmark = pytest.mark.live


def _ops():
    try:
        return live._get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")


def _build_slab_ops(*, pressure: float = -10.0) -> tuple[apeSees, tuple[int, ...]]:
    nodes = _NodesStub(
        ids=np.array([1, 2, 3, 4], dtype=np.int64),
        coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
            dtype=np.float64,
        ),
        node_pgs={"Corners": (1, 2, 3, 4)},
    )
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs={}))
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    ops.element.LadrunoMembrane(
        nodes=(1, 2, 3, 4),
        support_nodes="Corners",
        method="tributary",
        pressure=pressure,
        stiff=(30e6, 0.2, 0.2),
    )
    ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops, (1, 2, 3, 4)


def _build_knows_ladruno_membrane() -> bool:
    """True iff the loaded OpenSees binary recognises ``element LadrunoMembrane``."""
    ops_mod = _ops()
    emitter = live.LiveOpsEmitter(wipe=True)
    emitter.model(ndm=3, ndf=6)
    emitter.node(1, 0.0, 0.0, 0.0)
    emitter.node(2, 1.0, 0.0, 0.0)
    emitter.node(3, 0.0, 1.0, 0.0)
    try:
        ops_mod.element("LadrunoMembrane", 1, "-nodes", 1, 2, 3, "-method", "tributary")
        tags = ops_mod.getEleTags()
        if isinstance(tags, int):
            tags = [tags]
        known = 1 in (tags or [])
    except Exception:
        known = False
    finally:
        live.LiveOpsEmitter(wipe=True)
    return known


def test_live_emitter_refuses_ladruno_membrane_on_stock() -> None:
    """Stock side of LadrunoMembrane's warrant row (roster D10)."""
    if _build_knows_ladruno_membrane():
        pytest.skip("stock-only: fork build supports LadrunoMembrane")
    ops, _ = _build_slab_ops()
    emitter = live.LiveOpsEmitter(wipe=True)
    with pytest.raises(RuntimeError, match=r"(?s)LadrunoMembrane.*Ladruno fork"):
        ops.build().emit(emitter)
    live.LiveOpsEmitter(wipe=True)


@pytest.mark.ladruno_fork
def test_ladruno_membrane_distributes_pressure_to_supports_on_fork() -> None:
    """Fork side of LadrunoMembrane's warrant row (roster D10): exact equilibrium."""
    if not _build_knows_ladruno_membrane():
        pytest.skip("fork build predates LadrunoMembrane (ELE_TAG = 33022)")
    ops, corners = _build_slab_ops(pressure=-10.0)
    emitter = live.LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    lv = emitter.ops
    lv.reactions()
    rz = [lv.nodeReaction(n, 3) for n in corners]
    lv.wipe()
    assert sum(rz) == pytest.approx(160.0, rel=1e-9)
    for r in rz:
        assert r == pytest.approx(40.0, rel=1e-6)
