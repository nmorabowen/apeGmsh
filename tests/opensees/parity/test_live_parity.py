"""Parity: LiveOpsEmitter vs RecordingEmitter on shared fixtures.

Drives the same model through both emitters; verifies the in-process
state OpenSees holds matches what the recording captured.

Uses ``forceBeamColumn`` with a ``beamIntegration`` rule — the modern
openseespy contract. The previous workaround using ``elasticBeamColumn``
was needed when ``forceBeamColumn._emit`` emitted the legacy Tcl
``-section secTag n_ip`` form (rejected by openseespy). That has been
fixed: ``forceBeamColumn`` now composes a :class:`BeamIntegration` and
emits the modern ``transfTag integrationTag`` shape that both Tcl and
openseespy accept.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.section.fiber import FiberPoint

# openseespy is required.
pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402
from apeGmsh.opensees.emitter.recording import RecordingEmitter  # noqa: E402

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_node_beam,
)


def _build_force_beam() -> apeSees:
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
        GJ=1e9,  # openseespy requires torsion for 3-D fiber sections
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    return ops


@pytest.mark.live
def test_live_emitter_drives_full_force_beam_deck() -> None:
    """forceBeamColumn now drives openseespy successfully (post-Phase 4.5).

    The modern command shape (transfTag, integrationTag) is what
    openseespy parses; the Phase-3 ``-section`` form would have failed
    with "WARNING invalid int inputs"."""
    ops = _build_force_beam()
    bm = ops.build()
    emitter = LiveOpsEmitter(wipe=True)
    bm.emit(emitter)

    assert len(emitter.ops.getNodeTags()) == 2
    assert len(emitter.ops.getEleTags()) == 1


@pytest.mark.live
def test_live_emitter_node_tags_match_fem_snapshot() -> None:
    """The fan-out emits nodes from the FEM snapshot; openseespy ends
    up with exactly those tags."""
    rec_ops = _build_force_beam()
    rec = RecordingEmitter()
    rec_ops.build().emit(rec)

    live_ops = _build_force_beam()
    emitter = LiveOpsEmitter(wipe=True)
    live_ops.build().emit(emitter)

    rec_node_tags = sorted(c[1][0] for c in rec.calls if c[0] == "node")
    live_node_tags = sorted(emitter.ops.getNodeTags())
    assert rec_node_tags == live_node_tags == [1, 2]
