"""Keystone — the full apeGmsh -> fork -> .ladruno -> reader loop.

This is the test that proves the pipeline is "online": a model built
through the apeGmsh ``apeSees`` bridge is emitted into the *fork* build via
:class:`LiveOpsEmitter`, run, and writes a ``.ladruno`` file through the
fork-only ``recorder ladruno`` directive; ``Results.from_ladruno`` then
reads it back and the recorded nodal displacement matches the live
``ops.nodeDisp`` to machine precision.

It closes the seam the contract called "TO IMPLEMENT": composite/bridge
emit -> fork run -> ``.ladruno`` -> ``Results.from_ladruno``. The element
itself is a stock ``elasticBeamColumn`` (the *recorder* and the *run* are
the fork-only pieces under test); a fork-only *element* round-trip is
covered by the reinforce / coupling tests in this dir.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

from tests.opensees.fixtures.fem_stub import make_two_node_beam

pytestmark = pytest.mark.ladruno_fork


def test_recorder_roundtrip_matches_live_disp(tmp_path) -> None:
    path = str(tmp_path / "cantilever.ladruno")

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    E, A, Iz = 200e9, 0.01, 1e-4
    L, P = 1.0, 1000.0
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf, A=A, E=E, Iz=Iz, Iy=Iz, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(P, 0.0, 0.0, 0.0, 0.0, 0.0))

    # Fork-only recorder: writes the .ladruno HDF5 we read back below.
    ops.recorder.Ladruno(file=path, nodal_responses=("displacement",))

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0

    live_disp_x = emitter.ops.nodeDisp(2, 1)
    expected = P * L**3 / (3.0 * E * Iz)
    assert live_disp_x == pytest.approx(expected, rel=1e-3)

    # Flush the recorder so the .ladruno is complete, then read it back.
    emitter.ops.remove("recorders")

    r = Results.from_ladruno(path)
    nslab = r.nodes.get(component="displacement_x")
    i2 = nslab.node_ids.tolist().index(2)
    np.testing.assert_allclose(nslab.values[-1, i2], live_disp_x, atol=1e-12)
