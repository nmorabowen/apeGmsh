"""#1946 G1 — modes read back from a ``.ladruno`` written by the fork.

A 2-D cantilever with lumped masses is emitted into the fork through
:class:`LiveOpsEmitter`, solved by ``ops.eigen(3)``, and recorded by the
fork-only ``recorder ladruno -N modesOfVibration
modesOfVibrationRotational``. ``Results.from_ladruno(...).modes`` must
give three ``kind="mode"`` stages whose eigenvalues equal the live
``ops.eigen`` values and whose shapes equal ``ops.nodeEigenvector`` at
every node: the layout ``DomainCapture.capture_modes`` wrote, so the
capture can go (#1767).

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

from tests.opensees.live.test_readback_live import (
    A_BEAM,
    E,
    I_BEAM,
    build_cantilever_fem,
)

pytestmark = pytest.mark.ladruno_fork

N_MODES = 3
MASS = 25.0
MODAL_TOKENS = ("modesOfVibration", "modesOfVibrationRotational")


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


def modal_bridge(fem: Any) -> apeSees:
    """The 2-D cantilever at ``ndf=3`` with a lumped mass at every node."""
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beam", transf=t, A=A_BEAM, E=E,
                                  Iz=I_BEAM)
    ops.fix(label="base", dofs=(1, 1, 1))
    ops.mass(pg="Beam", values=(MASS, MASS, 0.0))
    return ops


def solve_and_record(ops: apeSees) -> "tuple[list[float], dict[int, list[list[float]]]]":
    """Emit, solve ``eigen(3)``, record once; return the live answer.

    Returns the eigenvalues and ``{node: [vector of mode k]}`` read from
    the live domain before the recorders close.
    """
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    live = emitter.ops
    lambdas = [float(v) for v in live.eigen(N_MODES)]
    live.record()
    vectors = {
        int(n): [list(live.nodeEigenvector(int(n), k)) for k in range(1, N_MODES + 1)]
        for n in live.getNodeTags()
    }
    live.remove("recorders")
    return lambdas, vectors


def assert_modes_match_live(
    res: Results, lambdas: "list[float]", vectors: "dict[int, list[list[float]]]",
) -> None:
    modes = res.modes
    assert len(modes) == N_MODES
    for k, (mode, lam) in enumerate(zip(modes, lambdas), start=1):
        assert mode.mode_index == k
        assert mode.eigenvalue == pytest.approx(lam, rel=1e-10)
        f_hz = math.sqrt(lam) / (2.0 * math.pi)
        assert mode.frequency_hz == pytest.approx(f_hz, rel=1e-10)
        assert mode.period_s == pytest.approx(1.0 / f_hz, rel=1e-10)
        for comp, dof in (("displacement_x", 0), ("displacement_y", 1),
                          ("rotation_z", 2)):
            slab = mode.nodes.get(component=comp)
            assert slab.values.shape == (1, len(vectors))
            got = dict(zip(slab.node_ids.tolist(), slab.values[0].tolist()))
            assert sorted(got) == sorted(vectors)
            for nid, vecs in vectors.items():
                assert got[nid] == pytest.approx(vecs[k - 1][dof], abs=1e-10)


def test_ladruno_modes_match_live_eigen(cantilever: Any, tmp_path: Path) -> None:
    path = tmp_path / "modes.ladruno"
    ops = modal_bridge(cantilever)
    ops.recorder.Ladruno(file=str(path), nodal_responses=MODAL_TOKENS)
    lambdas, vectors = solve_and_record(ops)
    assert all(lam > 0 for lam in lambdas)

    res = Results.from_ladruno(str(path))
    assert_modes_match_live(res, lambdas, vectors)
    # The host stage is not a mode: the recorder's KIND=eigen promotion
    # does not make it one.
    hosts = [s for s in res.stages if s.kind != "mode"]
    assert [s.kind for s in hosts] == ["static"]
