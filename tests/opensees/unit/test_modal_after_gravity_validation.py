"""Bridge-side validation of :meth:`apeSees.modal_after_gravity` (ADR 0124).

Every refusal here fires before the build, so no openseespy is needed. The
live solve is in ``tests/opensees/live/test_modal_after_gravity_live.py``.
"""
from __future__ import annotations

from typing import Callable, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.analysis.gravity_modal import GravityState, state_drift

from tests.opensees.fixtures.fem_stub import make_two_node_beam


def _bridge(
    integrator: Callable[[apeSees], object] | None = None,
    *,
    analysis: str = "Static",
) -> apeSees:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    if integrator is None:
        ops.integrator.LoadControl(dlam=0.1)
    else:
        integrator(ops)
    getattr(ops.analysis, analysis)()
    return ops


def test_rejects_zero_modes() -> None:
    with pytest.raises(ValueError, match="num_modes must be >= 1"):
        _bridge().modal_after_gravity(0)


def test_rejects_an_incomplete_chain() -> None:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    with pytest.raises(BridgeError, match="analysis chain is incomplete"):
        ops.modal_after_gravity(1)


def test_rejects_a_transient_analysis() -> None:
    ops = _bridge(
        lambda o: o.integrator.Newmark(gamma=0.5, beta=0.25),
        analysis="Transient",
    )
    with pytest.raises(BridgeError, match="needs ops.analysis.Static"):
        ops.modal_after_gravity(1)


def test_rejects_a_non_load_control_integrator() -> None:
    ops = _bridge(
        lambda o: o.integrator.DisplacementControl(node=2, dof=1, dU=1e-3))
    with pytest.raises(ValueError, match="needs ops.integrator.LoadControl"):
        ops.modal_after_gravity(1)


def test_rejects_an_adaptive_load_control() -> None:
    ops = _bridge(lambda o: o.integrator.LoadControl(dlam=0.1, num_iter=4))
    with pytest.raises(ValueError, match="adapts the increment"):
        ops.modal_after_gravity(1)


@pytest.mark.parametrize("dlam", [0.3, 0.0, -0.1, 1.5])
def test_rejects_a_dlam_that_does_not_reach_one(dlam: float) -> None:
    ops = _bridge(lambda o: o.integrator.LoadControl(dlam=dlam))
    with pytest.raises(ValueError, match="does not reach"):
        ops.modal_after_gravity(1)


def test_gravity_steps_reads_the_last_integrator() -> None:
    ops = _bridge(lambda o: o.integrator.LoadControl(dlam=0.25))
    assert ops._gravity_steps("ctx") == 4


def test_rejects_staged_models() -> None:
    ops = _bridge()
    ops._stage_records.append(cast("object", object()))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="staged"):
        ops.modal_after_gravity(1)


def _state(d: float, r: float) -> GravityState:
    return GravityState(
        node_tags=np.asarray([1, 2]),
        disp={1: np.zeros(3), 2: np.asarray([d, 2.0, 0.0])},
        reactions={1: np.asarray([0.0, -r, 0.0]), 2: np.zeros(3)},
        time=0.0,
    )


def test_state_drift_is_relative_to_the_reference() -> None:
    a = _state(1.0, 10.0)
    assert state_drift(a, a) == (0.0, 0.0)
    d, r = state_drift(a, _state(1.002, 10.5))
    assert d == pytest.approx(0.001)
    assert r == pytest.approx(0.05)
    assert a.total_reaction(2) == -10.0
    assert a.node_disp(2)[0] == 1.0
