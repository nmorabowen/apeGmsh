"""Live, fork-gated tests for ``critical_time_step`` / ``analyze_explicit``.

Gated by the ``live`` marker and the OpenSees *Ladruno fork* build
(``criticalTimeStep()`` + the explicit integrators are fork-only). The
``dt_cr`` machinery uses **element** mass+stiffness, so the model carries
a truss mass density (``rho``) — a pure nodal-mass model yields the
``-1.0`` not-applicable sentinel (exercised below).
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees

openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_node_beam,
)


def _fork_has_explicit() -> bool:
    ops = openseespy
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.node(2, 1.0)
    try:
        ops.integrator("ExplicitBathe", 0.54)
        return True
    except Exception:
        return False
    finally:
        ops.wipe()


pytestmark = pytest.mark.skipif(
    not _fork_has_explicit(),
    reason="OpenSees build lacks the Ladruno explicit integrators",
)


def _truss(ops: apeSees, *, rho: float | None, cfl: bool = True,
           explicit: bool = True) -> None:
    """2-node axial truss (along z) with optional element mass density."""
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    ops.element.Truss(pg="Cols", A=0.01, material=mat, rho=rho)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.Diagonal()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    if explicit:
        ops.integrator.ExplicitBathe(p=0.54, cfl=cfl)
    else:
        ops.integrator.Newmark(gamma=0.5, beta=0.25)
    ops.analysis.Transient()


@pytest.mark.live
def test_critical_time_step_positive() -> None:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    _truss(ops, rho=8000.0)
    dtcr = ops.critical_time_step()
    assert dtcr > 0.0


@pytest.mark.live
def test_analyze_explicit_runs_and_substeps() -> None:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    _truss(ops, rho=8000.0)
    # duration well above dt_cr (~2e-3) forces several sub-steps.
    res = ops.analyze_explicit(duration=0.01)
    assert res.dt_cr > 0.0
    assert res.n >= 1
    assert res.dt * res.n == pytest.approx(0.01)


@pytest.mark.live
def test_analyze_explicit_dt_max_caps_the_step() -> None:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    _truss(ops, rho=8000.0)
    res = ops.analyze_explicit(duration=0.01, dt_max=1e-4)
    assert res.dt <= 1e-4 + 1e-15


@pytest.mark.live
def test_critical_time_step_raises_without_cfl() -> None:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    _truss(ops, rho=8000.0, cfl=False)
    with pytest.raises(ValueError, match="not applicable"):
        ops.critical_time_step()


@pytest.mark.live
def test_critical_time_step_raises_pure_nodal_mass() -> None:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    # No element rho -> only nodal mass would exist; dt_cr uses element mass.
    _truss(ops, rho=None)
    ops.mass(nodes=[2], values=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="not applicable"):
        ops.critical_time_step()


@pytest.mark.live
def test_critical_time_step_raises_non_explicit() -> None:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    _truss(ops, rho=8000.0, explicit=False)
    with pytest.raises(ValueError, match="not applicable"):
        ops.critical_time_step()
