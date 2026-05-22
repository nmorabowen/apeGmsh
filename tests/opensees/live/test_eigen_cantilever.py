"""Live in-process smoke test for :meth:`apeSees.eigen`.

Build a 1-element elastic cantilever with a lumped tip mass and verify
the first natural frequency matches the closed-form

    ω_1 = √( 3 E I / (m L³) )

for a massless beam with a tip-concentrated mass.  This is exact for
the discrete model we build (no distributed mass on the element, all
inertia at the tip node).

Gated by the ``live`` marker — requires openseespy. Pinned against
the Ladruno build hash in the module docstring so a future binary
regression is caught here rather than in a downstream project.

Reference build hash for this smoke (May 2026 baseline):
    288f6d0f1e990111393aa26576482feef3e9f1f5
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees

# Module-level gate: skip every test if openseespy is not installed.
openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_node_beam,
)


@pytest.mark.live
def test_eigen_cantilever_first_mode_matches_closed_form() -> None:
    """Tip-mass cantilever: ω_1 = √(3EI/(mL³)) within rel=2e-2."""
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
    m_tip = 100.0

    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=A, E=E, Iz=Iz, Iy=Iy, G=G, J=J,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    # Tip mass — translational along X (DOF 1). Other DOFs get a tiny
    # mass so the eigen solver does not see singular rows for them; this
    # is the standard apeGmsh idiom for a lumped translational mass on
    # a 6-DOF node.  The tiny rotational inertia is small enough not to
    # influence the lowest translational mode.
    ops.mass(
        pg="Top",
        values=(m_tip, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6),
    )

    result = ops.eigen(num_modes=3)
    assert result.eigenvalues.shape == (3,)

    omega_1 = float(result.omega[0])
    expected = float(np.sqrt(3.0 * E * Iz / (m_tip * L**3)))
    assert omega_1 == pytest.approx(expected, rel=2e-2)


@pytest.mark.live
def test_eigen_derived_quantities_consistent() -> None:
    """omega / freq / periods are consistent on a real live solve."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(100.0, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6))

    result = ops.eigen(num_modes=2)
    np.testing.assert_allclose(result.omega, np.sqrt(result.eigenvalues))
    np.testing.assert_allclose(result.freq, result.omega / (2.0 * np.pi))
    np.testing.assert_allclose(result.periods, 1.0 / result.freq)


@pytest.mark.live
def test_eigen_mode_shape_returns_ndf_vector() -> None:
    """mode_shape(node, mode) returns a length-ndf numpy array."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(100.0, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6))

    result = ops.eigen(num_modes=1)

    # Tip node tag is 2 in make_two_node_beam.
    shape_tip = result.mode_shape(node=2, mode=1)
    assert shape_tip.shape == (6,)
    # At least one tip DOF must move in the first mode.
    assert float(np.abs(shape_tip).max()) > 0.0
