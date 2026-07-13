"""Live in-process tests for :meth:`apeSees.modal_properties`.

A tip-mass cantilever concentrates essentially all translational-X
mass in one DOF, so the modal properties have hand-checkable values:

* mode 1 (bending about Z, translation in X) carries ~100 % of the MX
  effective mass;
* the participation factor identity for a single massive DOF reduces
  to ``Γ_1 = 1 / φ_tip,x`` (since ``Γ = φᵀMr / φᵀMφ`` collapses to
  ``m·φ / (m·φ²)``), checkable against ``mode_shape``.

``modalProperties`` is upstream OpenSees (``DomainModalProperties``) —
these tests run on stock openseespy builds, no fork marker.
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


def _tip_mass_cantilever() -> "apeSees":
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
    return ops


@pytest.mark.live
def test_modal_properties_eigenvalues_match_eigen() -> None:
    """The result's eigenvalues equal a plain eigen solve's."""
    reference = _tip_mass_cantilever().eigen(num_modes=2)
    result = _tip_mass_cantilever().modal_properties(2)
    np.testing.assert_allclose(
        result.eigenvalues, reference.eigenvalues, rtol=1e-10,
    )
    np.testing.assert_allclose(result.omega, np.sqrt(result.eigenvalues))


@pytest.mark.live
def test_modal_properties_mx_mass_ratio_concentrates_in_mode_1() -> None:
    """~100 % of the MX effective mass sits in the first mode."""
    result = _tip_mass_cantilever().modal_properties(2)
    ratios = result.mass_ratios("MX")
    assert ratios.shape == (2,)
    assert float(ratios[0]) > 99.0
    cumulative = result.cumulative_mass_ratios("MX")
    assert float(cumulative[-1]) == pytest.approx(100.0, abs=0.5)


@pytest.mark.live
def test_modal_properties_participation_factor_hand_identity() -> None:
    """Single massive DOF: Γ_1 · φ_tip,x == 1 (any eigenvector scale)."""
    result = _tip_mass_cantilever().modal_properties(1)
    gamma_1 = float(result.participation_factors("MX")[0])
    # Tip node tag is 2 in make_two_node_beam; DOF 1 is X.
    phi_tip_x = float(result.mode_shape(node=2, mode=1)[0])
    assert gamma_1 * phi_tip_x == pytest.approx(1.0, rel=1e-3)


@pytest.mark.live
def test_modal_properties_total_mass_reports_lumped_mass() -> None:
    result = _tip_mass_cantilever().modal_properties(1)
    total = result.total_mass
    # MX component carries the 100.0 lumped tip mass.
    assert float(total[0]) == pytest.approx(100.0, rel=1e-6)
