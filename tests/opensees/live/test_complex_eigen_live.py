"""Live tests for :meth:`apeSees.complex_eigen` (ADR 0075 slice 5).

Classical-Rayleigh oracle (fork guide): under global Rayleigh
``rayleigh a0 a1 0 0`` the complex modes collapse to the classical
ones — ``ζ_k = a0/(2ω_k) + a1·ω_k/2`` and
``ω_d = ω₀·√(1−ζ²)`` — on BOTH projection routes (default assembled
Route B and ``-closedForm`` Route A).

Skips on builds without the fork ADR-46 ``complexEigen`` command; the
inverse test pins the friendly fork-required error (the path the
currently-deployed pre-ADR-46 build exercises).
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


def _has_complex_eigen() -> bool:
    return getattr(openseespy, "complexEigen", None) is not None


requires_complex_eigen = pytest.mark.skipif(
    not _has_complex_eigen(),
    reason=(
        "bound openseespy build lacks the Ladruno ADR-46 complexEigen "
        "command — rebuild the fork from ladruno HEAD"
    ),
)

_A0, _A1 = 4.0, 1.0e-5


def _damped_cantilever() -> "apeSees":
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(100.0, 100.0, 100.0, 1.0, 1.0, 1.0))
    ops.damping.rayleigh(alpha_m=_A0, beta_k=_A1)
    return ops


@requires_complex_eigen
@pytest.mark.live
@pytest.mark.parametrize("closed_form", [False, True])
def test_classical_rayleigh_zeta_matches_closed_form(
    closed_form: bool,
) -> None:
    """ζ_k == a0/(2ω_k) + a1·ω_k/2 on both projection routes."""
    result = _damped_cantilever().complex_eigen(
        4, solver="-fullGenLapack", closed_form=closed_form,
    )
    assert result.n_modes == 4
    zeta_expected = _A0 / (2.0 * result.omega0) + _A1 * result.omega0 / 2.0
    np.testing.assert_allclose(result.zeta, zeta_expected, rtol=1e-6)
    # Damped frequencies follow the classical relation.
    np.testing.assert_allclose(
        result.omega_d,
        result.omega0 * np.sqrt(1.0 - result.zeta**2),
        rtol=1e-6,
    )
    # All underdamped, tight residuals.
    assert set(result.kind.tolist()) == {0}
    assert float(result.resid.max()) < 1e-6


@requires_complex_eigen
@pytest.mark.live
def test_complex_eigen_omega0_matches_real_eigen() -> None:
    """The reported ω₀ echo the real eigen basis frequencies."""
    reference = _damped_cantilever().eigen(
        num_modes=3, solver="-fullGenLapack",
    )
    result = _damped_cantilever().complex_eigen(
        3, solver="-fullGenLapack",
    )
    np.testing.assert_allclose(
        np.sort(result.omega0), np.sort(reference.omega[:3]), rtol=1e-8,
    )


@pytest.mark.skipif(
    _has_complex_eigen(),
    reason="build HAS complexEigen — the fork-required error path is "
           "unreachable",
)
@pytest.mark.live
def test_complex_eigen_raises_friendly_error_on_pre_adr46_build() -> None:
    with pytest.raises(RuntimeError, match="Ladruno fork build"):
        _damped_cantilever().complex_eigen(2, solver="-fullGenLapack")
