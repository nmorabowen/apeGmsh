"""Live tests for the footfall FRF matrix (ADR 0109 S1):
:func:`~apeGmsh.opensees.analysis.footfall_frf.frf_matrix`,
:func:`~apeGmsh.opensees.analysis.footfall_frf.grid_for` and the D2
generalised-mass assertion.

Three oracles (ADR 0109 D8-2):

1. the tip-mass cantilever of ``test_modal_sweeps_live.py`` is an SDOF
   in global X, so ``|A(Ω)| = Ω² / (m·|ω² − Ω² + 2iξωΩ|)`` in closed
   form — with the alphaM-only Rayleigh ratio ``ξ = a0/(2ω)``. The
   ``1/m`` is the mass-normalised ``φ² = 1/m`` the modal sum carries,
   so a normalisation error shows up here as a factor of ``m``;
2. gated on the fork, the same ``(2, 2)`` pair against
   ``apeSees.frequency_response(load=<unit load at node 2 dof 1>,
   node=2, dof=1, resp="accel")`` — the fork divides by its own
   ``generalizedMasses()``, so the two agree to round-off, not "close":
   rtol 1e-6;
3. a mutation on the assertion itself: scaling one ``φ`` column by 2
   halves that mode's ``partiFactor`` in every component while leaving
   ``partiMass`` alone (``Γ = L/m̃``, ``partiMass = L²/m̃``), so
   ``partiMass/Γ² = 4`` and the refusal must name the mode.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.footfall_frf import (
    assert_unit_generalized_mass,
    frf_matrix,
    grid_for,
)

# Module-level gate: skip every test if openseespy is not installed.
openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_node_beam,
)


def _has_modal_family() -> bool:
    return getattr(openseespy, "frequencyResponse", None) is not None


requires_modal_family = pytest.mark.skipif(
    not _has_modal_family(),
    reason=(
        "bound openseespy build lacks the Ladruno ADR-44 modal family "
        "(frequencyResponse) — rebuild the fork from ladruno HEAD"
    ),
)

# Tip-mass cantilever (same fixture as test_modal_sweeps_live.py):
# k = 3EI/L^3 = 6e7, m = 100 -> omega ~ 774.6 rad/s, f_n ~ 123.3 Hz.
_E, _IZ, _L, _M_TIP = 200e9, 1e-4, 1.0, 100.0
_K = 3.0 * _E * _IZ / _L**3
_OMEGA = float(np.sqrt(_K / _M_TIP))
_FN = _OMEGA / (2.0 * np.pi)
_A0 = 20.0  # alphaM-only Rayleigh -> xi(omega) = a0 / (2*omega)
_XI = _A0 / (2.0 * _OMEGA)

# The whole basis (node 2 carries all six free DOFs), so the modal sum
# is the exact FRF of the model and the SDOF form is the only
# approximation being measured.
_NUM_MODES = 6
_SOLVER = "-fullGenLapack"


def _cantilever_with_unit_load() -> "tuple[apeSees, object]":
    """The cantilever plus a Plain pattern carrying a unit force at
    node 2 in global X — built through the public ``ops.pattern``
    surface (no emitter methods added)."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=_E, Iz=_IZ, Iy=_IZ, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(_M_TIP, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6))
    series = ops.timeSeries.Constant()
    pattern = ops.pattern.Plain(series=series)
    pattern.load(node=2, forces=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    return ops, pattern


@pytest.mark.live
def test_frf_matrix_matches_sdof_closed_form() -> None:
    """|A_22(Ω)| == Ω²/(m|ω² − Ω² + 2iξωΩ|) with the alphaM-only ξ."""
    ops, _ = _cantilever_with_unit_load()
    props = ops.modal_properties(_NUM_MODES, solver=_SOLVER)

    grid = grid_for(props, 0.2 * _FN, 2.0 * _FN, n_extra=40)
    # D4: every modal frequency in band is on the grid, and the fine
    # cluster brackets the peak.
    f_1 = float(props.freq[0])
    assert float(grid[0]) == pytest.approx(0.2 * _FN)
    assert np.min(np.abs(grid - f_1)) < 1e-12
    in_cluster = np.abs(grid - f_1) <= 0.05 * f_1
    assert int(np.count_nonzero(in_cluster)) >= 5

    matrix = frf_matrix(
        props, exc_nodes=[2], resp_nodes=[2], dof=1, freq=grid,
        rayleigh=(_A0, 0.0),
    )
    assert matrix.normalization == f"asserted({_NUM_MODES} of {_NUM_MODES})"
    assert matrix.exc_nodes == (2,) and matrix.resp_nodes == (2,)
    # D6: rayleigh=(a0, 0) converts per mode to xi_a = a0/(2 w_a).
    np.testing.assert_allclose(
        matrix.modal_damping, _A0 / (2.0 * props.omega), rtol=1e-12,
    )

    omega_grid = 2.0 * np.pi * grid
    exact = omega_grid**2 / (
        _M_TIP * np.abs(
            _OMEGA**2 - omega_grid**2 + 2j * _XI * _OMEGA * omega_grid
        )
    )
    np.testing.assert_allclose(matrix.magnitude(2, 2), exact, rtol=2e-2)

    # e^{+iOmega t} pin: A = -Omega^2 H, so below resonance the real
    # part is negative and the response lags (positive imaginary part).
    below = matrix.accel(2, 2)[grid < _FN]
    assert np.all(np.real(below) < 0.0)
    assert np.all(np.imag(below) > 0.0)


@requires_modal_family
@pytest.mark.live
def test_frf_matrix_matches_fork_frequency_response() -> None:
    """The (2, 2) pair == the fork's frequencyResponse to 1e-6.

    ``frequency_response`` rebuilds its own live domain (and wipes the
    one a held ``ModalPropertiesResult`` reads from), so the sweep runs
    FIRST and the Python matrix is built on a second, identical bridge
    over the grid the sweep returned.
    """
    sweep_ops, pattern = _cantilever_with_unit_load()
    reference = sweep_ops.frequency_response(
        f_min=0.2 * _FN, f_max=2.0 * _FN, n_freq=41, grid="lin",
        node=2, dof=1, num_modes=_NUM_MODES,
        load=pattern, damp=0.02, resp="accel", solver=_SOLVER,
    )

    props = _cantilever_with_unit_load()[0].modal_properties(
        _NUM_MODES, solver=_SOLVER,
    )
    mine = frf_matrix(
        props, exc_nodes=[2], resp_nodes=[2], dof=1,
        freq=reference.freq, damp=0.02,
    ).accel(2, 2)

    np.testing.assert_allclose(
        np.real(mine), np.real(reference.response), rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.imag(mine), np.imag(reference.response), rtol=1e-6,
    )


@pytest.mark.live
def test_assertion_refuses_a_rescaled_mode_and_names_it() -> None:
    """Scaling one φ column by 2 makes m̃ = 4 — refuse, naming mode 3."""
    ops, _ = _cantilever_with_unit_load()
    props = ops.modal_properties(_NUM_MODES, solver=_SOLVER)
    assert assert_unit_generalized_mass(props.properties) == (
        f"asserted({_NUM_MODES} of {_NUM_MODES})"
    )

    # phi_3 -> 2*phi_3 leaves partiMass = L^2/m~ alone and halves
    # Gamma = L/m~ in every component.
    mutated = {
        key: list(np.asarray(values, dtype=np.float64))
        for key, values in props.properties.items()
    }
    for key in list(mutated):
        if key.startswith("partiFactor"):
            mutated[key][2] = mutated[key][2] / 2.0

    with pytest.raises(ValueError, match=r"mode 3 component"):
        assert_unit_generalized_mass(mutated)
