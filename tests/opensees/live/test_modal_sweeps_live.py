"""Live tests for the frequency-domain sweep drivers (ADR 0075
slice 3): :meth:`apeSees.frequency_response`,
:meth:`apeSees.steady_state_dynamics`, :meth:`apeSees.random_response`.

The tip-mass cantilever is effectively an SDOF in global X (all
translational mass in one DOF), so closed-form oracles apply:

* relative-displacement FRF under unit base acceleration:
  ``H(Ω) = −1 / (ω² − Ω² + 2iξωΩ)`` with ``ξ(ω) = a0/(2ω)`` for
  alphaM-only Rayleigh — magnitude AND the ``e^{+iΩt}`` phase sign;
* static limit of the load channel: ``|u(Ω→0)| = P/k``;
* white-noise base-accel PSD anchor: ``σ_x² = G0/(8ξω³)`` — a √2
  error flags a one-sided/two-sided mixup, √(2π) an Hz/rad mixup
  (fork guide P3 convention pin).

Skip on builds without the ADR-44 family (same probe as
test_modal_response_live.py).
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


def _has_modal_family() -> bool:
    return getattr(openseespy, "frequencyResponse", None) is not None


requires_modal_family = pytest.mark.skipif(
    not _has_modal_family(),
    reason=(
        "bound openseespy build lacks the Ladruno ADR-44 modal family "
        "(frequencyResponse) — rebuild the fork from ladruno HEAD"
    ),
)

# Tip-mass cantilever: k = 3EI/L^3 = 6e7, m = 100 -> omega ~ 774.6
# rad/s, f_n ~ 123.3 Hz.
_E, _IZ, _L, _M_TIP = 200e9, 1e-4, 1.0, 100.0
_K = 3.0 * _E * _IZ / _L**3
_OMEGA = float(np.sqrt(_K / _M_TIP))
_FN = _OMEGA / (2.0 * np.pi)
_A0 = 20.0  # alphaM-only Rayleigh -> xi(omega) = a0 / (2*omega)
_XI = _A0 / (2.0 * _OMEGA)


def _cantilever() -> "apeSees":
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
    return ops


@requires_modal_family
@pytest.mark.live
def test_frf_matches_sdof_closed_form_and_phase_sign() -> None:
    """Base-accel FRF == SDOF H(Ω) in magnitude and phase."""
    result = _cantilever().frequency_response(
        f_min=0.2 * _FN, f_max=2.0 * _FN, n_freq=60,
        node=2, dof=1, num_modes=2,
        base_accel_dir=1, rayleigh=(_A0, 0.0),
        solver="-fullGenLapack",
    )
    omega_grid = 2.0 * np.pi * result.freq
    h_exact = -1.0 / (
        _OMEGA**2 - omega_grid**2 + 2j * _XI * _OMEGA * omega_grid
    )
    np.testing.assert_allclose(
        result.magnitude, np.abs(h_exact), rtol=2e-2,
    )
    # e^{+iOmega t} sign pin: response lags — phase of the exact H
    # matches, including the sign of the imaginary part.
    np.testing.assert_allclose(
        np.sign(np.imag(result.response)), np.sign(np.imag(h_exact)),
    )


@requires_modal_family
@pytest.mark.live
def test_frf_load_channel_static_limit_is_p_over_k() -> None:
    """|u(Ω→0)| -> P/k with all modes retained (fork guide P2)."""
    p_x = 1.0e5
    ops = _cantilever()
    ts = ops.timeSeries.Constant()
    pat = ops.pattern.Plain(series=ts)
    pat.load(node=2, forces=(p_x, 0, 0, 0, 0, 0))
    result = ops.frequency_response(
        f_min=1e-3, f_max=1.0, n_freq=3,
        node=2, dof=1, num_modes=6,
        load=pat, damp=0.02,
        solver="-fullGenLapack",
    )
    assert float(result.magnitude[0]) == pytest.approx(
        p_x / _K, rel=1e-3,
    )


@requires_modal_family
@pytest.mark.live
def test_steady_state_dynamics_magnitude_matches_frf() -> None:
    """|frequencyResponse| == steadyStateDynamics on the same sweep."""
    kwargs = dict(
        f_min=0.5 * _FN, f_max=1.5 * _FN, n_freq=40,
        node=2, dof=1, num_modes=2,
        base_accel_dir=1, damp=0.03,
        solver="-fullGenLapack",
    )
    frf = _cantilever().frequency_response(**kwargs)
    ssd = _cantilever().steady_state_dynamics(**kwargs)
    np.testing.assert_allclose(ssd.freq, frf.freq)
    np.testing.assert_allclose(ssd.magnitude, frf.magnitude, rtol=1e-10)


@requires_modal_family
@pytest.mark.live
def test_random_response_white_noise_sdof_anchor() -> None:
    """RMS == √(G0/(8ξω³)) for band-limited white base-accel noise.

    Tight rtol on purpose: a √2 error (~41 %) flags a one-sided /
    two-sided PSD mixup, √(2π) (~150 %) an Hz/rad mixup.
    """
    g0 = 0.5
    xi = 0.03
    ops = _cantilever()
    psd = ops.timeSeries.Constant(factor=g0)
    result = ops.random_response(
        f_min=0.01 * _FN, f_max=30.0 * _FN, n_freq=1200,
        node=2, dof=1, num_modes=1,
        input_psd=psd, base_accel_dir=1, damp=xi,
        solver="-fullGenLapack",
    )
    sigma_exact = float(np.sqrt(g0 / (8.0 * xi * _OMEGA**3)))
    assert result.rms == pytest.approx(sigma_exact, rel=2e-2)
    assert result.nu0 is None  # stats not requested


@requires_modal_family
@pytest.mark.live
def test_random_response_stats_and_duration_shape() -> None:
    """-stats + -duration returns nu0/m0/m2/peak; nu0 ~ f_n for the
    narrow-band response."""
    ops = _cantilever()
    psd = ops.timeSeries.Constant(factor=0.5)
    result = ops.random_response(
        f_min=0.01 * _FN, f_max=30.0 * _FN, n_freq=1200,
        node=2, dof=1, num_modes=1,
        input_psd=psd, base_accel_dir=1, damp=0.03,
        stats=True, duration=600.0,
        solver="-fullGenLapack",
    )
    assert result.nu0 is not None and result.m0 is not None
    assert result.m2 is not None and result.peak is not None
    assert result.rms == pytest.approx(float(np.sqrt(result.m0)), rel=1e-9)
    assert result.nu0 == pytest.approx(_FN, rel=5e-2)
    assert result.peak > result.rms


@requires_modal_family
@pytest.mark.live
def test_random_response_refuses_zero_damped_in_band_mode() -> None:
    """A zero-damped mode inside the band diverges — the fork refuses."""
    ops = _cantilever()
    psd = ops.timeSeries.Constant(factor=0.5)
    with pytest.raises(Exception):
        ops.random_response(
            f_min=0.1 * _FN, f_max=3.0 * _FN, n_freq=200,
            node=2, dof=1, num_modes=1,
            input_psd=psd, base_accel_dir=1, damp=0.0,
            solver="-fullGenLapack",
        )


@pytest.mark.skipif(
    _has_modal_family(),
    reason="build HAS the ADR-44 family — the fork-required error "
           "path is unreachable",
)
@pytest.mark.live
def test_sweeps_raise_friendly_error_on_pre_adr44_build() -> None:
    ops = _cantilever()
    with pytest.raises(RuntimeError, match="Ladruno fork build"):
        ops.frequency_response(
            f_min=1.0, f_max=10.0, n_freq=10, node=2, dof=1,
            num_modes=2, base_accel_dir=1, damp=0.02,
        )
    ops2 = _cantilever()
    psd = ops2.timeSeries.Constant(factor=1.0)
    with pytest.raises(RuntimeError, match="Ladruno fork build"):
        ops2.random_response(
            f_min=1.0, f_max=10.0, n_freq=10, node=2, dof=1,
            num_modes=2, input_psd=psd, base_accel_dir=1, damp=0.02,
        )
