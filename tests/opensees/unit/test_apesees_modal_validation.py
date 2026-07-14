"""Unit tests for :meth:`apeSees.modal_properties`,
:class:`ModalPropertiesResult`, and the shared modal-response
damping-channel helper.

The live solve is exercised in
``tests/opensees/live/test_modal_properties_live.py`` (gated by the
``live`` marker — requires openseespy). These tests cover the
bridge-side validation and the pure-Python accessors.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.modal import (
    ModalPropertiesResult,
    _damping_channel_args,
)

from tests.opensees.fixtures.fem_stub import make_two_node_beam


# ---------------------------------------------------------------------------
# apeSees.modal_properties — bridge-side validation
# ---------------------------------------------------------------------------


def test_modal_properties_rejects_zero_num_modes() -> None:
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="num_modes must be >= 1"):
        ops.modal_properties(0)


def test_modal_properties_rejects_negative_num_modes() -> None:
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="num_modes must be >= 1"):
        ops.modal_properties(-3)


# ---------------------------------------------------------------------------
# Standalone timeSeries emission pin — the fork ADR-44 excitation
# channels (-baseAccel / -series / -inputPSD) reference a timeSeries by
# tag WITHOUT any pattern carrying it. This pins that a registered but
# pattern-unreferenced timeSeries still emits in the pre-element group,
# so the modal-response drivers can rely on it.
# ---------------------------------------------------------------------------


def test_standalone_time_series_emits_without_a_pattern() -> None:
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    ts = ops.timeSeries.Path(values=(0.0, 1.0, 0.0), dt=0.01)
    tag = ops.tag_for(ts)
    assert tag is not None

    rec = RecordingEmitter()
    ops.build().emit(rec)

    ts_calls = [c for c in rec.calls if c[0] == "timeSeries"]
    assert any(c[1][0] == "Path" and c[1][1] == tag for c in ts_calls), (
        "registered standalone timeSeries must emit without a pattern "
        f"referencing it — timeSeries calls seen: {ts_calls}"
    )


# ---------------------------------------------------------------------------
# ModalPropertiesResult — pure-Python accessors
# ---------------------------------------------------------------------------


def _result_3d() -> ModalPropertiesResult:
    """Two-mode 3-D-shaped properties dict (component set MX..RMZ)."""
    properties = {
        "domainSize": [3.0],
        "eigenLambda": [1.0, 4.0],
        "totalMass": [10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
        "centerOfMass": [0.0, 0.0, 2.0],
        "partiFactorMX": [1.2, -0.3],
        "partiFactorMY": [0.0, 0.0],
        "partiFactorMZ": [0.0, 0.0],
        "partiFactorRMX": [0.0, 0.0],
        "partiFactorRMY": [0.0, 0.0],
        "partiFactorRMZ": [0.0, 0.0],
        "partiMassRatiosMX": [85.0, 10.0],
        "partiMassRatiosCumuMX": [85.0, 95.0],
    }
    return ModalPropertiesResult(
        eigenvalues=np.array([1.0, 4.0]),
        properties=properties,
        _live=cast(object, None),  # type: ignore[arg-type]
    )


def test_result_omega_freq_periods_derive_from_eigenvalues() -> None:
    r = _result_3d()
    np.testing.assert_allclose(r.omega, np.array([1.0, 2.0]))
    np.testing.assert_allclose(r.freq, r.omega / (2.0 * np.pi))
    np.testing.assert_allclose(r.periods, 1.0 / r.freq)


def test_result_participation_factors_by_component() -> None:
    r = _result_3d()
    np.testing.assert_allclose(
        r.participation_factors("MX"), np.array([1.2, -0.3]),
    )


def test_result_mass_ratios_and_cumulative_are_percent_series() -> None:
    r = _result_3d()
    np.testing.assert_allclose(r.mass_ratios("MX"), np.array([85.0, 10.0]))
    np.testing.assert_allclose(
        r.cumulative_mass_ratios("MX"), np.array([85.0, 95.0]),
    )


def test_result_unknown_component_fails_loud_with_available() -> None:
    r = _result_3d()
    with pytest.raises(KeyError, match="available components"):
        r.participation_factors("MQ")


def test_result_total_mass_and_center_of_mass() -> None:
    r = _result_3d()
    np.testing.assert_allclose(
        r.total_mass, np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(r.center_of_mass, np.array([0.0, 0.0, 2.0]))


def test_result_is_frozen() -> None:
    r = _result_3d()
    with pytest.raises((AttributeError, TypeError)):
        r.eigenvalues = np.array([2.0])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _damping_channel_args — the exactly-one-of validator shared by the
# fork ADR-44 drivers
# ---------------------------------------------------------------------------


def test_damping_channel_damp_renders_flag_pair() -> None:
    args = _damping_channel_args(
        damp=0.05, rayleigh=None, modal_damp=None, context="t",
    )
    assert args == ("-damp", 0.05)


def test_damping_channel_rayleigh_renders_two_factors() -> None:
    args = _damping_channel_args(
        damp=None, rayleigh=(0.1, 0.002), modal_damp=None, context="t",
    )
    assert args == ("-rayleigh", 0.1, 0.002)


def test_damping_channel_modal_damp_renders_per_mode_ratios() -> None:
    args = _damping_channel_args(
        damp=None, rayleigh=None, modal_damp=[0.02, 0.03, 0.05], context="t",
    )
    assert args == ("-modalDamp", 0.02, 0.03, 0.05)


def test_damping_channel_rejects_none_given() -> None:
    with pytest.raises(ValueError, match="exactly one damping channel"):
        _damping_channel_args(
            damp=None, rayleigh=None, modal_damp=None, context="t",
        )


def test_damping_channel_rejects_two_given() -> None:
    with pytest.raises(ValueError, match="exactly one damping channel"):
        _damping_channel_args(
            damp=0.05, rayleigh=(0.1, 0.002), modal_damp=None, context="t",
        )


def test_damping_channel_rejects_empty_modal_damp() -> None:
    with pytest.raises(ValueError, match="at least one ratio"):
        _damping_channel_args(
            damp=None, rayleigh=None, modal_damp=[], context="t",
        )


def test_damping_channel_rejects_negative_damp() -> None:
    """Adversarial-review hardening: the fork RSA parser has no >=0
    guard, and negative ratios silently zero the CQC combination."""
    with pytest.raises(ValueError, match="damp must be >= 0"):
        _damping_channel_args(
            damp=-0.05, rayleigh=None, modal_damp=None, context="t",
        )


def test_damping_channel_rejects_mixed_sign_modal_damp() -> None:
    """One typo'd sign in a per-mode list makes the fork's CQC
    rho_ij = sqrt(xi_i*xi_j) NaN -> committed all-zero design field."""
    with pytest.raises(ValueError, match="must be >= 0"):
        _damping_channel_args(
            damp=None, rayleigh=None, modal_damp=[0.05, -0.05, 0.05],
            context="t",
        )


def test_damping_channel_allows_zero_ratio() -> None:
    """xi = 0 stays legal — the fork's own boundary (randomResponse
    refuses zero-damped IN-BAND modes itself, downstream)."""
    assert _damping_channel_args(
        damp=0.0, rayleigh=None, modal_damp=None, context="t",
    ) == ("-damp", 0.0)


def test_damping_channel_error_names_the_context() -> None:
    with pytest.raises(ValueError, match="apeSees.modal_response_history"):
        _damping_channel_args(
            damp=None, rayleigh=None, modal_damp=None,
            context="apeSees.modal_response_history",
        )


# ---------------------------------------------------------------------------
# apeSees.modal_response_history — bridge-side validation (ADR 0075
# slice 2).  Every case fails BEFORE any live emitter is constructed,
# so no openseespy is needed.
# ---------------------------------------------------------------------------


def _mrh_ops() -> apeSees:
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    return ops


def test_mrh_rejects_nonpositive_dt() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    with pytest.raises(ValueError, match="dt must be > 0"):
        ops.modal_response_history(
            dt=0.0, n_steps=10, num_modes=2,
            base_accel=ts, direction=1, damp=0.05,
        )


def test_mrh_rejects_zero_num_modes() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    with pytest.raises(ValueError, match="num_modes must be >= 1"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=0,
            base_accel=ts, direction=1, damp=0.05,
        )


def test_mrh_rejects_no_excitation_channel() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="exactly one excitation channel"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2, damp=0.05,
        )


def test_mrh_rejects_both_excitation_channels() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    pat = ops.pattern.Plain(series=ts)
    with pytest.raises(ValueError, match="exactly one excitation channel"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2,
            base_accel=ts, direction=1, load=pat, series=ts, damp=0.05,
        )


def test_mrh_base_accel_needs_direction() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    with pytest.raises(ValueError, match="needs direction="):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2,
            base_accel=ts, damp=0.05,
        )


def test_mrh_load_channel_needs_series() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    pat = ops.pattern.Plain(series=ts)
    with pytest.raises(ValueError, match="needs series="):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2, load=pat, damp=0.05,
        )


def test_mrh_rejects_unregistered_time_series_handle() -> None:
    from apeGmsh.opensees._internal.build import BridgeError
    from apeGmsh.opensees.time_series.time_series import Path

    ops = _mrh_ops()
    stray = Path(values=(0.0, 1.0), dt=0.01)  # NOT registered
    with pytest.raises(BridgeError, match="not registered"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2,
            base_accel=stray, direction=1, damp=0.05,
        )


def test_mrh_rejects_wrong_kind_handle_as_base_accel() -> None:
    """Adversarial-review hardening: object handles bypass _resolve's
    kind check, and per-kind 1-based tags mean a pattern handle passed
    as base_accel= would emit a numerically-colliding timeSeries tag
    (silent wrong ground motion)."""
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    pat = ops.pattern.Plain(series=ts)
    with pytest.raises(TypeError, match="needs an ops.timeSeries"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2,
            base_accel=pat, direction=1, damp=0.05,  # type: ignore[arg-type]
        )


def test_mrh_rejects_wrong_kind_handle_as_load() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    with pytest.raises(TypeError, match="needs an ops.pattern.Plain"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2,
            load=ts, series=ts, damp=0.05,  # type: ignore[arg-type]
        )


def test_random_response_rejects_wrong_kind_handle_as_input_psd() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    pat = ops.pattern.Plain(series=ts)
    with pytest.raises(TypeError, match="needs an ops.timeSeries"):
        ops.random_response(
            f_min=0.1, f_max=10.0, n_freq=100, node=2, dof=1,
            num_modes=2, input_psd=pat,  # type: ignore[arg-type]
            base_accel_dir=1, damp=0.02,
        )


def test_mrh_rejects_pattern_with_sp_constraints() -> None:
    from apeGmsh.opensees._internal.build import BridgeError

    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    pat = ops.pattern.Plain(series=ts)
    pat.sp(node=2, dof=1, value=0.01)
    with pytest.raises(BridgeError, match="sp constraints"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2,
            load=pat, series=ts, damp=0.05,
        )


def test_mrh_requires_exactly_one_damping_channel() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    with pytest.raises(ValueError, match="exactly one damping channel"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=2,
            base_accel=ts, direction=1,
        )


def test_mrh_rejects_zero_based_modes() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    with pytest.raises(ValueError, match="1-based mode numbers"):
        ops.modal_response_history(
            dt=0.01, n_steps=10, num_modes=3,
            base_accel=ts, direction=1, damp=0.05, modes=[0, 1],
        )


# ---------------------------------------------------------------------------
# apeSees.response_spectrum_analysis — bridge-side validation
# ---------------------------------------------------------------------------


def test_rsa_rejects_unknown_combine_rule() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="combine must be one of"):
        ops.response_spectrum_analysis(
            1, periods=[0.1, 0.5], accels=[2.0, 1.0],
            combine="RMS", num_modes=2,
        )


def test_rsa_rejects_length_mismatch() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="equal-length"):
        ops.response_spectrum_analysis(
            1, periods=[0.1, 0.5], accels=[2.0],
            combine="SRSS", num_modes=2,
        )


def test_rsa_rejects_non_increasing_periods() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="strictly.*increasing"):
        ops.response_spectrum_analysis(
            1, periods=[0.5, 0.1], accels=[1.0, 2.0],
            combine="SRSS", num_modes=2,
        )


def test_rsa_rejects_negative_period() -> None:
    """Adversarial-review hardening: the fork refuses only NEGATIVE
    Tn — the bridge matches (a leading T=0 PGA anchor is legal; see
    the live test for the acceptance half)."""
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="non-negative"):
        ops.response_spectrum_analysis(
            1, periods=[-0.1, 0.5], accels=[1.0, 2.0],
            combine="SRSS", num_modes=2,
        )


def test_rsa_cqc_requires_damping() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="CQC needs a damping channel"):
        ops.response_spectrum_analysis(
            1, periods=[0.1, 0.5], accels=[2.0, 1.0],
            combine="CQC", num_modes=2,
        )


def test_rsa_rejects_zero_direction() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="direction is 1-based"):
        ops.response_spectrum_analysis(
            0, periods=[0.1, 0.5], accels=[2.0, 1.0],
            combine="SRSS", num_modes=2,
        )


# ---------------------------------------------------------------------------
# Frequency-domain sweep drivers — bridge-side validation (ADR 0075
# slice 3).  Every case fails BEFORE any live emitter is constructed.
# ---------------------------------------------------------------------------


def test_sweep_rejects_inverted_band() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="f_min < f_max"):
        ops.frequency_response(
            f_min=10.0, f_max=1.0, n_freq=100, node=2, dof=1,
            num_modes=2, base_accel_dir=1, damp=0.02,
        )


def test_sweep_rejects_unknown_grid() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="grid must be one of"):
        ops.frequency_response(
            f_min=0.1, f_max=10.0, n_freq=100, node=2, dof=1,
            num_modes=2, grid="geometric", base_accel_dir=1, damp=0.02,
        )


def test_sweep_rejects_unknown_resp() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="resp must be one of"):
        ops.steady_state_dynamics(
            f_min=0.1, f_max=10.0, n_freq=100, node=2, dof=1,
            num_modes=2, base_accel_dir=1, damp=0.02, resp="jerk",
        )


def test_sweep_rejects_both_excitation_channels() -> None:
    ops = _mrh_ops()
    ts = ops.timeSeries.Path(values=(0.0, 1.0), dt=0.01)
    pat = ops.pattern.Plain(series=ts)
    with pytest.raises(ValueError, match="exactly one excitation channel"):
        ops.frequency_response(
            f_min=0.1, f_max=10.0, n_freq=100, node=2, dof=1,
            num_modes=2, base_accel_dir=1, load=pat, damp=0.02,
        )


def test_sweep_requires_exactly_one_damping_channel() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="exactly one damping channel"):
        ops.frequency_response(
            f_min=0.1, f_max=10.0, n_freq=100, node=2, dof=1,
            num_modes=2, base_accel_dir=1,
        )


def test_random_response_requires_two_sweep_points() -> None:
    ops = _mrh_ops()
    psd = ops.timeSeries.Constant()
    with pytest.raises(ValueError, match="n_freq must be >= 2"):
        ops.random_response(
            f_min=0.1, f_max=10.0, n_freq=1, node=2, dof=1,
            num_modes=2, input_psd=psd, base_accel_dir=1, damp=0.02,
        )


def test_random_response_rejects_unregistered_psd_handle() -> None:
    from apeGmsh.opensees._internal.build import BridgeError
    from apeGmsh.opensees.time_series.time_series import Constant

    ops = _mrh_ops()
    stray = Constant()  # NOT registered
    with pytest.raises(BridgeError, match="not registered"):
        ops.random_response(
            f_min=0.1, f_max=10.0, n_freq=100, node=2, dof=1,
            num_modes=2, input_psd=stray, base_accel_dir=1, damp=0.02,
        )


def test_random_response_rejects_nonpositive_duration() -> None:
    ops = _mrh_ops()
    psd = ops.timeSeries.Constant()
    with pytest.raises(ValueError, match="duration must be > 0"):
        ops.random_response(
            f_min=0.1, f_max=10.0, n_freq=100, node=2, dof=1,
            num_modes=2, input_psd=psd, base_accel_dir=1, damp=0.02,
            duration=0.0,
        )


# ---------------------------------------------------------------------------
# apeSees.eigen_feast — bridge-side validation (ADR 0075 slice 4)
# ---------------------------------------------------------------------------


def test_eigen_feast_rejects_inverted_band() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="f_min < f_max"):
        ops.eigen_feast(50.0, 1.0)


def test_eigen_feast_rejects_negative_f_min() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="f_min < f_max"):
        ops.eigen_feast(-1.0, 50.0)


# ---------------------------------------------------------------------------
# apeSees.complex_eigen + ComplexEigenResult (ADR 0075 slice 5)
# ---------------------------------------------------------------------------


def test_complex_eigen_result_parses_flat_seven_per_mode() -> None:
    from apeGmsh.opensees.analysis.complex_eigen import ComplexEigenResult

    # Two modes: underdamped + rigid.
    flat = [
        10.0, 9.9, 0.05, -0.5, 9.9, 0.0, 1e-12,
        0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
    ]
    r = ComplexEigenResult.from_flat(flat)
    assert r.n_modes == 2
    np.testing.assert_allclose(r.omega0, [10.0, 0.0])
    np.testing.assert_allclose(r.zeta, [0.05, 0.0])
    np.testing.assert_allclose(r.lam[0], -0.5 + 9.9j)
    assert list(r.kind) == [0, 2]
    np.testing.assert_allclose(r.freq_d, r.omega_d / (2.0 * np.pi))


def test_complex_eigen_result_rejects_non_multiple_of_seven() -> None:
    from apeGmsh.opensees.analysis.complex_eigen import ComplexEigenResult

    with pytest.raises(ValueError, match="7 values per"):
        ComplexEigenResult.from_flat([1.0, 2.0, 3.0])


def test_complex_eigen_rejects_zero_num_modes() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="num_modes must be >= 1"):
        ops.complex_eigen(0)


def test_complex_eigen_rejects_nonpositive_tol() -> None:
    ops = _mrh_ops()
    with pytest.raises(ValueError, match="tol must be > 0"):
        ops.complex_eigen(2, tol=0.0)


def test_complex_eigen_warns_on_declared_plain_handler() -> None:
    """Fork guide trap #4 — Plain + MP constraints yields wrong shapes;
    the bridge warns whenever Plain is declared. The warn fires before
    any live emitter is constructed (tol guard placed after it), so
    pin it via the tol ValueError exit."""
    ops = _mrh_ops()
    ops.constraints.Plain()
    with pytest.warns(UserWarning, match="distributing handler"):
        with pytest.raises(ValueError, match="tol must be > 0"):
            ops.complex_eigen(2, tol=-1.0)


def test_sweep_result_dataclasses_derive_magnitude_and_phase() -> None:
    from apeGmsh.opensees.analysis.modal import (
        FrequencyResponseResult,
        RandomResponseResult,
    )

    r = FrequencyResponseResult(
        freq=np.array([1.0, 2.0]),
        response=np.array([1.0 + 0.0j, 0.0 + 2.0j]),
    )
    np.testing.assert_allclose(r.magnitude, np.array([1.0, 2.0]))
    np.testing.assert_allclose(r.phase, np.array([0.0, np.pi / 2.0]))

    rr = RandomResponseResult(rms=0.5)
    assert rr.rms == 0.5
    assert rr.nu0 is None and rr.peak is None
