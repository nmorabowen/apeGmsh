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


def test_damping_channel_error_names_the_context() -> None:
    with pytest.raises(ValueError, match="apeSees.modal_response_history"):
        _damping_channel_args(
            damp=None, rayleigh=None, modal_damp=None,
            context="apeSees.modal_response_history",
        )
