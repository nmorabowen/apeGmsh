"""Unit tests for the footfall FRF damping channel (no OpenSees run).

R-B finding 1: a zero ratio on any mode makes the modal denominator
``0 + 0j`` on the grid point ``grid_for`` places exactly at that mode,
and the evaluation went silently NaN.  Every channel must refuse it.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees.analysis.footfall_frf import _modal_damping_ratios

_OMEGA = 2.0 * np.pi * np.array([5.0, 12.0])


def test_uniform_zero_damping_is_refused() -> None:
    with pytest.raises(ValueError, match="damping ratio is zero on mode"):
        _modal_damping_ratios(
            omega=_OMEGA, damp=0.0, rayleigh=None, modal_damp=None, context="t",
        )


def test_modal_damp_with_one_zero_names_that_mode() -> None:
    with pytest.raises(ValueError, match=r"mode\(s\) \[2\]"):
        _modal_damping_ratios(
            omega=_OMEGA, damp=None, rayleigh=None, modal_damp=[0.02, 0.0],
            context="t",
        )


def test_rayleigh_zero_pair_is_refused() -> None:
    with pytest.raises(ValueError, match="damping ratio is zero"):
        _modal_damping_ratios(
            omega=_OMEGA, damp=None, rayleigh=(0.0, 0.0), modal_damp=None,
            context="t",
        )


def test_positive_channels_pass_through() -> None:
    out = _modal_damping_ratios(
        omega=_OMEGA, damp=0.03, rayleigh=None, modal_damp=None, context="t",
    )
    np.testing.assert_allclose(out, [0.03, 0.03])
