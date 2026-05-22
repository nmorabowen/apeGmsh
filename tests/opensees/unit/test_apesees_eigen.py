"""Unit tests for :meth:`apeSees.eigen` and :class:`EigenResult`.

The live solve itself is exercised in
``tests/opensees/live/test_eigen_cantilever.py`` (gated by the ``live``
marker — requires openseespy). These tests cover the bridge-side
validation and the pure-Python derived properties on
:class:`EigenResult`.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.eigen import EigenResult

from tests.opensees.fixtures.fem_stub import make_two_node_beam


def test_eigen_rejects_zero_num_modes() -> None:
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="num_modes must be >= 1"):
        ops.eigen(0)


def test_eigen_rejects_negative_num_modes() -> None:
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="num_modes must be >= 1"):
        ops.eigen(-2)


def test_eigen_result_omega_equals_sqrt_eigenvalues() -> None:
    # λ = ω² so ω = √λ.
    eigenvalues = np.array([1.0, 4.0, 9.0])
    r = EigenResult(eigenvalues=eigenvalues, _live=cast(object, None))  # type: ignore[arg-type]
    np.testing.assert_allclose(r.omega, np.array([1.0, 2.0, 3.0]))


def test_eigen_result_freq_equals_omega_over_two_pi() -> None:
    eigenvalues = np.array([(2.0 * np.pi) ** 2])  # ω = 2π → f = 1 Hz
    r = EigenResult(eigenvalues=eigenvalues, _live=cast(object, None))  # type: ignore[arg-type]
    np.testing.assert_allclose(r.freq, np.array([1.0]))


def test_eigen_result_periods_equal_inverse_freq() -> None:
    eigenvalues = np.array([(2.0 * np.pi) ** 2])  # f = 1 Hz → T = 1 s
    r = EigenResult(eigenvalues=eigenvalues, _live=cast(object, None))  # type: ignore[arg-type]
    np.testing.assert_allclose(r.periods, np.array([1.0]))


def test_eigen_result_is_frozen() -> None:
    """Per the dataclass contract — EigenResult must not be mutated."""
    r = EigenResult(
        eigenvalues=np.array([1.0]), _live=cast(object, None),  # type: ignore[arg-type]
    )
    with pytest.raises((AttributeError, TypeError)):
        r.eigenvalues = np.array([2.0])  # type: ignore[misc]
