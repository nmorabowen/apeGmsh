"""
``ComplexEigenResult`` — the return type of
:meth:`apeGmsh.opensees.apeSees.complex_eigen`.

The fork's ``complexEigen`` (Ladruno ADR 46, ``LadrunoComplexEigen``)
answers the question real modes cannot: the **true per-mode damping
ratio** of a non-classically damped model (localized dashpots,
bearings, radiation damping). It projects the model's actual M and C
onto the retained real-mode basis (element-by-element ``getDamp()`` /
``getMass()`` — the exact C a transient analysis feels) and solves the
reduced quadratic pencil, returning a flat list of 7 numbers per
reported physical mode::

    [omega0, omegaD, zeta, Re(lambda), Im(lambda), kind, resid]

``kind``: 0 = underdamped (one entry per conjugate pair),
1 = overdamped, 2 = rigid.  ``resid = ||(λ²M̃ + λC̃ + K̃) z||`` is the
per-mode quality metric.

Complex (phased) mode shapes are recorded via the Node recorder
response types ``complexEigenRe<k>`` / ``complexEigenIm<k>`` (the
``raw=`` escape hatch on recorder declarations) — not carried here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


__all__ = ["ComplexEigenResult"]


@dataclass(frozen=True, slots=True)
class ComplexEigenResult:
    """Per-mode complex-modal quantities from one ``complexEigen`` call.

    All arrays are parallel, one entry per reported physical mode (an
    underdamped conjugate pair reports once).
    """

    omega0: np.ndarray
    """Undamped natural circular frequencies ``ω₀`` (rad/s)."""

    omega_d: np.ndarray
    """Damped circular frequencies ``ω_d`` (rad/s; 0 for overdamped)."""

    zeta: np.ndarray
    """True per-mode damping ratios ``ζ``."""

    lam: np.ndarray
    """Complex eigenvalues ``λ`` (the reported branch of each pair)."""

    kind: np.ndarray
    """Mode kind: 0 underdamped, 1 overdamped, 2 rigid (int8)."""

    resid: np.ndarray
    """Per-mode residual ``||(λ²M̃ + λC̃ + K̃) z||`` — quality metric."""

    @property
    def n_modes(self) -> int:
        """Number of reported physical modes."""
        return int(self.omega0.shape[0])

    @property
    def freq_d(self) -> np.ndarray:
        """Damped frequencies ``f_d = ω_d / (2π)`` (Hz)."""
        return self.omega_d / (2.0 * np.pi)

    @classmethod
    def from_flat(cls, values: Sequence[float]) -> "ComplexEigenResult":
        """Parse the fork's flat 7-per-mode list."""
        flat = np.asarray(values, dtype=np.float64)
        if flat.size % 7 != 0:
            raise ValueError(
                "ComplexEigenResult.from_flat: expected 7 values per "
                f"mode, got a flat list of length {flat.size}."
            )
        table = flat.reshape(-1, 7)
        return cls(
            omega0=table[:, 0].copy(),
            omega_d=table[:, 1].copy(),
            zeta=table[:, 2].copy(),
            lam=table[:, 3] + 1j * table[:, 4],
            kind=table[:, 5].astype(np.int8),
            resid=table[:, 6].copy(),
        )
