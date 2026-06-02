"""
``rayleigh_from_ratio`` — two-target Rayleigh coefficient fit (ADR 0053, D1).

Classic Rayleigh damping ``C = αM·M + βK·K`` realises a chosen modal
damping ratio ξ at exactly two circular frequencies ω_i, ω_j::

    α = 2ξ ω_i ω_j / (ω_i + ω_j)
    β = 2ξ / (ω_i + ω_j)

with ω = 2πf for a frequency f in Hz.  This helper performs that fit and
returns the four positional coefficients OpenSees' ``rayleigh`` command
takes — ``(alpha_m, beta_k, beta_k_init, beta_k_comm)`` — with β placed in
the stiffness slot named by ``stiffness``.

The default slot is ``initial`` (β → ``beta_k_init``, OpenSees ``betaK0``):
for nonlinear runs the current tangent stiffness can vanish or go negative
on softening and destabilise the integration, so stiffness-proportional
damping on the *initial* stiffness is the safe default (ADR 0053, grounded
in the fork reference ``12_damping_channels.md``).  ``current`` (``betaK``)
and ``committed`` (``betaKc``) remain explicit opt-ins.
"""
from __future__ import annotations

import math
from typing import Literal

__all__ = ["RayleighCoefficients", "Stiffness", "rayleigh_from_ratio"]

Stiffness = Literal["initial", "current", "committed"]

#: ``(alpha_m, beta_k, beta_k_init, beta_k_comm)`` — the positional order of
#: the OpenSees ``rayleigh $alphaM $betaK $betaK0 $betaKc`` command.
RayleighCoefficients = tuple[float, float, float, float]


def rayleigh_from_ratio(
    *,
    ratio: float,
    f_i: float,
    f_j: float,
    stiffness: Stiffness = "initial",
) -> RayleighCoefficients:
    """Fit Rayleigh ``(alpha_m, beta_k, beta_k_init, beta_k_comm)`` to a
    target damping ratio ``ratio`` at two frequencies ``f_i``, ``f_j`` (Hz).

    β lands in the stiffness slot named by ``stiffness`` — ``initial``
    (default) → ``beta_k_init``, ``current`` → ``beta_k``, ``committed`` →
    ``beta_k_comm``; the other two stiffness slots are ``0.0``.  See the
    module docstring for the formulae and why ``initial`` is the default.

    Raises
    ------
    ValueError
        If either frequency is non-positive, the two frequencies are equal
        (the two-point fit is singular), or ``stiffness`` is not one of
        ``initial`` / ``current`` / ``committed``.
    """
    if f_i <= 0.0 or f_j <= 0.0:
        raise ValueError(
            "rayleigh_from_ratio: frequencies must be positive (Hz), "
            f"got f_i={f_i}, f_j={f_j}."
        )
    if f_i == f_j:
        raise ValueError(
            "rayleigh_from_ratio: f_i and f_j must differ — a single "
            f"frequency cannot pin both Rayleigh terms (got {f_i})."
        )
    w_i = 2.0 * math.pi * f_i
    w_j = 2.0 * math.pi * f_j
    alpha = 2.0 * ratio * w_i * w_j / (w_i + w_j)
    beta = 2.0 * ratio / (w_i + w_j)
    if stiffness == "initial":
        return (alpha, 0.0, beta, 0.0)
    if stiffness == "current":
        return (alpha, beta, 0.0, 0.0)
    if stiffness == "committed":
        return (alpha, 0.0, 0.0, beta)
    raise ValueError(
        "rayleigh_from_ratio: stiffness must be 'initial', 'current', or "
        f"'committed', got {stiffness!r}."
    )
