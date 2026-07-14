"""
``ModalPropertiesResult`` — the return type of
:meth:`apeGmsh.opensees.apeSees.modal_properties`.

``modalProperties`` (upstream OpenSees, Petracca's
``DomainModalProperties``) is — like ``eigen`` — a one-shot domain
directive, not an Analysis primitive: no ``analysis <Type>`` chain, no
stepping, values returned directly. It rides a preceding ``eigen``
solve and computes participation factors, modal masses, and mass
ratios per mode and per global component. Modelled as a bridge method
(``apeSees.modal_properties``) that drives a :class:`LiveOpsEmitter`
end-to-end (``eigen`` + ``modalProperties -return``) and wraps the
returned dict in this dataclass.

The dict keys mirror the OpenSees ``printDict`` layout
(``DomainModalProperties.cpp``): per-mode series are suffixed with a
component token — ``MX`` / ``MY`` / ``MZ`` translational, ``RMX`` /
``RMY`` / ``RMZ`` rotational (2-D models expose ``MX`` / ``MY`` /
``RMZ`` only). Mass *ratios* are percentages (the C++ side scales by
100 before returning).

This module also hosts :func:`_damping_channel_args`, the shared
exactly-one-of validator for the fork's modal-response damping flags
(``-damp`` | ``-rayleigh`` | ``-modalDamp``) consumed by the ADR-44
drivers (fork ADR 44, ``LadrunoModalResponse``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..emitter.live import LiveOpsEmitter
    from ..node import Node


__all__ = [
    "FrequencyResponseResult",
    "ModalHistoryResult",
    "ModalPropertiesResult",
    "RandomResponseResult",
    "ResponseSpectrumResult",
    "SteadyStateResult",
]


def _damping_channel_args(
    *,
    damp: float | None,
    rayleigh: tuple[float, float] | None,
    modal_damp: Sequence[float] | None,
    context: str,
) -> tuple[float | str, ...]:
    """Render exactly one modal-response damping channel to flag args.

    The fork's modal-response commands (fork ADR 44) accept exactly one
    of three damping channels; this validates the exactly-one-of
    contract at the bridge (fail-loud, before anything is emitted) and
    returns the verbatim flag tail:

    * ``damp=xi``              → ``('-damp', xi)`` — one ratio, all modes
    * ``rayleigh=(a0, a1)``    → ``('-rayleigh', a0, a1)`` — per-mode
      ``ξ_a = a0/(2ω_a) + a1·ω_a/2``
    * ``modal_damp=[xi1, ..]`` → ``('-modalDamp', xi1, ..)`` — explicit
      per-mode ratios in absolute mode order

    Negative ratios are refused here (adversarial-review hardening):
    the fork refuses ``ξ < 0`` on four of the five family parsers, but
    the ``responseSpectrumAnalysis`` parser does NOT — and a mixed-sign
    ``-modalDamp`` list under CQC makes ``ρ_ij = √(ξ_i·ξ_j)`` NaN,
    which the fork's combination kernel silently collapses to a
    committed all-zero design displacement field. ``ξ = 0`` stays
    legal (the fork's own boundary).

    ``context`` names the calling surface in error messages
    (e.g. ``"apeSees.modal_response_history"``).
    """
    given = [
        name
        for name, value in (
            ("damp", damp),
            ("rayleigh", rayleigh),
            ("modal_damp", modal_damp),
        )
        if value is not None
    ]
    if len(given) != 1:
        raise ValueError(
            f"{context}: supply exactly one damping channel — damp= "
            "(one ratio for all modes), rayleigh=(a0, a1), or "
            f"modal_damp=[xi1, ..] — got {given or 'none'}."
        )
    if damp is not None:
        if float(damp) < 0.0:
            raise ValueError(
                f"{context}: damp must be >= 0, got {damp} (negative "
                "damping ratios silently zero the fork's CQC "
                "combination)."
            )
        return ("-damp", float(damp))
    if rayleigh is not None:
        a0, a1 = rayleigh
        return ("-rayleigh", float(a0), float(a1))
    assert modal_damp is not None
    factors = tuple(float(x) for x in modal_damp)
    if not factors:
        raise ValueError(
            f"{context}: modal_damp must carry at least one ratio."
        )
    if any(x < 0.0 for x in factors):
        raise ValueError(
            f"{context}: every modal_damp ratio must be >= 0, got "
            f"{list(factors)} (a mixed-sign list makes the fork's CQC "
            "cross-correlation NaN and silently zeros the combined "
            "field)."
        )
    return ("-modalDamp", *factors)


@dataclass(frozen=True, slots=True)
class ModalPropertiesResult:
    """Eigenvalues + modal properties from one ``eigen`` +
    ``modalProperties`` pair.

    Attributes
    ----------
    eigenvalues
        1-D ``np.ndarray`` of ``λ_i = ω_i²`` in OpenSees order.
    properties
        The raw ``modalProperties -return`` dict — per-mode series keyed
        ``partiFactor<C>`` / ``partiMass<C>`` / ``partiMassesCumu<C>`` /
        ``partiMassRatios<C>`` / ``partiMassRatiosCumu<C>`` for each
        component ``C``, plus ``totalMass`` / ``totalFreeMass`` /
        ``centerOfMass`` / ``eigenLambda`` / ``eigenOmega`` /
        ``eigenFrequency`` / ``eigenPeriod`` / ``domainSize``.

    Notes
    -----
    Same staleness contract as :class:`EigenResult`: the eigenvectors
    live in openseespy's domain state and :meth:`mode_shape` reads them
    lazily via the retained live emitter; a later driver call or
    ``wipe`` invalidates them without detection.

    **Basis caveat under ``unorm=True``** — ``modalProperties -unorm``
    rescales its own *local* eigenvector copy (per-mode factor
    ``1/max|v|``) before computing the participation factors, so the
    factors are in the displacement-normalized basis while
    :meth:`mode_shape` always returns the RAW domain eigenvector
    (``ops.nodeEigenvector`` is untouched by ``-unorm``).  ``Γ_a·φ_a``
    recovery products mixing the two accessors are therefore only
    scale-consistent under the default ``unorm=False``.
    """

    eigenvalues: np.ndarray
    properties: Mapping[str, Sequence[float]]

    # Implementation handle for lazy mode-shape access. Underscore-
    # prefixed; not part of the user-facing surface.
    _live: "LiveOpsEmitter"

    @property
    def omega(self) -> np.ndarray:
        """Natural circular frequencies ``ω_i = √λ_i`` (rad/s)."""
        return np.asarray(np.sqrt(self.eigenvalues))

    @property
    def freq(self) -> np.ndarray:
        """Natural frequencies ``f_i = ω_i / (2π)`` (Hz)."""
        return self.omega / (2.0 * np.pi)

    @property
    def periods(self) -> np.ndarray:
        """Natural periods ``T_i = 1 / f_i`` (s)."""
        return 1.0 / self.freq

    @property
    def total_mass(self) -> np.ndarray:
        """Total structure mass per component (ndf-long)."""
        return self._array("totalMass")

    @property
    def center_of_mass(self) -> np.ndarray:
        """Center of mass (ndm-long)."""
        return self._array("centerOfMass")

    def participation_factors(self, component: str) -> np.ndarray:
        """Per-mode participation factors for ``component``.

        ``component`` is an OpenSees component token: ``"MX"`` /
        ``"MY"`` / ``"MZ"`` translational, ``"RMX"`` / ``"RMY"`` /
        ``"RMZ"`` rotational (2-D models carry ``MX`` / ``MY`` /
        ``RMZ`` only).
        """
        return self._series("partiFactor", component)

    def mass_ratios(self, component: str) -> np.ndarray:
        """Per-mode effective modal mass ratios for ``component``,
        in **percent** of the total free mass (OpenSees pre-scales by
        100)."""
        return self._series("partiMassRatios", component)

    def cumulative_mass_ratios(self, component: str) -> np.ndarray:
        """Cumulative modal mass ratios for ``component``, in
        **percent** — the ASCE/NEC "90 % of the mass" check reads the
        last entry."""
        return self._series("partiMassRatiosCumu", component)

    def mode_shape(self, node: "int | Node", mode: int) -> np.ndarray:
        """Return the mode shape for ``node`` in ``mode`` (1-indexed).

        Same lazy ``ops.nodeEigenvector`` access as
        :meth:`EigenResult.mode_shape` — always the RAW domain
        eigenvector.  Under ``unorm=True`` this is a DIFFERENT scaling
        than the basis the participation factors were computed in (see
        the class docstring's basis caveat).
        """
        from ..node import Node as _Node  # local import — avoid cycle

        if isinstance(node, _Node):
            tag = int(node.tag)
        else:
            tag = int(node)
        values: Any = self._live.ops.nodeEigenvector(tag, int(mode))
        return np.asarray(values, dtype=np.float64)

    # -- internals --------------------------------------------------------

    def _array(self, key: str) -> np.ndarray:
        try:
            values = self.properties[key]
        except KeyError:
            raise KeyError(
                f"ModalPropertiesResult: key {key!r} not present — "
                f"available: {sorted(self.properties)}."
            ) from None
        return np.asarray(values, dtype=np.float64)

    def _series(self, prefix: str, component: str) -> np.ndarray:
        key = f"{prefix}{component}"
        if key not in self.properties:
            available = sorted(
                k[len(prefix):]
                for k in self.properties
                if k.startswith(prefix)
                and not k.startswith(f"{prefix}Cumu")
            )
            raise KeyError(
                f"ModalPropertiesResult: component {component!r} not "
                f"present for {prefix!r} — available components: "
                f"{available} (2-D models expose MX/MY/RMZ only)."
            )
        return self._array(key)


@dataclass(frozen=True, slots=True)
class ModalHistoryResult:
    """Return type of :meth:`apeSees.modal_response_history`.

    The transient history itself lands in the user's **recorders** —
    the fork commits one domain step per time station, so every
    recorder declared on the model captures the run exactly as in a
    direct integration. This result carries the mode basis the
    superposition used plus lazy final-station state readers.

    Same staleness contract as :class:`EigenResult`: the readers query
    the live domain; a later driver call or ``wipe`` invalidates them
    without detection.
    """

    eigenvalues: np.ndarray
    dt: float
    n_steps: int

    _live: "LiveOpsEmitter"

    @property
    def omega(self) -> np.ndarray:
        """Natural circular frequencies ``ω_i = √λ_i`` (rad/s)."""
        return np.asarray(np.sqrt(self.eigenvalues))

    @property
    def freq(self) -> np.ndarray:
        """Natural frequencies ``f_i = ω_i / (2π)`` (Hz)."""
        return self.omega / (2.0 * np.pi)

    def node_disp(self, node: "int | Node", dof: int) -> float:
        """Displacement at the **final committed station** for
        ``(node, dof)`` (1-based dof)."""
        return float(self._live.ops.nodeDisp(_node_tag(node), int(dof)))

    def node_vel(self, node: "int | Node", dof: int) -> float:
        """Velocity at the final committed station."""
        return float(self._live.ops.nodeVel(_node_tag(node), int(dof)))

    def node_accel(self, node: "int | Node", dof: int) -> float:
        """Acceleration at the final committed station."""
        return float(self._live.ops.nodeAccel(_node_tag(node), int(dof)))


@dataclass(frozen=True, slots=True)
class ResponseSpectrumResult:
    """Return type of :meth:`apeSees.response_spectrum_analysis`.

    The fork's ``-combine`` stage commits the **combined** nodal design
    displacement field to the domain; :meth:`node_disp` reads it.

    Combination is per-quantity and nonlinear — element forces / drifts
    must NOT be derived from these combined displacements; combine
    those quantities' own per-mode peaks instead (fork ADR 44 guide).

    Same staleness contract as :class:`EigenResult`.
    """

    eigenvalues: np.ndarray
    combine: str

    _live: "LiveOpsEmitter"

    @property
    def omega(self) -> np.ndarray:
        """Natural circular frequencies ``ω_i = √λ_i`` (rad/s)."""
        return np.asarray(np.sqrt(self.eigenvalues))

    @property
    def periods(self) -> np.ndarray:
        """Natural periods ``T_i = 2π / ω_i`` (s)."""
        return 2.0 * np.pi / self.omega

    def node_disp(self, node: "int | Node", dof: int) -> float:
        """Combined design displacement for ``(node, dof)``
        (1-based dof, always >= 0 for SRSS/CQC/ABS)."""
        return float(self._live.ops.nodeDisp(_node_tag(node), int(dof)))


def _node_tag(node: "int | Node") -> int:
    """Accept a plain tag or a ``Node`` handle (mirrors
    :meth:`EigenResult.mode_shape`)."""
    from ..node import Node as _Node  # local import — avoid cycle

    if isinstance(node, _Node):
        return int(node.tag)
    return int(node)


# ---------------------------------------------------------------------------
# Frequency-domain sweep results (ADR 0075 tier 2) — EAGER: the sweep
# values are fully returned by the fork command, so no ``_live``
# back-reference and no staleness caveat.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrequencyResponseResult:
    """Return type of :meth:`apeSees.frequency_response`.

    The complex FRF of one response DOF over the sweep grid.  Sign
    convention is ``e^{+iΩt}`` — the response lags 90° at resonance.
    For base excitation the response is **relative** to the moving
    base; for the ``load=`` channel it is absolute.
    """

    freq: np.ndarray
    """Sweep frequencies in Hz."""

    response: np.ndarray
    """Complex FRF values (same length as :attr:`freq`)."""

    @property
    def magnitude(self) -> np.ndarray:
        """``|H(f)|`` per sweep point."""
        return np.abs(self.response)

    @property
    def phase(self) -> np.ndarray:
        """Phase angle ``atan2(Im, Re)`` in radians."""
        return np.angle(self.response)


@dataclass(frozen=True, slots=True)
class SteadyStateResult:
    """Return type of :meth:`apeSees.steady_state_dynamics` — the
    steady-state harmonic response amplitude ``|response|`` per sweep
    frequency."""

    freq: np.ndarray
    magnitude: np.ndarray


@dataclass(frozen=True, slots=True)
class RandomResponseResult:
    """Return type of :meth:`apeSees.random_response`.

    ``rms`` is always present (``√m0``).  The spectral moments and the
    Davenport expected peak are ``None`` unless requested via
    ``stats=`` / ``duration=``.  ``peak`` is ``NaN`` when
    ``ν₀·T <= 1`` (the fork flags the estimate unreliable below
    ``ν₀·T < 2``).
    """

    rms: float
    nu0: float | None = None
    m0: float | None = None
    m2: float | None = None
    peak: float | None = None
