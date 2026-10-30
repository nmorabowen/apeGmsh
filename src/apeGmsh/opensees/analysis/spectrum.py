"""
Per-mode response spectrum about the gravity state, and its combinations
(ADR 0124 D4–D7).

:meth:`apeGmsh.opensees.apeSees.spectrum_after_gravity` returns a
:class:`SpectrumModalResult`: the per-mode value of every response
extractor, for every :class:`DesignSpectrum`, as an increment about the
gravity state. This module holds the spectrum, three built-in extractors,
the result, and the combination kernels:

* :func:`cqc_correlation` — the Der Kiureghian (1981) ``ρ_ij`` with
  per-mode damping ratios;
* :func:`combine_modes` — CQC or SRSS over the modes of one quantity;
* :func:`combine_directions` — 100/30 or SRSS over the directions, from
  combined or per-mode values, optionally as every signed 100/30 case
  (:class:`DirectionalCases`);
* :func:`modal_cancellation` — why a combined quantity is small: each
  mode's signed term and contribution, and the cancellation ratio
  (:class:`ModalCancellation`);
* :func:`seismic_envelope` — ``S ± f·E``, the envelope over the sign of the
  seismic term, for a gravity combination ``S`` such as ``1.0 D + L_exp``.

An extractor is any callable ``extractor(ops) -> 1-D float array`` of fixed
length, optionally with a ``labels`` attribute. Every combination works on
one quantity's own per-mode values; none is derived from combined
displacements. A derived linear quantity (a drift, a sum of reactions, a
storey force) is formed per mode first (:meth:`SpectrumModalResult.linear`,
:meth:`SpectrumModalResult.drifts`, :meth:`SpectrumModalResult.storey_forces`)
and then combined (:meth:`SpectrumModalResult.combine_values`).
"""
from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .gravity_modal import GravityModalResult


__all__ = [
    "DesignSpectrum",
    "DirectionalCases",
    "ElementResponse",
    "InertiaBalanceWarning",
    "ModalCancellation",
    "NodeDisp",
    "NodeReaction",
    "SpectrumModalResult",
    "StoreyForces",
    "combine_directions",
    "combine_modes",
    "cqc_correlation",
    "modal_cancellation",
    "seismic_envelope",
]

#: ``extractor(ops) -> 1-D float array`` of fixed length.
ResponseExtractor = Callable[[Any], "npt.ArrayLike"]

#: The name the driver files the total base reaction under.
BASE_SHEAR = "base_shear"

#: The relative tolerance of the per-mode inertia / base reaction balance.
INERTIA_BALANCE_RTOL = 1e-3


class InertiaBalanceWarning(RuntimeWarning):
    """A mode's total inertia force ``Σ m φ Γ Sa`` does not balance its
    total base reaction (:meth:`SpectrumModalResult.inertia_balance`)."""


# ---------------------------------------------------------------------------
# The design spectrum
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DesignSpectrum:
    """A design spectrum ``Sa(T)`` acting along one global direction.

    ``DesignSpectrum(periods, accels, direction=1, scale=g)`` takes a table:
    ``periods`` non-negative and strictly increasing (a leading ``T = 0``
    PGA anchor is legal), ``accels`` non-negative; ``Sa(T)`` is interpolated
    linearly and clamped at both ends, as the fork's
    ``ResponseSpectrumAnalysis::getSa`` does.
    ``DesignSpectrum(spec, direction=1, scale=g)`` takes a function instead:
    any object with a vectorised ``spec.at(T)`` (a codeSpectra ``Spectrum``,
    Sa in g, satisfies it; codeSpectra is never imported) or a plain
    vectorised callable ``sa(T)``. Its own rules apply; nothing is tabulated.

    ``direction`` is the global translational direction, 1-based.
    ``scale`` multiplies every ``Sa`` (the g-to-model-units factor times any
    ``I/R``); it is applied here, the command's ``-scale`` is never used.
    """

    periods: tuple[float, ...]
    accels: tuple[float, ...]
    direction: int
    scale: float = 1.0
    function: "Callable[[np.ndarray], npt.ArrayLike] | None" = None

    def __init__(
        self,
        periods: "Sequence[float] | Callable[[np.ndarray], npt.ArrayLike] | Any",
        accels: "Sequence[float] | None" = None,
        *,
        direction: int,
        scale: float = 1.0,
    ) -> None:
        function = _sa_function(periods)
        tn: tuple[float, ...] = ()
        sa: tuple[float, ...] = ()
        if function is not None:
            if accels is not None:
                raise ValueError(
                    "DesignSpectrum: give a table (periods, accels) or a "
                    "spectrum function, not both."
                )
        else:
            tn = tuple(float(t) for t in cast("Sequence[float]", periods))
            sa = tuple(float(a) for a in (accels or ()))
            if not tn or len(tn) != len(sa):
                raise ValueError(
                    "DesignSpectrum: periods and accels must be equal-length, "
                    f"non-empty sequences; got {len(tn)} and {len(sa)}."
                )
            if any(t < 0.0 for t in tn) or any(
                    b <= a for a, b in zip(tn, tn[1:])):
                raise ValueError(
                    "DesignSpectrum: periods must be non-negative and "
                    "strictly increasing."
                )
            if any(a < 0.0 or not math.isfinite(a) for a in sa):
                raise ValueError(
                    "DesignSpectrum: accels must be finite and non-negative."
                )
        if int(direction) < 1:
            raise ValueError(
                f"DesignSpectrum: direction is 1-based, got {direction!r}."
            )
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(
                f"DesignSpectrum: scale must be finite and > 0, got {scale!r}."
            )
        object.__setattr__(self, "periods", tn)
        object.__setattr__(self, "accels", sa)
        object.__setattr__(self, "direction", int(direction))
        object.__setattr__(self, "scale", float(scale))
        object.__setattr__(self, "function", function)

    def sa(self, period: "npt.ArrayLike") -> np.ndarray:
        """Scaled ``Sa`` at ``period`` (scalar or array)."""
        t = np.asarray(period, dtype=np.float64)
        if self.function is None:
            return np.asarray(
                self.scale * np.interp(t, self.periods, self.accels))
        raw = np.asarray(self.function(t), dtype=np.float64)
        if raw.shape != t.shape:
            raise ValueError(
                f"DesignSpectrum: the Sa callable returned shape {raw.shape} "
                f"for periods of shape {t.shape}; it must be vectorised."
            )
        if np.any(raw < 0.0) or not np.all(np.isfinite(raw)):
            raise ValueError(
                "DesignSpectrum: the Sa callable returned a negative or "
                "non-finite value."
            )
        return np.asarray(self.scale * raw)


def _sa_function(
    obj: Any,
) -> "Callable[[np.ndarray], npt.ArrayLike] | None":
    """The ``Sa(T)`` function of a spectrum object: its ``.at`` when it has
    one, the object itself when it is callable, ``None`` for a table."""
    at = getattr(obj, "at", None)
    if callable(at):
        return cast("Callable[[np.ndarray], npt.ArrayLike]", at)
    if callable(obj):
        return cast("Callable[[np.ndarray], npt.ArrayLike]", obj)
    return None


# ---------------------------------------------------------------------------
# Built-in extractors
# ---------------------------------------------------------------------------

def _tags(values: Sequence[int], what: str) -> tuple[int, ...]:
    tags = tuple(int(v) for v in values)
    if not tags:
        raise ValueError(f"{what}: give at least one tag.")
    return tags


@dataclass(frozen=True, slots=True)
class NodeDisp:
    """Nodal displacements, node-major: ``[(n1, d1), (n1, d2), …]``.

    Reads the TRIAL displacement, ``nodeDisp`` (committed) plus
    ``nodeResponse(n, d, 4)`` (the increment trial − committed): after a
    solve the two agree, and on the fork's ``-increment`` path the trial
    state is gravity + mode while the committed one stays at gravity.
    (``nodeResponse(n, d, 1)`` is the committed displacement, on stock and
    on the fork.)
    """

    nodes: tuple[int, ...]
    dofs: tuple[int, ...]

    def __init__(self, nodes: Sequence[int], dofs: Sequence[int]) -> None:
        object.__setattr__(self, "nodes", _tags(nodes, "NodeDisp"))
        object.__setattr__(self, "dofs", _tags(dofs, "NodeDisp"))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f"node{n}.u{d}" for n in self.nodes for d in self.dofs)

    def __call__(self, ops: Any) -> np.ndarray:
        return np.asarray(
            [float(ops.nodeDisp(n, d)) + float(ops.nodeResponse(n, d, 4))
             for n in self.nodes for d in self.dofs],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class NodeReaction:
    """Nodal reactions, node-major; issues ``reactions`` first."""

    nodes: tuple[int, ...]
    dofs: tuple[int, ...]

    def __init__(self, nodes: Sequence[int], dofs: Sequence[int]) -> None:
        object.__setattr__(self, "nodes", _tags(nodes, "NodeReaction"))
        object.__setattr__(self, "dofs", _tags(dofs, "NodeReaction"))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f"node{n}.R{d}" for n in self.nodes for d in self.dofs)

    def __call__(self, ops: Any) -> np.ndarray:
        ops.reactions()
        return np.asarray(
            [ops.nodeReaction(n, d) for n in self.nodes for d in self.dofs],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class ElementResponse:
    """``eleResponse(tag, *response)`` of each element, concatenated in
    element order (for example ``response="localForce"``)."""

    elements: tuple[int, ...]
    response: tuple[str, ...]

    def __init__(
        self, elements: Sequence[int], response: "str | Sequence[str]",
    ) -> None:
        resp = (response,) if isinstance(response, str) else tuple(response)
        if not resp:
            raise ValueError("ElementResponse: response must name a response.")
        object.__setattr__(self, "elements", _tags(elements, "ElementResponse"))
        object.__setattr__(self, "response", tuple(str(r) for r in resp))

    def __call__(self, ops: Any) -> np.ndarray:
        parts = []
        for e in self.elements:
            values = ops.eleResponse(e, *self.response)
            arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
            if arr.size == 0:
                raise ValueError(
                    f"ElementResponse: eleResponse({e}, "
                    f"{', '.join(self.response)}) returned nothing; the "
                    "element does not answer that response."
                )
            parts.append(arr.ravel())
        return np.concatenate(parts)


@dataclass(frozen=True, slots=True)
class _TotalReaction:
    """The driver's own extractor: the sum of every fixed node's reaction
    along each translational direction (``ndm`` values)."""

    ndm: int

    def __call__(self, ops: Any) -> np.ndarray:
        ops.reactions()
        total = np.zeros(self.ndm)
        for n in ops.getFixedNodes():
            r = np.asarray(ops.nodeReaction(int(n)), dtype=np.float64)
            k = min(self.ndm, r.size)
            total[:k] += r[:k]
        return total


# ---------------------------------------------------------------------------
# Combination kernels
# ---------------------------------------------------------------------------

def cqc_correlation(
    omega: "npt.ArrayLike", damping: "float | npt.ArrayLike",
) -> np.ndarray:
    """Der Kiureghian (1981) modal correlation matrix ``ρ_ij``.

    ``ρ_ij = 8 sqrt(ξi ξj)(ξi + r ξj) r^1.5 /
    [(1 - r²)² + 4 ξi ξj r (1 + r²) + 4 (ξi² + ξj²) r²]``, ``r = ωj/ωi``,
    ``ρ_ii = 1``. ``damping`` is one ratio for every mode or one per mode;
    ratios must be ``>= 0``.
    """
    w = np.asarray(omega, dtype=np.float64).ravel()
    xi = np.broadcast_to(
        np.asarray(damping, dtype=np.float64), w.shape).astype(np.float64)
    if np.any(xi < 0.0) or not np.all(np.isfinite(xi)):
        raise ValueError("cqc_correlation: damping ratios must be >= 0.")
    if np.any(w <= 0.0):
        raise ValueError("cqc_correlation: every omega must be > 0.")
    r = w[None, :] / w[:, None]
    xi_i, xi_j = xi[:, None], xi[None, :]
    num = 8.0 * np.sqrt(xi_i * xi_j) * (xi_i + r * xi_j) * r ** 1.5
    den = ((1.0 - r * r) ** 2 + 4.0 * xi_i * xi_j * r * (1.0 + r * r)
           + 4.0 * (xi_i ** 2 + xi_j ** 2) * r * r)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(den > 0.0, num / den, 1.0)
    np.fill_diagonal(rho, 1.0)
    return np.asarray(rho)


def _modal_terms(
    r: np.ndarray,
    omega: "npt.ArrayLike | None",
    *,
    sa_periods: "npt.ArrayLike | None" = None,
    damping: "float | npt.ArrayLike | None" = None,
) -> "tuple[np.ndarray, npt.ArrayLike | None, float | npt.ArrayLike | None]":
    """The terms a combination rule sees, ``(r, omega, damping)``.

    ``sa_periods`` is one value per mode: ``0`` for a mode with no Sa-period
    override, else the period ``T`` its Sa was read at
    (``residualModes -period T``). The modes sharing one ``T`` are one
    rigid mode (ADR 213 D3b, ASCE 7-22 §12.9.1.1), as in the fork's
    ``responseSpectrumAnalysis -combine``. Their values are summed
    algebraically into one term, before any rule, in the slot of the
    group's first mode, at ``ω = 2π/T`` and with that mode's damping. An
    overridden mode alone in its group keeps its slot at ``ω = 2π/T``. With
    no override (or ``sa_periods=None``) every mode is its own term.
    """
    if sa_periods is None:
        return r, omega, damping
    t = np.asarray(sa_periods, dtype=np.float64).ravel()
    n = r.shape[0]
    if t.size != n:
        raise ValueError(
            f"combine_modes: {t.size} Sa periods for {n} modes of values.")
    over = t > 0.0
    if not np.any(over):
        return r, omega, damping
    slot = np.empty(n, dtype=np.intp)
    keep: list[int] = []
    group: dict[float, int] = {}
    for a in range(n):
        if over[a] and float(t[a]) in group:
            slot[a] = group[float(t[a])]
            continue
        slot[a] = len(keep)
        if over[a]:
            group[float(t[a])] = len(keep)
        keep.append(a)
    terms = np.zeros((len(keep),) + r.shape[1:])
    np.add.at(terms, slot, r)
    idx = np.asarray(keep)
    w_out: "np.ndarray | None" = None
    if omega is not None:
        w_all = np.asarray(omega, dtype=np.float64).ravel()
        if w_all.size != n:
            raise ValueError(
                f"combine_modes: {w_all.size} omegas for {n} modes of values.")
        w_out = w_all[idx].copy()
        w_out[over[idx]] = 2.0 * np.pi / t[idx][over[idx]]
    xi = damping
    if damping is not None and np.ndim(damping) > 0:
        xi = np.asarray(damping, dtype=np.float64).ravel()[idx]
    return terms, w_out, xi


def combine_modes(
    per_mode: "npt.ArrayLike",
    *,
    rule: str = "CQC",
    omega: "npt.ArrayLike | None" = None,
    damping: "float | npt.ArrayLike | None" = None,
    sa_periods: "npt.ArrayLike | None" = None,
) -> np.ndarray:
    """Combine the per-mode values of one quantity over the mode axis.

    ``per_mode`` has the modes on axis 0 and holds the SIGNED per-mode
    values of any linear quantity: a displacement, a reaction component, a
    drift, a storey force, an element force. The signs matter to CQC: two
    modes of opposite sign and correlation ``ρ_ij`` give the cross term
    ``2 ρ_ij r_i r_j < 0``. ``rule`` is ``"CQC"`` (needs ``omega`` and
    ``damping``; ``ρ_ij`` is :func:`cqc_correlation`, the formula of the
    fork's ``LadrunoModalCombination::rhoCQC``) or ``"SRSS"``.
    ``sa_periods`` (one per mode, ``0`` for none) marks the modes whose Sa
    was read at an override period: the modes sharing one are combined as
    one rigid mode (:func:`_modal_terms`). Returns the non-negative combined
    value, shape ``per_mode.shape[1:]``.
    """
    r = np.asarray(per_mode, dtype=np.float64)
    if r.ndim < 1:
        raise ValueError("combine_modes: per_mode needs a mode axis.")
    r, omega, damping = _modal_terms(
        r, omega, sa_periods=sa_periods, damping=damping)
    if rule == "SRSS":
        return np.asarray(np.sqrt(np.sum(r * r, axis=0)))
    if rule == "CQC":
        rho = _rho(r.shape[0], omega, damping, "combine_modes")
        flat = r.reshape(r.shape[0], -1)
        s = np.einsum("i...,ij,j...->...", flat, rho, flat)
        return np.asarray(np.sqrt(np.clip(s, 0.0, None)).reshape(r.shape[1:]))
    raise ValueError(
        f"combine_modes: rule must be 'CQC' or 'SRSS', got {rule!r}."
    )


def _rho(
    n_modes: int,
    omega: "npt.ArrayLike | None",
    damping: "float | npt.ArrayLike | None",
    what: str,
) -> np.ndarray:
    """``ρ_ij`` for ``n_modes`` modes, refused when CQC lacks an input."""
    if omega is None or damping is None:
        raise ValueError(f"{what}: CQC needs omega= and damping=.")
    rho = cqc_correlation(omega, damping)
    if rho.shape[0] != n_modes:
        raise ValueError(
            f"{what}: {n_modes} modes of values but {rho.shape[0]} omegas."
        )
    return rho


@dataclass(frozen=True, slots=True)
class DirectionalCases:
    """Every signed 100/30 case of one quantity (:func:`combine_directions`
    with ``permutations=True``).

    Attributes
    ----------
    factors
        ``(n_cases, n_directions)``: the factor on each direction's combined
        value in each case, ``±1`` on the leading direction and ``±0.3`` on
        the others. Case order: leading direction first (``0, 1, …``), then
        the sign patterns in :func:`itertools.product` order of
        ``(+1, -1)``. A design check takes every case, since a combined
        per-direction value has lost its sign.
    values
        ``(n_cases, …)``: ``Σ_d factors[c, d] E_d``, ``E_d`` the combined
        (non-negative) value of direction ``d``.
    """

    factors: np.ndarray
    values: np.ndarray

    @property
    def leading(self) -> np.ndarray:
        """``(n_cases,)``: the direction (0-based) that carries 100 %."""
        return np.asarray(np.argmax(np.abs(self.factors), axis=1))

    @property
    def envelope(self) -> np.ndarray:
        """The largest case value: the scalar ``"100-30"`` combination."""
        return np.asarray(np.max(self.values, axis=0))


@overload
def combine_directions(
    per_direction: "npt.ArrayLike", *, method: str = ...,
    rule: "str | None" = ..., omega: "npt.ArrayLike | None" = ...,
    damping: "float | npt.ArrayLike | None" = ...,
    permutations: Literal[False] = ...,
    sa_periods: "npt.ArrayLike | None" = ...,
) -> np.ndarray: ...


@overload
def combine_directions(
    per_direction: "npt.ArrayLike", *, method: str = ...,
    rule: "str | None" = ..., omega: "npt.ArrayLike | None" = ...,
    damping: "float | npt.ArrayLike | None" = ...,
    permutations: Literal[True],
    sa_periods: "npt.ArrayLike | None" = ...,
) -> DirectionalCases: ...


@overload
def combine_directions(
    per_direction: "npt.ArrayLike", *, method: str = ...,
    rule: "str | None" = ..., omega: "npt.ArrayLike | None" = ...,
    damping: "float | npt.ArrayLike | None" = ...,
    permutations: bool = ...,
    sa_periods: "npt.ArrayLike | None" = ...,
) -> "np.ndarray | DirectionalCases": ...


def combine_directions(
    per_direction: "npt.ArrayLike",
    *,
    method: str = "100-30",
    rule: "str | None" = None,
    omega: "npt.ArrayLike | None" = None,
    damping: "float | npt.ArrayLike | None" = None,
    permutations: bool = False,
    sa_periods: "npt.ArrayLike | None" = None,
) -> "np.ndarray | DirectionalCases":
    """Combine one quantity's per-direction values (axis 0).

    ``"100-30"``: the largest of ``|E_a| + 0.3 Σ_{b≠a} |E_b|`` over which
    direction ``a`` carries the 100 %. ``"SRSS"``: ``sqrt(Σ E_d²)``.

    With ``rule`` (``"CQC"`` or ``"SRSS"``), ``per_direction`` holds the
    per-mode sets instead, shape ``(n_directions, n_modes, …)``: each
    direction is first combined over its modes by :func:`combine_modes`
    (``omega``, ``damping`` and ``sa_periods`` as there: pass a result's
    :attr:`SpectrumModalResult.sa_periods` so its rigid term is one mode). With ``permutations=True``
    (``"100-30"`` only) the result is the :class:`DirectionalCases` of every
    signed case, ``n_directions · 2^n_directions`` of them, whose
    :attr:`~DirectionalCases.envelope` is the scalar ``"100-30"`` value.
    CQC3 is not offered.
    """
    raw = np.asarray(per_direction, dtype=np.float64)
    if raw.ndim < 1 or raw.shape[0] < 1:
        raise ValueError("combine_directions: needs a direction axis.")
    if method not in ("100-30", "SRSS"):
        raise ValueError(
            f"combine_directions: method must be '100-30' or 'SRSS', got "
            f"{method!r}."
        )
    if rule is not None:
        if raw.ndim < 2:
            raise ValueError(
                "combine_directions: with rule= the values need a direction "
                "and a mode axis, (n_directions, n_modes, ...)."
            )
        raw = np.stack([combine_modes(v, rule=rule, omega=omega,
                                      damping=damping, sa_periods=sa_periods)
                        for v in raw])
    e = np.abs(raw)
    if permutations:
        if method != "100-30":
            raise ValueError(
                "combine_directions: permutations= gives the signed 100/30 "
                "cases; SRSS of directions has one value."
            )
        n = e.shape[0]
        factors = np.asarray([
            [s * (1.0 if d == lead else 0.3) for d, s in enumerate(signs)]
            for lead in range(n)
            for signs in itertools.product((1.0, -1.0), repeat=n)
        ])
        values = np.tensordot(factors, e, axes=(1, 0))
        return DirectionalCases(factors=factors, values=np.asarray(values))
    if method == "SRSS":
        return np.asarray(np.sqrt(np.sum(e * e, axis=0)))
    total = np.sum(e, axis=0)
    return np.asarray(np.max(e + 0.3 * (total - e), axis=0))


@dataclass(frozen=True, slots=True)
class ModalCancellation:
    """Why one combined quantity is what it is (:func:`modal_cancellation`).

    Attributes
    ----------
    terms
        ``(n_modes,)``: each mode's signed value ``r_i``.
    contributions
        ``(n_modes,)``: each mode's signed share of :attr:`combined`,
        ``r_i Σ_j ρ_ij r_j / combined`` (``r_i² / combined`` under SRSS);
        they sum to :attr:`combined`. A negative share is a mode that
        cancels the others.
    combined
        The combined value, as :func:`combine_modes` gives it.
    abs_sum
        ``Σ |r_i|``: the value if no term cancelled or was discounted.
    ratio
        The cancellation ratio ``combined / Σ |r_i|`` (``0`` when every
        term is zero). A ratio ``≪ 1`` marks a quantity whose modal terms
        cancel: its value rests on their signs and correlations, so a small
        change in one mode (its period, its principal-axis angle) moves it
        a lot.
    angles
        ``(n_modes,)``: each mode's principal-axis angle in degrees,
        ``atan2(Γ_b, Γ_a)`` of its participation factors along the two plan
        directions modulo 180° (the sign of an eigenvector is arbitrary),
        or ``None`` when not known.
    rule
        ``"CQC"`` or ``"SRSS"``.
    """

    terms: np.ndarray
    contributions: np.ndarray
    combined: float
    abs_sum: float
    ratio: float
    angles: "np.ndarray | None"
    rule: str


def modal_cancellation(
    per_mode: "npt.ArrayLike",
    *,
    rule: str = "CQC",
    omega: "npt.ArrayLike | None" = None,
    damping: "float | npt.ArrayLike | None" = None,
    angles: "npt.ArrayLike | None" = None,
) -> ModalCancellation:
    """Diagnose the combination of ONE scalar quantity's per-mode values.

    ``per_mode`` is ``(n_modes,)`` signed values; ``rule``, ``omega`` and
    ``damping`` as in :func:`combine_modes`. ``angles`` (``(n_modes,)``
    degrees) is carried into the report beside the contributions;
    :meth:`SpectrumModalResult.cancellation` fills it in from the
    participation factors. Returns a :class:`ModalCancellation`; nothing
    is printed.
    """
    r = np.asarray(per_mode, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(
            "modal_cancellation: per_mode is one quantity, (n_modes,); got "
            f"shape {r.shape}."
        )
    if rule == "SRSS":
        rho = np.eye(r.size)
    elif rule == "CQC":
        rho = _rho(r.size, omega, damping, "modal_cancellation")
    else:
        raise ValueError(
            f"modal_cancellation: rule must be 'CQC' or 'SRSS', got {rule!r}."
        )
    ang = None
    if angles is not None:
        ang = np.asarray(angles, dtype=np.float64).ravel()
        if ang.shape != r.shape:
            raise ValueError(
                f"modal_cancellation: {ang.size} angles for {r.size} modes.")
    combined = float(combine_modes(r, rule=rule, omega=omega, damping=damping))
    shares = r * (rho @ r)
    abs_sum = float(np.sum(np.abs(r)))
    return ModalCancellation(
        terms=r.copy(),
        contributions=(shares / combined if combined > 0.0
                       else np.zeros_like(r)),
        combined=combined,
        abs_sum=abs_sum,
        ratio=combined / abs_sum if abs_sum > 0.0 else 0.0,
        angles=ang,
        rule=rule,
    )


def seismic_envelope(
    static: "npt.ArrayLike",
    seismic: "npt.ArrayLike",
    *,
    factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(max, min)`` of ``S + f·E`` and ``S - f·E``.

    ``static`` is a gravity combination already summed by the caller (for
    example ``1.0·D + L_exp``), ``seismic`` a combined (non-negative)
    spectrum response.
    """
    s = np.asarray(static, dtype=np.float64)
    e = np.asarray(seismic, dtype=np.float64)
    if s.shape != e.shape:
        raise ValueError(
            f"seismic_envelope: static {s.shape} and seismic {e.shape} "
            "differ in shape."
        )
    up, lo = s + factor * e, s - factor * e
    return np.maximum(up, lo), np.minimum(up, lo)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StoreyForces:
    """Per-mode storey shears and moments from the modal inertia
    (:meth:`SpectrumModalResult.storey_forces`).

    Attributes
    ----------
    elevations
        ``(n_cuts,)``: the cut elevations, as given.
    shear
        ``(n_spectra, n_modes, n_cuts, ndm)``: the inertia above each cut,
        every translational component, per mode.
    moment
        ``(n_spectra, n_modes, n_cuts, n_rot)``: its moment about the cut
        point (``n_rot`` = 1 in 2-D, 3 in 3-D), rotational inertia
        included.

    Both are signed per-mode values: combine them with
    :meth:`SpectrumModalResult.combine_values`.
    """

    elevations: np.ndarray
    shear: np.ndarray
    moment: np.ndarray


@dataclass(frozen=True, slots=True)
class SpectrumModalResult:
    """Per-mode spectrum responses about the gravity state.

    Attributes
    ----------
    modal
        The :class:`~apeGmsh.opensees.analysis.gravity_modal.GravityModalResult`
        the responses rest on (modes, participation, gravity state).
    spectra
        The spectra, in the order of the spectrum axis below.
    gravity_values
        Extractor name -> its value at the gravity state.
    per_mode
        Extractor name -> ``(n_spectra, n_modes, n_values)`` array: the
        increment of each value in each mode under each spectrum. Holds
        ``"base_shear"`` too: the total support reaction per translational
        direction (``ndm`` values), which :meth:`base_shear` combines.
    labels
        Extractor name -> its ``labels`` (``None`` when it has none).
    restore_drift
        ``(disp, reaction)`` relative drift of the domain from the
        reference state once every mode had run.
    inertia
        ``(n_spectra, n_modes, n_nodes, ndf)``: the modal inertia force
        ``m φ Γ Sa`` at each massed node, per mode and spectrum (the load
        whose static response is the mode's response), or ``None``.
        :meth:`storey_forces` sums it; :meth:`inertia_balance` checks it
        against the base reaction.
    inertia_nodes, inertia_coords
        The massed nodes' tags and their ``(n_nodes, ndm)`` coordinates,
        in the order of :attr:`inertia`'s node axis.
    sa_periods
        ``(n_modes,)``: the period each mode's Sa was read at when it is not
        the mode's own, ``0`` elsewhere, or ``None`` (no override). A
        ``residual_period=T`` gives ``T`` on every residual pseudo-mode;
        every combination here treats the modes sharing a ``T`` as one
        rigid mode at ``ω = 2π/T`` (ADR 213 D3b), as the fork's RSA does.
    """

    modal: "GravityModalResult"
    spectra: tuple[DesignSpectrum, ...]
    gravity_values: Mapping[str, np.ndarray]
    per_mode: Mapping[str, np.ndarray]
    labels: Mapping[str, "tuple[str, ...] | None"]
    restore_drift: tuple[float, float]
    inertia: "np.ndarray | None" = None
    inertia_nodes: tuple[int, ...] = ()
    inertia_coords: "np.ndarray | None" = None
    sa_periods: "np.ndarray | None" = None

    @property
    def omega(self) -> np.ndarray:
        """Circular frequencies of the modes (rad/s)."""
        return self.modal.modal.omega

    @property
    def periods(self) -> np.ndarray:
        """Periods of the modes (s): ``2π/√λ``. For a Ritz vector or a
        residual pseudo-mode that is its Rayleigh period, not a natural
        period (see :attr:`kinds`)."""
        return self.modal.periods

    @property
    def kinds(self) -> tuple[str, ...]:
        """Each mode's kind: ``"eigen"``, ``"residual"`` or ``"ritz"``
        (:attr:`~apeGmsh.opensees.analysis.modal.ModalPropertiesResult.kinds`)."""
        return self.modal.modal.kinds

    def modes(self, name: str, spectrum: int = 0) -> np.ndarray:
        """``(n_modes, n_values)`` per-mode values of ``name`` under
        spectrum number ``spectrum`` (0-based, in :attr:`spectra` order)."""
        return np.asarray(self._values(name)[int(spectrum)])

    def combine(
        self,
        name: str,
        *,
        rule: str = "CQC",
        damping: "float | npt.ArrayLike | None" = None,
    ) -> np.ndarray:
        """``(n_spectra, n_values)``: ``name`` combined over the modes,
        per spectrum. CQC needs ``damping`` (one ratio or one per mode)."""
        return np.stack([
            combine_modes(v, rule=rule, omega=self.omega, damping=damping,
                          sa_periods=self.sa_periods)
            for v in self._values(name)
        ])

    def directional(
        self,
        name: str,
        *,
        rule: str = "CQC",
        damping: "float | npt.ArrayLike | None" = None,
        method: str = "100-30",
    ) -> np.ndarray:
        """``(n_values,)``: ``name`` combined over the modes, then over the
        spectra (directions) by ``method`` (``"100-30"`` or ``"SRSS"``)."""
        return combine_directions(
            self.combine(name, rule=rule, damping=damping), method=method)

    def combine_values(
        self,
        per_mode: "npt.ArrayLike",
        *,
        rule: str = "CQC",
        damping: "float | npt.ArrayLike | None" = None,
    ) -> np.ndarray:
        """Combine any signed per-mode quantity with this result's modes.

        ``per_mode`` is ``(n_spectra, n_modes, …)``, for example from
        :meth:`linear`, :meth:`drifts` or :meth:`storey_forces`. It is
        combined over the modes per spectrum by :func:`combine_modes`, with
        the modes' own ``ω`` (CQC needs ``damping``, as :meth:`combine`).
        Returns ``(n_spectra, …)``.
        """
        v = np.asarray(per_mode, dtype=np.float64)
        shape = (len(self.spectra), self.omega.size)
        if v.ndim < 2 or v.shape[:2] != shape:
            raise ValueError(
                f"SpectrumModalResult.combine_values: per_mode must be "
                f"(n_spectra, n_modes, ...) = {shape + ('...',)}; got shape "
                f"{v.shape}."
            )
        return np.stack([
            combine_modes(x, rule=rule, omega=self.omega, damping=damping,
                          sa_periods=self.sa_periods)
            for x in v
        ])

    def linear(self, name: str, weights: "npt.ArrayLike") -> np.ndarray:
        """``(n_spectra, n_modes, n_out)``: the linear quantities
        ``weights @ values`` of extractor ``name``, per mode.

        ``weights`` is ``(n_out, n_values)`` (or ``(n_values,)`` for one):
        a difference of two displacements, a sum of reactions, a storey
        force from element forces. Combine the result with
        :meth:`combine_values`, never by combining ``name`` first.
        """
        v = self._values(name)
        w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
        if w.ndim != 2 or w.shape[1] != v.shape[2]:
            raise ValueError(
                f"SpectrumModalResult.linear: {name!r} has {v.shape[2]} "
                f"values per mode; weights must be (n_out, {v.shape[2]}), "
                f"got {w.shape}."
            )
        return np.asarray(np.einsum("smv,ov->smo", v, w))

    def drifts(
        self,
        name: str,
        pairs: "Sequence[tuple[int, int]]",
        *,
        dof: int,
        heights: "Sequence[float] | None" = None,
    ) -> np.ndarray:
        """``(n_spectra, n_modes, n_pairs)``: the drift of each joint pair
        in each mode, ``(u_top - u_bottom) / h``.

        ``name`` is a :class:`NodeDisp` extractor holding ``dof`` (1-based)
        at both nodes of every ``(top, bottom)`` pair; ``heights`` (one per
        pair) makes it a drift ratio, ``None`` leaves the relative
        displacement. The drift is taken per mode and then combined
        (:meth:`combine_values`): differencing two COMBINED displacements
        is not the combined drift, and is not offered.
        """
        labels = self.labels.get(name)
        if not labels:
            raise ValueError(
                f"SpectrumModalResult.drifts: {name!r} has no labels; it "
                "must be a NodeDisp extractor."
            )
        index = {label: i for i, label in enumerate(labels)}
        h = (np.ones(len(pairs)) if heights is None
             else np.asarray(heights, dtype=np.float64))
        if h.shape != (len(pairs),) or np.any(h == 0.0):
            raise ValueError(
                "SpectrumModalResult.drifts: give one non-zero height per "
                "pair, or none."
            )
        w = np.zeros((len(pairs), len(labels)))
        for p, (top, bottom) in enumerate(pairs):
            for node, sign in ((top, 1.0), (bottom, -1.0)):
                key = f"node{int(node)}.u{int(dof)}"
                if key not in index:
                    raise ValueError(
                        f"SpectrumModalResult.drifts: {name!r} does not hold "
                        f"{key}."
                    )
                w[p, index[key]] += sign / h[p]
        return self.linear(name, w)

    def storey_forces(
        self,
        elevations: "Sequence[float]",
        *,
        vertical: "int | None" = None,
        about: "Sequence[float] | None" = None,
        tol: "float | None" = None,
        check: bool = True,
    ) -> "StoreyForces":
        """Per-mode storey shears and moments from the modal inertia.

        The inertia force of mode ``k`` under a spectrum is
        ``F_n = m_n φ_kn Γ_k Sa(T_k)`` at every massed node ``n``, the load
        whose static response is the mode's response. The cut at elevation
        ``z`` (along the 1-based ``vertical`` axis, the last one by
        default) takes every node strictly above ``z + tol``: its shear is
        ``Σ F_n`` (all ``ndm`` components) and its moment ``Σ (x_n - p) ×
        F_n`` plus the rotational inertia, ``p`` the point at height ``z``
        whose other coordinates are ``about`` (the origin by default).
        Give the bottom of each storey to get that storey's forces; the cut
        at the base is the base shear. Combine with :meth:`combine_values`.

        ``check`` first runs :meth:`check_inertia_balance` (each mode's
        total inertia against its base reaction, relative ``1e-3``).

        The moment is that of the inertia alone. Under ``p_delta=True`` the
        gravity load acting through each mode's sway adds a P-Δ moment that
        is not included here: the overturning moment from the reactions
        exceeds this one by it (0.5 % on a slender 2-D stick). The shear
        has no such term.
        """
        if self.inertia is None or self.inertia_coords is None:
            raise ValueError(
                "SpectrumModalResult.storey_forces: this result carries no "
                "per-mode inertia (it was not built by "
                "spectrum_after_gravity)."
            )
        if check:
            self.check_inertia_balance()
        xyz = np.asarray(self.inertia_coords, dtype=np.float64)
        ndm = xyz.shape[1]
        axis = (ndm if vertical is None else int(vertical)) - 1
        if not 0 <= axis < ndm:
            raise ValueError(
                f"SpectrumModalResult.storey_forces: vertical must be in "
                f"1..{ndm}, got {vertical!r}."
            )
        z = np.asarray(elevations, dtype=np.float64).ravel()
        plan = np.zeros(ndm) if about is None else np.asarray(
            about, dtype=np.float64).ravel()
        if plan.size == ndm - 1:
            plan = np.insert(plan, axis, 0.0)
        if plan.shape != (ndm,):
            raise ValueError(
                f"SpectrumModalResult.storey_forces: about= takes "
                f"{ndm - 1} plan coordinates.")
        extent = float(np.max(np.ptp(xyz, axis=0))) if xyz.size else 0.0
        eps = 1e-9 * (extent or 1.0) if tol is None else float(tol)
        f = np.asarray(self.inertia)                  # (s, m, n, ndf)
        n_rot = 1 if ndm == 2 else 3
        force = f[..., :ndm]
        rot = np.zeros(f.shape[:3] + (n_rot,))
        k = min(n_rot, max(0, f.shape[3] - ndm))
        rot[..., :k] = f[..., ndm:ndm + k]
        shear = np.zeros(f.shape[:2] + (z.size, ndm))
        moment = np.zeros(f.shape[:2] + (z.size, n_rot))
        for c, zc in enumerate(z):
            above = xyz[:, axis] > zc + eps
            p = plan.copy()
            p[axis] = zc
            arm = xyz[above] - p                      # (n_above, ndm)
            fa = force[:, :, above, :]
            if ndm == 2:
                r_x_f = (arm[:, 0] * fa[..., 1] - arm[:, 1] * fa[..., 0])[..., None]
            else:
                r_x_f = np.cross(arm, fa)
            shear[:, :, c, :] = fa.sum(axis=2)
            moment[:, :, c, :] = (r_x_f + rot[:, :, above, :]).sum(axis=2)
        return StoreyForces(elevations=z, shear=shear, moment=moment)

    def inertia_balance(self) -> np.ndarray:
        """``(n_spectra, n_modes)``: the gap ``|Σ F + R|`` between each
        mode's total inertia ``Σ F`` and its total base reaction ``R``
        (``"base_shear"``), the largest over the translational components,
        relative to the spectrum's largest modal base force
        ``max_k |Σ F_k|``. Zero for a mode that takes no part.

        The scale is per spectrum, not per mode: a near-torsion mode's net
        force is round-off of the base shear (``1e-6`` N against ``1e5`` N),
        and its own net would turn that round-off into a false alarm, while
        a gap of ``1e-3`` of the governing base shear is what moves a storey
        force. Equilibrium makes it zero to round-off; above ``1e-3`` the
        inertia the storey forces rest on is not the load the reactions
        answered (mass the per-mode loads missed, an eigenvector off the
        stiffness the legs met)."""
        if self.inertia is None:
            raise ValueError(
                "SpectrumModalResult.inertia_balance: this result carries "
                "no per-mode inertia."
            )
        reaction = self._values(BASE_SHEAR)           # (s, m, ndm)
        ndm = reaction.shape[2]
        total = np.asarray(self.inertia)[..., :ndm].sum(axis=2)
        scale = np.max(np.abs(total), axis=(1, 2))[:, None]
        gap = np.max(np.abs(total + reaction), axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(scale > 0.0, gap / scale,
                           np.where(gap > 0.0, np.inf, 0.0))
        return np.asarray(rel)

    def check_inertia_balance(
        self, rtol: float = INERTIA_BALANCE_RTOL, *, warn: bool = False,
    ) -> None:
        """Raise (or, with ``warn=True``, warn :class:`InertiaBalanceWarning`)
        when a mode's :meth:`inertia_balance` exceeds ``rtol``."""
        rel = self.inertia_balance()
        if not np.any(rel > rtol):
            return
        s, k = np.unravel_index(int(np.argmax(rel)), rel.shape)
        msg = (
            f"SpectrumModalResult: mode {k + 1} under spectrum {s} "
            f"(direction {self.spectra[s].direction}): its total inertia "
            f"Σ m φ Γ Sa and its total base reaction differ by "
            f"{rel[s, k]:.3e} of the spectrum's largest modal base force "
            f"(limit {rtol:g}); "
            f"{int(np.sum(rel > rtol))} mode-spectrum pair(s) fail. The "
            "storey forces from inertia would not add up to the base shear."
        )
        if warn:
            warnings.warn(msg, InertiaBalanceWarning, stacklevel=2)
            return
        raise RuntimeError(msg)

    def cancellation(
        self,
        name: str,
        index: int,
        *,
        spectrum: int = 0,
        rule: str = "CQC",
        damping: "float | npt.ArrayLike | None" = None,
        plan: "tuple[int, int] | None" = (1, 2),
    ) -> ModalCancellation:
        """:func:`modal_cancellation` of value ``index`` of ``name`` under
        spectrum ``spectrum``, with each mode's principal-axis angle
        ``atan2(Γ_b, Γ_a)`` (degrees, modulo 180°) from its participation
        factors along the plan directions ``plan = (a, b)`` (1-based; ``None`` leaves the
        angles out). Refused when an Sa-period override groups modes into
        one rigid term: the per-mode shares would not add up to it."""
        if self.sa_periods is not None and np.any(
                np.asarray(self.sa_periods) > 0.0):
            raise ValueError(
                "SpectrumModalResult.cancellation: this result has an "
                "Sa-period override (residual_period=), so its modes sharing "
                "it combine as one rigid term; a per-mode cancellation report "
                "does not apply."
            )
        terms = self.modes(name, spectrum)[:, int(index)]
        angles = None
        if plan is not None:
            comp = ("MX", "MY", "MZ")
            ga, gb = (self.modal.modal.participation_factors(comp[int(d) - 1])
                      for d in plan)
            # an axis, not a direction: φ and -φ are the same mode
            angles = np.mod(np.degrees(np.arctan2(gb, ga)), 180.0)
        return modal_cancellation(terms, rule=rule, omega=self.omega,
                                  damping=damping, angles=angles)

    def base_shear(
        self,
        *,
        rule: str = "CQC",
        damping: "float | npt.ArrayLike | None" = None,
    ) -> np.ndarray:
        """``V_t`` per spectrum, shape ``(n_spectra,)``: the combined total
        base reaction along each spectrum's own direction (the total is
        combined, not the combined reactions summed)."""
        combined = self.combine(BASE_SHEAR, rule=rule, damping=damping)
        return np.asarray([
            combined[i, s.direction - 1] for i, s in enumerate(self.spectra)
        ])

    def scale_to_base_shear(
        self,
        v_target: "float | Sequence[float]",
        names: Sequence[str],
        *,
        rule: str = "CQC",
        damping: "float | npt.ArrayLike | None" = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Scale combined forces up to a target base shear, per spectrum.

        ``v_target`` is the minimum base shear per spectrum (one value for
        all, or one per spectrum), computed by the caller from the code in
        force: for example ASCE/SEI 7-22 §12.9.1.4.1 (100 % of the ELF
        ``V``), NEC-SE-DS (80 % / 90 % of the static shear), NCh433. The
        factor is ``max(1, v_target / V_t)`` with ``V_t`` from
        :meth:`base_shear`, and it multiplies the combined values of each
        quantity in ``names`` under that spectrum.

        Returns ``(factors, scaled)``: ``factors`` shape ``(n_spectra,)``,
        ``scaled[name]`` shape ``(n_spectra, n_values)`` (pass it to
        :func:`combine_directions` for the directional combination). Name
        force quantities only: drifts are not scaled here (apeETABS
        ``DriftScale`` owns drift scaling).
        """
        v_t = self.base_shear(rule=rule, damping=damping)
        target = np.broadcast_to(
            np.asarray(v_target, dtype=np.float64), v_t.shape)
        if np.any(target < 0.0) or not np.all(np.isfinite(target)):
            raise ValueError(
                "scale_to_base_shear: v_target must be finite and >= 0.")
        if np.any(v_t <= 0.0):
            raise ValueError(
                f"scale_to_base_shear: a combined base shear is zero "
                f"({v_t.tolist()}); nothing to scale against.")
        factors = np.maximum(1.0, target / v_t)
        scaled = {
            name: self.combine(name, rule=rule, damping=damping)
            * factors[:, None]
            for name in names
        }
        return np.asarray(factors), scaled

    def _values(self, name: str) -> np.ndarray:
        try:
            return np.asarray(self.per_mode[name])
        except KeyError:
            raise KeyError(
                f"SpectrumModalResult: no extractor {name!r}; have "
                f"{sorted(self.per_mode)}."
            ) from None
