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
* :func:`combine_directions` — 100/30 or SRSS over the directions;
* :func:`seismic_envelope` — ``S ± f·E``, the envelope over the sign of the
  seismic term, for a gravity combination ``S`` such as ``1.0 D + L_exp``.

An extractor is any callable ``extractor(ops) -> 1-D float array`` of fixed
length, optionally with a ``labels`` attribute. Every combination works on
one quantity's own per-mode values; none is derived from combined
displacements.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .gravity_modal import GravityModalResult


__all__ = [
    "DesignSpectrum",
    "ElementResponse",
    "NodeDisp",
    "NodeReaction",
    "SpectrumModalResult",
    "combine_directions",
    "combine_modes",
    "cqc_correlation",
    "seismic_envelope",
]

#: ``extractor(ops) -> 1-D float array`` of fixed length.
ResponseExtractor = Callable[[Any], "npt.ArrayLike"]

#: The name the driver files the total base reaction under.
BASE_SHEAR = "base_shear"


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


def combine_modes(
    per_mode: "npt.ArrayLike",
    *,
    rule: str = "CQC",
    omega: "npt.ArrayLike | None" = None,
    damping: "float | npt.ArrayLike | None" = None,
) -> np.ndarray:
    """Combine the per-mode values of one quantity over the mode axis.

    ``per_mode`` has the modes on axis 0. ``rule`` is ``"CQC"`` (needs
    ``omega`` and ``damping``) or ``"SRSS"``. Returns the non-negative
    combined value, shape ``per_mode.shape[1:]``.
    """
    r = np.asarray(per_mode, dtype=np.float64)
    if r.ndim < 1:
        raise ValueError("combine_modes: per_mode needs a mode axis.")
    if rule == "SRSS":
        return np.asarray(np.sqrt(np.sum(r * r, axis=0)))
    if rule == "CQC":
        if omega is None or damping is None:
            raise ValueError("combine_modes: CQC needs omega= and damping=.")
        rho = cqc_correlation(omega, damping)
        if rho.shape[0] != r.shape[0]:
            raise ValueError(
                f"combine_modes: {r.shape[0]} modes of values but "
                f"{rho.shape[0]} omegas."
            )
        flat = r.reshape(r.shape[0], -1)
        s = np.einsum("i...,ij,j...->...", flat, rho, flat)
        return np.asarray(np.sqrt(np.clip(s, 0.0, None)).reshape(r.shape[1:]))
    raise ValueError(
        f"combine_modes: rule must be 'CQC' or 'SRSS', got {rule!r}."
    )


def combine_directions(
    per_direction: "npt.ArrayLike", *, method: str = "100-30",
) -> np.ndarray:
    """Combine one quantity's per-direction values (axis 0).

    ``"100-30"``: the largest of ``|E_a| + 0.3 Σ_{b≠a} |E_b|`` over which
    direction ``a`` carries the 100 %. ``"SRSS"``: ``sqrt(Σ E_d²)``.
    """
    e = np.abs(np.asarray(per_direction, dtype=np.float64))
    if e.ndim < 1 or e.shape[0] < 1:
        raise ValueError("combine_directions: needs a direction axis.")
    if method == "SRSS":
        return np.asarray(np.sqrt(np.sum(e * e, axis=0)))
    if method == "100-30":
        total = np.sum(e, axis=0)
        return np.asarray(np.max(e + 0.3 * (total - e), axis=0))
    raise ValueError(
        f"combine_directions: method must be '100-30' or 'SRSS', got "
        f"{method!r}."
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
    """

    modal: "GravityModalResult"
    spectra: tuple[DesignSpectrum, ...]
    gravity_values: Mapping[str, np.ndarray]
    per_mode: Mapping[str, np.ndarray]
    labels: Mapping[str, "tuple[str, ...] | None"]
    restore_drift: tuple[float, float]

    @property
    def omega(self) -> np.ndarray:
        """Circular frequencies of the modes (rad/s)."""
        return self.modal.modal.omega

    @property
    def periods(self) -> np.ndarray:
        """Periods of the modes (s)."""
        return self.modal.periods

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
            combine_modes(v, rule=rule, omega=self.omega, damping=damping)
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
