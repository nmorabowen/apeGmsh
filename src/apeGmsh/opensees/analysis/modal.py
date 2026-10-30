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

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._core import default_artifact_dir
from .._internal.types import ModalBasis, Primitive
from ...results.readers._ladruno_writer import check_target, write_modes

if TYPE_CHECKING:
    from ..emitter.base import Emitter
    from ..emitter.live import LiveOpsEmitter
    from ..node import Node


__all__ = [
    "FrequencyResponseResult",
    "ModalHistoryResult",
    "ModalPropertiesResult",
    "ParallelModalResult",
    "RandomResponseResult",
    "ResidualModes",
    "ResponseSpectrumResult",
    "RitzVectors",
    "RotationRolesUnavailableWarning",
    "StaleModelArchiveWarning",
    "SteadyStateResult",
]


class StaleModelArchiveWarning(UserWarning):
    """:meth:`ParallelModalResult.to_native` found a file at the archive
    path ``modal_deck`` writes (``default_artifact_dir() /
    "<fem.model_name>.h5"``) and did not use it: its ``/meta/lineage``
    ``fem_hash`` is not ``fem.snapshot_id`` (another run of the model), or
    it has no lineage or no ``/opensees`` zone. The rotations then follow
    the rules for a call without ``model_h5``. The message names the file
    and the reason."""


class RotationRolesUnavailableWarning(UserWarning):
    """:meth:`ParallelModalResult.to_native`, given no ``model_h5=``,
    wrote no rotation for some nodes: the deck's DOF list reaches a
    rotation slot, and neither the deck nor ``fem`` settles what a node
    holds there (a 2-D node of ndf 3: a frame's rotation or a u-p node's
    pore pressure; a node whose ndf ``fem`` does not carry). Only the
    model's DOF roles can. The message names the nodes and says to pass
    ``model_h5=``."""


#: The spatial slots (``ux..rz`` = 0..5) a native mode stage writes as
#: ``rotation_*``, per ``ndm``: ``rotation_z`` in 2-D, all three in 3-D,
#: as the ``.ladruno`` writer writes ``Rz`` / ``Rx,Ry,Rz``.
_NATIVE_ROTATION_SLOTS: dict[int, tuple[int, ...]] = {2: (5,), 3: (3, 4, 5)}


_NO_SHAPES_MSG = (
    "ParallelModalResult: no mode-shape harvest in this run dir — "
    "neither mode_shapes.json (the replicated FEAST deck) nor "
    "mode_shapes_rank<P>.json (the partitioned ARPACK deck) was found "
    "next to the eigenvalue write-out. The deck must be emitted by an "
    "ADR 0077 P3+ apeSees.modal_deck, and the fetch must bring the "
    "sidecar(s) + per-mode files back with it."
)

# Cross-rank agreement tolerance for a shared boundary node (ADR 0077
# INV-12). Both ranks ran the SAME replicated ARPACK outer loop over the
# same global operator and received the same broadcast solution, so the
# components are bitwise identical doubles and the only slack needed is
# recorder text formatting. This is many orders tighter than the O(1)
# disagreement a private per-rank Lanczos produces — which is the whole
# point of the check.
_SHARED_NODE_RTOL = 1.0e-6
_SHARED_NODE_ATOL_FRAC = 1.0e-9

_MPI_BLIND_MSG = (
    "ParallelModalResult: participation factors / effective modal mass are "
    "not available from a distributed run — upstream modalProperties is "
    "MPI-blind (ADR 0077 INV-2). Run the single-process "
    "apeSees.modal_properties(...) on a node-sized model (ADR 0077 Tier 0) "
    "for participation."
)


def _modal_dirs(dirs: Sequence[int], *, context: str) -> tuple[int, ...]:
    """Global translations 1..3, each once (the fork's ``-dir`` rule)."""
    out = tuple(int(d) for d in dirs)
    if any(d not in (1, 2, 3) for d in out):
        raise ValueError(
            f"{context}: dirs are global translations 1, 2 or 3, got {dirs!r}."
        )
    if len(set(out)) != len(out):
        raise ValueError(f"{context}: a direction is given twice in {dirs!r}.")
    return out


def _unit_tol(tol: float | None, *, context: str) -> float | None:
    """``-tol``: ``None`` keeps the fork's default, else a value in (0, 1)."""
    if tol is None:
        return None
    t = float(tol)
    if not 0.0 < t < 1.0:
        raise ValueError(f"{context}: tol must be in (0, 1), got {tol!r}.")
    return t


@dataclass(frozen=True, kw_only=True, slots=True)
class ResidualModes(ModalBasis):
    """``residualModes -dir d1 <d2 <d3>> <-period $T> <-replace> <-tol $t>``.

    Fork WP-213 (Ladruno ADR 213): appends, per direction, the
    missing-mass pseudo-modes of the current mode set, so the modal mass
    reaches 100 % and the static response is exact. It runs after
    ``eigen`` (or ``ritzVectors``) and before ``modalProperties``, which
    must run again on the new set; a later ``eigen`` drops the record.

    ``period`` is ``"rayleigh"`` (the default, not emitted): a spectral
    analysis reads each pseudo-mode's Sa at its own Rayleigh period. A
    period ``T`` keeps the Rayleigh λ, and ``responseSpectrumAnalysis``
    then reads Sa at ``T`` and sums the pseudo-modes into one rigid mode
    (ADR 213 D3b). The fork refuses ``T`` on a Ritz set.
    """

    dirs: tuple[int, ...]
    period: float | str = "rayleigh"
    replace: bool = False
    tol: float | None = None

    def __post_init__(self) -> None:
        context = "ResidualModes"
        dirs = _modal_dirs(self.dirs, context=context)
        if not dirs:
            raise ValueError(f"{context}: dirs needs at least one direction.")
        period: float | str = self.period
        if period != "rayleigh":
            if isinstance(period, (str, bool)):
                raise ValueError(
                    f"{context}: period is 'rayleigh' or a period in "
                    f"seconds, got {period!r}."
                )
            period = float(period)
            if not (math.isfinite(period) and period > 0.0):
                raise ValueError(
                    f"{context}: period must be finite and > 0, got "
                    f"{self.period!r}."
                )
        object.__setattr__(self, "dirs", dirs)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "replace", bool(self.replace))
        object.__setattr__(self, "tol", _unit_tol(self.tol, context=context))

    def args(self) -> tuple[int | float | str, ...]:
        """The command's arguments after ``residualModes``."""
        out: list[int | float | str] = ["-dir", *self.dirs]
        if self.period != "rayleigh":
            out.extend(("-period", self.period))
        if self.replace:
            out.append("-replace")
        if self.tol is not None:
            out.extend(("-tol", self.tol))
        return tuple(out)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("residualModes", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class RitzVectors(ModalBasis):
    """``ritzVectors $n <-dir d ...> <-pattern $tag ...> <-cycles $c ...> <-tol $t>``.

    Fork WP-215 (Ladruno ADR 213 D7, D8): load-dependent Ritz vectors
    installed as the domain's mode set, in place of ``eigen``. The
    starting loads are ``M ι_d`` for each direction in ``dirs``, then the
    reference loads of each pattern tag in ``patterns``, in that order;
    ``cycles`` caps the blocks per start, one count per start in the same
    order (``0`` is no cap). The fork factors ``K`` through the defined
    analysis, so one must exist, and it refuses a partitioned
    (OpenSeesSP) domain. Ritz periods are Rayleigh periods, not natural
    periods.
    """

    n: int
    dirs: tuple[int, ...] = ()
    patterns: tuple[int, ...] = ()
    cycles: tuple[int, ...] | None = None
    tol: float | None = None

    def __post_init__(self) -> None:
        context = "RitzVectors"
        if isinstance(self.n, bool) or int(self.n) != self.n or self.n < 1:
            raise ValueError(
                f"{context}: n must be an integer >= 1, got {self.n!r}."
            )
        dirs = _modal_dirs(self.dirs, context=context)
        patterns = tuple(int(p) for p in self.patterns)
        if len(set(patterns)) != len(patterns):
            raise ValueError(
                f"{context}: a pattern is given twice in {self.patterns!r}."
            )
        if not dirs and not patterns:
            raise ValueError(
                f"{context}: give at least one starting load (dirs and/or "
                "patterns)."
            )
        cycles = self.cycles
        if cycles is not None:
            cycles = tuple(int(c) for c in cycles)
            if len(cycles) != len(dirs) + len(patterns):
                raise ValueError(
                    f"{context}: cycles takes one count per starting load, "
                    f"dirs then patterns ({len(dirs) + len(patterns)} "
                    f"starts, {len(cycles)} counts)."
                )
            if any(c < 0 for c in cycles):
                raise ValueError(
                    f"{context}: cycles are counts >= 0 (0 = no cap), got "
                    f"{self.cycles!r}."
                )
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "dirs", dirs)
        object.__setattr__(self, "patterns", patterns)
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "tol", _unit_tol(self.tol, context=context))

    def args(self) -> tuple[int | float | str, ...]:
        """The command's arguments after ``ritzVectors``."""
        out: list[int | float | str] = [self.n]
        if self.dirs:
            out.extend(("-dir", *self.dirs))
        if self.patterns:
            out.extend(("-pattern", *self.patterns))
        if self.cycles is not None:
            out.extend(("-cycles", *self.cycles))
        if self.tol is not None:
            out.extend(("-tol", self.tol))
        return tuple(out)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("ritzVectors", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


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
        return np.asarray(np.abs(self.response))

    @property
    def phase(self) -> np.ndarray:
        """Phase angle ``atan2(Im, Re)`` in radians."""
        return np.asarray(np.angle(self.response))


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


def _write_mode_stages(
    writer: Any,
    eigenvalues: Sequence[float] | np.ndarray,
    node_ids: np.ndarray,
    fields: Callable[[int], np.ndarray],
    n_trans: int,
    rotation_slots: Sequence[int] = (),
    stacklevel: int = 3,
) -> None:
    """Write one ``kind="mode"`` stage per eigenvalue into an open
    :class:`~apeGmsh.results.writers._native.NativeWriter`.

    The mode-stage layout, shared by
    :meth:`ParallelModalResult.to_native` and
    :meth:`~apeGmsh.opensees.analysis.eigen.EigenResult.to_native`:
    stage ``mode_<k>`` at ``time = [0.0]`` with the
    ``eigenvalue`` / ``frequency_hz`` / ``period_s`` / ``mode_index``
    attrs, ``displacement_*`` from columns ``0 .. n_trans-1`` of
    ``fields(k)`` and one ``rotation_*`` per entry of ``rotation_slots``
    (3, 4, 5 are ``rotation_x/y/z``). ``fields(k)`` is the ``(n_nodes,
    6)`` shape of mode ``k`` (1-indexed) by spatial slot ``ux..rz``, rows
    in ``node_ids`` order, with 0.0 in a slot that holds no rotation at
    its node; :func:`_native_fields` builds it from the DOF roles.

    A non-positive eigenvalue warns once for that mode and writes
    ``frequency_hz = period_s = 0``; ``stacklevel`` counts from this
    function to the caller of the public method (3 when that method
    calls it directly).
    """
    import math
    import warnings

    axes = ("x", "y", "z")
    for mode_idx, lam in enumerate(eigenvalues, start=1):
        lam_f = float(lam)
        if lam_f > 0:
            omega = math.sqrt(lam_f)
        else:
            warnings.warn(
                f"Mode {mode_idx} has a non-positive eigenvalue "
                f"({lam_f:.6g}); this indicates a spurious/"
                "unstable mode (rigid-body mechanism or "
                "unconverged eigensolve). Writing frequency=0 / "
                "period=0 for it.",
                UserWarning,
                stacklevel=stacklevel,
            )
            omega = 0.0
        freq_hz = omega / (2.0 * math.pi)
        period_s = (2.0 * math.pi / omega) if omega > 0 else 0.0

        sid = writer.begin_stage(
            name=f"mode_{mode_idx}",
            kind="mode",
            time=np.array([0.0]),
            eigenvalue=lam_f,
            frequency_hz=float(freq_hz),
            period_s=float(period_s),
            mode_index=mode_idx,
        )
        field = fields(mode_idx)
        components: dict[str, np.ndarray] = {}
        for axis_idx in range(n_trans):
            components[f"displacement_{axes[axis_idx]}"] = (
                field[:, axis_idx][None, :]
            )
        for slot in rotation_slots:
            components[f"rotation_{axes[int(slot) - 3]}"] = (
                field[:, int(slot)][None, :]
            )
        writer.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components=components,
        )
        writer.end_stage()


def _native_fields(
    modes: "Sequence[Any] | np.ndarray",
    layouts: "Sequence[tuple[int, ...]]",
    rotations: np.ndarray,
    ndm: int,
) -> "tuple[np.ndarray, tuple[int, ...]]":
    """The ``(n_modes, n_nodes, 6)`` spatial mode shapes a native writer
    writes, and the slots it writes as ``rotation_*``.

    ``modes[k]`` holds each node's DOF row (its own DOFs, padded or not),
    laid onto ``ux..rz`` by that node's ``layouts`` entry (the emit's
    ``_node_layout``). A node whose ``rotations`` entry is ``False`` (its
    DOF roles refuse a rotation: a u-p pore pressure, a node no beam or
    shell touches) writes 0.0 in every rotation slot, the native
    convention for a node without rotations. The slots returned are
    those of :data:`_NATIVE_ROTATION_SLOTS` for ``ndm`` that the layout of
    at least one rotation-holding node reaches; none when no node holds
    a rotation.
    """
    from .eigen import _spatial

    lays = list(layouts)
    if len(modes):
        spatial = np.stack([_spatial(rows, lays) for rows in modes])
    else:
        spatial = np.zeros((0, len(lays), 6), dtype=np.float64)
    rot = np.asarray(rotations, dtype=bool)
    spatial[:, ~rot, 3:] = 0.0
    slots = tuple(
        s for s in _NATIVE_ROTATION_SLOTS.get(int(ndm), ())
        if any(r and s in lay for r, lay in zip(rot, lays))
    )
    return spatial, slots


def _discover_archive(fem: Any, *, where: str) -> "Path | None":
    """The archive ``modal_deck`` left for ``fem``, or ``None``.

    The bridge writes it at ``default_artifact_dir() /
    f"{fem.model_name}.h5"`` after a deck emit (ADR 0112 D1). It is used
    only when its ``/meta/lineage`` ``fem_hash`` equals
    ``fem.snapshot_id``, it holds the ``/opensees`` zone the roles need,
    and it covers the model: every fem node on a solid (3-D) cell is
    touched by one of its elements. ``fem_hash`` names the mesh only, so a
    second bridge on the same fem and ``model_name`` (a frame-only one
    beside a frame + u-p one) leaves a matching hash over other elements;
    the coverage check refuses that file. A file there that fails any
    check is ignored with one :class:`StaleModelArchiveWarning` naming the
    reason. An unnamed snapshot has no such path, and no file there is
    nothing to say: both return ``None`` silently.
    """
    import warnings

    import h5py

    from .eigen import _lineage

    name = fem.model_name
    if not name:
        return None
    path = default_artifact_dir() / f"{name}.h5"
    if not path.is_file():
        return None
    lineage = _lineage(path)
    if lineage is None:
        why = "it has no /meta/lineage hashes (or is not HDF5)"
    elif lineage[1] != str(fem.snapshot_id):
        why = (f"its fem_hash {lineage[1]!r} is not this fem's snapshot_id "
               f"{str(fem.snapshot_id)!r} (another run of the model)")
    else:
        with h5py.File(path, "r") as f:
            has_zone = "opensees" in f
        if not has_zone:
            why = "it has no /opensees zone, so no element records to read"
        else:
            bare = _uncovered_solid_nodes(fem, path, where=where)
            if not bare:
                return path
            why = (
                f"none of its elements touches {len(bare)} node(s) on solid "
                f"cells of this fem ({bare[:10]}), so it is not the model "
                "this deck ran (another bridge on the same mesh and "
                "model_name wrote it last) and its roles would call those "
                "nodes' slots element-less rotations"
            )
    warnings.warn(
        f"{where}: ignored {path} as the deck's model archive: {why}. The "
        "rotations follow the rules for a call without model_h5=; pass "
        "model_h5= (the archive the deck was emitted from, ops.h5) to "
        "name it.",
        StaleModelArchiveWarning,
        stacklevel=3,
    )
    return None


def _uncovered_solid_nodes(
    fem: Any, model_h5: Path, *, where: str,
) -> list[int]:
    """The sorted fem nodes on a solid (3-D) cell that no element of
    ``model_h5`` touches (``RotationRoles.touched`` of
    :func:`~apeGmsh.opensees._recorder_translate.model_h5_rotation_roles`).
    Empty when the archive covers every solid node, or there is none."""
    from .._recorder_translate import model_h5_rotation_roles

    solid: set[int] = set()
    for group in fem.elements:
        if int(group.element_type.dim) == 3:
            solid.update(int(n) for n in np.unique(group.connectivity))
    if not solid:
        return []
    roles = model_h5_rotation_roles(model_h5, sorted(solid), where=where)
    return sorted(solid - roles.touched)


def _has_solid_cells(fem: Any) -> bool:
    """Whether ``fem`` holds any 3-D (solid) cell, by its gmsh element
    types (``fem.elements.types``, each with its ``dim``)."""
    return any(int(t.dim) == 3 for t in fem.elements.types)


def _fem_ndf_slots(
    fem: Any, rec: "Sequence[int]", ndf: int, ndm: int,
) -> "tuple[list[tuple[int, ...]], np.ndarray, list[tuple[int, str]]]":
    """Per node of ``rec``, without a model archive: its DOF layout,
    whether its rotation slots hold rotations, and the ``(node, reason)``
    of each node that only the DOF roles could settle.

    The node's own ndf comes from ``fem.nodes.ndf_for`` (a
    ``LookupError`` means ``fem`` does not carry it). A 3-D node of ndf 6
    holds rotations in DOFs 4-6; a node whose layout reaches no native
    rotation slot holds none (a 3-D u-p node, ndf 4); any other node (a
    2-D node of ndf 3: ``(ux, uy, rz)`` or ``(ux, uy, p)``) and a node
    of unknown ndf are unsettled. An unknown node is laid out on its
    translations only, so nothing of it lands in a rotation slot.
    """
    from .._internal.build.recorders import _node_layout

    native = _NATIVE_ROTATION_SLOTS.get(int(ndm), ())
    trans = tuple(range(min(3, ndm, ndf)))
    layouts: list[tuple[int, ...]] = []
    rotations = np.zeros(len(rec), dtype=bool)
    unsettled: list[tuple[int, str]] = []
    for i, n in enumerate(rec):
        try:
            own = int(fem.nodes.ndf_for(int(n)))
        except LookupError:
            layouts.append(trans)
            unsettled.append((int(n), "fem carries no ndf for it"))
            continue
        lay = _node_layout(own, ndm)
        layouts.append(lay)
        if not any(s in native for s in lay):
            continue                    # its own DOFs reach no rotation
        if ndm == 3 and own == 6:
            rotations[i] = True
        else:
            unsettled.append((int(n), (
                f"ndf {own} in {ndm}-D may carry a rotation (a frame) or a "
                "pore pressure (a u-p element) in that slot")))
    return layouts, rotations, unsettled


def _deck_archive(
    model_h5: "str | Any", rec: "Sequence[int]", shape_ndm: "int | None",
    *, where: str,
) -> Any:
    """The :class:`~apeGmsh.opensees.opensees_model.OpenSeesModel` of
    ``model_h5``, checked against modes over nodes ``rec``: the
    ``ndm`` of ``shape_ndm`` (when known), every node in its fem, and a
    ``/opensees/nodes_ndf`` entry for each. Raises ``ValueError`` naming
    ``where`` otherwise; the caller has written nothing yet."""
    from ..opensees_model import OpenSeesModel

    model = OpenSeesModel.from_h5(str(model_h5))
    ndm = int(model.ndm)
    if shape_ndm is not None and int(shape_ndm) != ndm:
        raise ValueError(
            f"{where}: the modes are ndm={shape_ndm} (the sidecar's, or "
            f"the solve's) but {model_h5} is an ndm={ndm} model; the "
            "modes and the archive come from different models. Nothing "
            "was written."
        )
    fem_ids = {int(n) for n in model.fem.nodes.ids}
    stray = [n for n in rec if n not in fem_ids]
    if stray:
        raise ValueError(
            f"{where}: node(s) "
            f"{stray[:5]} are not nodes of the model archive's fem; the "
            "mode shapes and the model come from different meshes. "
            "Nothing was written."
        )
    nodes_ndf = model.nodes_ndf
    unknown = [n for n in rec if n not in nodes_ndf]
    if unknown:
        raise ValueError(
            f"{where}: {model_h5} records no ndf "
            f"for node(s) {unknown[:5]} (/opensees/nodes_ndf), "
            "so their DOFs cannot be laid onto Ux..Rz. Re-write the "
            "archive with ops.h5(...). Nothing was written."
        )
    return model


def _read_mode_row(mode_file: Any, n_nodes: int, ndf: int) -> np.ndarray:
    """Read one ``mode_shape_*.out`` recorder row into ``(n_nodes, ndf)``.

    The file is a single whitespace row written by ``recorder Node ...
    "eigen k"`` over a pinned node list — node-major, ``ndf`` columns
    per node (DOFs a node does not carry are ``0.0`` padding). A width
    mismatch means the sidecar and the run dir are out of sync.
    """
    row = np.asarray(
        [float(t) for t in mode_file.read_text().split()],
        dtype=np.float64,
    )
    if row.shape[0] != n_nodes * ndf:
        raise ValueError(
            f"ParallelModalResult.from_job: {mode_file} has "
            f"{row.shape[0]} values, expected {n_nodes} nodes "
            f"x {ndf} dofs = {n_nodes * ndf} (sidecar "
            "mismatch — deck and run dir out of sync?)."
        )
    return row.reshape(n_nodes, ndf)


@dataclass(frozen=True, slots=True)
class ParallelModalResult:
    """Eager result of a distributed modal run (ADR 0077 Tier 1),
    harvested from a completed ``apeSees.modal_deck`` run dir.

    One surface for **both** Tier-1 backends — the replicated FEAST deck
    and the partitioned ARPACK deck. Carries the harvested eigenvalues
    (for FEAST a band output, so the count is dynamic; for ARPACK the
    requested ``num_modes``) plus derived ω / f / T, the optional
    completeness flag, and — when the run dir carries a mode-shape
    harvest — the full-field mode shapes. The two decks record shapes
    differently (rank-0 whole-field vs per-rank slices merged at read
    time, see :meth:`from_job`), but the accessors below see one
    ``(n_nodes, ndf)`` field either way. Unlike
    :class:`~apeGmsh.opensees.analysis.eigen.EigenResult` it is **eager**
    and holds no live domain — the run is remote / already complete;
    :meth:`mode_shape` reads the harvested arrays.

    Modal properties are NOT on this surface: ``modalProperties`` is
    MPI-blind upstream (wrong effective mass under any multi-rank run;
    INV-2). For participation factors run the single-process
    :meth:`apeSees.modal_properties` on a node-sized model (Tier 0).
    """

    eigenvalues: np.ndarray
    certified: bool | None = None

    # P3 mode-shape harvest (None when the run dir carries no sidecar —
    # e.g. a pre-P3 deck). Underscore-prefixed; read via mode_shape /
    # mode_shape_field / shape_nodes. ``_shape_ndm`` is the model ndm
    # the sidecar carries (P4 — maps shape columns to displacement/
    # rotation components in :meth:`to_native`).
    _shape_nodes: np.ndarray | None = None
    _shapes: np.ndarray | None = None
    _shape_ndm: int | None = None

    @property
    def n_modes(self) -> int:
        """Number of modes found in the band (the FEAST count)."""
        return int(self.eigenvalues.shape[0])

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

    @classmethod
    def from_job(
        cls,
        job_dir: "str | Any",
        *,
        out: str = "eigenvalues.out",
        certified: bool | None = None,
    ) -> "ParallelModalResult":
        """Harvest a completed ``modal_deck`` run dir.

        Reads the rank-0 eigenvalue write-out (``out``, default
        ``eigenvalues.out``) — a single whitespace-separated line of
        ``λ_i = ω_i²`` in solve order — and, when present, the
        mode-shape harvest. **Both Tier-1 deck shapes are read through
        this one entry point**, distinguished by which sidecar the run
        dir carries:

        * ``mode_shapes.json`` + ``mode_shape_<k>.out`` — the
          **replicated** FEAST deck (ADR 0077 P3). Rank 0 holds every
          node, so one recorder carried the whole field.
        * ``mode_shapes_rank<P>.json`` + ``mode_shape_<k>_rank<P>.out``
          — the **partitioned** ARPACK deck (Tier 1B). Each rank
          recorded only the nodes it owns, so the field is merged here:
          concatenate, sort by node tag, and **cross-check every shared
          boundary node** (INV-12 — see
          :meth:`_merge_partitioned_shapes`). Either way the result is
          the same ``(n_nodes, ndf)`` layout.

        A run dir with neither sidecar (a pre-P3 deck) still harvests
        eigenvalues; :meth:`mode_shape` then fails loud.

        Raises ``FileNotFoundError`` if the eigenvalue write-out is
        missing (the run did not complete or was not fetched back), or
        if a sidecar is present but a per-mode file is not; ``ValueError``
        if a per-mode row does not match its sidecar's ``nodes × ndf``
        width, if the per-rank sidecars disagree on ``ndf``, or if a
        shared boundary node's components disagree across ranks.
        """
        import json
        from pathlib import Path

        base = Path(job_dir)
        path = base / out
        if not path.is_file():
            raise FileNotFoundError(
                "ParallelModalResult.from_job: no eigenvalue write-out at "
                f"{path} — did the modal_deck run complete and fetch back? "
                f"(the rank-0 block writes '{out}')."
            )
        tokens = path.read_text().split()
        values = np.asarray([float(t) for t in tokens], dtype=np.float64)

        shape_nodes: np.ndarray | None = None
        shapes: np.ndarray | None = None
        shape_ndm: int | None = None
        sidecar = base / "mode_shapes.json"
        rank_sidecars = sorted(base.glob("mode_shapes_rank*.json"))
        if sidecar.is_file():
            meta = json.loads(sidecar.read_text())
            shape_nodes = np.asarray(meta["nodes"], dtype=np.int64)
            ndf = int(meta["ndf"])
            # "ndm" landed one format rev after "nodes"/"ndf" — a sidecar
            # without it is a 3-D deck (the only decks emitted before).
            shape_ndm = int(meta.get("ndm", 3))
            n_nodes = int(shape_nodes.shape[0])
            rows: list[np.ndarray] = []
            for k in range(1, values.shape[0] + 1):
                mode_file = base / f"mode_shape_{k}.out"
                if not mode_file.is_file():
                    raise FileNotFoundError(
                        "ParallelModalResult.from_job: mode_shapes.json "
                        f"promises {values.shape[0]} mode(s) but "
                        f"{mode_file} is missing — incomplete fetch or "
                        "the run died mid-harvest."
                    )
                row = _read_mode_row(mode_file, n_nodes, ndf)
                rows.append(row)
            shapes = (
                np.stack(rows)
                if rows
                else np.zeros((0, n_nodes, ndf), dtype=np.float64)
            )
        elif rank_sidecars:
            shape_nodes, shapes, shape_ndm = cls._merge_partitioned_shapes(
                base, rank_sidecars, n_modes=int(values.shape[0]),
            )
        return cls(
            eigenvalues=values,
            certified=certified,
            _shape_nodes=shape_nodes,
            _shapes=shapes,
            _shape_ndm=shape_ndm,
        )

    @staticmethod
    def _merge_partitioned_shapes(
        base: Any, rank_sidecars: "Sequence[Any]", *, n_modes: int,
    ) -> "tuple[np.ndarray, np.ndarray, int]":
        """Merge per-rank mode-shape harvests into one full field
        (ADR 0077 Tier 1B / INV-12).

        The partitioned ARPACK deck records each rank's OWN nodes into
        ``mode_shape_<k>_rank<P>.out``, described by
        ``mode_shapes_rank<P>.json``. This walks every rank, stacks the
        rows, and returns ``(node_tags, shapes, ndm)`` in the same
        ``(n_modes, n_nodes, ndf)`` layout the replicated path produces.

        **The boundary-node check is the reason this is not a plain
        concatenation.** A node on a partition boundary is defined on
        both touching ranks, so it appears in two sidecars — and the
        deck records it on both *deliberately*. Both ranks ran the same
        replicated ARPACK outer loop against the same global operator
        and received the same broadcast solution, so the two copies must
        agree. If they do not, each rank ran a **private Lanczos over
        its own subdomain** — precisely the failure mode this backend
        exists to prevent, and exactly what a pre-#668 binary (or a deck
        that lost its ``system Mumps``) produces. Since that failure is
        otherwise silent — you get a plausible-looking field, not an
        error — this comparison is the cheapest available proof that the
        run really was distributed, at one comparison per shared node.
        Raises ``ValueError`` naming the node and both values.
        """
        import json
        import re

        per_rank: "list[tuple[int, np.ndarray, list[np.ndarray]]]" = []
        ndf: int | None = None
        ndm: int | None = None
        for meta_path in rank_sidecars:
            match = re.search(r"mode_shapes_rank(\d+)\.json$", meta_path.name)
            if match is None:  # pragma: no cover - glob pins the shape
                continue
            rank = int(match.group(1))
            meta = json.loads(meta_path.read_text())
            nodes = np.asarray(meta["nodes"], dtype=np.int64)
            rank_ndf = int(meta["ndf"])
            if ndf is None:
                ndf = rank_ndf
                ndm = int(meta.get("ndm", 3))
            elif rank_ndf != ndf:
                raise ValueError(
                    "ParallelModalResult.from_job: per-rank sidecars "
                    f"disagree on ndf ({ndf} vs {rank_ndf} on rank "
                    f"{rank}) — the run dir mixes decks."
                )
            rows = []
            for k in range(1, n_modes + 1):
                mode_file = meta_path.parent / f"mode_shape_{k}_rank{rank}.out"
                if not mode_file.is_file():
                    raise FileNotFoundError(
                        "ParallelModalResult.from_job: "
                        f"{meta_path.name} promises {n_modes} mode(s) but "
                        f"{mode_file.name} is missing — incomplete fetch, "
                        "or that rank died mid-harvest."
                    )
                rows.append(
                    _read_mode_row(mode_file, int(nodes.shape[0]), rank_ndf)
                )
            per_rank.append((rank, nodes, rows))

        assert ndf is not None and ndm is not None  # rank_sidecars non-empty
        all_tags = np.unique(
            np.concatenate([n for _, n, _ in per_rank])
            if per_rank
            else np.zeros((0,), dtype=np.int64)
        )
        shapes = np.zeros((n_modes, all_tags.shape[0], ndf), dtype=np.float64)
        # Per-mode amplitude scale for the agreement tolerance, computed
        # ONCE over every rank's rows. Doing it inside the merge loop
        # would both be O(n_nodes) per shared node — O(N^{5/3}) overall
        # on a 3-D partition boundary — and measure a partially-filled
        # field, so the tolerance would drift with rank ordering.
        peaks = np.ones(n_modes, dtype=np.float64)
        for k in range(n_modes):
            scale = max(
                (float(np.max(np.abs(rows[k]))) for _, _, rows in per_rank),
                default=0.0,
            )
            peaks[k] = max(scale, 1.0)

        seen: "dict[int, int]" = {}  # node tag -> rank that wrote it first
        for rank, nodes, rows in per_rank:
            slots = np.searchsorted(all_tags, nodes)
            block = (
                np.stack(rows)
                if rows
                else np.zeros((0, nodes.shape[0], ndf), dtype=np.float64)
            )
            fresh = np.asarray(
                [int(t) not in seen for t in nodes], dtype=bool
            )
            shapes[:, slots[fresh]] = block[:, fresh]
            for tag in nodes[fresh]:
                seen[int(tag)] = rank
            if not (~fresh).any():
                continue
            # Shared boundary nodes: cross-check, never overwrite. The
            # vectorised compare below is the hot path; the Python loop
            # only runs to build the error message.
            dup_slots = slots[~fresh]
            have = shapes[:, dup_slots]
            got = block[:, ~fresh]
            tol = (
                _SHARED_NODE_RTOL * np.abs(got)
                + _SHARED_NODE_ATOL_FRAC * peaks[:, None, None]
            )
            bad = np.abs(have - got) > tol
            if not bad.any():
                continue
            k, j, _ = (int(v[0]) for v in np.nonzero(bad))
            tag_i = int(nodes[~fresh][j])
            raise ValueError(
                "ParallelModalResult.from_job: shared boundary "
                f"node {tag_i} disagrees between rank "
                f"{seen[tag_i]} and rank {rank} in mode {k + 1} "
                f"— {list(have[k, j])} vs {list(got[k, j])}. Both ranks "
                "solved the same global problem, so this means "
                "they did NOT: each ran a private Lanczos over "
                "its own subdomain. Check that the deck kept "
                "'system Mumps' (ADR 0077 INV-8) and that the "
                "binary is an OpenSeesMP built at or after fork "
                "5a522b03b (PR #668) — an older one gives no "
                "error, just this."
            )
        return all_tags, shapes, int(ndm)

    @property
    def shape_nodes(self) -> np.ndarray:
        """Node tags in mode-shape row order (the recorder column order
        pinned by the deck — sorted mesh node tags)."""
        if self._shape_nodes is None:
            raise FileNotFoundError(_NO_SHAPES_MSG)
        return self._shape_nodes

    def mode_shape(self, node: "int | Node", mode: int) -> np.ndarray:
        """Return the harvested mode shape for ``node`` in ``mode``
        (1-indexed) — a length-``ndf`` vector, matching
        :meth:`EigenResult.mode_shape`. DOFs the node does not carry are
        ``0.0`` (the recorder pads to the deck's uniform dof list)."""
        field = self.mode_shape_field(mode)
        tag = _node_tag(node)
        assert self._shape_nodes is not None  # mode_shape_field guarded
        idx = int(np.searchsorted(self._shape_nodes, tag))
        if (
            idx >= self._shape_nodes.shape[0]
            or int(self._shape_nodes[idx]) != tag
        ):
            raise KeyError(
                f"ParallelModalResult.mode_shape: node {tag} is not in the "
                "harvested field (mesh nodes only — bridge-declared extra "
                "nodes are not recorded)."
            )
        return np.asarray(field[idx], dtype=np.float64).copy()

    def mode_shape_field(self, mode: int) -> np.ndarray:
        """Full-field eigenvector for ``mode`` (1-indexed) — an
        ``(n_nodes, ndf)`` array, rows in :attr:`shape_nodes` order."""
        if self._shapes is None:
            raise FileNotFoundError(_NO_SHAPES_MSG)
        m = int(mode)
        if not 1 <= m <= self.n_modes:
            raise IndexError(
                f"ParallelModalResult.mode_shape_field: mode {mode} out of "
                f"range — the band found {self.n_modes} mode(s) "
                "(1-indexed)."
            )
        return np.asarray(self._shapes[m - 1], dtype=np.float64)

    def to_native(
        self, path: "str | Any", fem: Any, *, model_h5: "str | Any | None" = None,
    ) -> None:
        """Write the harvested modes as mode-kind stages in a native
        results H5 (ADR 0077 P4 viewer binding).

        Produces the mode-stage layout of :func:`_write_mode_stages`: one
        stage per mode (``name="mode_<k>"``, ``kind="mode"``, with
        ``eigenvalue`` / ``frequency_hz`` / ``period_s`` /
        ``mode_index``) at a single ``time = [0.0]`` station, carrying
        ``displacement_x/y/z`` (the first ``min(3, ndm, ndf)`` shape
        columns) and the ``rotation_*`` the model's DOF roles allow, so
        the existing surface consumes the distributed run with zero new
        viewer code::

            res = ParallelModalResult.from_job(job_dir)
            res.to_native("modes.h5", fem, model_h5="model.h5")
            r = Results.from_native("modes.h5", fem=fem)
            r.modes[0].frequency_hz;  r.viewer()

        **Rotations follow the DOF roles.** The deck records DOFs
        ``1..ndf`` for every node and pads a short node with zeros, so a
        row's entry in a rotation slot may be a rotation, a u-p node's
        pore pressure, or padding; a 2-D node of ndf 3 is ``(ux, uy, rz)``
        on a frame and ``(ux, uy, p)`` on a u-p element. With
        ``model_h5`` (the bridge archive the deck was emitted from,
        ``ops.h5``) each node's columns are laid onto ``ux..rz`` by its
        own ndf (``/opensees/nodes_ndf``), and a rotation slot holds the
        rotation only where
        :func:`~apeGmsh.opensees._recorder_translate.model_h5_rotation_roles`
        says so: a 2-D frame keeps its ``rotation_z``, and a pore
        pressure is never written as a rotation. A node whose slot holds
        no rotation writes 0.0 there; a slot the roles cannot settle (an
        element-less node beside a u-p class) also writes 0.0, with one
        :class:`~apeGmsh.opensees.analysis.eigen.RotationSlotUnresolvedWarning`.
        :meth:`to_ladruno` and :meth:`EigenResult.to_native` follow the
        same rule.

        Without ``model_h5`` nothing is guessed. A deck whose DOF list
        reaches no rotation slot (2-D ndf 2, 3-D ndf 3 or 4) writes
        translations only, silently. Otherwise, in this order:

        1. **The deck's own archive.** ``modal_deck`` leaves the bridge's
           archive at ``default_artifact_dir() / f"{fem.model_name}.h5"``.
           When that file's ``/meta/lineage`` ``fem_hash`` is
           ``fem.snapshot_id``, it holds ``/opensees``, and its elements
           touch every fem node on a solid cell, its roles are used as if
           passed. A file there that fails a check (another run, or
           another bridge on the same mesh and name that wrote it last)
           is ignored, with one :class:`StaleModelArchiveWarning`.
        2. **A 3-D deck of ndf 6 on a fem with no solid (3-D) cell**
           writes DOFs 4-6 of every node as rotations, silently, as main
           did: without a 3-D cell there is no u-p element, so the slot
           holds a frame or shell rotation or zero padding, never a
           pressure.
        3. **Otherwise** each node's own ndf is read from ``fem``
           (``fem.nodes.ndf_for``): a 3-D node of ndf 6 writes its
           rotations, and a node whose own DOFs reach no rotation slot (a
           3-D u-p node, ndf 4) writes 0.0 there, both silently. A node
           left unsettled, a 2-D node of ndf 3 (frame or u-p) or one
           whose ndf ``fem`` does not carry, writes no rotation (0.0 in
           the slot, or no ``rotation_*`` at all when no node holds one),
           with one :class:`RotationRolesUnavailableWarning` that names
           the nodes and says to pass ``model_h5=``. A fem snapshot the
           bridge built carries no per-node ndf, so for it every node of
           such a deck is unsettled.

        ``fem`` must be the same FEM snapshot the deck was emitted from
        (the recorder column order is the sorted mesh node tags).
        A non-positive eigenvalue warns and writes ``frequency_hz =
        period_s = 0`` (the mode-stage convention). Requires the
        P3 shape harvest; a run dir without the sidecar fails loud.

        Raises
        ------
        FileNotFoundError
            The run dir carries no mode-shape harvest.
        ValueError
            ``model_h5`` has another ``ndm`` than the sidecar, or a
            harvested node is not a node of its fem or has no ndf there.
            Nothing is written.
        """
        import warnings

        from ...results.writers._native import NativeWriter
        from .._internal.build.recorders import _node_layout
        from .eigen import _ladruno_slots, _warn_unresolved

        where = "ParallelModalResult.to_native"
        if self._shapes is None or self._shape_nodes is None:
            raise FileNotFoundError(_NO_SHAPES_MSG)
        shapes = self._shapes
        ndf = int(shapes.shape[2])
        rec = [int(n) for n in self._shape_nodes]
        rotation_slots: tuple[int, ...] = ()
        roles_h5: "str | Any | None" = model_h5
        if model_h5 is None:
            ndm = int(self._shape_ndm if self._shape_ndm is not None else 3)
            reach = [s for s in _node_layout(ndf, ndm)
                     if s in _NATIVE_ROTATION_SLOTS.get(ndm, ())]
            if reach:
                roles_h5 = _discover_archive(fem, where=where)
        if roles_h5 is None:
            n_trans = min(3, ndm, ndf)
            if not reach:
                spatial = np.zeros(shapes.shape[:2] + (6,), dtype=np.float64)
                spatial[..., :n_trans] = shapes[..., :n_trans]
            elif ndm == 3 and ndf == 6 and not _has_solid_cells(fem):
                # No 3-D cell, so no u-p (or other solid) element: every
                # node is a frame, shell or element-less node, and DOFs
                # 4-6 are rotations or zero padding (a truss or membrane
                # node), never a pressure. What main wrote, unchanged.
                spatial, rotation_slots = _native_fields(
                    shapes, [_node_layout(6, 3)] * len(rec),
                    np.ones(len(rec), dtype=bool), ndm)
            else:
                layouts, rotations, unsettled = _fem_ndf_slots(
                    fem, rec, ndf, ndm)
                spatial, rotation_slots = _native_fields(
                    shapes, layouts, rotations, ndm)
                if unsettled:
                    node, why = unsettled[0]
                    warnings.warn(
                        f"{where}: {len(unsettled)} node(s) "
                        f"({[n for n, _ in unsettled[:10]]}) write no "
                        f"rotation: for node {node}, {why}. The deck "
                        f"records DOFs 1..{ndf} of every node, so their "
                        "entry in a rotation slot may be a rotation, a u-p "
                        "node's pore pressure or zero padding, and only the "
                        "model's DOF roles can say which. Pass model_h5= "
                        "(the archive the deck was emitted from, ops.h5) "
                        "to write the rotations the roles allow.",
                        RotationRolesUnavailableWarning,
                        stacklevel=2,
                    )
        else:
            model = _deck_archive(roles_h5, rec, self._shape_ndm, where=where)
            ndm = int(model.ndm)
            n_trans = min(3, ndm, ndf)
            layouts, rotations, unresolved = _ladruno_slots(
                Path(roles_h5), rec, model.nodes_ndf, ndm, where=where)
            spatial, rotation_slots = _native_fields(
                shapes, layouts, rotations, ndm)
            _warn_unresolved(where, unresolved, 2)

        writer = NativeWriter(path)
        # No ``opensees_backend`` / ``opensees_build`` (F2-b): the modes
        # came from a modal_deck run by a separate OpenSees/OpenSeesMP
        # binary, which the run dir does not record. The in-process
        # module's BackendInfo would name the wrong build, so the attrs
        # stay absent (unknown), as for a recorder transcode.
        writer.open(
            fem=fem,
            source_type="parallel_modal",
            source_path="<modal_deck harvest>",
        )
        try:
            _write_mode_stages(
                writer,
                self.eigenvalues,
                self._shape_nodes,
                lambda k: spatial[k - 1],
                n_trans=n_trans,
                rotation_slots=rotation_slots,
            )
        finally:
            writer.close()

    def to_ladruno(self, path: "str | Any", *, model_h5: "str | Any") -> Path:
        """Write the harvested modes as a stamped ``.ladruno`` and return
        its path (ADR 0122; the twin of :meth:`to_native`).

        One ``KIND="eigen"`` ``MODEL_STAGE`` holds every mode in the
        layout the fork recorder writes for ``modesOfVibration`` +
        ``modesOfVibrationRotational`` (the table is in
        :mod:`apeGmsh.results.readers._ladruno_writer`), over
        :attr:`shape_nodes`. Each node's shape columns are laid onto
        ``Ux..Rz`` by its own DOF layout (its ndf in the archive's
        ``/opensees/nodes_ndf``; the deck pads every node to one dof
        list). A rotation slot holds the rotation only where the model's
        DOF roles say so, so a u-p node's pore pressure is never written
        as one; a slot the roles cannot settle is written as 0.0 with one
        :class:`~apeGmsh.opensees.analysis.eigen.RotationSlotUnresolvedWarning`.
        A non-positive eigenvalue writes 0 Hz / 0 s and warns once for
        its mode.

        ``model_h5`` is the bridge archive the deck was emitted from
        (``ops.h5``). ``MODEL/NODES`` is its fem, ``MODEL/ELEMENTS`` is
        empty (``INFO/PROVENANCE@unsupported`` says so), and the file is
        stamped against it::

            res = ParallelModalResult.from_job(job_dir)
            p = res.to_ladruno("modes.ladruno", model_h5="model.h5")
            r = Results.from_ladruno(p, model_h5="model.h5")

        Raises
        ------
        FileNotFoundError
            The run dir carries no mode-shape harvest.
        ValueError
            The archive's ``ndm`` differs from the sidecar's, or a
            harvested node is not a node of the archive's fem or has no
            ndf there, or ``path`` is ``model_h5`` or an ``.h5``. Nothing is
            written.
        """
        from .eigen import _ladruno_slots, _spatial, _warn_unresolved

        where = "ParallelModalResult.to_ladruno"
        if self._shapes is None or self._shape_nodes is None:
            raise FileNotFoundError(_NO_SHAPES_MSG)
        check_target(path, model_h5)
        shapes = self._shapes
        rec = [int(n) for n in self._shape_nodes]
        model = _deck_archive(model_h5, rec, self._shape_ndm, where=where)
        ndm = int(model.ndm)
        fem = model.fem
        layouts, rotations, unresolved = _ladruno_slots(
            Path(model_h5), rec, model.nodes_ndf, ndm, where=where)
        spatial = np.stack([_spatial(rows, layouts) for rows in shapes])
        _warn_unresolved(where, unresolved, 2)
        return write_modes(
            path, model_h5=model_h5,
            node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
            coords=np.asarray(fem.nodes.coords, dtype=np.float64),
            ndm=ndm,
            eigenvalues=self.eigenvalues,
            fields=lambda k: spatial[k - 1],
            rotations=rotations,
            record_ids=self._shape_nodes,
            source="modal_deck", stacklevel=3,
        )

    def participation_factors(self, component: str) -> np.ndarray:
        """Not available in a distributed run — modalProperties is MPI-blind."""
        _ = component
        raise NotImplementedError(_MPI_BLIND_MSG)

    @property
    def mass_ratios(self) -> np.ndarray:
        """Not available in a distributed run — modalProperties is MPI-blind."""
        raise NotImplementedError(_MPI_BLIND_MSG)
