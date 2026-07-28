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
    "ParallelModalResult",
    "RandomResponseResult",
    "ResponseSpectrumResult",
    "SteadyStateResult",
]


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

    def to_native(self, path: "str | Any", fem: Any) -> None:
        """Write the harvested modes as mode-kind stages in a native
        results H5 (ADR 0077 P4 viewer binding).

        Produces the exact layout ``DomainCapture.capture_modes`` writes
        — one stage per mode (``name="mode_<k>"``, ``kind="mode"``, with
        ``eigenvalue`` / ``frequency_hz`` / ``period_s`` /
        ``mode_index``) carrying ``displacement_x/y/z`` (first
        ``min(3, ndm)`` shape columns) and, when the deck recorded
        ``ndf >= 6``, ``rotation_x/y/z`` (columns 4–6) at a single
        ``time = [0.0]`` station — so the existing surface consumes the
        distributed run with zero new viewer code::

            res = ParallelModalResult.from_job(job_dir)
            res.to_native("modes.h5", fem)
            r = Results.from_native("modes.h5", fem=fem)
            r.modes[0].frequency_hz;  r.viewer()

        ``fem`` must be the same FEM snapshot the deck was emitted from
        (the recorder column order is the sorted mesh node tags).
        A non-positive eigenvalue warns and writes ``frequency_hz =
        period_s = 0`` (the ``capture_modes`` convention). Requires the
        P3 shape harvest — a run dir without the sidecar fails loud.
        """
        import math
        import warnings

        from ...results.writers._native import NativeWriter

        if self._shapes is None or self._shape_nodes is None:
            raise FileNotFoundError(_NO_SHAPES_MSG)
        ndf = int(self._shapes.shape[2])
        ndm = int(self._shape_ndm if self._shape_ndm is not None else 3)
        axes = ("x", "y", "z")

        writer = NativeWriter(path)
        writer.open(
            fem=fem,
            source_type="parallel_modal",
            source_path="<modal_deck harvest>",
        )
        try:
            for mode_idx, lam in enumerate(self.eigenvalues, start=1):
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
                        stacklevel=2,
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
                field = self._shapes[mode_idx - 1]
                components: dict[str, np.ndarray] = {}
                for axis_idx in range(min(3, ndm, ndf)):
                    components[f"displacement_{axes[axis_idx]}"] = (
                        field[:, axis_idx][None, :]
                    )
                if ndf >= 6:
                    for axis_idx in range(3):
                        components[f"rotation_{axes[axis_idx]}"] = (
                            field[:, 3 + axis_idx][None, :]
                        )
                writer.write_nodes(
                    sid, "partition_0",
                    node_ids=self._shape_nodes,
                    components=components,
                )
                writer.end_stage()
        finally:
            writer.close()

    def participation_factors(self, component: str) -> np.ndarray:
        """Not available in a distributed run — modalProperties is MPI-blind."""
        _ = component
        raise NotImplementedError(_MPI_BLIND_MSG)

    @property
    def mass_ratios(self) -> np.ndarray:
        """Not available in a distributed run — modalProperties is MPI-blind."""
        raise NotImplementedError(_MPI_BLIND_MSG)
