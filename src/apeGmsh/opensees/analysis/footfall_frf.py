"""Modal FRF matrix for the footfall evaluation (ADR 0109 D2 / D4 / D6).

The displacement frequency-response between an excitation node ``i``
and a response node ``j``, summed over the mode basis of ONE ``eigen``
+ ``modalProperties`` pair::

    H_ij(Ω) = Σ_a  φ_ia φ_ja / ( m̃_a (ω_a² − Ω² + 2 i ξ_a ω_a Ω) )
    A_ij(Ω) = −Ω² H_ij(Ω)

with the ``e^{+iΩt}`` sign convention of fork ADR-44 §4.4 (the response
lags 90° at resonance), ``φ`` read through
:meth:`~apeGmsh.opensees.analysis.modal.ModalPropertiesResult.mode_shape`
and ``ξ_a`` from the ADR 0075 damping channel (D6).

**``m̃_a`` is taken as 1 and asserted, not looked up** (D2).
``modalProperties -return`` does not export the generalised modal
masses, but it exports both ``partiFactor<C>`` (``Γ = L/m̃``) and
``partiMass<C>`` (``L²/m̃``), whose ratio is ``m̃`` for every component
with a non-zero participation factor.  Both stock eigensolvers happen
to return M-orthonormal vectors (``-fullGenLapack`` is LAPACK
``dsygv``; ``-genBandArpack`` is ARPACK with ``bmat='G'``) — a property
of the solvers, not a documented OpenSees contract, so
:func:`assert_unit_generalized_mass` checks it and **refuses** rather
than silently rescaling.  A mode whose every ``Γ_C`` is zero (a doubly
antisymmetric floor mode on a symmetric grid) cannot be checked this
way; at least one checkable mode must exist and all checkable modes
must pass, and the count is recorded as ``"asserted(k of p)"``.

The assertion takes the raw properties mapping, not the result object,
so it is testable — and mutation-testable — without a live domain.

Nothing here imports openseespy: the mode shapes arrive through the
``ModalPropertiesResult`` the caller already holds.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .modal import _damping_channel_args, _node_tag

if TYPE_CHECKING:
    from ..node import Node
    from .modal import ModalPropertiesResult


__all__ = ["FRFMatrix", "assert_unit_generalized_mass", "frf_matrix", "grid_for"]


# The six global components ``DomainModalProperties`` can report
# (2-D models expose MX / MY / RMZ only).
_COMPONENTS = ("MX", "MY", "MZ", "RMX", "RMY", "RMZ")

# D2 tolerances: a participation factor below the floor carries no
# usable information about the scale (0/0); a surviving ratio must be
# 1 to within round-off of the C++ side's own products.
_GAMMA_FLOOR = 1.0e-8
_MASS_RTOL = 1.0e-6

_UNORM_MSG = (
    "footfall FRF: modal_properties(unorm=True) is refused — "
    "``-unorm`` rescales modalProperties' own eigenvector copy "
    "(per-mode 1/max|v|) before computing the participation factors, "
    "while mode_shape() returns the RAW domain eigenvector, so the "
    "generalised-mass assertion and the modal sum would be in "
    "different bases.  Re-run modal_properties with the default "
    "unorm=False."
)


def assert_unit_generalized_mass(
    properties: Mapping[str, Sequence[float]],
    *,
    context: str = "footfall FRF",
) -> str:
    """Assert the eigenvector basis is mass-normalised (ADR 0109 D2).

    For every mode and every component ``C`` present in ``properties``
    with ``|partiFactor_C| > 1e-8``, ``partiMass_C / partiFactor_C²``
    is the generalised modal mass ``m̃`` and must be ``1 ± 1e-6``.

    Returns the ``"asserted(k of p)"`` provenance string — ``k``
    checkable modes out of ``p`` in the basis.

    Raises
    ------
    ValueError
        If no component pair is present at all, if the arrays disagree
        in length, if no mode is checkable, or if any checkable mode
        fails — the message names the failing mode.
    """
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for component in _COMPONENTS:
        factor_key = f"partiFactor{component}"
        mass_key = f"partiMass{component}"
        if factor_key in properties and mass_key in properties:
            pairs.append((
                component,
                np.asarray(properties[factor_key], dtype=np.float64),
                np.asarray(properties[mass_key], dtype=np.float64),
            ))
    if not pairs:
        raise ValueError(
            f"{context}: the modalProperties dict carries no "
            f"partiFactor<C>/partiMass<C> pair for any of "
            f"{list(_COMPONENTS)} — the eigenvector scale cannot be "
            "established (ADR 0109 D2)."
        )

    n_modes = int(pairs[0][1].shape[0])
    for component, factors, masses in pairs:
        if factors.shape != (n_modes,) or masses.shape != (n_modes,):
            raise ValueError(
                f"{context}: component {component!r} has "
                f"{factors.shape[0]} partiFactor / {masses.shape[0]} "
                f"partiMass entries, expected {n_modes} — the "
                "modalProperties dict is inconsistent."
            )

    checkable = 0
    failures: list[str] = []
    for mode in range(n_modes):
        mode_checked = False
        for component, factors, masses in pairs:
            gamma = float(factors[mode])
            if abs(gamma) <= _GAMMA_FLOOR:
                continue
            mode_checked = True
            m_tilde = float(masses[mode]) / (gamma * gamma)
            if abs(m_tilde - 1.0) > _MASS_RTOL:
                failures.append(
                    f"mode {mode + 1} component {component}: "
                    f"m~ = {m_tilde!r}"
                )
        if mode_checked:
            checkable += 1

    if failures:
        raise ValueError(
            f"{context}: the eigenvectors are NOT mass-normalised — "
            f"partiMass_C / partiFactor_C^2 must be 1 +/- {_MASS_RTOL} "
            f"for every checkable component, but {len(failures)} "
            f"check(s) failed: {'; '.join(failures[:8])}"
            + (" ..." if len(failures) > 8 else "")
            + ".  The modal sum assumes m~ = 1 and there is no silent "
            "rescale (ADR 0109 D2)."
        )
    if checkable == 0:
        raise ValueError(
            f"{context}: every one of the {n_modes} modes has "
            f"|partiFactor_C| <= {_GAMMA_FLOOR} in every component, so "
            "the eigenvector scale cannot be established from "
            "modalProperties (ADR 0109 D2 refusal path).  A basis of "
            "purely antisymmetric modes needs the fork's genMass "
            "export."
        )
    return f"asserted({checkable} of {n_modes})"


def grid_for(
    props: "ModalPropertiesResult",
    f_min: float,
    f_max: float,
    n_extra: int = 30,
    cluster: float = 0.05,
) -> np.ndarray:
    """Sweep grid for the FRF over ``[f_min, f_max]`` Hz (ADR 0109 D4).

    Every modal frequency in the band, a ``±cluster`` band of 5 points
    around each (a low-damping peak must not be stepped over), plus
    ``n_extra`` linearly spaced points — sorted, unique, clipped to the
    band.
    """
    if not (0.0 <= f_min < f_max):
        raise ValueError(
            f"grid_for: need 0 <= f_min < f_max, got f_min={f_min}, "
            f"f_max={f_max}."
        )
    if n_extra < 2:
        raise ValueError(f"grid_for: n_extra must be >= 2, got {n_extra}.")
    if not (0.0 <= cluster < 1.0):
        raise ValueError(
            f"grid_for: cluster must be in [0, 1), got {cluster}."
        )
    modal = np.asarray(props.freq, dtype=np.float64)
    in_band = modal[(modal >= f_min) & (modal <= f_max)]
    parts = [np.linspace(f_min, f_max, n_extra), in_band]
    for f_a in in_band:
        parts.append(
            np.linspace(f_a * (1.0 - cluster), f_a * (1.0 + cluster), 5)
        )
    grid = np.concatenate(parts)
    grid = grid[(grid >= f_min) & (grid <= f_max)]
    return np.unique(grid)


@dataclass(frozen=True, slots=True)
class FRFMatrix:
    """The ``N_exc × N_resp`` modal FRF over one sweep grid.

    Evaluated lazily per ``(i, j)`` pair from the stored mode-shape
    columns and the per-frequency modal denominator, so *full*
    excitation never materialises the ``(n_freq, N_exc, N_resp)``
    tensor (ADR 0109 D2).
    """

    freq: np.ndarray
    """Sweep frequencies in Hz."""

    exc_nodes: tuple[int, ...]
    """Excitation node tags, in the order given."""

    resp_nodes: tuple[int, ...]
    """Response node tags, in the order given."""

    dof: int
    """The 1-based DOF the mode shapes were read at (vertical = 3 on a
    3-D floor)."""

    normalization: str
    """``"asserted(k of p)"`` — the D2 provenance of ``m̃ = 1``."""

    modal_freq: np.ndarray
    """Natural frequencies of the basis, Hz."""

    modal_damping: np.ndarray
    """Damping ratio ``ξ_a`` per mode, as resolved from the D6
    channel."""

    _phi_exc: np.ndarray
    _phi_resp: np.ndarray
    _denom: np.ndarray

    def accel(self, i: "int | Node", j: "int | Node") -> np.ndarray:
        """Complex acceleration FRF ``A_ij = −Ω² H_ij`` per unit
        harmonic force at excitation node ``i``, read at response node
        ``j`` (``e^{+iΩt}``)."""
        phi_i = self._phi_exc[self._exc_index(i)]
        phi_j = self._phi_resp[self._resp_index(j)]
        h_ij = np.sum((phi_i * phi_j)[None, :] / self._denom, axis=1)
        omega = 2.0 * np.pi * self.freq
        return np.asarray(-(omega**2) * h_ij)

    def magnitude(self, i: "int | Node", j: "int | Node") -> np.ndarray:
        """``|A_ij(Ω)|`` — acceleration per unit force, model units."""
        return np.asarray(np.abs(self.accel(i, j)))

    # -- internals --------------------------------------------------------

    def _exc_index(self, node: "int | Node") -> int:
        return _index_of(node, self.exc_nodes, "exc_nodes")

    def _resp_index(self, node: "int | Node") -> int:
        return _index_of(node, self.resp_nodes, "resp_nodes")


def frf_matrix(
    props: "ModalPropertiesResult",
    *,
    exc_nodes: "Sequence[int | Node]",
    resp_nodes: "Sequence[int | Node]",
    dof: int,
    freq: "Sequence[float] | np.ndarray",
    damp: float | None = None,
    modal_damp: Sequence[float] | None = None,
    rayleigh: tuple[float, float] | None = None,
    unorm: bool = False,
) -> FRFMatrix:
    """Build the modal FRF matrix from one ``modal_properties`` result.

    Parameters
    ----------
    props
        The :class:`~apeGmsh.opensees.analysis.modal.ModalPropertiesResult`
        whose live domain still holds the eigenvectors.
    exc_nodes, resp_nodes
        Node tags (or ``Node`` handles) of the walker and occupant
        sets; duplicates are collapsed, order is kept.
    dof
        1-based DOF the FRF is built on — the vertical translation.
    freq
        Sweep frequencies in Hz (see :func:`grid_for`).
    damp, modal_damp, rayleigh
        Exactly one ADR 0075 damping channel (D6): a uniform ratio, an
        explicit per-mode list in absolute mode order, or
        ``(a0, a1)`` converted per mode as
        ``ξ_a = a0/(2ω_a) + a1·ω_a/2``.
    unorm
        Whether ``props`` came from ``modal_properties(unorm=True)``.
        Passed by the driver; ``True`` is refused (D2).
    """
    context = "footfall FRF"
    if unorm:
        raise ValueError(_UNORM_MSG)

    eigenvalues = np.asarray(props.eigenvalues, dtype=np.float64)
    if eigenvalues.size == 0:
        raise ValueError(f"{context}: the mode basis is empty.")
    if np.any(eigenvalues <= 0.0):
        bad = [int(a) + 1 for a in np.flatnonzero(eigenvalues <= 0.0)]
        raise ValueError(
            f"{context}: mode(s) {bad} have a non-positive eigenvalue "
            "(rigid-body or numerically negative) — the modal FRF sum "
            "is undefined there.  Constrain the mechanism or drop the "
            "mode from the basis."
        )
    omega = np.sqrt(eigenvalues)
    n_modes = int(omega.size)

    if dof < 1:
        raise ValueError(f"{context}: dof is 1-based, got {dof}.")
    exc = _unique_tags(exc_nodes, "exc_nodes", context)
    resp = _unique_tags(resp_nodes, "resp_nodes", context)

    frequencies = np.asarray(freq, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError(
            f"{context}: freq must be a non-empty 1-D sequence of Hz, "
            f"got shape {frequencies.shape}."
        )

    xi = _modal_damping_ratios(
        omega=omega, damp=damp, rayleigh=rayleigh,
        modal_damp=modal_damp, context=context,
    )
    normalization = assert_unit_generalized_mass(
        props.properties, context=context,
    )

    shapes = _mode_shape_columns(
        props, tags=(*exc, *resp), dof=dof, n_modes=n_modes,
        context=context,
    )
    phi_exc = np.stack([shapes[t] for t in exc]) if exc else np.empty((0, n_modes))
    phi_resp = (
        np.stack([shapes[t] for t in resp]) if resp else np.empty((0, n_modes))
    )

    omega_grid = 2.0 * np.pi * frequencies
    denom = (
        omega[None, :] ** 2
        - omega_grid[:, None] ** 2
        + 2.0j * xi[None, :] * omega[None, :] * omega_grid[:, None]
    )

    return FRFMatrix(
        freq=frequencies,
        exc_nodes=exc,
        resp_nodes=resp,
        dof=int(dof),
        normalization=normalization,
        modal_freq=omega / (2.0 * np.pi),
        modal_damping=xi,
        _phi_exc=phi_exc,
        _phi_resp=phi_resp,
        _denom=denom,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _unique_tags(
    nodes: "Sequence[int | Node]", name: str, context: str,
) -> tuple[int, ...]:
    tags = [_node_tag(n) for n in nodes]
    if not tags:
        raise ValueError(f"{context}: {name} must carry at least one node.")
    seen: dict[int, None] = {}
    for tag in tags:
        seen.setdefault(tag, None)
    return tuple(seen)


def _index_of(
    node: "int | Node", tags: tuple[int, ...], name: str,
) -> int:
    tag = _node_tag(node)
    try:
        return tags.index(tag)
    except ValueError:
        raise KeyError(
            f"FRFMatrix: node {tag} is not in {name} — the matrix was "
            f"built over {list(tags)}."
        ) from None


def _modal_damping_ratios(
    *,
    omega: np.ndarray,
    damp: float | None,
    rayleigh: tuple[float, float] | None,
    modal_damp: Sequence[float] | None,
    context: str,
) -> np.ndarray:
    """Resolve the D6 damping channel to one ratio per mode."""
    # Reuse the ADR 0075 exactly-one-of / non-negative validator; the
    # flag tail it renders is for the fork's parser, not for us.
    _damping_channel_args(
        damp=damp, rayleigh=rayleigh, modal_damp=modal_damp,
        context=context,
    )
    n_modes = int(omega.size)
    if damp is not None:
        return np.full(n_modes, float(damp), dtype=np.float64)
    if rayleigh is not None:
        a0, a1 = float(rayleigh[0]), float(rayleigh[1])
        ratios = a0 / (2.0 * omega) + a1 * omega / 2.0
        if np.any(ratios < 0.0):
            raise ValueError(
                f"{context}: rayleigh={rayleigh} converts to a negative "
                f"damping ratio on mode(s) "
                f"{[int(a) + 1 for a in np.flatnonzero(ratios < 0.0)]} "
                "(xi_a = a0/(2 w_a) + a1 w_a/2) — a negative ratio "
                "makes the FRF grow at resonance."
            )
        return np.asarray(ratios, dtype=np.float64)
    assert modal_damp is not None  # the validator above guarantees it
    ratios = np.asarray([float(x) for x in modal_damp], dtype=np.float64)
    if ratios.size != n_modes:
        raise ValueError(
            f"{context}: modal_damp carries {ratios.size} ratios but "
            f"the basis has {n_modes} modes — the list is in absolute "
            "mode order and must cover every extracted mode."
        )
    return ratios


def _mode_shape_columns(
    props: "ModalPropertiesResult",
    *,
    tags: tuple[int, ...],
    dof: int,
    n_modes: int,
    context: str,
) -> dict[int, np.ndarray]:
    """Read ``φ[:, dof-1]`` for every distinct node once, into
    ``{tag: (p,) array}``."""
    columns: dict[int, np.ndarray] = {}
    for tag in tags:
        if tag in columns:
            continue
        row = np.empty(n_modes, dtype=np.float64)
        for mode in range(n_modes):
            vector = props.mode_shape(tag, mode + 1)
            if vector.size < dof:
                raise ValueError(
                    f"{context}: node {tag} carries {vector.size} DOFs "
                    f"but dof={dof} was requested (1-based)."
                )
            row[mode] = float(vector[dof - 1])
        columns[tag] = row
    return columns
