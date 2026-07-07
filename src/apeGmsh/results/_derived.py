"""Derived stress/strain scalars — computed on read from the raw tensor.

Result files store the raw 6-component Voigt stress/strain tensor per
Gauss point (``stress_xx`` … ``stress_xz``; ``strain_xx`` … in 3D, the
three in-plane components in 2D). Invariant and principal measures
(von Mises, Tresca, J2, principal values, …) are **not** stored — they
are cheap functions of those columns, so we compute them here at read
time rather than bloating every recorder file.

The canonical names this module serves are declared in
``apeGmsh._vocabulary`` (``DERIVED_STRESS_SCALARS`` /
``DERIVED_STRAIN_SCALARS``). ``equivalent_plastic_strain`` is *not* here:
OpenSees emits it directly as a material-state scalar, so it is read from
storage, not derived.

Two conventions worth stating up front:

* **Engineering shear strain.** OpenSees reports ``strain_xy`` as the
  *engineering* shear γ_xy = 2·ε_xy. Building the strain tensor for an
  eigen/invariant computation therefore **halves** the shear components;
  skip this and every principal strain is wrong.

* **2-D out-of-plane component.** A plane element stores only the three
  in-plane components. We assemble a full 3×3 with the unavailable
  out-of-plane normal set to zero — exact for σ_zz=0 (plane stress) on
  the stress side and ε_zz=0 (plane strain) on the strain side. A model
  is one or the other, so at most one side is exact in 2-D; use a 3-D
  model (or the deferred ν-aware path) for exact out-of-plane recovery.

All functions are vectorized over a leading ``(T, N)`` shape (time steps
× points) and return a ``(T, N)`` float64 array.
"""
from __future__ import annotations

import numpy as np

from typing import Iterable

from .._vocabulary import (
    DERIVED_SHELL_SCALARS,
    DERIVED_STRAIN_SCALARS,
    DERIVED_STRESS_SCALARS,
    STRAIN_2D,
    STRESS_2D,
    expand_shorthand,
)

__all__ = [
    "is_derived",
    "is_shell_derived",
    "base_components_for",
    "shell_base_components",
    "available_derived",
    "compute",
    "compute_shell",
    "principal_frame",
]

_STRESS_DERIVED: frozenset[str] = frozenset(DERIVED_STRESS_SCALARS)
_STRAIN_DERIVED: frozenset[str] = frozenset(DERIVED_STRAIN_SCALARS)
_SHELL_DERIVED: frozenset[str] = frozenset(DERIVED_SHELL_SCALARS)

# In-plane shell resultants needed to recover surface stress.
_SHELL_BASE: tuple[str, ...] = (
    "membrane_force_xx", "membrane_force_yy", "membrane_force_xy",
    "bending_moment_xx", "bending_moment_yy", "bending_moment_xy",
)

# Voigt suffix → symmetric-tensor (row, col) placements.
_SUFFIX_SLOTS: dict[str, tuple[tuple[int, int], ...]] = {
    "xx": ((0, 0),),
    "yy": ((1, 1),),
    "zz": ((2, 2),),
    "xy": ((0, 1), (1, 0)),
    "yz": ((1, 2), (2, 1)),
    "xz": ((0, 2), (2, 0)),
}


def is_derived(name: str) -> bool:
    """True if ``name`` is a computed derived stress/strain scalar."""
    return name in _STRESS_DERIVED or name in _STRAIN_DERIVED


def is_shell_derived(name: str) -> bool:
    """True if ``name`` is a derived shell-resultant scalar (needs thickness)."""
    return name in _SHELL_DERIVED


def shell_base_components(name: str) -> tuple[str, ...]:
    """Shell resultant columns a shell-derived scalar needs."""
    if name in _SHELL_DERIVED:
        return _SHELL_BASE
    raise ValueError(f"'{name}' is not a derived shell scalar.")


def base_components_for(name: str, *, ndm: int) -> tuple[str, ...]:
    """Raw tensor columns a derived scalar needs, clipped to ``ndm``.

    Returns the ``stress_*`` set for a stress measure, the ``strain_*``
    set for a strain measure (3 components in 2-D, 6 in 3-D). Raises
    ``ValueError`` if ``name`` is not a derived scalar.
    """
    if name in _STRESS_DERIVED:
        return expand_shorthand("stress", ndm=ndm)
    if name in _STRAIN_DERIVED:
        return expand_shorthand("strain", ndm=ndm)
    raise ValueError(f"'{name}' is not a derived stress/strain scalar.")


def available_derived(stored: Iterable[str]) -> list[str]:
    """Derived scalars computable from a set of stored component names.

    Advertises the stress-derived set (for the viewer picker / matplotlib
    contour) only when a *complete* tensor is stored — the three in-plane
    components ``{xx, yy, xy}`` at minimum, which covers a genuine 2-D
    plane file (exactly those three) and a full 3-D file (all six). A
    partial tensor (e.g. only ``stress_xx``) is not advertised: computing
    an invariant from it would silently treat the missing components as
    zero. Explicitly requesting a derived scalar on a partial tensor still
    works — this only governs what the menu offers.
    """
    have = set(stored)
    out: list[str] = []
    if have.issuperset(STRESS_2D):
        out.extend(DERIVED_STRESS_SCALARS)
    if have.issuperset(STRAIN_2D):
        out.extend(DERIVED_STRAIN_SCALARS)
    if have.issuperset(_SHELL_BASE):
        out.extend(DERIVED_SHELL_SCALARS)
    return out


def compute(
    name: str, columns: dict[str, np.ndarray], *, ndm: int,
    plane: str | None = None, nu: float | None = None,
) -> np.ndarray:
    """Compute a derived scalar from its raw tensor columns.

    ``columns`` maps canonical base names (``stress_xx`` …) to ``(T, N)``
    arrays. Only the components returned by :func:`base_components_for`
    need be present; any not supplied are treated as zero. Returns a
    ``(T, N)`` float64 array.

    ``plane`` / ``nu`` control out-of-plane recovery for 2-D data (when
    the ``_zz`` component is absent). ``plane=None`` (default) leaves the
    out-of-plane component at zero — plane stress for a stress measure,
    plane strain for a strain measure. ``plane="strain"`` recovers
    σ_zz = ν(σ_xx+σ_yy) for stress (ν required); ``plane="stress"``
    recovers ε_zz = -ν/(1-ν)(ε_xx+ε_yy) for strain (ν required). See
    :func:`_assemble_tensor`.
    """
    if name in _STRESS_DERIVED:
        prefix, halve_shear = "stress", False
    elif name in _STRAIN_DERIVED:
        prefix, halve_shear = "strain", True
    else:
        raise ValueError(f"'{name}' is not a derived stress/strain scalar.")

    tensor = _assemble_tensor(
        columns, prefix=prefix, halve_shear=halve_shear, plane=plane, nu=nu,
    )

    if name in ("von_mises_stress", "von_mises_strain"):
        return _von_mises(tensor, engineering=halve_shear)
    if name in ("j2_stress", "j2_strain"):
        return _j2(tensor)
    if name == "mean_stress":
        return _mean(tensor)
    if name == "pressure_hydrostatic":
        return -_mean(tensor)
    if name == "volumetric_strain":
        return _trace(tensor)
    if name == "j3_stress":
        return _j3(tensor)
    if name == "lode_angle":
        return _lode_angle(tensor)
    if name == "stress_triaxiality":
        return _triaxiality(tensor)
    if name in ("tresca_stress", "max_shear_stress",
                "max_shear_strain",
                "principal_stress_1", "principal_stress_2",
                "principal_stress_3",
                "principal_strain_1", "principal_strain_2",
                "principal_strain_3"):
        # eigenvalues descending: p[..., 0] >= p[..., 1] >= p[..., 2]
        principals = np.linalg.eigvalsh(tensor)[..., ::-1]
        p1, p3 = principals[..., 0], principals[..., 2]
        if name == "tresca_stress":
            return np.ascontiguousarray(p1 - p3)
        if name in ("max_shear_stress", "max_shear_strain"):
            return np.ascontiguousarray(0.5 * (p1 - p3))
        idx = int(name[-1]) - 1
        return np.ascontiguousarray(principals[..., idx])

    raise ValueError(f"'{name}' is not a derived stress/strain scalar.")


def principal_frame(
    columns: dict[str, np.ndarray], *, prefix: str = "stress",
    plane: str | None = None, nu: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Principal values (descending) + eigenvectors of the tensor.

    ``columns`` maps ``stress_*`` / ``strain_*`` names to arrays of a
    common leading shape ``S``. Returns ``(values, vectors)`` where
    ``values`` is ``S + (3,)`` sorted descending (p₁ ≥ p₂ ≥ p₃) and
    ``vectors[..., :, i]`` is the unit eigenvector for ``values[..., i]``.
    For strain the engineering-shear halving is applied; 2-D out-of-plane
    recovery follows ``plane`` / ``nu`` (see :func:`_assemble_tensor`).
    """
    halve = prefix == "strain"
    tensor = _assemble_tensor(
        columns, prefix=prefix, halve_shear=halve, plane=plane, nu=nu,
    )
    w, v = np.linalg.eigh(tensor)      # ascending; columns of v are eigvecs
    return w[..., ::-1], v[..., ::-1]  # reorder to descending


def compute_shell(
    name: str, columns: dict[str, np.ndarray], *, thickness: float,
) -> np.ndarray:
    """Shell von Mises from stress resultants (through-thickness envelope).

    Recovers the extreme-fibre in-plane surface stress from the membrane
    forces ``N`` (per length) and bending moments ``M`` (per length):

        σ_surface = N / t ± 6 M / t²

    and returns the larger of the top / bottom von Mises per point — the
    worst-case surface demand. ``columns`` holds the six in-plane
    resultants (``membrane_force_*`` / ``bending_moment_*``); ``thickness``
    is the shell thickness ``t``. Transverse shears are not included.
    """
    if name not in _SHELL_DERIVED:
        raise ValueError(f"'{name}' is not a derived shell scalar.")
    t = float(thickness)
    if t <= 0.0:
        raise ValueError(f"thickness must be > 0 (got {thickness!r}).")

    membrane = {k: np.asarray(columns[f"membrane_force_{k}"], dtype=np.float64)
                for k in ("xx", "yy", "xy")}
    bending = {k: np.asarray(columns[f"bending_moment_{k}"], dtype=np.float64)
               for k in ("xx", "yy", "xy")}

    surfaces = []
    for sign in (1.0, -1.0):
        cols = {
            f"stress_{k}": membrane[k] / t + sign * 6.0 * bending[k] / (t * t)
            for k in ("xx", "yy", "xy")
        }
        # Plane-stress surface tensor (σ_zz = 0): reuse the continuum path.
        tensor = _assemble_tensor(cols, prefix="stress", halve_shear=False)
        surfaces.append(_von_mises(tensor, engineering=False))
    return np.maximum(surfaces[0], surfaces[1])


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------

def _assemble_tensor(
    columns: dict[str, np.ndarray], *, prefix: str, halve_shear: bool,
    plane: str | None = None, nu: float | None = None,
) -> np.ndarray:
    """Build a symmetric ``(T, N, 3, 3)`` tensor from Voigt columns.

    Missing components are zero. Off-diagonal (shear) terms are halved
    when ``halve_shear`` — the engineering→tensor strain correction.

    When the ``_zz`` normal is absent (2-D data), the out-of-plane
    component is recovered per the ``plane`` idealization:

    * ``plane=None`` — leave it zero (plane stress for σ, plane strain
      for ε; the historical default).
    * ``plane="strain"`` — σ_zz = ν(σ_xx+σ_yy) for a stress tensor
      (ν required); ε_zz stays 0 (plane strain kinematics).
    * ``plane="stress"`` — ε_zz = -ν/(1-ν)(ε_xx+ε_yy) for a strain
      tensor (ν required); σ_zz stays 0 (plane stress).

    ``nu`` is only consulted when a nonzero out-of-plane fill is required
    and the ``_zz`` component is missing; a missing ``nu`` in that case
    raises ``ValueError``.
    """
    if plane not in (None, "stress", "strain"):
        raise ValueError(
            f"plane must be None, 'stress', or 'strain' (got {plane!r})."
        )
    ref = next(iter(columns.values()))
    base_shape = np.asarray(ref).shape
    tensor = np.zeros(base_shape + (3, 3), dtype=np.float64)
    for suffix, slots in _SUFFIX_SLOTS.items():
        col = columns.get(f"{prefix}_{suffix}")
        if col is None:
            continue
        val = np.asarray(col, dtype=np.float64)
        if halve_shear and len(slots) == 2:  # off-diagonal shear term
            val = 0.5 * val
        for (i, j) in slots:
            tensor[..., i, j] = val

    if f"{prefix}_zz" not in columns:      # 2-D data: recover out-of-plane
        _fill_out_of_plane(tensor, prefix=prefix, plane=plane, nu=nu)
    return tensor


def _fill_out_of_plane(
    tensor: np.ndarray, *, prefix: str, plane: str | None, nu: float | None,
) -> None:
    """Set ``tensor[...,2,2]`` for 2-D data per the plane idealization."""
    xx = tensor[..., 0, 0]
    yy = tensor[..., 1, 1]
    if prefix == "stress" and plane == "strain":
        if nu is None:
            raise ValueError(
                "plane='strain' needs nu= to recover the out-of-plane "
                "stress σ_zz = ν(σ_xx+σ_yy) on 2-D data."
            )
        tensor[..., 2, 2] = nu * (xx + yy)
    elif prefix == "strain" and plane == "stress":
        if nu is None:
            raise ValueError(
                "plane='stress' needs nu= to recover the out-of-plane "
                "strain ε_zz = -ν/(1-ν)(ε_xx+ε_yy) on 2-D data."
            )
        tensor[..., 2, 2] = -nu / (1.0 - nu) * (xx + yy)
    # else: out-of-plane stays 0 (plane=None, or the naturally-zero combo)


def _trace(tensor: np.ndarray) -> np.ndarray:
    return (tensor[..., 0, 0] + tensor[..., 1, 1] + tensor[..., 2, 2])


def _mean(tensor: np.ndarray) -> np.ndarray:
    return _trace(tensor) / 3.0


def _deviator(tensor: np.ndarray) -> np.ndarray:
    dev = tensor.copy()
    m = _mean(tensor)
    for i in range(3):
        dev[..., i, i] -= m
    return dev


def _j2(tensor: np.ndarray) -> np.ndarray:
    """Second deviatoric invariant J2 = 1/2 s_ij s_ij."""
    dev = _deviator(tensor)
    return 0.5 * np.sum(dev * dev, axis=(-2, -1))


def _j3(tensor: np.ndarray) -> np.ndarray:
    """Third deviatoric invariant J3 = det(s)."""
    return np.linalg.det(_deviator(tensor))


def _lode_angle(tensor: np.ndarray) -> np.ndarray:
    """Lode angle in degrees, in ``[-30, 30]``.

    Convention: ``sin(3θ) = (3√3/2) · J3 / J2^{3/2}``. So a state with
    one distinct larger principal (σ₁ > σ₂ = σ₃, e.g. (2, -1, -1)) gives
    ``+30°``; the mirror (σ₁ = σ₂ > σ₃, e.g. (1, 1, -2)) gives ``-30°``;
    a state with the middle principal at the mean (pure shear) gives
    ``0°``. Undefined for a hydrostatic state (J2 = 0) → ``NaN``.
    """
    j2 = _j2(tensor)
    j3 = _j3(tensor)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (3.0 * np.sqrt(3.0) / 2.0) * j3 / np.power(j2, 1.5)
    ratio = np.clip(ratio, -1.0, 1.0)          # guards float overshoot
    return np.degrees(np.arcsin(ratio) / 3.0)  # NaN propagates where J2==0


def _triaxiality(tensor: np.ndarray) -> np.ndarray:
    """Stress triaxiality η = σ_mean / σ_von-Mises.

    Unbounded under near-hydrostatic states; returns ``NaN`` where the
    von Mises stress is ~0 rather than ``±inf``.
    """
    vm = _von_mises(tensor, engineering=False)
    mean = _mean(tensor)
    out = np.full_like(vm, np.nan)
    nz = vm > 0.0
    out[nz] = mean[nz] / vm[nz]
    return out


def _von_mises(tensor: np.ndarray, *, engineering: bool) -> np.ndarray:
    """von Mises equivalent measure.

    Stress: σ_vm = sqrt(3 J2). Strain: ε_vm = sqrt(2/3 · e_ij e_ij),
    the equivalent strain work-conjugate to σ_vm under J2 flow. The
    strain tensor is already engineering-corrected upstream, so both
    reduce to a coefficient on e:e = 2·J2.
    """
    j2 = _j2(tensor)
    coeff = (4.0 / 3.0) if engineering else 3.0  # ε_vm=sqrt(4/3 J2); σ_vm=sqrt(3 J2)
    return np.sqrt(np.maximum(coeff * j2, 0.0))
