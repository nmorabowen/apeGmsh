"""Canonical component names and shorthand expansion.

The Results module speaks one canonical name per result component,
chosen to be verbose and human-readable. Backend adapters (MPCO,
recorder transcoders, domain capture) translate to these names on
the way in. Users can also write shorthands (``"displacement"``,
``"stress"``) which expand here, clipped to the active ``ndm``/``ndf``.

This is the single source of truth for "what components exist".
"""
from __future__ import annotations

from typing import Iterable


# =====================================================================
# Tensor / vector index suffixes
# =====================================================================

_TENSOR_INDICES: tuple[str, ...] = ("xx", "yy", "zz", "xy", "yz", "xz")
_VECTOR_AXES: tuple[str, ...] = ("x", "y", "z")


# =====================================================================
# Canonical component names by category
# =====================================================================

NODAL_KINEMATICS: tuple[str, ...] = (
    "displacement_x", "displacement_y", "displacement_z",
    "rotation_x", "rotation_y", "rotation_z",
    "velocity_x", "velocity_y", "velocity_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "acceleration_x", "acceleration_y", "acceleration_z",
    "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z",
    "displacement_increment_x", "displacement_increment_y", "displacement_increment_z",
)

NODAL_FORCES: tuple[str, ...] = (
    "force_x", "force_y", "force_z",
    "moment_x", "moment_y", "moment_z",
    "reaction_force_x", "reaction_force_y", "reaction_force_z",
    "reaction_moment_x", "reaction_moment_y", "reaction_moment_z",
    "pore_pressure", "pore_pressure_rate",
)

PER_ELEMENT_NODAL_FORCES: tuple[str, ...] = (
    "nodal_resisting_force_x", "nodal_resisting_force_y", "nodal_resisting_force_z",
    "nodal_resisting_force_local_x",
    "nodal_resisting_force_local_y",
    "nodal_resisting_force_local_z",
    "nodal_resisting_moment_x", "nodal_resisting_moment_y", "nodal_resisting_moment_z",
    "nodal_resisting_moment_local_x",
    "nodal_resisting_moment_local_y",
    "nodal_resisting_moment_local_z",
)

LINE_DIAGRAMS: tuple[str, ...] = (
    "axial_force",
    "shear_y", "shear_z",
    "torsion",
    "bending_moment_y", "bending_moment_z",
)
# Note: ``bending_moment_y/z`` (line diagrams, in section local frame)
# is intentionally distinct from ``moment_x/y/z`` (applied nodal moments,
# in global frame) — same physical dimension but different reference
# frame and topology level, so they get different canonical names.

STRESS: tuple[str, ...] = tuple(f"stress_{idx}" for idx in _TENSOR_INDICES)
STRAIN: tuple[str, ...] = tuple(f"strain_{idx}" for idx in _TENSOR_INDICES)

DERIVED_SCALARS: tuple[str, ...] = (
    "von_mises_stress", "pressure_hydrostatic",
    "principal_stress_1", "principal_stress_2", "principal_stress_3",
    "equivalent_plastic_strain",
)

FIBER: tuple[str, ...] = ("fiber_stress", "fiber_strain")

MATERIAL_STATE: tuple[str, ...] = ("damage",)
# ``state_variable_<n>`` is also valid — handled via a regex match in
# ``is_canonical()``, not enumerated here.


ALL_CANONICAL: frozenset[str] = frozenset(
    NODAL_KINEMATICS
    + NODAL_FORCES
    + PER_ELEMENT_NODAL_FORCES
    + LINE_DIAGRAMS
    + STRESS
    + STRAIN
    + DERIVED_SCALARS
    + FIBER
    + MATERIAL_STATE
)


# =====================================================================
# Shorthand expansion table
# =====================================================================
#
# Each shorthand maps to the *full* (3D, ndf=6) expansion. The
# ``expand_shorthand()`` function clips it to the active ``ndm``/``ndf``.

_SHORTHAND_TRANSLATIONAL: dict[str, tuple[str, ...]] = {
    "displacement": ("displacement_x", "displacement_y", "displacement_z"),
    "velocity": ("velocity_x", "velocity_y", "velocity_z"),
    "acceleration": ("acceleration_x", "acceleration_y", "acceleration_z"),
    "displacement_increment": (
        "displacement_increment_x",
        "displacement_increment_y",
        "displacement_increment_z",
    ),
    "force": ("force_x", "force_y", "force_z"),
    # Granular reaction shorthands — separate from the all-in-one
    # "reaction" shorthand that expands to forces+moments.
    "reaction_force": (
        "reaction_force_x", "reaction_force_y", "reaction_force_z",
    ),
}

_SHORTHAND_ROTATIONAL: dict[str, tuple[str, ...]] = {
    "rotation": ("rotation_x", "rotation_y", "rotation_z"),
    "angular_velocity": (
        "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    ),
    "angular_acceleration": (
        "angular_acceleration_x",
        "angular_acceleration_y",
        "angular_acceleration_z",
    ),
    "moment": ("moment_x", "moment_y", "moment_z"),
    "reaction_moment": (
        "reaction_moment_x", "reaction_moment_y", "reaction_moment_z",
    ),
}

_SHORTHAND_TENSOR: dict[str, tuple[str, ...]] = {
    "stress": STRESS,
    "strain": STRAIN,
}

# Reaction is a single OpenSees recorder token covering both forces
# and moments; we expose it as one shorthand.
_SHORTHAND_REACTION: tuple[str, ...] = (
    "reaction_force_x", "reaction_force_y", "reaction_force_z",
    "reaction_moment_x", "reaction_moment_y", "reaction_moment_z",
)

ALL_SHORTHANDS: frozenset[str] = frozenset(
    list(_SHORTHAND_TRANSLATIONAL.keys())
    + list(_SHORTHAND_ROTATIONAL.keys())
    + list(_SHORTHAND_TENSOR.keys())
    + ["reaction"]
)


# =====================================================================
# Public API
# =====================================================================

def is_canonical(name: str) -> bool:
    """True if ``name`` is a known canonical component name."""
    if name in ALL_CANONICAL:
        return True
    # Pattern: state_variable_<integer>
    if name.startswith("state_variable_"):
        suffix = name[len("state_variable_"):]
        return suffix.isdigit()
    return False


def is_shorthand(name: str) -> bool:
    """True if ``name`` is a known shorthand."""
    return name in ALL_SHORTHANDS


def expand_shorthand(
    name: str, *, ndm: int = 3, ndf: int = 6,
) -> tuple[str, ...]:
    """Expand a shorthand or pass through a canonical name.

    Translational shorthands clip to ``ndm`` axes (e.g. ``ndm=2`` →
    ``displacement_x/y`` only). Rotational shorthands require
    rotational DOFs in the active ``ndf`` and return ``()`` if there
    are none. Tensor shorthands (``"stress"``, ``"strain"``) clip to
    3 components in ``ndm=2`` (xx, yy, xy) and 6 in ``ndm=3``.

    Raises ``ValueError`` if ``name`` is neither a known shorthand
    nor a canonical name.
    """
    if is_canonical(name):
        return (name,)

    if name in _SHORTHAND_TRANSLATIONAL:
        full = _SHORTHAND_TRANSLATIONAL[name]
        return _clip_translational(full, ndm)

    if name in _SHORTHAND_ROTATIONAL:
        full = _SHORTHAND_ROTATIONAL[name]
        return _clip_rotational(full, ndm, ndf)

    if name in _SHORTHAND_TENSOR:
        full = _SHORTHAND_TENSOR[name]
        return _clip_tensor(full, ndm)

    if name == "reaction":
        forces = _clip_translational(_SHORTHAND_REACTION[:3], ndm)
        moments = _clip_rotational(_SHORTHAND_REACTION[3:], ndm, ndf)
        return forces + moments

    raise ValueError(
        f"Unknown component '{name}'. Must be a canonical name "
        f"(e.g. 'displacement_x', see ALL_CANONICAL) or a known shorthand "
        f"({sorted(ALL_SHORTHANDS)})."
    )


def expand_many(
    names: Iterable[str], *, ndm: int = 3, ndf: int = 6,
) -> tuple[str, ...]:
    """Expand multiple names, deduplicating while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        for canonical in expand_shorthand(name, ndm=ndm, ndf=ndf):
            if canonical not in seen:
                seen.add(canonical)
                out.append(canonical)
    return tuple(out)


# =====================================================================
# Internal clipping helpers
# =====================================================================

def _clip_translational(
    full: tuple[str, ...], ndm: int,
) -> tuple[str, ...]:
    if ndm <= 0 or ndm > 3:
        raise ValueError(f"ndm must be 1, 2, or 3 (got {ndm}).")
    return full[:ndm]


def _clip_rotational(
    full: tuple[str, ...], ndm: int, ndf: int,
) -> tuple[str, ...]:
    """Return rotational components available in the active DOF space.

    - 1D (ndm=1): no rotations (returns ``()``).
    - 2D (ndm=2): one rotation about the out-of-plane axis (z), only
      if ``ndf >= 3``.
    - 3D (ndm=3): three rotations, only if ``ndf >= 6``.
    """
    if ndm == 1:
        return ()
    if ndm == 2:
        if ndf < 3:
            return ()
        # Convention: in 2D, the rotational DOF is about the z axis.
        return (full[2],)
    if ndm == 3:
        if ndf < 6:
            return ()
        return full
    raise ValueError(f"ndm must be 1, 2, or 3 (got {ndm}).")


def _clip_tensor(full: tuple[str, ...], ndm: int) -> tuple[str, ...]:
    """Tensor (stress/strain) clip: 3 components in 2D, 6 in 3D."""
    if ndm == 1:
        # 1D: just the axial component.
        return (full[0],)
    if ndm == 2:
        # Plane: xx, yy, xy.
        return (full[0], full[1], full[3])
    if ndm == 3:
        return full
    raise ValueError(f"ndm must be 1, 2, or 3 (got {ndm}).")
