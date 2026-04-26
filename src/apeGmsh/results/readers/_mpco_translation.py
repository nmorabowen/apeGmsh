"""STKO ``<RESULT_NAME> + component`` ↔ apeGmsh canonical names.

MPCO encodes nodal results as ``RESULTS/ON_NODES/<RESULT_NAME>``
groups, where ``<RESULT_NAME>`` is a fixed STKO label
(``DISPLACEMENT``, ``ROTATION``, ``VELOCITY`` …) and the dataset
``DATA/STEP_<k>`` is a ``(nNodes, nComp)`` array. The group's
``COMPONENTS`` attribute names each column (``"Ux,Uy,Uz"``,
``"Rz"``, etc.).

This module translates STKO labels into apeGmsh canonical names:

- The result name maps to a canonical *prefix*
  (``DISPLACEMENT`` → ``"displacement"``, ``ROTATION`` →
  ``"rotation"``, …).
- The per-column component name's last character (``x``/``y``/``z``,
  Greek or Latin) maps to the canonical axis suffix.
- A scalar result (``PRESSURE`` → ``pore_pressure``) has no suffix.

Example: ``DISPLACEMENT`` group with ``COMPONENTS="Ux,Uy,Uz"``
column 1 (zero-based) → canonical ``displacement_y``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Mapping:
    canonical_prefix: str   # e.g. "displacement"
    is_scalar: bool = False


# MPCO RESULT_NAME → canonical prefix.
# All entries except scalar are vector-valued; the column index maps
# to the axis suffix via ``component_axis``.
_NODAL_RESULT_NAME_MAP: dict[str, _Mapping] = {
    # Kinematics
    "DISPLACEMENT": _Mapping("displacement"),
    "ROTATION": _Mapping("rotation"),
    "VELOCITY": _Mapping("velocity"),
    "ANGULAR_VELOCITY": _Mapping("angular_velocity"),
    "ACCELERATION": _Mapping("acceleration"),
    "ANGULAR_ACCELERATION": _Mapping("angular_acceleration"),

    # Reactions — three flavors all map to the same canonical names
    # because apeGmsh doesn't distinguish them at the vocabulary level.
    # The MPCO result name remains accessible via inspection if needed.
    "REACTION_FORCE": _Mapping("reaction_force"),
    "REACTION_MOMENT": _Mapping("reaction_moment"),
    "REACTION_FORCE_INCLUDING_INERTIA": _Mapping("reaction_force"),
    "REACTION_MOMENT_INCLUDING_INERTIA": _Mapping("reaction_moment"),
    # Real MPCO files use ``RAYLEIGH_FORCE`` / ``RAYLEIGH_MOMENT``
    # — the skill docs that show ``REACTION_FORCE_RAYLEIGH`` are out of
    # date. Keep both spellings for resilience.
    "RAYLEIGH_FORCE": _Mapping("reaction_force"),
    "RAYLEIGH_MOMENT": _Mapping("reaction_moment"),
    "REACTION_FORCE_RAYLEIGH": _Mapping("reaction_force"),
    "REACTION_MOMENT_RAYLEIGH": _Mapping("reaction_moment"),

    # Unbalanced — same canonical names as forces; flavor surfaced via
    # `MPCOReader` if a future caller needs it.
    "UNBALANCED_FORCE": _Mapping("force"),
    "UNBALANCED_MOMENT": _Mapping("moment"),
    "UNBALANCED_FORCE_INCLUDING_INERTIA": _Mapping("force"),
    "UNBALANCED_MOMENT_INCLUDING_INERTIA": _Mapping("moment"),

    # Pressure — scalar, no axis suffix.
    "PRESSURE": _Mapping("pore_pressure", is_scalar=True),
}


def canonical_node_component(
    mpco_result_name: str, component_label: str,
) -> str | None:
    """Translate one ``(RESULT_NAME, component)`` to a canonical name.

    Parameters
    ----------
    mpco_result_name : str
        The HDF5 group name under ``RESULTS/ON_NODES``, e.g. ``"DISPLACEMENT"``.
    component_label : str
        The component label from the group's ``COMPONENTS`` attribute,
        e.g. ``"Ux"`` or ``"Rz"``.

    Returns
    -------
    str or None
        The canonical name, or ``None`` if the combination is not
        recognized (the caller should skip unmapped entries).
    """
    mapping = _NODAL_RESULT_NAME_MAP.get(mpco_result_name)
    if mapping is None:
        return None

    if mapping.is_scalar:
        return mapping.canonical_prefix

    axis = component_axis(component_label)
    if axis is None:
        return None
    return f"{mapping.canonical_prefix}_{axis}"


def component_axis(label: str) -> str | None:
    """Return the canonical axis suffix (``"x"``, ``"y"``, ``"z"``)
    inferred from a component label.

    Recognizes both Latin (``Ux``, ``Vy``, ``Rz``) and Greek
    (``ωx``, ``αz``) prefixes — only the trailing axis character
    matters. Case-insensitive.
    """
    if not label:
        return None
    last = label[-1].lower()
    if last in ("x", "y", "z"):
        return last
    return None


def has_canonical_mapping(mpco_result_name: str) -> bool:
    """True if ``mpco_result_name`` has any canonical translation."""
    return mpco_result_name in _NODAL_RESULT_NAME_MAP


def all_known_mpco_result_names() -> list[str]:
    """Sorted list of MPCO result names this module knows about.

    Useful for testing translation coverage.
    """
    return sorted(_NODAL_RESULT_NAME_MAP.keys())


def canonical_to_mpco_lookup(canonical_name: str) -> tuple[str, str] | None:
    """Reverse lookup: canonical name → ``(mpco_result_name, axis)``.

    Returns the *first* MPCO name whose prefix matches
    (``"reaction_force_x"`` could come from
    ``REACTION_FORCE``, ``REACTION_FORCE_INCLUDING_INERTIA``, or
    ``REACTION_FORCE_RAYLEIGH`` — this returns the first registered).
    Used by the reader to find which MPCO group a requested
    canonical component lives in.

    Returns ``None`` if no mapping exists.
    """
    # Split canonical_name into prefix + axis (or treat as scalar).
    axis: str = ""
    prefix: str = canonical_name
    if len(canonical_name) > 2 and canonical_name[-2] == "_":
        last = canonical_name[-1]
        if last in ("x", "y", "z"):
            axis = last
            prefix = canonical_name[:-2]

    for mpco_name, mapping in _NODAL_RESULT_NAME_MAP.items():
        if mapping.canonical_prefix != prefix:
            continue
        if mapping.is_scalar:
            return (mpco_name, "")
        return (mpco_name, axis)
    return None
