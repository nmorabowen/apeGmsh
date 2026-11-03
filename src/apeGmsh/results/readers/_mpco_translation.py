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


from ._buckets import (
    _NODAL_NOT_CANONICAL,
    _NODAL_PRECEDENCE,
    _NODAL_RESULT_NAME_MAP,
    canonical_modal_component,
    component_axis,
    is_modal_result_name,
)


def _check_precedence() -> None:
    """Fail at import when a many-to-one prefix has no total precedence."""
    by_prefix: dict[str, set[str]] = {}
    for name, mapping in _NODAL_RESULT_NAME_MAP.items():
        by_prefix.setdefault(mapping.canonical_prefix, set()).add(name)
    for prefix, names in by_prefix.items():
        order = _NODAL_PRECEDENCE.get(prefix)
        if len(names) == 1 and order is None:
            continue
        if order is None or sorted(order) != sorted(names):
            raise RuntimeError(
                f"_NODAL_PRECEDENCE[{prefix!r}] is {order!r}, but the groups "
                f"mapping to {prefix!r} are {sorted(names)}: list each "
                f"exactly once."
            )
    stray = set(_NODAL_PRECEDENCE).difference(by_prefix)
    overlap = set(_NODAL_NOT_CANONICAL).intersection(_NODAL_RESULT_NAME_MAP)
    if stray or overlap:
        raise RuntimeError(
            f"_NODAL_PRECEDENCE names unmapped prefixes {sorted(stray)}, or "
            f"_NODAL_NOT_CANONICAL names mapped groups {sorted(overlap)}."
        )


_check_precedence()


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

    When several MPCO names share the prefix (``"reaction_force_x"``
    comes from ``REACTION_FORCE`` or ``REACTION_FORCE_INCLUDING_INERTIA``)
    this returns the one :data:`_NODAL_PRECEDENCE` ranks first.

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

    names = _NODAL_PRECEDENCE.get(prefix) or tuple(
        name for name, mapping in _NODAL_RESULT_NAME_MAP.items()
        if mapping.canonical_prefix == prefix
    )
    if not names:
        return None
    if _NODAL_RESULT_NAME_MAP[names[0]].is_scalar:
        return (names[0], "")
    return (names[0], axis)
