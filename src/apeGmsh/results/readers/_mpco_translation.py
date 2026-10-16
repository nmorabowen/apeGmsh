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
from typing import Iterable, Sequence


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

    # Reactions — the static and the inertia-including flavors map to the
    # same canonical names; :data:`_NODAL_PRECEDENCE` picks between them
    # when a file records both. The Rayleigh groups are not reactions and
    # are left out (see :data:`_NODAL_NOT_CANONICAL`).
    "REACTION_FORCE": _Mapping("reaction_force"),
    "REACTION_MOMENT": _Mapping("reaction_moment"),
    "REACTION_FORCE_INCLUDING_INERTIA": _Mapping("reaction_force"),
    "REACTION_MOMENT_INCLUDING_INERTIA": _Mapping("reaction_moment"),

    # Unbalanced — same canonical names as forces; precedence as for
    # reactions.
    "UNBALANCED_FORCE": _Mapping("force"),
    "UNBALANCED_MOMENT": _Mapping("moment"),
    "UNBALANCED_FORCE_INCLUDING_INERTIA": _Mapping("force"),
    "UNBALANCED_MOMENT_INCLUDING_INERTIA": _Mapping("moment"),

    # Pressure — scalar, no axis suffix.
    "PRESSURE": _Mapping("pore_pressure", is_scalar=True),

    # Constraint tie force (Ladruno fork, ADR-30 P4 / ADR 0068 P5) — the
    # ``recorder ladruno -N constraintTieForce`` channel writes
    # ``RESULTS/ON_NODES/CONSTRAINT_TIE_FORCE`` with COMPONENTS
    # ``"TFx,TFy,TFz"`` (the projection constraint force M(a_raw - a_proj)
    # at each tied node). ``component_axis("TFx")`` already resolves the
    # trailing axis, so this single entry unlocks readback via
    # ``results.nodes.get(component="constraint_tie_force_x")``.
    "CONSTRAINT_TIE_FORCE": _Mapping("constraint_tie_force"),
}


# Several ON_NODES groups translate to one canonical prefix. A read takes
# the first group of the prefix's tuple that the file records, never the
# first in HDF5 iteration order: a file written without ``track_order``
# iterates its groups by name. The static reaction leads because
# it is what ``ops.reactions()`` / ``ops.nodeReaction`` report by default.
# Every prefix with more than one group must be listed here, and list
# exactly those groups; :func:`_check_precedence` enforces it at import.
_NODAL_PRECEDENCE: dict[str, tuple[str, ...]] = {
    "reaction_force": (
        "REACTION_FORCE", "REACTION_FORCE_INCLUDING_INERTIA",
    ),
    "reaction_moment": (
        "REACTION_MOMENT", "REACTION_MOMENT_INCLUDING_INERTIA",
    ),
    "force": ("UNBALANCED_FORCE", "UNBALANCED_FORCE_INCLUDING_INERTIA"),
    "moment": ("UNBALANCED_MOMENT", "UNBALANCED_MOMENT_INCLUDING_INERTIA"),
}


# ON_NODES groups the readers recognise but translate to no canonical
# name, with the canonical prefix each is mistaken for. A Rayleigh group
# holds the Rayleigh damping force alone (OpenSees
# ``calculateNodalReactions(2)``), not a reaction. A read of that prefix from a file that records only
# these groups is refused by name. Real MPCO files use ``RAYLEIGH_*``;
# the ``*_RAYLEIGH`` spelling is kept for older skill docs.
_NODAL_NOT_CANONICAL: dict[str, str] = {
    "RAYLEIGH_FORCE": "reaction_force",
    "RAYLEIGH_MOMENT": "reaction_moment",
    "REACTION_FORCE_RAYLEIGH": "reaction_force",
    "REACTION_MOMENT_RAYLEIGH": "reaction_moment",
}


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


class NodeComponentAmbiguous(ValueError):
    """The ON_NODES groups of a stage cannot decide a canonical component.

    Raised when the group that precedence picks holds no column for the
    component while a lower-ranked group does (the read would mix
    sources), when one group holds the component in two columns, and when
    the only groups of the prefix are ones that translate to no canonical
    name (:data:`_NODAL_NOT_CANONICAL`).
    """


def _maps_to(mapping: _Mapping, component: str) -> bool:
    if mapping.is_scalar:
        return component == mapping.canonical_prefix
    return component in {
        f"{mapping.canonical_prefix}_{axis}" for axis in ("x", "y", "z")
    }


def locate_node_component(
    groups: Iterable[tuple[str, Sequence[str]]], component: str,
) -> tuple[str, int] | None:
    """``(result_name, column)`` of ``component`` among ON_NODES groups.

    ``groups`` yields ``(result_name, component_labels)`` for each group
    under ``RESULTS/ON_NODES`` that carries a ``COMPONENTS`` attribute.
    The group comes from :data:`_NODAL_PRECEDENCE`, never from the order
    of ``groups``. Returns ``None`` when no group of the component's
    prefix is recorded; raises :class:`NodeComponentAmbiguous` when the
    groups cannot decide (see the class).
    """
    holders: dict[str, int] = {}
    contenders: list[str] = []
    impostors: list[str] = []
    for name, labels in groups:
        mapping = _NODAL_RESULT_NAME_MAP.get(name)
        if mapping is None:
            prefix = _NODAL_NOT_CANONICAL.get(name)
            if prefix is not None and _maps_to(_Mapping(prefix), component):
                impostors.append(name)
            continue
        if not _maps_to(mapping, component):
            continue
        contenders.append(name)
        cols = [
            col for col, label in enumerate(labels)
            if canonical_node_component(name, label.strip()) == component
        ]
        if len(cols) > 1:
            raise NodeComponentAmbiguous(
                f"ON_NODES/{name} holds {component!r} in columns {cols}; "
                f"its COMPONENTS cannot name one column."
            )
        if cols:
            holders[name] = cols[0]
    if not contenders:
        if impostors:
            raise NodeComponentAmbiguous(
                f"{component!r} is not recorded: ON_NODES holds only "
                f"{sorted(impostors)}, the Rayleigh damping force, which "
                f"is not a reaction. Record reactionForce (or "
                f"reactionForceIncludingInertia) to read {component!r}."
            )
        return None
    order = _NODAL_PRECEDENCE.get(_NODAL_RESULT_NAME_MAP[contenders[0]]
                                  .canonical_prefix, (contenders[0],))
    winner = min(contenders, key=order.index)
    if winner in holders:
        return winner, holders[winner]
    if holders:
        raise NodeComponentAmbiguous(
            f"ON_NODES/{winner} takes precedence for {component!r} but holds "
            f"no column for it, while {sorted(holders)} hold it; refusing "
            f"to read {component!r} from a different group than its sibling "
            f"components. Precedence: {list(order)}."
        )
    return None


# Modal result groups (``-N modesOfVibration`` / ``modesOfVibrationRotational``)
# → the canonical prefix of the *mode shape*. They are deliberately NOT in
# :data:`_NODAL_RESULT_NAME_MAP`: a mode shape is not a displacement history,
# and a host stage that also records ``DISPLACEMENT`` must never resolve
# ``displacement_x`` to an eigenvector column. The readers expose each mode
# as its own ``kind="mode"`` stage instead, the layout a native results
# file holds (``ParallelModalResult.to_native``).
_MODAL_RESULT_NAME_MAP: dict[str, str] = {
    "MODES_OF_VIBRATION(U)": "displacement",
    "MODES_OF_VIBRATION(R)": "rotation",
}


def is_modal_result_name(mpco_result_name: str) -> bool:
    """True for the ``MODES_OF_VIBRATION(U|R)`` groups under ``ON_NODES``."""
    return mpco_result_name in _MODAL_RESULT_NAME_MAP


def canonical_modal_component(
    mpco_result_name: str, component_label: str,
) -> str | None:
    """Canonical mode-shape component of one modal-group column.

    ``("MODES_OF_VIBRATION(U)", "Ux")`` → ``"displacement_x"``;
    ``("MODES_OF_VIBRATION(R)", "Rz")`` → ``"rotation_z"``. ``None`` for a
    non-modal group or a label with no axis.
    """
    prefix = _MODAL_RESULT_NAME_MAP.get(mpco_result_name)
    if prefix is None:
        return None
    axis = component_axis(component_label)
    if axis is None:
        return None
    return f"{prefix}_{axis}"


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


# How the copies of a node on a partition interface combine when the
# per-rank files of an OpenSeesMP run are stitched (Ladruno schema §7.1,
# attribute ``PARTITION_REDUCTION`` on each result group; WP-126).
PARTITION_REDUCTION_NONE = "NONE"                # every copy is the same
PARTITION_REDUCTION_SUM = "SUM"                  # each copy is a partial
PARTITION_REDUCTION_UNSUPPORTED = "UNSUPPORTED"  # partial, no sum recovers it
PARTITION_REDUCTIONS = (
    PARTITION_REDUCTION_NONE,
    PARTITION_REDUCTION_SUM,
    PARTITION_REDUCTION_UNSUPPORTED,
)


def partition_reduction_from_name(result_name: str) -> str:
    """Reduction kind for a result group that carries no attribute.

    The fallback for ``.ladruno`` files written before WP-126 and for
    every ``.mpco``. Each rank computes nodal reactions and unbalanced
    loads from its own elements only, so those are summed; everything
    else (kinematics, pressure, tie force) is the same in every copy.
    ``RAYLEIGH_*`` is the damping part of the reaction, a partial too.
    """
    name = result_name.upper()
    if name.startswith(("REACTION", "UNBALANCED", "RAYLEIGH")):
        return PARTITION_REDUCTION_SUM
    return PARTITION_REDUCTION_NONE


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
