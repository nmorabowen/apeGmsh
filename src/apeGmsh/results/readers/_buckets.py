"""Shared reader code that sits between the ``_mpco*`` and ``.ladruno`` readers.

The slab constructors, the multi-partition stitching helpers, the node
component translation, the mode-of-vibration readers and the skip warnings
used by both the MPCO and the ``.ladruno`` readers live here, so a
non-MPCO reader imports nothing from an ``_mpco*`` module (ADR 0131 D10,
stage 0). The code was moved out of ``_mpco``, ``_mpco_translation``,
``_mpco_multi``, ``_mpco_nodal_io`` and ``_mpco_layer_io`` unchanged.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Sequence

import numpy as np
from numpy import ndarray

from .._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    NodeSlab,
    SpringSlab,
)
from .._time import resolve_time_slice
from ._gauss_refusals import refuse_mixed_gauss_dims
from ._mpco_refusals import MPCOElementInTwoBuckets

if TYPE_CHECKING:
    from collections.abc import Callable

    import h5py

    from typing import TypeAlias

    from ...mesh.FEMData import FEMData
    from ...opensees._readback import RotationRoles

    #: What :meth:`RotationMask.attach` takes for the model's DOF roles:
    #: called once, on the first read that needs them.
    RolesSource: TypeAlias = "Callable[[], RotationRoles]"


#: The kinds a program stage takes (:meth:`MPCOReader.attach_stage_names`).
_PROGRAM_STAGE_KINDS = frozenset({"static", "transient"})


class MPCOBucketSkippedWarning(UserWarning):
    """A bucket the MPCO reader leaves unread, so its elements return no
    values.

    It covers two kinds of bucket:

    * **Refused by its decoder** at read time, at the material-state,
      spring and line-station levels (malformed ``META``, a
      ``NUM_COLUMNS`` mismatch, a missing ``GP_X``, a section code with
      no canonical). One warning per read names the file, every such
      bucket and the decoder's reason.
    * **Left out at discovery**, one warning per bucket naming its path,
      the reason and the file:

      - at the Gauss and material-state levels, a bucket on a nonzero
        custom-rule index (#2115), and a standard-rule bucket whose
        class and rule have no ``RESPONSE_CATALOG`` row (#2160);
      - at the spring level, a ZeroLength-family bucket on a nonzero
        custom-rule index (#2160);
      - at the Gauss, material-state, spring, fiber and line-station
        levels, a bucket whose bracket key does not parse as
        ``<tag>-<Class>[<rule>:<cust>:<hdr>]`` (#2165). A key under two
        alias groups of one token is named once.

    Discovery only records a skipped bucket; the read names it, and only
    when the bucket holds an element the read asked for (``element_ids``,
    or a selection that resolves to them). An unfiltered read and a
    component listing name every one, and a bucket whose ``ID`` cannot
    be read is always named. The same filter holds for a bucket refused
    by its decoder (#2165).

    The layer level names its unread buckets, an unparseable key among
    them, in
    :class:`~apeGmsh.results.readers._mpco_layer_io.LayerBucketUnreadWarning`,
    and the nodal-force level in
    :class:`~apeGmsh.results.readers._mpco_nodal_io.NodalForceBucketSkippedWarning`.
    """


def _empty_node_slab(component: str, time: ndarray, t_idx) -> NodeSlab:
    if t_idx is None:
        t_idx = np.arange(time.size, dtype=np.int64)
    return NodeSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        node_ids=np.array([], dtype=np.int64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_element_slab(component: str, time: ndarray, time_slice) -> ElementSlab:
    t_idx = resolve_time_slice(time_slice, time)
    return ElementSlab(
        component=component,
        values=np.zeros((t_idx.size, 0, 0), dtype=np.float64),
        element_ids=np.array([], dtype=np.int64),
        time=time[t_idx],
    )


def _empty_gauss_slab(component: str, time: ndarray, t_idx) -> GaussSlab:
    return GaussSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        natural_coords=np.zeros((0, 3), dtype=np.float64),
        local_axes_quaternion=None,
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_line_station_slab(
    component: str, time: ndarray, t_idx,
) -> LineStationSlab:
    return LineStationSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        station_natural_coord=np.array([], dtype=np.float64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_fiber_slab(component: str, time: ndarray, t_idx) -> FiberSlab:
    return FiberSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        gp_index=np.array([], dtype=np.int64),
        y=np.array([], dtype=np.float64),
        z=np.array([], dtype=np.float64),
        area=np.array([], dtype=np.float64),
        material_tag=np.array([], dtype=np.int64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_layer_slab(component: str, time: ndarray, t_idx) -> LayerSlab:
    return LayerSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        gp_index=np.array([], dtype=np.int64),
        layer_index=np.array([], dtype=np.int64),
        sub_gp_index=np.array([], dtype=np.int64),
        thickness=np.array([], dtype=np.float64),
        local_axes_quaternion=np.zeros((0, 4), dtype=np.float64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_spring_slab(component: str, time: ndarray, t_idx) -> SpringSlab:
    return SpringSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


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


def _validate_warning_once_per_mode(validate, paths) -> None:
    """Run ``validate()``, which reads every part's stages, and re-emit
    the warnings it raised.

    Each part file sees the same eigen solution, so a non-positive
    eigenvalue would warn once per part. It is re-emitted once per mode
    (the first part's message); every other warning passes through as is.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate()
    seen: set[str] = set()
    for w in caught:
        text = str(w.message)
        if NON_POSITIVE_EIGENVALUE in text:
            key = text
            for p in paths:
                key = key.replace(Path(p).name, "<part>")
            if key in seen:
                continue
            seen.add(key)
        warnings.warn(w.message, stacklevel=3)


def _first_nonempty_time(slabs) -> ndarray:
    for s in slabs:
        if s.time.size:
            return s.time
    return np.array([], dtype=np.float64)


def _node_reduction(readers, paths, stage_id: str, component: str) -> str:
    """The one reduction kind every partition reports for ``component``.

    Partitions that do not record the component (``None``) have no say.
    Partitions that disagree are refused: the stitch would have to guess.
    """
    per_part = [
        (p.name, r.node_partition_reduction(stage_id, component))
        for r, p in zip(readers, paths)
    ]
    kinds = {k for _, k in per_part if k is not None}
    if len(kinds) > 1:
        raise ValueError(
            f"Partitions disagree on PARTITION_REDUCTION for {component!r}: "
            f"{[(n, k) for n, k in per_part if k is not None]}. The part "
            "files are not from one run."
        )
    return kinds.pop() if kinds else PARTITION_REDUCTION_NONE


def _merge_node_slabs(
    slabs: list[NodeSlab], component: str,
    reduction: str = PARTITION_REDUCTION_NONE,
) -> NodeSlab:
    """Union node IDs across partitions, one column per unique ID.

    A node on a partition interface is in several partitions.
    ``reduction`` says how its copies combine (Ladruno schema §7.1):

    - ``"NONE"``: the copies are the same (kinematics); the first
      partition that has the node wins.
    - ``"SUM"``: each rank stored only its own elements' share
      (reactions, unbalanced loads); the copies are added. A node held
      by one partition keeps its value.
    - ``"UNSUPPORTED"``: each copy is a partial that no sum recovers;
      refused.
    """
    if reduction == PARTITION_REDUCTION_UNSUPPORTED:
        raise ValueError(
            f"{component!r} is PARTITION_REDUCTION=UNSUPPORTED: each "
            "partition file holds only its own rank's partial, and no "
            "sum of the partials gives the model value. Read the part "
            "files one at a time, or rerun serially."
        )
    if not slabs:
        return _empty_node(component)
    time = _first_nonempty_time(slabs)
    T = int(time.size)
    pieces = [sl for sl in slabs if sl.node_ids.size]
    if not pieces:
        return NodeSlab(
            component=component,
            values=np.zeros((T, 0), dtype=np.float64),
            node_ids=np.array([], dtype=np.int64),
            time=time,
        )
    all_ids = np.concatenate([sl.node_ids for sl in pieces])
    all_vals = np.concatenate([sl.values for sl in pieces], axis=1)
    # ``return_index`` uses a stable sort: ``first`` is the column of
    # each node's first copy in partition order.
    master_ids, first, inverse = np.unique(
        all_ids, return_index=True, return_inverse=True,
    )
    if reduction == PARTITION_REDUCTION_SUM:
        # Group the copies of each node next to each other, then add
        # each group. Every master ID has at least one copy, so the
        # group starts are strictly increasing.
        order = np.argsort(inverse, kind="stable")
        starts = np.searchsorted(inverse[order], np.arange(master_ids.size))
        values = np.add.reduceat(all_vals[:, order], starts, axis=1)
    else:
        values = all_vals[:, first]
    return NodeSlab(
        component=component, values=np.asarray(values, dtype=np.float64),
        node_ids=master_ids, time=time,
    )


def _refuse_element_in_two_partitions(
    ids_per_part: Sequence[ndarray], sources: Sequence[str], *, what: str,
) -> None:
    """Raise :class:`MPCOElementInTwoBuckets` when an element id comes
    back from two partitions.

    Each element lives on exactly one rank, so the stitch concatenates
    without dedupe; an id in two parts would read twice (#2221). Within
    one part an id repeats legitimately (one row per Gauss point, station
    or fiber), and the part's own reader refuses a repeat inside one
    file. Nodes are not checked here: boundary nodes are shared by
    design, and :func:`_merge_node_slabs` merges them.
    """
    if len(ids_per_part) != len(sources):
        raise ValueError(
            f"{len(ids_per_part)} partition slabs but {len(sources)} "
            f"partition names; the stitch cannot name a part."
        )
    uniq = [np.unique(np.asarray(i, dtype=np.int64)) for i in ids_per_part]
    if sum(1 for u in uniq if u.size) < 2:
        return
    all_ids = np.concatenate(uniq)
    owners = np.repeat(np.arange(len(uniq)), [u.size for u in uniq])
    order = np.argsort(all_ids, kind="stable")
    sorted_ids = all_ids[order]
    repeat = np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])
    if repeat.size == 0:
        return
    j = int(repeat[0])
    eid = int(sorted_ids[j])
    first = sources[int(owners[order[j]])]
    second = sources[int(owners[order[j + 1]])]
    n_ids = np.unique(sorted_ids[repeat]).size
    more = (
        f" {n_ids - 1} more element id(s) the read returns repeat too."
        if n_ids > 1 else ""
    )
    raise MPCOElementInTwoBuckets(
        f"Partitioned {what}: element {eid} comes back from two partition "
        f"files, {first} and {second}. Each element lives on exactly one "
        f"rank, so the part files are corrupt or not from one run; the "
        f"read refuses rather than return element {eid} twice.{more}"
    )


def _concat_element_slabs(
    slabs: list[ElementSlab], component: str, sources: Sequence[str],
) -> ElementSlab:
    """Concatenate element slabs along the element axis (disjoint by
    partition: an element in two parts refuses, see
    :func:`_refuse_element_in_two_partitions`)."""
    time = _first_nonempty_time(slabs)
    _refuse_element_in_two_partitions(
        [s.element_ids for s in slabs], sources,
        what=f"element read of {component!r}",
    )
    nonempty = [s for s in slabs if s.element_ids.size]
    if not nonempty:
        return ElementSlab(
            component=component,
            values=np.zeros((time.size, 0, 0), dtype=np.float64),
            element_ids=np.array([], dtype=np.int64),
            time=time,
        )
    return ElementSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_ids=np.concatenate([s.element_ids for s in nonempty]),
        time=time,
    )


def _concat_line_station_slabs(
    slabs: list[LineStationSlab], component: str, sources: Sequence[str],
) -> LineStationSlab:
    time = _first_nonempty_time(slabs)
    _refuse_element_in_two_partitions(
        [s.element_index for s in slabs], sources,
        what=f"line-station read of {component!r}",
    )
    nonempty = [s for s in slabs if s.element_index.size]
    if not nonempty:
        return LineStationSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            station_natural_coord=np.array([], dtype=np.float64),
            time=time,
        )
    # Stitch the per-row beam quaternion when any partition carries one
    # (Ladruno MODEL/LOCAL_AXES); parts without a frame array contribute
    # NaN rows so the result stays row-aligned. None when no part has one
    # (MPCO/native carry no LOCAL_AXES → all None → None).
    quats = None
    if any(s.local_axes_quaternion is not None for s in nonempty):
        parts = []
        for s in nonempty:
            if s.local_axes_quaternion is not None:
                parts.append(np.asarray(s.local_axes_quaternion))
            else:
                parts.append(np.full((s.element_index.size, 4), np.nan))
        quats = np.concatenate(parts, axis=0)
    return LineStationSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        station_natural_coord=np.concatenate(
            [s.station_natural_coord for s in nonempty],
        ),
        time=time,
        local_axes_quaternion=quats,
    )


def _concat_gauss_slabs(
    slabs: list[GaussSlab],
    component: str,
    sources: Sequence[str],
    *,
    reader: str,
    mixed_error: type[ValueError],
) -> GaussSlab:
    """Concatenate Gauss slabs along the GP axis.

    An element in two parts refuses
    (:func:`_refuse_element_in_two_partitions`). So do parts whose
    natural coordinates differ in width (a 2-D-parent shell part beside a
    3-D-parent solid part): ``mixed_error`` names a part of each width,
    as the single-file read does (#2221). A part whose points have no
    natural coordinates at all (width 0) is padded with NaN to the others'
    width, as before.
    """
    time = _first_nonempty_time(slabs)
    _refuse_element_in_two_partitions(
        [s.element_index for s in slabs], sources,
        what=f"gauss read of {component!r}",
    )
    pairs = [(s, src) for s, src in zip(slabs, sources, strict=True)
             if s.element_index.size]
    nonempty = [s for s, _ in pairs]
    if not nonempty:
        return GaussSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            natural_coords=np.zeros((0, 3), dtype=np.float64),
            local_axes_quaternion=None,
            time=time,
        )
    refuse_mixed_gauss_dims(
        [(f"{src} (element {int(s.element_index[0])})",
          int(s.natural_coords.shape[1])) for s, src in pairs],
        component=component, reader=reader,
        source=f"the {len(sources)} partition files", error=mixed_error,
    )
    max_dim = max(s.natural_coords.shape[1] for s in nonempty)
    coords = [
        s.natural_coords if s.natural_coords.shape[1] == max_dim
        else np.full((s.natural_coords.shape[0], max_dim), np.nan)
        for s in nonempty
    ]
    return GaussSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        natural_coords=np.concatenate(coords, axis=0),
        local_axes_quaternion=None,
        time=time,
    )


def _concat_fiber_slabs(
    slabs: list[FiberSlab], component: str, sources: Sequence[str],
) -> FiberSlab:
    time = _first_nonempty_time(slabs)
    _refuse_element_in_two_partitions(
        [s.element_index for s in slabs], sources,
        what=f"fiber read of {component!r}",
    )
    nonempty = [s for s in slabs if s.element_index.size]
    if not nonempty:
        return FiberSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            gp_index=np.array([], dtype=np.int64),
            y=np.array([], dtype=np.float64),
            z=np.array([], dtype=np.float64),
            area=np.array([], dtype=np.float64),
            material_tag=np.array([], dtype=np.int64),
            time=time,
        )
    # Station ξ: NaN-fill partitions that carry no stations iff at
    # least one does; all-None stays None (matches the slab contract).
    if any(s.station_natural_coord is not None for s in nonempty):
        station = np.concatenate([
            s.station_natural_coord
            if s.station_natural_coord is not None
            else np.full(s.element_index.size, np.nan, dtype=np.float64)
            for s in nonempty
        ])
    else:
        station = None
    return FiberSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        gp_index=np.concatenate([s.gp_index for s in nonempty]),
        y=np.concatenate([s.y for s in nonempty]),
        z=np.concatenate([s.z for s in nonempty]),
        area=np.concatenate([s.area for s in nonempty]),
        material_tag=np.concatenate([s.material_tag for s in nonempty]),
        time=time,
        station_natural_coord=station,
    )


def _concat_layer_slabs(
    slabs: list[LayerSlab], component: str, sources: Sequence[str],
) -> LayerSlab:
    time = _first_nonempty_time(slabs)
    _refuse_element_in_two_partitions(
        [s.element_index for s in slabs], sources,
        what=f"layer read of {component!r}",
    )
    nonempty = [s for s in slabs if s.element_index.size]
    if not nonempty:
        return LayerSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            gp_index=np.array([], dtype=np.int64),
            layer_index=np.array([], dtype=np.int64),
            sub_gp_index=np.array([], dtype=np.int64),
            thickness=np.array([], dtype=np.float64),
            local_axes_quaternion=np.zeros((0, 4), dtype=np.float64),
            time=time,
        )
    return LayerSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        gp_index=np.concatenate([s.gp_index for s in nonempty]),
        layer_index=np.concatenate([s.layer_index for s in nonempty]),
        sub_gp_index=np.concatenate([s.sub_gp_index for s in nonempty]),
        thickness=np.concatenate([s.thickness for s in nonempty]),
        local_axes_quaternion=np.concatenate(
            [s.local_axes_quaternion for s in nonempty], axis=0,
        ),
        time=time,
    )


def _concat_spring_slabs(
    slabs: list[SpringSlab], component: str, sources: Sequence[str],
) -> SpringSlab:
    time = _first_nonempty_time(slabs)
    _refuse_element_in_two_partitions(
        [s.element_index for s in slabs], sources,
        what=f"spring read of {component!r}",
    )
    nonempty = [s for s in slabs if s.element_index.size]
    if not nonempty:
        return SpringSlab(
            component=component,
            values=np.zeros((time.size, 0), dtype=np.float64),
            element_index=np.array([], dtype=np.int64),
            time=time,
        )
    return SpringSlab(
        component=component,
        values=np.concatenate([s.values for s in nonempty], axis=1),
        element_index=np.concatenate([s.element_index for s in nonempty]),
        time=time,
    )


def _merge_partition_fems(
    fems: list["Optional[FEMData]"],
) -> "Optional[FEMData]":
    """Merge per-partition FEMData snapshots into one model-wide snapshot.

    Strategy:

    - **Nodes**: union by ID. First partition that owns a node ID
      keeps its coordinates; later duplicates (boundary copies) are
      dropped.
    - **Elements**: per-class concatenation. Different partitions may
      hold disjoint elements of the same class — merge their groups.
    - **Physical groups / labels**: dropped from the merged FEMData.
      Per-partition MPCO regions describe rank-local node sets that
      don't correspond to anything meaningful at the merged level.
    """
    from ...mesh._element_types import ElementGroup
    from ...mesh._group_set import LabelSet, PhysicalGroupSet
    from ...mesh.FEMData import (
        ElementComposite, FEMData, MeshInfo, NodeComposite,
    )

    real = [f for f in fems if f is not None]
    if not real:
        return None
    if len(real) == 1:
        return real[0]

    # Node merge — union by ID, first-occurrence wins on coords.
    seen_ids: dict[int, int] = {}     # node_id → row in merged array
    merged_ids: list[int] = []
    merged_coords: list[ndarray] = []
    for f in real:
        ids = np.asarray(f.nodes.ids, dtype=np.int64)
        coords = np.asarray(f.nodes.coords, dtype=np.float64)
        for i, nid in enumerate(ids):
            int_nid = int(nid)
            if int_nid in seen_ids:
                continue
            seen_ids[int_nid] = len(merged_ids)
            merged_ids.append(int_nid)
            merged_coords.append(coords[i])
    node_ids_arr = np.array(merged_ids, dtype=np.int64)
    coords_arr = (
        np.stack(merged_coords, axis=0)
        if merged_coords
        else np.zeros((0, 3), dtype=np.float64)
    )

    # Element merge — per-class concatenation. ``code`` is the
    # negated class_tag (synthetic) and matches across partitions for
    # the same OpenSees class. ``ElementComposite`` exposes its
    # underlying type-keyed dict via ``_groups`` (no public accessor).
    merged_groups: dict[int, ElementGroup] = {}
    for f in real:
        for code, grp in f.elements._groups.items():
            if code in merged_groups:
                existing = merged_groups[code]
                merged_ids_e = np.concatenate(
                    [existing.ids, grp.ids],
                ).astype(np.int64)
                merged_conn = np.concatenate(
                    [existing.connectivity, grp.connectivity], axis=0,
                ).astype(np.int64)
                # Element type info — npe and dim are immutable; bump
                # the count on a fresh ElementTypeInfo.
                from ...mesh._element_types import make_type_info
                info = make_type_info(
                    code=existing.element_type.code,
                    gmsh_name=existing.element_type.gmsh_name,
                    dim=existing.element_type.dim,
                    order=existing.element_type.order,
                    npe=existing.element_type.npe,
                    count=int(merged_ids_e.size),
                )
                merged_groups[code] = ElementGroup(
                    element_type=info,
                    ids=merged_ids_e,
                    connectivity=merged_conn,
                )
            else:
                merged_groups[code] = grp

    nodes = NodeComposite(
        node_ids=node_ids_arr,
        node_coords=coords_arr,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups=merged_groups,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    n_elems = sum(int(g.ids.size) for g in merged_groups.values())
    info = MeshInfo(
        n_nodes=int(node_ids_arr.size),
        n_elems=n_elems,
        bandwidth=0,
        types=[g.element_type for g in merged_groups.values()],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


def _empty_node(component: str) -> NodeSlab:
    return NodeSlab(
        component=component,
        values=np.zeros((0, 0), dtype=np.float64),
        node_ids=np.array([], dtype=np.int64),
        time=np.array([], dtype=np.float64),
    )


class NodalForceBucketSkippedWarning(UserWarning):
    """An MPCO nodal-force bucket the reader is unable to decode: the
    elements it holds read no rows."""


@dataclass(frozen=True)
class ModeEntry:
    """One recorded mode: where it lives and its eigen data."""

    step_key: str          # "STEP_<s>" under the modal group's DATA
    mode_key: str          # "MODE_<k>" (k 0-based)
    mode_index: int        # 1-based, as in a native results file
    eigenvalue: float      # LAMBDA = omega^2
    frequency_hz: float
    period_s: float


def _modal_groups(on_nodes: "h5py.Group") -> "list[str]":
    from ._mpco_translation import is_modal_result_name
    return [n for n in on_nodes if is_modal_result_name(n)]


def _keyed(group: "h5py.Group", prefix: str) -> "list[str]":
    keys = [k for k in group if k.startswith(prefix)]
    return sorted(keys, key=lambda k: int(k[len(prefix):]))


def _components_of(group: "h5py.Group") -> "list[str]":
    if "COMPONENTS" not in group.attrs:
        return []
    raw = _attr_scalar(group.attrs["COMPONENTS"])
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


#: Marker in the warning :func:`eigen_scalars` raises; the multi-partition
#: readers key on it to warn once per mode for the whole part set.
NON_POSITIVE_EIGENVALUE = "non-positive eigenvalue"


def eigen_scalars(
    lam: float, mode_index: int, source: str,
) -> "tuple[float, float]":
    """``(frequency_hz, period_s)`` from an eigenvalue.

    The ``ParallelModalResult.to_native`` rule: a non-positive eigenvalue (a rigid-body mechanism or an
    unconverged eigensolve) warns and reads as 0 Hz / 0 s rather than
    passing as a valid mode.
    """
    import math
    import warnings

    if lam > 0:
        omega = math.sqrt(lam)
        return omega / (2.0 * math.pi), 2.0 * math.pi / omega
    warnings.warn(
        f"{source}: mode {mode_index} has a {NON_POSITIVE_EIGENVALUE} "
        f"({lam:.6g}); this indicates a spurious/unstable mode (rigid-body "
        "mechanism or unconverged eigensolve). Reading frequency=0 / "
        "period=0 for it.",
        UserWarning,
        stacklevel=3,
    )
    return 0.0, 0.0


def _mode_sets_of(
    data: "h5py.Group", group_name: str, source: str,
) -> "list[tuple[str, list[tuple[str, float]]]]":
    sets: "list[tuple[str, list[tuple[str, float]]]]" = []
    for step_key in _keyed(data, "STEP_"):
        modes: "list[tuple[str, float]]" = []
        for mode_key in _keyed(data[step_key], "MODE_"):
            attrs = data[step_key][mode_key].attrs
            if "LAMBDA" not in attrs:
                raise ValueError(
                    f"{source}: ON_NODES/{group_name}/DATA/{step_key}/"
                    f"{mode_key} has no LAMBDA attribute; the eigenvalue "
                    "of this mode is not recorded."
                )
            modes.append((mode_key, float(_attr_scalar(attrs["LAMBDA"]))))
        if modes:
            sets.append((step_key, modes))
    return sets


def discover_modes(
    on_nodes: "h5py.Group", *, source: str,
) -> "list[tuple[str, list[ModeEntry]]]":
    """Every modal set under ``on_nodes``, as ``[(step_key, [ModeEntry])]``.

    Sets are ordered by step, modes by index. The translational group is
    the reference when present; the rotational group, when both exist,
    must list the same steps, modes and eigenvalues, or the file is
    refused. A ``MODE_<k>`` without a ``LAMBDA`` attribute is refused too:
    a mode stage never invents its eigenvalue.
    """
    names = sorted(_modal_groups(on_nodes), key=lambda n: (not n.endswith("(U)"), n))
    if not names:
        return []
    per_group = []
    for name in names:
        if "DATA" not in on_nodes[name]:
            raise ValueError(
                f"{source}: ON_NODES/{name} has no DATA group; the modal "
                "record is incomplete."
            )
        per_group.append(_mode_sets_of(on_nodes[name]["DATA"], name, source))
    for name, other in zip(names[1:], per_group[1:]):
        if other != per_group[0]:
            raise ValueError(
                f"{source}: ON_NODES/{name} lists different modes or "
                f"eigenvalues than ON_NODES/{names[0]}; the modal record is "
                "inconsistent."
            )
    out: "list[tuple[str, list[ModeEntry]]]" = []
    for step_key, modes in per_group[0]:
        entries = []
        for mode_key, lam in modes:
            index = int(mode_key[len("MODE_"):]) + 1
            f_hz, t_s = eigen_scalars(lam, index, source)
            entries.append(ModeEntry(
                step_key=step_key, mode_key=mode_key, mode_index=index,
                eigenvalue=lam, frequency_hz=f_hz, period_s=t_s,
            ))
        out.append((step_key, entries))
    return out


def modal_components(on_nodes: "h5py.Group") -> "list[str]":
    """Canonical mode-shape components the modal groups carry."""
    from ._mpco_translation import canonical_modal_component

    out: "set[str]" = set()
    for name in _modal_groups(on_nodes):
        for label in _components_of(on_nodes[name]):
            canon = canonical_modal_component(name, label)
            if canon is not None:
                out.add(canon)
    return sorted(out)


def read_mode_shape(
    on_nodes: "h5py.Group",
    entry: ModeEntry,
    component: str,
    *,
    node_ids: "ndarray | None" = None,
) -> "tuple[ndarray, ndarray] | None":
    """``(values[nSel], node_ids[nSel])`` of one mode's ``component``.

    ``None`` when no modal group carries ``component`` for this mode.
    """
    from ._mpco_translation import canonical_modal_component

    for name in _modal_groups(on_nodes):
        grp = on_nodes[name]
        cols = [
            i for i, lab in enumerate(_components_of(grp))
            if canonical_modal_component(name, lab) == component
        ]
        if not cols:
            continue
        data = grp["DATA"]
        if entry.step_key not in data or entry.mode_key not in data[entry.step_key]:
            return None
        ids = np.asarray(grp["ID"][...], dtype=np.int64).flatten()
        rows = np.arange(ids.size)
        if node_ids is not None:
            rows = np.flatnonzero(
                np.isin(ids, np.asarray(node_ids, dtype=np.int64)),
            )
        block = data[entry.step_key][entry.mode_key]
        col = np.asarray(block[:, cols[0]], dtype=np.float64)
        return col[rows], ids[rows]
    return None


def mode_stage_layout(
    host_ids: "list[str]",
    host_names: "list[str]",
    host_sets: "list[list[tuple[str, list[ModeEntry]]]]",
) -> "list[tuple[int, str, str, ModeEntry]]":
    """``(host_ordinal, stage_id, name, entry)`` for every recorded mode.

    Ids are ``<host_id>_mode_<k>``, or ``<host_id>_step_<s>_mode_<k>``
    when the host holds more than one modal set. Names are ``mode_<k>``
    (the names a native results file uses) when the whole file holds one modal
    set, else ``<host name>/STEP_<s>/mode_<k>`` so every name is unique.
    """
    n_sets = sum(len(sets) for sets in host_sets)
    out: "list[tuple[int, str, str, ModeEntry]]" = []
    for h, (hid, hname, sets) in enumerate(zip(host_ids, host_names, host_sets)):
        for step_key, entries in sets:
            for e in entries:
                if len(sets) == 1:
                    sid = f"{hid}_mode_{e.mode_index}"
                else:
                    sid = f"{hid}_{step_key.lower()}_mode_{e.mode_index}"
                name = (
                    f"mode_{e.mode_index}" if n_sets == 1
                    else f"{hname}/{step_key}/mode_{e.mode_index}"
                )
                out.append((h, sid, name, e))
    return out


class AbsentDofWarning(UserWarning):
    """A rotation read without per-node ndf may hold 0.0 at nodes that
    have no rotational DOF (#1971)."""


def load_nodes_ndf(model_h5: "str | Path") -> "dict[int, int] | None":
    """``{node: ndf}`` from ``model_h5``'s ``/opensees/nodes_ndf``.

    ``None`` when the archive predates that group.
    """
    from ...opensees.emitter import h5_reader

    with h5_reader.open(str(model_h5)) as m:
        return m.nodes_ndf()


def _continuum_classes() -> "frozenset[str]":
    """C++ element classes whose nodes may lack rotational DOFs.

    Two derived sources, so a class joins through either table: the
    RESPONSE_CATALOG classes whose ``stress`` is a continuum tensor, and
    the element registry's ``nd``-material entries (by ``cpp_class_name``,
    else the key). The registry adds the u-p family, whose nodes carry a
    pressure DOF in a slot the recorder writes as a rotation.
    """
    from ...opensees._element_capabilities import _ELEM_REGISTRY
    from ...opensees._response_catalog import RESPONSE_CATALOG

    catalog = {
        cls for (cls, _rule, token), layout in RESPONSE_CATALOG.items()
        if token == "stress" and layout.component_layout[0] == "stress_xx"
    }
    registry = {
        spec.cpp_class_name or key for key, spec in _ELEM_REGISTRY.items()
        if spec.mat_family == "nd"
    }
    return frozenset(catalog | registry)


def _has_continuum_class(model_elements: "h5py.Group | None") -> bool:
    """True when ``MODEL/ELEMENTS`` holds a ``<tag>-<Class>[..]`` entry of
    a continuum class."""
    if model_elements is None:
        return False
    solids = _continuum_classes()
    return any(
        key.partition("-")[2].partition("[")[0] in solids
        for key in model_elements
    )


def model_node_ids(stage_groups: "list[h5py.Group]") -> "set[int]":
    """Every node id under the stages' ``MODEL/NODES/ID``."""
    out: "set[int]" = set()
    for grp in stage_groups:
        if "MODEL" in grp and "NODES" in grp["MODEL"] and "ID" in grp["MODEL/NODES"]:
            out.update(int(n) for n in np.asarray(grp["MODEL/NODES/ID"][...]).flat)
    return out


_NO_MODEL = (
    "no per-node ndf is attached. Pass model_h5= (the run's archive, with "
    "/opensees/nodes_ndf) to read NaN at those nodes"
)


class RotationMask:
    """Per-reader NaN mask for rotation columns a node lacks.

    :meth:`attach` takes the model's per-node ndf and its DOF roles.
    :meth:`absent` then names the nodes whose DOF layout has no slot for
    a component, or whose elements say that slot holds no rotation (a
    2-D ``LadrunoUP`` node lays its pore pressure where a 2-D frame node
    lays ``rotation_z``, #2029); :meth:`apply` writes NaN there. Without
    an attached ndf, nothing is masked, and the first rotation read over
    a continuum class warns.
    """

    #: The warn-once cell a partitioned reader shares across its parts
    #: (:meth:`one_warning`); unset, each mask warns on its own.
    _shared_warning: "ContextVar[list[bool] | None]" = ContextVar(
        "rotation_mask_shared_warning", default=None)

    def __init__(self, source: str) -> None:
        self._source = source
        self._nodes_ndf: "dict[int, int] | None" = None
        self._roles: "RolesSource | None" = None
        self._ndm = 3
        self._warned = False
        self._why_unattached = _NO_MODEL

    @staticmethod
    @contextmanager
    def one_warning(cell: "list[bool]") -> "Iterator[None]":
        """Within the block, every mask shares ``cell``: the first one to
        warn sets ``cell[0]`` and the others stay silent, so a partitioned
        read warns once, not once per part (#2028)."""
        token = RotationMask._shared_warning.set(cell)
        try:
            yield
        finally:
            RotationMask._shared_warning.reset(token)

    def attach(
        self, nodes_ndf: "dict[int, int] | None", ndm: int,
        file_nodes: "set[int]", *, roles: "RolesSource",
    ) -> None:
        """Use ``nodes_ndf`` if it describes every node in ``file_nodes``.

        ``roles`` gives the model's :class:`~apeGmsh.opensees._readback.RotationRoles`
        over those nodes; it is called (once) by the first read whose
        layout keeps a rotational slot, since only then can a slot be a
        rotation or a pore pressure.

        ``None`` is an archive older than ``/opensees/nodes_ndf``. A model
        that misses some of the file's nodes is another model (the bind
        warns ``ModelFemMismatchWarning`` for it). Neither is used; the
        warning then names which it was.
        """
        if nodes_ndf is None:
            self._why_unattached = (
                "model_h5 is an archive older than /opensees/nodes_ndf, so "
                "it holds no per-node ndf. Re-save the model with this "
                "apeGmsh to read NaN at those nodes"
            )
            return
        missing = sorted(file_nodes - {int(n) for n in nodes_ndf})
        if missing:
            self._why_unattached = (
                f"model_h5's /opensees/nodes_ndf misses nodes "
                f"{missing[:10]} of this file, so it is another model's "
                "and its ndf is not used. Pass the archive of the run "
                "that wrote this file"
            )
            return
        self._nodes_ndf = {int(n): int(d) for n, d in nodes_ndf.items()}
        self._ndm = int(ndm)
        self._roles = roles

    def absent(
        self, component: str, node_ids: "ndarray",
        model_elements: "h5py.Group | None",
    ) -> "ndarray | None":
        """``(N,)`` bool: the nodes without ``component``'s DOF, or
        ``None`` when nothing is masked (not a rotation, or no ndf)."""
        from ...opensees._recorder_translate import node_component_spatial

        spatial = node_component_spatial(component)
        if spatial is None or spatial[1] < 3:
            return None                 # not a rotational component
        slot = spatial[1]
        if self._nodes_ndf is None or self._roles is None:
            self._warn_unattached(component, model_elements)
            return None
        from ...opensees._internal.build.recorders import _node_layout

        ids = np.asarray(node_ids, dtype=np.int64)
        missing = sorted({int(n) for n in ids} - self._nodes_ndf.keys())
        if missing:
            raise ValueError(
                f"{self._source}: {component!r} has nodes "
                f"{missing[:10]} that are neither in the file's MODEL/NODES "
                "nor in the model's /opensees/nodes_ndf, so whether their "
                "column is a value or padding is unknown."
            )
        absent = np.array([
            slot not in _node_layout(self._nodes_ndf[int(n)], self._ndm)
            for n in ids
        ], dtype=bool)
        if not absent.all():
            # The layout keeps the slot at these nodes; their elements say
            # whether it holds a rotation (a frame) or not (a u-p pressure).
            roles = self._roles()
            absent |= np.array(
                [roles.refusal(int(n)) is not None for n in ids], dtype=bool)
        return absent

    def _warn_unattached(
        self, component: str, model_elements: "h5py.Group | None",
    ) -> None:
        """Warn once (per mask, or per :meth:`one_warning` block) that a
        rotation read over continuum elements is unmasked."""
        if self._warned or not _has_continuum_class(model_elements):
            return
        self._warned = True
        shared = RotationMask._shared_warning.get()
        if shared is not None:
            if shared[0]:
                return
            shared[0] = True
        warnings.warn(
            f"{self._source}: {component!r} is read over continuum "
            "elements, whose nodes may have no rotational DOF; the "
            "recorder writes 0.0 there (or a pressure, at a u-p "
            "node), so a value may be an absent DOF. "
            f"{self._why_unattached}. The fork recording per-node "
            "ndf is the permanent fix (#2015).",
            AbsentDofWarning, stacklevel=6,
        )

    def apply(
        self, slab: "NodeSlab", model_elements: "h5py.Group | None",
    ) -> "NodeSlab":
        """``slab`` with NaN at the nodes :meth:`absent` names."""

        absent = self.absent(slab.component, slab.node_ids, model_elements)
        if absent is None or not absent.any():
            return slab
        values = np.where(absent[None, :], np.nan, slab.values)
        return replace(slab, values=values)


def attach_nodes_ndf(
    reader: object, accepts: "tuple[type, ...]", model_h5: "str | Path",
    ndm: int,
) -> None:
    """Attach ``model_h5``'s per-node ndf and DOF roles to ``reader``
    (#1971, #2029).

    ``accepts`` are the reader classes that take them: the one-file
    reader, which carries a :class:`RotationMask`, and the partitioned
    façade, which forwards them to every part (#2028). Any other reader
    raises :class:`TypeError`, so a reader never goes unmasked unnoticed.
    The roles are read from ``model_h5`` once, on the first rotation read
    whose layout keeps a rotational slot, and shared by every part.
    """
    if not isinstance(reader, accepts):
        raise TypeError(
            f"attach_nodes_ndf: {type(reader).__name__} is none of "
            f"{[c.__name__ for c in accepts]}, which carry the rotation "
            "mask; it has no per-node ndf to attach."
        )
    from functools import cache, partial
    from pathlib import Path

    from ...opensees._recorder_translate import model_h5_rotation_roles

    nodes_ndf = load_nodes_ndf(model_h5)
    roles = cache(partial(
        model_h5_rotation_roles, model_h5, tuple(nodes_ndf or ()),
        where=f"{Path(model_h5).name}: the rotation mask",
    ))
    reader.attach_nodes_ndf(          # type: ignore[attr-defined]
        nodes_ndf, ndm, roles=roles,
    )


def _attr_scalar(value):
    """Coerce an h5py attribute value to a scalar."""
    if hasattr(value, "size") and getattr(value, "size", 0) == 1:
        return value.item() if hasattr(value, "item") else value[0]
    if hasattr(value, "__len__") and len(value) == 1:
        return value[0]
    return value


class LayerBucketUnreadWarning(UserWarning):
    """A layered-shell bucket the reader leaves unread, so neither the fiber
    nor the layer level returns its values. The warning names the bucket,
    its class and why: on ``.ladruno`` a fiber-token bucket whose blocks of
    several components per fiber it does not decode (#1995); on MPCO a
    layer-group bucket that bucket discovery leaves out (#2090)."""
