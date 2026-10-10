"""Declarations of an :class:`~apeGmsh.assembly.Assembly`: instances and ties.

Both are frozen records validated when they are declared, so a refused
call leaves the assembly unchanged (ADR 0117 INV-1, INV-7).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, cast

from ._errors import AssemblyError

__all__ = [
    "Coupling", "Instance", "RefNode", "Tie", "check_dense_ranks",
    "check_label", "check_partition_rank", "check_point",
    "check_unranked_source",
    "check_rotate", "check_translate", "merged_port", "split_port",
]


@dataclass(frozen=True)
class Instance:
    """One placed copy of a saved ``model.h5`` (ADR 0117 D1).

    ``rotate`` is ``((ax, ay, az), theta)`` with ``theta`` in radians,
    applied about the origin before ``translate``.
    """

    label: str
    source: Path
    translate: tuple[float, float, float]
    rotate: "tuple[tuple[float, float, float], float] | None"
    #: The OpenSeesMP rank (``getPID``) that owns the instance (ADR 0038
    #: Layer 2), or ``None``. Every instance of an assembly carries one or
    #: none does: an unranked assembly is serial.
    partition_rank: "int | None" = None

    def compose_rotate(self) -> "tuple[float, float, float, float] | None":
        """``rotate`` in the merge engine's axis-angle form ``(x, y, z, theta)``."""
        if self.rotate is None:
            return None
        (ax, ay, az), theta = self.rotate
        return (ax, ay, az, theta)


@dataclass(frozen=True)
class Tie:
    """One assembly-level ``tie`` between two instance ports (ADR 0117 D3)."""

    master: str
    slave: str
    enforce: str
    method: str
    dofs: "tuple[int, ...] | None"
    tolerance: float
    name: "str | None"
    #: The ``TieDef`` built (and so validated) when ``tie()`` was called.
    definition: Any = field(compare=False, repr=False)
    #: The penalty knobs ``g.constraints.tie`` takes, passed to the
    #: ``TieDef`` unchanged (AS5-b, the v1 ``couple(kind="tie")`` options).
    stiffness: "float | str" = "auto"
    stiffness_p: "float | None" = None
    rotational: bool = False
    pressure: bool = False
    #: A ``CouplingControl`` (``enforce="penalty_al"`` only), or ``None``.
    control: Any = None
    outward: "tuple[float, float, float] | None" = None

    @classmethod
    def of(cls, master: str, slave: str, definition: Any) -> "Tie":
        """The record of a ``TieDef`` that ``tie_definition`` built."""
        d = definition
        return cls(
            master=master, slave=slave, enforce=d.enforce, method=d.method,
            dofs=tuple(d.dofs) if d.dofs is not None else None,
            tolerance=d.tolerance, name=d.name, definition=d,
            stiffness=d.stiffness, stiffness_p=d.stiffness_p,
            rotational=d.rotational, pressure=d.pressure, control=d.control,
            outward=d.outward,
        )


@dataclass(frozen=True)
class RefNode:
    """An assembly-owned reference node (ADR 0117 D3).

    It is not part of any instance: ``bridge()`` adds it to the merged FEM
    as an element-less decoupled node labelled ``name`` (which has no
    ``.``), at FEM id ``k`` for the ``k``-th declared node, below every
    instance window.
    """

    name: str
    coords: tuple[float, float, float]


@dataclass(frozen=True)
class Coupling:
    """One assembly-level coupling between two ports (ADR 0117 D3).

    ``kind`` is a ``/assembly/ties`` kind other than ``tie`` and ``node``.
    For ``kinematic_coupling`` / ``distributing_coupling`` the master is
    the reference node and the slave the target port; for ``embedded``
    the master is the host. ``params`` is the canonical JSON of the
    options, exactly as the ``/assembly/ties`` row stores it.
    """

    kind: str
    master: str
    slave: str
    params: str
    name: "str | None"
    #: The constraint def built (and so validated) when the verb was called.
    definition: Any = field(compare=False, repr=False)


def check_label(label: object, *, what: str) -> str:
    """Return ``label`` if it is a valid instance or assembly-object name.

    The rule is the merge engine's (``Compose._validate_label``), checked
    here so ``instance()`` refuses before it records anything: a
    non-empty string with no ``.``, ``/`` or whitespace, and no leading
    or trailing ``_``.
    """
    if not isinstance(label, str) or not label:
        raise AssemblyError(f"{what} must be a non-empty string, got {label!r}.")
    if "." in label or "/" in label:
        raise AssemblyError(
            f"{what} {label!r} contains '.' or '/'; an instance-owned name "
            f"is '{{instance}}.{{name}}', so neither may appear in a label."
        )
    if any(ch.isspace() for ch in label):
        raise AssemblyError(f"{what} {label!r} contains whitespace.")
    if label.startswith("_") or label.endswith("_"):
        raise AssemblyError(f"{what} {label!r} cannot start or end with '_'.")
    return label


def check_rotate(
    rotate: object,
) -> "tuple[tuple[float, float, float], float] | None":
    """Normalise ``rotate=((ax, ay, az), theta)``; refuse a zero axis and
    a non-finite (``nan``, ``inf``) component."""
    if rotate is None:
        return None
    try:
        axis, theta = cast("tuple[Sequence[float], float]", rotate)
        ax, ay, az = (float(a) for a in axis)
        th = float(theta)
    except (TypeError, ValueError) as exc:
        raise AssemblyError(
            f"rotate={rotate!r}: expected ((ax, ay, az), theta_radians)."
        ) from exc
    if not all(math.isfinite(v) for v in (ax, ay, az, th)):
        raise AssemblyError(
            f"rotate={rotate!r}: the axis and angle must be finite numbers.")
    if math.hypot(ax, ay, az) == 0.0:
        raise AssemblyError(f"rotate={rotate!r}: the rotation axis is zero.")
    return ((ax, ay, az), th)


def check_partition_rank(
    rank: object, label: str, placed: Sequence[Instance],
) -> "int | None":
    """Return ``rank`` for a new instance ``label`` beside ``placed``.

    ``None`` or an ``int >= 0`` (a ``bool`` is refused). Either every
    instance of an assembly carries a rank or none does, and one rank
    holds one instance: the merge engine places each instance on its own
    rank, and an unranked assembly is serial.
    """
    if rank is not None and (
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 0):
        raise AssemblyError(
            f"instance {label!r}: partition_rank must be an int >= 0 or "
            f"None, got {rank!r}.")
    if placed and (placed[0].partition_rank is None) != (rank is None):
        raise AssemblyError(
            f"instance {label!r}: partition_rank={rank!r}, but instance "
            f"{placed[0].label!r} has partition_rank="
            f"{placed[0].partition_rank!r}; give every instance a rank, or "
            f"none for a serial assembly.")
    for other in placed:
        if rank is not None and other.partition_rank == rank:
            raise AssemblyError(
                f"instance {label!r}: rank {rank} already holds instance "
                f"{other.label!r}; one rank holds one instance.")
    return rank


def check_unranked_source(path: Path, label: str) -> None:
    """Refuse a source whose composed modules carry a partition rank.

    The merge engine reads each ``/composed_from/*@partition_rank`` as a
    rank hint, so a ranked assembly archive (or a ranked ``g.compose``
    model) instanced here would add ranks the assembly never declared:
    an empty rank, or a raw rank collision between two instances. An
    assembly's ranks come from ``instance(partition_rank=)`` only.
    """
    import h5py

    try:
        with h5py.File(str(path), "r") as f:
            grp = f.get("composed_from")
            ranked = [] if grp is None else sorted(
                str(sub.attrs["label"]) for sub in grp.values()
                if "partition_rank" in sub.attrs)
    except OSError as exc:
        raise AssemblyError(
            f"instance {label!r}: {str(path)!r} is not a readable model.h5 "
            f"({exc}).") from exc
    if ranked:
        raise AssemblyError(
            f"instance {label!r}: {str(path)!r} composes modules {ranked} "
            f"that carry a partition_rank; an instance source must be "
            f"unranked (rank the assembly with instance(partition_rank=)).")


def check_dense_ranks(placed: Sequence[Instance]) -> None:
    """Refuse a ranked assembly whose ranks are not ``0 .. n-1``: a rank
    with no instance would emit an empty ``getPID`` block."""
    ranks = {i.partition_rank for i in placed}
    if ranks == {None}:
        return
    empty = sorted(set(range(len(placed))) - ranks)
    if empty:
        raise AssemblyError(
            f"partition ranks {sorted(r for r in ranks if r is not None)} "
            f"leave rank(s) {empty} with no instance; ranks run 0 .. "
            f"{len(placed) - 1}, one per instance.")


def check_translate(translate: Sequence[float]) -> tuple[float, float, float]:
    """Normalise ``translate=(x, y, z)``; refuse a non-finite component."""
    return check_point(translate, what="translate")


def check_point(value: object, *, what: str) -> tuple[float, float, float]:
    """Normalise a 3-vector ``(x, y, z)``; refuse a non-finite component."""
    try:
        t = tuple(float(v) for v in cast("Sequence[float]", value))
    except (TypeError, ValueError) as exc:
        raise AssemblyError(f"{what}={value!r}: expected (x, y, z).") from exc
    if len(t) != 3:
        raise AssemblyError(f"{what}={value!r}: expected (x, y, z).")
    if not all(math.isfinite(v) for v in t):
        raise AssemblyError(
            f"{what}={value!r}: every component must be a finite number.")
    return (t[0], t[1], t[2])


def split_port(
    port: object, labels: Sequence[str], nodes: Sequence[str] = (),
) -> tuple[str, str]:
    """Split ``"{instance}.{pg|label}"`` on its first dot.

    A port with no dot names an assembly-owned object: one of ``nodes``
    (the reference nodes this verb accepts), returned as ``("", port)``.
    Any other bare port raises listing the instances and those nodes. The
    instance must already be declared, and the local name must be
    non-empty.
    """
    if not isinstance(port, str) or not port:
        raise AssemblyError(f"port must be a non-empty string, got {port!r}.")
    inst, dot, local = port.partition(".")
    if not dot:
        if port in nodes:
            return "", port
        owned = (f"reference nodes {list(nodes)}" if nodes
                 else "none this verb accepts")
        raise AssemblyError(
            f"port {port!r} names no assembly object (declared: {owned}); "
            f"an instance port is '{{instance}}.{{pg}}', with instances "
            f"{list(labels)}."
        )
    if inst not in labels:
        raise AssemblyError(
            f"port {port!r} names unknown instance {inst!r}; declared "
            f"instances are {list(labels)}."
        )
    if not local:
        raise AssemblyError(
            f"port {port!r} has an empty name after the instance label."
        )
    return inst, local


def merged_port(
    port: object, labels: Sequence[str], nodes: Sequence[str] = (),
) -> str:
    """The name the merged FEM carries for ``port`` (checked as
    :func:`split_port` checks it).

    A reference node keeps its name. An instance port ``"{inst}.{name}"``
    maps through the merge engine's own prefix rule
    (``_prefix_namespaced_name``, ADR 0038 nested composition): a plain
    ``name`` becomes ``"{inst}.{name}"``, and a dotted source name such as
    ``deck.slab`` becomes ``"{inst}/deck.slab"``, the name compose gave it.
    """
    from apeGmsh.mesh._compose import _prefix_namespaced_name

    inst, local = split_port(port, labels, nodes)
    if not inst:
        return local
    merged = _prefix_namespaced_name(inst, local)
    assert merged is not None  # a str in gives a str out
    return merged
