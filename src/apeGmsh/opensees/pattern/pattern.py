"""
Typed ``pattern`` primitives — load patterns and ground-motion patterns.

Per ADR 0005 patterns are **explicit context managers** in apeSees.
The class is BOTH a :class:`Primitive` (registered with the bridge,
gets a pattern tag) AND a context manager (``__enter__`` / ``__exit__``
open the load-recording scope). Calling ``p.load(...)`` / ``p.sp(...)``
inside a ``with`` block records :class:`_LoadRecord` /
:class:`_SPRecord` entries on the pattern instance; ``_emit`` later
plays them back through the emitter as ``pattern_open`` →
``emitter.load`` / ``emitter.sp`` → ``pattern_close``.

Phase 3A scope
==============

Two classes ship in this slice:

* :class:`Plain`  — ``pattern Plain tag tsTag { ... loads + sps ... }``;
  the workhorse.
* :class:`UniformExcitation` — ``pattern UniformExcitation tag dir
  -accel tsTag``; no body, the payload is the pattern itself.

:class:`MultiSupport` is deferred — rare; can be added in a follow-up
without churn.

Load-record fan-out (pg= → element / node tags) is **the bridge
build pipeline's responsibility** (Phase 4), consistent with Phase 2
elements + Phase 1D csys-driven transforms. ``_emit`` for the Plain
pattern raises :class:`NotImplementedError` when it sees a ``pg=``
record so test suites that drive ``_emit`` directly use ``node=``
records.

See :doc:`../architecture/patterns-and-loads` for the OpenSees-driven
rationale (``Domain.h`` exposes one overload of ``addSP_Constraint``
that takes a ``loadPatternTag`` and a second that does not — the
patterns-vs-flat split mirrors that).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .._internal.tag_resolution import resolve_tag
from .._internal.types import Pattern, Primitive, TimeSeries

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Plain",
    "UniformExcitation",
    "_LoadRecord",
    "_SPRecord",
]


# ---------------------------------------------------------------------------
# Internal record types — one per load / sp call inside a Plain pattern
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _LoadRecord:
    """One nodal-load entry inside a :class:`Plain` pattern.

    A value object — not a :class:`Primitive`. Held in a tuple by the
    parent :class:`Plain` instance.

    Parameters
    ----------
    target_kind
        ``"pg"`` or ``"node"``. Picks which of ``target`` is meaningful.
    target
        For ``target_kind="pg"`` the physical-group name; for
        ``target_kind="node"`` the node tag rendered as ``str``
        (a tag is an ``int`` at the OpenSees level — kept as ``str``
        here so the record carries one ``target`` field of one type).
    forces
        The per-DOF nodal-load magnitudes.
    """

    target_kind: str
    target: str
    forces: tuple[float, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class _SPRecord:
    """One non-zero (prescribed) SP_Constraint entry inside a
    :class:`Plain` pattern.

    Same value-object shape as :class:`_LoadRecord`, but for
    pattern-scoped SPs (``sp`` Tcl command). Homogeneous SPs (``fix``)
    are model-level and live on the bridge, not on the pattern.
    """

    target_kind: str
    target: str
    dof: int
    value: float


# ---------------------------------------------------------------------------
# Plain — the workhorse pattern (loads + prescribed SPs in a block)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Plain(Pattern):
    """``pattern Plain tag tsTag { ... loads + sps ... }``.

    The workhorse pattern: aggregates :class:`_LoadRecord` and
    :class:`_SPRecord` instances inside a ``with`` block. The
    accumulators are private list fields excluded from the generated
    constructor / repr / equality (``init=False, compare=False,
    repr=False``); ``load()`` / ``sp()`` mutate the lists in place
    even though the dataclass is frozen at the field level.

    Parameters
    ----------
    series
        The :class:`TimeSeries` instance multiplying the loads /
        prescribed displacements during analysis. The reference is
        a typed primitive; emit-time tag resolution picks up its
        bridge-allocated tag.
    """

    series: TimeSeries

    # ``slots=True`` requires every attribute used at runtime to be
    # declared on the class; private accumulators must appear in the
    # slot list. ``field(init=False, repr=False, compare=False,
    # default_factory=list)`` keeps them out of the user-facing
    # constructor / equality / repr while still adding them to the
    # generated ``__slots__``. The ``_`` suffix signals "internal only".
    _loads_: list[_LoadRecord] = field(
        init=False, repr=False, compare=False, default_factory=list,
    )
    _sps_: list[_SPRecord] = field(
        init=False, repr=False, compare=False, default_factory=list,
    )

    # -- Inspection -----------------------------------------------------

    @property
    def loads(self) -> tuple[_LoadRecord, ...]:
        """Tuple snapshot of the recorded :class:`_LoadRecord` entries."""
        return tuple(self._loads_)

    @property
    def sps(self) -> tuple[_SPRecord, ...]:
        """Tuple snapshot of the recorded :class:`_SPRecord` entries."""
        return tuple(self._sps_)

    # -- Context manager ------------------------------------------------

    def __enter__(self) -> "Plain":
        return self

    def __exit__(self, *exc: object) -> None:
        # No special action — ``_emit`` plays back the recorded entries
        # at build time. The block boundary is purely textual: the
        # ``with`` makes pattern-scoped commands visually clear at the
        # call site (per ADR 0005).
        return None

    # -- Recording API --------------------------------------------------

    def load(
        self,
        *,
        pg: str | None = None,
        node: int | None = None,
        forces: tuple[float, ...],
    ) -> None:
        """Record a nodal load inside this pattern.

        Exactly one of ``pg`` (physical-group name; the bridge fans the
        load across the group's nodes at build time) or ``node`` (an
        explicit OpenSees node tag) must be supplied.
        """
        if (pg is None) == (node is None):
            raise ValueError(
                "Plain.load: supply exactly one of pg= or node= "
                f"(got pg={pg!r}, node={node!r})."
            )
        if pg is not None:
            rec = _LoadRecord(
                target_kind="pg", target=pg, forces=tuple(forces),
            )
        else:
            assert node is not None  # validation above guarantees
            rec = _LoadRecord(
                target_kind="node", target=str(node),
                forces=tuple(forces),
            )
        self._loads_.append(rec)

    def sp(
        self,
        *,
        pg: str | None = None,
        node: int | None = None,
        dof: int,
        value: float,
    ) -> None:
        """Record a non-zero (prescribed) SP_Constraint inside this pattern.

        Same target-resolution shape as :meth:`load`. Homogeneous SPs
        (``value=0``) are model-level — use ``ops.fix(...)`` for those.
        """
        if (pg is None) == (node is None):
            raise ValueError(
                "Plain.sp: supply exactly one of pg= or node= "
                f"(got pg={pg!r}, node={node!r})."
            )
        if pg is not None:
            rec = _SPRecord(
                target_kind="pg", target=pg, dof=dof, value=value,
            )
        else:
            assert node is not None  # validation above guarantees
            rec = _SPRecord(
                target_kind="node", target=str(node),
                dof=dof, value=value,
            )
        self._sps_.append(rec)

    # -- Primitive surface ---------------------------------------------

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.series,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        """Emit ``pattern_open`` + recorded load/sp calls + ``pattern_close``.

        The per-record fan-out (PG → node tags) is the bridge build
        pipeline's responsibility (Phase 4); for Phase 3A we surface
        the still-needed work by raising :class:`NotImplementedError`
        when ``_emit`` sees a ``pg=`` record. Tests construct
        :class:`Plain` with explicit ``node=`` records to exercise emit.
        """
        ts_tag = resolve_tag(emitter, self.series)
        emitter.pattern_open("Plain", tag, ts_tag)
        for rec in self._loads_:
            if rec.target_kind == "node":
                emitter.load(int(rec.target), *rec.forces)
            else:
                raise NotImplementedError(
                    "Plain._emit: pg= load fan-out is the bridge build "
                    "pipeline's job (Phase 4). Tests should use node= "
                    "records to exercise emit."
                )
        for rec in self._sps_:
            if rec.target_kind == "node":
                emitter.sp(int(rec.target), rec.dof, rec.value)
            else:
                raise NotImplementedError(
                    "Plain._emit: pg= sp fan-out is the bridge build "
                    "pipeline's job (Phase 4). Tests should use node= "
                    "records to exercise emit."
                )
        emitter.pattern_close()


# ---------------------------------------------------------------------------
# UniformExcitation — ground-motion pattern, no body
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class UniformExcitation(Pattern):
    """``pattern UniformExcitation tag dir -accel tsTag``.

    A ground-motion pattern: applies a uniform base acceleration in the
    given direction. The payload IS the pattern — there is no body to
    record, so the ``with`` block is permitted but optional (per
    ``patterns-and-loads.md``).

    Parameters
    ----------
    direction
        DOF index for the excitation: ``1, 2, 3`` are translations
        (typically X, Y, Z); ``4, 5, 6`` are rotations. The OpenSees
        manual restricts the value to this range.
    series
        :class:`TimeSeries` providing the acceleration history.
    """

    direction: int
    series: TimeSeries

    def __post_init__(self) -> None:
        if self.direction not in (1, 2, 3, 4, 5, 6):
            raise ValueError(
                "UniformExcitation: direction must be 1-6, got "
                f"{self.direction!r}"
            )

    def __enter__(self) -> "UniformExcitation":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.series,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        ts_tag = resolve_tag(emitter, self.series)
        emitter.pattern_open(
            "UniformExcitation", tag, self.direction, "-accel", ts_tag,
        )
        emitter.pattern_close()
