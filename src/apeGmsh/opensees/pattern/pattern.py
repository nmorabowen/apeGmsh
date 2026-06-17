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
elements + Phase 1D orientation-driven transforms. ``_emit`` for the Plain
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
    from ..node import Node


def _node_to_tag(node: "int | Node") -> int:
    """Coerce a :class:`Node` or plain int to its tag.

    Duck-typing via ``int(node)`` (Node defines ``__int__``) would work
    too, but explicit isinstance keeps the code searchable when we
    later add other node-bearing primitives.
    """
    # Defer the Node import to call-time so module load order doesn't
    # matter (pattern.py vs node.py — both eventually load when
    # apesees.py is imported, but isinstance against an unimported
    # class would always be False).
    from ..node import Node as _N
    if isinstance(node, _N):
        return node.tag
    return int(node)


__all__ = [
    "Plain",
    "UniformExcitation",
    "_LoadRecord",
    "_SPRecord",
    "_MomentTensorRecord",
]

# Frame / method vocabularies for the moment-tensor source (ADR 0062).
_MT_FRAMES = ("z-up", "z-down")
_MT_METHODS = ("consistent", "dipole")


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


@dataclass(frozen=True, kw_only=True, slots=True)
class _MomentTensorRecord:
    """One moment-tensor seismic source inside a :class:`Plain` pattern
    (ADR 0062).

    A value object resolved by the bridge build pipeline (Phase 4): it
    needs the FEM snapshot to locate the host element and evaluate
    ``∂N/∂x`` at ``position``, so it cannot expand at the primitive level
    (like ``pg=`` loads and ``from_model`` cases). The forces it produces
    are constant nodal-load vectors ``M0·m_ij·∂N_a/∂x_j``; the pattern's
    shared :class:`TimeSeries` supplies the **normalized moment function**
    ``S(t)``.

    Supply exactly one mechanism source: ``(strike, dip, rake)`` *or*
    ``m_ij`` (a 3×3 unit tensor stored as nested tuples for frozen-ness).
    """

    position: tuple[float, ...]
    frame: str
    method: str
    t0: float
    M0: float
    strike: float | None
    dip: float | None
    rake: float | None
    m_ij: tuple[tuple[float, ...], ...] | None
    region: str | None


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
    _from_model_cases_: list[str] = field(
        init=False, repr=False, compare=False, default_factory=list,
    )
    _moment_tensors_: list[_MomentTensorRecord] = field(
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

    @property
    def from_model_cases(self) -> tuple[str, ...]:
        """Tuple snapshot of the imported geometry load-case names.

        Populated by :meth:`from_model`.  The bridge expands each case
        into per-node ``load`` / ``sp`` lines at emit time (it owns the
        FEM snapshot and the model ``ndf``).
        """
        return tuple(self._from_model_cases_)

    @property
    def moment_tensors(self) -> tuple[_MomentTensorRecord, ...]:
        """Tuple snapshot of the recorded moment-tensor sources (ADR 0062).

        Populated by :meth:`moment_tensor`.  The bridge resolves each into
        nodal ``load`` lines at emit time (it owns the FEM snapshot needed
        to locate the host and evaluate ``∂N/∂x``).
        """
        return tuple(self._moment_tensors_)

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
        node: "int | Node | None" = None,
        forces: tuple[float, ...],
    ) -> None:
        """Record a nodal load inside this pattern.

        Exactly one of ``pg`` (physical-group name; the bridge fans the
        load across the group's nodes at build time) or ``node`` (an
        explicit OpenSees node tag or :class:`Node` instance returned
        by ``ops.nodes.get(...)``) must be supplied.
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
            node_tag = _node_to_tag(node)
            rec = _LoadRecord(
                target_kind="node", target=str(node_tag),
                forces=tuple(forces),
            )
        self._loads_.append(rec)

    def sp(
        self,
        *,
        pg: str | None = None,
        node: "int | Node | None" = None,
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
            node_tag = _node_to_tag(node)
            rec = _SPRecord(
                target_kind="node", target=str(node_tag),
                dof=dof, value=value,
            )
        self._sps_.append(rec)

    def from_model(self, case: str) -> None:
        """Import the geometry-declared loads tagged with load *case*.

        Pulls the resolved nodal records that ``g.loads.case(case)`` /
        ``g.displacements.case(case)`` produced — concentrated/distributed
        forces become ``load`` lines and prescribed (non-homogeneous)
        displacements become ``sp`` lines, all inside this pattern (scaled
        by its time series).  Homogeneous fixes are *not* imported (they
        are model-level — use ``ops.fix(...)``).

        The expansion happens on the bridge at emit time (it owns the FEM
        snapshot and the model ``ndf``); this call only records the case
        name.  Mix freely with explicit :meth:`load` / :meth:`sp`.

        Example
        -------
        ::

            with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
                p.from_model("dead")
                p.from_model("live")
        """
        if not isinstance(case, str) or not case:
            raise ValueError(
                f"Plain.from_model: case must be a non-empty str (got "
                f"{case!r})."
            )
        self._from_model_cases_.append(case)

    def moment_tensor(
        self,
        *,
        position: "tuple[float, ...] | list[float]",
        frame: str,
        M0: float = 1.0,
        mech: "dict[str, float] | None" = None,
        m_ij: "object | None" = None,
        t0: float = 0.0,
        method: str = "consistent",
        region: str | None = None,
    ) -> None:
        """Record a moment-tensor seismic source inside this pattern (ADR 0062).

        Embeds a seismic point source at ``position`` (mesh coordinates)
        and emits the representation-theorem equivalent **nodal forces**
        ``M0·m_ij·∂N_a/∂x_j`` — a pure right-hand-side contribution, no
        stiffness / constraint, integrator-agnostic. The pattern's time
        series carries the **normalized moment function** ``S(t)`` (rising
        ``0 → 1``); build it with ``ops.timeSeries.MomentStep`` / ``Yoffe``.

        Parameters
        ----------
        position
            Source point in mesh coordinates ``(x, y, z)`` (or ``(x, y)``
            for a 2D model). The host element is found automatically; the
            point must lie inside the intact continuum.
        frame
            Mesh vertical convention — ``"z-up"`` (typical apeGmsh: free
            surface on top, depth positive downward; flips the A&R z-down
            tensor) or ``"z-down"``. **Required** — wrong frame silently
            mirrors the radiation pattern.
        M0
            Scalar seismic moment ``M0 = μ·A·D̄`` in deck moment units
            (e.g. kN·m). Default ``1.0`` (unit source).
        mech
            Fault mechanism ``dict(strike=, dip=, rake=)`` in degrees, OR
            pass ``m_ij`` — exactly one.
        m_ij
            A 3×3 unit moment tensor (A&R z-down), as an alternative to
            ``mech``.
        t0
            Rupture onset (s). Only ``0.0`` is supported in v1 — a
            per-source onset delay (a finite fault) is MT-4.
        method
            ``"consistent"`` (host ``∂N/∂x`` — works on any solid mesh) or
            ``"dipole"`` (±neighbour couples on a structured grid).
        region
            Optional PG / label name restricting the host search to that
            region's elements. Use it to confine the source to the intact
            continuum (e.g. ``region="soil"``) so a point in the absorbing
            skin or outside the region fails loud instead of silently
            sourcing into boundary cells.
        """
        if (mech is None) == (m_ij is None):
            raise ValueError(
                "Plain.moment_tensor: supply exactly one of mech= "
                "(dict(strike=, dip=, rake=)) or m_ij= (a 3x3 unit tensor), "
                f"got mech={mech!r}, m_ij={'<array>' if m_ij is not None else None}."
            )
        if frame not in _MT_FRAMES:
            raise ValueError(
                f"Plain.moment_tensor: frame must be one of {_MT_FRAMES}, "
                f"got {frame!r} (no default — the flip mirrors the radiation "
                f"pattern if wrong)."
            )
        if method not in _MT_METHODS:
            raise ValueError(
                f"Plain.moment_tensor: method must be one of {_MT_METHODS}, "
                f"got {method!r}."
            )
        strike = dip = rake = None
        m_tuple: tuple[tuple[float, ...], ...] | None = None
        if mech is not None:
            missing = {"strike", "dip", "rake"} - set(mech)
            if missing:
                raise ValueError(
                    f"Plain.moment_tensor: mech= needs strike, dip, rake "
                    f"(missing {sorted(missing)})."
                )
            strike = float(mech["strike"])
            dip = float(mech["dip"])
            rake = float(mech["rake"])
        else:
            rows = [tuple(float(v) for v in row) for row in m_ij]  # type: ignore[union-attr]
            if len(rows) != 3 or any(len(r) != 3 for r in rows):
                raise ValueError(
                    "Plain.moment_tensor: m_ij must be a 3x3 array-like."
                )
            m_tuple = tuple(rows)
        self._moment_tensors_.append(_MomentTensorRecord(
            position=tuple(float(c) for c in position),
            frame=frame,
            method=method,
            t0=float(t0),
            M0=float(M0),
            strike=strike,
            dip=dip,
            rake=rake,
            m_ij=m_tuple,
            region=region,
        ))

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
        if self._from_model_cases_:
            raise NotImplementedError(
                "Plain._emit: from_model(case) import is the bridge build "
                "pipeline's job (it owns the FEM snapshot + ndf). Emit via "
                "the bridge (ops.tcl/py/build), not the primitive directly."
            )
        if self._moment_tensors_:
            raise NotImplementedError(
                "Plain._emit: moment_tensor(...) resolution is the bridge "
                "build pipeline's job (it owns the FEM snapshot needed to "
                "locate the host + evaluate ∂N/∂x). Emit via the bridge "
                "(ops.tcl/py/build), not the primitive directly."
            )
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
        for sp in self._sps_:
            if sp.target_kind == "node":
                emitter.sp(int(sp.target), sp.dof, sp.value)
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
