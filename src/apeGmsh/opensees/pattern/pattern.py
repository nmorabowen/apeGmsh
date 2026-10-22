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

import math

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import numbers
from typing import TYPE_CHECKING

from .._internal.build.errors import BridgeError
from .._internal.tag_resolution import resolve_tag
from .._internal.types import Element, Pattern, Primitive, TimeSeries

if TYPE_CHECKING:
    from ..emitter.base import Emitter
    from ..node import Node


class _BridgeValueError(BridgeError, ValueError):
    """Error that satisfies both BridgeError and ValueError checks."""


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
    "H5DRM",
    "_LoadRecord",
    "_SPRecord",
    "_EleLoadRecord",
    "_MomentTensorRecord",
]

# Identity transform — the default DRM frame handshake (km dataset built
# centred in a z-down metres model ⇒ T = I, x0 = 0, crd_scale rescales).
_IDENTITY_3X3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

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
class _EleLoadRecord:
    """One element-load entry inside a :class:`Plain` pattern.

    A value object — not a :class:`Primitive`. Held in a tuple by the
    parent :class:`Plain` instance.
    """

    load_type: str
    target_kind: str
    target: object
    wy: tuple[float, float] = (0.0, 0.0)
    wz: tuple[float, float] | None = None
    wx: tuple[float, float] = (0.0, 0.0)
    aL: float = 0.0
    bL: float = 1.0


def _resolve_ele_target(
    method_name: str,
    *,
    pg: str | None,
    label: str | None,
    ele: "int | Element | Sequence[int | Element] | None",
) -> tuple[str, object]:
    """Validate and normalize ``pg=`` / ``label=`` / ``ele=`` for ``Plain`` element loads."""
    given = int(pg is not None) + int(label is not None) + int(ele is not None)
    if given != 1:
        raise ValueError(
            f"{method_name}: supply exactly one of pg=, label=, or ele= "
            f"(got pg={pg!r}, label={label!r}, ele={ele!r})."
        )
    if pg is not None:
        if not isinstance(pg, str) or not pg:
            raise ValueError(f"{method_name}: pg must be a non-empty str, got {pg!r}.")
        return "pg", pg
    if label is not None:
        if not isinstance(label, str) or not label:
            raise ValueError(f"{method_name}: label must be a non-empty str, got {label!r}.")
        return "label", label
    assert ele is not None
    if isinstance(ele, Element) or (isinstance(ele, numbers.Integral) and not isinstance(ele, bool)):
        raw_items: tuple[object, ...] = (ele,)
    elif isinstance(ele, Iterable) and not isinstance(ele, (str, bytes)):
        raw_items = tuple(ele)
        if not raw_items:
            raise ValueError(f"{method_name}: ele= sequence must not be empty.")
    else:
        raise ValueError(
            f"{method_name}: ele= must be an int, Element, or sequence thereof, got {ele!r}."
        )
    out: list[int | Element] = []
    for item in raw_items:
        if isinstance(item, Element):
            out.append(item)
        elif isinstance(item, numbers.Integral) and not isinstance(item, bool):
            out.append(int(item))
        else:
            raise ValueError(
                f"{method_name}: invalid ele= item {item!r}; expected int or Element."
            )
    return "ele", tuple(out)


def _normalize_beam_w(
    val: float | tuple[float, float], name: str,
) -> tuple[float, float]:
    """Normalize a scalar or ``(w_a, w_b)`` 2-tuple into a float 2-tuple."""
    if isinstance(val, (tuple, list)):
        if len(val) != 2:
            raise ValueError(
                f"Plain.beam_uniform: {name} tuple must have length 2 (w_a, w_b), got {val!r}."
            )
        return (float(val[0]), float(val[1]))
    v = float(val)
    return (v, v)


def _beam_uniform_args(rec: _EleLoadRecord, ndm: int) -> tuple[float, ...]:
    """Build trailing float args for ``eleLoad ... -type -beamUniform``."""
    wy_a, wy_b = rec.wy
    wx_a, wx_b = rec.wx
    a_l, b_l = rec.aL, rec.bL
    is_full_span = a_l == 0.0 and b_l == 1.0
    if ndm == 2:
        if rec.wz is not None and (rec.wz[0] != 0.0 or rec.wz[1] != 0.0):
            raise _BridgeValueError(
                "Plain.beam_uniform: wz is not supported in 2D (ndm=2); "
                f"got wz={rec.wz!r}."
            )
        is_uniform = wy_a == wy_b and wx_a == wx_b
        if is_full_span and is_uniform:
            return (wy_a, wx_a)
        if is_uniform:
            return (wy_a, wx_a, a_l, b_l)
        return (wy_a, wx_a, a_l, b_l, wy_b, wx_b)
    wz_a, wz_b = rec.wz if rec.wz is not None else (0.0, 0.0)
    is_uniform = wy_a == wy_b and wz_a == wz_b and wx_a == wx_b
    if is_full_span and is_uniform:
        return (wy_a, wz_a, wx_a)
    if is_uniform:
        return (wy_a, wz_a, wx_a, a_l, b_l)
    return (wy_a, wz_a, wx_a, a_l, b_l, wy_b, wz_b, wx_b)


def _membrane_pressure_args(rec: _EleLoadRecord, ndm: int) -> tuple[float, ...]:
    """Trailing args of a ``surface_load(pressure=q)`` on a LadrunoMembrane:
    ``-beamUniform 0.0 q 0.0`` (``wy wz wx``; the fork reads ``wz`` along the
    membrane normal)."""
    if ndm != 3:
        raise _BridgeValueError(
            "Plain.surface_load(pressure=...): a per-pattern membrane "
            f"pressure needs a 3-D model (ndm=3); got ndm={ndm}. In 2-D the "
            "fork reads -beamUniform as an in-plane load."
        )
    q = rec.wz[0] if rec.wz is not None else 0.0
    return (0.0, float(q), 0.0)


def _self_weight_args(rec: _EleLoadRecord, ndm: int) -> tuple[float, ...]:
    """Build trailing float args for ``eleLoad ... -type -selfWeight``."""
    wx_a = rec.wx[0]
    wy_a = rec.wy[0]
    if ndm == 2:
        if rec.wz is not None and rec.wz[0] != 0.0:
            raise _BridgeValueError(
                "Plain.self_weight: wz is not supported in 2D (ndm=2); "
                f"got wz={rec.wz[0]!r}."
            )
        return (wx_a, wy_a)
    wz_a = rec.wz[0] if rec.wz is not None else 0.0
    return (wx_a, wy_a, wz_a)


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
    _from_model_allow_empty_: list[str] = field(
        init=False, repr=False, compare=False, default_factory=list,
    )
    _moment_tensors_: list[_MomentTensorRecord] = field(
        init=False, repr=False, compare=False, default_factory=list,
    )

    @property
    def _ele_loads_(self) -> list[_EleLoadRecord]:
        lst: list[_EleLoadRecord] = self.__dict__.setdefault("_ele_loads_", [])
        return lst

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
    def ele_loads(self) -> tuple[_EleLoadRecord, ...]:
        """Tuple snapshot of the recorded :class:`_EleLoadRecord` entries."""
        return tuple(self.__dict__.get("_ele_loads_", ()))

    @property
    def from_model_cases(self) -> tuple[str, ...]:
        """Tuple snapshot of the imported geometry load-case names.

        Populated by :meth:`from_model`.  The bridge expands each case
        into per-node ``load`` / ``sp`` lines at emit time (it owns the
        FEM snapshot and the model ``ndf``).
        """
        return tuple(self._from_model_cases_)

    @property
    def from_model_allow_empty(self) -> tuple[str, ...]:
        """Case names imported with ``allow_empty=True``.

        These are exempt from the build-time zero-match guard (a
        ``from_model`` case matching zero importable records raises
        :class:`BridgeError` otherwise).
        """
        return tuple(self._from_model_allow_empty_)

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

        There is no ``label=``. To load the one node a geometry label
        names, select it with ``ops.nodes.one``, which raises unless the
        label picks exactly one node::

            with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
                p.load(node=ops.nodes.one(label="tip"),
                       forces=(0.0, -5e3, 0.0))
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

    def beam_uniform(
        self,
        *,
        pg: str | None = None,
        label: str | None = None,
        ele: "int | Element | Sequence[int | Element] | None" = None,
        wy: float | tuple[float, float] = 0.0,
        wz: float | tuple[float, float] | None = None,
        wx: float | tuple[float, float] = 0.0,
        aL: float = 0.0,
        bL: float = 1.0,
        span: tuple[float, float] | None = None,
    ) -> None:
        """Record a beam distributed load (``eleLoad -type -beamUniform``).

        All load components (``wy``, ``wz``, ``wx``) are in **local element
        coordinates**. Each component may be a scalar ``float`` (uniform over
        the loaded segment) or a 2-tuple ``(w_a, w_b)`` (linearly varying /
        trapezoidal over ``[aL, bL]``).
        """
        target_kind, target = _resolve_ele_target(
            "Plain.beam_uniform", pg=pg, label=label, ele=ele,
        )
        if span is not None:
            if aL != 0.0 or bL != 1.0:
                raise ValueError(
                    "Plain.beam_uniform: pass either span=(aL, bL) or aL=/bL=, not both."
                )
            if not isinstance(span, (tuple, list)) or len(span) != 2:
                raise ValueError(
                    f"Plain.beam_uniform: span must be a 2-tuple (aL, bL), got {span!r}."
                )
            a_val, b_val = float(span[0]), float(span[1])
        else:
            a_val, b_val = float(aL), float(bL)
        if not (0.0 <= a_val < b_val <= 1.0):
            raise ValueError(
                f"Plain.beam_uniform: requires 0.0 <= aL < bL <= 1.0, "
                f"got aL={a_val}, bL={b_val}."
            )
        wy_pair = _normalize_beam_w(wy, "wy")
        wx_pair = _normalize_beam_w(wx, "wx")
        wz_pair = _normalize_beam_w(wz, "wz") if wz is not None else None
        self._ele_loads_.append(
            _EleLoadRecord(
                load_type="beamUniform",
                target_kind=target_kind,
                target=target,
                wy=wy_pair,
                wz=wz_pair,
                wx=wx_pair,
                aL=a_val,
                bL=b_val,
            )
        )

    def surface_load(
        self,
        *,
        pg: str | None = None,
        label: str | None = None,
        ele: "int | Element | Sequence[int | Element] | None" = None,
        pressure: float | None = None,
    ) -> None:
        """Record a surface-load activation (``eleLoad -type -surfaceLoad``).

        Without ``pressure=`` the targeted elements apply their own reference
        load (a ``LadrunoMembrane``'s ``pressure=``), scaled by the pattern's
        time series.

        ``pressure=q`` gives this pattern its own magnitude, so one panel can
        carry D, SD and L in three patterns with three values (an ETABS shell
        uniform load set). It is for ``LadrunoMembrane`` targets only
        (``ele=`` handles or the membrane's ``pg=``) and 3-D models. ``q``
        has the sign and normal of the element's ``pressure=``: the traction
        is ``q * n``, with ``n`` the membrane's normal (from its boundary
        loop's winding), exactly as ``pressure=q`` on the element would
        give. The bridge emits ``eleLoad -ele <m> -type -beamUniform 0.0 q
        0.0``: the fork's ``LadrunoMembrane::addLoad`` reads a
        ``-beamUniform`` load as the traction ``wy e2 + wz n + wx e1`` and
        distributes it through the same tributary transfer as
        ``-surfaceLoad`` (``qz = wz n``), so the load is exact. It never
        reads the element's ``pressure=``.
        """
        target_kind, target = _resolve_ele_target(
            "Plain.surface_load", pg=pg, label=label, ele=ele,
        )
        if pressure is None:
            self._ele_loads_.append(
                _EleLoadRecord(
                    load_type="surfaceLoad",
                    target_kind=target_kind,
                    target=target,
                )
            )
            return
        q = float(pressure)
        if not math.isfinite(q):
            raise ValueError(
                f"Plain.surface_load: pressure must be finite, got {pressure!r}."
            )
        if target_kind == "label":
            raise ValueError(
                "Plain.surface_load(pressure=...): label= resolves to mesh "
                "cells, and pressure= loads LadrunoMembrane macro-elements, "
                "which have none. Target the membrane by its handle (ele=) or "
                "its pg=."
            )
        if target_kind == "ele":
            assert isinstance(target, tuple)
            for item in target:
                if type(item).__name__ != "LadrunoMembrane":
                    raise ValueError(
                        "Plain.surface_load(pressure=...): pressure= is for "
                        "LadrunoMembrane targets only (it emits -beamUniform, "
                        "which another element reads as a beam load); got "
                        f"{item!r}. Pass the membrane handle."
                    )
        self._ele_loads_.append(
            _EleLoadRecord(
                load_type="membranePressure",
                target_kind=target_kind,
                target=target,
                wz=(q, q),
            )
        )

    def self_weight(
        self,
        *,
        pg: str | None = None,
        label: str | None = None,
        ele: "int | Element | Sequence[int | Element] | None" = None,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float | None = None,
    ) -> None:
        """Record a continuum self-weight multiplier (``eleLoad -type -selfWeight``)."""
        target_kind, target = _resolve_ele_target(
            "Plain.self_weight", pg=pg, label=label, ele=ele,
        )
        wx_f = float(wx)
        wy_f = float(wy)
        wz_pair = (float(wz), float(wz)) if wz is not None else None
        self._ele_loads_.append(
            _EleLoadRecord(
                load_type="selfWeight",
                target_kind=target_kind,
                target=target,
                wy=(wy_f, wy_f),
                wz=wz_pair,
                wx=(wx_f, wx_f),
            )
        )

    def from_model(self, case: str, *, allow_empty: bool = False) -> None:
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

        At build time the bridge verifies each imported case matches at
        least one importable record (a nodal load or a prescribed SP) in
        the broker and raises :class:`BridgeError` otherwise — a pattern
        import that silently imports nothing is almost always a typo or
        a case name lost to a stale ``model.h5``.  Pass
        ``allow_empty=True`` to exempt a case you expect may legitimately
        carry no records for this deck.

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
        if allow_empty:
            self._from_model_allow_empty_.append(case)

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
        for erec in self._ele_loads_:
            if erec.target_kind != "ele":
                raise NotImplementedError(
                    f"Plain._emit: {erec.target_kind}= ele_load fan-out is "
                    "the bridge build pipeline's job (Phase 4). Tests should "
                    "use integer ele= tags to exercise emit directly."
                )
            assert isinstance(erec.target, tuple)
            if any(not isinstance(t, int) for t in erec.target):
                raise NotImplementedError(
                    "Plain._emit: Element spec handle resolution is the "
                    "bridge build pipeline's job (Phase 4). Tests should use "
                    "integer ele= tags to exercise emit directly."
                )
            ele_tags = tuple(int(t) for t in erec.target)
            em_ndm = getattr(emitter, "_ndm", None) or getattr(
                emitter, "_model_ndm", None
            )
            ndm = int(em_ndm) if em_ndm is not None else (3 if erec.wz is not None else 2)
            if erec.load_type == "beamUniform":
                emitter.eleLoad(
                    "-ele",
                    *ele_tags,
                    "-type",
                    "-beamUniform",
                    *_beam_uniform_args(erec, ndm),
                )
            elif erec.load_type == "surfaceLoad":
                emitter.eleLoad("-ele", *ele_tags, "-type", "-surfaceLoad")
            elif erec.load_type == "membranePressure":
                emitter.eleLoad(
                    "-ele", *ele_tags, "-type", "-beamUniform",
                    *_membrane_pressure_args(erec, ndm),
                )
            elif erec.load_type == "selfWeight":
                emitter.eleLoad(
                    "-ele",
                    *ele_tags,
                    "-type",
                    "-selfWeight",
                    *_self_weight_args(erec, ndm),
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


# ---------------------------------------------------------------------------
# H5DRM — Domain Reduction Method pattern, field read from an .h5drm (ADR 0066)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class H5DRM(Pattern):
    """``pattern H5DRM tag file factor crd_scale dist_tol 1 T00..T22 x00..x02``.

    Drives a soil box with a regional incident wavefield read from an
    ``.h5drm`` HDF5 dataset (e.g. a ShakerMaker synthetic) via OpenSees'
    H5DRM load pattern. It is **field-carrying** — the payload IS the
    dataset, so there is no body (``p.load`` / ``p.from_model``) and **no
    TimeSeries** (the motion history lives in the file). The ``with`` block
    is permitted but optional, for symmetry with :class:`UniformExcitation`.

    This primitive owns the **frame handshake** the fork study proved is
    error-prone (ADR 0066 / OpenSees PR #296). The C++ transform is

        ``xyz_model = T · ((xyz_station − drmbox_x0) · crd_scale) + x0``

    where ``drmbox_x0`` is the box centre read from the file. Build the FE
    model **centred at the lateral origin, z-down, in metres** and the
    default ``crd_scale=1000`` (km→m) with identity ``transform`` / zero
    ``x0`` reproduces the station coordinates exactly (the study's 98/98
    node match). ``do_transform`` is emitted as ``1`` unconditionally — the
    transform is *how* the km→m rescale + centring happen.

    Parameters
    ----------
    h5drm
        Path to the ``.h5drm`` dataset. Emitted verbatim (Tcl tokens are
        unquoted per the apeGmsh convention) — use a forward-slash,
        space-free path, as OpenSees expects on Windows.
    factor
        Scalar applied to the DRM effective forces. Default ``1.0``.
    crd_scale
        Station-units → model-units scale. ShakerMaker stations are in km,
        FE models in m ⇒ default ``1000.0``.
    distance_tolerance
        Node-matching tolerance, in **model** units (m). Default ``1.0``.
    transform
        Optional 3×3 rotation ``T`` (row-major nested tuples / lists);
        ``None`` ⇒ identity (the built-in centred frame).
    x0
        Translation added after the scale/rotate. Default ``(0, 0, 0)``.

    Notes
    -----
    Mutually exclusive with a base-input absorbing drive (ADR 0054 §7.2:
    DRM ring **or** ``-fx/-fy/-fz`` base input, never both) and valid only
    for 3-DOF, ≤8-node elements (``stdBrick``). Those two
    checks need the FEM snapshot / the full pattern set, so they are
    enforced by the bridge build pipeline (and guaranteed by construction
    once ``g.parts.add_DRM_box_from_h5drm`` lands in D-2), not at the
    primitive level.
    """

    h5drm: str
    factor: float = 1.0
    crd_scale: float = 1000.0
    distance_tolerance: float = 1.0
    transform: tuple[tuple[float, ...], ...] | None = None
    x0: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.h5drm, str) or not self.h5drm:
            raise ValueError(
                f"H5DRM: h5drm must be a non-empty path str (got "
                f"{self.h5drm!r})."
            )
        if self.transform is not None:
            rows = tuple(tuple(float(c) for c in row) for row in self.transform)
            if len(rows) != 3 or any(len(r) != 3 for r in rows):
                raise ValueError(
                    "H5DRM: transform must be a 3x3 array-like (row-major), "
                    f"got {self.transform!r}."
                )
            object.__setattr__(self, "transform", rows)
        x0 = tuple(float(c) for c in self.x0)
        if len(x0) != 3:
            raise ValueError(f"H5DRM: x0 must be 3 floats, got {self.x0!r}.")
        object.__setattr__(self, "x0", x0)

    def __enter__(self) -> "H5DRM":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def dependencies(self) -> tuple[Primitive, ...]:
        # No TimeSeries — the motion history lives in the .h5drm file.
        return ()

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        T = self.transform if self.transform is not None else _IDENTITY_3X3
        flat_T = tuple(float(c) for row in T for c in row)
        emitter.pattern_open(
            "H5DRM", tag,
            self.h5drm,
            float(self.factor),
            float(self.crd_scale),
            float(self.distance_tolerance),
            1,                       # do_transform — always on (carries km→m + centring)
            *flat_T,
            self.x0[0], self.x0[1], self.x0[2],
        )
        emitter.pattern_close()
