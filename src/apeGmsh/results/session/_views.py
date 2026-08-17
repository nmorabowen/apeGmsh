"""Mesh and plot views — the panes of a ``ResultsSession``.

ADR 0098 §3 (mesh view), §4 (slots on the view), §5 (legends derived
from slots), §6 (plot view). Views are mutable session state with
value-record fields: every mutator replaces a frozen record and fires
the owning session's change tick, so the Qt client is a projection and
never a second copy of this state.

The pose (``deform``) is NOT a slot and never causes a legend; the
four style buttons are view chrome, not slots; edges/outlines are a
style of cells that are on, never independent objects.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Union

from ._slots import (
    COLOUR_MAPPED,
    SLOT_CATALOG,
    Contour,
    Gauss,
    Line,
    Loads,
    Reactions,
    Sand,
    Slot,
    Vector,
    slot_field,
)
from ._time import Instant

_Notify = Optional[Callable[[], None]]


# ======================================================================
# Value records of the mesh view
# ======================================================================

_DEFORM_FIELDS = frozenset({"displacement", "velocity", "acceleration"})


@dataclass(frozen=True)
class Deform:
    """The pose of a mesh view — off (``view.deform = None``) or
    ``(field, scale, mode?)``. A pose, never a picture: it emits no
    legend (§5 INV-LEGEND-1).

    ``scale=None`` means auto-fit at realize (the existing attach-time
    convention). ``mode`` set makes this a mode pose: the view has no
    ``(stage, step)`` and is frozen under the session time link (§4/§7).
    The mode index is an opaque token here; S1 resolves it against the
    broker's modal stage.
    """

    field: str = "displacement"
    scale: Optional[float] = None
    mode: Optional[int] = None

    def __post_init__(self) -> None:
        if self.field not in _DEFORM_FIELDS:
            raise ValueError(
                f"Deform.field must be one of "
                f"{sorted(_DEFORM_FIELDS)}; got {self.field!r}."
            )
        if self.scale is not None:
            object.__setattr__(self, "scale", float(self.scale))
        if self.mode is not None:
            if isinstance(self.mode, bool) or not isinstance(self.mode, int):
                raise TypeError(
                    f"Deform.mode must be an int mode index; got "
                    f"{self.mode!r}."
                )


@dataclass(frozen=True)
class MeshStyle:
    """The four style buttons (§3 INV-MESH-4) — view chrome, not slots.

    ``gauss`` here draws integration-point *locations* (required to
    click-pick Gauss); the ``gauss`` slot draws *values*. They must not
    draw two clouds — the realize layer owns that merge.

    Boot picture: mesh + outlines on, nodes / gauss off (§3).
    """

    mesh: bool = True
    outlines: bool = True
    nodes: bool = False
    gauss: bool = False

    def __post_init__(self) -> None:
        # Coerce to real bools: the S2 reconciler diffs records by
        # equality, and MeshStyle(mesh="yes") != MeshStyle(mesh=True)
        # would read as a change that repaints nothing (probe H11).
        for field_name in ("mesh", "outlines", "nodes", "gauss"):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))


_SCOPE_AXES = frozenset({"physical_groups", "materials", "element_types"})


@dataclass(frozen=True)
class Scope:
    """What a mesh view draws — ONE composition axis plus one or more
    checked names on it, or all names on that axis (§3/§9; never a
    boolean ``concrete AND hex``).

    ``names=None`` = every name on the axis; ``view.scope = None`` = no
    scope at all (the whole analysis mesh). Names are opaque tokens
    here — resolution against the data (including the materials /
    element-type indexes) is S1/S3 work.
    """

    axis: str
    names: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.axis not in _SCOPE_AXES:
            raise ValueError(
                f"Scope.axis must be one of {sorted(_SCOPE_AXES)}; got "
                f"{self.axis!r}."
            )
        if self.names is not None:
            names = tuple(self.names)
            if not names or any(
                not isinstance(n, str) or not n for n in names
            ):
                raise ValueError(
                    "Scope.names must be a non-empty sequence of "
                    f"non-empty name strings, or None for all; got "
                    f"{self.names!r}."
                )
            object.__setattr__(self, "names", names)


def _unit(normal: Sequence[float]) -> tuple[float, float, float]:
    """``normal`` as a unit vector; degenerate input falls back to +X
    (same tolerance-free semantics as the 0083 controller, copied — the
    controller itself is never imported here)."""
    x, y, z = (float(normal[0]), float(normal[1]), float(normal[2]))
    mag = math.sqrt(x * x + y * y + z * z)
    if mag <= 0.0:
        return (1.0, 0.0, 0.0)
    return (x / mag, y / mag, z / mag)


@dataclass(frozen=True)
class ViewClip:
    """One section plane of one view — copies the ADR 0083 ``ClipPlane``
    field shape verbatim (the 0083 machinery is kept; *ownership* moved
    viewer → view). Frozen here: edits go through
    :meth:`MeshView.set_clip`, which replaces the record and ticks.

    The six-active-planes GL cap is a render-time constraint enforced
    where planes meet a backend (S1/S3), not by the IR.
    """

    plane_id: str
    name: str
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    offset: float = 0.0
    active: bool = True
    flipped: bool = False
    gizmo_visible: bool = True

    def __post_init__(self) -> None:
        # Coerce to an immutable float triple: a directly-constructed
        # record (the S5 restore path) holding a caller's list would be
        # mutable through the frozen shell and equality-fragile (probe
        # H10). Normalisation to unit length stays in add_clip/set_clip.
        x, y, z = self.normal
        object.__setattr__(self, "normal", (float(x), float(y), float(z)))
        object.__setattr__(self, "offset", float(self.offset))
        for field_name in ("active", "flipped", "gizmo_visible"):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))


@dataclass(frozen=True)
class Legend:
    """One derived colour scale of one mesh view (§5). Never stored:
    ``legends = f(occupied colour-mapped slots)``. ``field`` names the
    slot quantity (INV-LEGEND-4); ``categories`` are the slots that
    cause it — two slots of the same field share the one scale.
    ``hidden`` is view chrome (INV-LEGEND-3): hiding the scale does not
    hide the picture, and does not clear the slot."""

    field: str
    categories: tuple[str, ...]
    hidden: bool = False


PICK_TARGETS = ("nodes", "gauss")

#: Legend derivation walks colour-mapped categories in §4 row order so
#: the emitted tuple is deterministic.
_COLOUR_ORDER = tuple(c for c in SLOT_CATALOG if c in COLOUR_MAPPED)


# ======================================================================
# MeshView
# ======================================================================

def _slot_property(category: str, record_type: type) -> property:
    """One §4 slot as a view property: read the occupant (or ``None``),
    assign to fill (replacing any occupant — the §4 law), assign
    ``None`` to clear. A record of the wrong category refuses loudly —
    the catalog is closed and typed."""

    def fget(self: "MeshView") -> Optional[Slot]:
        return self._slots.get(category)

    def fset(self: "MeshView", value: Optional[Slot]) -> None:
        if value is not None and not isinstance(value, record_type):
            got = type(value).__name__
            raise TypeError(
                f"The {category!r} slot takes a "
                f"{record_type.__name__} record or None; got {got}."
            )
        self._set_slot(category, value)

    fget.__name__ = category
    return property(
        fget, fset,
        doc=(
            f"The {category!r} slot ({record_type.__name__} or None). "
            "Assigning replaces the occupant; None clears."
        ),
    )


class MeshView:
    """A pane on the **analysis mesh** (§3). No BRep, no CAD.

    Booted with the valid empty picture: grey mesh, mesh + outlines
    on, nodes / gauss off, no slots, no legends, unscoped, undeformed.
    """

    def __init__(
        self,
        pane_id: str,
        name: Optional[str] = None,
        on_changed: _Notify = None,
    ) -> None:
        self._id = str(pane_id)
        self._name = name
        self._notify = on_changed
        self._scope: Optional[Scope] = None
        self._deform: Optional[Deform] = None
        self._time: Optional[Instant] = None
        self._style = MeshStyle()
        self._overlay = False
        self._pick_target = "nodes"
        self._slots: dict[str, Slot] = {}
        self._legend_hidden: dict[str, bool] = {}
        self._clips: list[ViewClip] = []
        self._clip_ids = itertools.count(1)

    # -- identity ------------------------------------------------------

    @property
    def id(self) -> str:
        """Stable pane id — how render, the MCP verb, the outline and
        the snapshot address this pane (§1)."""
        return self._id

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is not None:
            value = str(value)
        if value == self._name:
            return
        self._name = value
        self._changed()

    # -- scope / pose / time / chrome ----------------------------------

    @property
    def scope(self) -> Optional[Scope]:
        """One composition axis + checked names, or None = whole mesh.
        Scope chooses the ONE cell set every layer of this view is a
        function of (INV-MESH-1); it is a different knob from selection
        (§8)."""
        return self._scope

    @scope.setter
    def scope(self, value: Optional[Scope]) -> None:
        if value is not None and not isinstance(value, Scope):
            raise TypeError(
                f"MeshView.scope takes a Scope or None; got "
                f"{type(value).__name__}."
            )
        if value == self._scope:
            return
        self._scope = value
        self._changed()

    @property
    def deform(self) -> Optional[Deform]:
        """The pose — off (None) or on (field, scale, mode?). Never a
        picture, never a legend (§4/§5)."""
        return self._deform

    @deform.setter
    def deform(self, value: Optional[Deform]) -> None:
        if value is not None and not isinstance(value, Deform):
            raise TypeError(
                f"MeshView.deform takes a Deform or None; got "
                f"{type(value).__name__}."
            )
        if value == self._deform:
            return
        self._deform = value
        self._changed()

    @property
    def time(self) -> Optional[Instant]:
        """This pane's own instant. Ignored while the session link is
        on (§7/§9 — set it here; the link ignores it); irrelevant for a
        mode pose, which has no instant."""
        return self._time

    @time.setter
    def time(self, value: Optional[Instant]) -> None:
        if value is not None and not isinstance(value, Instant):
            raise TypeError(
                f"MeshView.time takes an Instant or None; got "
                f"{type(value).__name__}."
            )
        if value == self._time:
            return
        self._time = value
        self._changed()

    @property
    def style(self) -> MeshStyle:
        """The four style buttons (INV-MESH-4) — independent toggles of
        cells that are ON. They never change the cell set or the
        selection set (§8)."""
        return self._style

    @style.setter
    def style(self, value: MeshStyle) -> None:
        if not isinstance(value, MeshStyle):
            raise TypeError(
                f"MeshView.style takes a MeshStyle; got "
                f"{type(value).__name__}."
            )
        if value == self._style:
            return
        self._style = value
        self._changed()

    @property
    def overlay(self) -> bool:
        """Undeformed overlay — this mesh at scale 0 in the Outlines
        style, same cell set (§3). Never CAD, never a second Geometry."""
        return self._overlay

    @overlay.setter
    def overlay(self, value: bool) -> None:
        value = bool(value)
        if value == self._overlay:
            return
        self._overlay = value
        self._changed()

    @property
    def pick_target(self) -> str:
        """The Nodes | Gauss radio (§8). Aims clicks and windows in
        THIS view only; it neither owns nor clears the session's one
        selection set."""
        return self._pick_target

    @pick_target.setter
    def pick_target(self, value: str) -> None:
        if value not in PICK_TARGETS:
            raise ValueError(
                f"MeshView.pick_target must be one of {PICK_TARGETS}; "
                f"got {value!r}."
            )
        if value == self._pick_target:
            return
        self._pick_target = value
        self._changed()

    @property
    def is_mode_posed(self) -> bool:
        """Whether the pose is a mode shape — no instant, frozen under
        the session time link (§4/§7)."""
        return self._deform is not None and self._deform.mode is not None

    # -- slots (§4) ----------------------------------------------------

    contour = _slot_property("contour", Contour)
    vector = _slot_property("vector", Vector)
    gauss = _slot_property("gauss", Gauss)
    line = _slot_property("line", Line)
    sand = _slot_property("sand", Sand)
    loads = _slot_property("loads", Loads)
    reactions = _slot_property("reactions", Reactions)

    @property
    def slots(self) -> dict[str, Slot]:
        """Occupied categories → occupant, in §4 catalog order."""
        return {
            c: self._slots[c] for c in SLOT_CATALOG if c in self._slots
        }

    def _set_slot(self, category: str, record: Optional[Slot]) -> None:
        if record is None:
            if category not in self._slots:
                return
            del self._slots[category]
        else:
            if self._slots.get(category) == record:
                return
            self._slots[category] = record
        # Legend chrome lives only as long as its legend (INV-LEGEND-2:
        # turning the slot off destroys the legend) — prune hidden
        # flags for fields no longer caused by any occupied slot.
        live = {
            f for f in (
                slot_field(c, r) for c, r in self._slots.items()
            ) if f is not None
        }
        self._legend_hidden = {
            f: h for f, h in self._legend_hidden.items() if f in live
        }
        self._changed()

    # -- legends (§5) --------------------------------------------------

    def legends(self) -> tuple[Legend, ...]:
        """The colour scales this view carries — derived, never stored:
        one per distinct field over the occupied colour-mapped slots
        (INV-LEGEND-1/-2/-5). Same quantity on two slots → one scale.
        Deform on with every slot empty → zero legends."""
        fields: dict[str, list[str]] = {}
        for category in _COLOUR_ORDER:
            record = self._slots.get(category)
            if record is None:
                continue
            field = slot_field(category, record)
            fields.setdefault(field, []).append(category)
        return tuple(
            Legend(
                field=field,
                categories=tuple(categories),
                hidden=self._legend_hidden.get(field, False),
            )
            for field, categories in fields.items()
        )

    def set_legend_hidden(self, field: str, hidden: bool = True) -> None:
        """Hide/show one scale — view chrome (INV-LEGEND-3). Does not
        touch the slot: the picture stays painted. Refuses a field no
        occupied colour-mapped slot causes (that legend does not
        exist — INV-LEGEND-2)."""
        live = {legend.field for legend in self.legends()}
        if field not in live:
            raise ValueError(
                f"No legend for field {field!r} on this view — legends "
                f"exist only for occupied colour-mapped slots (have: "
                f"{sorted(live)})."
            )
        hidden = bool(hidden)
        if self._legend_hidden.get(field, False) == hidden:
            return
        self._legend_hidden[field] = hidden
        self._changed()

    # -- clips (§3, 0083 machinery view-owned) -------------------------

    @property
    def clips(self) -> tuple[ViewClip, ...]:
        """This view's section planes, in creation order."""
        return tuple(self._clips)

    def add_clip(
        self,
        normal: Sequence[float],
        *,
        offset: float = 0.0,
        name: Optional[str] = None,
        active: bool = True,
        flipped: bool = False,
        gizmo_visible: bool = True,
    ) -> ViewClip:
        clip = ViewClip(
            plane_id=f"clip-{next(self._clip_ids)}",
            name=name or f"Plane {len(self._clips) + 1}",
            normal=_unit(normal),
            offset=float(offset),
            active=bool(active),
            flipped=bool(flipped),
            gizmo_visible=bool(gizmo_visible),
        )
        self._clips.append(clip)
        self._changed()
        return clip

    def remove_clip(self, plane_id: str) -> None:
        for clip in self._clips:
            if clip.plane_id == plane_id:
                self._clips.remove(clip)
                self._changed()
                return
        raise KeyError(f"No clip {plane_id!r} on view {self._id!r}.")

    def set_clip(self, plane_id: str, /, **changes: object) -> ViewClip:
        """Replace fields of one clip (frozen record swap). ``plane_id``
        is identity and cannot be changed."""
        if "plane_id" in changes:
            raise ValueError("ViewClip.plane_id is identity — not settable.")
        if "normal" in changes:
            changes["normal"] = _unit(changes["normal"])  # type: ignore[arg-type]
        if "offset" in changes:
            changes["offset"] = float(changes["offset"])  # type: ignore[arg-type]
        for i, clip in enumerate(self._clips):
            if clip.plane_id == plane_id:
                new = replace(clip, **changes)  # type: ignore[arg-type]
                if new == clip:
                    return clip
                self._clips[i] = new
                self._changed()
                return new
        raise KeyError(f"No clip {plane_id!r} on view {self._id!r}.")

    # -- internal ------------------------------------------------------

    def _changed(self) -> None:
        if self._notify is not None:
            self._notify()

    def __repr__(self) -> str:
        occupied = ", ".join(self.slots) or "no slots"
        return f"<MeshView {self._id!r} ({occupied})>"


# ======================================================================
# PlotView (§6)
# ======================================================================

PLOT_KINDS = ("history", "path", "xy")

_SOURCE_KINDS = ("node", "gauss", "label", "physical_group")


@dataclass(frozen=True)
class PlotSource:
    """One series source (§6): a node, a Gauss point, a label, or a
    physical group. The session selection is NOT a source kind — the
    "select → New plot" flow COPIES the membership into concrete
    node/gauss sources at creation (a live alias is not v1). "Live"
    means live against ``Results`` at the cursor, not live membership.
    """

    kind: str
    key: Union[int, tuple[int, int], str]

    def __post_init__(self) -> None:
        if self.kind not in _SOURCE_KINDS:
            raise ValueError(
                f"PlotSource.kind must be one of {_SOURCE_KINDS}; got "
                f"{self.kind!r}."
            )
        if self.kind == "node":
            object.__setattr__(self, "key", int(self.key))  # type: ignore[arg-type]
        elif self.kind == "gauss":
            eid, gp = self.key  # type: ignore[misc]
            object.__setattr__(self, "key", (int(eid), int(gp)))
        else:
            if not isinstance(self.key, str) or not self.key:
                raise ValueError(
                    f"PlotSource({self.kind!r}).key must be a non-empty "
                    f"name string; got {self.key!r}."
                )

    @classmethod
    def node(cls, node_id: int) -> "PlotSource":
        return cls("node", node_id)

    @classmethod
    def gauss(cls, element_id: int, gp_index: int) -> "PlotSource":
        return cls("gauss", (element_id, gp_index))

    @classmethod
    def label(cls, name: str) -> "PlotSource":
        return cls("label", name)

    @classmethod
    def physical_group(cls, name: str) -> "PlotSource":
        return cls("physical_group", name)


@dataclass(frozen=True)
class PlotSeries:
    """One curve: a source + a quantity token. Several series on one
    chart are one plot view (§6)."""

    source: PlotSource
    quantity: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, PlotSource):
            raise TypeError(
                f"PlotSeries.source must be a PlotSource; got "
                f"{type(self.source).__name__}."
            )
        if not isinstance(self.quantity, str) or not self.quantity:
            raise ValueError(
                f"PlotSeries.quantity must be a non-empty string token; "
                f"got {self.quantity!r}."
            )


class PlotView:
    """A plot is a pane, not a dock of a contour (§6).

    ``kind`` is fixed at creation; series are live queries against
    ``Results`` evaluated at the cursor. A history plot shows the whole
    record with the cursor as time; a path plot is evaluated at the
    current instant (§7)."""

    def __init__(
        self,
        pane_id: str,
        kind: str = "history",
        series: Sequence[PlotSeries] = (),
        name: Optional[str] = None,
        on_changed: _Notify = None,
    ) -> None:
        if kind not in PLOT_KINDS:
            raise ValueError(
                f"PlotView.kind must be one of {PLOT_KINDS}; got "
                f"{kind!r}."
            )
        self._id = str(pane_id)
        self._kind = kind
        self._name = name
        self._notify = on_changed
        self._series: tuple[PlotSeries, ...] = ()
        self._cursor: Optional[Instant] = None
        self._series = self._coerce_series(series)

    @property
    def id(self) -> str:
        return self._id

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is not None:
            value = str(value)
        if value == self._name:
            return
        self._name = value
        self._changed()

    @property
    def series(self) -> tuple[PlotSeries, ...]:
        return self._series

    @series.setter
    def series(self, value: Sequence[PlotSeries]) -> None:
        new = self._coerce_series(value)
        if new == self._series:
            return
        self._series = new
        self._changed()

    @property
    def cursor(self) -> Optional[Instant]:
        """This plot's own instant. Rides the session instant while the
        link is on (§7)."""
        return self._cursor

    @cursor.setter
    def cursor(self, value: Optional[Instant]) -> None:
        if value is not None and not isinstance(value, Instant):
            raise TypeError(
                f"PlotView.cursor takes an Instant or None; got "
                f"{type(value).__name__}."
            )
        if value == self._cursor:
            return
        self._cursor = value
        self._changed()

    @staticmethod
    def _coerce_series(value: Sequence[PlotSeries]) -> tuple[PlotSeries, ...]:
        series = tuple(value)
        for s in series:
            if not isinstance(s, PlotSeries):
                raise TypeError(
                    f"PlotView.series items must be PlotSeries; got "
                    f"{type(s).__name__}."
                )
        return series

    def _changed(self) -> None:
        if self._notify is not None:
            self._notify()

    def __repr__(self) -> str:
        return (
            f"<PlotView {self._id!r} {self._kind} "
            f"({len(self._series)} series)>"
        )


__all__ = [
    "Deform",
    "MeshStyle",
    "Scope",
    "ViewClip",
    "Legend",
    "PICK_TARGETS",
    "MeshView",
    "PLOT_KINDS",
    "PlotSource",
    "PlotSeries",
    "PlotView",
]
