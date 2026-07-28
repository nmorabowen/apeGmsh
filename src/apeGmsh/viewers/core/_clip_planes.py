"""ClipPlaneSetController — the viewer's section planes (ADR 0083).

A section plane is a property of the **view**, not of any diagram that
happens to be cut by it: the substrate, the node cloud and every
diagram share one cut, and two of those three have no diagram to own
them. So the plane set gets exactly one owner, a viewer-level
controller, sibling of :class:`~apeGmsh.viewers.core._legend.LegendController`
and bound the same way (ADR 0056 Part 2: mutators live on the owner,
each fires exactly one dispatcher event).

Three rules shape the state:

* **Planes are world-space by definition.** They cut whatever is
  currently rendered, so the deformed shape moves *through* a fixed
  plane as the scrubber plays — the wanted behaviour, and the reason
  the cut needs no per-step maintenance. A per-geometry ``offset``
  (ADR 0058 S3a) likewise slides its geometry through the plane;
  documented, not a bug.
* **At most six planes cut at once.** OpenGL guarantees six user clip
  distances and VTK inherits the limit, so a seventh would be silently
  dropped by the driver. The controller refuses it out loud instead
  (:data:`MAX_ACTIVE_PLANES`); creating more than six *inactive*
  planes is fine.
* **The backend receives resolved half-spaces.** ``offset`` and
  ``flipped`` are folded into :class:`ClipPlaneSpec` here, so the
  render side builds planes and makes no decisions.

The controller is a plain object (no Qt, no VTK), so it is fully
testable headless. One controller exists per render target, keyed
weakly by backend in :func:`controller_for` — the backend is
per-viewer by construction, the same argument ADR 0081 made for
legends.
"""
from __future__ import annotations

import itertools
import math
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
from weakref import WeakKeyDictionary


#: OpenGL's guaranteed minimum number of user clip distances, which VTK
#: inherits. Past this the driver is free to ignore planes, so the
#: controller refuses the activation rather than let the cut lie.
MAX_ACTIVE_PLANES = 6

#: Signed distances inside this band count as *on* the plane rather
#: than behind it, so a pick that lands exactly on the cut face is not
#: rejected as clipped-away by floating-point noise.
_ON_PLANE_TOL = 1e-9


def _weak(obj: Any) -> Any:
    """``weakref.ref(obj)``, or a plain thunk when ``obj`` forbids it."""
    if obj is None:
        return None
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


def _unit(normal: Sequence[float]) -> tuple[float, float, float]:
    """``normal`` as a unit vector; degenerate input falls back to +X.

    A zero-length normal has no half-space, and the numeric normal
    fields in the panel are typed into live — so the degenerate state
    is reachable by ordinary use and must not raise.
    """
    x, y, z = (float(normal[0]), float(normal[1]), float(normal[2]))
    mag = math.sqrt(x * x + y * y + z * z)
    if mag <= 0.0:
        return (1.0, 0.0, 0.0)
    return (x / mag, y / mag, z / mag)


@dataclass
class ClipPlane:
    """One section plane. Mutable — the controller owns it."""

    plane_id: str
    name: str
    #: Unit, world-space. The half-space this points into survives.
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    #: Signed distance origin → plane along ``normal``.
    offset: float = 0.0
    active: bool = True
    #: Which half survives (SketchUp's "Reverse").
    flipped: bool = False
    #: ADR 0083 S2 — honoured by the gizmo, stored here from S1 so a
    #: session saved now restores into the slice that draws them.
    gizmo_visible: bool = True

    def spec(self) -> Any:
        """This plane as a resolved :class:`ClipPlaneSpec`."""
        from ..scene_ir import ClipPlaneSpec

        n = _unit(self.normal)
        origin = tuple(c * float(self.offset) for c in n)
        if self.flipped:
            n = (-n[0], -n[1], -n[2])
        return ClipPlaneSpec(origin=origin, normal=n)


class ClipPlaneSetController:
    """Owns every section plane in one viewport (ADR 0083 Part 1).

    Every mutator reconciles — it pushes the resolved set to the
    backend — and then fires exactly one ``CLIP_PLANES_CHANGED``. The
    dock panel and (slice 2) the gizmo are projections of this state,
    never a second copy of it.
    """

    def __init__(self, backend: Any) -> None:
        # Weak, for the reason :func:`controller_for` is a
        # WeakKeyDictionary: a strong reference here would make the
        # value keep its own key alive and the backend, its plotter and
        # the whole scene would outlive the viewer that owned them.
        self._backend_ref = _weak(backend)
        self._planes: list[ClipPlane] = []
        self._ids = itertools.count(1)
        self._apply_cuts: bool = True
        self._show_gizmos: bool = True
        #: Injected by ``DiagramRegistry.bind`` (ADR 0056 Part 2 — owner
        #: mutators fire their own dispatcher events). ``None`` for a
        #: controller built outside a director, i.e. in unit tests.
        self.dispatcher: Any = None
        #: Optional ``callable(text)`` for user-facing refusals (the
        #: seventh active plane). Set by the viewer to its status bar.
        self.on_status: Optional[Callable[[str], None]] = None

    @property
    def _backend(self) -> Any:
        """The render target, or ``None`` once it has been collected."""
        return self._backend_ref() if self._backend_ref is not None else None

    # -- set-level state -----------------------------------------------

    @property
    def apply_cuts(self) -> bool:
        """Master enable — off leaves the planes defined but not cutting."""
        return self._apply_cuts

    @property
    def show_gizmos(self) -> bool:
        """Whether plane gizmos are drawn (ADR 0083 S2; inert in S1)."""
        return self._show_gizmos

    def set_apply_cuts(self, enabled: bool) -> None:
        if bool(enabled) == self._apply_cuts:
            return
        self._apply_cuts = bool(enabled)
        self._reconcile_and_fire()

    def set_show_gizmos(self, enabled: bool) -> None:
        if bool(enabled) == self._show_gizmos:
            return
        self._show_gizmos = bool(enabled)
        # Fires like any other mutator even though S1 draws nothing:
        # the flag is view state with one owner, and S2's gizmo layer
        # subscribes to the same event the panel already does.
        self._reconcile_and_fire()

    # -- queries -------------------------------------------------------

    def planes(self) -> list[ClipPlane]:
        """Every plane, in creation order (the panel's row order)."""
        return list(self._planes)

    def plane(self, plane_id: str) -> Optional[ClipPlane]:
        for p in self._planes:
            if p.plane_id == plane_id:
                return p
        return None

    def active_count(self) -> int:
        return sum(1 for p in self._planes if p.active)

    def specs(self) -> tuple:
        """The half-spaces the backend should currently cut with."""
        if not self._apply_cuts:
            return ()
        return tuple(p.spec() for p in self._planes if p.active)

    def is_clipped(self, point: Sequence[float]) -> bool:
        """Whether ``point`` sits on the discarded side of any plane.

        The pick-coherence filter (ADR 0083 Part 5). ``vtkCellPicker``
        is a geometric ray-cast and knows nothing about mapper clip
        planes, so without this a click on apparently-empty space
        returns the cell the cut is hiding and drives the probe HUD
        with data the user cannot see.
        """
        try:
            px, py, pz = (
                float(point[0]), float(point[1]), float(point[2]),
            )
        except (TypeError, ValueError, IndexError):
            return False
        for spec in self.specs():
            ox, oy, oz = spec.origin
            nx, ny, nz = spec.normal
            side = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz
            if side < -_ON_PLANE_TOL:
                return True
        return False

    # -- mutators ------------------------------------------------------

    def add(
        self,
        normal: Sequence[float],
        *,
        offset: float = 0.0,
        name: Optional[str] = None,
    ) -> ClipPlane:
        """Create a plane and return it.

        Created **inactive** when six planes are already cutting — the
        plane still exists, ready to be swapped in for one of the six,
        which is friendlier than refusing to create it at all.
        """
        plane_id = f"plane-{next(self._ids)}"
        active = self.active_count() < MAX_ACTIVE_PLANES
        plane = ClipPlane(
            plane_id=plane_id,
            name=name or f"Plane {len(self._planes) + 1}",
            normal=_unit(normal),
            offset=float(offset),
            active=active,
        )
        self._planes.append(plane)
        if not active:
            self._refuse()
        self._reconcile_and_fire()
        return plane

    def remove(self, plane_id: str) -> bool:
        plane = self.plane(plane_id)
        if plane is None:
            return False
        self._planes.remove(plane)
        self._reconcile_and_fire()
        return True

    def clear(self) -> None:
        """Drop every plane (one event, not one per plane)."""
        if not self._planes:
            return
        self._planes.clear()
        self._reconcile_and_fire()

    def set_active(self, plane_id: str, active: bool) -> bool:
        """Activate / deactivate a plane. ``False`` when refused.

        Refused is exactly one case: a seventh plane trying to cut.
        """
        plane = self.plane(plane_id)
        if plane is None or plane.active == bool(active):
            return False
        if active and self.active_count() >= MAX_ACTIVE_PLANES:
            self._refuse()
            return False
        plane.active = bool(active)
        self._reconcile_and_fire()
        return True

    def set_flipped(self, plane_id: str, flipped: bool) -> None:
        self._mutate(plane_id, "flipped", bool(flipped))

    def set_offset(self, plane_id: str, offset: float) -> None:
        self._mutate(plane_id, "offset", float(offset))

    def set_normal(self, plane_id: str, normal: Sequence[float]) -> None:
        self._mutate(plane_id, "normal", _unit(normal))

    def set_name(self, plane_id: str, name: str) -> None:
        self._mutate(plane_id, "name", str(name))

    def set_gizmo_visible(self, plane_id: str, visible: bool) -> None:
        self._mutate(plane_id, "gizmo_visible", bool(visible))

    def _mutate(self, plane_id: str, field_name: str, value: Any) -> None:
        plane = self.plane(plane_id)
        if plane is None or getattr(plane, field_name) == value:
            return
        setattr(plane, field_name, value)
        self._reconcile_and_fire()

    def _refuse(self) -> None:
        message = (
            f"Section planes: at most {MAX_ACTIVE_PLANES} planes can cut "
            f"at once (the OpenGL clip-distance limit). Deactivate one "
            f"first."
        )
        if self.on_status is not None:
            try:
                self.on_status(message)
            except Exception:
                pass

    # -- reconciliation ------------------------------------------------

    def _reconcile(self) -> None:
        """Push the resolved half-spaces to the render target."""
        backend = self._backend
        if backend is None:
            return
        try:
            backend.set_clip_planes(self.specs())
        except Exception:
            pass

    def _reconcile_and_fire(self) -> None:
        """Reconcile, then announce (ADR 0056 Part 2).

        The owner fires; call sites only call mutators. Reconciliation
        runs first so the dispatcher's coalesced render paints a scene
        that is already cut correctly — and so the RENDER-lane
        subscriber that re-stamps the off-seam actors sees the same
        set the backend just took.
        """
        self._reconcile()
        if self.dispatcher is None:
            return
        try:
            from ..diagrams._dispatch import CLIP_PLANES_CHANGED
            self.dispatcher.fire(CLIP_PLANES_CHANGED)
        except Exception:
            pass

    # -- session persistence (ADR 0083 Part 5) -------------------------

    def snapshot(self) -> "list[dict]":
        """Every plane as a plain dict, for the viewer session."""
        return [
            {
                "plane_id": p.plane_id,
                "name": p.name,
                "normal": tuple(float(c) for c in p.normal),
                "offset": float(p.offset),
                "active": bool(p.active),
                "flipped": bool(p.flipped),
                "gizmo_visible": bool(p.gizmo_visible),
            }
            for p in self._planes
        ]

    def restore(
        self,
        snapshots: Any,
        *,
        apply_cuts: bool = True,
        show_gizmos: bool = True,
    ) -> None:
        """Replace the set from a session (one event for the lot).

        Unlike legends, planes are not owned by anything that has to
        attach first — they exist as soon as the controller does — so
        restore is a straight replacement with no parking. Malformed
        records are skipped rather than aborting the whole restore; the
        active-count cap is re-applied, so a session written before a
        future cap change still loads legally.
        """
        self._planes = []
        for raw in snapshots or ():
            try:
                plane = ClipPlane(
                    plane_id=str(raw["plane_id"]),
                    name=str(raw.get("name", "Plane")),
                    normal=_unit(raw["normal"]),
                    offset=float(raw.get("offset", 0.0)),
                    active=bool(raw.get("active", True)),
                    flipped=bool(raw.get("flipped", False)),
                    gizmo_visible=bool(raw.get("gizmo_visible", True)),
                )
            except Exception:
                continue
            if plane.active and self.active_count() >= MAX_ACTIVE_PLANES:
                plane.active = False
            self._planes.append(plane)
        # Keep the id counter ahead of anything restored so a plane
        # added after a restore cannot collide with a restored id.
        # Ahead of the restored ids' own numbering, not the list
        # length: a session saved after a deletion has sparse ids
        # (plane-1, plane-3), and counting from len+1 would mint
        # plane-3 again — every id-keyed mutator then drives the
        # wrong plane.
        max_n = 0
        for p in self._planes:
            try:
                max_n = max(max_n, int(str(p.plane_id).rsplit("-", 1)[-1]))
            except ValueError:
                pass
        self._ids = itertools.count(max_n + 1)
        self._apply_cuts = bool(apply_cuts)
        self._show_gizmos = bool(show_gizmos)
        self._reconcile_and_fire()


# =====================================================================
# One controller per render target
# =====================================================================

_CONTROLLERS: "WeakKeyDictionary[Any, ClipPlaneSetController]" = (
    WeakKeyDictionary()
)


def controller_for(backend: Any) -> ClipPlaneSetController:
    """The :class:`ClipPlaneSetController` owning ``backend``'s viewport.

    Keyed weakly by backend for the same reason legends are: a backend
    *is* a viewport (the registry wraps the plotter once at bind, ADR
    0042 R-B.final), so this gives every viewport exactly one plane
    owner without threading an injection parameter through the diagram
    lifecycle.
    """
    controller = _CONTROLLERS.get(backend)
    if controller is None:
        controller = ClipPlaneSetController(backend)
        _CONTROLLERS[backend] = controller
    return controller


__all__ = [
    "MAX_ACTIVE_PLANES",
    "ClipPlane",
    "ClipPlaneSetController",
    "controller_for",
]
