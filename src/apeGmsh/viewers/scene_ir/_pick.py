"""``PickBackend`` Protocol + pick IR — the viewer-side *pick* contract.

Defined by
[ADR 0044](../../opensees/architecture/decisions/0044-pick-backend-and-export.md)
(Phase R-D), the pick-side sibling of the render seam (ADR 0042
:mod:`RenderBackend <apeGmsh.viewers.scene_ir._backend>`).  The backend
owns *all* VTK ray-casting and screen↔world geometry; the domain layer
(``viewers/core/``, the two viewers) interprets the geometric hit into a
substrate-native entity.  The clean line (ADR 0044 INV-3):

* the backend resolves **screen → scene geometry** — a ``vtkCellPicker``
  hit (which prop, which cell, the world point) and the camera
  projection — and nothing else;
* mode routing, ``EntityRegistry`` / ``cell_to_element_id`` resolution,
  box-candidate sourcing, highlight overlays, ``Alt``-pick-through, and
  hover *interpretation* stay in the domain layer.

INV-2 (ADR 0044, mirroring ADR 0042 INV-1): this module imports neither
``vtk`` nor ``pyvista``.  Screen coordinates in, an opaque prop id + cell
id + world point out.  Enforced by ``tests/test_scene_ir_pure.py``.

Why ``prop_id`` and not the VTK actor (the keystone refinement over the
ADR's draft body): both domain registries already key on ``id(actor)`` —
``EntityRegistry._actor_id_to_dim`` (mesh, ``(dim, tag)``) and the
results pick inventory (``id(vtkProp) → (kind, …)``).  Returning that
plain integer lets a :class:`PickHit` cross the VTK-free seam while both
domain consumers map it back with their existing dict lookup — no VTK
object ever crosses (INV-2), and the two viewers keep their distinct
entity vocabularies (INV-3, ADR 0045 ``Substrate``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


class PickMode(str, Enum):
    """What a click is *for* — domain-level routing, not a backend concern.

    The backend resolves a screen hit to a prop + cell + world point; the
    *mode* decides how the domain layer interprets it (snap-to-node,
    resolve to element id, route to a gauss overlay).  Subclassing ``str``
    keeps it JSON/keyboard-shortcut friendly and lets the results
    controller's existing ``"node"`` / ``"element"`` / ``"gp"`` strings
    compare equal to these members during the R-D.2 migration.

    ``FIBER`` is **reserved**: fiber selection is driven from the 2-D
    section side panel, not a viewport click (see
    ``results_pick_engine`` module docstring), so no click-pick path
    resolves to it yet.  It is named here so the vocabulary is complete
    when a viewport fiber pick is added.
    """

    NODE = "node"
    ELEMENT = "element"
    GP = "gp"
    FIBER = "fiber"  # reserved — side-panel-driven, no click path yet


@dataclass(frozen=True)
class PickModifiers:
    """Keyboard modifiers live at pick time — a backend-neutral envelope.

    Carries the modifier state the desktop event face reads off the live
    VTK event (``GetControlKey`` / ``GetAltKey``) so the *domain*
    interprets them (``ctrl`` → add/remove toggle in the selection set;
    ``alt`` → pick-through).  ``shift`` is **not** here: Shift+LMB is
    owned by navigation (priority 11) and never reaches the pick path.
    """

    ctrl: bool = False
    alt: bool = False


@dataclass(frozen=True)
class PickRequest:
    """A single click-pick query in display-pixel space — the stateless
    core's only input.

    ``(x, y)`` is a display-pixel coordinate (VTK event-position
    convention: origin bottom-left).  ``mode`` and ``modifiers`` travel so
    the backend can run the correct picker, but the *interpretation* of
    the result stays in the domain layer (INV-3).

    Box picking is **not** a ``PickRequest``: its candidate sourcing is
    substrate-specific (mesh vertices vs. GP centers vs. node coords), so
    only the shared *projection* primitive (:meth:`PickBackend.project_points`
    / :meth:`PickBackend.frustum_planes`) is on the seam; the domain owns
    the in-box test (ADR 0044 §Rejected C caveat).
    """

    x: int
    y: int
    mode: PickMode = PickMode.NODE
    modifiers: PickModifiers = PickModifiers()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PickMode):
            # Accept the bare strings the results controller uses today
            # (one-release migration shim, ADR 0044 Open-Q 2 resolution).
            object.__setattr__(self, "mode", PickMode(self.mode))
        if not isinstance(self.modifiers, PickModifiers):
            raise TypeError(
                "PickRequest.modifiers must be a PickModifiers; got "
                f"{type(self.modifiers).__name__}."
            )
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))


@dataclass(frozen=True)
class PickHit:
    """A resolved screen → scene hit, VTK-free — the geometric click result.

    The backend fills the geometry it can resolve from the render:

    ``prop_id``
        ``id()`` of the picked VTK prop (the actor's Python identity).
        The domain registry key — ``EntityRegistry`` (mesh) /
        the results pick inventory both look entities up by exactly this
        integer.  ``None`` on a miss.
    ``cell_id``
        The picked cell index within that prop's mapped dataset, or
        ``None`` on a miss.
    ``world``
        The world-space point under the cursor (``GetPickPosition``), or
        ``None`` on a miss.

    A *miss* (the ray hit no pickable prop) is reported as ``None`` from
    :meth:`PickBackend.resolve_pick`, not a ``PickHit`` with empty fields —
    the consumer never has to distinguish the two.  A ``PickHit`` always
    carries at least ``world``.

    The IR widens **additively** (ADR 0044 INV-7): a future hover-detail
    or multi-hit ray channel is a new field / type, never a VTK object
    smuggled back across the seam.
    """

    world: tuple[float, float, float]
    cell_id: Optional[int] = None
    prop_id: Optional[int] = None


@dataclass(frozen=True)
class BoxGesture:
    """A completed rubber-band drag, *before* entity resolution.

    The backend draws the rubber-band overlay and detects the drag; on
    release it hands the domain this geometric envelope and the domain
    resolves it against its own candidates via
    :meth:`PickBackend.project_points` / :meth:`PickBackend.frustum_planes`
    (INV-3).  ``crossing`` mirrors the established convention: a
    right→left drag (``x1 < x0``) selects any entity *touching* the box;
    left→right selects only entities fully *inside* it.

    No ``mode`` field: the gesture machine is mode-agnostic geometry, and
    the domain already knows its own mode when the callback fires.
    """

    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) display pixels
    crossing: bool
    modifiers: PickModifiers = PickModifiers()


# Desktop event-face callback shapes.  The backend installs interactor
# observers, runs the press/move/release state machine + rubber-band
# overlay, pre-resolves the geometric hit, and fires these; the domain
# does the entity *interpretation* (mode routing, FEM-id resolution).
# ``on_pick`` / ``on_hover`` carry ``None`` on a miss (the ray hit no
# pickable prop) so the domain can clear / ignore as it sees fit.
OnPick = Callable[[Optional[PickHit], PickModifiers], None]
OnHover = Callable[[Optional[PickHit]], None]
OnBox = Callable[[BoxGesture], None]


@runtime_checkable
class PickBackend(Protocol):
    """Optional capability: resolve a screen pick to a scene hit.

    A :class:`RenderBackend <apeGmsh.viewers.scene_ir._backend.RenderBackend>`
    exposes a ``PickBackend`` only when ``supports_picking() is True``
    (ADR 0042 Part 2 / ADR 0044 INV-1).  View-only backends (trame
    client-only, ``ParaViewExportBackend``) expose none and are legal.

    Structural, not nominal — like ``RenderBackend`` / ``H5ModelReader``,
    implementers do **not** subclass; conformance is by
    :class:`typing.Protocol`.  ``runtime_checkable`` so a consumer can
    ``isinstance``-probe, but that verifies method *presence* only.

    The Protocol carries two faces (ADR 0044 Open-Q 1 resolution — keep
    both):

    * **stateless core** — :meth:`resolve_pick` (+ the projection
      primitives) is pure screen↔scene geometry the web request/response
      face will reuse verbatim;
    * **desktop event face** — :meth:`install` / :meth:`uninstall` layer
      the interactor-observer gesture machine over that core, preserving
      the priority-10/11 abort chain shared with navigation.
    """

    # -- stateless geometric core (web reuses this verbatim in R-D.3) --

    def resolve_pick(self, request: "PickRequest") -> "Optional[PickHit]":
        """Ray-cast one click; return the geometric hit, or ``None`` on a
        miss.

        Pure screen → scene geometry: ``vtkCellPicker.Pick`` → prop id +
        cell id + world point.  No mode routing, no FEM-id resolution, no
        highlight — those are domain logic (INV-3)."""
        ...

    def project_points(self, world: "np.ndarray") -> "np.ndarray":
        """Project ``(N, 3)`` world points to ``(N, 2)`` display pixels.

        The shared box-pick core (today the ``_project_points_to_display``
        helper both engines already call).  The domain feeds its own
        candidate points and runs the in-box test itself (INV-3)."""
        ...

    def frustum_planes(
        self, box: "tuple[int, int, int, int]"
    ) -> "Optional[np.ndarray]":
        """Un-project a display box to its 6 world-space frustum planes.

        ``(6, 4)`` ``[nx, ny, nz, d]`` rows for an exact 3-D box test, or
        ``None`` when the camera cannot un-project (the domain then falls
        back to the 2-D :meth:`project_points` test).  Optional capability:
        a backend may always return ``None``."""
        ...

    # -- desktop event face (layered over the stateless core) --

    def install(
        self,
        *,
        on_pick: "OnPick",
        on_hover: "Optional[OnHover]" = None,
        on_box: "Optional[OnBox]" = None,
    ) -> None:
        """Install the event-driven desktop face (interactor observers).

        The backend runs the click/hover/drag state machine + rubber-band
        overlay and fires the callbacks with the *geometric* result.  A
        request/response backend (web) may leave this a no-op and be
        driven purely through :meth:`resolve_pick`."""
        ...

    def uninstall(self) -> None:
        """Remove any installed observers and overlay actors. Idempotent.

        Closes the observer leak both legacy engines carry (neither had a
        teardown path)."""
        ...


__all__ = [
    "PickMode",
    "PickModifiers",
    "PickRequest",
    "PickHit",
    "BoxGesture",
    "OnPick",
    "OnHover",
    "OnBox",
    "PickBackend",
]
