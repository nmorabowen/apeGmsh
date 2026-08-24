"""``ViewClipController`` — the ADR 0083 clip-controller contract, served
off a session ``MeshView`` (ADR 0098 R1, slice 1).

The gizmo renderer and its interactor were written against
:class:`~...viewers.core._clip_planes.ClipPlaneSetController`, which
OWNS its planes. Under ADR 0098 §3 the planes moved viewer → view: the
session's :class:`~....results.session._views.ViewClip` copies
``ClipPlane``'s field shape verbatim, and ``MeshView.set_clip`` is the
writer. So the session needs no second copy of the state and no second
owner — it needs an **adapter**, which is all this is.

The contract is exactly five members, extracted from both consumers
rather than from the old controller's much wider surface:

============================  =========================================
``planes()``                  ``_clip_gizmo.ClipGizmoRenderer.refresh``
``show_gizmos``               (same)
``plane(plane_id)``           ``_clip_gizmo_interactor`` press
``set_offset(id, offset)``    translate drag
``set_pose(id, normal, off)`` rotate drag
============================  =========================================

``ViewClip`` already exposes every field the renderer reads off a plane
(``plane_id`` / ``normal`` / ``offset`` / ``flipped`` / ``gizmo_visible``),
so :meth:`planes` hands the records over untouched. Nothing is copied,
adapted or shadowed — which is the point: a second plane record would
be a second thing to keep in sync, and ADR 0083's whole discipline is
that the renderer is a *projection* of the owner's state.

Three semantic differences from the old controller are reconciled here,
and each one is load-bearing:

* **An unknown ``plane_id`` is silent.** ``ClipPlaneSetController``'s
  mutators return quietly when the plane is gone; ``MeshView.set_clip``
  raises ``KeyError``, deliberately, because a *scripted* edit naming a
  plane that does not exist is a bug the user needs told about. A drag
  is not a scripted edit: these mutators run inside a VTK observer
  callback, and the plane can vanish under an in-flight gesture (an
  outline delete, a snapshot restore). Raising there would escape into
  the interactor's event handler. So the gesture path swallows exactly
  ``KeyError`` and nothing else.
* **``set_pose`` stays ONE mutation.** The old controller writes
  ``normal`` and ``offset`` together on purpose — two mutators would
  fire two events per mouse-move and paint a transient plane (new
  normal, stale offset) swinging through the model between them. One
  ``set_clip`` call with both fields is one frozen-record swap and one
  change tick, so the property survives the port for free.
* **``show_gizmos`` is DERIVED, not stored.** See :meth:`show_gizmos`.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


class ViewClipController:
    """The ADR 0083 clip-controller contract over one ``MeshView``.

    Holds no plane state: every read goes to ``view.clips`` and every
    write goes through ``view.set_clip``, so a gizmo drag is an
    ordinary session edit and lands in the snapshot like any other.
    """

    def __init__(self, view: Any) -> None:
        self._view = view

    @property
    def view(self) -> Any:
        """The view this controller serves (the pane, 1:1 — ADR 0098 §3)."""
        return self._view

    # -- reads ---------------------------------------------------------

    def planes(self) -> "list[Any]":
        """Every plane of this view, in creation order.

        The ``ViewClip`` records themselves, not copies: they are frozen,
        so handing them out cannot let a projection mutate the view.
        """
        return list(self._view.clips)

    def plane(self, plane_id: str) -> Optional[Any]:
        """One plane by id, or ``None`` — the old controller's shape.

        ``None`` rather than ``KeyError`` because the interactor's press
        handler tests it (``if plane is None: return``) instead of
        catching.
        """
        for clip in self._view.clips:
            if clip.plane_id == plane_id:
                return clip
        return None

    @property
    def show_gizmos(self) -> bool:
        """Whether any gizmo is drawn at all — DERIVED, per A5.3's rule
        that placement is state and *existence* stays derived.

        ``ClipPlaneSetController`` carries a viewer-wide ``_show_gizmos``
        flag beside the per-plane ``gizmo_visible``. The session record
        has only the per-plane flag, and adding a second, coarser one
        would make two fields answer the same question — the state
        duplication ADR 0098 §3 moved the planes to avoid.

        Deriving it is not a shortcut, it is exact. The renderer draws
        plane ``P`` iff ``show_gizmos and P.gizmo_visible``; with
        ``show_gizmos = any(P.gizmo_visible)`` that reduces to
        ``P.gizmo_visible`` for every ``P`` — the identical picture the
        stored flag produces, for every combination of flags. The master
        toggle a UI offers is therefore "clear every plane's flag",
        which is a real gesture on state that already round-trips
        through the snapshot, rather than a new field that would need
        its own persistence and its own restore rule.
        """
        return any(clip.gizmo_visible for clip in self._view.clips)

    # -- gesture mutators ----------------------------------------------

    def set_offset(self, plane_id: str, offset: float) -> None:
        """Slide the plane along its normal (the translate drag)."""
        self._set(plane_id, offset=float(offset))

    def set_pose(
        self, plane_id: str, normal: Sequence[float], offset: float,
    ) -> None:
        """Rotate and re-anchor in ONE mutation (the rotate drag).

        Both fields in a single ``set_clip`` for the reason ADR 0083 S2
        gives: split across two writes, the intermediate record is a
        plane with the new normal and the old offset, which sweeps
        across the model for one frame.
        """
        self._set(plane_id, normal=tuple(normal), offset=float(offset))

    def _set(self, plane_id: str, **changes: Any) -> None:
        """One ``set_clip``, with a vanished plane treated as a no-op.

        Scoped to ``KeyError`` on purpose: that is the one failure a
        live gesture can legitimately race into. Anything else —
        a bad normal, a frozen-record violation — is a real bug and
        must not be swallowed inside a VTK callback where it would
        vanish without a traceback.
        """
        try:
            self._view.set_clip(plane_id, **changes)
        except KeyError:
            pass


__all__ = ["ViewClipController"]
