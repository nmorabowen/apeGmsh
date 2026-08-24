"""PanePick — one pane's clicks and windows, into the ONE selection.

ADR 0098 §8 (S4-1). Results selection is **nodes or Gauss points**,
never elements, never faces, never BRep: an element is a membership
query ("all Gauss of this hex"), not a hit. This module is therefore
the slim rewrite of ``core/results_pick.py`` rather than an extension
of it — that controller carries a third mode, a dim-pick gate, a
geometry resolver and a GP-candidate callback, all of which exist for
the old window's element vocabulary. The old file stays untouched
until S6a.

It owns **no VTK**. The ``vtkCellPicker`` ray-cast, the
press/move/release gesture machine, the rubber-band overlay and the
screen↔world projection all live in
:class:`~apeGmsh.viewers.backends._pyvista_pick.PyVistaPickBackend`,
reused verbatim (ADR 0047 INV-3 — the backend resolves geometry, the
domain interprets it). What is new here is only the interpretation:

* **The targets come from realize.** ``RealizedPane.targets`` carries
  the node ids / ``(element_id, gp_index)`` pairs this pane last put
  on screen, with their posed coordinates. A pick can therefore only
  ever resolve to a point the pane is drawing — right scope, right
  pose, right instant — and "window … scoped to this view's visible
  cells" needs no second derivation.
* **The radio only aims.** ``MeshView.pick_target`` decides which
  family THIS pane's clicks and windows hit. It neither owns nor
  clears the set, and two panes may aim differently over the one set.
* **The style button is the gate.** "Click-pick requires the matching
  style button on" — with the button off the family has no targets,
  and the pick is not a miss but a non-event: it leaves the set alone
  rather than clearing it.
* **Last writer replaces.** Every write goes through
  ``session.selection`` (the ADR 0045 store, INV-5), so a Gauss write
  over a node set replaces it and the log records one gesture.

Modifiers: plain click / drag REPLACES the set, Ctrl extends it
(Ctrl+click toggles the one target). A plain click that hits nothing
clears — the conventional reading of "click empty space to deselect";
a Ctrl+click on nothing does nothing, because Ctrl means "keep what I
have".

**A hit the cut has hidden is a miss** (ADR 0083 Part 5, and now on
this path too). ``vtkCellPicker`` is a geometric ray-cast that knows
nothing about mapper clip planes, so without this a click on
apparently-empty space returns the node the section plane is hiding.
The rule applies to the window as well: a target the cut has hidden is
not IN the rubber band, which matters more than it sounds — a band
dragged over a cut model would otherwise sweep up everything behind
the cut, invisibly, in one gesture.

Both read :func:`~._realize.resolved_clip_specs`, the same function
that answers the backend, so the pick and the picture cannot disagree
about which planes are cutting — cap included.

Resolution rule for a click: the backend's ray answers with the world
point under the cursor, and the nearest target in 3-D to that point
wins. Snapping from the RAY (not from the pixel) is what makes
occlusion work — the ray stops at the visible surface, so the nearest
target to it is a front one, never a node hidden behind the model.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ..core._clip_planes import _ON_PLANE_TOL
from ._realize import resolved_clip_specs


class PanePick:
    """Click + window picking for one mesh pane.

    ``realized_fn`` returns the pane's current
    :class:`~._realize.RealizedPane` (or ``None`` before the first
    flush) — one source for both the view identity and the targets, so
    the pick and the picture can never disagree about which pane, which
    scope or which pose is on screen.
    """

    def __init__(
        self,
        session: Any,
        pick_backend: Any,
        realized_fn: Callable[[], Optional[Any]],
    ) -> None:
        self._session = session
        self._backend = pick_backend
        self._realized_fn = realized_fn
        self._installed = False
        pick_backend.install(on_pick=self._on_pick, on_box=self._on_box)
        self._installed = True

    # -- surface -------------------------------------------------------

    @property
    def installed(self) -> bool:
        """Whether the gesture machine is live on the pane's
        interactor. ``False`` after :meth:`dispose`."""
        return self._installed

    @property
    def pick_backend(self) -> Any:
        return self._backend

    def dispose(self) -> None:
        """Remove the interactor observers and the rubber-band actor.

        Idempotent, and NOT optional: the pane owns this installation
        the way it owns its GL context, so it dies on the same path
        (Amendment 1 caution 1). Observers left on a dead interactor
        are the leak both legacy engines carried.
        """
        if not self._installed:
            return
        self._installed = False
        try:
            self._backend.uninstall()
        except Exception:
            pass

    # -- internals -----------------------------------------------------

    def _aim(self) -> "tuple[Optional[Any], Optional[Any]]":
        """``(view, targets)`` this pane's next gesture acts on.

        ``targets`` is the family the view's radio aims at, or ``None``
        when the matching style button is off (§8's gate) — the pane
        has nothing of that kind on screen to hit.
        """
        realized = self._realized_fn()
        if realized is None:
            return None, None
        try:
            view = self._session.pane(realized.pane_id)
        except KeyError:
            return None, None
        # ``PaneTargets`` names its families after the PICK_TARGETS
        # tokens, so the radio addresses one directly. A token with no
        # family would read as "nothing to hit" — which is why the two
        # sets are pinned equal by test_pane_selection.
        family = getattr(realized.targets, view.pick_target, None)
        return view, family

    def _on_pick(self, hit: Any, mods: Any) -> None:
        view, targets = self._aim()
        if view is None or targets is None:
            return
        selection = self._session.selection
        if hit is not None and _is_clipped(view, hit.world):
            # ADR 0083 Part 5 — the cut has hidden this point, so the
            # user clicked on nothing they can see. Read as a miss, in
            # both senses: plain clears, Ctrl keeps. Checked BEFORE any
            # target resolution, because the point is behind a plane
            # regardless of what it would have resolved to.
            hit = None
        if hit is None:
            # Clicked past the model. Plain: deselect. Ctrl: the user
            # asked to keep what they have.
            if not getattr(mods, "ctrl", False):
                selection.clear()
            return
        row = _nearest(targets.coords, hit.world)
        if row is None:
            return
        if view.pick_target == "nodes":
            node_id = int(targets.ids[row])
            if getattr(mods, "ctrl", False):
                selection.toggle_node(node_id)
            else:
                selection.set_nodes([node_id])
            return
        pair = (int(targets.element_ids[row]), int(targets.gp_indices[row]))
        if getattr(mods, "ctrl", False):
            selection.toggle_gauss(*pair)
        else:
            selection.set_gauss([pair])

    def _on_box(self, gesture: Any) -> None:
        view, targets = self._aim()
        if view is None or targets is None:
            return
        x0, y0, x1, y1 = gesture.box
        if x0 == x1 or y0 == y1:
            return  # Degenerate rectangle — nothing to pick.
        try:
            display = self._backend.project_points(targets.coords)
        except Exception:
            return
        inside = _inside_box(np.asarray(display), x0, y0, x1, y1)
        # Same law, applied to the set rather than the ray: a hidden
        # target is not in the window. Without it one drag over a cut
        # model selects the whole interior the user cannot see.
        inside &= _visible_mask(view, targets.coords)
        selection = self._session.selection
        mods = getattr(gesture, "modifiers", None)
        ctrl = bool(mods is not None and mods.ctrl)
        if view.pick_target == "nodes":
            ids = [int(i) for i in np.asarray(targets.ids)[inside]]
            if ctrl:
                selection.add_nodes(ids)
            else:
                selection.set_nodes(ids)
            return
        pairs = [
            (int(e), int(g))
            for e, g in zip(
                np.asarray(targets.element_ids)[inside],
                np.asarray(targets.gp_indices)[inside],
            )
        ]
        if ctrl:
            selection.add_gauss(pairs)
        else:
            selection.set_gauss(pairs)


def _visible_mask(view: Any, coords: Any) -> "np.ndarray":
    """Boolean mask: which of ``coords`` the cut leaves visible.

    Vectorised because it runs over every target of a window gesture —
    a 100k-node pane is an ordinary rubber band.

    The tolerance is imported rather than restated for the same reason
    the plane cap is: a point exactly ON the cut face must not be
    rejected by floating-point noise, and the two paths must agree on
    where "on" ends. Two copies of a tolerance drift.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(pts.shape[0], dtype=bool)
    for spec in _specs(view):
        origin = np.asarray(spec.origin, dtype=np.float64)
        normal = np.asarray(spec.normal, dtype=np.float64)
        keep &= ((pts - origin) @ normal) >= -_ON_PLANE_TOL
    return keep


def _is_clipped(view: Any, world: Any) -> bool:
    """Whether ``world`` sits on the discarded side of any live plane."""
    try:
        pt = np.asarray(world, dtype=np.float64).reshape(1, 3)
    except (TypeError, ValueError):
        return False
    specs = _specs(view)
    if not specs:
        return False
    return not bool(_visible_mask(view, pt)[0])


def _specs(view: Any) -> tuple:
    """This view's live half-spaces, or none if they cannot be read.

    A pane that cannot answer must not start rejecting picks — the
    failure mode of this filter is silently swallowing clicks, which is
    worse than the defect it fixes.
    """
    try:
        return resolved_clip_specs(view)
    except Exception:
        return ()


def _nearest(coords: "np.ndarray", world: Any) -> Optional[int]:
    """Row of ``coords`` closest in 3-D to the ray's world point.

    No distance tolerance: the ray already landed on the picture, so
    the question is which target it landed nearest to, not whether one
    is near enough. A tolerance here would make a click on a coarse
    element's face silently do nothing.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.size == 0:
        return None
    delta = pts - np.asarray(world, dtype=np.float64).reshape(1, 3)
    return int(np.argmin(np.einsum("ij,ij->i", delta, delta)))


def _inside_box(
    xy: "np.ndarray", x0: float, y0: float, x1: float, y1: float,
) -> "np.ndarray":
    """Boolean mask for points whose display coords fall in the box.

    ``crossing`` (the right→left drag) is not consulted: crossing vs
    inside distinguishes entities a box TOUCHES from entities it
    CONTAINS, and a point is either in the rectangle or out of it.
    """
    bx0, bx1 = (x0, x1) if x0 <= x1 else (x1, x0)
    by0, by1 = (y0, y1) if y0 <= y1 else (y1, y0)
    return (
        (xy[:, 0] >= bx0) & (xy[:, 0] <= bx1)
        & (xy[:, 1] >= by0) & (xy[:, 1] <= by1)
    )


__all__ = ["PanePick"]
