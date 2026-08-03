"""ScopeController — a spatial SCOPE BOX as an ElementVisibility layer.

The catalog's second region tool, and the sibling of
:mod:`.threshold_controller`. A scope box is a per-geometry
axis-aligned box; cells outside it are hidden. Like the threshold it is
**not a diagram kind** — it takes a layer of its own,
:data:`~.element_visibility.LAYER_SCOPE`, and rides the existing
:class:`~.element_visibility.ElementVisibility` mechanism instead of
adding a second writer of ``vtkGhostType``:

* the ghost array is mutated in place, so there is no polydata rebuild;
* hidden cells become non-pickable for free (VTK skips HIDDENCELL and
  the box-pick path ANDs the mask);
* it COMPOSES — the effective hidden set is the OR of manual ∪ dim ∪
  stage ∪ threshold ∪ scope, so clearing the scope cannot reveal a cell
  the user hid by hand.

Per geometry, not per viewer, for the same reason the threshold is:
each geometry owns its own :class:`FEMSceneData` (ADR 0058 S1/S2a) and
therefore its own ``ElementVisibility``, so keying this controller's
state by geometry id makes the feature per-geometry by construction.

The box itself is the canonical :class:`~..scene_ir.BBox` — validated,
axis-aligned, world frame (ADR 0045 INV-2: there is exactly ONE
bounding-box value type). Nothing here invents a second one.

Two ratified behaviours
-----------------------
**Cell rule — ANY NODE INSIDE KEEPS THE CELL.** A cell is visible if at
least one of its nodes lies in the box. This deliberately DIVERGES from
the threshold's all-values rule, and the divergence is the point: a
threshold asks *"are these values in range"*, which only has a
conservative answer (one node outside and the cell no longer describes
the band). A scope box asks *"show me this region"*, which only has an
inclusive one. All-nodes would erode the boundary — every element
straddling a box face would vanish, so the visible set would sit a full
element inside the box the user drew — and on a coarse mesh a box
smaller than one element would hide EVERYTHING: the user drags a box
around the region they care about and gets an empty viewport. Do not
"fix" the inconsistency with :mod:`.threshold_controller`; the two
questions are different questions.

**REFERENCE geometry, never the deformed shape.** Membership is
evaluated against ``scene.reference_points + geometry.offset`` — where
the undeformed geometry sits in the world (ADR 0058 S3a keeps the
per-geometry spatial offset as a pump-time term, and world = grid, the
S2c picking invariant). Two consequences, both wanted: which cells are
in scope is FIXED, so nothing pops in or out of the scope while the
user scrubs a seismic run and watches the structure sway; and the mask
is static with respect to time, so there is **zero per-step cost** —
this controller is deliberately absent from the STEP path. It is
refreshed exactly when its inputs change: the box is set or cleared,
the geometry's offset moves it relative to the world-frame box, a scene
is materialized mid-session, or a session is restored.

No VTK, no render backend, no Qt: the writes go through
``ElementVisibility``, the sanctioned ghost writer, and the points come
off the scene. Unlike the threshold there is no injected reader — a
scope needs no ``Results`` handle and no step, which is exactly why it
costs nothing to scrub.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ..scene_ir import BBox
from .element_visibility import LAYER_SCOPE


def scope_points(geometry: Any, scene: Any) -> "np.ndarray":
    """The world points a scope box is tested against.

    ``scene.reference_points + geometry.offset`` — the UNDEFORMED
    geometry where it actually sits (ADR 0058 S3a: the offset is a
    pump-time translation, never baked into ``reference_points``). The
    deform field is deliberately not read; see the module docstring.
    """
    pts = np.asarray(scene.reference_points, dtype=np.float64)
    off = getattr(geometry, "offset", None)
    if off is None:
        return pts
    off = np.asarray(off, dtype=np.float64).reshape(-1)
    if off.shape != (3,) or not bool(np.any(off)):
        return pts
    return pts + off


def points_in_box(box: BBox, points: "np.ndarray") -> "np.ndarray":
    """Boolean "inside ``box``" mask, one entry per point.

    The vectorised twin of :meth:`BBox.contains` — same inclusive
    ``min <= p <= max`` test, run over an ``(n, 3)`` cloud instead of a
    single 3-vector, because these are FE meshes with 100k+ points.
    NaN compares False against both bounds and therefore lands OUTSIDE,
    which is the right reading: a point with no coordinate is not
    evidence that its cell is in the region.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros(0 if pts.ndim == 0 else pts.shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        return np.all((pts >= box.min) & (pts <= box.max), axis=1)


def cells_with_any_node_in_box(
    grid: Any, point_in_box: "np.ndarray",
) -> "np.ndarray":
    """Reduce a per-POINT boolean mask to per-CELL by logical OR.

    The any-node rule, vectorised — the structural mirror of
    :func:`~.threshold_controller.cells_with_all_nodes_in_range` with
    ``logical_and`` swapped for ``logical_or``; that one operator IS the
    ratified divergence between the two features (see the module
    docstring), so it is the thing to look at first if a straddling cell
    ever starts disappearing.

    ``grid.cell_connectivity`` is the flat node-index stream and
    ``grid.offset`` is the ``n_cells + 1`` array of start positions into
    it (VTK 9's split representation, which pyvista surfaces directly).
    Gathering ``point_in_box[connectivity]`` gives one boolean per
    (cell, node) incidence; ``np.logical_or.reduceat`` at the offsets
    ORs each cell's run in one pass. Mixed cell types need no special
    case — a tri contributes a run of 3 and a hex a run of 8, and the
    offsets already say so.

    ``reduceat`` reduces ``a[idx[i]:idx[i+1]]`` for each index and
    ``a[idx[-1]:]`` for the last, which is why the trailing offset is
    dropped: ``offset[:-1]`` has length ``n_cells`` and yields exactly
    ``n_cells`` results. Keeping it would silently merge the last two
    cells — cross-checked against a per-cell Python oracle over
    randomised ragged meshes in ``tests/viewers/test_scope_box.py``.

    Returns a boolean array of length ``n_cells``: True = keep.
    """
    n_cells = int(grid.n_cells)
    if n_cells == 0:
        return np.zeros(0, dtype=bool)
    conn = np.asarray(grid.cell_connectivity)
    offsets = np.asarray(grid.offset)
    if conn.size == 0 or offsets.size != n_cells + 1:
        # Nothing sane to reduce (an empty or non-standard grid) — keep
        # every cell rather than blanking the viewport.
        return np.ones(n_cells, dtype=bool)
    per_incidence = np.asarray(point_in_box, dtype=bool)[conn]
    return np.logical_or.reduceat(per_incidence, offsets[:-1])


def compute_hidden_mask(
    grid: Any, points: "np.ndarray", box: BBox,
) -> "np.ndarray":
    """The per-cell hide mask for ``points`` under ``box``.

    ``True`` = hide. Membership is decided per point by
    :func:`points_in_box` and reduced to cells by
    :func:`cells_with_any_node_in_box`, so a cell survives iff at least
    one of its nodes is inside.
    """
    return ~np.asarray(
        cells_with_any_node_in_box(grid, points_in_box(box, points)),
        dtype=bool,
    )


class ScopeController:
    """Owns the per-geometry scope box and applies it.

    Takes no collaborators: a scope is a function of the geometry's
    reference points, its offset and the box, all of which are already
    to hand at :meth:`refresh` time. That is the whole reason the
    feature is free to scrub — there is no reader to call and no step
    to resolve, so nothing here belongs on the STEP path.
    """

    __slots__ = (
        "_by_geometry", "_applied", "_show_gizmo", "on_show_gizmo_changed",
    )

    def __init__(self) -> None:
        self._by_geometry: dict[str, BBox] = {}
        # Geometry ids whose scene currently CARRIES the layer. Tracked
        # so that turning the last scope off still gets one more refresh
        # pass to take the layer back down — see :meth:`needs_refresh`.
        self._applied: set[str] = set()
        # Whether the in-viewport drag gizmo is drawn (viewer-wide, like
        # the clip gizmos' ``show_gizmos``). It is a VIEW preference and
        # never touches the mask, which is why it is the one piece of
        # state here that :meth:`refresh` ignores.
        self._show_gizmo: bool = True
        #: Called with no arguments after :meth:`set_show_gizmo` really
        #: changes the flag. A plain callback, not a dispatcher handle:
        #: this controller takes no collaborators (that is what keeps
        #: the scope off the STEP path), so the viewer wires the fire at
        #: install, exactly as it does for the drag gizmo's
        #: ``on_changed``.
        self.on_show_gizmo_changed: "Optional[Callable[[], None]]" = None

    # ------------------------------------------------------------------
    # State (the programmatic API — usable from a script, no UI)
    # ------------------------------------------------------------------

    def set_scope(self, geometry_id: str, box: BBox) -> None:
        """Enable (or move) the scope box on one geometry.

        Records intent only — the mask lands on the next
        :meth:`refresh`. ``box`` must be a :class:`BBox`, which has
        already validated ``min <= max`` at construction; an inverted
        box therefore never reaches this controller.
        """
        if not isinstance(box, BBox):
            raise TypeError(
                f"set_scope needs a scene_ir.BBox; got {type(box).__name__}."
            )
        self._by_geometry[str(geometry_id)] = box

    def clear_scope(self, geometry_id: str) -> None:
        """Disable the scope box on one geometry.

        Records intent only; the next :meth:`refresh` takes the layer
        down (and :meth:`needs_refresh` stays True until it has).
        """
        self._by_geometry.pop(str(geometry_id), None)

    def box_for(self, geometry_id: str) -> Optional[BBox]:
        """The geometry's scope box, or ``None`` when disabled."""
        return self._by_geometry.get(str(geometry_id))

    @property
    def show_gizmo(self) -> bool:
        """Whether the in-viewport drag gizmo is drawn."""
        return self._show_gizmo

    def set_show_gizmo(self, enabled: bool) -> None:
        """Show / hide the drag gizmo. Never affects the mask.

        **Announces** through :attr:`on_show_gizmo_changed`, which is
        what makes the programmatic call and the panel checkbox behave
        the same way (review finding F7). Before, a script calling this
        mutated the flag and told nobody: the renderer never reconciled,
        so a "hidden" gizmo stayed drawn AND stayed in the interactor's
        hit-test surface, still claiming presses. Meanwhile the checkbox
        announced through ``GEOMETRY_SCOPE_CHANGED`` — the MASK event —
        so a view-only toggle re-ran the mask over every geometry.

        Idempotent: setting the flag to what it already is announces
        nothing.
        """
        enabled = bool(enabled)
        if enabled == self._show_gizmo:
            return
        self._show_gizmo = enabled
        if self.on_show_gizmo_changed is not None:
            self.on_show_gizmo_changed()

    def needs_refresh(self) -> bool:
        """The caller's zero-cost gate.

        True while any scope is configured OR any layer is still
        applied. The second half matters: after the last
        :meth:`clear_scope` the configured set is empty but a layer is
        still hiding cells, so one more pass has to run to take it
        down.
        """
        return bool(self._by_geometry) or bool(self._applied)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def refresh(self, geometry: Any, scene: Any) -> None:
        """Recompute and apply ``geometry``'s scope layer on ``scene``.

        Called when an INPUT changes — the box, the geometry's offset,
        a scene materializing, a session restoring — and pointedly NOT
        per step: the mask is evaluated against reference geometry, so
        it is invariant under scrubbing (pinned by
        ``test_scope_box.py``).

        A scene with no ``element_visibility`` is a no-op; a geometry
        with no scope whose layer is still on the scene gets it
        cleared, so turning the scope off restores exactly the cells it
        hid and nothing else.
        """
        ev = getattr(scene, "element_visibility", None)
        if ev is None:
            return
        gid = str(getattr(geometry, "id", geometry))
        box = self._by_geometry.get(gid)
        if box is None:
            ev.clear_layer(LAYER_SCOPE)
            self._applied.discard(gid)
            return
        ev.set_layer(
            LAYER_SCOPE, compute_hidden_mask(
                scene.grid, scope_points(geometry, scene), box,
            ),
        )
        self._applied.add(gid)


__all__ = [
    "LAYER_SCOPE",
    "ScopeController",
    "cells_with_any_node_in_box",
    "compute_hidden_mask",
    "points_in_box",
    "scope_points",
]
