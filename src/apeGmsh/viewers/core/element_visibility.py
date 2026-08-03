"""ElementVisibility — per-cell hide via vtkGhostType, in place.

Mutates the substrate grid's ``cell_data["vtkGhostType"]`` array
directly so VTK pickers and renderers skip cells with the HIDDENCELL
bit set. No polydata rebuild — the array is allocated once, then
recomposed with vectorised writes that share the underlying buffer.

The box-pick path (``results_pick._build_box_result`` in MODE_ELEMENT)
ANDs the hidden-mask into its hit set so rubber-band picks don't
return cells the user just hid.

Layered model (ADR 0045 — results visual dim-hide)
--------------------------------------------------
A single substrate actor carries every element, so more than one
feature wants to hide cells at once: the user's manual hide/isolate,
and the ``0/1/2/3/4`` dimension filter. If each wrote the ghost array
directly they would clobber one another (a dim-filter ``show_all``
would reveal cells the user had isolated, and vice-versa).

So ElementVisibility is the SOLE writer of ``vtkGhostType`` and keeps
one boolean mask per named *layer*. The effective hidden set is the
union (logical OR) of all layers, materialised into the HIDDENCELL bit
on every change. ``hide`` / ``show`` / ``show_all`` operate on the
default :data:`LAYER_MANUAL` layer (manual hide, isolate, box-hide,
future diagram hides); the dimension filter owns :data:`LAYER_DIM` via
:meth:`set_layer` / :meth:`clear_layer`, and the scalar threshold filter
owns :data:`LAYER_THRESHOLD` the same way. Because the layers are
independent, ``show_all`` reveals only manually-hidden cells and leaves
the dim filter intact, and clearing the threshold cannot reveal a cell
the user hid by hand or the dim filter dropped.

Separate from :class:`VisibilityManager` (mesh viewer, BRep entities
via ``actor.SetVisibility``). The two coexist:

* VisibilityManager hides whole entities at the actor level (used in
  mesh / model viewers).
* ElementVisibility hides individual FE cells inside a single grid
  (used by the results viewer for interior-selection drilling).

Publishes ``ELEMENT_VISIBILITY_CHANGED`` through the injected
dispatcher on every mutation so RENDER-lane subscribers (e.g., a
"hidden count" status label) get a fresh frame without polling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

# vtkDataSetAttributes::HIDDENCELL — which is **0x20**, not 0x01.
#
# This constant read 0x01 from the day it shipped, with a comment that
# had VTK's enum backwards ("0x01 hide ... 0x20 dup cell"). VTK's
# CellGhostTypes is DUPLICATECELL=0x01, HIDDENPOINT=0x02,
# HIDDENCELL=0x20 — ask it yourself:
# ``vtkDataSetAttributes.HIDDENCELL``. On VTK 9.5.2 the mapper renders
# 0x01-flagged cells normally, so every per-element hide in the results
# viewer was **visually inert**: manual hide / isolate, the 0/1/2/3/4
# dim filter, and stage-activation masks all wrote a bit nothing reads.
# Pixel-probed through the real substrate path: 0x01 changed 0 of 39204
# painted pixels; 0x20 hid exactly the half it was given.
#
# ``backends/pyvista_qt.py`` already carried the correct 0x20 with a
# comment naming this same trap — the two halves of the codebase simply
# disagreed. They now share this constant so they cannot drift again.
#
# One inherited caveat from that backend comment, render-verified there:
# only the *pure* 0x20 byte hides 0-D vertex cells (even 0x21 fails),
# so the recompose below must not leave other bits set alongside it.
HIDDENCELL: int = 0x20

# Layer names. LAYER_MANUAL backs hide/show/show_all (manual hide,
# isolate, box-hide); LAYER_DIM is owned by the 0/1/2/3/4 dim filter;
# LAYER_THRESHOLD is owned by the scalar threshold filter
# (``core/threshold_controller.py``), which recomputes it on every STEP
# so the thresholded region follows the time cursor.
LAYER_MANUAL: str = "manual"
LAYER_DIM: str = "dim"
LAYER_THRESHOLD: str = "threshold"


class ElementVisibility:
    """Per-cell hide controller for the results substrate grid.

    Holds a reference to ``scene.grid``; recomposes the ghost array in
    place from its layer masks. ``dispatcher`` is wired by
    :class:`ResultsViewer`; in headless tests it stays ``None`` and the
    controller still works (no event fires).
    """

    def __init__(self, grid: "pv.UnstructuredGrid") -> None:
        self._grid = grid
        self._n = int(grid.n_cells)
        # name -> boolean mask (length n_cells). Union is the hidden set.
        self._layers: dict[str, np.ndarray] = {}
        self._ensure_ghost_array()
        # Seed the manual layer from any pre-existing HIDDENCELL bits so
        # a later recompose preserves them (a grid may arrive with cells
        # already hidden).
        seed = (np.asarray(self._ghosts()) & HIDDENCELL).astype(bool)
        self._layers[LAYER_MANUAL] = seed.copy()
        # Set by ResultsViewer for ELEMENT_VISIBILITY_CHANGED dispatch.
        self.dispatcher: Any = None

    # ------------------------------------------------------------------
    # Manual-layer mutations (hide / show / show_all)
    # ------------------------------------------------------------------

    def hide(self, cell_ids: "np.ndarray | Iterable[int]") -> None:
        """Mark ``cell_ids`` hidden in the manual layer. Idempotent."""
        ids = self._to_ids(cell_ids)
        if ids.size == 0:
            return
        self._layer(LAYER_MANUAL)[ids] = True
        self._recompose()
        self._fire_changed(ids)

    def show(self, cell_ids: "np.ndarray | Iterable[int]") -> None:
        """Unhide ``cell_ids`` in the manual layer. Idempotent.

        Does not affect other layers — a cell hidden by the dim filter
        stays hidden even after a manual ``show``."""
        ids = self._to_ids(cell_ids)
        if ids.size == 0:
            return
        self._layer(LAYER_MANUAL)[ids] = False
        self._recompose()
        self._fire_changed(ids)

    def show_all(self) -> None:
        """Clear the manual layer (reveal manually-hidden cells).

        Leaves other layers (e.g. the dim filter) untouched — "reveal
        all" means undo manual hides, not override the active filter."""
        self._layer(LAYER_MANUAL).fill(False)
        self._recompose()
        self._fire_changed(None)

    # ------------------------------------------------------------------
    # Named-layer API (dim filter and any future independent hide source)
    # ------------------------------------------------------------------

    def set_layer(
        self, name: str, hidden_mask: "np.ndarray | Iterable[bool]"
    ) -> None:
        """Replace layer ``name`` with ``hidden_mask`` (length n_cells).

        ``True`` = hide. Recomposes so the new layer ORs with the rest;
        other layers are preserved."""
        mask = np.asarray(hidden_mask, dtype=bool)
        if mask.shape != (self._n,):
            raise ValueError(
                f"layer mask must have length {self._n}, got {mask.shape}"
            )
        self._layers[name] = mask.copy()
        self._recompose()
        self._fire_changed(None)

    def clear_layer(self, name: str) -> None:
        """Drop layer ``name`` (no-op if absent) and recompose."""
        if self._layers.pop(name, None) is None:
            return
        self._recompose()
        self._fire_changed(None)

    # ------------------------------------------------------------------
    # Queries (read the materialised composite on the ghost array)
    # ------------------------------------------------------------------

    def hidden_mask(self) -> np.ndarray:
        """Boolean mask of length ``n_cells``; True where HIDDENCELL is set."""
        ghosts = self._ghosts()
        return (np.asarray(ghosts) & HIDDENCELL).astype(bool)

    def n_hidden(self) -> int:
        return int(self.hidden_mask().sum())

    def is_hidden(self, cell_id: int) -> bool:
        ghosts = self._ghosts()
        try:
            return bool(int(ghosts[int(cell_id)]) & HIDDENCELL)
        except (IndexError, KeyError):
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _layer(self, name: str) -> np.ndarray:
        """Lazily create an all-visible layer mask for ``name``."""
        m = self._layers.get(name)
        if m is None:
            m = np.zeros(self._n, dtype=bool)
            self._layers[name] = m
        return m

    def _recompose(self) -> None:
        """OR every layer into the HIDDENCELL bit, in place.

        Preserves non-HIDDENCELL ghost bits; never rebinds the array so
        VTK's backing buffer (and any filter that shares it) stays put."""
        ghosts = self._ghosts()
        union = np.zeros(self._n, dtype=bool)
        for mask in self._layers.values():
            union |= mask
        keep = np.uint8(~HIDDENCELL & 0xFF)
        ghosts[:] = (np.asarray(ghosts) & keep) | (
            union.astype(np.uint8) * np.uint8(HIDDENCELL)
        )
        self._grid.Modified()

    def _ensure_ghost_array(self) -> None:
        """Lazy-allocate ``vtkGhostType`` (unsigned-char, all zero) if
        the substrate grid doesn't carry one yet. PyVista may report
        ``not in cell_data`` even with the key present in older
        versions — fall back to a try/except read for robustness."""
        try:
            _ = self._grid.cell_data["vtkGhostType"]
            return
        except (KeyError, IndexError):
            pass
        self._grid.cell_data["vtkGhostType"] = np.zeros(
            self._n, dtype=np.uint8,
        )

    def _ghosts(self) -> np.ndarray:
        return self._grid.cell_data["vtkGhostType"]

    @staticmethod
    def _to_ids(cell_ids: "np.ndarray | Iterable[int]") -> np.ndarray:
        if isinstance(cell_ids, np.ndarray):
            return cell_ids.astype(np.int64, copy=False)
        return np.fromiter(
            (int(c) for c in cell_ids), dtype=np.int64,
        )

    def _fire_changed(self, payload: Optional[np.ndarray]) -> None:
        if self.dispatcher is None:
            return
        try:
            from ..diagrams._dispatch import ELEMENT_VISIBILITY_CHANGED
            self.dispatcher.fire(
                ELEMENT_VISIBILITY_CHANGED, payload=payload,
            )
        except Exception:
            pass


def apply_dim_filter(
    ev: "ElementVisibility",
    cell_dim: "np.ndarray",
    active: "Iterable[int]",
    all_dims: "Iterable[int]",
    *,
    layer: str = LAYER_DIM,
) -> None:
    """Ghost-hide cells whose dimension is not in ``active``, via ``layer``.

    The results-viewer ``0/1/2/3/4`` filter callback's pure core: when
    every dim is active the layer is cleared (nothing dim-hidden);
    otherwise cells of inactive dims are hidden. Composes with the
    manual layer because it only touches ``layer`` (default
    :data:`LAYER_DIM`). No render — the caller renders."""
    active_set = {int(d) for d in active}
    if active_set == {int(d) for d in all_dims}:
        ev.clear_layer(layer)
        return
    hide = ~np.isin(np.asarray(cell_dim), list(active_set))
    ev.set_layer(layer, hide)


__all__ = [
    "ElementVisibility",
    "HIDDENCELL",
    "LAYER_MANUAL",
    "LAYER_DIM",
    "LAYER_THRESHOLD",
    "apply_dim_filter",
]
