"""ElementVisibility — per-cell hide via vtkGhostType, in place.

Mutates the substrate grid's ``cell_data["vtkGhostType"]`` array
directly so VTK pickers and renderers skip cells with the HIDDENCELL
bit set. No polydata rebuild — the array is allocated once, then
masked with ``|=`` / ``&=`` operations that share the underlying
buffer.

The box-pick path (``results_pick._build_box_result`` in MODE_ELEMENT)
ANDs the hidden-mask into its hit set so rubber-band picks don't
return cells the user just hid.

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

# vtkDataSetAttributes::HIDDENCELL. The full vtkGhostType byte is
# bitmask-style: 0x01 hide, 0x02 refined, 0x04 boundary, 0x08 exterior,
# 0x10 dup point, 0x20 dup cell. We only touch 0x01 here.
HIDDENCELL: int = 0x01


class ElementVisibility:
    """Per-cell hide controller for the results substrate grid.

    Holds a reference to ``scene.grid``; mutates the ghost array in
    place. ``dispatcher`` is wired by :class:`ResultsViewer`; in
    headless tests it stays ``None`` and the controller still works
    (no event fires).
    """

    def __init__(self, grid: "pv.UnstructuredGrid") -> None:
        self._grid = grid
        self._ensure_ghost_array()
        # Set by ResultsViewer for ELEMENT_VISIBILITY_CHANGED dispatch.
        self.dispatcher: Any = None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def hide(self, cell_ids: "np.ndarray | Iterable[int]") -> None:
        """Mark ``cell_ids`` as hidden. Idempotent (no-op on already-hidden)."""
        ids = self._to_ids(cell_ids)
        if ids.size == 0:
            return
        ghosts = self._ghosts()
        ghosts[ids] |= HIDDENCELL
        self._grid.Modified()
        self._fire_changed(ids)

    def show(self, cell_ids: "np.ndarray | Iterable[int]") -> None:
        """Unhide ``cell_ids``. Idempotent."""
        ids = self._to_ids(cell_ids)
        if ids.size == 0:
            return
        ghosts = self._ghosts()
        ghosts[ids] &= np.uint8(~HIDDENCELL & 0xFF)
        self._grid.Modified()
        self._fire_changed(ids)

    def show_all(self) -> None:
        """Clear the hidden bit on every cell."""
        ghosts = self._ghosts()
        ghosts &= np.uint8(~HIDDENCELL & 0xFF)
        self._grid.Modified()
        self._fire_changed(None)

    # ------------------------------------------------------------------
    # Queries
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
        n_cells = int(self._grid.n_cells)
        self._grid.cell_data["vtkGhostType"] = np.zeros(
            n_cells, dtype=np.uint8,
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


__all__ = ["ElementVisibility", "HIDDENCELL"]
