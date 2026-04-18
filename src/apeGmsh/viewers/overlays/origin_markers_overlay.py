"""
Origin-marker overlay runtime manager.

Wraps :func:`build_origin_markers` so the live viewer can add/remove
individual markers and toggle visibility without rebuilding the whole
scene. Owns the glyph + label actors and rebuilds them on any mutation.

Markers are purely visual artifacts — they never touch geometry, mesh,
or the entity registry.

Usage::

    overlay = OriginMarkerOverlay(
        plotter,
        origin_shift=registry.origin_shift,
        model_diagonal=diag,
        points=[(0.0, 0.0, 0.0)],
    )
    overlay.add((10.0, 0.0, 0.0))
    overlay.set_show_coords(False)
    overlay.set_visible(False)
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class OriginMarkerOverlay:
    """Live manager for the reference-point marker overlay."""

    def __init__(
        self,
        plotter: Any,
        *,
        origin_shift: np.ndarray,
        model_diagonal: float,
        points: Sequence[tuple[float, float, float]] | None = None,
        show_coords: bool = True,
        visible: bool = True,
        size: float = 10.0,
    ) -> None:
        self.plotter = plotter
        self.origin_shift = np.asarray(origin_shift, dtype=np.float64)
        self.model_diagonal = float(model_diagonal)
        self.points: list[tuple[float, float, float]] = [
            tuple(map(float, p)) for p in (points or [])
        ]
        self.show_coords = bool(show_coords)
        self.visible = bool(visible)
        self.size = float(size)

        self._glyph_actor: Any = None
        self._label_actor: Any = None
        self._rebuild()

    # ── mutations ────────────────────────────────────────────────────

    def add(self, point: tuple[float, float, float]) -> None:
        self.points.append(tuple(map(float, point)))
        self._rebuild()

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.points):
            del self.points[index]
            self._rebuild()

    def clear(self) -> None:
        self.points.clear()
        self._rebuild()

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        self._rebuild()

    def set_show_coords(self, show: bool) -> None:
        self.show_coords = bool(show)
        self._rebuild()

    def set_size(self, size: float) -> None:
        self.size = float(size)
        self._rebuild()

    def set_origin_shift(self, shift: np.ndarray) -> None:
        """Re-sync after an external scene rebuild changed the shift."""
        self.origin_shift = np.asarray(shift, dtype=np.float64)
        self._rebuild()

    # ── internal ─────────────────────────────────────────────────────

    def _remove_actors(self) -> None:
        for a in (self._glyph_actor, self._label_actor):
            if a is None:
                continue
            try:
                self.plotter.remove_actor(a)
            except Exception:
                pass
        self._glyph_actor = None
        self._label_actor = None

    def _rebuild(self) -> None:
        self._remove_actors()
        if not self.visible or not self.points:
            try:
                self.plotter.render()
            except Exception:
                pass
            return
        from ..scene.origin_markers import build_origin_markers
        g, l = build_origin_markers(
            self.plotter,
            self.points,
            origin_shift=self.origin_shift,
            model_diagonal=self.model_diagonal,
            show_coords=self.show_coords,
            size=self.size,
        )
        self._glyph_actor = g
        self._label_actor = l
        try:
            self.plotter.render()
        except Exception:
            pass
