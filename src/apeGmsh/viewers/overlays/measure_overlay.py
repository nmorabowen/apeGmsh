"""
MeasureOverlay — point-to-point distance probe between entity centroids.

State machine (cycles each measurement):

    [empty] -- add_entity() -->  [one] -- add_entity() --> [two]
       ^                                                     |
       |--------------- next add_entity() resets ------------|
       |                                                     |
       +-- reset() ------------------------------------------+

When the first entity is captured, a yellow sphere marks its centroid.
When the second is captured, a yellow segment is drawn between the
centroids and a label at the midpoint shows ``|d|`` and the per-axis
deltas.  The next ``add_entity()`` after that wipes both and starts
fresh — so the viewer never accumulates more than one measurement.

Coordinate system
-----------------
``EntityRegistry`` stores centroids already shifted by
``registry.origin_shift`` — i.e. they are in the *rendered* coordinate
frame.  We draw all overlay actors in that same frame, and report the
distance scalar (which is translation-invariant).

Lifecycle
---------
After ``_rebuild_scene`` the centroids are recomputed; any in-flight
measurement points reference the previous registry state and are
stale.  Call :meth:`reset` from the rebuild path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._types import DimTag
    from ..core.entity_registry import EntityRegistry


_DIM_ABBR = {0: "P", 1: "C", 2: "S", 3: "V"}


class MeasureOverlay:
    """Two-click distance probe."""

    def __init__(self, plotter, registry: "EntityRegistry") -> None:
        self._plotter = plotter
        self._registry = registry
        self._points: list[np.ndarray] = []  # shifted coords
        self._labels: list[str] = []
        self._segment_actor = None
        self._marker_actor = None
        self._label_actor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear in-flight points and any drawn segment / label."""
        self._points.clear()
        self._labels.clear()
        self._clear_actors()
        self._plotter.render()

    def add_entity(self, dt: "DimTag") -> bool:
        """Capture the centroid of *dt* as the next measurement point.

        Returns
        -------
        bool
            True if the point was captured (centroid available).
            False if the entity has no centroid (caller should ignore).
        """
        c = self._registry.centroid(dt)
        if c is None:
            return False

        # Cycle: a third add wipes the previous measurement.
        if len(self._points) >= 2:
            self._points.clear()
            self._labels.clear()
            self._clear_actors()

        self._points.append(np.asarray(c, dtype=np.float64).copy())
        self._labels.append(f"{_DIM_ABBR.get(dt[0], '?')}{dt[1]}")

        if len(self._points) == 1:
            self._draw_marker()
        elif len(self._points) == 2:
            self._draw_segment()
        return True

    @property
    def num_points(self) -> int:
        return len(self._points)

    @property
    def last_distance(self) -> float | None:
        if len(self._points) < 2:
            return None
        return float(np.linalg.norm(self._points[1] - self._points[0]))

    @property
    def last_delta(self) -> tuple[float, float, float] | None:
        if len(self._points) < 2:
            return None
        d = self._points[1] - self._points[0]
        return (float(d[0]), float(d[1]), float(d[2]))

    @property
    def last_endpoints(self) -> tuple[str, str] | None:
        """Entity labels of the two endpoints, e.g. (``"P3"``, ``"S7"``)."""
        if len(self._labels) < 2:
            return None
        return (self._labels[0], self._labels[1])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clear_actors(self) -> None:
        for actor in (self._segment_actor, self._marker_actor, self._label_actor):
            if actor is None:
                continue
            try:
                self._plotter.remove_actor(actor)
            except Exception:
                pass
        self._segment_actor = None
        self._marker_actor = None
        self._label_actor = None

    def _draw_marker(self) -> None:
        """Yellow sphere at the first captured point."""
        try:
            self._marker_actor = self._plotter.add_points(
                self._points[0].reshape(1, 3),
                color="yellow",
                point_size=15,
                render_points_as_spheres=True,
                name="_measure_marker",
            )
        except Exception:
            pass
        self._plotter.render()

    def _draw_segment(self) -> None:
        # Drop the single-point marker before drawing the segment.
        if self._marker_actor is not None:
            try:
                self._plotter.remove_actor(self._marker_actor)
            except Exception:
                pass
            self._marker_actor = None

        p0, p1 = self._points[0], self._points[1]
        line_pts = np.vstack([p0, p1])
        try:
            self._segment_actor = self._plotter.add_lines(
                line_pts, color="yellow", width=3,
            )
        except Exception:
            pass

        d = float(np.linalg.norm(p1 - p0))
        dx, dy, dz = p1 - p0
        midpoint = ((p0 + p1) * 0.5).reshape(1, 3)
        text = (
            f"|d| = {d:.6g}\n"
            f"Δ = ({dx:.4g}, {dy:.4g}, {dz:.4g})"
        )
        try:
            self._label_actor = self._plotter.add_point_labels(
                midpoint, [text],
                font_size=12,
                shape_color="black",
                text_color="yellow",
                shape_opacity=0.85,
                show_points=False,
                always_visible=True,
                name="_measure_label",
            )
        except Exception:
            pass
        self._plotter.render()


__all__ = ["MeasureOverlay"]
