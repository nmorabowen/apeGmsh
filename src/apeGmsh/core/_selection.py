"""
Geometric selection primitives and the Selection result type.

Users never import from this module directly — everything is accessed
through ``m.model.queries.select()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import gmsh

DimTag = tuple[int, int]

if TYPE_CHECKING:
    from ._model_queries import _Queries


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bb_corners(bb: tuple) -> np.ndarray:
    """Return the 8 corners of an axis-aligned bounding box as (8, 3) array."""
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    return np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmin, ymax, zmin], [xmax, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmin, ymax, zmax], [xmax, ymax, zmax],
    ], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Geometric primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Plane:
    """Infinite plane defined by a unit normal and an anchor point."""
    normal: np.ndarray   # shape (3,), unit vector
    anchor: np.ndarray   # shape (3,), any point on the plane

    @classmethod
    def at(cls, **kwargs) -> "Plane":
        """Axis-aligned plane.  E.g. ``Plane.at(z=0)``, ``Plane.at(x=5)``."""
        if len(kwargs) != 1:
            raise ValueError("Plane.at() takes exactly one keyword, e.g. z=0")
        axis, value = next(iter(kwargs.items()))
        axes = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axes:
            raise ValueError(f"Unknown axis {axis!r}. Use 'x', 'y', or 'z'.")
        normal = np.zeros(3)
        normal[axes[axis]] = 1.0
        anchor = np.zeros(3)
        anchor[axes[axis]] = float(value)
        return cls(normal=normal, anchor=anchor)

    @classmethod
    def through(cls, p1, p2, p3) -> "Plane":
        """Plane through three non-collinear points."""
        p1, p2, p3 = np.array(p1, float), np.array(p2, float), np.array(p3, float)
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            raise ValueError("Points are collinear — cannot define a plane.")
        return cls(normal=n / norm, anchor=p1)

    def signed_distances(self, bb: tuple) -> np.ndarray:
        """Signed distance of each bounding-box corner from this plane."""
        corners = _bb_corners(bb)                       # (8, 3)
        return (corners - self.anchor) @ self.normal    # (8,)


@dataclass
class Line:
    """
    Infinite line used to cut 2-D geometry.

    The 'signed distance' is computed as the component of each bounding-box
    corner along the line's in-plane normal — the axis perpendicular to the
    line direction projected onto the dominant plane (XY, XZ, or YZ).
    """
    normal: np.ndarray   # shape (3,), unit vector perpendicular to line
    anchor: np.ndarray   # shape (3,), any point on the line

    @classmethod
    def through(cls, p1, p2) -> "Line":
        """Line through two points."""
        p1, p2 = np.array(p1, float), np.array(p2, float)
        d = p2 - p1
        norm = np.linalg.norm(d)
        if norm < 1e-14:
            raise ValueError("Points are coincident — cannot define a line.")
        d = d / norm
        # Build a normal perpendicular to d in the plane that best contains it
        # Try cross with Z, then Y, then X to avoid degeneracy
        for ref in (np.array([0., 0., 1.]), np.array([0., 1., 0.]), np.array([1., 0., 0.])):
            n = np.cross(d, ref)
            if np.linalg.norm(n) > 1e-6:
                break
        n = n / np.linalg.norm(n)
        return cls(normal=n, anchor=p1)

    def signed_distances(self, bb: tuple) -> np.ndarray:
        corners = _bb_corners(bb)
        return (corners - self.anchor) @ self.normal


# ─────────────────────────────────────────────────────────────────────────────
# Primitive parser — converts raw user input to Plane or Line
# ─────────────────────────────────────────────────────────────────────────────

def _parse_primitive(spec) -> Plane | Line:
    """
    Infer a geometric primitive from the user's raw input.

    Accepted formats
    ----------------
    {'z': 0}                         → Plane.at(z=0)
    [(x1,y1,z1), (x2,y2,z2)]        → Line through 2 points
    [(x1,y1,z1), (x2,y2,z2),
     (x3,y3,z3)]                     → Plane through 3 points
    Plane / Line instance            → passed through unchanged
    """
    if isinstance(spec, (Plane, Line)):
        return spec
    if isinstance(spec, dict):
        return Plane.at(**spec)
    pts = list(spec)
    if len(pts) == 2:
        return Line.through(pts[0], pts[1])
    if len(pts) == 3:
        return Plane.through(pts[0], pts[1], pts[2])
    raise ValueError(
        f"Cannot infer primitive from {spec!r}. "
        "Pass a dict ({'z': 0}), 2 points (line), or 3 points (plane)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core filter
# ─────────────────────────────────────────────────────────────────────────────

_DIM_NAMES = {0: 'points', 1: 'curves', 2: 'surfaces', 3: 'volumes'}


def _select_impl(dimtags: Iterable[DimTag], *, on=None, crossing=None,
                 not_on=None, not_crossing=None,
                 tol: float = 1e-6, _queries: "_Queries | None" = None) -> "Selection":
    """Apply one (possibly negated) predicate and return a new Selection."""
    given = [(label, val) for label, val in
             [('on', on), ('crossing', crossing),
              ('not_on', not_on), ('not_crossing', not_crossing)]
             if val is not None]
    if len(given) != 1:
        raise ValueError(
            "Pass exactly one of on=, crossing=, not_on=, not_crossing=."
        )
    label, spec = given[0]
    primitive   = _parse_primitive(spec)
    base_mode   = 'on' if 'on' in label else 'crossing'
    invert      = label.startswith('not_')

    result = []
    for d, t in dimtags:
        bb = gmsh.model.getBoundingBox(d, t)
        sd = primitive.signed_distances(bb)
        if base_mode == 'on':
            hit = bool(np.all(np.abs(sd) <= tol))
        else:
            hit = bool(sd.min() < -tol and sd.max() > tol)
        if hit ^ invert:
            result.append((d, t))

    return Selection(result, _queries=_queries)


# ─────────────────────────────────────────────────────────────────────────────
# Selection — chainable result type
# ─────────────────────────────────────────────────────────────────────────────

class Selection(list):
    """
    A filtered list of ``(dim, tag)`` pairs returned by
    ``m.model.queries.select()``.

    Chain ``.select()`` to narrow further::

        bottom_left = (m.model.queries
            .select(curves, on={'z': 0})
            .select(on={'x': 0}))

    Iterate directly or call ``.tags()`` for bare integers.
    """

    def __init__(self, dimtags: Iterable[DimTag] = (), *,
                 _queries: "_Queries | None" = None) -> None:
        super().__init__(dimtags)
        self._queries = _queries

    def select(self, *, on=None, crossing=None, not_on=None, not_crossing=None,
               tol: float = 1e-6) -> "Selection":
        """Filter this selection further.  Same arguments as ``queries.select()``."""
        return _select_impl(self, on=on, crossing=crossing,
                            not_on=not_on, not_crossing=not_crossing,
                            tol=tol, _queries=self._queries)

    def tags(self) -> list[int]:
        """Return bare integer tags (drops dim)."""
        return [t for _, t in self]

    def to_label(self, name: str) -> "Selection":
        """
        Register every entity in this selection as a label.

        Groups by dimension before calling ``session.labels.add`` so a
        mixed-dim Selection is handled correctly.  Returns ``self`` for
        chaining.

        Example
        -------
        ::

            (m.model.queries
                .select(curves, on={'x': 0})
                .select(on={'y': 5})
                .to_label('left_top_edge'))

            m.mesh.sizing.set_size('left_top_edge', size=0.1)
        """
        import warnings
        session = self._queries._model._parent
        dims    = sorted({d for d, _ in self})
        with warnings.catch_warnings():
            # Re-using the same name across multiple dims is the documented
            # intent here, not a mistake — silence the labels-composite warning
            # so a mixed-dim selection labels cleanly.
            if len(dims) > 1:
                warnings.filterwarnings(
                    "ignore", message=r".*already exists at dim.*",
                )
            for d in dims:
                tags = [t for dim, t in self if dim == d]
                session.labels.add(d, tags, name=name)
        return self

    def to_physical(self, name: str) -> "Selection":
        """
        Register every entity in this selection as a physical group.

        Groups by dimension before calling ``session.physical.add`` so a
        mixed-dim Selection is handled correctly.  Returns ``self`` for
        chaining.

        Example
        -------
        ::

            (m.model.queries
                .select(faces, on={'z': 0})
                .to_physical('Base'))

            g.constraints.fix('Base', dofs=[1, 2, 3])
        """
        session = self._queries._model._parent
        for d in sorted({d for d, _ in self}):
            tags = [t for dim, t in self if dim == d]
            session.physical.add(d, tags, name=name)
        return self

    # ── Set operations ──────────────────────────────────────────────────────

    def __or__(self, other) -> "Selection":
        """Union with deduplication.  Preserves order of *self* first."""
        seen   = set(self)
        merged = list(self) + [dt for dt in other if dt not in seen]
        return Selection(merged, _queries=self._queries)

    def __and__(self, other) -> "Selection":
        """Intersection."""
        other_set = set(other)
        return Selection([dt for dt in self if dt in other_set],
                         _queries=self._queries)

    def __sub__(self, other) -> "Selection":
        """Set difference — entities in *self* but not in *other*."""
        other_set = set(other)
        return Selection([dt for dt in self if dt not in other_set],
                         _queries=self._queries)

    # ── Partitioning ────────────────────────────────────────────────────────

    def partition_by(self, axis: str | None = None):
        """
        Group entities by their dominant bounding-box axis.

        Returns
        -------
        If ``axis`` is ``None``: ``dict[str, Selection]`` keyed by ``'x'``,
        ``'y'``, ``'z'``.
        If ``axis`` is one of ``'x'``, ``'y'``, ``'z'``: a single
        ``Selection`` for that axis only.

        Semantics by entity dimension
        -----------------------------
        - **dim = 1 (curves)** — dominant axis is the **largest** BB extent
          (the direction the curve runs along).
        - **dim = 2 (surfaces)** — dominant axis is the **smallest** BB extent
          (the surface normal — for axis-aligned faces this picks the
          perpendicular direction).
        - Mixed dims partition independently per dim using the right rule.

        Example
        -------
        ::

            curves = m.model.queries.boundary_curves('box')
            groups = curves.partition_by()
            m.mesh.structured.set_transfinite_curve(groups['x'].tags(), nx)
            m.mesh.structured.set_transfinite_curve(groups['y'].tags(), ny)
            m.mesh.structured.set_transfinite_curve(groups['z'].tags(), nz)
        """
        if axis is not None and axis not in ('x', 'y', 'z'):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

        groups: dict[str, list] = {'x': [], 'y': [], 'z': []}
        AXES = ('x', 'y', 'z')

        for d, t in self:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(d, t)
            spans = [xmax - xmin, ymax - ymin, zmax - zmin]
            if d == 1:
                # Curve → direction of largest extent
                idx = int(np.argmax(spans))
            elif d == 2:
                # Surface → axis with smallest extent (≈ normal direction)
                idx = int(np.argmin(spans))
            elif d == 3:
                # Volume → largest extent (most useful for transfinite hints)
                idx = int(np.argmax(spans))
            else:
                continue                       # dim 0 — points have no axis
            groups[AXES[idx]].append((d, t))

        if axis is not None:
            return Selection(groups[axis], _queries=self._queries)
        return {ax: Selection(items, _queries=self._queries)
                for ax, items in groups.items()}

    def __repr__(self) -> str:
        by_dim: dict[int, int] = {}
        for d, _ in self:
            by_dim[d] = by_dim.get(d, 0) + 1
        parts = [f"{n} {_DIM_NAMES.get(d, f'dim{d}')}" for d, n in sorted(by_dim.items())]
        summary = ', '.join(parts) if parts else 'empty'
        return f"Selection({summary}) — .select(on=..., crossing=...) to filter further"
