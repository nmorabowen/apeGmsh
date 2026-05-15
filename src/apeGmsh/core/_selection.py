"""
Geometric selection primitives and the Selection result type.

Users never import from this module directly — everything is accessed
through ``m.model.queries.select()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import gmsh

DimTag = tuple[int, int]

_AXIS_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}

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
    if len(given) > 1:
        raise ValueError(
            "Pass at most one of on=, crossing=, not_on=, not_crossing=."
        )
    if not given:
        # No predicate — return the resolved entities unfiltered.  Lets
        # callers use queries.select("name", dim=N) as a "resolve only"
        # entry point that returns a chainable Selection.
        return Selection(list(dimtags), _queries=_queries)
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
# Direction helpers — for Selection.parallel_to() and .normal_along()
# ─────────────────────────────────────────────────────────────────────────────

def _parse_direction(d) -> np.ndarray:
    """Resolve an axis alias or 3-vector to a unit vector."""
    if isinstance(d, str):
        key = d.lower()
        if key not in _AXIS_VECTORS:
            raise ValueError(
                f"Unknown axis alias {d!r}. Use 'x', 'y', 'z', or a 3-vector."
            )
        return _AXIS_VECTORS[key].copy()
    v = np.asarray(d, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"Direction must be a 3-vector; got shape {v.shape}.")
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Direction vector has zero magnitude.")
    return v / n


def _chord_direction(dt: DimTag) -> np.ndarray:
    """Endpoint-to-endpoint unit vector for a curve (dim=1 entity)."""
    bnd = gmsh.model.getBoundary([dt], oriented=False, recursive=False)
    if len(bnd) < 2:
        raise ValueError(
            f"Curve {dt} has fewer than 2 endpoints (closed curve?); "
            "cannot compute a chord direction."
        )
    p0 = np.array(gmsh.model.getValue(0, bnd[0][1], []), dtype=float)
    p1 = np.array(gmsh.model.getValue(0, bnd[1][1], []), dtype=float)
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Curve {dt} has coincident endpoints.")
    return v / n


def _face_normal(dt: DimTag) -> np.ndarray:
    """Unit normal of a flat surface, computed from 3 boundary points.

    Works for both the built-in and OCC kernels (no kernel-specific calls).
    For a flat face this is exact.  For curved faces it returns the normal
    of the chord plane through 3 sampled boundary points — a coarse
    approximation; prefer ``on=`` predicates for curved surfaces.
    """
    # Collect every boundary point of the surface (boundary curves' endpoints).
    bnd_curves = gmsh.model.getBoundary([dt], oriented=False, recursive=False)
    pt_tags: list[int] = []
    seen: set[int] = set()
    for cd, ct in bnd_curves:
        for pd, pt in gmsh.model.getBoundary(
            [(cd, ct)], oriented=False, recursive=False,
        ):
            if pt not in seen:
                seen.add(pt)
                pt_tags.append(pt)
    if len(pt_tags) < 3:
        raise ValueError(
            f"Surface {dt} has fewer than 3 boundary points; "
            "cannot compute a normal."
        )

    coords = [np.array(gmsh.model.getValue(0, pt, []), dtype=float)
              for pt in pt_tags]
    p0 = coords[0]
    v1 = coords[1] - p0
    # Find a third point not collinear with p0, p1.
    for p in coords[2:]:
        v2 = p - p0
        n = np.cross(v1, v2)
        nlen = np.linalg.norm(n)
        if nlen > 1e-12:
            return n / nlen
    raise ValueError(
        f"Surface {dt}: boundary points are collinear; cannot compute normal."
    )


def _require_dim(sel: "Selection", expected_dim: int, *, method: str) -> None:
    """Raise an educational error if ``sel`` contains entities of other dims."""
    bad = [dt for dt in sel if dt[0] != expected_dim]
    if bad:
        preview = bad[:3] + (["..."] if len(bad) > 3 else [])
        raise ValueError(
            f"Selection.{method}() requires dim={expected_dim} entities, "
            f"but got {len(bad)} entity(ies) of other dims: {preview}\n"
            f"Either narrow your Selection first, e.g.\n"
            f"    queries.select('your_target', dim={expected_dim}).{method}(...)\n"
            f"or filter by dim before calling this method."
        )


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

    # ------------------------------------------------------------------
    # Direction-based filters — dim-restricted
    # ------------------------------------------------------------------

    def parallel_to(
        self,
        direction: "str | tuple[float, float, float] | np.ndarray",
        *,
        angle_tol: float = 1.0,
    ) -> "Selection":
        """Keep curves whose endpoint chord is parallel to ``direction``.

        Only meaningful for curves (dim=1).  Raises ``ValueError`` if the
        Selection contains entities of any other dim.

        Parameters
        ----------
        direction : str or 3-vector
            ``"x"``, ``"y"``, ``"z"`` for axis aliases, or any non-zero
            3-vector for an arbitrary direction.  Anti-parallel matches
            count as parallel — a z-edge with reversed endpoint order is
            still a z-edge.
        angle_tol : float, default 1.0
            Maximum angle (in degrees) between the curve's chord direction
            and ``direction`` for the curve to be kept.

        Returns
        -------
        Selection
            New Selection of curves that match.

        Example
        -------
        ::

            edges = m.model.queries.select("layer_1", dim=1)
            verticals = edges.parallel_to("z")
            obliques  = edges.parallel_to((1, 1, 0), angle_tol=2.0)

            m.mesh.structured.set_transfinite_curve(verticals.tags(), n=21)
        """
        _require_dim(self, 1, method="parallel_to")
        target = _parse_direction(direction)
        cos_tol = math.cos(math.radians(angle_tol))
        kept = [
            dt for dt in self
            if abs(float(_chord_direction(dt) @ target)) >= cos_tol
        ]
        return Selection(kept, _queries=self._queries)

    def normal_along(
        self,
        direction: "str | tuple[float, float, float] | np.ndarray",
        *,
        angle_tol: float = 1.0,
    ) -> "Selection":
        """Keep surfaces whose face normal is along ``direction``.

        Only meaningful for surfaces (dim=2).  Raises ``ValueError`` if
        the Selection contains entities of any other dim.

        Same direction grammar and tolerance as :meth:`parallel_to`.  The
        normal is computed from three boundary points — exact for flat
        faces, an approximation for curved faces (prefer ``on=`` for those).
        Anti-parallel matches count as parallel.

        Example
        -------
        ::

            faces = m.model.queries.select("layer_1", dim=2)
            horizontals = faces.normal_along("z")
            verticals   = faces.normal_along("x").select(...)
        """
        _require_dim(self, 2, method="normal_along")
        target = _parse_direction(direction)
        cos_tol = math.cos(math.radians(angle_tol))
        kept = [
            dt for dt in self
            if abs(float(_face_normal(dt) @ target)) >= cos_tol
        ]
        return Selection(kept, _queries=self._queries)

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
