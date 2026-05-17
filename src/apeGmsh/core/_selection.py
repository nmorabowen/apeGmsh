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

# Root leaf — stdlib/typing only (see docs/plans/selection-unification.md
# §3 and tests/test_import_dag_polarity.py).  Importing it here adds NO
# eager cross-package edge among {core, mesh, viz, results}: ``_chain``
# is the package-root leaf, not one of those four packages, so the
# polarity baseline is unaffected (identical idiom to
# ``mesh/_node_chain.py``).
from .._chain import SelectionChain

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


def _cluster_edge_directions(
    curve_dimtags: list[DimTag],
    *,
    angle_tol_deg: float = 5.0,
) -> list[tuple[np.ndarray, list[DimTag]]]:
    """Group curves by chord direction (anti-parallel = same cluster).

    Returns a list of ``(mean_direction, [curve_dimtags])`` tuples — one
    per distinct principal direction.  A clean axis-aligned hex volume
    yields exactly 3 clusters of 4 curves each.

    The ``mean_direction`` of each cluster is sign-canonicalised so the
    first non-zero component is positive — gives deterministic axis
    ordering downstream.
    """
    import math
    cos_tol = math.cos(math.radians(angle_tol_deg))
    clusters: list[dict] = []
    for dt in curve_dimtags:
        d = _chord_direction(dt)
        matched = False
        for cluster in clusters:
            if abs(float(d @ cluster["mean"])) >= cos_tol:
                # Flip d to align with cluster mean before averaging.
                d_aligned = d if float(d @ cluster["mean"]) >= 0 else -d
                cluster["dts"].append(dt)
                m = (cluster["mean"] * (len(cluster["dts"]) - 1) + d_aligned)
                cluster["mean"] = m / np.linalg.norm(m)
                matched = True
                break
        if not matched:
            clusters.append({"mean": d.copy(), "dts": [dt]})

    # Sign-canonicalise: first non-zero component positive.
    result = []
    for c in clusters:
        m = c["mean"]
        for v in m:
            if abs(v) > 1e-9:
                if v < 0:
                    m = -m
                break
        result.append((m, c["dts"]))
    return result


def _order_clusters_by_global_axis(
    clusters: list[tuple[np.ndarray, list[DimTag]]],
) -> list[tuple[np.ndarray, list[DimTag]]]:
    """Greedy-assign clusters to global axes (X, Y, Z) in that order.

    For each of X, Y, Z (in order), picks the unclaimed cluster with the
    largest ``|dot|`` against that global axis.  Tie-breaks by lex order
    on the cluster mean direction.  Deterministic.

    Returns clusters reordered so position 0 is the X-aligned cluster,
    position 1 is Y-aligned, position 2 is Z-aligned (skipping unused
    positions when there are fewer clusters than 3).
    """
    if not clusters:
        return []
    global_axes = [_AXIS_VECTORS["x"], _AXIS_VECTORS["y"], _AXIS_VECTORS["z"]]
    remaining = list(clusters)
    ordered: list[tuple[np.ndarray, list[DimTag]]] = []
    for gax in global_axes:
        if not remaining:
            break
        remaining.sort(
            key=lambda c: (-abs(float(c[0] @ gax)), tuple(c[0].tolist())),
        )
        ordered.append(remaining.pop(0))
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Selection — chainable result type
# ─────────────────────────────────────────────────────────────────────────────

class Selection(list):
    """
    A filtered list of ``(dim, tag)`` pairs returned by
    ``m.model.queries.select()`` and the ``select_all_*`` entry points.

    A Selection is a ``list`` subclass, so it iterates as ``(dim, tag)``
    pairs and supports indexing.  It is also chainable — every method
    that narrows or combines returns a new Selection.

    Refine (narrow what you have)
    -----------------------------
    ============================== ==========================================
    ``.select(...)``               position predicates: ``on``, ``crossing``,
                                   ``not_on``, ``not_crossing``
    ``.parallel_to(direction)``    curves whose chord is along a direction
    ``.normal_along(direction)``   surfaces whose normal is along a direction
    ``.partition_by(axis=None)``   group entities by dominant BB axis
    ============================== ==========================================

    Combine (set algebra on two Selections)
    ---------------------------------------
    Set semantics with deduplication — appropriate for ``(dim, tag)``
    pairs, where logical identity matters and duplicates would cause
    downstream calls (e.g. ``to_physical``) to register the same
    entity twice.

    Each operation has both an **operator** form (terse, for one-liners)
    and a **named-method** form (discoverable via autocomplete, keeps the
    chain fluent).

    ===================== ===================== ===================== ===========================
    Operator              Method                Meaning               Example
    ===================== ===================== ===================== ===========================
    ``a | b``             ``a.union(b)``        union                 ``nx | ny``
    ``a & b``             ``a.intersect(b)``    intersection          ``top & front``
    ``a - b``             ``a.difference(b)``   set difference        ``all - horizontal``
    ===================== ===================== ===================== ===========================

    Why ``|`` and not ``+``: ``Selection`` subclasses ``list``, where
    ``+`` is concatenation with duplicates preserved.  ``|`` follows
    the ``set`` / ``dict`` convention for combining-with-dedup, and is
    the right semantics for selection sets.

    Consume (turn a Selection into something else)
    ----------------------------------------------
    ================================ ===========================================
    ``.tags()``                      bare integer tags (drops dim)
    ``.to_label(name)``              register entities as a label
    ``.to_physical(name)``           register entities as a physical group
    ================================ ===========================================

    Example
    -------
    ::

        surf = m.model.queries.select_all_surfaces()

        # Lateral sides of an axis-aligned box — three equivalent forms:
        (surf.normal_along("x") | surf.normal_along("y")).to_physical("sides")
        (surf - surf.normal_along("z")).to_physical("sides")
        surf.normal_along("x").union(surf.normal_along("y")).to_physical("sides")

        # Chain refine → consume
        (m.model.queries
            .select(curves, on={'z': 0})
            .select(on={'x': 0})
            .to_label("bottom_left_edge"))
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
        """Union with deduplication — ``a | b``.

        Returns a Selection containing every ``(dim, tag)`` in *self*
        or *other*, with duplicates removed.  Order of *self* is
        preserved first, then new entries from *other* appended.

        Example
        -------
        ::

            sides = surf.normal_along("x") | surf.normal_along("y")
        """
        seen   = set(self)
        merged = list(self) + [dt for dt in other if dt not in seen]
        return Selection(merged, _queries=self._queries)

    def __and__(self, other) -> "Selection":
        """Intersection — ``a & b``.

        Returns a Selection containing only ``(dim, tag)`` pairs that
        appear in **both** *self* and *other*.

        Example
        -------
        ::

            # Curves shared by the top face and the front face (the top-front edge)
            edge = top_face_curves & front_face_curves
        """
        other_set = set(other)
        return Selection([dt for dt in self if dt in other_set],
                         _queries=self._queries)

    def __sub__(self, other) -> "Selection":
        """Set difference — ``a - b``.

        Returns a Selection containing entities in *self* but not in
        *other*.  Useful for "everything except…" patterns.

        Example
        -------
        ::

            # All faces except the horizontal ones (top, bottom, interfaces)
            laterals = surf - surf.normal_along("z")
        """
        other_set = set(other)
        return Selection([dt for dt in self if dt not in other_set],
                         _queries=self._queries)

    # Named aliases — discoverable via autocomplete; operators stay for terse code.
    def union(self, other) -> "Selection":
        """Alias for ``self | other``.  See :meth:`__or__`."""
        return self.__or__(other)

    def intersect(self, other) -> "Selection":
        """Alias for ``self & other``.  See :meth:`__and__`."""
        return self.__and__(other)

    def difference(self, other) -> "Selection":
        """Alias for ``self - other``.  See :meth:`__sub__`."""
        return self.__sub__(other)

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


# ─────────────────────────────────────────────────────────────────────────────
# GeometryChain — the entity-family chainable (S3a)
# ─────────────────────────────────────────────────────────────────────────────
#
# This is the fluent *intermediate* for the geometry level of the unified
# selection family (docs/plans/selection-unification.md §5, ratified
# decision R3).  It is a sibling of the legacy ``Selection`` (above) —
# the legacy class is left **byte-unchanged** as the terminal type, and
# ``GeometryChain._materialize()`` returns exactly that legacy
# ``Selection`` so ``.result().to_label()/.to_physical()/.tags()`` keep
# working through the existing terminal with zero behavior change.
#
# FAMILY = "entity": atoms are ``(dim, tag)`` CAD dimtags.  Per R3 the
# spatial contract is *entity-typed*, NOT the point-family
# coordinate-containment contract:
#
#   * ``in_box``  -> ``gmsh.model.getEntitiesInBoundingBox`` (Gmsh's
#     own BRep entity query).  Empirically this is bbox-CONTAINMENT,
#     not bbox-intersect: an entity is returned iff its *whole*
#     bounding box lies inside the query box, with the box expanded by
#     ``Geometry.Tolerance`` (~1e-8) — strict on the raw bound, so a
#     query box exactly equal to an entity's own extent does NOT
#     contain it.  Either way it is a single closed-ish entity test
#     with no half-open notion, so the point-family ``inclusive=`` knob
#     is *inexpressible*: passing it raises ``TypeError`` (fail loud —
#     R3; never silently ignored).
#   * ``in_sphere`` / ``on_plane`` / ``nearest_to`` / ``where`` use the
#     entity **bounding-box** (centre for sphere/nearest/where; all 8
#     corners for on_plane), reusing the legacy ``Plane`` /
#     ``_bb_corners`` primitives.  These are coarse *entity* tests by
#     construction — for exact geometric on/crossing semantics use the
#     legacy ``queries.select(on=/crossing=)`` predicate, unchanged.
#
# The opaque ``_engine`` is the ``_Queries`` instance: it gives access
# to the owning session (``_engine._model._parent``) for the locked
# tiered name resolver, and is exactly the object the legacy
# ``Selection`` wants as ``_queries=`` so the terminal stays wired.


class GeometryChain(SelectionChain):
    """Daisy-chainable CAD-entity selection (entity family).

    Atoms are ``(dim, tag)`` dimtags.  Constructed by the additive
    ``g.model.select(...)`` host hook (see :meth:`core.Model.Model.select`),
    which delegates *all* name resolution to the existing, contract-locked
    geometry resolver — this class never re-implements tier logic.

    Chaining / set-algebra / verb-name enforcement are inherited from
    :class:`apeGmsh._chain.SelectionChain` unchanged.  Only the
    entity-family spatial hooks and the ``in_box`` override live here.

    Example
    -------
    ::

        sel = (g.model.select("box", dim=2)        # tiered name resolve
                 .in_box((0, 0, 0), (1, 1, 0.5))   # gmsh BRep query
                 .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6))
        legacy = sel.result()        # -> core/_selection.Selection
        legacy.to_physical("Base")   # legacy terminal, unchanged
    """

    FAMILY = "entity"

    __slots__ = ()

    # ── coordinate access — entity bounding-box centre ──────
    def _coords_of(self, atoms: tuple) -> np.ndarray:
        """Bounding-box **centre** of each ``(dim, tag)`` entity.

        Entity-family semantics: an entity has no single coordinate, so
        ``nearest_to`` / ``where`` operate on its bbox centre.  This is a
        coarse proxy (a long curve and a point can share a centre) — for
        exact geometric predicates use the legacy ``on=``/``crossing=``.
        """
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        rows = []
        for d, t in atoms:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                int(d), int(t)
            )
            rows.append((
                0.5 * (xmin + xmax),
                0.5 * (ymin + ymax),
                0.5 * (zmin + zmax),
            ))
        return np.asarray(rows, dtype=np.float64)

    # ── in_box override — gmsh BRep, inclusive= forbidden ───
    def in_box(self, lo, hi, **kw) -> "GeometryChain":
        """Refine to entities whose BRep bounding box lies in ``[lo, hi]``.

        Delegates to ``gmsh.model.getEntitiesInBoundingBox`` — Gmsh's
        own entity query.  Its semantics are bbox-CONTAINMENT (the
        whole entity bounding box must lie inside the query box, the
        box expanded by ``Geometry.Tolerance`` ~1e-8), *not*
        coordinate half-open and *not* bbox-intersect — so to capture a
        boundary entity the query box must comfortably enclose it (a
        box exactly equal to the entity's own extent will exclude it).
        This is the honest *entity-family* contract (R3): there is no
        half-open notion, so the point-family ``inclusive=`` knob is
        inexpressible and is rejected loudly rather than silently
        ignored.

        Parameters
        ----------
        lo, hi : 3-sequences
            Lower / upper corners of the query box.

        Raises
        ------
        TypeError
            If ``inclusive=`` (or any keyword) is passed — the
            half-open knob does not exist for the gmsh BRep query
            (ratified R3).
        """
        if kw:
            raise TypeError(
                "GeometryChain.in_box() does not accept "
                f"{sorted(kw)!r}. The entity family uses "
                "gmsh.model.getEntitiesInBoundingBox (BRep "
                "bbox-intersect), which is inherently closed — the "
                "half-open / 'inclusive=' knob is point-family only "
                "and inexpressible here (selection-unification R3). "
                "Drop the keyword; use queries.select(on=/crossing=) "
                "for an exact geometric predicate."
            )
        return self._wrap(self._spatial_box(self._items, lo, hi))

    def _spatial_box(self, atoms: tuple, lo, hi) -> tuple:
        """gmsh BRep containment query, intersected with the chain.

        ``getEntitiesInBoundingBox`` (bbox-CONTAINMENT — see
        :meth:`in_box`) is queried per distinct dim present in
        ``atoms`` (querying per-dim is deterministic and avoids the
        ``dim=-1`` all-dims ambiguity).  The gmsh hit set is then
        intersected with the chain's current atoms so the verb
        *refines* (chain protocol), preserving the chain's insertion
        order.
        """
        if not atoms:
            return ()
        lo = [float(v) for v in lo]
        hi = [float(v) for v in hi]
        dims = sorted({int(d) for d, _ in atoms})
        hits: set = set()
        for d in dims:
            for hd, ht in gmsh.model.getEntitiesInBoundingBox(
                lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], d
            ):
                hits.add((int(hd), int(ht)))
        return tuple(a for a in atoms if (int(a[0]), int(a[1])) in hits)

    # ── in_sphere — entity bbox centre within radius ────────
    def _spatial_sphere(self, atoms: tuple, center, radius: float) -> tuple:
        """Keep entities whose **bbox centre** is within ``radius``.

        Closed ball on the entity bounding-box centre — an entity-family
        proxy, not exact geometric containment.
        """
        r = float(radius)
        if r < 0:
            raise ValueError(f"radius must be non-negative, got {r}.")
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        ctr = np.asarray(center, dtype=np.float64).reshape(3)
        mask = np.linalg.norm(c - ctr, axis=1) <= r
        return tuple(a for a, k in zip(atoms, mask) if k)

    # ── on_plane — all 8 bbox corners within tol of plane ───
    def _spatial_plane(self, atoms: tuple, point, normal, tol: float) -> tuple:
        """Keep entities lying **entirely on** the plane.

        Reuses the legacy :class:`Plane` + :func:`_bb_corners`
        primitives — an entity is kept iff *all 8* of its bounding-box
        corners are within ``tol`` of the plane.  This is exactly the
        legacy ``select(on=...)`` "on" test (entity family), expressed
        through the unified chain surface.
        """
        t = float(tol)
        if t < 0:
            raise ValueError(f"tolerance must be non-negative, got {t}.")
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        nn = np.linalg.norm(n)
        if nn == 0:
            raise ValueError("normal vector has zero length.")
        if not atoms:
            return ()
        plane = Plane(normal=n / nn, anchor=np.asarray(point, dtype=np.float64))
        kept = []
        for d, t_ in atoms:
            bb = gmsh.model.getBoundingBox(int(d), int(t_))
            if bool(np.all(np.abs(plane.signed_distances(bb)) <= t)):
                kept.append((d, t_))
        return tuple(kept)

    # ── terminal — the LEGACY Selection, unchanged ──────────
    def _materialize(self) -> "Selection":
        """Return the legacy :class:`Selection`, constructed as today.

        ``_engine`` is the ``_Queries`` instance, so the terminal is
        wired exactly like ``queries.select(...)`` output: ``.to_label``
        / ``.to_physical`` / ``.tags`` / ``.select(on=)`` all work
        through the byte-unchanged legacy class.
        """
        return Selection(list(self._items), _queries=self._engine)
