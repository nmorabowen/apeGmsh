"""
Geometric selection primitives and the Selection result type.

Users never import from this module directly — everything is accessed
through ``m.model.select()``, which returns an
:class:`EntitySelection`; its ``.result()`` terminal yields the
:class:`Selection` payload defined here.
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
from .._kernel.chain import SelectionChain

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
# Entity-family crossing/straddle hook for EntitySelection
# (selection-unification-v2 P2-G, HT9).
#
# This is the unified-idiom ``crossing_plane`` straddle.  It ports
# ``_select_impl``'s 8-bounding-box-corner semantics **exactly**
# (``core/_selection.py`` :166-194 — the in-module engine):
#
#   * ``spec`` is the ``_parse_primitive`` grammar (dict→Plane,
#     2 points→``Line``, 3 points→Plane, ``Plane``/``Line`` instance
#     passthrough);
#   * ``mode`` ∈ {on, crossing, not_on, not_crossing} is the exactly-
#     one-of straddle predicate;
#   * per dimtag ``bb = gmsh.model.getBoundingBox(d, t)``;
#     ``sd = primitive.signed_distances(bb)``;
#     on  = ``bool(np.all(np.abs(sd) <= tol))``;
#     crossing = ``bool(sd.min() < -tol and sd.max() > tol)``;
#     ``not_*`` inverts (``hit ^ invert``);
#   * ``tol`` default ``1e-6``.
#
# It refines the chain (intersects with ``atoms``, preserving the
# chain's insertion order — the chain-protocol contract), so the
# returned tuple is exactly the ``Selection`` membership set restricted
# to (and ordered by) the chain's current atoms.  Defined in THIS
# module, so it reuses the in-module ``Plane`` / ``Line`` /
# ``_bb_corners`` / ``_parse_primitive`` primitives and adds **no** new
# import edge (the import-DAG-polarity BASELINE is unaffected).

_CROSSING_MODES = ("on", "crossing", "not_on", "not_crossing")


def _crossing_impl(atoms: tuple, spec, *, tol: float, mode: str) -> tuple:
    """Entity-family straddle filter (the ``crossing_plane`` engine).

    Returns the subset of ``atoms`` (``(dim, tag)`` dimtags) that
    satisfies the ``mode`` predicate against ``spec``, preserving the
    chain's insertion order.  Semantics are byte-identical to
    :func:`_select_impl`.
    """
    if mode not in _CROSSING_MODES:
        raise ValueError(
            f"crossing_plane(mode={mode!r}) is invalid; expected one of "
            f"{_CROSSING_MODES} (the on=/crossing=/not_on=/not_crossing= "
            "straddle predicate set)."
        )
    t = float(tol)
    if t < 0:
        raise ValueError(f"tolerance must be non-negative, got {t}.")
    primitive = _parse_primitive(spec)
    base_mode = 'on' if 'on' in mode else 'crossing'
    invert = mode.startswith('not_')

    kept = []
    for d, tg in atoms:
        bb = gmsh.model.getBoundingBox(int(d), int(tg))
        sd = primitive.signed_distances(bb)
        if base_mode == 'on':
            hit = bool(np.all(np.abs(sd) <= t))
        else:
            hit = bool(sd.min() < -t and sd.max() > t)
        if hit ^ invert:
            kept.append((d, tg))
    return tuple(kept)


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
            f"    m.model.select('your_target', dim={expected_dim}).{method}(...)\n"
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
    A filtered list of ``(dim, tag)`` pairs — the payload yielded by
    the ``.result()`` terminal of an :class:`EntitySelection` (itself
    returned by ``m.model.select(...)``).  **Retained by architecture**
    as the entity-side terminal payload; it is not a legacy or
    backward-compat type.

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

        surf = m.model.select(dim=2).result()

        # Lateral sides of an axis-aligned box — three equivalent forms:
        (surf.normal_along("x") | surf.normal_along("y")).to_physical("sides")
        (surf - surf.normal_along("z")).to_physical("sides")
        surf.normal_along("x").union(surf.normal_along("y")).to_physical("sides")

        # Chain refine → consume
        (m.model.select(curves, dim=1).result()
            .select(on={'z': 0})
            .select(on={'x': 0})
            .to_label("bottom_left_edge"))
    """

    def __init__(self, dimtags: Iterable[DimTag] = (), *,
                 _queries: "_Queries | None" = None) -> None:
        super().__init__(dimtags)
        self._queries = _queries

    def select(self, *, on=None, crossing=None, not_on=None, not_crossing=None,
               tol: float = 1e-6) -> "Selection":
        """Filter this selection further by position predicates
        (``on`` / ``crossing`` / ``not_on`` / ``not_crossing``)."""
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

            (m.model.select(curves, dim=1).result()
                .select(on={'x': 0})
                .select(on={'y': 5})
                .to_label('left_top_edge'))

            m.mesh.sizing.set_size('left_top_edge', size=0.1)
        """
        import warnings
        if self._queries is None:
            raise RuntimeError(
                "Selection.to_label()/.to_physical() requires a Selection "
                "bound to the model-queries engine — build it via "
                "m.model.select(...).result(). This Selection has "
                "_queries=None (constructed standalone), so it has no "
                "session to register the label/physical-group on."
            )
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

            (m.model.select(faces, dim=2).result()
                .select(on={'z': 0})
                .to_physical('Base'))

            g.constraints.fix('Base', dofs=[1, 2, 3])
        """
        if self._queries is None:
            raise RuntimeError(
                "Selection.to_label()/.to_physical() requires a Selection "
                "bound to the model-queries engine — build it via "
                "m.model.select(...).result(). This Selection has "
                "_queries=None (constructed standalone), so it has no "
                "session to register the label/physical-group on."
            )
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

            edges = m.model.select("layer_1", dim=1).result()
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

            faces = m.model.select("layer_1", dim=2).result()
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
# EntitySelection — the v2 entity-family terminal (selection-unification-v2 P2-I)
# ─────────────────────────────────────────────────────────────────────────────
#
# This is the **chain==terminal** entity-family type the
# ``g.model.select(...)`` host hook returns
# (``docs/plans/selection-unification-v2.md`` §4/§5 R-v2-2/-3, §6 P2-I).
# It is defined in this same module, so it adds *no* new import edge
# (the import-DAG-polarity BASELINE is unaffected — ``core/_selection.py``
# already imports the package-root leaf ``_kernel.chain``).
#
# The entity-family spatial contract reuses the same in-module
# ``Plane`` / ``Line`` / ``_bb_corners`` / ``_parse_primitive``
# primitives and the same gmsh-BRep / bbox hooks.  On top of the
# composable verbs it adds the v2 **direct terminals** so the chain
# *is* the terminal (no ``.result()`` ceremony required):
#
#   * ``.to_label(name)``   — Tier-1, ``_label:``-prefixed,
#     boolean-op-stable (``session.labels.add`` per dim);
#   * ``.to_physical(name)`` — Tier-2, raw gmsh PG
#     (``session.physical.add`` per dim);
#   * ``.to_dataframe()``   — local mirror of ``viz/Selection.py``'s
#     columns (no ``viz`` import; gmsh bbox/mass + session label
#     reverse-map);
#   * ``.result()`` / ``._materialize()`` — zero-cost identity alias to
#     the ``core/_selection.Selection`` payload (R-v2-2), so
#     ``.tags()`` / ``.to_label`` / ``.to_physical`` / ``.select(on=)``
#     are all available on it.
#
# Tier-1 (``.to_label``) and Tier-2 (``.to_physical``) are **separate
# registries that must never be merged** — ADR 0015.  A ``Clash``
# user-name can legitimately coexist as ``(d, 'Clash')`` (Tier-2) *and*
# ``(d, '_label:Clash')`` (Tier-1); merging them silently destroys
# Tier-1 boolean-op identity.


class EntitySelection(SelectionChain):
    """Daisy-chainable + terminal CAD-entity selection (entity family).

    Atoms are ``(dim, tag)`` dimtags.  Constructed by the
    ``g.model.select(...)`` host hook (see
    :meth:`core.Model.Model.select`), which delegates *all* name
    resolution to the existing, contract-locked geometry resolver —
    this class never re-implements tier logic.

    Every inherited verb / set-algebra / spatial hook composes; the
    direct terminals are ``.to_label`` / ``.to_physical`` /
    ``.to_dataframe``, and ``.result()`` is a zero-cost identity alias
    to the :class:`Selection` payload (retained by architecture as the
    entity-side terminal type).

    Example
    -------
    ::

        (g.model.select("BottomFaces")
            .in_box((0, 0, 0), (1, 1, 0.5))
            .to_physical("lower_faces"))      # Tier-2, direct terminal
    """

    FAMILY = "entity"

    __slots__ = ()

    # ── coordinate access — entity bounding-box centre ──────
    def _coords_of(self, atoms: tuple) -> np.ndarray:
        """Bounding-box **centre** of each ``(dim, tag)`` entity.

        Entity-family ``nearest_to`` / ``where`` operate on the bbox
        centre (a coarse proxy; for exact geometric predicates use
        ``.on_plane(...)`` / ``.crossing_plane(...)``).
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
    def in_box(self, lo, hi, **kw) -> "EntitySelection":
        """Refine to entities whose BRep bounding box lies in ``[lo, hi]``.

        Delegates to ``gmsh.model.getEntitiesInBoundingBox`` (BRep
        CONTAINMENT, closed, ``Geometry.Tolerance`` ~1e-8 expanded).
        The point-family ``inclusive=`` half-open knob is inexpressible
        for the entity family and is rejected **loudly** (R3 / ADR
        precedent) — never silently ignored.

        Raises
        ------
        TypeError
            If ``inclusive=`` (or any keyword) is passed.
        """
        if kw:
            raise TypeError(
                "EntitySelection.in_box() does not accept "
                f"{sorted(kw)!r}. The entity family uses "
                "gmsh.model.getEntitiesInBoundingBox (BRep "
                "bbox-intersect), which is inherently closed — the "
                "half-open / 'inclusive=' knob is point-family only "
                "and inexpressible here (selection-unification R3). "
                "Drop the keyword; use .on_plane(...) / "
                ".crossing_plane(...) for an exact geometric predicate."
            )
        return self._wrap(self._spatial_box(self._items, lo, hi))

    def _spatial_box(self, atoms: tuple, lo, hi) -> tuple:
        """gmsh BRep containment query, intersected with the chain.

        ``getEntitiesInBoundingBox`` queried per distinct dim present
        in ``atoms``, then intersected with the chain's current atoms
        so the verb *refines* (chain protocol), preserving insertion
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
        """Closed ball on the entity bbox centre (entity-family
        proxy)."""
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
        """An entity is kept iff *all 8* of its bbox corners are within
        ``tol`` of the plane (the ``on=`` straddle test, entity family,
        via the in-module :class:`Plane`)."""
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

    # ── crossing_plane — legacy on/crossing/not_* straddle ──
    def _spatial_crossing(
        self, atoms: tuple, spec, *, tol: float, mode: str
    ) -> tuple:
        """The ``crossing_plane`` straddle fold.  Delegates to the
        module-level :func:`_crossing_impl`: the
        ``on``/``crossing``/``not_on``/``not_crossing`` predicate over
        the 8-bbox-corner signed-distance semantics
        (:func:`_select_impl`) through the unified chain surface,
        refining the chain (intersected with ``atoms``, insertion order
        preserved).
        """
        return _crossing_impl(atoms, spec, tol=tol, mode=mode)

    # ── session access (same wiring Selection carries) ──
    def _session(self):
        """The owning session.

        ``_engine`` is the ``_Queries`` instance — exactly the object
        :class:`Selection` carries as ``_queries=``.  The session is
        ``_engine._model._parent`` (the same wiring
        ``Selection.to_label`` uses at ``core/_selection.py``).
        """
        if self._engine is None:
            raise RuntimeError(
                "EntitySelection.to_label()/.to_physical()/"
                ".to_dataframe() requires a selection bound to the "
                "model-queries engine — build it via "
                "g.model.select(...). This selection has _engine=None "
                "(constructed standalone), so it has no session to "
                "register on / read metadata from."
            )
        return self._engine._model._parent

    # ── direct Tier-1 terminal (boolean-op-stable labels) ───
    def to_label(self, name: str) -> "EntitySelection":
        """Register every entity as a **Tier-1 label** (``_label:``).

        Per-dim ``session.labels.add(d, tags, name=name)`` — the
        boolean-op-stable, ``_label:``-prefixed registry (ADR 0015,
        distinct from :meth:`to_physical`'s raw Tier-2 PG).  Behaves
        identically to ``Selection.to_label``, including its multi-dim
        warning suppression (re-using one name across dims is the
        documented intent here, not a mistake).  Returns ``self`` for
        chaining.
        """
        import warnings
        session = self._session()
        dims = sorted({d for d, _ in self._items})
        with warnings.catch_warnings():
            if len(dims) > 1:
                warnings.filterwarnings(
                    "ignore", message=r".*already exists at dim.*",
                )
            for d in dims:
                tags = [t for dim, t in self._items if dim == d]
                session.labels.add(d, tags, name=name)
        return self

    # ── direct Tier-2 terminal (raw physical groups) ────────
    def to_physical(self, name: str) -> "EntitySelection":
        """Register every entity as a **Tier-2 physical group** (raw).

        Per-dim ``session.physical.add(d, tags, name=name)`` — the raw
        gmsh-PG registry (ADR 0015, distinct from :meth:`to_label`'s
        Tier-1 ``_label:`` registry; the two are never merged).
        Behaves identically to ``Selection.to_physical``.  Returns
        ``self`` for chaining.
        """
        session = self._session()
        for d in sorted({d for d, _ in self._items}):
            tags = [t for dim, t in self._items if dim == d]
            session.physical.add(d, tags, name=name)
        return self

    # ── direct dataframe terminal (NEW; no viz import) ──────
    def to_dataframe(self):
        """Return a DataFrame ``dim, tag, kind, label, x, y, z, mass``.

        Direct terminal — the ``core/_selection.Selection`` payload has
        no ``to_dataframe`` (only ``viz/Selection`` does).  Implemented
        **locally** (no ``viz`` import — keeps the import-DAG polarity
        intact, R8): ``kind`` from the session's
        ``model._metadata`` entity registry, ``label`` from the
        session label reverse-map (Tier-1), ``x/y/z`` from the gmsh
        **bounding-box centre**, ``mass`` from ``gmsh.model.occ.getMass``
        (length/area/volume; ``0.0`` for points).  Mirrors
        ``viz/Selection.py``'s column set without importing it.
        """
        import pandas as pd

        session = self._session()
        reg = getattr(session.model, "_metadata", {}) or {}
        label_map: dict = {}
        labels_comp = getattr(session, "labels", None)
        if labels_comp is not None:
            try:
                label_map = labels_comp.reverse_map()
            except Exception:
                label_map = {}

        rows = []
        for d, t in self._items:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                int(d), int(t)
            )
            if int(d) == 0:
                mass = 0.0
            else:
                try:
                    mass = float(gmsh.model.occ.getMass(int(d), int(t)))
                except Exception:
                    mass = float("nan")
            info = reg.get((int(d), int(t)), {})
            rows.append({
                "dim": int(d),
                "tag": int(t),
                "kind": info.get("kind"),
                "label": label_map.get((int(d), int(t)), ""),
                "x": 0.5 * (xmin + xmax),
                "y": 0.5 * (ymin + ymax),
                "z": 0.5 * (zmin + zmax),
                "mass": mass,
            })
        return pd.DataFrame(
            rows,
            columns=["dim", "tag", "kind", "label", "x", "y", "z", "mass"],
        )

    # ── terminal — the Selection payload (R-v2-2) ──
    def result(self) -> "Selection":
        return self._materialize()

    def _materialize(self) -> "Selection":
        """Identity alias → the :class:`Selection` payload.

        Zero-cost (1-line) alias (R-v2-2): ``_engine`` is the
        ``_Queries`` instance so the returned terminal is wired exactly
        like ``m.model.select(...).result()`` — ``.tags()`` /
        ``.to_label`` / ``.to_physical`` / ``.select(on=)`` are all
        available on it.
        """
        return Selection(list(self._items), _queries=self._engine)
