"""User-facing result composites — ``results.nodes``, ``results.elements.gauss``, …

These mirror the ``FEMData`` composite shape: same ``pg=`` / ``label=`` /
``ids=`` selection vocabulary, same per-topology accessors. Each
composite resolves the user's selectors to concrete IDs and
delegates to the bound ``ResultsReader`` for the actual slab read.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from numpy import ndarray

from . import _derived, _expr
from ._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    LocalAxes,
    NodeSlab,
    SpringSlab,
)
from .readers._protocol import ResultLevel, TimeSlice

if TYPE_CHECKING:
    from .Results import Results


# =====================================================================
# Geometric helpers — coordinate / box queries against the bound FEM
# =====================================================================

def _require_fem(results: "Results", method: str):
    fem = results._fem
    if fem is None:
        raise RuntimeError(
            f"{method} requires a bound FEMData. Pass fem= when "
            f"constructing Results, or call .bind(fem)."
        )
    return fem


def _nearest_node_id(fem, point, *, candidate_ids=None) -> int:
    """Return the node ID closest to ``point`` in 3D Euclidean distance.

    If ``candidate_ids`` is provided, restrict the search to that set
    (additive composition with named selectors).
    """
    target = np.asarray(point, dtype=np.float64).reshape(3)
    all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if all_ids.size == 0:
        raise RuntimeError("FEMData has no nodes.")
    if candidate_ids is None:
        cand_idx = np.arange(all_ids.size)
    else:
        cand_idx = np.flatnonzero(np.isin(all_ids, candidate_ids))
        if cand_idx.size == 0:
            raise RuntimeError(
                "Candidate set has no nodes — check pg/label/selection/ids."
            )
    i = cand_idx[int(np.argmin(
        np.linalg.norm(coords[cand_idx] - target, axis=1),
    ))]
    return int(all_ids[i])


def _node_ids_in_box(fem, box_min, box_max) -> ndarray:
    """Return node IDs whose coordinates lie inside the AABB.

    Box is half-open on the upper side:
    ``box_min <= xyz < box_max`` per axis. Coordinates equal to
    ``box_max`` are excluded so adjacent boxes don't double-count a
    shared face.
    """
    lo = np.asarray(box_min, dtype=np.float64).reshape(3)
    hi = np.asarray(box_max, dtype=np.float64).reshape(3)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if coords.size == 0:
        return np.array([], dtype=np.int64)
    mask = np.all((coords >= lo) & (coords < hi), axis=1)
    return np.asarray(fem.nodes.ids, dtype=np.int64)[mask]


def _node_ids_in_sphere(fem, center, radius) -> ndarray:
    """Return node IDs within ``radius`` of ``center`` (closed ball)."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = float(radius)
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {r}.")
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if coords.size == 0:
        return np.array([], dtype=np.int64)
    mask = np.linalg.norm(coords - c, axis=1) <= r
    return np.asarray(fem.nodes.ids, dtype=np.int64)[mask]


def _node_ids_on_plane(fem, point_on_plane, normal, tolerance) -> ndarray:
    """Return node IDs within ``tolerance`` of the plane.

    Plane is defined by a point on it and a normal vector. The normal
    is normalised internally; perpendicular distance is computed via
    the signed dot product.
    """
    p = np.asarray(point_on_plane, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("normal vector has zero length.")
    n_hat = n / n_norm
    tol = float(tolerance)
    if tol < 0:
        raise ValueError(f"tolerance must be non-negative, got {tol}.")
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if coords.size == 0:
        return np.array([], dtype=np.int64)
    distance = np.abs((coords - p) @ n_hat)
    mask = distance <= tol
    return np.asarray(fem.nodes.ids, dtype=np.int64)[mask]


def _element_centroids(fem) -> tuple[ndarray, ndarray]:
    """Return ``(ids, centroids)`` for every element in the FEM.

    Walks every element-type group, computes the centroid as the mean
    of its node coordinates, and concatenates the results. O(E) work
    per call — callers should cache when calling repeatedly.
    """
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    sorted_ids = np.argsort(np.asarray(fem.nodes.ids, dtype=np.int64))
    sorted_node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)[sorted_ids]
    inverse_perm = sorted_ids   # sorted index -> original index

    out_ids: list[ndarray] = []
    out_cent: list[ndarray] = []

    # selection-unification v2 P3-R / §6.3 M-STOP-3: iterate element
    # groups directly — byte-equivalent to the removed per-type
    # ``fem.elements.resolve(element_type=)`` (``_groups`` is one group
    # per type code, same dict order as the old ``fem.elements.types``).
    for grp in fem.elements._groups.values():
        ids, conn = grp.ids, grp.connectivity
        if ids.size == 0:
            continue
        flat = np.asarray(conn, dtype=np.int64).ravel()
        loc = np.searchsorted(sorted_node_ids, flat)
        # Fail loud on any connectivity entry pointing at a node ID
        # absent from the FEM. searchsorted returns len(sorted_node_ids)
        # for an ID past the maximum; for any other missing ID it lands
        # on an in-range slot whose value does not equal the requested
        # ID. Either way the centroid would be silently corrupted.
        in_range = loc < sorted_node_ids.size
        known = np.zeros(flat.shape, dtype=bool)
        known[in_range] = sorted_node_ids[loc[in_range]] == flat[in_range]
        if not known.all():
            bad_flat = int(np.flatnonzero(~known)[0])
            elem_idx = int(np.unravel_index(bad_flat, conn.shape)[0])
            bad_elem = int(np.asarray(ids, dtype=np.int64)[elem_idx])
            bad_node = int(flat[bad_flat])
            raise KeyError(
                f"element {bad_elem} references node {bad_node} which is "
                f"not in the FEM node set — refusing to compute a "
                f"corrupted centroid (fail loud)"
            )
        orig = inverse_perm[loc]
        node_xyz = coords[orig].reshape(conn.shape + (3,))
        cent = node_xyz.mean(axis=1)

        out_ids.append(np.asarray(ids, dtype=np.int64))
        out_cent.append(cent)

    if not out_ids:
        # Fail loud, never silent-empty (resolution-contract Rule 6).
        # Every element-geometry verb (nearest_to / in_box / in_sphere /
        # on_plane, both the legacy helpers and the .select() chain that
        # inherits this contract) funnels through here. A reader-
        # synthesised MPCO FEMData omits element connectivity for any
        # element class MPCO does not write into MODEL/ELEMENTS, so the
        # centroid filter would otherwise return an empty slab with no
        # signal that the geometry was simply unavailable.
        raise RuntimeError(
            "Element-geometry spatial filtering (nearest_to / in_box / "
            "in_sphere / on_plane) needs a bound FEMData that carries "
            "element connectivity, but the bound FEMData has no "
            f"resolvable elements ({sorted_node_ids.size} node(s), 0 "
            "element groups). This is the case for a reader-synthesised "
            "MPCO FEMData whose MODEL/ELEMENTS group does not carry "
            "connectivity for the requested element class (e.g. "
            "ZeroLength springs in models where STKO did not emit them) "
            "— element centroids cannot be computed. Re-attach a "
            "session-built FEMData that carries element geometry via "
            "results.bind(fem)."
        )
    return np.concatenate(out_ids), np.vstack(out_cent)


def _nearest_element_id(fem, point, *, candidate_ids=None) -> int:
    """Return the element ID whose centroid is closest to ``point``.

    If ``candidate_ids`` is provided, restrict the search to that set.
    """
    target = np.asarray(point, dtype=np.float64).reshape(3)
    # _element_centroids fails loud when the bound FEMData carries no
    # resolvable element geometry, so ids is always non-empty here.
    ids, cent = _element_centroids(fem)
    if candidate_ids is None:
        cand_mask = np.ones(ids.size, dtype=bool)
    else:
        cand_mask = np.isin(ids, candidate_ids)
        if not cand_mask.any():
            raise RuntimeError(
                "Candidate set has no elements — check pg/label/selection/ids."
            )
    sub_ids = ids[cand_mask]
    sub_cent = cent[cand_mask]
    i = int(np.argmin(np.linalg.norm(sub_cent - target, axis=1)))
    return int(sub_ids[i])


def _element_ids_in_box(fem, box_min, box_max) -> ndarray:
    """Return element IDs whose centroid lies inside the AABB."""
    lo = np.asarray(box_min, dtype=np.float64).reshape(3)
    hi = np.asarray(box_max, dtype=np.float64).reshape(3)
    ids, cent = _element_centroids(fem)   # fails loud if no element geometry
    mask = np.all((cent >= lo) & (cent < hi), axis=1)
    return ids[mask]


def _element_ids_in_sphere(fem, center, radius) -> ndarray:
    """Return element IDs whose centroid is within ``radius`` of ``center``."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = float(radius)
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {r}.")
    ids, cent = _element_centroids(fem)   # fails loud if no element geometry
    mask = np.linalg.norm(cent - c, axis=1) <= r
    return ids[mask]


def _element_ids_on_plane(fem, point_on_plane, normal, tolerance) -> ndarray:
    """Return element IDs whose centroid is within ``tolerance`` of the plane."""
    p = np.asarray(point_on_plane, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("normal vector has zero length.")
    n_hat = n / n_norm
    tol = float(tolerance)
    if tol < 0:
        raise ValueError(f"tolerance must be non-negative, got {tol}.")
    ids, cent = _element_centroids(fem)   # fails loud if no element geometry
    distance = np.abs((cent - p) @ n_hat)
    mask = distance <= tol
    return ids[mask]


def _element_ids_of_type(fem, element_type: str) -> ndarray:
    """Return element IDs matching the given broker element-type name.

    The element-type name is the broker's label (e.g. ``"Tet4"``,
    ``"Hex8"``, ``"Quad4"``) — see ``fem.elements.types``. If the
    name doesn't match any group, returns an empty array.
    """
    available = [t.name for t in fem.elements.types]
    if element_type not in available:
        return np.array([], dtype=np.int64)
    # selection-unification v2 P3-R / §6.3 M-STOP-3: direct group
    # lookup — byte-equivalent to the removed
    # ``fem.elements.resolve(element_type=)`` (the guard above already
    # restricts to a broker type name; ``_groups`` is one group per
    # type, 1:1 and same order as ``fem.elements.types``).
    for t, grp in zip(fem.elements.types, fem.elements._groups.values()):
        if t.name == element_type:
            return np.asarray(grp.ids, dtype=np.int64)
    return np.array([], dtype=np.int64)


# =====================================================================
# Selection helpers
# =====================================================================

class _SelectionMixin:
    """Shared PG / label / selection / ids resolution for composites."""

    _r: "Results"

    def _resolve_node_ids(
        self,
        *,
        pg: str | Iterable[str] | None,
        label: str | Iterable[str] | None,
        selection: str | Iterable[str] | None,
        ids: Iterable[int] | ndarray | None,
    ) -> Optional[ndarray]:
        """Return a node ID array, or None for 'all nodes'.

        Selection vocabulary mirrors ``FEMData.nodes.get()``:
        ``pg=`` (physical groups), ``label=`` (apeGmsh labels),
        ``selection=`` (post-mesh ``g.mesh_selection`` sets), or
        ``ids=`` (raw IDs). Provide at most one.
        """
        named = [x for x in (pg, label, selection) if x is not None]
        if ids is not None:
            if named:
                raise ValueError(
                    "Provide one of pg=, label=, selection=, or ids= "
                    "(not multiple)."
                )
            return np.asarray(list(ids), dtype=np.int64)

        if not named:
            return None     # all nodes

        fem = self._r._fem
        if fem is None:
            raise RuntimeError(
                "Cannot resolve pg= / label= / selection= without a bound "
                "FEMData. Pass fem= when constructing Results, or "
                "call .bind(fem)."
            )

        all_ids: list[ndarray] = []
        if pg is not None:
            for name in _as_iter(pg):
                all_ids.append(np.asarray(
                    fem.nodes.physical.node_ids(name), dtype=np.int64,
                ))
        if label is not None:
            for name in _as_iter(label):
                all_ids.append(np.asarray(
                    fem.nodes.labels.node_ids(name), dtype=np.int64,
                ))
        if selection is not None:
            store = getattr(fem, "mesh_selection", None)
            if store is None:
                raise RuntimeError(
                    "selection= requires fem.mesh_selection to be present "
                    "(it's captured at get_fem_data() time when the "
                    "session has a g.mesh_selection composite)."
                )
            for name in _as_iter(selection):
                all_ids.append(store.node_ids(name))
        if not all_ids:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(all_ids))

    def _resolve_element_ids(
        self,
        *,
        pg: str | Iterable[str] | None,
        label: str | Iterable[str] | None,
        selection: str | Iterable[str] | None,
        ids: Iterable[int] | ndarray | None,
    ) -> Optional[ndarray]:
        """Return an element ID array, or None for 'all elements'."""
        named = [x for x in (pg, label, selection) if x is not None]
        if ids is not None:
            if named:
                raise ValueError(
                    "Provide one of pg=, label=, selection=, or ids= "
                    "(not multiple)."
                )
            return np.asarray(list(ids), dtype=np.int64)

        if not named:
            return None

        fem = self._r._fem
        if fem is None:
            raise RuntimeError(
                "Cannot resolve pg= / label= / selection= without a bound "
                "FEMData. Pass fem= when constructing Results, or "
                "call .bind(fem)."
            )

        all_ids: list[ndarray] = []
        if pg is not None:
            for name in _as_iter(pg):
                all_ids.append(np.asarray(
                    fem.elements.physical.element_ids(name), dtype=np.int64,
                ))
        if label is not None:
            for name in _as_iter(label):
                all_ids.append(np.asarray(
                    fem.elements.labels.element_ids(name), dtype=np.int64,
                ))
        if selection is not None:
            store = getattr(fem, "mesh_selection", None)
            if store is None:
                raise RuntimeError(
                    "selection= requires fem.mesh_selection to be present."
                )
            for name in _as_iter(selection):
                all_ids.append(store.element_ids(name))
        if not all_ids:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(all_ids))


def _as_iter(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    return list(x)


class _ElementGeometryMixin:
    """Geometric helpers for element-level composites.

    Provides ``nearest_to`` / ``in_box`` / ``in_sphere`` / ``on_plane``,
    each accepting the standard selectors (``pg=`` / ``label=`` /
    ``selection=`` / ``ids=`` / ``element_type=``) for **additive**
    composition: the named selectors restrict the candidate set first,
    then the geometric filter narrows that subset. Distance and
    containment are computed against element centroids (mean of the
    element's node coordinates).
    """

    _r: "Results"

    # ------------------------------------------------------------------
    # Internal — combine named selectors plus element_type
    # ------------------------------------------------------------------

    def _combine_candidates(
        self,
        *,
        pg, label, selection, ids,
        element_type: str | None,
    ):
        """Return the candidate ID array (or None for "all elements")."""
        cand = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        if element_type is not None:
            fem = _require_fem(self._r, "element_type=")
            type_ids = _element_ids_of_type(fem, element_type)
            cand = (
                np.intersect1d(cand, type_ids, assume_unique=False)
                if cand is not None else type_ids
            )
        return cand

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    def nearest_to(
        self,
        point,
        *,
        component: str,
        pg=None, label=None, selection=None, ids=None,
        element_type: str | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ):
        """Read ``component`` at the element whose centroid is nearest ``point``."""
        fem = _require_fem(
            self._r, f"{type(self).__name__}.nearest_to(...)",
        )
        cand = self._combine_candidates(
            pg=pg, label=label, selection=selection, ids=ids,
            element_type=element_type,
        )
        eid = _nearest_element_id(fem, point, candidate_ids=cand)
        return self.get(
            ids=[eid], component=component, time=time, stage=stage,
        )

    def in_box(
        self,
        box_min,
        box_max,
        *,
        component: str,
        pg=None, label=None, selection=None, ids=None,
        element_type: str | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ):
        """Read ``component`` at every element whose centroid is in ``[box_min, box_max)``."""
        fem = _require_fem(
            self._r, f"{type(self).__name__}.in_box(...)",
        )
        cand = self._combine_candidates(
            pg=pg, label=label, selection=selection, ids=ids,
            element_type=element_type,
        )
        in_box_ids = _element_ids_in_box(fem, box_min, box_max)
        narrowed = (
            np.intersect1d(cand, in_box_ids, assume_unique=False)
            if cand is not None else in_box_ids
        )
        return self.get(
            ids=narrowed, component=component, time=time, stage=stage,
        )

    def in_sphere(
        self,
        center,
        radius: float,
        *,
        component: str,
        pg=None, label=None, selection=None, ids=None,
        element_type: str | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ):
        """Read ``component`` at every element whose centroid is within ``radius`` of ``center``."""
        fem = _require_fem(
            self._r, f"{type(self).__name__}.in_sphere(...)",
        )
        cand = self._combine_candidates(
            pg=pg, label=label, selection=selection, ids=ids,
            element_type=element_type,
        )
        in_sphere_ids = _element_ids_in_sphere(fem, center, radius)
        narrowed = (
            np.intersect1d(cand, in_sphere_ids, assume_unique=False)
            if cand is not None else in_sphere_ids
        )
        return self.get(
            ids=narrowed, component=component, time=time, stage=stage,
        )

    def on_plane(
        self,
        point_on_plane,
        normal,
        tolerance: float,
        *,
        component: str,
        pg=None, label=None, selection=None, ids=None,
        element_type: str | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ):
        """Read ``component`` at every element whose centroid is within ``tolerance`` of the plane."""
        fem = _require_fem(
            self._r, f"{type(self).__name__}.on_plane(...)",
        )
        cand = self._combine_candidates(
            pg=pg, label=label, selection=selection, ids=ids,
            element_type=element_type,
        )
        on_plane_ids = _element_ids_on_plane(
            fem, point_on_plane, normal, tolerance,
        )
        narrowed = (
            np.intersect1d(cand, on_plane_ids, assume_unique=False)
            if cand is not None else on_plane_ids
        )
        return self.get(
            ids=narrowed, component=component, time=time, stage=stage,
        )

    # ------------------------------------------------------------------
    # Daisy-chainable selection (point family, element level)
    # ------------------------------------------------------------------

    def select(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        element_type: str | None = None,
    ):
        """Select a subset of elements for result extraction
        (sub-composite level).

        Shared by ``results.elements.gauss``, ``.fibers``,
        ``.layers``, ``.line_stations``, and ``.springs``.
        Spatial verbs test element **centroids**::

            # fiber stress at the first GP, in a spatial region
            slab = (results.elements.fibers.select(pg="Cols")
                        .in_box(lo, hi)
                        .values(component="fiber_stress",
                                gp_indices=[0]))

            # bending moment along beams in a region
            slab = (results.elements.line_stations.select(pg="Beams")
                        .in_box(lo, hi)
                        .values(component="bending_moment_y"))

        No arguments seeds every domain element.

        Parameters
        ----------
        pg :
            Physical group name or list of names.
        label :
            Geometry-time label name or list.
        selection :
            Named mesh-selection set.
        ids :
            Explicit element id list.
        element_type :
            Restrict to a specific element type by name (e.g.
            ``"tet4"``) or Gmsh type code (int).

        Refining verbs
        --------------
        Each returns a new ``MeshSelection`` and composes freely
        (spatial verbs test element **centroids**):

        - ``.in_box(lo, hi, *, inclusive=False)``
        - ``.in_sphere(center, radius)``
        - ``.on_plane(point, normal, *, tol)`` — ``tol=`` is
          **required**; raises ``TypeError`` if omitted.
        - ``.nearest_to(point, *, count=1)``
        - ``.where(predicate)``
        - ``|`` ``&`` ``-`` ``^`` (set algebra).

        Terminal
        --------
        ``.values(component, *, time=None, stage=None, **extra)`` —
        reads the result slab for the selected elements.  Extra kwargs
        (e.g. ``gp_indices=`` for fibers, ``layer_indices=`` for
        layers) are forwarded to the sub-composite's ``.get`` reader.
        """
        # selection-unification-v2 P2-I (§6.1 STOP-2): return the v2
        # terminal ``MeshSelection`` (legacy ``ResultChain`` left
        # defined-but-unwired; P3 deletes it).  ``engine_for`` is kept
        # verbatim — the memoised ``_ResultChainEngine`` singleton is
        # what ``SelectionChain._compatible`` gates set-algebra on, and
        # ``MeshSelection`` delegates the per-engine read back to a
        # ``ResultChain`` built from this exact same engine (byte-faithful
        # results-engine behaviour).  ``MeshSelection`` is a new leaf
        # importing only ``_kernel.chain`` at load — no new eager
        # cross-package edge (one declared downward BASELINE triple).
        from ._result_engine import engine_for
        from ..mesh._mesh_selection import MeshSelection

        seed = self._combine_candidates(
            pg=pg, label=label, selection=selection, ids=ids,
            element_type=element_type,
        )
        if seed is None:                 # no selector → every domain element
            fem = _require_fem(
                self._r, f"{type(self).__name__}.select(...)",
            )
            atoms = [int(e) for e in np.asarray(fem.elements.ids)]
        else:
            atoms = [int(e) for e in np.asarray(seed)]
        engine = engine_for(self._r, self, "element")
        return MeshSelection(atoms, _engine=engine)


# =====================================================================
# Nodes
# =====================================================================

class _CustomScalarMixin:
    """User-defined scalar expressions on a result composite (ADR 0076).

    Shared ``define`` / ``undefine`` / ``definitions`` surface plus the
    on-read evaluation, for the node and Gauss composites. The host must
    provide ``self._r`` (the Results), initialize ``self._registry`` (a
    ``dict[str, _expr.ExprDef]``) in ``__init__``, and implement
    ``_reader_components(stage=...)`` returning the stored + derived
    component names (no custom names) for a stage — the current
    ``available_components`` body.

    A custom scalar is evaluated through the host's own ``get()`` for
    each operand (so a custom scalar may reference stored, derived, or
    earlier custom names, all read for the identical selection / stage /
    time), then combined by :func:`_expr.evaluate`.
    """

    _registry: "dict[str, _expr.ExprDef]"

    def define(
        self, name: str, expr: str, *,
        label: str | None = None, units: str | None = None,
    ) -> "_expr.ExprDef":
        """Register a custom scalar ``name`` = ``expr`` on this composite.

        ``expr`` is a restricted arithmetic expression over this
        composite's ``available_components()`` (stored, derived, or
        earlier custom names). It becomes a first-class ``component=``
        everywhere this composite reads, and appears in
        ``available_components()`` / the viewer picker. ``label`` /
        ``units`` are display-only. Raises on a bad expression, an
        unknown operand, or a name that shadows an existing component
        (:func:`_expr.ExprError` / ``ValueError``).
        """
        name = str(name)
        reserved = self._union_reader_components()
        if name in reserved:
            raise ValueError(
                f"'{name}' shadows a stored or derived component; pick "
                f"another name."
            )
        if name in self._registry:
            raise ValueError(
                f"'{name}' is already defined on this composite; "
                f"undefine('{name}') first."
            )
        resolvable = reserved | set(self._registry)
        exprdef = _expr.compile_expr(
            name, expr, available=resolvable, label=label, units=units,
        )
        if not exprdef.operands:
            raise ValueError(
                f"'{name}' references no recorded component, so there is no "
                f"point set to evaluate it over; reference at least one "
                f"available component."
            )
        self._registry[name] = exprdef
        return exprdef

    def undefine(self, name: str) -> None:
        """Remove a custom scalar. Raises if another definition depends on it."""
        if name not in self._registry:
            raise KeyError(f"no custom scalar named '{name}' on this composite.")
        dependents = sorted(
            d.name for d in self._registry.values() if name in d.operands
        )
        if dependents:
            raise ValueError(
                f"cannot undefine '{name}': {dependents} depend on it. "
                f"Remove them first."
            )
        del self._registry[name]

    @property
    def definitions(self) -> "dict[str, _expr.ExprDef]":
        """The custom scalars registered on this composite (name → ExprDef)."""
        return dict(self._registry)

    def available_components(self, *, stage: str | None = None) -> list[str]:
        base = list(self._reader_components(stage=stage))
        extra = [n for n in self._registry if n not in base]
        return base + extra

    def _union_reader_components(self) -> set[str]:
        """Stored + derived components across *all* stages (define-time
        validation namespace — see ADR 0076 INV-3)."""
        out: set[str] = set()
        for s in self._r._all_stages():
            out |= set(self._reader_components(stage=s.id))
        return out

    def _reader_components(self, *, stage: str | None = None) -> list[str]:
        raise NotImplementedError  # provided by the host composite

    def _evaluate_custom(self, name: str, read_operand):
        """Evaluate custom scalar ``name``; ``read_operand(comp)`` returns
        the slab for one operand under the caller's selection."""
        exprdef = self._registry[name]
        base_slabs = {op: read_operand(op) for op in exprdef.operands}
        namespace = {op: slab.values for op, slab in base_slabs.items()}
        values = _expr.evaluate(exprdef, namespace)
        template = next(iter(base_slabs.values()))
        return dataclasses.replace(template, component=name, values=values)


class NodeResultsComposite(_CustomScalarMixin, _SelectionMixin):
    """``results.nodes`` — node-level result access."""

    def __init__(self, results: "Results") -> None:
        self._r = results
        self._registry: "dict[str, _expr.ExprDef]" = {}

    def select(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
    ):
        """Select a subset of nodes for result extraction.

        Returns a :class:`~apeGmsh.mesh._mesh_selection.MeshSelection`
        (point family) that chains spatial-refinement verbs and then
        reads data with ``.values(component=...)``::

            # displacement at base nodes in a spatial region
            slab = (results.nodes.select(pg="Base")
                        .in_box(lo, hi)
                        .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
                        .values(component="displacement_x"))
            slab.values   # ndarray (N_nodes, N_steps)
            slab.node_ids # list[int]

            # set algebra across groups
            top = (results.nodes.select(pg="TopFlange")
                   | results.nodes.select(pg="TopPlate"))

        No arguments seeds every domain node.

        Parameters
        ----------
        pg :
            Physical group name or list of names.
        label :
            Geometry-time label name or list.  Labels survive
            boolean operations.
        selection :
            Named mesh-selection set (registered via
            ``g.mesh_selection``).
        ids :
            Explicit node id list.  When given, other selectors
            are ignored.

        Refining verbs
        --------------
        Each returns a new ``MeshSelection`` and composes freely:

        - ``.in_box(lo, hi, *, inclusive=False)``
        - ``.in_sphere(center, radius)``
        - ``.on_plane(point, normal, *, tol)`` — ``tol=`` is
          **required**; raises ``TypeError`` if omitted.
        - ``.nearest_to(point, *, count=1)``
        - ``.where(predicate)``
        - ``|`` ``&`` ``-`` ``^`` (set algebra).

        Terminal
        --------
        ``.values(component, *, time=None, stage=None)`` — reads the
        result slab for the selected nodes; returns a
        :class:`~apeGmsh.results.NodeSlab`.
        """
        # selection-unification-v2 P2-I (§6.1 STOP-2): return the v2
        # terminal ``MeshSelection``; ``engine_for`` kept verbatim (the
        # ``_ResultChainEngine`` singleton gates set-algebra identity);
        # ``MeshSelection`` delegates the node-level read back to a
        # ``ResultChain`` on this same engine (byte-faithful).
        from ._result_engine import engine_for
        from ..mesh._mesh_selection import MeshSelection

        seed = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        if seed is None:                 # no selector → every domain node
            fem = _require_fem(self._r, "results.nodes.select(...)")
            atoms = [int(n) for n in np.asarray(fem.nodes.ids)]
        else:
            atoms = [int(n) for n in np.asarray(seed)]
        engine = engine_for(self._r, self, "node")
        return MeshSelection(atoms, _engine=engine)

    def get(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        component: str,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> NodeSlab:
        if component in self._registry:
            return self._evaluate_custom(
                component,
                lambda op: self.get(
                    pg=pg, label=label, selection=selection, ids=ids,
                    component=op, time=time, stage=stage,
                ),
            )
        sid = self._r._resolve_stage(stage)
        node_ids = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        return self._r._reader.read_nodes(
            sid, component, node_ids=node_ids, time_slice=time,
        )

    def _reader_components(self, *, stage: str | None = None) -> list[str]:
        sid = self._r._resolve_stage(stage)
        return self._r._reader.available_components(sid, ResultLevel.NODES)

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    def nearest_to(
        self,
        point,
        *,
        component: str,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> NodeSlab:
        """Read ``component`` at the node nearest ``point`` (3D coord)."""
        fem = _require_fem(self._r, "results.nodes.nearest_to(...)")
        candidate = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        nid = _nearest_node_id(fem, point, candidate_ids=candidate)
        return self.get(
            ids=[nid], component=component, time=time, stage=stage,
        )

    def in_box(
        self,
        box_min,
        box_max,
        *,
        component: str,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> NodeSlab:
        """Read ``component`` at every node inside ``[box_min, box_max)``.

        Box bounds are half-open on the upper side so adjacent boxes
        don't double-count shared faces. Use ``-np.inf`` / ``np.inf``
        to relax an axis.
        """
        fem = _require_fem(self._r, "results.nodes.in_box(...)")
        candidate = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        in_box_ids = _node_ids_in_box(fem, box_min, box_max)
        narrowed = (
            np.intersect1d(candidate, in_box_ids, assume_unique=False)
            if candidate is not None else in_box_ids
        )
        return self.get(
            ids=narrowed, component=component, time=time, stage=stage,
        )

    def in_sphere(
        self,
        center,
        radius: float,
        *,
        component: str,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> NodeSlab:
        """Read ``component`` at every node within ``radius`` of ``center``."""
        fem = _require_fem(self._r, "results.nodes.in_sphere(...)")
        candidate = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        in_sphere_ids = _node_ids_in_sphere(fem, center, radius)
        narrowed = (
            np.intersect1d(candidate, in_sphere_ids, assume_unique=False)
            if candidate is not None else in_sphere_ids
        )
        return self.get(
            ids=narrowed, component=component, time=time, stage=stage,
        )

    def on_plane(
        self,
        point_on_plane,
        normal,
        tolerance: float,
        *,
        component: str,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> NodeSlab:
        """Read ``component`` at every node within ``tolerance`` of the plane.

        The plane is defined by a point on it and a normal vector.
        Useful for slicing through a model — e.g. mid-span or a
        story-level cut.
        """
        fem = _require_fem(self._r, "results.nodes.on_plane(...)")
        candidate = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        on_plane_ids = _node_ids_on_plane(
            fem, point_on_plane, normal, tolerance,
        )
        narrowed = (
            np.intersect1d(candidate, on_plane_ids, assume_unique=False)
            if candidate is not None else on_plane_ids
        )
        return self.get(
            ids=narrowed, component=component, time=time, stage=stage,
        )


# =====================================================================
# Elements (per-element-node forces)
# =====================================================================

class ElementResultsComposite(_SelectionMixin, _ElementGeometryMixin):
    """``results.elements`` — per-element-node forces, plus sub-composites."""

    def __init__(self, results: "Results") -> None:
        self._r = results
        self.gauss = GaussResultsComposite(results)
        self.fibers = FibersResultsComposite(results)
        self.layers = LayersResultsComposite(results)
        self.line_stations = LineStationsResultsComposite(results)
        self.springs = SpringsResultsComposite(results)

    def select(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        element_type: str | None = None,
    ):
        """Select a subset of elements for result extraction.

        Returns a :class:`~apeGmsh.mesh._mesh_selection.MeshSelection`
        (point family; spatial verbs test element **centroids**) that
        chains refinement verbs and then reads data with
        ``.values(component=...)``::

            # global forces on beams in a spatial region
            slab = (results.elements.select(pg="Beams")
                        .in_box(lo, hi)
                        .values(component="globalForce"))
            slab.values     # ndarray (N_elems, N_steps)

            # set algebra
            cols = (results.elements.select(pg="ColsA")
                    | results.elements.select(pg="ColsB"))

        No arguments seeds every domain element.

        Parameters
        ----------
        pg :
            Physical group name or list of names.
        label :
            Geometry-time label name or list.
        selection :
            Named mesh-selection set.
        ids :
            Explicit element id list.
        element_type :
            Restrict to a specific element type by name (e.g.
            ``"tet4"``) or Gmsh type code (int).

        Refining verbs
        --------------
        Each returns a new ``MeshSelection`` and composes freely
        (spatial verbs test element **centroids**):

        - ``.in_box(lo, hi, *, inclusive=False)``
        - ``.in_sphere(center, radius)``
        - ``.on_plane(point, normal, *, tol)`` — ``tol=`` is
          **required**; raises ``TypeError`` if omitted.
        - ``.nearest_to(point, *, count=1)``
        - ``.where(predicate)``
        - ``|`` ``&`` ``-`` ``^`` (set algebra).

        Terminal
        --------
        ``.values(component, *, time=None, stage=None)`` — reads the
        result slab for the selected elements; returns an
        :class:`~apeGmsh.results.ElementSlab`.
        """
        # selection-unification-v2 P2-I (§6.1 STOP-2): return the v2
        # terminal ``MeshSelection``; ``engine_for`` kept verbatim (the
        # ``_ResultChainEngine`` singleton gates set-algebra identity).
        from ._result_engine import engine_for
        from ..mesh._mesh_selection import MeshSelection

        seed = self._combine_candidates(
            pg=pg, label=label, selection=selection, ids=ids,
            element_type=element_type,
        )
        if seed is None:                 # no selector → every domain element
            fem = _require_fem(self._r, "results.elements.select(...)")
            atoms = [int(e) for e in np.asarray(fem.elements.ids)]
        else:
            atoms = [int(e) for e in np.asarray(seed)]
        engine = engine_for(self._r, self, "element")
        return MeshSelection(atoms, _engine=engine)

    def get(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        component: str,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> ElementSlab:
        """Per-element-node values (``globalForce`` / ``localForce``)."""
        sid = self._r._resolve_stage(stage)
        eids = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        return self._r._reader.read_elements(
            sid, component, element_ids=eids, time_slice=time,
        )

    def available_components(self, *, stage: str | None = None) -> list[str]:
        sid = self._r._resolve_stage(stage)
        return self._r._reader.available_components(sid, ResultLevel.ELEMENTS)

    def local_axes(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        stage: str | None = None,
    ) -> "LocalAxes":
        """Per-element local frames (beam / shell orientation) from a
        ``.ladruno`` ``MODEL/LOCAL_AXES``.

        This is the orientation a ``.mpco`` does **not** carry for
        beam-columns: returns a :class:`LocalAxes` with per-element
        scalar-first quaternions and ``.matrices`` / ``.x_axis`` /
        ``.y_axis`` / ``.z_axis`` (the local axes are the **rows** of each
        matrix, in global coordinates).

        No selector → every recorded frame. Requires a Ladruno results
        object (raises :class:`TypeError` otherwise — MPCO / native beam
        frames are not surfaced here).
        """
        read = getattr(self._r._reader, "read_local_axes", None)
        if read is None:
            raise TypeError(
                "results.elements.local_axes(...) requires a .ladruno "
                "(MODEL/LOCAL_AXES). Open one via Results.from_ladruno(...); "
                "MPCO / native beam frames are not surfaced here."
            )
        sid = self._r._resolve_stage(stage)
        eids = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        return read(sid, element_ids=eids)


# =====================================================================
# Gauss points
# =====================================================================

class GaussResultsComposite(
    _CustomScalarMixin, _SelectionMixin, _ElementGeometryMixin,
):
    """``results.elements.gauss`` — continuum Gauss-point values."""

    def __init__(self, results: "Results") -> None:
        self._r = results
        self._registry: "dict[str, _expr.ExprDef]" = {}

    def get(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        component: str,
        time: TimeSlice = None,
        stage: str | None = None,
        plane: str | None = "auto",
        nu: float | None = None,
        thickness: float | None = None,
    ) -> GaussSlab:
        if component in self._registry:
            return self._evaluate_custom(
                component,
                lambda op: self.get(
                    pg=pg, label=label, selection=selection, ids=ids,
                    component=op, time=time, stage=stage,
                    plane=plane, nu=nu, thickness=thickness,
                ),
            )
        sid = self._r._resolve_stage(stage)
        eids = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        if _derived.is_shell_derived(component):
            return self._compute_shell_derived(
                component, sid, eids, time, thickness=thickness,
            )
        if _derived.is_derived(component):
            return self._compute_derived(
                component, sid, eids, time, plane=plane, nu=nu,
            )
        return self._r._reader.read_gauss(
            sid, component, element_ids=eids, time_slice=time,
        )

    def _compute_shell_derived(
        self, component: str, sid, eids, time: TimeSlice,
        *, thickness: float | None,
    ) -> GaussSlab:
        """Compute a shell-resultant derived scalar (needs ``thickness``)."""
        if thickness is None:
            raise ValueError(
                f"'{component}' needs the shell thickness — pass "
                f"thickness=<t> to results.elements.gauss.get(...)."
            )
        base = _derived.shell_base_components(component)
        stored = set(
            self._r._reader.available_components(sid, ResultLevel.GAUSS)
        )
        missing = [b for b in base if b not in stored]
        if missing:
            raise ValueError(
                f"Cannot compute '{component}': missing shell resultants "
                f"{missing} (available: {sorted(stored)}). Record the shell "
                f"stress resultants (component='stress' on shell elements)."
            )
        base_slabs = {
            name: self._r._reader.read_gauss(
                sid, name, element_ids=eids, time_slice=time,
            )
            for name in base
        }
        columns = {name: slab.values for name, slab in base_slabs.items()}
        values = _derived.compute_shell(component, columns, thickness=thickness)
        return dataclasses.replace(
            base_slabs[base[0]], component=component, values=values,
        )

    def _compute_derived(
        self, component: str, sid, eids, time: TimeSlice,
        *, plane: str | None = "auto", nu: float | None = None,
    ) -> GaussSlab:
        """Compute a derived scalar (von Mises, principal, …) on read.

        Reads the stored tensor columns for the same selection, assembles
        them, and returns a synthesized slab. For 2-D data the out-of-plane
        component is recovered per ``plane`` / ``nu``:

        * ``"auto"`` (default) — read each element's plane type + ν from
          the model and recover σ_zz / ε_zz *per element* (falls back to
          zero where the model can't be parsed).
        * ``"strain"`` / ``"stress"`` — force that idealization globally
          (``nu`` required for the nonzero fill).
        * ``None`` — leave the out-of-plane component at zero.
        """
        base = _derived.base_components_for(component, ndm=3)
        stored = set(
            self._r._reader.available_components(sid, ResultLevel.GAUSS)
        )
        present = [b for b in base if b in stored]
        if not present:
            raise ValueError(
                f"Cannot compute '{component}': none of the required raw "
                f"tensor components {list(base)} are stored in this results "
                f"file (available: {sorted(stored)}). Derived stress/strain "
                f"scalars need the raw tensor recorded (component='stress' / "
                f"'strain')."
            )
        base_slabs = {
            name: self._r._reader.read_gauss(
                sid, name, element_ids=eids, time_slice=time,
            )
            for name in present
        }
        columns = {name: slab.values for name, slab in base_slabs.items()}
        eff_plane = plane
        if plane == "auto":
            # Elastic out-of-plane recovery applies to the stress / total-
            # strain tensors only. The plastic-strain tensor has its own
            # (material-dependent) out-of-plane rule, so it is left as
            # stored (σ_zzᵖ = 0 in 2-D unless recorded).
            if not base[0].startswith("plastic_strain"):
                from . import _plane_recovery
                prefix = "strain" if base[0].startswith("strain") else "stress"
                _plane_recovery.inject_out_of_plane(
                    columns, base_slabs[present[0]].element_index,
                    prefix=prefix, model=self._r.model,
                )
            eff_plane = None      # zz now supplied per element (or absent → 0)
        values = _derived.compute(
            component, columns, ndm=3, plane=eff_plane, nu=nu,
        )
        return dataclasses.replace(
            base_slabs[present[0]], component=component, values=values,
        )

    def _reader_components(self, *, stage: str | None = None) -> list[str]:
        sid = self._r._resolve_stage(stage)
        stored = list(
            self._r._reader.available_components(sid, ResultLevel.GAUSS)
        )
        derived = [
            d for d in _derived.available_derived(stored) if d not in stored
        ]
        return stored + derived


# =====================================================================
# Fibers
# =====================================================================

class FibersResultsComposite(_SelectionMixin, _ElementGeometryMixin):
    """``results.elements.fibers`` — fiber values within fiber-section GPs."""

    def __init__(self, results: "Results") -> None:
        self._r = results

    def get(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        gp_indices: Iterable[int] | ndarray | None = None,
        component: str,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> FiberSlab:
        sid = self._r._resolve_stage(stage)
        eids = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        gpi = (
            None if gp_indices is None
            else np.asarray(list(gp_indices), dtype=np.int64)
        )
        return self._r._reader.read_fibers(
            sid, component,
            element_ids=eids, gp_indices=gpi, time_slice=time,
        )

    def available_components(self, *, stage: str | None = None) -> list[str]:
        sid = self._r._resolve_stage(stage)
        return self._r._reader.available_components(sid, ResultLevel.FIBERS)


# =====================================================================
# Layers
# =====================================================================

class LayersResultsComposite(_SelectionMixin, _ElementGeometryMixin):
    """``results.elements.layers`` — layered shell layer values."""

    def __init__(self, results: "Results") -> None:
        self._r = results

    def get(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        gp_indices: Iterable[int] | ndarray | None = None,
        layer_indices: Iterable[int] | ndarray | None = None,
        component: str,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> LayerSlab:
        sid = self._r._resolve_stage(stage)
        eids = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        gpi = (
            None if gp_indices is None
            else np.asarray(list(gp_indices), dtype=np.int64)
        )
        lyr = (
            None if layer_indices is None
            else np.asarray(list(layer_indices), dtype=np.int64)
        )
        return self._r._reader.read_layers(
            sid, component,
            element_ids=eids, gp_indices=gpi, layer_indices=lyr,
            time_slice=time,
        )

    def available_components(self, *, stage: str | None = None) -> list[str]:
        sid = self._r._resolve_stage(stage)
        return self._r._reader.available_components(sid, ResultLevel.LAYERS)


# =====================================================================
# Line stations
# =====================================================================

class LineStationsResultsComposite(_SelectionMixin, _ElementGeometryMixin):
    """``results.elements.line_stations`` — beam section forces along the length."""

    def __init__(self, results: "Results") -> None:
        self._r = results

    def get(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        component: str,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> LineStationSlab:
        sid = self._r._resolve_stage(stage)
        eids = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        return self._r._reader.read_line_stations(
            sid, component, element_ids=eids, time_slice=time,
        )

    def available_components(self, *, stage: str | None = None) -> list[str]:
        sid = self._r._resolve_stage(stage)
        return self._r._reader.available_components(
            sid, ResultLevel.LINE_STATIONS,
        )


# =====================================================================
# Springs (ZeroLength)
# =====================================================================

class SpringsResultsComposite(_SelectionMixin, _ElementGeometryMixin):
    """``results.elements.springs`` — ZeroLength spring force / deformation."""

    def __init__(self, results: "Results") -> None:
        self._r = results

    def get(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        component: str,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> SpringSlab:
        """Return spring force or deformation for one spring direction.

        Parameters
        ----------
        component
            Canonical name such as ``"spring_force_0"`` (force in the
            first configured spring direction) or
            ``"spring_deformation_2"`` (deformation in the third).
            Use ``available_components()`` to discover what the file
            contains.
        """
        sid = self._r._resolve_stage(stage)
        eids = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        return self._r._reader.read_springs(
            sid, component, element_ids=eids, time_slice=time,
        )

    def available_components(self, *, stage: str | None = None) -> list[str]:
        sid = self._r._resolve_stage(stage)
        return self._r._reader.available_components(sid, ResultLevel.SPRINGS)
