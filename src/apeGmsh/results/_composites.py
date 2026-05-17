"""User-facing result composites — ``results.nodes``, ``results.elements.gauss``, …

These mirror the ``FEMData`` composite shape: same ``pg=`` / ``label=`` /
``ids=`` selection vocabulary, same per-topology accessors. Each
composite resolves the user's selectors to concrete IDs and
delegates to the bound ``ResultsReader`` for the actual slab read.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from numpy import ndarray

from ._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
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

    for type_info in fem.elements.types:
        ids, conn = fem.elements.resolve(element_type=type_info.name)
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
        return np.array([], dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    return np.concatenate(out_ids), np.vstack(out_cent)


def _nearest_element_id(fem, point, *, candidate_ids=None) -> int:
    """Return the element ID whose centroid is closest to ``point``.

    If ``candidate_ids`` is provided, restrict the search to that set.
    """
    target = np.asarray(point, dtype=np.float64).reshape(3)
    ids, cent = _element_centroids(fem)
    if ids.size == 0:
        raise RuntimeError("FEMData has no elements.")
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
    ids, cent = _element_centroids(fem)
    if ids.size == 0:
        return np.array([], dtype=np.int64)
    mask = np.all((cent >= lo) & (cent < hi), axis=1)
    return ids[mask]


def _element_ids_in_sphere(fem, center, radius) -> ndarray:
    """Return element IDs whose centroid is within ``radius`` of ``center``."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = float(radius)
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {r}.")
    ids, cent = _element_centroids(fem)
    if ids.size == 0:
        return np.array([], dtype=np.int64)
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
    ids, cent = _element_centroids(fem)
    if ids.size == 0:
        return np.array([], dtype=np.int64)
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
    ids, _ = fem.elements.resolve(element_type=element_type)
    return np.asarray(ids, dtype=np.int64)


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


# =====================================================================
# Nodes
# =====================================================================

class NodeResultsComposite(_SelectionMixin):
    """``results.nodes`` — node-level result access."""

    def __init__(self, results: "Results") -> None:
        self._r = results

    def select(
        self,
        *,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
    ):
        """Start a daisy-chainable node-result selection.

        Returns a :class:`~apeGmsh.results._result_chain.ResultChain`
        (point family, node level) that composes fluently and is then
        read with a terminal ``.get(component=...)``::

            (results.nodes.select(pg="Base")
                 .in_box(lo, hi)
                 .on_plane(p, n, tol=1e-6)
                 .get(component="displacement_x"))
            results.nodes.select(ids=a) | results.nodes.select(ids=b)

        Accepts the **same selectors** as :meth:`get`
        (``pg`` / ``label`` / ``selection`` / ``ids``); no selector
        seeds every domain node.  Name resolution is **not**
        re-implemented here: the seed ids are obtained by delegating
        verbatim to :meth:`_SelectionMixin._resolve_node_ids` — the
        exact method :meth:`get` / :meth:`in_box` / :meth:`nearest_to`
        already use — so ``select(...).get(component=)`` is id-for-id the
        same selection as ``get(...)`` (no extra scoping).  The spatial
        verbs narrow the chain *before* the terminal read.

        ``ResultChain`` is imported **deferred** (mirrors
        ``mesh/_node_chain.py``): it imports only the package-root leaf
        ``apeGmsh._chain`` + numpy, so this adds no eager cross-package
        edge and ``results`` stays runtime-clean of ``core``/``mesh``
        (``tests/test_import_dag_polarity.py`` baseline unchanged).
        """
        from ._result_chain import ResultChain, engine_for

        seed = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        if seed is None:                 # no selector → every domain node
            fem = _require_fem(self._r, "results.nodes.select(...)")
            atoms = [int(n) for n in np.asarray(fem.nodes.ids)]
        else:
            atoms = [int(n) for n in np.asarray(seed)]
        engine = engine_for(self._r, self, "node")
        return ResultChain(atoms, _engine=engine)

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
        sid = self._r._resolve_stage(stage)
        node_ids = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        return self._r._reader.read_nodes(
            sid, component, node_ids=node_ids, time_slice=time,
        )

    def available_components(self, *, stage: str | None = None) -> list[str]:
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
        """Start a daisy-chainable element-result selection.

        Returns a :class:`~apeGmsh.results._result_chain.ResultChain`
        (point family, element level; spatial verbs operate on element
        **centroids**) read with a terminal ``.get(component=...)``::

            (results.elements.select(pg="Beams")
                 .in_box(lo, hi)
                 .get(component="globalForce"))

        Accepts the **same selectors** as :meth:`get` plus
        ``element_type=`` (the same additive narrowing
        :meth:`_ElementGeometryMixin.in_box` / :meth:`nearest_to`
        accept).  No selector seeds every domain element.  Resolution
        is **not** re-implemented: the seed ids are obtained by
        delegating verbatim to
        :meth:`_ElementGeometryMixin._combine_candidates` (which itself
        calls the existing :meth:`_resolve_element_ids`) — the exact
        path the element geometry helpers use — so
        ``select(...).get(component=)`` is id-for-id the same selection
        as ``get(...)``.

        ``ResultChain`` is imported **deferred** (mirrors
        ``mesh/_elem_chain.py``): only the package-root leaf
        ``apeGmsh._chain`` + numpy, so ``results`` stays runtime-clean
        of ``core``/``mesh`` and the
        ``tests/test_import_dag_polarity.py`` baseline is unchanged.
        Element centroids are computed **fail-loud** in ``ResultChain``
        (an unknown node id raises; never the ``np.clip`` silent
        substitution ``_element_centroids`` does).
        """
        from ._result_chain import ResultChain, engine_for

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
        return ResultChain(atoms, _engine=engine)

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


# =====================================================================
# Gauss points
# =====================================================================

class GaussResultsComposite(_SelectionMixin, _ElementGeometryMixin):
    """``results.elements.gauss`` — continuum Gauss-point values."""

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
    ) -> GaussSlab:
        sid = self._r._resolve_stage(stage)
        eids = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        return self._r._reader.read_gauss(
            sid, component, element_ids=eids, time_slice=time,
        )

    def available_components(self, *, stage: str | None = None) -> list[str]:
        sid = self._r._resolve_stage(stage)
        return self._r._reader.available_components(sid, ResultLevel.GAUSS)


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
