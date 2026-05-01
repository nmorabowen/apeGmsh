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


def _nearest_node_id(fem, xyz, *, candidate_ids=None) -> int:
    """Return the node ID closest to ``xyz`` in 3D Euclidean distance.

    If ``candidate_ids`` is provided, restrict the search to that set
    (additive composition with named selectors).
    """
    target = np.asarray(xyz, dtype=np.float64).reshape(3)
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


def _node_ids_in_box(fem, p_min, p_max) -> ndarray:
    """Return node IDs whose coordinates lie inside the AABB.

    Box is half-open on the upper side: ``p_min <= xyz < p_max`` per
    axis. Coordinates equal to ``p_max`` are excluded so adjacent
    boxes don't double-count a shared face.
    """
    lo = np.asarray(p_min, dtype=np.float64).reshape(3)
    hi = np.asarray(p_max, dtype=np.float64).reshape(3)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if coords.size == 0:
        return np.array([], dtype=np.int64)
    mask = np.all((coords >= lo) & (coords < hi), axis=1)
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
        # Guard against any node IDs not in the FEM (shouldn't happen,
        # but if it does, np.searchsorted returns len(sorted_node_ids)).
        loc = np.clip(loc, 0, len(sorted_node_ids) - 1)
        orig = inverse_perm[loc]
        node_xyz = coords[orig].reshape(conn.shape + (3,))
        cent = node_xyz.mean(axis=1)

        out_ids.append(np.asarray(ids, dtype=np.int64))
        out_cent.append(cent)

    if not out_ids:
        return np.array([], dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    return np.concatenate(out_ids), np.vstack(out_cent)


def _nearest_element_id(fem, xyz, *, candidate_ids=None) -> int:
    """Return the element ID whose centroid is closest to ``xyz``.

    If ``candidate_ids`` is provided, restrict the search to that set.
    """
    target = np.asarray(xyz, dtype=np.float64).reshape(3)
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


def _element_ids_in_box(fem, p_min, p_max) -> ndarray:
    """Return element IDs whose centroid lies inside the AABB."""
    lo = np.asarray(p_min, dtype=np.float64).reshape(3)
    hi = np.asarray(p_max, dtype=np.float64).reshape(3)
    ids, cent = _element_centroids(fem)
    if ids.size == 0:
        return np.array([], dtype=np.int64)
    mask = np.all((cent >= lo) & (cent < hi), axis=1)
    return ids[mask]


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
    """Geometric ``nearest_to`` / ``in_box`` helpers for element-level
    composites. The subclass must expose a ``.get(...)`` accepting
    ``ids=`` / ``component=`` / ``time=`` / ``stage=`` and an ``_r``
    attribute pointing at the parent :class:`Results`.

    Filters compose **additively** — passing ``pg=`` / ``label=`` /
    ``selection=`` / ``ids=`` restricts the candidate set first; the
    spatial filter then narrows that subset. Distance and containment
    are computed against element centroids (mean of the element's
    node coordinates).
    """

    _r: "Results"

    def nearest_to(
        self,
        xyz,
        *,
        component: str,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ):
        """Read ``component`` at the element whose centroid is nearest ``xyz``.

        ``pg=`` / ``label=`` / ``selection=`` / ``ids=`` restrict the
        search to a candidate set first.
        """
        fem = _require_fem(
            self._r, f"results.elements.{type(self).__name__}.nearest_to(...)",
        )
        candidate = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        eid = _nearest_element_id(fem, xyz, candidate_ids=candidate)
        return self.get(
            ids=[eid], component=component, time=time, stage=stage,
        )

    def in_box(
        self,
        p_min,
        p_max,
        *,
        component: str,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ):
        """Read ``component`` at every element whose centroid is in ``[p_min, p_max)``.

        ``pg=`` / ``label=`` / ``selection=`` / ``ids=`` restrict the
        search to a candidate set first; the box filter narrows that
        subset.
        """
        fem = _require_fem(
            self._r, f"results.elements.{type(self).__name__}.in_box(...)",
        )
        candidate = self._resolve_element_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        in_box_ids = _element_ids_in_box(fem, p_min, p_max)
        narrowed = (
            np.intersect1d(candidate, in_box_ids, assume_unique=False)
            if candidate is not None else in_box_ids
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
        xyz,
        *,
        component: str,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> NodeSlab:
        """Read ``component`` at the node nearest ``xyz``.

        Filters compose **additively** — pass any of ``pg=`` /
        ``label=`` / ``selection=`` / ``ids=`` and the search is
        restricted to that candidate set first. Distance is 3D
        Euclidean against the bound FEMData's node coordinates.

        Examples::

            # Globally nearest
            results.nodes.nearest_to((1.0, 0.0, 0.0), component="...")

            # Nearest within a PG
            results.nodes.nearest_to(
                (1.0, 0.0, 0.0), component="...", pg="Top",
            )
        """
        fem = _require_fem(self._r, "results.nodes.nearest_to(...)")
        candidate = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        nid = _nearest_node_id(fem, xyz, candidate_ids=candidate)
        return self.get(
            ids=[nid], component=component, time=time, stage=stage,
        )

    def in_box(
        self,
        p_min,
        p_max,
        *,
        component: str,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | ndarray | None = None,
        time: TimeSlice = None,
        stage: str | None = None,
    ) -> NodeSlab:
        """Read ``component`` at every node inside the AABB ``[p_min, p_max)``.

        Filters compose **additively** — passing any of ``pg=`` /
        ``label=`` / ``selection=`` / ``ids=`` restricts the search to
        that candidate set first, then keeps only those whose
        coordinates fall inside the box.

        Box bounds are half-open on the upper side so adjacent boxes
        don't double-count shared faces. Use ``-np.inf`` / ``np.inf``
        to relax an axis.

        Examples::

            # Plain box query
            results.nodes.in_box((-1, -1, 0), (1, 1, 5), component="...")

            # Box intersected with a PG
            results.nodes.in_box(
                (-1, -1, 0), (1, 1, 5),
                component="...",
                pg="Top",
            )
        """
        fem = _require_fem(self._r, "results.nodes.in_box(...)")
        candidate = self._resolve_node_ids(
            pg=pg, label=label, selection=selection, ids=ids,
        )
        in_box_ids = _node_ids_in_box(fem, p_min, p_max)
        narrowed = (
            np.intersect1d(candidate, in_box_ids, assume_unique=False)
            if candidate is not None else in_box_ids
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
