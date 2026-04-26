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
)
from .readers._protocol import ResultLevel, TimeSlice

if TYPE_CHECKING:
    from .Results import Results


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


# =====================================================================
# Elements (per-element-node forces)
# =====================================================================

class ElementResultsComposite(_SelectionMixin):
    """``results.elements`` — per-element-node forces, plus sub-composites."""

    def __init__(self, results: "Results") -> None:
        self._r = results
        self.gauss = GaussResultsComposite(results)
        self.fibers = FibersResultsComposite(results)
        self.layers = LayersResultsComposite(results)
        self.line_stations = LineStationsResultsComposite(results)

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

class GaussResultsComposite(_SelectionMixin):
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

class FibersResultsComposite(_SelectionMixin):
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

class LayersResultsComposite(_SelectionMixin):
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

class LineStationsResultsComposite(_SelectionMixin):
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
