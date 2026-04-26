"""Reader protocol — backend-agnostic contract for the composite layer.

Two implementations:

- ``NativeReader`` reads apeGmsh native HDF5 (Phase 1).
- ``MPCOReader`` reads STKO MPCO HDF5 (Phase 3).

The composite layer above (``Results.nodes.get(...)`` etc.) talks
only to this protocol — it never branches on backend type.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Protocol, Union, runtime_checkable

from numpy import ndarray

from .._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    NodeSlab,
)

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData


class ResultLevel(Enum):
    """The topology level a component lives at."""

    NODES = "nodes"
    ELEMENTS = "elements"               # per-element-node (globalForce / localForce)
    LINE_STATIONS = "line_stations"     # per-station (section.force on beams)
    GAUSS = "gauss"                     # continuum integration points
    FIBERS = "fibers"                   # within fiber-section GPs
    LAYERS = "layers"                   # layered shells


@dataclass(frozen=True)
class StageInfo:
    """Stage metadata returned by ``ResultsReader.stages()``.

    For ``kind="mode"``, the eigenvalue/frequency/period/index
    fields are populated and ``n_steps`` is 1. For other kinds,
    the mode-only fields are ``None``.
    """

    id: str
    name: str
    kind: str                           # "transient" | "static" | "mode"
    n_steps: int

    eigenvalue: Optional[float] = None
    frequency_hz: Optional[float] = None
    period_s: Optional[float] = None
    mode_index: Optional[int] = None


# A time slice can be:
#   None              — full time axis
#   int               — single step index
#   list[int]         — explicit step indices
#   slice             — half-open [start, stop) over time values (numpy semantics)
#   float             — single time value, reader picks nearest step
TimeSlice = Union[None, int, list[int], slice, float]


@runtime_checkable
class ResultsReader(Protocol):
    """Backend-agnostic reader protocol.

    Implementations must support all six topology levels even if a
    given file has no data at some of them; they should return empty
    component lists from ``available_components()`` in that case
    rather than raising.
    """

    # ------------------------------------------------------------------
    # Stage discovery
    # ------------------------------------------------------------------

    def stages(self) -> list[StageInfo]:
        """All stages in this file, in write order."""
        ...

    def time_vector(self, stage_id: str) -> ndarray:
        """Time vector for a stage. Shape ``(n_steps,)``."""
        ...

    def partitions(self, stage_id: str) -> list[str]:
        """Partition IDs for a stage (always at least one)."""
        ...

    # ------------------------------------------------------------------
    # FEM access
    # ------------------------------------------------------------------

    def fem(self) -> "Optional[FEMData]":
        """Embedded / synthesized FEMData snapshot.

        - ``NativeReader``: reconstructs from ``/model/`` (always available).
        - ``MPCOReader``: synthesizes a partial FEMData from ``/MODEL/``
          (no apeGmsh labels, no Part provenance).
        """
        ...

    # ------------------------------------------------------------------
    # Component discovery
    # ------------------------------------------------------------------

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        """Canonical component names available at the given level."""
        ...

    # ------------------------------------------------------------------
    # Slab reads
    # ------------------------------------------------------------------

    def read_nodes(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        ...

    def read_elements(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> ElementSlab:
        ...

    def read_line_stations(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        ...

    def read_gauss(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> GaussSlab:
        ...

    def read_fibers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> FiberSlab:
        ...

    def read_layers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        layer_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LayerSlab:
        ...
