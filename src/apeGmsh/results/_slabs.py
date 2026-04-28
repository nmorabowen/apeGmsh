"""Slab dataclasses returned by ``ResultsReader`` implementations.

A slab carries one component's values plus enough location metadata
that the caller can interpret each row without re-deriving it. They
are numpy-native and immutable; the viewer wraps them in ``xarray``
when it wants labeled axes.

Shape conventions (single-stage, post-stitching across partitions):

================  ========================  =================================================
Slab              ``values`` shape          Location index fields
================  ========================  =================================================
NodeSlab          ``(T, N)``                ``node_ids: (N,)``
ElementSlab       ``(T, E, npe)``           ``element_ids: (E,)``
LineStationSlab   ``(T, sum_S)``            ``element_index, station_natural_coord: (sum_S,)``
GaussSlab         ``(T, sum_GP)``           ``element_index: (sum_GP,)``,
                                            ``natural_coords: (sum_GP, dim)``
FiberSlab         ``(T, sum_F)``            ``element_index, gp_index, y, z, area,
                                            material_tag: (sum_F,)``
LayerSlab         ``(T, sum_L)``            ``element_index, gp_index, layer_index,
                                            sub_gp_index, thickness: (sum_L,)``
================  ========================  =================================================

For a single time step (``time_slice`` was a scalar), ``T`` is 1 and
the leading axis is preserved — the caller can squeeze if desired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from numpy import ndarray


@dataclass(frozen=True)
class NodeSlab:
    """Node-level result values."""
    component: str
    values: ndarray              # (T, N)
    node_ids: ndarray            # (N,)
    time: ndarray                # (T,)


@dataclass(frozen=True)
class ElementSlab:
    """Per-element-node values (e.g. globalForce / localForce)."""
    component: str
    values: ndarray              # (T, E, npe)
    element_ids: ndarray         # (E,)
    time: ndarray                # (T,)


@dataclass(frozen=True)
class LineStationSlab:
    """Beam line-diagram values per integration station."""
    component: str
    values: ndarray              # (T, sum_S)
    element_index: ndarray       # (sum_S,) — parent element per row
    station_natural_coord: ndarray  # (sum_S,) — in [-1, +1]
    time: ndarray                # (T,)


@dataclass(frozen=True)
class GaussSlab:
    """Continuum Gauss-point values.

    ``natural_coords`` are in parent space ``[-1, +1]``. To get
    global coordinates, call ``Results.elements.gauss.get(...)
    .global_coords()`` which interpolates through the bound
    FEMData's element shape functions.
    """
    component: str
    values: ndarray              # (T, sum_GP)
    element_index: ndarray       # (sum_GP,)
    natural_coords: ndarray      # (sum_GP, dim)
    local_axes_quaternion: Optional[ndarray]  # (E, 4) for shells, else None
    time: ndarray                # (T,)


@dataclass(frozen=True)
class FiberSlab:
    """Fiber-level values within fiber-section GPs."""
    component: str
    values: ndarray              # (T, sum_F)
    element_index: ndarray       # (sum_F,) — parent element
    gp_index: ndarray            # (sum_F,) — parent GP within element
    y: ndarray                   # (sum_F,) — section-local y
    z: ndarray                   # (sum_F,) — section-local z
    area: ndarray                # (sum_F,)
    material_tag: ndarray        # (sum_F,)
    time: ndarray                # (T,)


@dataclass(frozen=True)
class LayerSlab:
    """Layered shell layer values (one row per (elem, surf_gp, layer, sub_gp))."""
    component: str
    values: ndarray              # (T, sum_L)
    element_index: ndarray       # (sum_L,)
    gp_index: ndarray            # (sum_L,) — surface GP
    layer_index: ndarray         # (sum_L,)
    sub_gp_index: ndarray        # (sum_L,) — through-thickness GP
    thickness: ndarray           # (sum_L,)
    local_axes_quaternion: ndarray  # (sum_L, 4)
    time: ndarray                # (T,)


@dataclass(frozen=True)
class SpringSlab:
    """Zero-length spring values (one column per element, one spring index).

    ``component`` encodes which spring is represented (e.g.
    ``"spring_force_0"`` for the force in the first configured spring
    direction). Each column in ``values`` corresponds to one element;
    ``element_index`` carries the raw OpenSees element tag so the
    caller can correlate columns with elements without needing a
    separate ID array.

    ================  ========================
    ``values``        ``(T, E)``
    ``element_index`` ``(E,)``
    ================  ========================
    """
    component: str
    values: ndarray              # (T, E)
    element_index: ndarray       # (E,)
    time: ndarray                # (T,)
