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

import numpy as np
from numpy import ndarray


@dataclass(frozen=True)
class LocalAxes:
    """Per-element local coordinate frames (beam / shell orientation).

    Read from a ``.ladruno`` ``MODEL/LOCAL_AXES`` group — the orientation a
    ``.mpco`` does **not** carry for beam-columns (so apeGmsh can orient
    line / section-force diagrams straight from ``.ladruno`` for wired
    element classes, instead of the native ``vecxz`` path).

    ``quaternions`` are scalar-first ``(w, x, y, z)``, mapping global →
    element-local at the reference configuration. The local axes are the
    **rows** of the rotation matrix (OpenSees ``quatFromMat`` stores the
    transpose convention), so ``matrices[k]`` has row 0 = local x, row 1 =
    local y, row 2 = local z — each expressed in global coordinates.
    Elements with no recorded frame get the identity quaternion.
    """

    element_ids: ndarray         # (n,)
    quaternions: ndarray         # (n, 4) scalar-first (w, x, y, z)

    @property
    def matrices(self) -> ndarray:
        """``(n, 3, 3)`` per-element rotations; **rows are the local axes**
        in global coordinates."""
        q = np.asarray(self.quaternions, dtype=np.float64).reshape(-1, 4)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        m = np.empty((q.shape[0], 3, 3), dtype=np.float64)
        m[:, 0, 0] = 1 - 2 * (y * y + z * z)
        m[:, 0, 1] = 2 * (x * y - z * w)
        m[:, 0, 2] = 2 * (x * z + y * w)
        m[:, 1, 0] = 2 * (x * y + z * w)
        m[:, 1, 1] = 1 - 2 * (x * x + z * z)
        m[:, 1, 2] = 2 * (y * z - x * w)
        m[:, 2, 0] = 2 * (x * z - y * w)
        m[:, 2, 1] = 2 * (y * z + x * w)
        m[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return m

    @property
    def x_axis(self) -> ndarray:
        """``(n, 3)`` — each element's local x-axis (beam axis) in global coords."""
        return self.matrices[:, 0, :]

    @property
    def y_axis(self) -> ndarray:
        """``(n, 3)`` — each element's local y-axis in global coords."""
        return self.matrices[:, 1, :]

    @property
    def z_axis(self) -> ndarray:
        """``(n, 3)`` — each element's local z-axis in global coords."""
        return self.matrices[:, 2, :]


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
    global coordinates, call ``slab.global_coords(fem)`` — interpolates
    through the bound ``FEMData``'s element shape functions for hex8 /
    quad4, falling back to a centroid + bbox-scaled approximation for
    element types that don't yet have explicit shape-fn support.
    """
    component: str
    values: ndarray              # (T, sum_GP)
    element_index: ndarray       # (sum_GP,)
    natural_coords: ndarray      # (sum_GP, dim)
    local_axes_quaternion: Optional[ndarray]  # (E, 4) for shells, else None
    time: ndarray                # (T,)

    def global_coords(self, fem) -> ndarray:
        """Map per-GP natural coords to ``(sum_GP, 3)`` world coords.

        Uses element shape functions for supported types (hex8, quad4);
        falls back to ``centroid + 0.5 * bbox_span * natural`` for
        others — visualization-faithful for axis-aligned elements.
        """
        from ._gauss_world_coords import compute_global_coords
        return compute_global_coords(self, fem)


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
