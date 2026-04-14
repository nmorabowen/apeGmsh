"""
Results — Self-contained FEM post-processing container.
=======================================================

Combines mesh geometry with analysis result fields (displacements,
stresses, mode shapes) in a single object that requires no live Gmsh
or OpenSees session.

Construction
------------
::

    # From FEMData + numpy arrays (primary workflow)
    results = Results.from_fem(fem,
        point_data={"Displacement": u_array},
        cell_data={"Stress_xx": s_array},
    )

    # From VTU/VTK/PVD file
    results = Results.from_file("output.vtu")

    # Time-series (modal / transient)
    results = Results.from_fem(fem, steps=[
        {"time": freq1, "point_data": {"ModeShape": phi1}},
        {"time": freq2, "point_data": {"ModeShape": phi2}},
    ])

Usage
-----
::

    results.viewer()                  # open apeGmshViewer (no session needed)
    results.to_vtu("output.vtu")     # export single step
    results.to_pvd("modes")          # export time-series (.pvd + .vtu files)
    results.get_point_field("Displacement")       # retrieve field
    results.get_point_field("ModeShape", step=2)  # time-series field
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


# ======================================================================
# VTK cell type constants
# ======================================================================
VTK_VERTEX     = 1
VTK_LINE       = 3
VTK_TRIANGLE   = 5
VTK_QUAD       = 9
VTK_TETRA      = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE      = 13

# (element_dim, nodes_per_element) -> VTK cell type
_DIM_NPE_TO_VTK: dict[tuple[int, int], int] = {
    (0, 1): VTK_VERTEX,
    (1, 2): VTK_LINE,
    (2, 3): VTK_TRIANGLE,
    (2, 4): VTK_QUAD,
    (3, 4): VTK_TETRA,
    (3, 6): VTK_WEDGE,
    (3, 8): VTK_HEXAHEDRON,
}

# Fallback: npe-only lookup (for legacy or when dim is unknown)
_NPE_TO_VTK: dict[int, int] = {1: VTK_VERTEX, 2: VTK_LINE, 3: VTK_TRIANGLE,
                                 4: VTK_TETRA, 6: VTK_WEDGE, 8: VTK_HEXAHEDRON}


# ======================================================================
# VTK cell-array builders
# ======================================================================

def _remap_connectivity(
    conn: ndarray,
    tag_to_idx: dict[int, int],
) -> ndarray:
    """Convert node-tag connectivity to 0-based VTK indices.

    Parameters
    ----------
    conn : ndarray (E, npe)
        Connectivity in Gmsh node tags.
    tag_to_idx : dict
        Mapping from node tag -> 0-based point index.

    Returns
    -------
    ndarray (E, npe), dtype int64
        Connectivity in 0-based indices.
    """
    out = np.empty_like(conn, dtype=np.int64)
    for i in range(conn.shape[0]):
        for j in range(conn.shape[1]):
            out[i, j] = tag_to_idx[int(conn[i, j])]
    return out


def _build_vtk_cells_from_fem(
    fem: "FEMData",
    primary_dim: int = 2,
) -> tuple[ndarray, ndarray, int, dict[int, int]]:
    """Build mixed-type VTK cell arrays from FEMData.

    Collects:
    1. Primary-dim elements from ``fem.elements.connectivity``
    2. Elements from physical groups of *other* dimensions

    Parameters
    ----------
    fem : FEMData
    primary_dim : int
        The dimension of the primary connectivity (passed to
        ``get_fem_data(dim=...)``).

    Returns
    -------
    cells_flat : ndarray
        Flat VTK cell array ``[npe0, idx0, idx1, …, npe1, …]``.
    cell_types : ndarray (uint8)
        Per-cell VTK type codes.
    n_primary : int
        Number of primary-dim cells (first in the array).
    tag_to_idx : dict
        Node-tag -> 0-based index mapping used for the conversion.
    """
    # Node ID -> 0-based index mapping
    tag_to_idx: dict[int, int] = {
        int(t): i for i, t in enumerate(fem.nodes.ids)
    }

    primary_elem_ids: set[int] = set(int(e) for e in fem.elements.ids)
    cell_blocks: list[tuple[ndarray, int]] = []   # (conn_0based, vtk_type)

    # ── 1.  Primary connectivity (per element-type group) ────────
    for group in fem.elements:
        if group.connectivity.size == 0:
            continue
        vtk_type = _DIM_NPE_TO_VTK.get(
            (group.dim, group.npe),
            _NPE_TO_VTK.get(group.npe, VTK_TRIANGLE),
        )
        conn_0 = _remap_connectivity(group.connectivity, tag_to_idx)
        cell_blocks.append((conn_0, vtk_type))

    n_primary = len(fem.elements.ids)

    # ── 2.  Extra elements from physical groups of other dims ────
    if fem.nodes.physical is not None:
        for pg_dim, pg_tag in fem.nodes.physical.get_all():
            if pg_dim < 1 or pg_dim == primary_dim:
                continue
            try:
                pg_elem_ids = fem.nodes.physical.element_ids((pg_dim, pg_tag))
                pg_conn = fem.nodes.physical.connectivity((pg_dim, pg_tag))
            except (ValueError, KeyError):
                continue

            # Filter out elements already counted in primary or earlier
            mask = np.array(
                [int(eid) not in primary_elem_ids for eid in pg_elem_ids],
                dtype=bool,
            )
            if not mask.any():
                continue

            new_conn = pg_conn[mask]
            new_ids = pg_elem_ids[mask]
            new_npe = new_conn.shape[1]
            vtk_type = _DIM_NPE_TO_VTK.get(
                (pg_dim, new_npe),
                _NPE_TO_VTK.get(new_npe, VTK_LINE),
            )

            conn_0 = _remap_connectivity(new_conn, tag_to_idx)
            cell_blocks.append((conn_0, vtk_type))
            primary_elem_ids.update(int(e) for e in new_ids)

    # ── 3.  Assemble flat VTK cell array ─────────────────────────
    flat_parts: list[ndarray] = []
    type_parts: list[ndarray] = []

    for conn_0, vtk_type in cell_blocks:
        n_cells = conn_0.shape[0]
        npe = conn_0.shape[1]
        # Flat format: [npe, n0, n1, ..., npe, n0, n1, ...]
        prefix = np.full((n_cells, 1), npe, dtype=np.int64)
        block = np.hstack([prefix, conn_0])
        flat_parts.append(block.ravel())
        type_parts.append(np.full(n_cells, vtk_type, dtype=np.uint8))

    if flat_parts:
        cells_flat = np.concatenate(flat_parts)
        cell_types = np.concatenate(type_parts)
    else:
        cells_flat = np.array([], dtype=np.int64)
        cell_types = np.array([], dtype=np.uint8)

    return cells_flat, cell_types, n_primary, tag_to_idx


def _pad_cell_data(
    cell_data: dict[str, ndarray] | None,
    n_primary: int,
    n_total: int,
) -> dict[str, ndarray]:
    """Pad cell-data arrays with NaN when extra (lower-dim) cells exist.

    If an array has exactly ``n_primary`` entries and ``n_total > n_primary``,
    it is padded with ``NaN`` for the extra cells.  Arrays that already
    match ``n_total`` are passed through unchanged.
    """
    if cell_data is None:
        return {}
    if n_primary == n_total:
        return dict(cell_data)

    padded: dict[str, ndarray] = {}
    n_extra = n_total - n_primary

    for name, arr in cell_data.items():
        arr = np.asarray(arr, dtype=np.float64)

        if arr.shape[0] == n_primary:
            # Pad with NaN for the extra cells
            if arr.ndim == 1:
                pad = np.full(n_extra, np.nan, dtype=np.float64)
            else:
                pad = np.full((n_extra, arr.shape[1]), np.nan, dtype=np.float64)
            padded[name] = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] == n_total:
            padded[name] = arr
        else:
            raise ValueError(
                f"Cell field '{name}' has {arr.shape[0]} entries, "
                f"expected {n_primary} (primary) or {n_total} (total)."
            )

    return padded


# ======================================================================
# Legacy single-type builder (kept for from_file path)
# ======================================================================

def _fem_to_vtk_cells(
    fem: "FEMData",
) -> tuple[ndarray, ndarray]:
    """Convert FEMData connectivity to VTK cell arrays (legacy).

    Used only by ``from_file`` path.  For ``from_fem``, use
    :func:`_build_vtk_cells_from_fem` which handles mixed types and
    proper index remapping.
    """
    flat_parts: list[ndarray] = []
    type_parts: list[ndarray] = []
    for group in fem.elements:
        if group.connectivity.size == 0:
            continue
        npe = group.npe
        n_cells = len(group)
        vtk_type = _DIM_NPE_TO_VTK.get(
            (group.dim, npe), _NPE_TO_VTK.get(npe, VTK_TRIANGLE))
        prefix = np.full((n_cells, 1), npe, dtype=np.int64)
        block = np.hstack([prefix, group.connectivity.astype(np.int64)])
        flat_parts.append(block.ravel())
        type_parts.append(np.full(n_cells, vtk_type, dtype=np.uint8))
    if flat_parts:
        return np.concatenate(flat_parts), np.concatenate(type_parts)
    return np.array([], dtype=np.int64), np.array([], dtype=np.uint8)


# ======================================================================
# Results
# ======================================================================

class Results:
    """Self-contained FEM results container.

    Holds mesh geometry + result fields (static or time-series).
    No dependency on ``gmsh`` module.  Optional dependencies on
    ``pyvista`` and ``apeGmshViewer`` are deferred to call-time.

    Use the classmethods :meth:`from_fem` or :meth:`from_file` to
    construct instances.
    """

    # Type declarations for __slots__ (consumed by mypy / pyright).
    # The underscore-prefixed backing fields are written via
    # object.__setattr__ in __init__ and exposed through public
    # @property accessors below.
    _node_coords: ndarray
    _cells: Any
    _cell_types: ndarray
    _point_fields: dict[str, ndarray]
    _cell_fields: dict[str, ndarray]
    _time_steps: list[float] | None
    _step_point_fields: dict[str, list[ndarray]] | None
    _step_cell_fields: dict[str, list[ndarray]] | None
    _physical_groups: Any
    _name: str
    _n_primary_cells: int

    __slots__ = (
        "_node_coords",
        "_cells",
        "_cell_types",
        "_point_fields",
        "_cell_fields",
        "_time_steps",
        "_step_point_fields",
        "_step_cell_fields",
        "_physical_groups",
        "_name",
        "_n_primary_cells",
    )

    def __init__(
        self,
        *,
        node_coords: ndarray,
        cells: Any,
        cell_types: ndarray,
        point_fields: dict[str, ndarray] | None = None,
        cell_fields: dict[str, ndarray] | None = None,
        time_steps: list[float] | None = None,
        step_point_fields: dict[str, list[ndarray]] | None = None,
        step_cell_fields: dict[str, list[ndarray]] | None = None,
        physical_groups: Any = None,
        name: str = "results",
        n_primary_cells: int | None = None,
    ) -> None:
        object.__setattr__(self, "_node_coords", np.asarray(node_coords))
        object.__setattr__(self, "_cells", cells)
        object.__setattr__(self, "_cell_types", np.asarray(cell_types))
        object.__setattr__(self, "_point_fields", dict(point_fields or {}))
        object.__setattr__(self, "_cell_fields", dict(cell_fields or {}))
        object.__setattr__(self, "_time_steps", list(time_steps) if time_steps else None)
        object.__setattr__(self, "_step_point_fields", dict(step_point_fields) if step_point_fields else None)
        object.__setattr__(self, "_step_cell_fields", dict(step_cell_fields) if step_cell_fields else None)
        object.__setattr__(self, "_physical_groups", physical_groups)
        object.__setattr__(self, "_name", name)

        n_total = len(np.asarray(cell_types))
        object.__setattr__(
            self, "_n_primary_cells",
            n_primary_cells if n_primary_cells is not None else n_total,
        )

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_fem(
        cls,
        fem: "FEMData",
        *,
        point_data: dict[str, ndarray] | None = None,
        cell_data: dict[str, ndarray] | None = None,
        steps: list[dict] | None = None,
        name: str = "results",
    ) -> "Results":
        """Create from a :class:`FEMData` + numpy result arrays.

        Automatically includes elements from **all** physical groups
        (e.g., 1-D column lines + 2-D slab triangles) to produce a
        mixed-type VTK grid.  Node-tag connectivity is remapped to
        0-based indices internally.

        Parameters
        ----------
        fem : FEMData
            Mesh geometry (from ``g.mesh.queries.get_fem_data()``).
        point_data : dict, optional
            Static nodal fields: ``{name: ndarray (N,) or (N,3)}``.
        cell_data : dict, optional
            Static element fields: ``{name: ndarray (E,) or (E,3)}``.
            If the array length matches only the primary-dim elements,
            it is automatically padded with ``NaN`` for extra cells.
        steps : list[dict], optional
            Time-series steps.  Each dict has ``"time"`` (float) and
            optional ``"point_data"`` / ``"cell_data"`` dicts.
            Mutually exclusive with ``point_data`` / ``cell_data``.
        name : str
            Display name.
        """
        if steps is not None and (point_data or cell_data):
            raise ValueError(
                "Cannot provide both point_data/cell_data and steps. "
                "Use steps for time-series, or point_data/cell_data "
                "for a single static result set."
            )

        # Infer primary dimension from connectivity npe
        primary_dim = _guess_primary_dim(fem)

        cells_flat, cell_types, n_primary, tag_to_idx = \
            _build_vtk_cells_from_fem(fem, primary_dim=primary_dim)

        n_total = len(cell_types)

        if steps is not None:
            return cls._from_fem_steps(
                fem, cells_flat, cell_types,
                n_primary, n_total, steps, name,
            )

        # Pad cell_data if needed (primary-only -> full)
        padded_cell_data = _pad_cell_data(cell_data, n_primary, n_total)

        return cls(
            node_coords=fem.nodes.coords,
            cells=cells_flat,
            cell_types=cell_types,
            point_fields=point_data,
            cell_fields=padded_cell_data if padded_cell_data else None,
            physical_groups=fem.nodes.physical,
            name=name,
            n_primary_cells=n_primary,
        )

    @classmethod
    def _from_fem_steps(
        cls,
        fem: "FEMData",
        cells: ndarray,
        cell_types: ndarray,
        n_primary: int,
        n_total: int,
        steps: list[dict],
        name: str,
    ) -> "Results":
        """Build a time-series Results from FEMData + step dicts."""
        time_values: list[float] = []
        step_pf: dict[str, list[ndarray]] = {}
        step_cf: dict[str, list[ndarray]] = {}

        for i, step in enumerate(steps):
            t = step.get("time", float(i))
            time_values.append(t)

            for field_name, arr in step.get("point_data", {}).items():
                step_pf.setdefault(field_name, []).append(
                    np.asarray(arr),
                )
            for field_name, arr in step.get("cell_data", {}).items():
                arr = np.asarray(arr, dtype=np.float64)
                # Pad if only primary-dim cells provided
                if arr.shape[0] == n_primary and n_total > n_primary:
                    n_extra = n_total - n_primary
                    if arr.ndim == 1:
                        pad = np.full(n_extra, np.nan, dtype=np.float64)
                    else:
                        pad = np.full(
                            (n_extra, arr.shape[1]), np.nan, dtype=np.float64,
                        )
                    arr = np.concatenate([arr, pad], axis=0)
                step_cf.setdefault(field_name, []).append(arr)

        # Validate consistent step counts
        n = len(steps)
        for field_name, arrays in step_pf.items():
            if len(arrays) != n:
                raise ValueError(
                    f"Point field '{field_name}' has {len(arrays)} "
                    f"arrays but {n} steps were provided."
                )
        for field_name, arrays in step_cf.items():
            if len(arrays) != n:
                raise ValueError(
                    f"Cell field '{field_name}' has {len(arrays)} "
                    f"arrays but {n} steps were provided."
                )

        return cls(
            node_coords=fem.nodes.coords,
            cells=cells,
            cell_types=cell_types,
            time_steps=time_values,
            step_point_fields=step_pf if step_pf else None,
            step_cell_fields=step_cf if step_cf else None,
            physical_groups=fem.nodes.physical,
            name=name,
            n_primary_cells=n_primary,
        )

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        *,
        name: str | None = None,
    ) -> "Results":
        """Load results from a VTU / VTK / PVD file.

        Parameters
        ----------
        filepath : str or Path
            Path to the results file.
        name : str, optional
            Display name (defaults to filename stem).
        """
        from apeGmshViewer.loaders.vtu_loader import load_file

        filepath = Path(filepath)
        mesh_data = load_file(filepath)
        display_name = name or mesh_data.name or filepath.stem

        grid = mesh_data.mesh
        node_coords = np.array(grid.points)
        cells = grid.cells
        cell_types = np.array(grid.celltypes)

        point_fields = {
            k: np.array(grid.point_data[k])
            for k in mesh_data.point_field_names
        }
        cell_fields = {
            k: np.array(grid.cell_data[k])
            for k in mesh_data.cell_field_names
        }

        # Time-series (PVD)
        time_steps: list[float] | None = None
        step_pf: dict[str, list[ndarray]] | None = None
        step_cf: dict[str, list[ndarray]] | None = None

        if mesh_data.has_time_series and mesh_data.step_meshes:
            time_steps = list(mesh_data.time_steps)
            step_pf = {}
            step_cf = {}

            for step_mesh in mesh_data.step_meshes:
                for k in step_mesh.point_data:
                    step_pf.setdefault(k, []).append(
                        np.array(step_mesh.point_data[k]),
                    )
                for k in step_mesh.cell_data:
                    step_cf.setdefault(k, []).append(
                        np.array(step_mesh.cell_data[k]),
                    )

            # Clear static fields if time-series is present
            point_fields = {}
            cell_fields = {}

        return cls(
            node_coords=node_coords,
            cells=cells,
            cell_types=cell_types,
            point_fields=point_fields if point_fields else None,
            cell_fields=cell_fields if cell_fields else None,
            time_steps=time_steps,
            step_point_fields=step_pf if step_pf else None,
            step_cell_fields=step_cf if step_cf else None,
            name=display_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_coords(self) -> ndarray:
        return self._node_coords

    @property
    def cells(self) -> Any:
        return self._cells

    @property
    def cell_types(self) -> ndarray:
        return self._cell_types

    @property
    def point_fields(self) -> dict[str, ndarray]:
        return self._point_fields

    @property
    def cell_fields(self) -> dict[str, ndarray]:
        return self._cell_fields

    @property
    def time_steps(self) -> list[float] | None:
        return self._time_steps

    @property
    def physical_groups(self) -> Any:
        return self._physical_groups

    @property
    def has_time_series(self) -> bool:
        return self._time_steps is not None and len(self._time_steps) > 1

    @property
    def n_steps(self) -> int:
        if self._time_steps is None:
            return 1
        return len(self._time_steps)

    @property
    def n_primary_cells(self) -> int:
        """Number of primary-dimension cells (before extra groups)."""
        return self._n_primary_cells

    @property
    def n_total_cells(self) -> int:
        """Total cell count (primary + extra physical-group elements)."""
        return len(self._cell_types)

    @property
    def name(self) -> str:
        return self._name

    @property
    def field_names(self) -> dict[str, list[str]]:
        """All field names: ``{"point": [...], "cell": [...]}``."""
        point = list(self._point_fields.keys())
        cell = list(self._cell_fields.keys())
        if self._step_point_fields:
            for k in self._step_point_fields:
                if k not in point:
                    point.append(k)
        if self._step_cell_fields:
            for k in self._step_cell_fields:
                if k not in cell:
                    cell.append(k)
        return {"point": point, "cell": cell}

    # ------------------------------------------------------------------
    # Field access
    # ------------------------------------------------------------------

    def get_point_field(
        self, name: str, step: int | None = None,
    ) -> ndarray:
        """Retrieve a nodal field array.

        Parameters
        ----------
        name : str
            Field name.
        step : int, optional
            Time-step index (required for time-series fields).
        """
        # Static field
        if name in self._point_fields and step is None:
            return self._point_fields[name]

        # Step field
        if self._step_point_fields and name in self._step_point_fields:
            if step is None:
                raise ValueError(
                    f"Field '{name}' is time-series — provide step=<int>. "
                    f"Available steps: 0..{self.n_steps - 1}"
                )
            arrays = self._step_point_fields[name]
            if step < 0 or step >= len(arrays):
                raise IndexError(
                    f"Step {step} out of range for field '{name}' "
                    f"(0..{len(arrays) - 1})"
                )
            return arrays[step]

        available = self.field_names["point"]
        raise KeyError(
            f"Point field '{name}' not found. "
            f"Available: {available}"
        )

    def get_cell_field(
        self, name: str, step: int | None = None,
    ) -> ndarray:
        """Retrieve an element field array.

        Parameters
        ----------
        name : str
            Field name.
        step : int, optional
            Time-step index (required for time-series fields).
        """
        if name in self._cell_fields and step is None:
            return self._cell_fields[name]

        if self._step_cell_fields and name in self._step_cell_fields:
            if step is None:
                raise ValueError(
                    f"Field '{name}' is time-series — provide step=<int>. "
                    f"Available steps: 0..{self.n_steps - 1}"
                )
            arrays = self._step_cell_fields[name]
            if step < 0 or step >= len(arrays):
                raise IndexError(
                    f"Step {step} out of range for field '{name}' "
                    f"(0..{len(arrays) - 1})"
                )
            return arrays[step]

        available = self.field_names["cell"]
        raise KeyError(
            f"Cell field '{name}' not found. "
            f"Available: {available}"
        )

    # ------------------------------------------------------------------
    # Internal: build PyVista grid
    # ------------------------------------------------------------------

    def _build_grid(self, step: int | None = None):
        """Build a ``pv.UnstructuredGrid`` on demand.

        Parameters
        ----------
        step : int, optional
            Time-step index.  If None, uses static fields.
        """
        import pyvista as pv

        grid = pv.UnstructuredGrid(
            self._cells, self._cell_types, self._node_coords,
        )

        if step is None:
            # Attach static fields
            for k, v in self._point_fields.items():
                grid.point_data[k] = v
            for k, v in self._cell_fields.items():
                grid.cell_data[k] = v
        else:
            # Attach fields from the requested step
            if self._step_point_fields:
                for k, arrays in self._step_point_fields.items():
                    if step < len(arrays):
                        grid.point_data[k] = arrays[step]
            if self._step_cell_fields:
                for k, arrays in self._step_cell_fields.items():
                    if step < len(arrays):
                        grid.cell_data[k] = arrays[step]

        return grid

    # ------------------------------------------------------------------
    # Viewer bridge
    # ------------------------------------------------------------------

    def to_mesh_data(self):
        """Convert to a apeGmshViewer ``MeshData`` (no file I/O).

        Transfers mesh geometry and all fields directly in memory.

        Returns
        -------
        apeGmshViewer.loaders.vtu_loader.MeshData
        """
        from apeGmshViewer.loaders.vtu_loader import from_arrays

        if self.has_time_series:
            return from_arrays(
                self._node_coords,
                self._cells,
                self._cell_types,
                time_steps=self._time_steps,
                step_point_data=self._step_point_fields,
                step_cell_data=self._step_cell_fields,
                name=self._name,
            )
        return from_arrays(
            self._node_coords,
            self._cells,
            self._cell_types,
            point_data=self._point_fields,
            cell_data=self._cell_fields,
            name=self._name,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_vtu(self, filepath: str | Path) -> Path:
        """Write a single VTU file.

        For time-series, writes step 0 only.  Use :meth:`to_pvd` for
        the full series.

        Returns the written file path.
        """
        filepath = Path(filepath)
        step = 0 if self.has_time_series else None
        grid = self._build_grid(step=step)
        grid.save(str(filepath))
        return filepath

    def to_pvd(self, base_path: str | Path) -> list[Path]:
        """Write a PVD time-series (multiple VTU files + collection).

        Parameters
        ----------
        base_path : str or Path
            Base path *without* extension.  Produces::

                base_path.pvd
                base_path_000.vtu
                base_path_001.vtu
                ...

        Returns list of all written file paths (PVD first).
        """
        base = Path(base_path)
        parent = base.parent
        stem = base.stem
        parent.mkdir(parents=True, exist_ok=True)

        n = self.n_steps
        written: list[Path] = []

        # Write individual VTU files
        vtu_names: list[str] = []
        for i in range(n):
            vtu_name = f"{stem}_{i:03d}.vtu"
            vtu_path = parent / vtu_name
            grid = self._build_grid(step=i)
            grid.save(str(vtu_path))
            vtu_names.append(vtu_name)
            written.append(vtu_path)

        # Write PVD collection file
        pvd_path = parent / f"{stem}.pvd"
        times = self._time_steps or [float(i) for i in range(n)]

        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1">',
            "  <Collection>",
        ]
        for t, vtu_name in zip(times, vtu_names):
            lines.append(
                f'    <DataSet timestep="{t}" file="{vtu_name}"/>',
            )
        lines.append("  </Collection>")
        lines.append("</VTKFile>")

        pvd_path.write_text("\n".join(lines), encoding="utf-8")
        written.insert(0, pvd_path)
        return written

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def viewer(self, *, blocking: bool = False) -> None:
        """Open the results in apeGmshViewer.

        No live Gmsh session required.

        Parameters
        ----------
        blocking : bool
            If False (default), writes temp files and launches a
            subprocess so the notebook / script keeps running.
            If True, opens the viewer in-process with direct memory
            transfer (no temp files) and blocks until closed.
        """
        if blocking:
            from apeGmshViewer import show_mesh_data
            show_mesh_data(self.to_mesh_data(), blocking=True)
        else:
            import tempfile
            import shutil
            import atexit
            from apeGmshViewer import show

            tmp_dir = tempfile.mkdtemp(prefix="apeGmsh_results_")
            atexit.register(shutil.rmtree, tmp_dir, True)

            if self.has_time_series:
                paths = self.to_pvd(Path(tmp_dir) / self._name)
                show(str(paths[0]), blocking=False)
            else:
                vtu_path = Path(tmp_dir) / f"{self._name}.vtu"
                self.to_vtu(vtu_path)
                show(str(vtu_path), blocking=False)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [f"Results: '{self._name}'"]

        n_total = self.n_total_cells
        n_primary = self._n_primary_cells
        if n_total == n_primary:
            lines.append(
                f"  Mesh: {len(self._node_coords)} nodes, "
                f"{n_total} elements"
            )
        else:
            lines.append(
                f"  Mesh: {len(self._node_coords)} nodes, "
                f"{n_total} elements "
                f"({n_primary} primary + {n_total - n_primary} extra)"
            )

        pf = self.field_names["point"]
        cf = self.field_names["cell"]
        if pf:
            lines.append(f"  Point fields: {', '.join(pf)}")
        if cf:
            lines.append(f"  Cell fields:  {', '.join(cf)}")
        if self.has_time_series:
            lines.append(
                f"  Time-series: {self.n_steps} steps "
                f"({self._time_steps[0]:.4g} .. {self._time_steps[-1]:.4g})"
            )
        if self._physical_groups is not None:
            lines.append(f"  Physical groups: {len(self._physical_groups)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_fields = len(self.field_names["point"]) + len(self.field_names["cell"])
        parts = [
            f"'{self._name}'",
            f"{len(self._node_coords)} nodes",
            f"{self.n_total_cells} cells",
            f"{n_fields} fields",
        ]
        if self.has_time_series:
            parts.append(f"{self.n_steps} steps")
        return f"<Results {', '.join(parts)}>"


# ======================================================================
# Helpers
# ======================================================================

def _guess_primary_dim(fem: "FEMData") -> int:
    """Infer the primary element dimension from element groups.

    Returns the highest dimension present in the mesh.
    """
    dims = {g.dim for g in fem.elements}
    return max(dims) if dims else 2
