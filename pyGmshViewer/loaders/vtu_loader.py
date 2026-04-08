"""
VTU / PVD file loader — reads VTK UnstructuredGrid files into PyVista meshes.

Supports:
    - Single .vtu files (mesh + optional point/cell data)
    - .pvd time-series collections (modal analysis, transient results)
    - .msh Gmsh native format (via meshio fallback)
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv


@dataclass
class MeshData:
    """Container for a loaded mesh and its metadata."""

    name: str
    mesh: pv.UnstructuredGrid
    filepath: Path
    point_field_names: list[str] = field(default_factory=list)
    cell_field_names: list[str] = field(default_factory=list)
    time_steps: list[float] | None = None
    step_meshes: list[pv.UnstructuredGrid] | None = None

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def bounds(self) -> tuple:
        return self.mesh.bounds

    @property
    def has_time_series(self) -> bool:
        return self.time_steps is not None and len(self.time_steps) > 1


def load_vtu(filepath: str | Path) -> MeshData:
    """Load a single .vtu file.

    Parameters
    ----------
    filepath : path
        Path to a .vtu (VTK UnstructuredGrid XML) file.

    Returns
    -------
    MeshData with the loaded mesh and field names.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"VTU file not found: {filepath}")

    mesh = pv.read(str(filepath))

    point_fields = list(mesh.point_data.keys())
    cell_fields = list(mesh.cell_data.keys())

    return MeshData(
        name=filepath.stem,
        mesh=mesh,
        filepath=filepath,
        point_field_names=point_fields,
        cell_field_names=cell_fields,
    )


def load_pvd(filepath: str | Path) -> MeshData:
    """Load a .pvd time-series collection.

    Each <DataSet> in the PVD references a .vtu file at a specific time step.
    All individual meshes are loaded, and the first step becomes the
    primary mesh.

    Parameters
    ----------
    filepath : path
        Path to a .pvd (ParaView Data) file.

    Returns
    -------
    MeshData with time_steps and step_meshes populated.
    """
    import xml.etree.ElementTree as ET

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PVD file not found: {filepath}")

    tree = ET.parse(filepath)
    root = tree.getroot()
    collection = root.find("Collection")
    if collection is None:
        raise ValueError(f"No <Collection> found in PVD file: {filepath}")

    parent_dir = filepath.parent
    time_steps = []
    step_meshes = []

    for ds in collection.findall("DataSet"):
        t = float(ds.get("timestep", 0.0))
        vtu_name = ds.get("file", "")
        vtu_path = parent_dir / vtu_name

        if vtu_path.exists():
            m = pv.read(str(vtu_path))
            time_steps.append(t)
            step_meshes.append(m)

    if not step_meshes:
        raise ValueError(f"No valid VTU files found in PVD: {filepath}")

    # Primary mesh is the first time step
    primary = step_meshes[0]
    point_fields = list(primary.point_data.keys())
    cell_fields = list(primary.cell_data.keys())

    return MeshData(
        name=filepath.stem,
        mesh=primary,
        filepath=filepath,
        point_field_names=point_fields,
        cell_field_names=cell_fields,
        time_steps=time_steps,
        step_meshes=step_meshes,
    )


def load_file(filepath: str | Path) -> MeshData:
    """Auto-detect file type and load accordingly.

    Supports: .vtu, .pvd, .vtk, .msh
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == ".pvd":
        return load_pvd(filepath)
    elif ext in (".vtu", ".vtk", ".vtp"):
        return load_vtu(filepath)
    elif ext == ".msh":
        # Gmsh native format — PyVista can read it via meshio
        return load_vtu(filepath)
    else:
        # Try generic PyVista reader
        return load_vtu(filepath)


def create_deformed_mesh(
    mesh_data: MeshData,
    displacement_field: str = "Displacement",
    scale_factor: float = 1.0,
    time_step: int | None = None,
) -> pv.UnstructuredGrid:
    """Create a deformed copy of the mesh.

    Parameters
    ----------
    mesh_data : MeshData
        Source mesh data.
    displacement_field : str
        Name of the vector field containing displacements.
    scale_factor : float
        Amplification factor for displacements.
    time_step : int, optional
        If mesh has time series, which step to use.

    Returns
    -------
    New mesh with displaced coordinates.
    """
    if time_step is not None and mesh_data.step_meshes:
        source = mesh_data.step_meshes[time_step]
    else:
        source = mesh_data.mesh

    if displacement_field not in source.point_data:
        raise KeyError(
            f"Displacement field '{displacement_field}' not found. "
            f"Available: {list(source.point_data.keys())}"
        )

    deformed = source.copy()
    disp = np.asarray(source.point_data[displacement_field])

    # Ensure 3D displacement
    if disp.ndim == 1:
        n = source.n_points
        disp = disp.reshape(n, -1)
    if disp.shape[1] == 2:
        disp = np.column_stack([disp, np.zeros(disp.shape[0])])

    deformed.points = np.array(source.points) + scale_factor * disp
    return deformed


def from_arrays(
    node_coords: np.ndarray,
    cells: list | np.ndarray,
    cell_types: np.ndarray,
    *,
    point_data: dict[str, np.ndarray] | None = None,
    cell_data: dict[str, np.ndarray] | None = None,
    time_steps: list[float] | None = None,
    step_point_data: dict[str, list[np.ndarray]] | None = None,
    step_cell_data: dict[str, list[np.ndarray]] | None = None,
    name: str = "mesh",
) -> MeshData:
    """Create a :class:`MeshData` from numpy arrays (no file I/O).

    Parameters
    ----------
    node_coords : ndarray (N, 3)
        Node coordinates.
    cells : list or ndarray
        VTK-style cell array.
    cell_types : ndarray
        VTK cell type codes, one per cell.
    point_data : dict, optional
        Nodal fields: ``{name: ndarray}``.
    cell_data : dict, optional
        Element fields: ``{name: ndarray}``.
    time_steps : list[float], optional
        Time values for each step (time-series data).
    step_point_data : dict, optional
        Per-step nodal fields: ``{name: [arr_step0, arr_step1, ...]}``.
    step_cell_data : dict, optional
        Per-step element fields: ``{name: [arr_step0, arr_step1, ...]}``.
    name : str
        Display name for the mesh.

    Returns
    -------
    MeshData ready for the viewer.
    """
    ct = np.asarray(cell_types)

    # Time-series path: build one grid per step
    if time_steps is not None and (step_point_data or step_cell_data):
        step_meshes = []
        all_point_names: set[str] = set()
        all_cell_names: set[str] = set()

        for i in range(len(time_steps)):
            step_grid = pv.UnstructuredGrid(cells, ct, node_coords)
            if step_point_data:
                for fname, arrays in step_point_data.items():
                    step_grid.point_data[fname] = np.asarray(arrays[i])
                    all_point_names.add(fname)
            if step_cell_data:
                for fname, arrays in step_cell_data.items():
                    step_grid.cell_data[fname] = np.asarray(arrays[i])
                    all_cell_names.add(fname)
            step_meshes.append(step_grid)

        return MeshData(
            name=name,
            mesh=step_meshes[0],
            filepath=Path(""),
            point_field_names=sorted(all_point_names),
            cell_field_names=sorted(all_cell_names),
            time_steps=time_steps,
            step_meshes=step_meshes,
        )

    # Static path
    grid = pv.UnstructuredGrid(cells, ct, node_coords)
    for field_name, arr in (point_data or {}).items():
        grid.point_data[field_name] = np.asarray(arr)
    for field_name, arr in (cell_data or {}).items():
        grid.cell_data[field_name] = np.asarray(arr)
    return MeshData(
        name=name,
        mesh=grid,
        filepath=Path(""),
        point_field_names=list(grid.point_data.keys()),
        cell_field_names=list(grid.cell_data.keys()),
    )
