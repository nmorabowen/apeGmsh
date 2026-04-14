"""
VTKExport — Write Gmsh mesh + FEM results to VTU files for ParaView.

Zero external dependencies beyond numpy and the Python standard library.
Produces XML-based VTK UnstructuredGrid (.vtu) files that ParaView,
VisIt, and any VTK-compatible viewer can open directly.

Typical usage
-------------
>>> from VTKExport import VTKExport
>>> vtk = VTKExport(g)                         # pass the apeGmsh instance
>>> vtk.add_node_scalar("Temperature", T)       # T is (nNode,)
>>> vtk.add_node_vector("Displacement", disp)   # disp is (nNode, 3)
>>> vtk.add_elem_scalar("Stress_xx", sig_xx)    # sig_xx is (nElem,)
>>> vtk.write("results.vtu")                     # -> open in ParaView

Or standalone (without apeGmsh):
>>> VTKExport.write_vtu("out.vtu", coords, connectivity,
...     point_data={"u": disp}, cell_data={"sig": sig_xx},
...     cell_type="quad")
"""

from __future__ import annotations

import base64
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh

# ---------------------------------------------------------------------------
# VTK cell type codes (subset relevant to structural FEM)
# https://vtk.org/doc/nightly/html/vtkCellType_8h.html
# ---------------------------------------------------------------------------
VTK_VERTEX        = 1
VTK_LINE          = 3
VTK_TRIANGLE      = 5
VTK_QUAD          = 9
VTK_TETRA         = 10
VTK_HEXAHEDRON    = 12
VTK_WEDGE         = 13
VTK_PYRAMID       = 14
VTK_QUAD_QUADRATIC = 23   # 8-node quad
VTK_TETRA_QUADRATIC = 24  # 10-node tet
VTK_HEX_QUADRATIC  = 25   # 20-node hex

# Gmsh element type -> (VTK cell type, nodes per element)
_GMSH_TO_VTK: dict[int, tuple[int, int]] = {
    1:  (VTK_LINE,        2),
    2:  (VTK_TRIANGLE,    3),
    3:  (VTK_QUAD,        4),
    4:  (VTK_TETRA,       4),
    5:  (VTK_HEXAHEDRON,  8),
    6:  (VTK_WEDGE,       6),
    7:  (VTK_PYRAMID,     5),
    8:  (VTK_LINE,        3),   # 2nd-order line -> VTK_QUADRATIC_EDGE=21
    9:  (VTK_TRIANGLE,    6),   # 2nd-order tri  -> 22
    10: (VTK_QUAD_QUADRATIC, 9),
    11: (VTK_TETRA_QUADRATIC, 10),
    15: (VTK_VERTEX,      1),
    16: (VTK_QUAD_QUADRATIC, 8),
    17: (VTK_HEX_QUADRATIC, 20),
}


# ---------------------------------------------------------------------------
# Helper: numpy array -> base64-encoded binary string for VTU
# ---------------------------------------------------------------------------
def _encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 binary for VTU AppendedData / inline."""
    flat = np.ascontiguousarray(arr)
    raw = flat.tobytes()
    # VTK binary inline: 4-byte header with byte-count, then raw data
    header = struct.pack("<I", len(raw))
    return base64.b64encode(header + raw).decode("ascii")


# ---------------------------------------------------------------------------
# Core: write a .vtu file from raw arrays
# ---------------------------------------------------------------------------
def write_vtu(
    filename:   str | Path,
    points:     np.ndarray,                         # (N, 3) float64
    cells:      np.ndarray,                         # (M, npe) int — 0-based indices
    *,
    vtk_cell_type: int = VTK_QUAD,                  # single type for all cells
    point_data: dict[str, np.ndarray] | None = None,  # name -> (N,) or (N,3)
    cell_data:  dict[str, np.ndarray] | None = None,  # name -> (M,) or (M,3)
    binary:     bool = True,
) -> Path:
    """Write a VTK UnstructuredGrid (.vtu) file.

    Parameters
    ----------
    filename : path
        Output file path (should end in ``.vtu``).
    points : ndarray (N, 3)
        Node coordinates.
    cells : ndarray (M, npe)
        Element connectivity as **0-based** node indices.
    vtk_cell_type : int
        VTK cell type code (same for all cells).
    point_data : dict
        Nodal fields. Each value is (N,) for scalars or (N, 3) for vectors.
    cell_data : dict
        Element fields. Each value is (M,) for scalars or (M, 3) for vectors.
    binary : bool
        If True (default), encode data as base64 binary.  If False, write
        ASCII (larger files but human-readable for debugging).

    Returns
    -------
    Path to the written file.
    """
    filename = Path(filename)
    point_data = point_data or {}
    cell_data  = cell_data or {}

    nPoints = points.shape[0]
    nCells  = cells.shape[0]
    npe     = cells.shape[1]

    fmt = "binary" if binary else "ascii"

    # --- Root element ---
    root = ET.Element("VTKFile", type="UnstructuredGrid",
                      version="1.0", byte_order="LittleEndian")
    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece",
                          NumberOfPoints=str(nPoints),
                          NumberOfCells=str(nCells))

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------
    pts_el = ET.SubElement(piece, "Points")
    _add_data_array(pts_el, "Points", points.astype(np.float64),
                    n_components=3, fmt=fmt)

    # ------------------------------------------------------------------
    # Cells: connectivity + offsets + types
    # ------------------------------------------------------------------
    cells_el = ET.SubElement(piece, "Cells")

    # Connectivity (flat)
    conn_flat = cells.astype(np.int32).ravel()
    _add_data_array(cells_el, "connectivity", conn_flat, fmt=fmt)

    # Offsets (cumulative node count per cell)
    offsets = np.arange(1, nCells + 1, dtype=np.int32) * npe
    _add_data_array(cells_el, "offsets", offsets, fmt=fmt)

    # Types
    types = np.full(nCells, vtk_cell_type, dtype=np.uint8)
    _add_data_array(cells_el, "types", types, fmt=fmt)

    # ------------------------------------------------------------------
    # PointData (nodal fields)
    # ------------------------------------------------------------------
    if point_data:
        pd_el = ET.SubElement(piece, "PointData")
        for name, arr in point_data.items():
            arr = np.asarray(arr, dtype=np.float64)
            nc = arr.shape[1] if arr.ndim == 2 else 1
            _add_data_array(pd_el, name, arr, n_components=nc, fmt=fmt)

    # ------------------------------------------------------------------
    # CellData (element fields)
    # ------------------------------------------------------------------
    if cell_data:
        cd_el = ET.SubElement(piece, "CellData")
        for name, arr in cell_data.items():
            arr = np.asarray(arr, dtype=np.float64)
            nc = arr.shape[1] if arr.ndim == 2 else 1
            _add_data_array(cd_el, name, arr, n_components=nc, fmt=fmt)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(filename), xml_declaration=True, encoding="utf-8")

    return filename


# ---------------------------------------------------------------------------
# Helper: add a <DataArray> element
# ---------------------------------------------------------------------------
_NP_TO_VTK_TYPE = {
    np.dtype("float64"): "Float64",
    np.dtype("float32"): "Float32",
    np.dtype("int32"):   "Int32",
    np.dtype("int64"):   "Int64",
    np.dtype("uint8"):   "UInt8",
}


def _add_data_array(parent: ET.Element, name: str, arr: np.ndarray,
                    n_components: int = 1, fmt: str = "binary") -> ET.Element:
    """Append a <DataArray> XML element under *parent*."""
    arr = np.ascontiguousarray(arr)
    vtk_type = _NP_TO_VTK_TYPE.get(arr.dtype, "Float64")
    if arr.dtype not in _NP_TO_VTK_TYPE:
        arr = arr.astype(np.float64)

    attrib: dict[str, str] = {
        "type": vtk_type,
        "Name": name,
        "NumberOfComponents": str(n_components),
        "format": fmt,
    }
    da = ET.SubElement(parent, "DataArray", attrib)

    if fmt == "binary":
        da.text = _encode_array(arr)
    else:
        flat = arr.ravel()
        da.text = " ".join(f"{v}" for v in flat)

    return da


# ---------------------------------------------------------------------------
# Convenience: write multiple steps (time series) for animation
# ---------------------------------------------------------------------------
def write_vtu_series(
    base_name: str,
    points:    np.ndarray,
    cells:     np.ndarray,
    *,
    vtk_cell_type: int = VTK_QUAD,
    steps:     list[dict],  # each dict: {"time": float, "point_data": {...}, "cell_data": {...}}
) -> list[Path]:
    """Write a time series of .vtu files + a .pvd collection file.

    ParaView opens the ``.pvd`` file and treats each ``.vtu`` as a
    time step, enabling the animation toolbar.

    Parameters
    ----------
    base_name : str
        Base filename without extension (e.g. ``"modal"``).
        Files will be ``modal_000.vtu``, ``modal_001.vtu``, … and ``modal.pvd``.
    steps : list[dict]
        Each entry has ``"time"`` (float), ``"point_data"`` (dict), and
        optionally ``"cell_data"`` (dict).

    Returns
    -------
    List of written file paths (VTU files + PVD file).
    """
    base = Path(base_name)
    stem = base.stem
    parent = base.parent

    vtu_files = []
    for i, step in enumerate(steps):
        vtu_name = parent / f"{stem}_{i:03d}.vtu"
        write_vtu(
            vtu_name, points, cells,
            vtk_cell_type=vtk_cell_type,
            point_data=step.get("point_data", {}),
            cell_data=step.get("cell_data", {}),
        )
        vtu_files.append(vtu_name)

    # --- PVD collection file ---
    pvd_path = parent / f"{stem}.pvd"
    root = ET.Element("VTKFile", type="Collection", version="1.0")
    coll = ET.SubElement(root, "Collection")
    for i, (step, vtu) in enumerate(zip(steps, vtu_files)):
        ET.SubElement(coll, "DataSet",
                      timestep=str(step.get("time", float(i))),
                      file=str(vtu.name))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(pvd_path), xml_declaration=True, encoding="utf-8")

    return vtu_files + [pvd_path]


# ---------------------------------------------------------------------------
# High-level class that integrates with apeGmsh
# ---------------------------------------------------------------------------
class VTKExport:
    """Convenience wrapper: accumulate fields, then write one .vtu.

    >>> vtk = VTKExport(g)             # g is a apeGmsh instance
    >>> vtk.add_node_scalar("T", T)
    >>> vtk.add_node_vector("u", disp)
    >>> vtk.add_elem_scalar("sig_xx", sig)
    >>> vtk.write("results.vtu")
    """

    def __init__(self, ctx: apeGmsh, dim: int | None = 2) -> None:
        fem = ctx.mesh.queries.get_fem_data(dim=dim)
        self._node_coords = fem.nodes.coords
        self._elem_tags   = fem.elements.ids
        self._node_ids    = fem.nodes.ids
        self._fem         = fem

        # Build node-ID -> 0-based array index mapping
        tag_to_idx = {int(t): i for i, t in enumerate(fem.nodes.ids)}

        # Build per-group 0-based connectivity and VTK types
        self._cell_blocks: list[tuple[np.ndarray, int]] = []
        for group in fem.elements:
            if group.connectivity.size == 0:
                continue
            conn_0 = np.array(
                [[tag_to_idx[int(n)] for n in row]
                 for row in group.connectivity])
            vtk_type = {
                (1, 2): VTK_LINE,
                (2, 3): VTK_TRIANGLE, (2, 4): VTK_QUAD,
                (3, 4): 10, (3, 6): VTK_WEDGE, (3, 8): VTK_HEXAHEDRON,
            }.get((group.dim, group.npe), VTK_QUAD)
            self._cell_blocks.append((conn_0, vtk_type))

        self._point_data: dict[str, np.ndarray] = {}
        self._cell_data:  dict[str, np.ndarray] = {}

    # ---- Add fields -------------------------------------------------------

    def add_node_scalar(self, name: str, data: np.ndarray) -> None:
        """Add a nodal scalar field (one value per node)."""
        self._point_data[name] = np.asarray(data, dtype=np.float64).ravel()

    def add_node_vector(self, name: str, data: np.ndarray) -> None:
        """Add a nodal vector field (3 components per node).

        If data is (N, 2), a zero z-component is appended automatically.
        """
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            raise ValueError(f"Vector field '{name}' must be 2D, got shape {arr.shape}")
        if arr.shape[1] == 2:
            arr = np.column_stack([arr, np.zeros(arr.shape[0])])
        self._point_data[name] = arr

    def add_elem_scalar(self, name: str, data: np.ndarray) -> None:
        """Add an element scalar field (one value per element)."""
        self._cell_data[name] = np.asarray(data, dtype=np.float64).ravel()

    def add_elem_vector(self, name: str, data: np.ndarray) -> None:
        """Add an element vector field (3 components per element)."""
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape[1] == 2:
            arr = np.column_stack([arr, np.zeros(arr.shape[0])])
        self._cell_data[name] = arr

    # ---- Internal helpers --------------------------------------------------

    def _primary_conn_and_type(self):
        """Return (conn_0, vtk_type) for single-type meshes.

        For mixed-type meshes, uses the first block.
        """
        if self._cell_blocks:
            return self._cell_blocks[0]
        return np.empty((0, 0), dtype=np.int64), VTK_QUAD

    # ---- Write ------------------------------------------------------------

    def write(self, filename: str | Path = "results.vtu") -> Path:
        """Write accumulated fields to a .vtu file."""
        conn_0, vtk_type = self._primary_conn_and_type()
        return write_vtu(
            filename,
            self._node_coords,
            conn_0,
            vtk_cell_type=vtk_type,
            point_data=self._point_data,
            cell_data=self._cell_data,
        )

    def write_mode_series(self, base_name: str,
                          mode_shapes: list[np.ndarray],
                          frequencies: list[float]) -> list[Path]:
        """Write mode shapes as a time-series PVD for ParaView animation."""
        steps = []
        for i, (phi, freq) in enumerate(zip(mode_shapes, frequencies)):
            phi3 = np.asarray(phi, dtype=np.float64)
            if phi3.ndim == 2 and phi3.shape[1] > 3:
                phi3 = phi3[:, :3]
            if phi3.ndim == 2 and phi3.shape[1] == 2:
                phi3 = np.column_stack([phi3, np.zeros(phi3.shape[0])])

            mag = np.sqrt(np.sum(phi3**2, axis=1))

            steps.append({
                "time": freq,
                "point_data": {
                    "ModeShape": phi3,
                    "Magnitude": mag,
                },
            })

        conn_0, vtk_type = self._primary_conn_and_type()
        return write_vtu_series(
            base_name,
            self._node_coords,
            conn_0,
            vtk_cell_type=vtk_type,
            steps=steps,
        )

    def __repr__(self) -> str:
        n_cells = sum(c.shape[0] for c, _ in self._cell_blocks)
        return (f"<VTKExport: {self._node_coords.shape[0]} nodes, "
                f"{n_cells} cells, "
                f"{len(self._point_data)} point fields, "
                f"{len(self._cell_data)} cell fields>")
