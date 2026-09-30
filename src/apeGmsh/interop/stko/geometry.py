"""Extract the OCC geometry of an STKO ``.scd`` document as BREP.

STKO stores its geometry as one OCC compound in binary BRep
(``GEOMETRIES/COMPOUND_SHAPE``); each ``GEOMETRIES/GEOM_<id>`` is one of
the compound's children (``SHAPE_ID``, 1-based). Reading binary BRep
needs OpenCASCADE's Python bindings (``pip install cadquery-ocp``, the
``apeGmsh[stko]`` extra); the result is text BRep, which
``g.model.io.load_brep`` imports.

Sub-shape order survives the round trip for faces and edges (STKO face
``i`` is gmsh face tag ``i + 1``, edge likewise); vertices are reordered,
so match them by coordinate (``ScdModel.mesh.vertex_nodes``).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py

from .model import ScdModel


def write_brep(
    scd: ScdModel | str | Path,
    out: str | Path,
    *,
    geometry: int | None = None,
) -> Path:
    """Write the document's geometry to a text BREP file.

    Parameters
    ----------
    scd
        An :class:`ScdModel` or the ``.scd`` path.
    out
        The ``.brep`` file to write.
    geometry
        A geometry ID (``GEOMETRIES/GEOM_<id>``) to write alone; the
        default writes the whole compound.
    """
    try:
        from OCP.BinTools import BinTools
        from OCP.BRepTools import BRepTools
        from OCP.TopoDS import TopoDS_Iterator, TopoDS_Shape
    except ImportError as exc:
        raise ImportError(
            "write_brep needs OpenCASCADE's Python bindings: "
            "pip install cadquery-ocp  (or apeGmsh[stko])"
        ) from exc

    path = scd.path if isinstance(scd, ScdModel) else Path(scd)
    with h5py.File(path, "r") as f:
        blob = f["GEOMETRIES/COMPOUND_SHAPE"][()].tobytes()
        shape_index = None
        if geometry is not None:
            key = f"GEOMETRIES/GEOM_{geometry}"
            if key not in f:
                raise KeyError(f"{path.name} has no geometry {geometry}")
            shape_index = int(f[key].attrs["SHAPE_ID"].ravel()[0])

    compound = TopoDS_Shape()
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "compound.bin"
        binary.write_bytes(blob)
        BinTools.Read_s(compound, str(binary))

    shape = compound
    if shape_index is not None:
        children = TopoDS_Iterator(compound)
        for _ in range(shape_index - 1):
            children.Next()
        if not children.More():
            raise ValueError(
                f"{path.name}: geometry {geometry} is child {shape_index} "
                f"of a compound with fewer children"
            )
        shape = children.Value()

    out = Path(out)
    if not BRepTools.Write_s(shape, str(out)):
        raise OSError(f"could not write {out}")
    return out
