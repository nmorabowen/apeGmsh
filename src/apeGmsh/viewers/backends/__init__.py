"""Render backends — concrete implementers of
:class:`~apeGmsh.viewers.scene_ir.RenderBackend`.

Each backend consumes the backend-agnostic ``scene_ir`` value types and
owns *all* VTK / pyvista / trame construction (ADR 0042, INV-2).  The
domain layer never imports a backend's render library directly.

``PyVistaBackend`` is the generic pyvista core; ``PyVistaQtBackend`` is
the reference (desktop) subclass; ``TrameBackend`` (web/Jupyter) is the
Phase R-C sibling. All three share the same ``scene_ir`` translation.
"""
from __future__ import annotations

from .pyvista_qt import (
    PyVistaBackend,
    PyVistaQtBackend,
    apply_visibility_mask,
    cellblocks_from_grid,
    mesh_layer_from_grid,
    mesh_layer_to_grid,
)
from .trame import TrameBackend

__all__ = [
    "PyVistaBackend",
    "PyVistaQtBackend",
    "TrameBackend",
    "mesh_layer_to_grid",
    "apply_visibility_mask",
    "cellblocks_from_grid",
    "mesh_layer_from_grid",
]
