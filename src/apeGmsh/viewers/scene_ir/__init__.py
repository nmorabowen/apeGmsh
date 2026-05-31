"""Backend-agnostic scene description for the viewer subsystem.

The render-side seam introduced by
[ADR 0042](../../opensees/architecture/decisions/0042-render-backend-seam.md):
domain logic emits :class:`SceneLayer` value types and drives a
:class:`RenderBackend`; backends (PyVistaQt desktop, trame web) own all
VTK/pyvista/trame construction.

INV-1: this package imports neither ``vtk`` nor ``pyvista``.  Enforced
by ``tests/test_scene_ir_pure.py``.
"""
from __future__ import annotations

from ._backend import LayerHandle, RenderBackend
from ._bbox import BBox
from ._layers import (
    CellBlocks,
    ColorSpec,
    GlyphLayer,
    LabelLayer,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarBarSpec,
    ScalarField,
    SceneLayer,
    VisibilityMask,
)
from ._pick import (
    BoxGesture,
    PickBackend,
    PickHit,
    PickMode,
    PickModifiers,
    PickRequest,
)
from ._targets import SelectionTarget, Substrate

__all__ = [
    # value types
    "PointSet",
    "CellBlocks",
    "ScalarField",
    "LutSpec",
    "ColorSpec",
    "VisibilityMask",
    "MeshLayer",
    "GlyphLayer",
    "LabelLayer",
    "ScalarBarSpec",
    "SceneLayer",
    # selection IR (ADR 0045)
    "Substrate",
    "SelectionTarget",
    "BBox",
    # pick IR (ADR 0044, Phase R-D)
    "PickMode",
    "PickModifiers",
    "PickRequest",
    "PickHit",
    "BoxGesture",
    # protocols
    "LayerHandle",
    "RenderBackend",
    "PickBackend",
]
