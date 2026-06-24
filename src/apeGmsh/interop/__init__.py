"""apeGmsh interop — import analytical structural models from external tools.

Phase 2 entry point for the apeETABS -> apeGmsh pipeline (ADR 0009).
"""
from .etabs_import import (
    AreaGroup,
    DiaphragmSpec,
    FrameGroup,
    ImportResult,
    RestraintGroup,
    build_opensees,
    import_structural_model,
)
from .model import StructuralModel

__all__ = [
    "StructuralModel",
    "import_structural_model",
    "build_opensees",
    "ImportResult",
    "FrameGroup",
    "AreaGroup",
    "RestraintGroup",
    "DiaphragmSpec",
]
