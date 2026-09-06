"""apeGmsh interop — import analytical structural models from external tools.

Phase 2 entry point for the apeETABS -> apeGmsh pipeline (ADR 0009).
"""
from .etabs_import import (
    AreaGroup,
    DiaphragmSpec,
    FrameGroup,
    ImportResult,
    RestraintGroup,
    SpringGround,
    apply_subgrade_springs,
    build_opensees,
    import_structural_model,
)
from .model import AREA_MODIFIER_NAMES, AreaModifiers, StructuralModel
from .solve import SolveResult, solve_and_extract

__all__ = [
    "StructuralModel",
    "AreaModifiers",
    "AREA_MODIFIER_NAMES",
    "import_structural_model",
    "apply_subgrade_springs",
    "build_opensees",
    "solve_and_extract",
    "SolveResult",
    "ImportResult",
    "FrameGroup",
    "AreaGroup",
    "RestraintGroup",
    "DiaphragmSpec",
    "SpringGround",
]
