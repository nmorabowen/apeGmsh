"""Diagram catalogue + Director + Registry.

Phase 0 shipped the scaffolding. Phase 1 adds ``ContourDiagram``.
Subsequent phases add line force, fiber, layer, vector glyph, gauss
marker, and spring force diagrams.
"""
from ._base import Diagram, DiagramSpec, NoDataError
from ._contour import ContourDiagram
from ._director import ResultsDirector, TimeMode
from ._fiber_section import FiberSectionDiagram
from ._gauss_marker import GaussPointDiagram
from ._layer_stack import LayerStackDiagram
from ._line_force import LineForceDiagram
from ._loads import LoadsDiagram
from ._reactions import ReactionsDiagram
from ._registry import DiagramRegistry
from ._sand import SandDiagram
from ._section_cut import SectionCutDiagram
from ._selectors import SlabSelector, normalize as normalize_selector
from ._spring_force import SpringForceDiagram
from ._styles import (
    ContourStyle,
    DiagramStyle,
    FiberSectionStyle,
    GaussMarkerStyle,
    LayerStackStyle,
    LineForceStyle,
    LoadsStyle,
    ReactionsStyle,
    SandStyle,
    SectionCutStyle,
    SpringForceStyle,
    VectorGlyphStyle,
)
from ._vector_glyph import VectorGlyphDiagram

__all__ = [
    "ContourDiagram",
    "ContourStyle",
    "Diagram",
    "DiagramRegistry",
    "DiagramSpec",
    "DiagramStyle",
    "FiberSectionDiagram",
    "FiberSectionStyle",
    "GaussMarkerStyle",
    "GaussPointDiagram",
    "LayerStackDiagram",
    "LayerStackStyle",
    "LineForceDiagram",
    "LineForceStyle",
    "LoadsDiagram",
    "LoadsStyle",
    "NoDataError",
    "ReactionsDiagram",
    "ReactionsStyle",
    "ResultsDirector",
    "SandDiagram",
    "SandStyle",
    "SectionCutDiagram",
    "SectionCutStyle",
    "SlabSelector",
    "SpringForceDiagram",
    "SpringForceStyle",
    "TimeMode",
    "VectorGlyphDiagram",
    "VectorGlyphStyle",
    "normalize_selector",
]
