"""Diagram catalogue + Director + Registry.

Phase 0 shipped the scaffolding. Phase 1 adds ``ContourDiagram`` and
``DeformedShapeDiagram``. Subsequent phases add line force, fiber,
layer, vector glyph, gauss marker, and spring force diagrams.
"""
from ._base import Diagram, DiagramSpec, NoDataError
from ._contour import ContourDiagram
from ._deformed_shape import DeformedShapeDiagram
from ._director import ResultsDirector, TimeMode
from ._fiber_section import FiberSectionDiagram
from ._gauss_marker import GaussPointDiagram
from ._layer_stack import LayerStackDiagram
from ._line_force import LineForceDiagram
from ._loads import LoadsDiagram
from ._reactions import ReactionsDiagram
from ._registry import DiagramRegistry
from ._selectors import SlabSelector, normalize as normalize_selector
from ._spring_force import SpringForceDiagram
from ._styles import (
    ContourStyle,
    DeformedShapeStyle,
    DiagramStyle,
    FiberSectionStyle,
    GaussMarkerStyle,
    LayerStackStyle,
    LineForceStyle,
    LoadsStyle,
    ReactionsStyle,
    SpringForceStyle,
    VectorGlyphStyle,
)
from ._vector_glyph import VectorGlyphDiagram

__all__ = [
    "ContourDiagram",
    "ContourStyle",
    "DeformedShapeDiagram",
    "DeformedShapeStyle",
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
    "SlabSelector",
    "SpringForceDiagram",
    "SpringForceStyle",
    "TimeMode",
    "VectorGlyphDiagram",
    "VectorGlyphStyle",
    "normalize_selector",
]
