"""apeGmsh interop — read STKO ``.scd`` documents.

``read_scd`` reads the document (mesh, selection sets, properties,
conditions, interactions, definitions, analysis steps) into plain data;
``write_brep`` extracts its OCC geometry for ``g.model.io.load_brep``.
"""

from .geometry import write_brep
from .model import (
    ELEMENT_TYPES,
    KINDS,
    AnalysisElement,
    Condition,
    Geometry,
    Interaction,
    LocalAxes,
    Mesh,
    MeshElement,
    ScdModel,
    SelectionSet,
    SubShapeRef,
    SubShapes,
    XObject,
)
from .read_scd import read_scd

__all__ = [
    "ELEMENT_TYPES",
    "KINDS",
    "AnalysisElement",
    "Condition",
    "Geometry",
    "Interaction",
    "LocalAxes",
    "Mesh",
    "MeshElement",
    "ScdModel",
    "SelectionSet",
    "SubShapeRef",
    "SubShapes",
    "XObject",
    "read_scd",
    "write_brep",
]
