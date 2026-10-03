"""apeGmsh interop — read STKO ``.scd`` documents and translate them.

``read_scd`` reads the document (mesh, selection sets, properties,
conditions, interactions, definitions, analysis steps) into plain data;
``write_brep`` extracts its OCC geometry for ``g.model.io.load_brep``.

``translate_scd`` + ``build_opensees`` (ADR 0111) translate it, mesh-faithful
(STKO's own nodes and elements, same ids), into an apeGmsh session and an
``apeSees`` bridge; anything this version cannot translate raises
``UnsupportedSTKOTypes`` listing every offender.
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
from .translate import (
    build_opensees,
    collect_unsupported,
    declare_translation,
    translate_scd,
)
from .translate_types import (
    ConditionsPlan,
    Ignored,
    MeshMap,
    TranslateResult,
    Unsupported,
    UnsupportedSTKOTypes,
)

__all__ = [
    "ELEMENT_TYPES",
    "KINDS",
    "AnalysisElement",
    "Condition",
    "ConditionsPlan",
    "Geometry",
    "Ignored",
    "Interaction",
    "LocalAxes",
    "Mesh",
    "MeshElement",
    "MeshMap",
    "ScdModel",
    "SelectionSet",
    "SubShapeRef",
    "SubShapes",
    "TranslateResult",
    "Unsupported",
    "UnsupportedSTKOTypes",
    "XObject",
    "build_opensees",
    "collect_unsupported",
    "declare_translation",
    "read_scd",
    "translate_scd",
    "write_brep",
]
