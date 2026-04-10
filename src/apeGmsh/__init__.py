"""
apeGmsh — Gmsh wrapper for structural FEM workflows.
====================================================

Two usage modes:

1. **Standalone** (single-model, quick prototyping)::

       from apeGmsh import apeGmsh

       g = apeGmsh(model_name="plate", verbose=True)
       g.begin()
       g.model.add_point(0, 0, 0)
       ...
       g.end()

2. **Multi-part** (assembly workflow via ``g.parts``)::

       from apeGmsh import apeGmsh, Part

       web = Part("web")
       web.begin()
       web.model.add_box(0, 0, 0, 1, 0.5, 10)
       web.save("web.step")
       web.end()

       g = apeGmsh(model_name="bridge")
       g.begin()
       g.parts.add(web, label="web")
       g.parts.fragment_all()
       g.constraints.equal_dof("web", "slab", tolerance=1e-3)
       g.mesh.generate(dim=2)
       g.end()
"""

from apeGmsh._session import _SessionBase
from apeGmsh._core import apeGmsh
from apeGmsh.core.Part import Part
from apeGmsh.core._parts_registry import PartsRegistry, Instance
from apeGmsh.core.ConstraintsComposite import ConstraintsComposite
from apeGmsh.mesh.FEMData import FEMData, MeshInfo, PhysicalGroupSet
from apeGmsh.mesh.Mesh import (
    Algorithm2D,
    Algorithm3D,
    MeshAlgorithm2D,
    MeshAlgorithm3D,
    ALGORITHM_2D,
    ALGORITHM_3D,
    OptimizeMethod,
)
from apeGmsh.mesh.MshLoader import MshLoader
from apeGmsh.solvers.Numberer import Numberer, NumberedMesh
from apeGmsh.viz.Selection import Selection, SelectionComposite
from apeGmsh.viewers.model_viewer import ModelViewer
from apeGmsh.viewers._mesh_viewer import MeshViewer as MeshViewerV2

# Backward-compatible aliases
SelectionPicker = ModelViewer
MeshViewer = MeshViewerV2
from apeGmsh.results.Results import Results
import apeGmsh.solvers.Constraints as Constraints

__all__ = [
    "_SessionBase",
    "apeGmsh",
    "Part",
    "PartsRegistry",
    "Instance",
    "ConstraintsComposite",
    "FEMData",
    "MeshInfo",
    "PhysicalGroupSet",
    "Algorithm2D",
    "Algorithm3D",
    "MeshAlgorithm2D",
    "MeshAlgorithm3D",
    "ALGORITHM_2D",
    "ALGORITHM_3D",
    "OptimizeMethod",
    "MshLoader",
    "Results",
    "Numberer",
    "NumberedMesh",
    "Selection",
    "SelectionComposite",
    "ModelViewer",
    "MeshViewerV2",
    "SelectionPicker",
    "MeshViewer",
    "Constraints",
]