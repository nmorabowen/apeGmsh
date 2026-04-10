"""
pyGmsh — Gmsh wrapper for structural FEM workflows.
====================================================

Two usage modes:

1. **Standalone** (single-model, quick prototyping)::

       from pyGmsh import pyGmsh

       g = pyGmsh(model_name="plate", verbose=True)
       g.begin()
       g.model.add_point(0, 0, 0)
       ...
       g.end()

2. **Multi-part** (assembly workflow via ``g.parts``)::

       from pyGmsh import pyGmsh, Part

       web = Part("web")
       web.begin()
       web.model.add_box(0, 0, 0, 1, 0.5, 10)
       web.save("web.step")
       web.end()

       g = pyGmsh(model_name="bridge")
       g.begin()
       g.parts.add(web, label="web")
       g.parts.fragment_all()
       g.constraints.equal_dof("web", "slab", tolerance=1e-3)
       g.mesh.generate(dim=2)
       g.end()
"""

from pyGmsh._session import _SessionBase
from pyGmsh._core import pyGmsh
from pyGmsh.core.Part import Part
from pyGmsh.core._parts_registry import PartsRegistry, Instance
from pyGmsh.core.ConstraintsComposite import ConstraintsComposite
from pyGmsh.mesh.FEMData import FEMData, MeshInfo, PhysicalGroupSet
from pyGmsh.mesh.MshLoader import MshLoader
from pyGmsh.solvers.Numberer import Numberer, NumberedMesh
from pyGmsh.viz.Selection import Selection, SelectionComposite
from pyGmsh.viewers.model_viewer import ModelViewer
from pyGmsh.viewers._mesh_viewer import MeshViewer as MeshViewerV2

# Backward-compatible aliases
SelectionPicker = ModelViewer
MeshViewer = MeshViewerV2
from pyGmsh.results.Results import Results
import pyGmsh.solvers.Constraints as Constraints

__all__ = [
    "_SessionBase",
    "pyGmsh",
    "Part",
    "PartsRegistry",
    "Instance",
    "ConstraintsComposite",
    "FEMData",
    "MeshInfo",
    "PhysicalGroupSet",
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