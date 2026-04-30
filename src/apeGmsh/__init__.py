"""
apeGmsh — Gmsh wrapper for structural FEM workflows.
====================================================

Composition-based API with sub-composites for focused surfaces:

1. **Standalone** (single-model, quick prototyping)::

       from apeGmsh import apeGmsh

       g = apeGmsh(model_name="plate", verbose=True)
       g.begin()
       p = g.model.geometry.add_point(0, 0, 0)
       ...
       g.end()

2. **Multi-part** (assembly workflow via ``g.parts``)::

       from apeGmsh import apeGmsh, Part

       web = Part("web")
       web.begin()
       web.model.geometry.add_box(0, 0, 0, 1, 0.5, 10)
       web.save("web.step")
       web.end()

       g = apeGmsh(model_name="bridge")
       g.begin()
       g.parts.add(web, label="web")
       g.parts.fragment_all()
       g.constraints.equal_dof("web", "slab", tolerance=1e-3)
       with g.loads.pattern("dead"):
           g.loads.gravity("web", g=(0, 0, -9.81), density=7850)
       g.masses.volume("web", density=7850)
       g.mesh.generation.generate(dim=3)
       fem = g.mesh.queries.get_fem_data(dim=3)
       g.end()
"""

from apeGmsh._session import _SessionBase
from apeGmsh._core import apeGmsh
from apeGmsh.core.Part import Part
from apeGmsh.core._parts_registry import PartsRegistry, Instance
from apeGmsh.core.ConstraintsComposite import ConstraintsComposite
from apeGmsh.mesh.FEMData import FEMData, MeshInfo
from apeGmsh.mesh._group_set import PhysicalGroupSet, LabelSet
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
from apeGmsh.results.Results import Results
import apeGmsh.solvers.Constraints as Constraints
from apeGmsh.solvers.Numberer import Numberer, NumberedMesh
from apeGmsh.mesh._mesh_partitioning import RenumberResult, PartitionInfo
from apeGmsh.viewers.mesh_viewer import MeshViewer
from apeGmsh.viewers.model_viewer import ModelViewer
from apeGmsh.viewers.results_viewer import ResultsViewer
from apeGmsh.viewers import settings, theme_editor
from apeGmsh.viz.NotebookPreview import preview
from apeGmsh.viz.Selection import Selection, SelectionComposite

# Backward-compatible alias (SelectionPicker was the pre-v1 name)
SelectionPicker = ModelViewer

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
    "LabelSet",
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
    "RenumberResult",
    "PartitionInfo",
    "Selection",
    "SelectionComposite",
    "ModelViewer",
    "MeshViewer",
    "ResultsViewer",
    "SelectionPicker",
    "Constraints",
    "settings",
    "theme_editor",
    "preview",
]