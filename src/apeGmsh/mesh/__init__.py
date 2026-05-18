from .Mesh import (
    Mesh,
    Algorithm2D,
    Algorithm3D,
    MeshAlgorithm2D,
    MeshAlgorithm3D,
    ALGORITHM_2D,
    ALGORITHM_3D,
    OptimizeMethod,
)
from .Partition import Partition
from ._mesh_partitioning import RenumberResult, PartitionInfo
from .PhysicalGroups import PhysicalGroups
from .MeshSelectionSet import MeshSelectionSet, MeshSelectionStore
from .View import View
from .FEMData import FEMData, MeshInfo
from ._element_types import ElementTypeInfo
from ._group_set import NamedGroupSet, PhysicalGroupSet, LabelSet
# Relocated to apeGmsh._kernel (selection-unification-v2 P1-K, the
# keystone cycle-break).  Re-exported here (a downward mesh -> _kernel
# edge — the intended layering direction) so the public
# ``apeGmsh.mesh`` surface is byte-stable for callers.
from .._kernel.payloads import NodeResult, ElementGroup, GroupResult
from .._kernel.record_sets import (
    ConstraintKind, LoadKind,
    NodeConstraintSet, SurfaceConstraintSet,
    NodalLoadSet, ElementLoadSet, MassSet,
)

__all__ = [
    "Mesh",
    "Algorithm2D", "Algorithm3D",
    "MeshAlgorithm2D", "MeshAlgorithm3D",
    "ALGORITHM_2D", "ALGORITHM_3D",
    "OptimizeMethod",
    "Partition", "RenumberResult", "PartitionInfo", "PhysicalGroups",
    "MeshSelectionSet", "MeshSelectionStore",
    "View",
    "FEMData", "MeshInfo", "NodeResult",
    "ElementTypeInfo", "ElementGroup", "GroupResult",
    "NamedGroupSet", "PhysicalGroupSet", "LabelSet",
    "ConstraintKind", "LoadKind",
    "NodeConstraintSet", "SurfaceConstraintSet",
    "NodalLoadSet", "ElementLoadSet", "MassSet",
]
