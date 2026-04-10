from __future__ import annotations

from typing import TYPE_CHECKING

from ._session import _SessionBase

if TYPE_CHECKING:
    from .viz.Inspect import Inspect
    from .core.Model import Model
    from .core.ConstraintsComposite import ConstraintsComposite
    from .core.LoadsComposite import LoadsComposite
    from .core.MassesComposite import MassesComposite
    from .core._parts_registry import PartsRegistry
    from .mesh.Mesh import Mesh
    from .mesh.MshLoader import MshLoader
    from .mesh.PhysicalGroups import PhysicalGroups
    from .mesh.MeshSelectionSet import MeshSelectionSet
    from .mesh.Partition import Partition
    from .mesh.View import View
    from .solvers.Gmsh2OpenSees import Gmsh2OpenSees
    from .solvers.OpenSees import OpenSees
    from .viz.Plot import Plot


class apeGmsh(_SessionBase):
    """Standalone single-model Gmsh session with all composites.

    Parameters
    ----------
    model_name : str
        Name passed to ``gmsh.model.add()``.
    verbose : bool
        If True, composites print diagnostic messages.
    """

    _COMPOSITES = (
        ("inspect",         ".viz.Inspect",                "Inspect",               False),
        ("model",           ".core.Model",                 "Model",                 False),
        ("parts",           ".core._parts_registry",       "PartsRegistry",         False),
        ("constraints",     ".core.ConstraintsComposite",  "ConstraintsComposite",  False),
        ("loads",           ".core.LoadsComposite",        "LoadsComposite",        False),
        ("masses",          ".core.MassesComposite",       "MassesComposite",       False),
        ("mesh",            ".mesh.Mesh",                  "Mesh",                  False),
        ("loader",          ".mesh.MshLoader",             "MshLoader",             False),
        ("physical",        ".mesh.PhysicalGroups",        "PhysicalGroups",        False),
        ("mesh_selection",  ".mesh.MeshSelectionSet",      "MeshSelectionSet",      False),
        ("partition",       ".mesh.Partition",             "Partition",             False),
        ("view",            ".mesh.View",                  "View",                  False),
        ("g2o",             ".solvers.Gmsh2OpenSees",      "Gmsh2OpenSees",         False),
        ("opensees",        ".solvers.OpenSees",           "OpenSees",              False),
        ("plot",            ".viz.Plot",                   "Plot",                  True),
    )

    # -- Static type declarations for composites (created at runtime by begin()) --
    inspect: Inspect
    model: Model
    parts: PartsRegistry
    constraints: ConstraintsComposite
    loads: LoadsComposite
    masses: MassesComposite
    mesh: Mesh
    loader: MshLoader
    physical: PhysicalGroups
    mesh_selection: MeshSelectionSet
    partition: Partition
    view: View
    g2o: Gmsh2OpenSees
    opensees: OpenSees
    plot: Plot

    def __init__(
        self,
        *,
        model_name: str = "ModelName",
        verbose: bool = False,
    ) -> None:
        super().__init__(name=model_name, verbose=verbose)
