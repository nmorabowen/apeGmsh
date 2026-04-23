from __future__ import annotations

from typing import TYPE_CHECKING

from ._session import _SessionBase

if TYPE_CHECKING:
    from .viz.Inspect import Inspect
    from .core.Model import Model
    from .core.Labels import Labels
    from .core.ConstraintsComposite import ConstraintsComposite
    from .core.LoadsComposite import LoadsComposite
    from .core.MassesComposite import MassesComposite
    from .core._parts_registry import PartsRegistry
    from .sections._builder import SectionsBuilder
    from .mesh.Mesh import Mesh
    from .mesh.MshLoader import MshLoader
    from .mesh.PhysicalGroups import PhysicalGroups
    from .mesh.MeshSelectionSet import MeshSelectionSet
    from .mesh.Partition import Partition  # noqa: F401 (backward compat)
    from .mesh.View import View
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
        ("labels",          ".core.Labels",                "Labels",                False),
        ("sections",        ".sections._builder",          "SectionsBuilder",       False),
        ("parts",           ".core._parts_registry",       "PartsRegistry",         False),
        ("constraints",     ".core.ConstraintsComposite",  "ConstraintsComposite",  False),
        ("loads",           ".core.LoadsComposite",        "LoadsComposite",        False),
        ("masses",          ".core.MassesComposite",       "MassesComposite",       False),
        ("mesh",            ".mesh.Mesh",                  "Mesh",                  False),
        ("loader",          ".mesh.MshLoader",             "MshLoader",             False),
        ("physical",        ".mesh.PhysicalGroups",        "PhysicalGroups",        False),
        ("mesh_selection",  ".mesh.MeshSelectionSet",      "MeshSelectionSet",      False),
        # ("partition",    ".mesh.Partition",             "Partition",             False),
        # ^ Removed: consolidated into g.mesh.partitioning
        ("view",            ".mesh.View",                  "View",                  False),
        ("opensees",        ".solvers.OpenSees",           "OpenSees",              False),
        ("plot",            ".viz.Plot",                   "Plot",                  True),
    )

    # -- Static type declarations for composites (created at runtime by begin()) --
    inspect: Inspect
    model: Model
    labels: Labels
    sections: SectionsBuilder
    parts: PartsRegistry
    constraints: ConstraintsComposite
    loads: LoadsComposite
    masses: MassesComposite
    mesh: Mesh
    loader: MshLoader
    physical: PhysicalGroups
    mesh_selection: MeshSelectionSet
    # partition: Partition  # removed — use g.mesh.partitioning
    view: View
    opensees: OpenSees
    plot: Plot

    def __init__(
        self,
        *,
        model_name: str = "ModelName",
        verbose: bool = False,
    ) -> None:
        super().__init__(name=model_name, verbose=verbose)
        # Labels (Tier 1 naming) are auto-created from label= kwargs
        # on geometry methods in both Part and Assembly sessions.
        self._auto_pg_from_label = True
