from __future__ import annotations

from typing import TYPE_CHECKING

from ._session import _SessionBase

if TYPE_CHECKING:
    from .viz.Inspect import Inspect
    from .core.Model import Model
    from .mesh.Mesh import Mesh
    from .mesh.MshLoader import MshLoader
    from .mesh.PhysicalGroups import PhysicalGroups
    from .mesh.Partition import Partition
    from .mesh.View import View
    from .solvers.Gmsh2OpenSees import Gmsh2OpenSees
    from .solvers.OpenSees import OpenSees
    from .viz.Plot import Plot


class pyGmsh(_SessionBase):
    """Standalone single-model Gmsh session with all composites.

    Parameters
    ----------
    model_name : str
        Name passed to ``gmsh.model.add()``.
    verbose : bool
        If True, composites print diagnostic messages.
    """

    _COMPOSITES = (
        ("inspect",   ".viz.Inspect",           "Inspect",        False),
        ("model",     ".core.Model",            "Model",          False),
        ("mesh",      ".mesh.Mesh",             "Mesh",           False),
        ("loader",    ".mesh.MshLoader",        "MshLoader",      False),
        ("physical",        ".mesh.PhysicalGroups",    "PhysicalGroups",    False),
        ("mesh_selection",  ".mesh.MeshSelectionSet",  "MeshSelectionSet",  False),
        ("partition",       ".mesh.Partition",         "Partition",         False),
        ("view",      ".mesh.View",             "View",           False),
        ("g2o",       ".solvers.Gmsh2OpenSees", "Gmsh2OpenSees",  False),
        ("opensees",  ".solvers.OpenSees",      "OpenSees",       False),
        ("plot",      ".viz.Plot",              "Plot",           True),
    )

    # -- Static type declarations for composites (created at runtime by begin()) --
    inspect: Inspect
    model: Model
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

    # ------------------------------------------------------------------
    # model_name setter (backward compat: g.model_name = "foo")
    # ------------------------------------------------------------------

    @_SessionBase.model_name.getter  # type: ignore[attr-defined]
    def model_name(self) -> str:  # type: ignore[no-redef]
        return self.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.name = value

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    @property
    def _initialized(self) -> bool:
        """Legacy alias — use ``is_active`` instead."""
        return self._active

    @_initialized.setter
    def _initialized(self, value: bool) -> None:
        self._active = value

    def initialize(self) -> "pyGmsh":
        """Alias for ``begin()``."""
        return self.begin()  # type: ignore[return-value]

    def finalize(self) -> None:
        """Alias for ``end()``."""
        self.end()

    # ------------------------------------------------------------------
    # Convenience delegates
    # ------------------------------------------------------------------

    def remove_duplicates(
        self,
        *,
        tolerance: float | None = None,
        sync: bool = True,
    ) -> None:
        """Delegate to ``self.model.remove_duplicates()``."""
        self.model.remove_duplicates(tolerance=tolerance, sync=sync)

    def make_conformal(
        self,
        *,
        dims: list[int] | None = None,
        tolerance: float | None = None,
        sync: bool = True,
    ) -> None:
        """Delegate to ``self.model.make_conformal()``."""
        self.model.make_conformal(dims=dims, tolerance=tolerance, sync=sync)
