from __future__ import annotations

from pathlib import Path
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
        # ("opensees", ".solvers.OpenSees", "OpenSees", False)
        # ^ Removed in PR γ of the Phase-8 bridge teardown.  The
        #   OpenSees deck is now constructed explicitly via
        #   ``apeGmsh.opensees.apeSees(fem)``, where ``fem`` is the
        #   FEMData snapshot from ``g.mesh.queries.get_fem_data(...)``.
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
    # opensees: removed in PR γ — use apeGmsh.opensees.apeSees(fem)
    plot: Plot

    def __init__(
        self,
        *,
        model_name: str = "ModelName",
        verbose: bool = False,
        save_to: str | Path | None = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__(name=model_name, verbose=verbose)
        # Labels (Tier 1 naming) are auto-created from label= kwargs
        # on geometry methods in both Part and Assembly sessions.
        self._auto_pg_from_label = True
        # Autosave configuration. ``save_to=None`` disables autosave;
        # otherwise ``end()`` writes the neutral-zone HDF5 to this path
        # before finalizing gmsh.  Manual ``g.save()`` uses the same path.
        self._save_to: Path | None = Path(save_to) if save_to else None
        self._overwrite: bool = overwrite

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Write the neutral-zone ``model.h5`` for this session.

        Persists what the session knows about the model: nodes,
        elements, physical groups, labels, constraints, loads, masses.
        Downstream solver enrichment (e.g. ``apeSees(fem).h5(p)``) is
        a separate user-driven action and not invoked here.

        Parameters
        ----------
        path : str, Path, or None
            Destination file.  ``None`` (default) uses the ``save_to``
            given to the constructor.  Raises if neither is set.

        Returns the resolved path.
        """
        target = Path(path) if path is not None else self._save_to
        if target is None:
            raise RuntimeError(
                "g.save() requires a path — either pass one explicitly "
                "or construct the session with save_to=<path>."
            )
        if target.exists() and not self._overwrite:
            raise FileExistsError(
                f"{target} already exists and overwrite=False."
            )
        self._do_save(target)
        return target

    def _do_save(self, path: Path) -> None:
        """Extract the broker snapshot and write it to ``path``."""
        from . import __version__ as _ver

        fem = self.mesh.queries.get_fem_data()
        fem.to_h5(
            str(path),
            model_name=self.name,
            apegmsh_version=_ver,
        )
