"""
MshLoader — Load external ``.msh`` files into the apeGmsh pipeline.
===================================================================

Two usage modes:

1. **Standalone** (just give me a FEMData, no ceremony)::

       from apeGmsh import MshLoader

       fem = MshLoader.load("bridge.msh", dim=2)

   Opens a temporary Gmsh session, merges, extracts, and tears down.
   Returns a self-contained ``FEMData`` — no session management needed.

2. **Composite** (load into a live session for plotting, inspection, etc.)::

       from apeGmsh import apeGmsh

       g = apeGmsh(model_name="imported")
       g.begin()
       fem = g.loader.from_msh("bridge.msh", dim=2)
       g.physical.get_all()   # live queries still work
       g.end()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase
    from apeGmsh.mesh.FEMData import FEMData


class MshLoader:
    """
    Load ``.msh`` files and produce solver-ready :class:`FEMData`.

    Can be used standalone via the :meth:`load` classmethod, or as a
    composite on a ``apeGmsh`` / ``Assembly`` session via ``g.loader``.

    Parameters
    ----------
    parent : _SessionBase or None
        The owning session when used as a composite.  ``None`` when
        used standalone.
    """

    def __init__(self, parent: "_SessionBase | None" = None) -> None:
        self._parent = parent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent is not None and self._parent._verbose:
            print(f"[MshLoader] {msg}")

    @staticmethod
    def _validate_path(path: str | Path) -> Path:
        """Resolve and check the .msh file exists."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MSH file not found: {p}")
        return p

    @staticmethod
    def _log_fem(fem: "FEMData", label: str, verbose: bool) -> None:
        """Print a summary of the loaded FEMData."""
        if not verbose:
            return
        print(
            f"[MshLoader] {label} → "
            f"{fem.info.n_nodes} nodes, "
            f"{fem.info.n_elems} elements, "
            f"bw={fem.info.bandwidth}"
        )
        pg_list = fem.physical.get_all()
        if pg_list:
            print(
                f"[MshLoader] physical groups ({len(pg_list)}): "
                + ", ".join(
                    f"({d},{t}) {fem.physical.get_name(d, t)!r}"
                    for d, t in pg_list
                )
            )
        else:
            print("[MshLoader] no physical groups found in the .msh file")

    # ------------------------------------------------------------------
    # Standalone class method — no session needed
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        dim: int = 2,
        verbose: bool = False,
    ) -> "FEMData":
        """
        Load a ``.msh`` file and return a :class:`FEMData`.

        Manages its own Gmsh session internally — no ``apeGmsh``
        instance, no ``begin()``/``end()`` needed.  Supports MSH2
        and MSH4 formats.

        Parameters
        ----------
        path : str or Path
            Path to the ``.msh`` file.
        dim : int
            Element dimension to extract (1 = lines, 2 = tri/quad,
            3 = tet/hex).  Default is 2.
        verbose : bool
            Print a summary of what was loaded.

        Returns
        -------
        FEMData
            Self-contained solver-ready mesh data with physical
            groups, mesh statistics, and connectivity.

        Example
        -------
        ::

            from apeGmsh import MshLoader, Numberer

            fem = MshLoader.load("bridge.msh", dim=2)

            print(fem.info)
            print(fem.physical.summary())

            numb = Numberer(fem)
            data = numb.renumber(method="rcm")
        """
        from ._fem_extract import build_fem_data

        p = cls._validate_path(path)

        gmsh.initialize()
        try:
            gmsh.model.add(p.stem)
            gmsh.merge(str(p))
            fem = build_fem_data(dim=dim)
        finally:
            gmsh.finalize()

        cls._log_fem(fem, f"load({p.name!r})", verbose)
        return fem

    # ------------------------------------------------------------------
    # Composite method — loads into the active session
    # ------------------------------------------------------------------

    def from_msh(
        self,
        path: str | Path,
        *,
        dim: int = 2,
    ) -> "FEMData":
        """
        Load a ``.msh`` file into the **active** Gmsh session.

        The mesh is merged via ``gmsh.merge()``, so all composites
        (``g.physical``, ``g.plot``, ``g.inspect``, etc.) remain
        usable afterwards.

        Parameters
        ----------
        path : str or Path
            Path to the ``.msh`` file.
        dim : int
            Element dimension to extract.  Default is 2.

        Returns
        -------
        FEMData
            Self-contained solver-ready mesh data.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        RuntimeError
            If no Gmsh session is active (call ``g.begin()`` first).

        Example
        -------
        ::

            g = apeGmsh(model_name="imported")
            g.begin()

            fem = g.loader.from_msh("model.msh", dim=2)
            print(fem.physical.summary())

            g.end()
        """
        from ._fem_extract import build_fem_data

        p = self._validate_path(path)

        if self._parent is None or not self._parent.is_active:
            raise RuntimeError(
                "No active Gmsh session. Call g.begin() first, "
                "or use MshLoader.load() for standalone loading."
            )

        self._log(f"merging {p.name} ...")
        gmsh.merge(str(p))

        fem = build_fem_data(dim=dim)

        verbose = self._parent._verbose if self._parent else False
        self._log_fem(fem, f"from_msh({p.name!r})", verbose)

        return fem
