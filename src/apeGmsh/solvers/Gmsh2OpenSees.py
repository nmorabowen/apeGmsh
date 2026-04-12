"""
Gmsh2OpenSees  —  Optional convenience layer around jaabell's gmsh2opensees.

This module wraps the lightweight ``gmsh2opensees`` library by José Abell
(https://github.com/jaabell/gmsh2opensees) so that apeGmsh users can transfer
a mesh into an active OpenSees model with a single call.

Design philosophy
-----------------
* **Optional** — apeGmsh works perfectly fine without ``gmsh2opensees`` installed.
  Users who prefer full control can keep using ``g.mesh.queries.get_fem_data()`` and
  build their OpenSees model manually (the "manual path").
* **Thin wrapper** — we add safety checks and convenience but delegate the
  real work to ``gmsh2opensees``.
* **No solver dependency** — this module only *calls* gmsh2opensees when
  the user explicitly invokes a method.  ``import Gmsh2OpenSees`` alone does
  NOT import ``openseespy`` or ``gmsh2opensees``.

Typical usage
-------------
>>> with apeGmsh(model_name="demo") as g:
...     # ... build geometry, set physical groups, generate mesh ...
...     g.g2o.transfer()          # one-liner: nodes + elements -> OpenSees
...     g.g2o.transfer_from_file("model.msh")   # alternative: from .msh file
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def _import_g2o():
    """Import gmsh2opensees at call-time, not at module-import time."""
    try:
        import gmsh2opensees as g2o
        return g2o
    except ImportError:
        raise ImportError(
            "gmsh2opensees is not installed.\n"
            "Install it from source:\n"
            "    git clone https://github.com/jaabell/gmsh2opensees.git\n"
            "    cd gmsh2opensees && pip install .\n\n"
            "Or continue using the manual path:\n"
            "    fem = g.mesh.queries.get_fem_data(dim=2)\n"
            "See the example notebooks for both approaches."
        ) from None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Gmsh2OpenSees:
    """Thin optional bridge between a live apeGmsh session and OpenSees.

    Accessed as ``g.g2o`` after ``g = apeGmsh(...)``.

    Two transfer modes:

    1. **Live session** — ``g.g2o.transfer()``
       Requires that Gmsh is initialised *and* the mesh has been generated.
       Internally calls ``gmsh2opensees.gmsh2ops()``.

    2. **From file** — ``g.g2o.transfer_from_file("model.msh")``
       Reads a ``.msh`` file and pushes it into the current OpenSees model.
       Internally calls ``gmsh2opensees.msh2ops(path)``.

    Both modes assume that the caller has already:
    * Created an OpenSees model: ``ops.model('basic', '-ndm', ..., '-ndf', ...)``
    * Defined any materials that the elements will reference (if needed by
      the element-creation callbacks inside gmsh2opensees).
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent

    def _is_session_active(self) -> bool:
        """Check whether the owning session is open."""
        return self._parent.is_active

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------
    @staticmethod
    def is_available() -> bool:
        """Return ``True`` if gmsh2opensees is importable."""
        try:
            import gmsh2opensees  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Live-session transfer
    # ------------------------------------------------------------------
    def transfer(self, *, verbose: bool = False) -> None:
        """Transfer the current Gmsh mesh into the active OpenSees model.

        This is the "one-liner" convenience path.  It calls
        ``gmsh2opensees.gmsh2ops()`` which:

        * Iterates over every element in the Gmsh mesh.
        * Creates corresponding ``ops.node()`` and ``ops.element()`` calls.

        Prerequisites
        -------------
        * Gmsh must be initialised (``g.__enter__()`` or context manager).
        * ``g.mesh.generation.generate(...)`` must have been called first.
        * An OpenSees model must already be active::

              import openseespy.opensees as ops
              ops.wipe()
              ops.model('basic', '-ndm', 2, '-ndf', 2)

        Parameters
        ----------
        verbose : bool
            If ``True``, print a summary of what was transferred.

        Raises
        ------
        ImportError
            If ``gmsh2opensees`` is not installed.
        RuntimeError
            If Gmsh is not initialised or no mesh exists.
        """
        # --- Safety checks ---
        if not self._is_session_active():
            raise RuntimeError(
                "Gmsh is not initialised. Open a apeGmsh/Assembly session "
                "before transferring."
            )

        g2o = _import_g2o()

        if verbose:
            print("[Gmsh2OpenSees] Calling gmsh2ops() — transferring mesh …")

        g2o.gmsh2ops()

        if verbose:
            print("[Gmsh2OpenSees] Transfer complete.")

    # ------------------------------------------------------------------
    # File-based transfer
    # ------------------------------------------------------------------
    def transfer_from_file(self, msh_path: str | Path, *,
                           verbose: bool = False) -> None:
        """Read a ``.msh`` file and push nodes/elements into OpenSees.

        This is useful when the mesh was generated in a different session
        or by a different tool, and you only have the ``.msh`` file.

        Parameters
        ----------
        msh_path : str or Path
            Path to a Gmsh ``.msh`` file (ASCII or binary).
        verbose : bool
            If ``True``, print a summary.

        Raises
        ------
        ImportError
            If ``gmsh2opensees`` is not installed.
        FileNotFoundError
            If ``msh_path`` does not exist.
        """
        msh_path = Path(msh_path)
        if not msh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {msh_path}")

        g2o = _import_g2o()

        if verbose:
            print(f"[Gmsh2OpenSees] Reading {msh_path.name} …")

        g2o.msh2ops(str(msh_path))

        if verbose:
            print("[Gmsh2OpenSees] Transfer complete.")

    # ------------------------------------------------------------------
    # Convenience: export .msh then transfer
    # ------------------------------------------------------------------
    def save_and_transfer(self, msh_path: str | Path = "model.msh", *,
                          verbose: bool = False) -> Path:
        """Save the current mesh to a ``.msh`` file, then transfer it.

        Combines ``gmsh.write()`` + ``g2o.msh2ops()`` in one call.
        Useful when you want to keep a mesh file on disk for
        reproducibility **and** load it into OpenSees.

        Returns the resolved path to the written ``.msh`` file.
        """
        import gmsh

        if not self._is_session_active():
            raise RuntimeError("Gmsh is not initialised.")

        msh_path = Path(msh_path)
        gmsh.write(str(msh_path))

        if verbose:
            print(f"[Gmsh2OpenSees] Saved mesh -> {msh_path}")

        self.transfer_from_file(msh_path, verbose=verbose)
        return msh_path

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        avail = "installed" if self.is_available() else "NOT installed"
        return f"<Gmsh2OpenSees wrapper — gmsh2opensees {avail}>"
