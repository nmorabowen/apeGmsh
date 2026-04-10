"""
Part — Pure geometry builder.  Meshing lives in the Assembly.
==============================================================

Mirrors the Abaqus Part concept:

* **Part**  =  geometry only (points, curves, surfaces, volumes).
  No meshing, no physical groups, no mesh settings.
* **Assembly**  =  imports Parts, positions them, fragments,
  assigns physical groups, controls mesh, generates mesh,
  extracts FEM data, resolves constraints.

A Part is created in its own isolated Gmsh session, geometry is
built with the full ``apeGmsh`` API, then saved to **STEP** (the
default and recommended format — preserves parametric OCC geometry
so the Assembly can re-mesh freely).

Typical usage
-------------
::

    from apeGmsh import Part

    # ── Build a plate ──────────────────────────────────────────
    plate = Part("plate")
    plate.begin()

    p1 = plate.model.geometry.add_point(0, 0, 0, sync=False)
    ...                                   # build geometry
    plate.model.sync()

    plate.save()                          # → plate.step (default)
    plate.end()

    # ── Reuse it in a apeGmsh session ─────────────────────────
    from apeGmsh import apeGmsh

    g = apeGmsh(model_name="bridge")
    g.begin()
    g.parts.add(plate, translate=(0, 0, 0))
    g.parts.add(plate, translate=(5000, 0, 0), label="plate_2")
    g.parts.fragment_all()
    g.mesh.generate(dim=2)
    g.end()

Why STEP?
~~~~~~~~~
STEP preserves the full parametric OCC geometry (exact NURBS,
topology, tolerances).  After import, apeGmsh can:

* Apply boolean operations (fragment, fuse, cut)
* Set transfinite meshing / recombine / mesh fields
* Generate and re-generate the mesh with any settings

IGES is also supported for legacy compatibility, but STEP is
preferred (better fidelity, modern spec, avoids IGES surface
junction gaps).

Design notes
~~~~~~~~~~~~
* Each Part owns its own Gmsh session.  ``begin()`` initialises
  Gmsh and creates only the composites it needs (Model, Inspect).
  ``end()`` finalises Gmsh.
* The Part remembers which file it was last saved to, so the
  Assembly can locate it automatically.
* Parts carry **metadata** in ``properties`` (thickness, material
  name, section type, etc.) that the Assembly passes through to
  the solver model.
* ``save()`` with no arguments writes ``"{name}.step"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import gmsh

from .._session import _SessionBase

if TYPE_CHECKING:
    from .Model import Model
    from ..viz.Inspect import Inspect
    from ..viz.Plot import Plot


class Part(_SessionBase):
    """
    An isolated geometry unit — no meshing, no physical groups.

    Parameters
    ----------
    name : str
        Descriptive name (also used as the Gmsh model name).
    """

    # Supported CAD export extensions
    _VALID_EXT = {'.step', '.stp', '.iges', '.igs'}

    _COMPOSITES = (
        ("model",   ".Model",   "Model",   False),
        ("inspect", ".viz.Inspect",   "Inspect", False),
        ("plot",    ".viz.Plot",      "Plot",    True),
    )

    # -- Static type declarations for composites --
    model: Model
    inspect: Inspect
    plot: Plot

    def __init__(self, name: str) -> None:
        super().__init__(name=name, verbose=False)
        self.file_path: Path | None = None       # set by save()
        self.properties: dict[str, Any] = {}     # user metadata

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(
        self,
        file_path: str | Path | None = None,
        *,
        fmt: str | None = None,
    ) -> Path:
        """
        Export the Part geometry to a CAD file.

        Parameters
        ----------
        file_path : str, Path, or None
            Destination path.  If ``None``, defaults to
            ``"{name}.step"``.  The extension determines the format
            unless *fmt* overrides it.
        fmt : str, optional
            Force format: ``"step"`` or ``"iges"``.

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        if not self._active:
            raise RuntimeError("Part session is not active — call begin() first.")

        # Default: save as STEP using the Part name
        if file_path is None:
            file_path = Path(f"{self.name}.step")

        file_path = Path(file_path)

        # Override extension if fmt is given
        if fmt is not None:
            fmt = fmt.lower().strip(".")
            ext_map = {"step": ".step", "stp": ".step",
                       "iges": ".iges", "igs": ".iges"}
            ext = ext_map.get(fmt)
            if ext is None:
                raise ValueError(f"Unknown format '{fmt}'. Use 'step' or 'iges'.")
            file_path = file_path.with_suffix(ext)

        if file_path.suffix.lower() not in self._VALID_EXT:
            raise ValueError(
                f"Extension '{file_path.suffix}' is not a supported CAD format. "
                f"Use one of {self._VALID_EXT}."
            )

        # Sync OCC kernel before export
        gmsh.model.occ.synchronize()
        gmsh.write(str(file_path))
        self.file_path = file_path.resolve()
        return self.file_path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def has_file(self) -> bool:
        """True if the Part has been saved to disk."""
        return self.file_path is not None and self.file_path.exists()

    def __repr__(self) -> str:
        status = "active" if self._active else "closed"
        saved  = f", file={self.file_path.name}" if self.file_path else ""
        return f"Part('{self.name}', {status}{saved})"
