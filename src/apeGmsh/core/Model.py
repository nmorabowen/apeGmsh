from __future__ import annotations

import gmsh
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tag      = int
DimTag   = tuple[int, int]
TagsLike = Tag | list[Tag] | DimTag | list[DimTag]   # flexible input accepted everywhere

# ---------------------------------------------------------------------------
# Mixin imports — each file holds one logical category of methods
# ---------------------------------------------------------------------------
from ._model_geometry import _GeometryMixin
from ._model_boolean import _BooleanMixin
from ._model_transforms import _TransformsMixin
from ._model_io import _IOMixin
from ._model_queries import _QueriesMixin


class Model(
    _GeometryMixin,
    _BooleanMixin,
    _TransformsMixin,
    _IOMixin,
    _QueriesMixin,
):
    """
    Geometry-construction composite attached to a ``apeGmsh`` instance as
    ``self.model``.

    Wraps ``gmsh.model.occ`` with a clean, parametric-friendly API:

    * **Points**       — ``add_point``
    * **Curves**       — ``add_line``, ``add_arc``, ``add_circle``,
      ``add_ellipse``, ``add_spline``, ``add_bspline``, ``add_bezier``
    * **Wire / faces** — ``add_curve_loop``, ``add_plane_surface``,
      ``add_surface_filling``
    * **Solids**       — ``add_box``, ``add_sphere``, ``add_cylinder``,
      ``add_cone``, ``add_torus``, ``add_wedge``
    * **Boolean ops**  — ``fuse``, ``cut``, ``intersect``, ``fragment``
    * **Transforms**   — ``translate``, ``rotate``, ``scale``, ``mirror``,
      ``copy``
    * **Sweep ops**    — ``extrude``, ``revolve``
    * **IO**           — ``load_iges``, ``load_step``, ``load_dxf``,
      ``save_iges``, ``save_step``, ``save_dxf``, ``heal_shapes``
    * **Queries**      — ``bounding_box``, ``center_of_mass``, ``mass``,
      ``boundary``, ``adjacencies``, ``entities_in_bounding_box``
    * **Utilities**    — ``sync``, ``remove``, ``gui``, ``registry``

    All creation methods return plain integer tags so they compose
    naturally — the dimension is implied by context::

        # solid boolean workflow
        box  = g.model.add_box(0, 0, 0, 10, 10, 10)
        hole = g.model.add_cylinder(5, 5, 0, 0, 0, 10, 2)
        part = g.model.cut(box, hole)

        # wire-frame → surface workflow
        p1   = g.model.add_point(0, 0, 0)
        p2   = g.model.add_point(10, 0, 0)
        p3   = g.model.add_point(10, 5, 0)
        p4   = g.model.add_point(0, 5, 0)
        l1   = g.model.add_line(p1, p2)
        l2   = g.model.add_line(p2, p3)
        l3   = g.model.add_line(p3, p4)
        l4   = g.model.add_line(p4, p1)
        loop = g.model.add_curve_loop([l1, l2, l3, l4])
        surf = g.model.add_plane_surface(loop)

    Parameters
    ----------
    parent : _SessionBase
        Owning instance — used to read ``_verbose``.
    """

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent   = parent
        # (dim, tag) → {label, kind}
        self._registry : dict[DimTag, dict] = {}
        # Entity-selection sub-composite (model.selection.select_points(...))
        from apeGmsh.viz.Selection import SelectionComposite
        self.selection = SelectionComposite(parent=parent, model=self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[Model] {msg}")

    def _resolve_dim(self, tag: int, default_dim: int) -> int:
        """Resolve tag dimension from registry. See :func:`_helpers.resolve_dim`."""
        from ._helpers import resolve_dim
        return resolve_dim(tag, default_dim, self._registry)

    def _as_dimtags(self, tags: TagsLike, default_dim: int = 3) -> list[DimTag]:
        """Normalize tag input to [(dim, tag), ...]. See :func:`_helpers.as_dimtags`."""
        from ._helpers import as_dimtags
        return as_dimtags(tags, default_dim, registry=self._registry)

    def _register(self, dim: int, tag: Tag, label: str | None, kind: str) -> Tag:
        self._registry[(dim, tag)] = {
            'label': label if label else f'{kind}_{tag}',
            'kind' : kind,
        }
        return tag

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self) -> "Model":
        """
        Synchronise the OCC kernel with the gmsh model topology.

        Call this explicitly when you have been batching operations with
        ``sync=False``.  Returns ``self`` for chaining.
        """
        gmsh.model.occ.synchronize()
        self._log("OCC kernel synchronised")
        return self

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def viewer(self, **kwargs):
        """Open the interactive Qt model viewer.

        Displays the BRep geometry with selectable entities, parts,
        physical groups, and (when ``fem=`` is provided) loads/mass
        arrow and sphere overlays.

        Parameters
        ----------
        fem : FEMData, optional
            Resolved snapshot from :meth:`Mesh.get_fem_data`.  Required
            to enable the **Loads** and **Mass** tab overlays — the
            viewer reads node coordinates and resolved per-node records
            from this snapshot.  Without ``fem=``, those tabs show the
            load/mass definition list but the 3-D glyph overlays are
            disabled with an amber warning.

        **kwargs :
            Forwarded to :meth:`selection.picker` (e.g. ``dims``,
            ``point_size``, ``line_width``, ``surface_opacity``).

        Why ``fem=`` is required for overlays
        --------------------------------------
        Loads and mass are **snapshot semantics**: the user defines
        them symbolically (``g.loads.gravity(...)``, ``g.mass.volume(...)``),
        then resolves them against a specific mesh.  The viewer draws
        exactly what was in the snapshot — not whatever the session
        currently has.  This means:

        * If you re-mesh or re-resolve, the viewer does not
          auto-follow.  You re-open with the new snapshot.
        * The same entry point works for live sessions and for
          loaded ``.msh`` files via ``g.mesh.loader.from_msh(...)``
          (a pre-existing snapshot).
        * Forgetting to call ``get_fem_data()`` shows empty overlays
          with a clear warning — safer than drawing stale data.

        Example
        -------
        ::

            g.mesh.generate(3)
            fem = g.mesh.get_fem_data(3)   # resolve loads + mass
            g.model.viewer(fem=fem)         # overlays enabled
        """
        return self.selection.picker(**kwargs)

    def viewer_fast(self, **kwargs):
        """Open the model viewer with fast mesh-based tessellation."""
        return self.selection.picker(fast=True, **kwargs)

    def gui(self) -> None:
        """Open the interactive Gmsh FLTK GUI window."""
        gmsh.fltk.run()

    def launch_picker(
        self,
        *,
        show_points   : bool = True,
        show_curves   : bool = True,
        show_surfaces : bool = True,
        show_volumes  : bool = False,
        verbose       : bool = True,
    ) -> None:
        """Open Gmsh's native FLTK viewer with entity labels pre-enabled."""
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Geometry.PointLabels",   int(show_points))
        gmsh.option.setNumber("Geometry.CurveLabels",   int(show_curves))
        gmsh.option.setNumber("Geometry.SurfaceLabels", int(show_surfaces))
        gmsh.option.setNumber("Geometry.VolumeLabels",  int(show_volumes))
        gmsh.option.setNumber("Geometry.Points",   1)
        gmsh.option.setNumber("Geometry.Curves",   1)
        gmsh.option.setNumber("Geometry.Surfaces", 1)
        if verbose:
            print("[launch_picker] Opening Gmsh FLTK window.")
            print("  Labels visible — read tags off the 3D view.")
            print("  Close the window to return here.")
        gmsh.fltk.run()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Model(name={self._parent.model_name!r}, registered={len(self._registry)})"
