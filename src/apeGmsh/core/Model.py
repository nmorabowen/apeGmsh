from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

from ._model_boolean import _Boolean
from ._model_geometry import _Geometry
from ._model_io import _IO
from ._model_queries import _Queries
from ._model_transforms import _Transforms

if TYPE_CHECKING:
    from apeGmsh._types import SessionProtocol as _SessionBase

from apeGmsh._logging import _HasLogging
from apeGmsh._types import Tag, DimTag, TagsLike


class Model(_HasLogging):
    """
    Geometry composite attached to an ``apeGmsh`` instance as
    ``g.model``.  Owns five focused sub-composites:

    * ``g.model.geometry`` — point / curve / surface / solid primitives
    * ``g.model.boolean`` — fuse, cut, intersect, fragment
    * ``g.model.transforms`` — translate, rotate, scale, mirror, copy,
      extrude, revolve, sweep, thru_sections
    * ``g.model.io`` — load/save STEP, IGES, DXF, MSH, heal_shapes
    * ``g.model.queries`` — bounding_box, center_of_mass, mass,
      boundary, adjacencies, entities_in_bounding_box, registry

    Plus entity selection:

    * ``g.model.selection`` — spatial entity queries

    And top-level utilities on the Model itself:

    * ``g.model.sync()`` — flush the OCC kernel
    * ``g.model.viewer()`` — open the interactive Qt viewer
    * ``g.model.gui()`` / ``g.model.launch_picker()`` — native Gmsh viewers

    Example
    -------
    ::

        # Solid boolean workflow
        box  = g.model.geometry.add_box(0, 0, 0, 10, 10, 10)
        hole = g.model.geometry.add_cylinder(5, 5, 0, 0, 0, 10, 2)
        part = g.model.boolean.cut(box, hole)

        # Wire-frame -> surface workflow
        p1   = g.model.geometry.add_point(0, 0, 0)
        p2   = g.model.geometry.add_point(10, 0, 0)
        p3   = g.model.geometry.add_point(10, 5, 0)
        p4   = g.model.geometry.add_point(0, 5, 0)
        l1   = g.model.geometry.add_line(p1, p2)
        l2   = g.model.geometry.add_line(p2, p3)
        l3   = g.model.geometry.add_line(p3, p4)
        l4   = g.model.geometry.add_line(p4, p1)
        loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
        surf = g.model.geometry.add_plane_surface(loop)

    Parameters
    ----------
    parent : _SessionBase
        Owning session — used to read ``_verbose`` and ``name``.
    """

    _log_prefix = "Model"

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent
        # (dim, tag) -> {kind, ...}  (labels live in g.labels, not here)
        self._metadata: dict[DimTag, dict] = {}

        # Five focused sub-composites — each one holds a reference to self
        self.geometry = _Geometry(self)
        self.boolean = _Boolean(self)
        self.transforms = _Transforms(self)
        self.io = _IO(self)
        self.queries = _Queries(self)

        # Entity-selection sub-composite (g.model.selection.select_points(...))
        # NB: SelectionComposite's methods that touch `physical` require a
        # full ``apeGmsh`` session — ``Part`` sessions have a ``model``
        # composite but never touch ``model.selection``, so the runtime
        # cast is safe in practice.
        from apeGmsh.viz.Selection import SelectionComposite
        self.selection = SelectionComposite(parent=parent, model=self)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internal helpers (used by sub-composites via self._model._*)
    # ------------------------------------------------------------------

    def _resolve_dim(self, tag: int, default_dim: int) -> int:
        """Resolve tag dimension from the live Gmsh model. See :func:`_helpers.resolve_dim`."""
        from ._helpers import resolve_dim
        return resolve_dim(tag, default_dim)

    def _as_dimtags(self, tags: TagsLike, default_dim: int = 3) -> list[DimTag]:
        """Normalize tag input to [(dim, tag), ...]. See :func:`_helpers.as_dimtags`."""
        from ._helpers import as_dimtags
        return as_dimtags(tags, default_dim)

    def _register(self, dim: int, tag: Tag, label: str | None, kind: str) -> Tag:
        # Metadata only — labels live exclusively in g.labels (Gmsh PGs).
        self._metadata[(dim, tag)] = {'kind': kind}

        # When the owning session has ``_auto_pg_from_label`` set
        # (both Part and apeGmsh sessions), automatically create a
        # **label PG** (Tier 1 — geometry bookkeeping, prefixed with
        # ``_label:``) so the user-supplied label can be resolved by
        # ``g.labels.entities("name")`` and travels through the STEP
        # sidecar into the Assembly.  This does NOT create a solver-
        # facing physical group — the user promotes labels to PGs
        # explicitly via ``g.labels.promote_to_physical("name")``.
        # Label creation must never break geometry creation, but
        # failures are logged so they don't hide real bugs.
        if label and getattr(self._parent, '_auto_pg_from_label', False):
            labels_comp = getattr(self._parent, 'labels', None)
            if labels_comp is not None:
                try:
                    labels_comp.add(dim, [tag], name=label)
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"Label {label!r} (dim={dim}, tag={tag}) could "
                        f"not be created: {exc}",
                        stacklevel=3,
                    )
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

        Displays BRep geometry with selectable entities, parts,
        physical groups, and labels.  This is a **geometry-only**
        viewer — loads, constraints, and masses are mesh-resolved
        concepts and live on ``g.mesh.viewer()`` instead.

        Parameters
        ----------
        **kwargs :
            Forwarded to :meth:`selection.picker` (e.g. ``dims``,
            ``point_size``, ``line_width``, ``surface_opacity``).
        """
        return self.selection.picker(**kwargs)

    def gui(self) -> None:
        """Open the interactive Gmsh FLTK GUI window."""
        gmsh.fltk.run()

    def launch_picker(
        self,
        *,
        show_points: bool = True,
        show_curves: bool = True,
        show_surfaces: bool = True,
        show_volumes: bool = False,
        verbose: bool = True,
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
        return f"Model(name={self._parent.name!r}, entities={len(self._metadata)})"
