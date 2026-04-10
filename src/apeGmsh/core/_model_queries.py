from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import pandas as pd

from ._helpers import Tag, DimTag, TagsLike

if TYPE_CHECKING:
    from .Model import Model


class _Queries:
    """Queries sub-composite — remove, topology queries, and registry."""

    def __init__(self, model: "Model") -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(
        self,
        tags     : TagsLike,
        *,
        dim      : int  = 3,
        recursive: bool = False,
        sync     : bool = True,
    ) -> None:
        """
        Delete entities from the model.

        Parameters
        ----------
        recursive : bool
            If True, also delete all lower-dimensional entities that are
            exclusively owned by these entities.
        """
        dim_tags = self._model._as_dimtags(tags, dim)
        gmsh.model.occ.remove(dim_tags, recursive=recursive)
        if sync:
            gmsh.model.occ.synchronize()
        for dt in dim_tags:
            self._model._registry.pop(dt, None)
        self._model._log(f"removed {dim_tags} (recursive={recursive})")

    def remove_duplicates(
        self,
        *,
        tolerance: float | None = None,
        sync     : bool         = True,
    ) -> _Queries:
        """
        Merge all coincident OCC entities in the current model.

        Calls ``gmsh.model.occ.removeAllDuplicates()``, which walks every
        dimension (points → curves → surfaces → volumes) and collapses
        entities that are geometrically identical within the OCC tolerance.
        The internal registry is then reconciled so only entities that
        survive the merge are tracked.

        This is the recommended post-processing step after importing IGES
        or STEP files, which routinely produce coincident points and
        overlapping curves at shared frame joints.

        Parameters
        ----------
        tolerance : float | None
            Geometric merge tolerance.  When provided, temporarily overrides
            ``Geometry.Tolerance`` and ``Geometry.ToleranceBoolean`` for the
            duration of this call, then restores the previous values.
            Use this when the IGES exporter introduced small coordinate
            imprecisions (e.g. ``tolerance=1e-3`` for mm-scale models).
            When None (default), the current Gmsh tolerance is used unchanged.
        sync : bool
            Synchronise the OCC kernel after merging (default True).
            Set to False only if you intend to call ``model.sync()``
            manually as part of a larger batch operation.

        Returns
        -------
        self — for method chaining

        Example
        -------
        ::

            imported = g.model.io.load_iges("Frame3D.iges", highest_dim_only=False)
            g.model.queries.remove_duplicates(tolerance=1e-3)
            g.plot.geometry(label_tags=True)
        """
        before = {d: len(gmsh.model.getEntities(d)) for d in range(4)}

        # Temporarily widen the OCC tolerance if the caller asked for it,
        # then restore both options to their original values afterwards.
        _tol_keys = ("Geometry.Tolerance", "Geometry.ToleranceBoolean")
        _saved: dict[str, float] = {}
        if tolerance is not None:
            for key in _tol_keys:
                _saved[key] = gmsh.option.getNumber(key)
                gmsh.option.setNumber(key, tolerance)

        try:
            gmsh.model.occ.removeAllDuplicates()
        finally:
            for key, val in _saved.items():
                gmsh.option.setNumber(key, val)

        if sync:
            gmsh.model.occ.synchronize()

        # Reconcile registry — drop any (dim, tag) pairs that no longer
        # exist in the gmsh model after the merge.
        surviving: set[tuple[int, int]] = {
            (dim, tag)
            for dim in range(4)
            for _, tag in gmsh.model.getEntities(dim)
        }
        stale_dts = [dt for dt in self._model._registry if dt not in surviving]
        for dt in stale_dts:
            del self._model._registry[dt]

        after = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
        removed = {d: before[d] - after[d] for d in range(4) if before[d] != after[d]}
        tol_str = f"tolerance={tolerance}" if tolerance is not None else ""
        self._model._log(
            f"remove_duplicates({tol_str}): merged {removed} entities "
            f"(before={before}, after={after})"
        )
        return self

    def make_conformal(
        self,
        *,
        dims     : list[int] | None = None,
        tolerance: float | None     = None,
        sync     : bool             = True,
    ) -> _Queries:
        """
        Fragment all entities against each other to produce a conformal model.

        IGES/STEP files exported from CAD tools often create topologically
        disconnected entities at shared joints — for example, column endpoints
        and beam endpoints that are coincident in space but belong to separate
        BRep objects with no shared vertex.  A conformal model is required for
        FEM meshing because elements must share nodes at junctions rather than
        having two independent nodes at the same location.

        This method calls ``gmsh.model.occ.fragment()`` with all entities of
        the requested dimensions as both objects and tools.  OCC computes all
        intersections, splits curves at shared points, and merges coincident
        vertices — leaving a single connected topology.

        Parameters
        ----------
        dims : list[int] | None
            Dimensions to fragment.  Defaults to all non-empty dimensions
            present in the model (typically ``[1]`` for wireframe frames,
            ``[1, 2]`` for mixed models).  Pass ``[1]`` explicitly to
            restrict to curves only and avoid fragmenting surfaces.
        tolerance : float | None
            Geometric tolerance for OCC's intersection / coincidence detection.
            Temporarily overrides ``Geometry.ToleranceBoolean`` for the duration
            of the fragment call, then restores the original value.
            Use this when curves only touch at endpoints (no proper crossing)
            and the default OCC tolerance is too tight to detect them —
            e.g. ``tolerance=1.0`` for mm-scale models.
            When None (default), the current Gmsh tolerance is used unchanged.
        sync : bool
            Synchronise the OCC kernel after fragmenting (default True).

        Returns
        -------
        self — for method chaining

        Example
        -------
        ::

            m1.model.io.load_iges("Frame3D.iges", highest_dim_only=False)
            m1.remove_duplicates(tolerance=1.0)
            m1.model.queries.make_conformal(dims=[1], tolerance=1.0)
            m1.plot.geometry(label_tags=True)
        """
        before = {d: len(gmsh.model.getEntities(d)) for d in range(4)}

        if dims is None:
            dims = [d for d in range(4) if gmsh.model.getEntities(d)]

        all_dimtags: list[tuple[int, int]] = [
            (d, tag)
            for d in dims
            for _, tag in gmsh.model.getEntities(d)
        ]

        if not all_dimtags:
            self._model._log("make_conformal(): no entities found, nothing to do")
            return self

        _tol_keys = ("Geometry.ToleranceBoolean",)
        _saved: dict[str, float] = {}
        if tolerance is not None:
            for key in _tol_keys:
                _saved[key] = gmsh.option.getNumber(key)
                gmsh.option.setNumber(key, tolerance)

        try:
            gmsh.model.occ.fragment(all_dimtags, [], removeObject=True, removeTool=True)
        finally:
            for key, val in _saved.items():
                gmsh.option.setNumber(key, val)

        if sync:
            gmsh.model.occ.synchronize()

        # Rebuild registry from scratch — fragment renumbers entities
        old_registry = dict(self._model._registry)
        self._model._registry.clear()
        for d in range(4):
            for _, tag in gmsh.model.getEntities(d):
                old_entry = old_registry.get((d, tag))
                if old_entry:
                    self._model._registry[(d, tag)] = old_entry
                else:
                    self._model._registry[(d, tag)] = {'label': f'entity_{tag}', 'kind': 'fragment'}

        after = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
        delta = {d: after[d] - before[d] for d in range(4) if before[d] != after[d]}
        tol_str = f", tolerance={tolerance}" if tolerance is not None else ""
        self._model._log(
            f"make_conformal(dims={dims}{tol_str}): entity delta={delta} "
            f"(before={before}, after={after})"
        )
        return self

    # Alias so both ``model.fragment_all()`` and ``model.make_conformal()``
    # work — mirrors the Assembly API naming convention.
    fragment_all = make_conformal

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    def bounding_box(
        self,
        tag: Tag,
        *,
        dim: int = 3,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Return the axis-aligned bounding box of an entity.

        Returns
        -------
        (xmin, ymin, zmin, xmax, ymax, zmax)

        Example
        -------
        ``xmin, ymin, zmin, xmax, ymax, zmax = g.model.queries.bounding_box(vol)``
        """
        d = self._model._resolve_dim(tag, dim)
        return gmsh.model.getBoundingBox(d, tag)

    def center_of_mass(
        self,
        tag: Tag,
        *,
        dim: int = 3,
    ) -> tuple[float, float, float]:
        """
        Return the center of mass of an entity.

        Example
        -------
        ``cx, cy, cz = g.model.queries.center_of_mass(vol)``
        """
        d = self._model._resolve_dim(tag, dim)
        return gmsh.model.occ.getCenterOfMass(d, tag)

    def mass(
        self,
        tag: Tag,
        *,
        dim: int = 3,
    ) -> float:
        """
        Return the mass (volume for 3D, area for 2D, length for 1D)
        of an entity.

        Example
        -------
        ``vol = g.model.queries.mass(solid_tag)``
        """
        d = self._model._resolve_dim(tag, dim)
        return gmsh.model.occ.getMass(d, tag)

    def boundary(
        self,
        tags: TagsLike,
        *,
        dim      : int  = 3,
        oriented : bool = False,
        combined : bool = True,
        recursive: bool = False,
    ) -> list[DimTag]:
        """
        Return the boundary entities of the given entities.

        Parameters
        ----------
        tags : entities whose boundary to query.
        dim : default dimension for bare integer tags.
        oriented : if True, return oriented boundary (signs on tags).
        combined : if True, return the boundary of the combined entities.
        recursive : if True, recurse down to dimension 0.

        Returns
        -------
        list[DimTag]
            Boundary entities as (dim, tag) pairs.

        Example
        -------
        ::

            faces = g.model.queries.boundary(vol_tag)  # surfaces bounding a volume
        """
        dt = self._model._as_dimtags(tags, dim)
        return gmsh.model.getBoundary(
            dt,
            combined=combined,
            oriented=oriented,
            recursive=recursive,
        )

    def adjacencies(
        self,
        tag: Tag,
        *,
        dim: int = 3,
    ) -> tuple[list[Tag], list[Tag]]:
        """
        Return entities adjacent to the given entity.

        Returns
        -------
        (upward, downward)
            ``upward`` — tags of entities of ``dim + 1`` that contain
            this entity.
            ``downward`` — tags of entities of ``dim - 1`` on this
            entity's boundary.

        Example
        -------
        ::

            up, down = g.model.queries.adjacencies(face_tag, dim=2)
            # up   = volumes bounded by this face
            # down = curves on this face's boundary
        """
        d = self._model._resolve_dim(tag, dim)
        up, down = gmsh.model.getAdjacencies(d, tag)
        return list(up), list(down)

    def entities_in_bounding_box(
        self,
        xmin: float, ymin: float, zmin: float,
        xmax: float, ymax: float, zmax: float,
        *,
        dim: int = -1,
    ) -> list[DimTag]:
        """
        Return all entities inside a bounding box.

        Parameters
        ----------
        xmin, ymin, zmin, xmax, ymax, zmax : box limits.
        dim : restrict to this dimension (-1 = all dimensions).

        Returns
        -------
        list[DimTag]

        Example
        -------
        ::

            # Find all entities in a region
            found = g.model.queries.entities_in_bounding_box(
                0, 0, 0,  10, 10, 10, dim=3
            )
        """
        return gmsh.model.getEntitiesInBoundingBox(
            xmin, ymin, zmin, xmax, ymax, zmax, dim,
        )

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def registry(self) -> pd.DataFrame:
        """
        Return a DataFrame of all entities created through this helper.

        Indexed by ``(dim, tag)`` — matching Gmsh's identity model where
        tags are only unique within a dimension.

        Columns: ``kind``, ``label``
        """
        if not self._model._registry:
            return pd.DataFrame(columns=['dim', 'tag', 'kind', 'label'])
        rows = [
            {'dim': dim, 'tag': tag, **info}
            for (dim, tag), info in self._model._registry.items()
        ]
        return (
            pd.DataFrame(rows)
            .set_index(['dim', 'tag'])
            .sort_index()
        )
