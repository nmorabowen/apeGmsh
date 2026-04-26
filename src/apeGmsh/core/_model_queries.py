from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import gmsh
import pandas as pd

from ._helpers import Tag, DimTag, TagsLike
from .Labels import (
    pg_preserved,
    cleanup_label_pgs,
    reconcile_label_pgs,
)
from ._selection import Selection, _select_impl, Plane, Line

if TYPE_CHECKING:
    from .Model import Model


@contextmanager
def _temporary_tolerance(
    tolerance: float | None,
    keys: tuple[str, ...] = ("Geometry.Tolerance", "Geometry.ToleranceBoolean"),
) -> Iterator[None]:
    """Temporarily override Gmsh tolerance options, restoring on exit."""
    saved: dict[str, float] = {}
    if tolerance is not None:
        for key in keys:
            saved[key] = gmsh.option.getNumber(key)
            gmsh.option.setNumber(key, tolerance)
    try:
        yield
    finally:
        for key, val in saved.items():
            gmsh.option.setNumber(key, val)


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
            self._model._metadata.pop(dt, None)
        cleanup_label_pgs(dim_tags)
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
        dimension (points -> curves -> surfaces -> volumes) and collapses
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

        with _temporary_tolerance(tolerance):
            gmsh.model.occ.removeAllDuplicates()

        if sync:
            gmsh.model.occ.synchronize()

        # Reconcile metadata — drop any (dim, tag) pairs that no longer
        # exist in the gmsh model after the merge.
        surviving: set[tuple[int, int]] = {
            (dim, tag)
            for dim in range(4)
            for _, tag in gmsh.model.getEntities(dim)
        }
        stale_dts = [dt for dt in self._model._metadata if dt not in surviving]
        for dt in stale_dts:
            del self._model._metadata[dt]

        # Reconcile label PGs — removeAllDuplicates doesn't provide a
        # result_map, so we walk all label PGs and drop dead entity tags.
        reconcile_label_pgs()

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

        with pg_preserved() as pg, \
             _temporary_tolerance(tolerance, keys=("Geometry.ToleranceBoolean",)):
            _, result_map = gmsh.model.occ.fragment(
                all_dimtags, [], removeObject=True, removeTool=True,
            )
            if sync:
                gmsh.model.occ.synchronize()
            pg.set_result(all_dimtags, result_map)

        # Rebuild metadata from scratch — fragment renumbers entities.
        # Only kind/normal/point metadata is preserved; labels live
        # in g.labels (Gmsh PGs) and are remapped by remap_physical_groups.
        old_metadata = dict(self._model._metadata)
        self._model._metadata.clear()
        for d in range(4):
            for _, tag in gmsh.model.getEntities(d):
                old_entry = old_metadata.get((d, tag))
                if old_entry:
                    self._model._metadata[(d, tag)] = old_entry
                else:
                    self._model._metadata[(d, tag)] = {'kind': 'fragment'}

        # Remap Instance.entities if the session has a parts registry.
        # Without this, instances track stale tags after make_conformal.
        parts = getattr(self._model._parent, 'parts', None)
        if parts is not None:
            # Build (dim, old_tag) → [new_tags at same dim]
            dt_remap: dict[tuple[int, int], list[int]] = {}
            for old_dt, new_dts in zip(all_dimtags, result_map):
                od, ot = int(old_dt[0]), int(old_dt[1])
                dt_remap[(od, ot)] = [int(t) for d, t in new_dts if int(d) == od]

            for inst in parts._instances.values():
                for d in list(inst.entities.keys()):
                    old_tags = inst.entities.get(d, [])
                    new_tags: list[int] = []
                    for ot in old_tags:
                        mapped = dt_remap.get((d, ot))
                        if mapped is not None:
                            new_tags.extend(mapped)
                        else:
                            new_tags.append(ot)
                    inst.entities[d] = new_tags

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
        tag,
        *,
        dim: int = 3,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Return the axis-aligned bounding box of an entity.

        ``tag`` accepts an int, a label, a PG name, or a ``(dim, tag)``
        tuple.  Must resolve to exactly one entity.  When ``tag`` is a
        bare int, ``dim`` is honoured as an explicit dimension hint
        (no live-model lookup) — important because Gmsh tag spaces are
        per-dimension, so the same int can refer to different entities
        at different dims.

        Returns
        -------
        (xmin, ymin, zmin, xmax, ymax, zmax)

        Example
        -------
        ``xmin, ymin, zmin, xmax, ymax, zmax = g.model.queries.bounding_box("box")``
        """
        if isinstance(tag, int) and not isinstance(tag, bool):
            return gmsh.model.getBoundingBox(dim, tag)
        from ._helpers import resolve_to_single_dimtag
        d, t = resolve_to_single_dimtag(
            tag, default_dim=dim, session=self._model._parent,
            what="bounding_box target",
        )
        return gmsh.model.getBoundingBox(d, t)

    def center_of_mass(
        self,
        tag,
        *,
        dim: int = 3,
    ) -> tuple[float, float, float]:
        """
        Return the center of mass of an entity.

        ``tag`` accepts an int, a label, a PG name, or a ``(dim, tag)``
        tuple.  Must resolve to exactly one entity.  Bare ints are
        interpreted at ``dim`` directly (no live-model lookup).

        Example
        -------
        ``cx, cy, cz = g.model.queries.center_of_mass("box")``
        """
        if isinstance(tag, int) and not isinstance(tag, bool):
            return gmsh.model.occ.getCenterOfMass(dim, tag)
        from ._helpers import resolve_to_single_dimtag
        d, t = resolve_to_single_dimtag(
            tag, default_dim=dim, session=self._model._parent,
            what="center_of_mass target",
        )
        return gmsh.model.occ.getCenterOfMass(d, t)

    def mass(
        self,
        tag,
        *,
        dim: int = 3,
    ) -> float:
        """
        Return the mass (volume for 3D, area for 2D, length for 1D)
        of an entity.

        ``tag`` accepts an int, a label, a PG name, or a ``(dim, tag)``
        tuple.  Must resolve to exactly one entity.  Bare ints are
        interpreted at ``dim`` directly (no live-model lookup).

        Example
        -------
        ``vol = g.model.queries.mass("box")``
        """
        if isinstance(tag, int) and not isinstance(tag, bool):
            return gmsh.model.occ.getMass(dim, tag)
        from ._helpers import resolve_to_single_dimtag
        d, t = resolve_to_single_dimtag(
            tag, default_dim=dim, session=self._model._parent,
            what="mass target",
        )
        return gmsh.model.occ.getMass(d, t)

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

    def _resolve_to_dimtags(self, tag) -> list[DimTag]:
        """Resolve int / label / PG / dimtag (or list of) to dimtags."""
        if isinstance(tag, str):
            from ._helpers import _resolve_string_to_dimtags
            return _resolve_string_to_dimtags(
                tag, default_dim=3, session=self._model._parent,
            )
        return self._model._as_dimtags(tag)

    def boundary_curves(self, tag) -> list[DimTag]:
        """
        Return all unique curves (dim = 1) on the boundary of an entity.

        Wraps the two-step query needed to get a volume's edges:
        ``boundary(vol)`` skips straight to vertices when ``recursive=True``,
        so the correct pattern is to fetch faces first, then walk each face's
        boundary individually with ``combined=False`` (so shared edges are not
        cancelled), and deduplicate the result.

        Parameters
        ----------
        tag : int, label, PG name, ``(dim, tag)`` tuple, or list thereof.

        Returns
        -------
        list[DimTag]
            ``(1, curve_tag)`` pairs, deduplicated.

        Example
        -------
        ::

            edges = g.model.queries.boundary_curves('box')   # 12 edges
            edges = g.model.queries.boundary_curves(surf)    # 4 edges of a face
        """
        owners = self._resolve_to_dimtags(tag)
        # If the entities are already curves, return them deduplicated.
        if all(d == 1 for d, _ in owners):
            return list(dict.fromkeys(owners))
        # Surfaces → walk one more level with combined=False to keep shared edges.
        if all(d == 2 for d, _ in owners):
            return list(dict.fromkeys(
                self.boundary(owners, combined=False, oriented=False)
            ))
        # Volumes (or mixed): faces first, then their individual boundaries.
        faces = self.boundary(owners, oriented=False)
        return list(dict.fromkeys(
            self.boundary(faces, combined=False, oriented=False)
        ))

    def boundary_points(self, tag) -> list[DimTag]:
        """
        Return all unique points (dim = 0) on the boundary of an entity.

        Equivalent to ``boundary(tag, recursive=True)`` for volumes —
        Gmsh's recursive walk goes straight to dim=0 — but provided as a
        named alias for symmetry with ``boundary_curves``.

        Example
        -------
        ::

            corners = g.model.queries.boundary_points('box')   # 8 corners
        """
        owners = self._resolve_to_dimtags(tag)
        return list(dict.fromkeys(
            dt for dt in self.boundary(owners, oriented=False, recursive=True)
            if dt[0] == 0
        ))

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
    # Geometric primitives — factories so users never need to import
    # ------------------------------------------------------------------

    def plane(self, *args, **kwargs) -> Plane:
        """
        Construct a :class:`Plane` for use with ``select(on=...)`` /
        ``select(crossing=...)`` (or any future API that accepts a plane).

        Forms accepted
        --------------
        ``plane(z=0)`` / ``plane(x=5)``
            Axis-aligned plane.
        ``plane(p1, p2, p3)``
            Plane through three non-collinear points.
        ``plane(normal=(0, 0, 1), through=(0, 0, 5))``
            Direct construction from a normal and an anchor point.

        Example
        -------
        ::

            mid = m.model.queries.plane(z=2.5)
            faces_cut = m.model.queries.select(faces, crossing=mid)
            below     = m.model.queries.select(faces, not_crossing=mid)
        """
        if args and not kwargs:
            if len(args) == 3:
                return Plane.through(*args)
            raise ValueError(
                "plane(*args) needs 3 points; got "
                f"{len(args)}.  Did you mean plane(z=0)?"
            )
        if kwargs and not args:
            if 'normal' in kwargs and 'through' in kwargs:
                import numpy as _np
                n = _np.asarray(kwargs['normal'], dtype=float)
                n = n / _np.linalg.norm(n)
                a = _np.asarray(kwargs['through'], dtype=float)
                return Plane(normal=n, anchor=a)
            return Plane.at(**kwargs)
        raise ValueError(
            "plane() takes either an axis kwarg (z=0), 3 positional points, "
            "or normal=/through= kwargs."
        )

    def line(self, p1, p2) -> Line:
        """
        Construct a :class:`Line` for use with ``select(on=...)`` /
        ``select(crossing=...)``.

        Example
        -------
        ::

            mid_line = m.model.queries.line((0, 5, 0), (5, 5, 0))
            crossing = m.model.queries.select(curves, crossing=mid_line)
        """
        return Line.through(p1, p2)

    # ------------------------------------------------------------------
    # Geometric selection
    # ------------------------------------------------------------------

    def select(
        self,
        tags: "TagsLike | Selection | str",
        *,
        dim: int | None = None,
        on=None,
        crossing=None,
        not_on=None,
        not_crossing=None,
        tol: float = 1e-6,
    ) -> Selection:
        """
        Filter entities by a geometric predicate.

        Parameters
        ----------
        tags :
            A dimtag list, a ``Selection``, or anything accepted by
            ``boundary()`` — e.g. a label string or bare integer with ``dim``.
        on :
            Keep entities that lie **entirely on** the primitive
            (all bounding-box corners within ``tol``).
        crossing :
            Keep entities that **straddle** the primitive
            (bounding-box corners on both sides).
        not_on :
            Keep entities that **do not** lie on the primitive (negation
            of ``on``).  Useful for "all faces except the bottom".
        not_crossing :
            Keep entities that **do not** straddle the primitive (negation
            of ``crossing``).  Selects entities entirely on one side.
        tol :
            Distance tolerance. Default ``1e-6``.

        Primitive formats — no imports needed
        --------------------------------------
        ``{'z': 0}``
            Axis-aligned plane z = 0.
        ``{'x': 5}``
            Axis-aligned plane x = 5.
        ``[(x1,y1,z1), (x2,y2,z2)]``
            Infinite line through 2 points  (use for curves in 2-D).
        ``[(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]``
            Infinite plane through 3 points (use for surfaces / volumes).

        Returns
        -------
        Selection
            A list subclass of ``(dim, tag)`` pairs. Call ``.select()``
            on it to filter further, or ``.tags()`` for bare integers.

        Examples
        --------
        ::

            surf   = m.model.geometry.add_rectangle(0, 0, 0, 5, 10)
            curves = m.model.queries.boundary(surf, dim=2, oriented=False)

            # by axis-aligned plane
            bottom = m.model.queries.select(curves, on={'y': 0})
            left   = m.model.queries.select(curves, on={'x': 0})

            # by arbitrary line (2 points)
            mid    = m.model.queries.select(curves, crossing=[(0,5,0),(5,5,0)])

            # by arbitrary plane (3 points) on surfaces
            faces  = m.model.queries.boundary('box', dim=3, oriented=False)
            bottom_face = m.model.queries.select(faces, on={'z': 0})
            cut_faces   = m.model.queries.select(faces,
                              crossing=[(0,0,2),(5,0,2),(0,5,2)])

            # stack: curves on z=0 AND crossing x=2.5
            result = (m.model.queries
                          .select(curves, on={'z': 0})
                          .select(crossing={'x': 2.5}))

            # bare tags for downstream calls
            m.mesh.structured.set_transfinite_curve(bottom.tags(), n=11)
        """
        # Normalise input to list of dimtags
        if isinstance(tags, Selection):
            dimtags = list(tags)
        elif isinstance(tags, str):
            # Resolve a label or PG name, then walk down to *dim* if requested.
            from ._helpers import _resolve_string_to_dimtags
            owners = _resolve_string_to_dimtags(
                tags, default_dim=dim or 3, session=self._model._parent,
            )
            if dim is None or all(d == dim for d, _ in owners):
                dimtags = owners
            elif dim == 0:
                dimtags = self.boundary_points(owners)
            elif dim == 1:
                dimtags = self.boundary_curves(owners)
            elif dim == 2:
                # 3-D owners → faces (single-step boundary).
                dimtags = list(dict.fromkeys(
                    self.boundary(owners, oriented=False)
                ))
            else:
                dimtags = owners
        else:
            dimtags = self._model._as_dimtags(tags)

        return _select_impl(dimtags, on=on, crossing=crossing,
                            not_on=not_on, not_crossing=not_crossing,
                            tol=tol, _queries=self)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def registry(self) -> pd.DataFrame:
        """
        Return a DataFrame of all entities created through this helper.

        Indexed by ``(dim, tag)`` — matching Gmsh's identity model where
        tags are only unique within a dimension.

        Columns: ``kind``, ``label``

        The ``label`` column is populated from ``g.labels`` (the single
        source of truth), not from the metadata dict.
        """
        if not self._model._metadata:
            return pd.DataFrame(columns=['dim', 'tag', 'kind', 'label'])

        # Build label reverse map from g.labels
        labels_comp = getattr(self._model._parent, 'labels', None)
        label_map: dict[tuple[int, int], str] = {}
        if labels_comp is not None:
            try:
                label_map = labels_comp.reverse_map()
            except Exception:
                pass

        rows = []
        for (dim, tag), info in self._model._metadata.items():
            row = {'dim': dim, 'tag': tag, **info}
            row['label'] = label_map.get((dim, tag), '')
            rows.append(row)
        return (
            pd.DataFrame(rows)
            .set_index(['dim', 'tag'])
            .sort_index()
        )
