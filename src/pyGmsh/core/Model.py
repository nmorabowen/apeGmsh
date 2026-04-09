from __future__ import annotations

import math
from pathlib import Path

import gmsh
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tag      = int
DimTag   = tuple[int, int]
TagsLike = Tag | list[Tag] | DimTag | list[DimTag]   # flexible input accepted everywhere


class Model:
    """
    Geometry-construction composite attached to a ``pyGmsh`` instance as
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
    * **IO**           — ``load_iges``, ``load_step``, ``load_dxf``,
      ``save_iges``, ``save_step``, ``save_dxf``
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

    def __init__(self, parent: _SessionBase) -> None:
        self._parent   = parent
        # (dim, tag) → {label, kind}
        self._registry : dict[DimTag, dict] = {}
        # Entity-selection sub-composite (model.selection.select_points(...))
        from pyGmsh.viz.Selection import SelectionComposite
        self.selection = SelectionComposite(parent=parent, model=self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[Model] {msg}")

    def _resolve_dim(self, tag: int, default_dim: int) -> int:
        """Return the dimension of *tag* from the registry.

        If the tag appears at exactly one dimension, return that dimension.
        If it appears at multiple dimensions (e.g. curve 1 *and* volume 1)
        or is not in the registry at all, fall back to *default_dim*.
        """
        found_dims = [d for (d, t) in self._registry if t == tag]
        if len(found_dims) == 1:
            return found_dims[0]
        return default_dim

    def _as_dimtags(
        self,
        tags: TagsLike,
        default_dim: int = 3,
    ) -> list[DimTag]:
        """
        Normalise any of the accepted input forms into a list of (dim, tag)
        tuples.

        For bare integer tags the dimension is resolved automatically from
        the internal registry when possible.  This allows, e.g.,
        ``fragment(objects=[vol], tools=[surf], dim=3)`` to work without
        explicit ``(dim, tag)`` tuples for the tools.

        Accepted forms
        --------------
        * ``5``                  → ``[(resolved_dim, 5)]``
        * ``[1, 2, 3]``          → ``[(resolved_dim, 1), …]``
        * ``(2, 5)``             → ``[(2, 5)]``
        * ``[(2, 5), (2, 6)]``   → ``[(2, 5), (2, 6)]``

        When the tag is not in the registry or is ambiguous (same tag at
        multiple dimensions), *default_dim* is used as before.
        """
        if isinstance(tags, int):
            return [(self._resolve_dim(tags, default_dim), tags)]

        # single (dim, tag) tuple
        if (
            isinstance(tags, tuple)
            and len(tags) == 2
            and all(isinstance(x, int) for x in tags)
        ):
            return [tags]

        out: list[DimTag] = []
        for item in tags:
            if isinstance(item, int):
                out.append((self._resolve_dim(item, default_dim), item))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                out.append((int(item[0]), int(item[1])))
            else:
                raise TypeError(f"Cannot convert {item!r} to a (dim, tag) pair.")
        return out

    def _register(self, dim: int, tag: Tag, label: str | None, kind: str) -> Tag:
        self._registry[(dim, tag)] = {
            'label': label if label else f'{kind}_{tag}',
            'kind' : kind,
        }
        return tag

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self) -> Model:
        """
        Synchronise the OCC kernel with the gmsh model topology.

        Call this explicitly when you have been batching operations with
        ``sync=False``.  Returns ``self`` for chaining.
        """
        gmsh.model.occ.synchronize()
        self._log("OCC kernel synchronised")
        return self

    # ------------------------------------------------------------------
    # Points  (dim = 0)
    # ------------------------------------------------------------------

    def add_point(
        self,
        x: float, y: float, z: float,
        *,
        mesh_size: float      = 0.0,
        lc       : float | None = None,
        label    : str | None = None,
        sync     : bool       = True,
    ) -> Tag:
        """
        Add a single point.

        Parameters
        ----------
        x, y, z   : coordinates
        mesh_size : target element size at this point (0 = use global size)
        lc        : alias for *mesh_size* (Gmsh characteristic length)

        Returns
        -------
        int tag of the new point.
        """
        if lc is not None:
            mesh_size = lc
        tag = gmsh.model.occ.addPoint(x, y, z, meshSize=mesh_size)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_point({x}, {y}, {z}) → tag {tag}")
        return self._register(0, tag, label, 'point')

    # ------------------------------------------------------------------
    # Curves  (dim = 1)
    # ------------------------------------------------------------------

    def add_line(
        self,
        start: Tag,
        end  : Tag,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a straight line segment between two existing points.

        Parameters
        ----------
        start, end : point tags
        """
        tag = gmsh.model.occ.addLine(start, end)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_line({start} → {end}) → tag {tag}")
        return self._register(1, tag, label, 'line')

    def add_arc(
        self,
        start : Tag,
        center: Tag,
        end   : Tag,
        *,
        label : str | None = None,
        sync  : bool       = True,
    ) -> Tag:
        """
        Add a circular arc defined by three existing points.

        Parameters
        ----------
        start  : point tag — start of the arc
        center : point tag — centre of the circle (not on the arc)
        end    : point tag — end of the arc

        Note
        ----
        All three points must be equidistant from the implied circle centre.
        The arc is the *shorter* of the two possible arcs unless you reverse
        the start/end order.
        """
        tag = gmsh.model.occ.addCircleArc(start, center, end)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_arc(start={start}, centre={center}, end={end}) → tag {tag}")
        return self._register(1, tag, label, 'arc')

    def add_circle(
        self,
        cx: float, cy: float, cz: float,
        radius: float,
        *,
        angle1: float     = 0.0,
        angle2: float     = 2 * math.pi,
        label : str | None = None,
        sync  : bool       = True,
    ) -> Tag:
        """
        Add a full circle (or arc sector) as a single curve entity.

        Unlike ``add_arc``, this does **not** require pre-existing point
        tags — it creates the circle directly from centre + radius.

        Parameters
        ----------
        cx, cy, cz : centre
        radius     : radius
        angle1     : start angle in radians (default 0)
        angle2     : end angle in radians   (default 2π = full circle)
        """
        tag = gmsh.model.occ.addCircle(cx, cy, cz, radius,
                                        angle1=angle1, angle2=angle2)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(
            f"add_circle(centre=({cx},{cy},{cz}), r={radius}, "
            f"[{math.degrees(angle1):.1f}°→{math.degrees(angle2):.1f}°]) → tag {tag}"
        )
        return self._register(1, tag, label, 'circle')

    def add_ellipse(
        self,
        cx: float, cy: float, cz: float,
        r_major: float, r_minor: float,
        *,
        angle1: float      = 0.0,
        angle2: float      = 2 * math.pi,
        label : str | None = None,
        sync  : bool       = True,
    ) -> Tag:
        """
        Add a full ellipse (or elliptic arc) as a single curve entity.

        Parameters
        ----------
        cx, cy, cz : centre
        r_major    : semi-major axis (along X before any rotation)
        r_minor    : semi-minor axis
        angle1     : start angle in radians
        angle2     : end angle in radians
        """
        tag = gmsh.model.occ.addEllipse(cx, cy, cz, r_major, r_minor,
                                         angle1=angle1, angle2=angle2)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(
            f"add_ellipse(centre=({cx},{cy},{cz}), a={r_major}, b={r_minor}) → tag {tag}"
        )
        return self._register(1, tag, label, 'ellipse')

    def add_spline(
        self,
        point_tags: list[Tag],
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a C2-continuous spline curve **through** the given points
        (interpolating spline).

        Parameters
        ----------
        point_tags : ordered list of point tags the spline passes through.
                     Minimum 2 points; for a closed spline repeat the first
                     tag at the end.

        Example
        -------
        ::

            p1 = g.model.add_point(0, 0, 0)
            p2 = g.model.add_point(1, 1, 0)
            p3 = g.model.add_point(2, 0, 0)
            s  = g.model.add_spline([p1, p2, p3])
        """
        if len(point_tags) < 2:
            raise ValueError("add_spline requires at least 2 point tags.")
        tag = gmsh.model.occ.addSpline(point_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_spline({point_tags}) → tag {tag}")
        return self._register(1, tag, label, 'spline')

    def add_bspline(
        self,
        point_tags   : list[Tag],
        *,
        degree        : int         = 3,
        weights       : list[float] | None = None,
        knots         : list[float] | None = None,
        multiplicities: list[int]   | None = None,
        label         : str | None  = None,
        sync          : bool        = True,
    ) -> Tag:
        """
        Add a B-spline curve with explicit control points.

        Control points are **not** interpolated (the curve is attracted to
        them, not forced through them), which is different from
        ``add_spline``.

        Parameters
        ----------
        point_tags     : control-point tags
        degree         : polynomial degree (default 3 = cubic)
        weights        : optional rational weights (len = len(point_tags))
        knots          : optional knot vector
        multiplicities : optional knot multiplicities
        """
        if len(point_tags) < 2:
            raise ValueError("add_bspline requires at least 2 point tags.")
        tag = gmsh.model.occ.addBSpline(
            point_tags,
            degree=degree,
            weights=weights        or [],
            knots=knots            or [],
            multiplicities=multiplicities or [],
        )
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_bspline(ctrl_pts={point_tags}, degree={degree}) → tag {tag}")
        return self._register(1, tag, label, 'bspline')

    def add_bezier(
        self,
        point_tags: list[Tag],
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a Bézier curve.

        Parameters
        ----------
        point_tags : control-point tags.  The curve starts at the first
                     point and ends at the last; intermediate points are
                     control handles (not interpolated).
        """
        if len(point_tags) < 2:
            raise ValueError("add_bezier requires at least 2 point tags.")
        tag = gmsh.model.occ.addBezier(point_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_bezier({point_tags}) → tag {tag}")
        return self._register(1, tag, label, 'bezier')

    # ------------------------------------------------------------------
    # Wire / surface builders  (dim = 1 → 2)
    # ------------------------------------------------------------------

    def add_curve_loop(
        self,
        curve_tags: list[Tag],
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Assemble an ordered list of curve tags into a closed wire (curve
        loop).  The result is used as input to ``add_plane_surface`` or
        ``add_surface_filling``.

        Parameters
        ----------
        curve_tags : ordered curve tags forming a closed loop.
                     Use negative tags to reverse orientation of a curve.

        Example
        -------
        ::

            loop = g.model.add_curve_loop([l1, l2, l3, l4])
            surf = g.model.add_plane_surface(loop)
        """
        tag = gmsh.model.occ.addCurveLoop(curve_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_curve_loop({curve_tags}) → tag {tag}")
        return self._register(1, tag, label, 'curve_loop')

    def add_plane_surface(
        self,
        wire_tags: Tag | list[Tag],
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Create a planar surface bounded by one or more curve loops.

        Parameters
        ----------
        wire_tags : tag (or list of tags) of curve loops.  The first loop
                    is the outer boundary; any additional loops define holes.

        Example
        -------
        ::

            outer = g.model.add_curve_loop([l1, l2, l3, l4])
            hole  = g.model.add_curve_loop([h1, h2, h3, h4])
            surf  = g.model.add_plane_surface([outer, hole])
        """
        if isinstance(wire_tags, int):
            wire_tags = [wire_tags]
        tag = gmsh.model.occ.addPlaneSurface(wire_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_plane_surface(wires={wire_tags}) → tag {tag}")
        return self._register(2, tag, label, 'plane_surface')

    def add_surface_filling(
        self,
        wire_tag: Tag,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Create a surface filling bounded by a single curve loop, using a
        Coons-patch style interpolation (non-planar surfaces).

        Parameters
        ----------
        wire_tag : tag of the bounding curve loop
        """
        tag = gmsh.model.occ.addSurfaceFilling(wire_tag)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_surface_filling(wire={wire_tag}) → tag {tag}")
        return self._register(2, tag, label, 'surface_filling')

    def add_rectangle(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float,
        *,
        rounded_radius: float = 0.0,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a rectangular planar surface in the XY plane.

        The rectangle is created at **(x, y, z)** with extents **dx** along
        X and **dy** along Y.  Combine with :meth:`rotate` and
        :meth:`translate` to orient it arbitrarily.

        Useful as a cutting tool for :meth:`fragment` — a 2D rectangle
        fragmented against a 3D solid splits the solid along the
        rectangle's plane.

        Parameters
        ----------
        x, y, z : float
            Corner of the rectangle.
        dx, dy : float
            Extents along X and Y.
        rounded_radius : float
            If > 0, rounds the four corners with this radius.
        label : str, optional
            Human-readable label stored in the internal registry.
        sync : bool
            Synchronise the OCC kernel after creation (default True).

        Returns
        -------
        Tag
            Surface tag of the new rectangle.

        Example
        -------
        ::

            # Split a solid at mid-height with a cutting plane
            bb = gmsh.model.getBoundingBox(3, 1)
            xmin, ymin, zmin, xmax, ymax, zmax = bb
            zmid = (zmin + zmax) / 2
            pad = 1.0
            rect = m1.model.add_rectangle(
                xmin - pad, ymin - pad, zmid,
                (xmax - xmin) + 2*pad,
                (ymax - ymin) + 2*pad,
            )
            result = m1.model.fragment(objects=[1], tools=[rect], dim=3)
        """
        tag = gmsh.model.occ.addRectangle(x, y, z, dx, dy, roundedRadius=rounded_radius)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(
            f"add_rectangle(origin=({x},{y},{z}), size=({dx},{dy})"
            f"{f', r={rounded_radius}' if rounded_radius else ''}) → tag {tag}"
        )
        return self._register(2, tag, label, 'rectangle')

    # ------------------------------------------------------------------
    # Primitives  (dim = 3 solids)
    # ------------------------------------------------------------------

    def add_box(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add an axis-aligned box.

        Parameters
        ----------
        x, y, z       : origin corner
        dx, dy, dz    : extents along X, Y, Z
        """
        tag = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_box(origin=({x},{y},{z}), size=({dx},{dy},{dz})) → tag {tag}")
        return self._register(3, tag, label, 'box')

    def add_sphere(
        self,
        cx: float, cy: float, cz: float,
        radius: float,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """Add a sphere centred at (cx, cy, cz) with the given radius."""
        tag = gmsh.model.occ.addSphere(cx, cy, cz, radius)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_sphere(centre=({cx},{cy},{cz}), r={radius}) → tag {tag}")
        return self._register(3, tag, label, 'sphere')

    def add_cylinder(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        radius: float,
        *,
        angle: float      = 2 * math.pi,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a cylinder.

        Parameters
        ----------
        x, y, z    : base-circle centre
        dx, dy, dz : axis direction vector (length = height of cylinder)
        radius     : base radius
        angle      : sweep angle in radians (default 2π = full cylinder)
        """
        tag = gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, radius, angle=angle)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_cylinder(base=({x},{y},{z}), axis=({dx},{dy},{dz}), r={radius}) → tag {tag}")
        return self._register(3, tag, label, 'cylinder')

    def add_cone(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        r1: float, r2: float,
        *,
        angle: float      = 2 * math.pi,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a cone / truncated cone.

        Parameters
        ----------
        x, y, z    : base-circle centre
        dx, dy, dz : axis vector
        r1         : base radius
        r2         : top radius (0 = sharp cone)
        angle      : sweep angle in radians
        """
        tag = gmsh.model.occ.addCone(x, y, z, dx, dy, dz, r1, r2, angle=angle)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_cone(base=({x},{y},{z}), r1={r1}, r2={r2}) → tag {tag}")
        return self._register(3, tag, label, 'cone')

    def add_torus(
        self,
        cx: float, cy: float, cz: float,
        r1: float, r2: float,
        *,
        angle: float      = 2 * math.pi,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a torus.

        Parameters
        ----------
        cx, cy, cz : centre
        r1         : major radius (axis to tube centre)
        r2         : minor radius (tube cross-section)
        angle      : sweep angle in radians
        """
        tag = gmsh.model.occ.addTorus(cx, cy, cz, r1, r2, angle=angle)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_torus(centre=({cx},{cy},{cz}), R={r1}, r={r2}) → tag {tag}")
        return self._register(3, tag, label, 'torus')

    def add_wedge(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        ltx: float,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a right-angle wedge.

        Parameters
        ----------
        x, y, z    : origin corner
        dx, dy, dz : extents
        ltx        : top X extent (0 = sharp wedge)
        """
        tag = gmsh.model.occ.addWedge(x, y, z, dx, dy, dz, ltx)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_wedge(origin=({x},{y},{z}), size=({dx},{dy},{dz}), ltx={ltx}) → tag {tag}")
        return self._register(3, tag, label, 'wedge')

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------
    # Every boolean op returns a list[Tag] of the surviving volumes.
    # ``objects`` and ``tools`` accept any TagsLike form.

    def _bool_op(
        self,
        fn_name   : str,
        objects   : TagsLike,
        tools     : TagsLike,
        default_dim   : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        sync          : bool = True,
    ) -> list[Tag]:
        obj_dt  = self._as_dimtags(objects, default_dim)
        tool_dt = self._as_dimtags(tools,   default_dim)
        fn      = getattr(gmsh.model.occ, fn_name)
        result, _ = fn(
            obj_dt, tool_dt,
            removeObject=remove_object,
            removeTool=remove_tool,
        )
        if sync:
            gmsh.model.occ.synchronize()

        # Clean up registry: remove consumed objects/tools
        result_set = set(result)
        if remove_object:
            for dt in obj_dt:
                if dt not in result_set:
                    self._registry.pop(dt, None)
        if remove_tool:
            for dt in tool_dt:
                if dt not in result_set:
                    self._registry.pop(dt, None)

        tags = [t for _, t in result]
        for d, t in result:
            self._register(d, t, None, fn_name)
        self._log(f"{fn_name}(obj={obj_dt}, tool={tool_dt}) → tags {tags}")
        return tags

    def fuse(
        self,
        objects : TagsLike,
        tools   : TagsLike,
        *,
        dim           : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        sync          : bool = True,
    ) -> list[Tag]:
        """
        Boolean union (A ∪ B).  Returns surviving volume tags.

        Example
        -------
        ``result = g.model.fuse(box, sphere)``
        """
        return self._bool_op('fuse', objects, tools, dim, remove_object, remove_tool, sync)

    def cut(
        self,
        objects : TagsLike,
        tools   : TagsLike,
        *,
        dim           : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        sync          : bool = True,
    ) -> list[Tag]:
        """
        Boolean difference (A − B).  Returns surviving volume tags.

        Example
        -------
        ``result = g.model.cut(box, cylinder)``
        """
        return self._bool_op('cut', objects, tools, dim, remove_object, remove_tool, sync)

    def intersect(
        self,
        objects : TagsLike,
        tools   : TagsLike,
        *,
        dim           : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        sync          : bool = True,
    ) -> list[Tag]:
        """Boolean intersection (A ∩ B).  Returns surviving volume tags."""
        return self._bool_op('intersect', objects, tools, dim, remove_object, remove_tool, sync)

    def fragment(
        self,
        objects : TagsLike,
        tools   : TagsLike,
        *,
        dim           : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        cleanup_free  : bool = True,
        sync          : bool = True,
    ) -> list[Tag]:
        """
        Boolean fragment — splits all shapes at their intersections and
        preserves all sub-volumes (useful for conformal meshing).

        Parameters
        ----------
        objects : tag(s) of the entities to fragment.
        tools : tag(s) of the cutting entities (e.g. rectangles).
            Dimensions are auto-resolved from the registry, so bare
            integer tags work even when tools have a different dimension
            than *dim*.
        dim : target dimension for bare integer tags in *objects*
            (default 3).
        remove_object, remove_tool : passed to OCC (default True).
        cleanup_free : bool
            When True (default), remove any "free" surfaces that do not
            bound a volume after the fragment operation.  This cleans up
            exterior remnants of cutting planes that fall outside the
            solid.  Set to False to keep all surface fragments.
        sync : synchronise the OCC kernel (default True).

        Returns
        -------
        list[Tag]
            Tags of all surviving entities at the target dimension.
        """
        result = self._bool_op(
            'fragment', objects, tools, dim,
            remove_object, remove_tool, sync,
        )

        if cleanup_free:
            free: list[tuple[int, int]] = []
            for _, tag_s in gmsh.model.getEntities(2):
                up, _ = gmsh.model.getAdjacencies(2, tag_s)
                if len(up) == 0:
                    free.append((2, tag_s))
            if free:
                gmsh.model.occ.remove(free, recursive=True)
                if sync:
                    gmsh.model.occ.synchronize()
                for dt in free:
                    self._registry.pop(dt, None)
                self._log(
                    f"fragment cleanup: removed {len(free)} free surface(s)"
                )

        return result

    # ------------------------------------------------------------------
    # Transforms  (all return self for chaining)
    # ------------------------------------------------------------------

    def translate(
        self,
        tags: TagsLike,
        dx: float, dy: float, dz: float,
        *,
        dim : int  = 3,
        sync: bool = True,
    ) -> Model:
        """
        Translate entities by (dx, dy, dz).

        Example
        -------
        ``g.model.translate(box, 5, 0, 0)``
        """
        gmsh.model.occ.translate(self._as_dimtags(tags, dim), dx, dy, dz)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"translate by ({dx}, {dy}, {dz})")
        return self

    def rotate(
        self,
        tags  : TagsLike,
        angle : float,
        *,
        ax: float = 0.0, ay: float = 0.0, az: float = 1.0,
        cx: float = 0.0, cy: float = 0.0, cz: float = 0.0,
        dim : int  = 3,
        sync: bool = True,
    ) -> Model:
        """
        Rotate entities around an axis through (cx, cy, cz) with direction
        (ax, ay, az) by ``angle`` radians.

        Example
        -------
        ``g.model.rotate(box, math.pi / 4, az=1)``
        """
        gmsh.model.occ.rotate(
            self._as_dimtags(tags, dim),
            cx, cy, cz,
            ax, ay, az,
            angle,
        )
        if sync:
            gmsh.model.occ.synchronize()
        self._log(
            f"rotate {math.degrees(angle):.2f}° about axis=({ax},{ay},{az}) "
            f"through ({cx},{cy},{cz})"
        )
        return self

    def scale(
        self,
        tags: TagsLike,
        sx: float, sy: float, sz: float,
        *,
        cx: float = 0.0, cy: float = 0.0, cz: float = 0.0,
        dim : int  = 3,
        sync: bool = True,
    ) -> Model:
        """
        Scale (dilate) entities by (sx, sy, sz) from centre (cx, cy, cz).

        Example
        -------
        ``g.model.scale(box, 2, 2, 2)``   # uniform double
        """
        gmsh.model.occ.dilate(
            self._as_dimtags(tags, dim),
            cx, cy, cz,
            sx, sy, sz,
        )
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"scale ({sx},{sy},{sz}) about ({cx},{cy},{cz})")
        return self

    def mirror(
        self,
        tags: TagsLike,
        a: float, b: float, c: float, d: float,
        *,
        dim : int  = 3,
        sync: bool = True,
    ) -> Model:
        """
        Mirror entities through the plane ax + by + cz + d = 0.

        Example
        -------
        ``g.model.mirror(box, 1, 0, 0, 0)``   # reflect through YZ plane
        """
        gmsh.model.occ.mirror(self._as_dimtags(tags, dim), a, b, c, d)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"mirror through plane {a}x + {b}y + {c}z + {d} = 0")
        return self

    def copy(
        self,
        tags: TagsLike,
        *,
        dim : int  = 3,
        sync: bool = True,
    ) -> list[Tag]:
        """
        Duplicate entities.  Returns the tags of the new copies.

        Example
        -------
        ``copies = g.model.copy([box, sphere])``
        """
        new_dimtags = gmsh.model.occ.copy(self._as_dimtags(tags, dim))
        if sync:
            gmsh.model.occ.synchronize()
        new_tags = [t for _, t in new_dimtags]
        self._log(f"copy → new tags {new_tags}")
        return new_tags

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
        dim_tags = self._as_dimtags(tags, dim)
        gmsh.model.occ.remove(dim_tags, recursive=recursive)
        if sync:
            gmsh.model.occ.synchronize()
        for dt in dim_tags:
            self._registry.pop(dt, None)
        self._log(f"removed {dim_tags} (recursive={recursive})")

    def remove_duplicates(
        self,
        *,
        tolerance: float | None = None,
        sync     : bool         = True,
    ) -> Model:
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

            imported = g.model.load_iges("Frame3D.iges", highest_dim_only=False)
            g.model.remove_duplicates(tolerance=1e-3)
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
        stale_dts = [dt for dt in self._registry if dt not in surviving]
        for dt in stale_dts:
            del self._registry[dt]

        after = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
        removed = {d: before[d] - after[d] for d in range(4) if before[d] != after[d]}
        tol_str = f"tolerance={tolerance}" if tolerance is not None else ""
        self._log(
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
    ) -> Model:
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

            m1.model.load_iges("Frame3D.iges", highest_dim_only=False)
            m1.remove_duplicates(tolerance=1.0)
            m1.model.make_conformal(dims=[1], tolerance=1.0)
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
            self._log("make_conformal(): no entities found, nothing to do")
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
        old_registry = dict(self._registry)
        self._registry.clear()
        for d in range(4):
            for _, tag in gmsh.model.getEntities(d):
                old_entry = old_registry.get((d, tag))
                if old_entry:
                    self._registry[(d, tag)] = old_entry
                else:
                    self._registry[(d, tag)] = {'label': f'entity_{tag}', 'kind': 'fragment'}

        after = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
        delta = {d: after[d] - before[d] for d in range(4) if before[d] != after[d]}
        tol_str = f", tolerance={tolerance}" if tolerance is not None else ""
        self._log(
            f"make_conformal(dims={dims}{tol_str}): entity delta={delta} "
            f"(before={before}, after={after})"
        )
        return self

    # Alias so both ``model.fragment_all()`` and ``model.make_conformal()``
    # work — mirrors the Assembly API naming convention.
    fragment_all = make_conformal

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def _import_shapes(
        self,
        file_path      : Path,
        kind           : str,
        highest_dim_only: bool,
        sync           : bool,
    ) -> dict[int, list[Tag]]:
        """
        Core import helper shared by ``load_iges`` and ``load_step``.

        Calls ``gmsh.model.occ.importShapes``, captures the returned
        (dim, tag) pairs, registers every imported entity, and returns a
        dimension-indexed dict so callers can address entities immediately.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` — e.g. ``{3: [1, 2], 2: [5, 6, 7]}``
        """
        raw: list[tuple[int, int]] = gmsh.model.occ.importShapes(
            str(file_path),
            highestDimOnly=highest_dim_only,
        )
        if sync:
            gmsh.model.occ.synchronize()

        result: dict[int, list[Tag]] = {}
        for dim, tag in raw:
            self._register(dim, tag, None, kind)
            result.setdefault(dim, []).append(tag)

        dim_summary = {d: len(ts) for d, ts in result.items()}
        self._log(f"loaded {kind.upper()} ← {file_path.name}  {dim_summary}")
        return result

    def load_iges(
        self,
        file_path       : Path | str,
        *,
        highest_dim_only: bool = True,
        sync            : bool = True,
    ) -> dict[int, list[Tag]]:
        """
        Import an IGES file into the current model.

        All imported entities are registered and their tags are returned so
        you can immediately use them in boolean ops or transforms.

        Parameters
        ----------
        highest_dim_only : bool
            If True (default) only the highest-dimension entities are
            returned and registered (volumes for solids, surfaces for
            surface models).  Set to False to capture every sub-entity
            (faces, edges, vertices) as well.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` indexed by dimension.

        Example
        -------
        ::

            imported = g.model.load_iges("part.iges")
            bodies   = imported[3]           # all imported volume tags
            flange   = bodies[0]             # first imported volume

            boss = g.model.add_cylinder(10, 10, 0,  0, 0, 5,  3)
            result = g.model.fuse(flange, boss)
        """
        return self._import_shapes(
            Path(file_path), 'iges', highest_dim_only, sync
        )

    def load_step(
        self,
        file_path       : Path | str,
        *,
        highest_dim_only: bool = True,
        sync            : bool = True,
    ) -> dict[int, list[Tag]]:
        """
        Import a STEP file into the current model.

        All imported entities are registered and their tags are returned so
        you can immediately use them in boolean ops or transforms.

        Parameters
        ----------
        highest_dim_only : bool
            If True (default) only the highest-dimension entities are
            returned and registered.  Set to False to include all
            sub-entities.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` indexed by dimension.

        Example
        -------
        ::

            imported = g.model.load_step("assembly.step")
            bodies   = imported[3]
            g.model.translate(bodies, 0, 0, 50)   # lift the whole import
        """
        return self._import_shapes(
            Path(file_path), 'step', highest_dim_only, sync
        )

    def save_iges(self, file_path: Path | str) -> None:
        """
        Export the current model to IGES.

        The ``.iges`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.iges')
        gmsh.write(str(file_path))
        self._log(f"saved IGES → {file_path}")

    def save_step(self, file_path: Path | str) -> None:
        """
        Export the current model to STEP.

        The ``.step`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.step')
        gmsh.write(str(file_path))
        self._log(f"saved STEP → {file_path}")

    # ------------------------------------------------------------------
    # DXF (AutoCAD) — parsed with ezdxf, geometry built via OCC kernel
    # ------------------------------------------------------------------

    @staticmethod
    def _dxf_point_key(
        x: float, y: float, z: float, tol: float,
    ) -> tuple[int, int, int]:
        """Discretise coordinates into a grid cell for O(1) dedup."""
        inv = 1.0 / tol
        return (round(x * inv), round(y * inv), round(z * inv))

    def load_dxf(
        self,
        file_path: Path | str,
        *,
        point_tolerance: float = 1e-6,
        create_physical_groups: bool = True,
        sync: bool = True,
    ) -> dict[str, dict[int, list[Tag]]]:
        """
        Import a DXF file into the current model.

        Uses ``ezdxf`` to parse the DXF (supports all AutoCAD versions
        from R12 to 2024+), then builds Gmsh geometry through the OCC
        kernel.  AutoCAD **layers** become Gmsh physical groups
        automatically.

        Supported DXF entity types: ``LINE``, ``ARC``, ``CIRCLE``,
        ``LWPOLYLINE``, ``POLYLINE``, ``SPLINE``, ``POINT``.

        Parameters
        ----------
        file_path : Path or str
            Path to the ``.dxf`` file.
        point_tolerance : float
            Distance below which two DXF endpoints are considered
            coincident and share a single Gmsh point.  Default ``1e-6``.
        create_physical_groups : bool
            If True (default), a physical group is created for each DXF
            layer.  If False, entities are created but no physical groups
            are made (useful when you want to assign groups manually).
        sync : bool
            Synchronise the OCC kernel after import (default True).

        Returns
        -------
        dict[str, dict[int, list[Tag]]]
            ``{layer_name: {dim: [tag, ...]}}``

            Each key is a DXF layer name.  Values map entity dimension
            to lists of Gmsh tags created from that layer.

        Example
        -------
        ::

            # AutoCAD drawing with layers: "C80x80", "V30x50"
            layers = g.model.load_dxf("frame_2D.dxf")

            # layers == {
            #     "C80x80": {1: [1, 2, 3, 4]},
            #     "V30x50": {1: [5, 6, 7, 8, 9]},
            # }

            # Physical groups are already created — ready for meshing.
            # Access beam curves:
            beam_curves = layers["V30x50"][1]
        """
        try:
            import ezdxf
        except ImportError:
            raise ImportError(
                "ezdxf is required for DXF import.  "
                "Install it with:  pip install ezdxf"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"DXF file not found: {file_path}")

        doc = ezdxf.readfile(str(file_path))
        msp = doc.modelspace()

        # -- Point deduplication ------------------------------------------
        tol = point_tolerance
        _pt_cache: dict[tuple[int, int, int], Tag] = {}

        def _get_or_add_point(x: float, y: float, z: float) -> Tag:
            key = self._dxf_point_key(x, y, z, tol)
            if key in _pt_cache:
                return _pt_cache[key]
            tag = gmsh.model.occ.addPoint(x, y, z)
            _pt_cache[key] = tag
            self._register(0, tag, None, 'dxf_point')
            return tag

        # -- Entity conversion by type ------------------------------------
        # We store each curve's geometry fingerprint (sorted bounding-box
        # coords) → layer name so we can rebuild the mapping after
        # removeAllDuplicates() potentially renumbers tags.

        def _bbox_key(
            x0: float, y0: float, z0: float,
            x1: float, y1: float, z1: float,
        ) -> tuple[float, ...]:
            """Canonical bounding box: min coords then max coords."""
            return (
                round(min(x0, x1), 8), round(min(y0, y1), 8),
                round(min(z0, z1), 8), round(max(x0, x1), 8),
                round(max(y0, y1), 8), round(max(z0, z1), 8),
            )

        # fingerprint → layer name  (for curves, dim=1)
        _geom_to_layer: dict[tuple[float, ...], str] = {}
        # point key → layer name  (for dim=0 DXF POINT entities)
        _pt_to_layer: dict[tuple[int, int, int], str] = {}

        for entity in msp:
            layer = entity.dxf.layer
            etype = entity.dxftype()

            if etype == 'POINT':
                pt = entity.dxf.location
                _get_or_add_point(pt.x, pt.y, pt.z)
                key = self._dxf_point_key(pt.x, pt.y, pt.z, tol)
                _pt_to_layer[key] = layer

            elif etype == 'LINE':
                s = entity.dxf.start
                e = entity.dxf.end
                p1 = _get_or_add_point(s.x, s.y, s.z)
                p2 = _get_or_add_point(e.x, e.y, e.z)
                gmsh.model.occ.addLine(p1, p2)
                _geom_to_layer[_bbox_key(s.x, s.y, s.z, e.x, e.y, e.z)] = layer

            elif etype == 'ARC':
                c = entity.dxf.center
                r = entity.dxf.radius
                a1 = math.radians(entity.dxf.start_angle)
                a2 = math.radians(entity.dxf.end_angle)
                if a2 <= a1:
                    a2 += 2.0 * math.pi
                gmsh.model.occ.addCircle(
                    c.x, c.y, c.z, r, angle1=a1, angle2=a2,
                )
                # Compute arc endpoints for the fingerprint
                sx = c.x + r * math.cos(a1)
                sy = c.y + r * math.sin(a1)
                ex = c.x + r * math.cos(a2)
                ey = c.y + r * math.sin(a2)
                _geom_to_layer[_bbox_key(sx, sy, c.z, ex, ey, c.z)] = layer

            elif etype == 'CIRCLE':
                c = entity.dxf.center
                r = entity.dxf.radius
                gmsh.model.occ.addCircle(c.x, c.y, c.z, r)
                _geom_to_layer[_bbox_key(
                    c.x - r, c.y - r, c.z, c.x + r, c.y + r, c.z,
                )] = layer

            elif etype in ('LWPOLYLINE', 'POLYLINE'):
                pts: list[Tag] = []
                if etype == 'LWPOLYLINE':
                    vertices = list(entity.get_points(format='xyz'))
                else:
                    vertices = [
                        (v.dxf.location.x, v.dxf.location.y,
                         v.dxf.location.z)
                        for v in entity.vertices
                    ]
                for vx, vy, vz in vertices:
                    pts.append(_get_or_add_point(vx, vy, vz))

                is_closed = (
                    getattr(entity.dxf, 'flags', 0) & 1
                    if etype == 'POLYLINE' else entity.closed
                )
                vert_pairs = list(zip(vertices, vertices[1:]))
                if is_closed and len(vertices) > 2:
                    vert_pairs.append((vertices[-1], vertices[0]))

                pt_pairs = list(zip(pts, pts[1:]))
                if is_closed and len(pts) > 2:
                    pt_pairs.append((pts[-1], pts[0]))

                for (v_s, v_e), (p1, p2) in zip(vert_pairs, pt_pairs):
                    gmsh.model.occ.addLine(p1, p2)
                    _geom_to_layer[_bbox_key(
                        v_s[0], v_s[1], v_s[2],
                        v_e[0], v_e[1], v_e[2],
                    )] = layer

            elif etype == 'SPLINE':
                ctrl_pts: list[Tag] = []
                for cp in entity.control_points:
                    ctrl_pts.append(
                        _get_or_add_point(
                            cp[0], cp[1],
                            cp[2] if len(cp) > 2 else 0.0,
                        )
                    )
                if len(ctrl_pts) >= 2:
                    gmsh.model.occ.addBSpline(ctrl_pts)
                    cps = entity.control_points
                    xs = [c[0] for c in cps]
                    ys = [c[1] for c in cps]
                    zs = [c[2] if len(c) > 2 else 0.0 for c in cps]
                    _geom_to_layer[_bbox_key(
                        min(xs), min(ys), min(zs),
                        max(xs), max(ys), max(zs),
                    )] = layer

            else:
                self._log(f"DXF: skipped unsupported entity {etype} "
                          f"on layer '{layer}'")

        # -- Merge duplicate points & synchronise --------------------------
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # -- Rebuild layer mapping from surviving entities -----------------
        layers: dict[str, dict[int, list[Tag]]] = {}

        for dim, tag in gmsh.model.getEntities(1):
            bb = gmsh.model.getBoundingBox(dim, tag)
            key = _bbox_key(*bb)
            layer_name = _geom_to_layer.get(key)
            if layer_name:
                self._register(dim, tag, None, 'dxf')
                layers.setdefault(layer_name, {}).setdefault(1, []).append(tag)
            else:
                # Fallback: assign to "_unmatched"
                self._register(dim, tag, None, 'dxf')
                layers.setdefault("_unmatched", {}).setdefault(1, []).append(tag)

        for dim, tag in gmsh.model.getEntities(0):
            self._register(dim, tag, None, 'dxf_point')

        # -- Physical groups from layers ----------------------------------
        if create_physical_groups:
            for layer_name, dim_tags in layers.items():
                for dim, tags in dim_tags.items():
                    if tags:
                        gmsh.model.addPhysicalGroup(
                            dim, tags, name=layer_name,
                        )

        # -- Summary ------------------------------------------------------
        layer_summary = {
            name: {d: len(ts) for d, ts in ents.items()}
            for name, ents in layers.items()
        }
        self._log(f"loaded DXF ← {file_path.name}  layers={layer_summary}")
        return layers

    def save_dxf(self, file_path: Path | str) -> None:
        """
        Export the current model to DXF.

        The ``.dxf`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.dxf')
        gmsh.write(str(file_path))
        self._log(f"saved DXF → {file_path}")

    def save_msh(self, file_path: Path | str) -> None:
        """
        Export the current model to Gmsh's native MSH format.

        Unlike STEP/IGES, this preserves **everything**: geometry, mesh,
        physical groups, and partition data.

        The ``.msh`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.msh')
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(file_path))
        self._log(f"saved MSH → {file_path}")

    def load_msh(
        self,
        file_path: Path | str,
    ) -> dict[int, list[Tag]]:
        """
        Import a Gmsh ``.msh`` file using ``gmsh.merge``.

        Unlike ``load_iges`` / ``load_step``, this preserves physical
        groups, mesh data, and partition info — because ``.msh`` is
        Gmsh's native format.

        Parameters
        ----------
        file_path : Path or str
            Path to the ``.msh`` file.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` of all entities after merge.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"MSH file not found: {file_path}")

        gmsh.merge(str(file_path))

        result: dict[int, list[Tag]] = {}
        for d in range(4):
            for dim, tag in gmsh.model.getEntities(d):
                result.setdefault(dim, []).append(tag)

        dim_summary = {d: len(ts) for d, ts in result.items()}
        self._log(f"loaded MSH ← {file_path.name}  {dim_summary}")
        return result

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def viewer(self, **kwargs):
        """Open the interactive Qt model viewer.

        This is the main entry point for visual model inspection,
        physical-group management, and entity picking.  The viewer
        provides a 3D viewport with:

        * **Browser** — model tree (physical groups, unassigned, instances)
        * **View** — toggle entity labels / tags on screen
        * **Filter** — dimension + label checkboxes
        * **Physical Groups** — create / apply / modify / delete groups
        * **Preferences** — visual tuning (sizes, opacity, colors, font)
        * **Console / Entity Info** — collapsible bottom docks

        All kwargs are forwarded to :meth:`selection.picker` (e.g.
        ``dims``, ``point_size``, ``line_width``, ``surface_opacity``).

        Typical usage::

            m.model.viewer()

        Returns
        -------
        SelectionPicker — the picker instance after the window closes.
        """
        return self.selection.picker(**kwargs)

    def viewer_fast(self, **kwargs):
        """Open the model viewer with fast mesh-based tessellation.

        Same as :meth:`viewer` but generates a temporary coarse mesh
        and builds actors from ``getNodes``/``getElements`` instead of
        parametric sampling.  Much faster for large BRep models
        (hundreds of surfaces/curves).

        All kwargs are forwarded to :meth:`selection.picker`.
        """
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
        """
        Open Gmsh's native FLTK viewer with entity labels pre-enabled
        so every point/curve/surface shows its tag directly on the
        3-D view.

        Usage inside the FLTK window:
            - Hover an entity -> its type + tag appears in the status bar.
            - Ctrl+Shift+V -> Visibility -> 'Elementary entities' to
              isolate/toggle specific entities.
            - Close the window to return control to Python.

        The picks do **not** flow back into Python automatically — read
        the tags from the labels and feed them to
        ``m1.model.selection.select_*(tags=[...])``.  For an end-to-end
        click-to-selection workflow use the Qt picker
        (``m1.model.selection.picker()``) — create, rename and delete
        physical groups from its toolbar or the tree's right-click menu.

        Parameters
        ----------
        show_points    : pre-enable point-label rendering
        show_curves    : pre-enable curve-label rendering
        show_surfaces  : pre-enable surface-label rendering
        show_volumes   : pre-enable volume-label rendering
        verbose        : print instructions before opening the window
        """
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
            print("  Hover an entity to see its type + tag in the status bar.")
            print("  Ctrl+Shift+V -> Visibility -> 'Elementary entities'")
            print("  Close the window to return here.")
        gmsh.fltk.run()

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
        if not self._registry:
            return pd.DataFrame(columns=['dim', 'tag', 'kind', 'label'])
        rows = [
            {'dim': dim, 'tag': tag, **info}
            for (dim, tag), info in self._registry.items()
        ]
        return (
            pd.DataFrame(rows)
            .set_index(['dim', 'tag'])
            .sort_index()
        )

    def __repr__(self) -> str:
        return f"Model(name={self._parent.model_name!r}, registered={len(self._registry)})"
