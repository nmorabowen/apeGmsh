from __future__ import annotations
import math
import gmsh

from ._helpers import Tag, DimTag, TagsLike


class _GeometryMixin:
    """Points, curves, surfaces, and solid primitive creation methods."""

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

    def add_wire(
        self,
        curve_tags: list[Tag],
        *,
        check_closed: bool       = False,
        label       : str | None = None,
        sync        : bool       = True,
    ) -> Tag:
        """
        Assemble an ordered list of curve tags into an OpenCASCADE wire
        (open or closed).  Wires are the path input for sweep operations
        (:meth:`sweep`) and the section input for lofted volumes
        (:meth:`thru_sections`).

        Unlike :meth:`add_curve_loop`, a wire does **not** need to be
        closed.  This is what makes it suitable as a sweep path.

        Parameters
        ----------
        curve_tags : ordered curve tags.  Curves must be connected
            end-to-end but may share only geometrically identical
            endpoints (OCC allows topologically distinct but coincident
            points).
        check_closed : if True, the underlying OCC call verifies that
            the wire forms a closed loop and raises otherwise.
        label : registry label (for later resolution by name).

        Returns
        -------
        int tag of the new wire (a dim-1 entity).

        Example
        -------
        ::

            p0 = g.model.add_point(0, 0, 0, sync=False)
            p1 = g.model.add_point(1, 0, 0, sync=False)
            p2 = g.model.add_point(1, 1, 0, sync=False)
            p3 = g.model.add_point(1, 1, 2, sync=False)
            l1 = g.model.add_line(p0, p1, sync=False)
            l2 = g.model.add_line(p1, p2, sync=False)
            l3 = g.model.add_line(p2, p3, sync=False)
            path = g.model.add_wire([l1, l2, l3], label="sweep_path")
        """
        tag = gmsh.model.occ.addWire(curve_tags, checkClosed=check_closed)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"add_wire({curve_tags}, closed={check_closed}) → tag {tag}")
        return self._register(1, tag, label, 'wire')

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
