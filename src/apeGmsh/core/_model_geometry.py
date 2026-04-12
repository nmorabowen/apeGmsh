from __future__ import annotations
import math
from typing import TYPE_CHECKING, Literal

import gmsh
import numpy as np
from numpy import ndarray

from ._helpers import Tag

if TYPE_CHECKING:
    from .Model import Model


# Unit vectors for the three coordinate axes — used by
# :meth:`_Geometry.add_axis_cutting_plane`.  Module-level constants so
# they are not rebuilt on every call.  Callers that rotate the
# returned vector MUST ``.copy()`` first.
_AXIS_UNIT_VEC: dict[str, ndarray] = {
    'x': np.array([1.0, 0.0, 0.0]),
    'y': np.array([0.0, 1.0, 0.0]),
    'z': np.array([0.0, 0.0, 1.0]),
}


class _Geometry:
    """Points, curves, surfaces, and solid primitive creation methods."""

    def __init__(self, model: "Model") -> None:
        self._model = model

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
        self._model._log(f"add_point({x}, {y}, {z}) -> tag {tag}")
        return self._model._register(0, tag, label, 'point')

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
        self._model._log(f"add_line({start} -> {end}) -> tag {tag}")
        return self._model._register(1, tag, label, 'line')

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
        self._model._log(f"add_arc(start={start}, centre={center}, end={end}) -> tag {tag}")
        return self._model._register(1, tag, label, 'arc')

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
        self._model._log(
            f"add_circle(centre=({cx},{cy},{cz}), r={radius}, "
            f"[{math.degrees(angle1):.1f}°->{math.degrees(angle2):.1f}°]) -> tag {tag}"
        )
        return self._model._register(1, tag, label, 'circle')

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
        self._model._log(
            f"add_ellipse(centre=({cx},{cy},{cz}), a={r_major}, b={r_minor}) -> tag {tag}"
        )
        return self._model._register(1, tag, label, 'ellipse')

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

            p1 = g.model.geometry.add_point(0, 0, 0)
            p2 = g.model.geometry.add_point(1, 1, 0)
            p3 = g.model.geometry.add_point(2, 0, 0)
            s  = g.model.geometry.add_spline([p1, p2, p3])
        """
        if len(point_tags) < 2:
            raise ValueError("add_spline requires at least 2 point tags.")
        tag = gmsh.model.occ.addSpline(point_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_spline({point_tags}) -> tag {tag}")
        return self._model._register(1, tag, label, 'spline')

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
        self._model._log(f"add_bspline(ctrl_pts={point_tags}, degree={degree}) -> tag {tag}")
        return self._model._register(1, tag, label, 'bspline')

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
        self._model._log(f"add_bezier({point_tags}) -> tag {tag}")
        return self._model._register(1, tag, label, 'bezier')

    # ------------------------------------------------------------------
    # Wire / surface builders  (dim = 1 -> 2)
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

            p0 = g.model.geometry.add_point(0, 0, 0, sync=False)
            p1 = g.model.geometry.add_point(1, 0, 0, sync=False)
            p2 = g.model.geometry.add_point(1, 1, 0, sync=False)
            p3 = g.model.geometry.add_point(1, 1, 2, sync=False)
            l1 = g.model.geometry.add_line(p0, p1, sync=False)
            l2 = g.model.geometry.add_line(p1, p2, sync=False)
            l3 = g.model.geometry.add_line(p2, p3, sync=False)
            path = g.model.geometry.add_wire([l1, l2, l3], label="sweep_path")
        """
        tag = gmsh.model.occ.addWire(curve_tags, checkClosed=check_closed)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_wire({curve_tags}, closed={check_closed}) -> tag {tag}")
        return self._model._register(1, tag, label, 'wire')

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

            loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
            surf = g.model.geometry.add_plane_surface(loop)
        """
        tag = gmsh.model.occ.addCurveLoop(curve_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_curve_loop({curve_tags}) -> tag {tag}")
        return self._model._register(1, tag, label, 'curve_loop')

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

            outer = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
            hole  = g.model.geometry.add_curve_loop([h1, h2, h3, h4])
            surf  = g.model.geometry.add_plane_surface([outer, hole])
        """
        if isinstance(wire_tags, int):
            wire_tags = [wire_tags]
        tag = gmsh.model.occ.addPlaneSurface(wire_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_plane_surface(wires={wire_tags}) -> tag {tag}")
        return self._model._register(2, tag, label, 'plane_surface')

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
        self._model._log(f"add_surface_filling(wire={wire_tag}) -> tag {tag}")
        return self._model._register(2, tag, label, 'surface_filling')

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
            rect = m1.model.geometry.add_rectangle(
                xmin - pad, ymin - pad, zmid,
                (xmax - xmin) + 2*pad,
                (ymax - ymin) + 2*pad,
            )
            result = m1.model.boolean.fragment(objects=[1], tools=[rect], dim=3)
        """
        tag = gmsh.model.occ.addRectangle(x, y, z, dx, dy, roundedRadius=rounded_radius)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(
            f"add_rectangle(origin=({x},{y},{z}), size=({dx},{dy})"
            f"{f', r={rounded_radius}' if rounded_radius else ''}) -> tag {tag}"
        )
        return self._model._register(2, tag, label, 'rectangle')

    def add_cutting_plane(
        self,
        point          : list[float] | ndarray,
        normal_vector  : list[float] | ndarray,
        *,
        size           : float | None = None,
        label          : str | None   = None,
        sync           : bool         = True,
    ) -> Tag:
        """
        Create a square planar surface through ``point`` with the given
        normal, suitable for clipping / section / visualisation views.

        The surface is a plain BRep face built from 4 points + 4 lines
        + a curve loop + a plane surface, so it behaves exactly like
        any other registered surface (it can be selected, meshed as a
        discrete 2-D grid, exported to STEP, etc.).  It is *not* a
        Gmsh clipping plane in the rendering sense — it is real
        geometry.

        Parameters
        ----------
        point : array-like of 3 floats
            A point on the plane.  The square is centred here.
        normal_vector : array-like of 3 floats
            Plane normal.  Need not be unit-length — it is normalised
            internally.
        size : float, optional
            Edge length of the square.  When ``None`` (default), size
            is picked as ``2 × max(model_bbox_diagonal, 1.0)`` so the
            square comfortably overhangs the current model.
        label : str, optional
            Human-readable label stored in the internal registry.
        sync : bool, optional
            Synchronise the OCC kernel after creation (default True).

        Returns
        -------
        Tag
            Surface tag of the new cutting plane.

        Example
        -------
        ::

            # A vertical plane through (0, 0, 0) with normal (1, 0, 0)
            g.model.geometry.add_cutting_plane(
                point=(0, 0, 0), normal_vector=(1, 0, 0),
            )
        """
        p = np.asarray(point, dtype=float)
        n = np.asarray(normal_vector, dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm == 0.0:
            raise ValueError("normal_vector must be non-zero")
        n = n / n_norm

        # Pick size from the current model bounding box when not given.
        # ``getBoundingBox`` requires a synchronised OCC state, so this
        # is the only hard reason for a pre-sync.  When the caller
        # supplies an explicit ``size``, we do not sync until the end.
        if size is None:
            gmsh.model.occ.synchronize()
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
            size = 2.0 * max(diag, 1.0)

        # Orthonormal basis (u, v) spanning the plane.  v does not need
        # explicit normalisation: if ``n`` and ``u`` are unit and
        # orthogonal, then ``cross(n, u)`` is also unit.
        ref = (
            np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = np.cross(n, ref)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        half = size / 2.0
        corner_coeffs = [(-half, -half), (half, -half), (half, half), (-half, half)]
        corners = [p + a * u + b * v for a, b in corner_coeffs]

        # Delegate to the existing BRep primitives so the new points,
        # lines, loop, and surface all flow through ``_register`` /
        # ``_log`` with the correct kinds.  Defer every sub-sync so we
        # sync exactly once at the end (or zero times if sync=False).
        pt_tags  = [
            self.add_point(float(c[0]), float(c[1]), float(c[2]), sync=False)
            for c in corners
        ]
        ln_tags  = [
            self.add_line(pt_tags[i], pt_tags[(i + 1) % 4], sync=False)
            for i in range(4)
        ]
        loop_tag = self.add_curve_loop(ln_tags, sync=False)
        tag      = self.add_plane_surface(loop_tag, sync=False, label=label)

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"add_cutting_plane(point={tuple(p)}, normal={tuple(n)}, "
            f"size={size}) -> tag {tag}"
        )
        # ``add_plane_surface`` already registered the tag as
        # ``plane_surface``.  Re-label it as a cutting plane and stash
        # the defining point + unit normal so downstream operations
        # (``cut_by_plane``) can recover the orientation without
        # re-querying Gmsh or re-parsing the geometry.
        entry = self._model._registry.get((2, tag))
        if entry is not None:
            entry['kind']   = 'cutting_plane'
            entry['point']  = tuple(float(x) for x in p)
            entry['normal'] = tuple(float(x) for x in n)
        return tag

    def add_axis_cutting_plane(
        self,
        axis          : Literal['x', 'y', 'z'],
        offset        : float = 0.0,
        *,
        origin        : list[float] | ndarray | None = None,
        rotation      : float                        = 0.0,
        rotation_about: Literal['x', 'y', 'z'] | None = None,
        label         : str | None                   = None,
        sync          : bool                         = True,
    ) -> Tag:
        """
        Add an axis-aligned cutting plane, optionally tilted by a rotation.

        Convenience wrapper around :meth:`add_cutting_plane`.  The plane is
        initially defined as **normal to** ``axis`` (so ``axis='z'`` produces
        a horizontal XY-plane).  It is then:

        1. Offset along its base normal by ``offset``.
        2. Rotated by ``rotation`` degrees about ``rotation_about``
           (if both are given), producing a tilted plane through the same
           anchor point.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Axis the plane is **normal to**.  ``'z'`` -> XY plane, etc.
        offset : float, optional
            Signed distance along the base normal from ``origin``
            (or from the global origin if ``origin`` is None).
        origin : array-like of 3 floats, optional
            Anchor point before the offset is applied.  Defaults to (0, 0, 0).
        rotation : float, optional
            Rotation angle in **degrees**.  Requires ``rotation_about``
            to have any effect — passing ``rotation`` without
            ``rotation_about`` raises ``ValueError`` so silent
            no-ops do not sneak through.
        rotation_about : {'x', 'y', 'z'}, optional
            Axis about which the base normal is rotated.  Must differ
            from ``axis`` for the rotation to have any effect.
        label : str, optional
            Human-readable label stored in the internal registry.
        sync : bool, optional
            Synchronise the OCC kernel after creation (default True).

        Returns
        -------
        Tag
            Surface tag of the new cutting plane.

        Examples
        --------
        Horizontal plane at z = 3::

            g.model.geometry.add_axis_cutting_plane('z', offset=3.0)

        Vertical YZ-plane passing through x = 1.5::

            g.model.geometry.add_axis_cutting_plane('x', offset=1.5)

        Horizontal plane tilted 15° about the y-axis::

            g.model.geometry.add_axis_cutting_plane(
                'z', offset=0.0,
                rotation=15.0, rotation_about='y',
            )
        """
        if axis not in _AXIS_UNIT_VEC:
            raise ValueError(
                f"axis must be one of 'x', 'y', 'z'; got {axis!r}"
            )
        if rotation != 0.0 and rotation_about is None:
            raise ValueError(
                "rotation was provided without rotation_about — pass "
                "rotation_about='x'|'y'|'z' to tilt the plane, or drop "
                "the rotation argument to keep it axis-aligned."
            )

        base_normal = _AXIS_UNIT_VEC[axis].copy()

        if rotation_about is not None and rotation != 0.0:
            if rotation_about not in _AXIS_UNIT_VEC:
                raise ValueError(
                    f"rotation_about must be one of 'x', 'y', 'z'; "
                    f"got {rotation_about!r}"
                )
            if rotation_about == axis:
                self._model._log(
                    f"add_axis_cutting_plane: rotation_about='{rotation_about}' "
                    f"equals axis='{axis}'; rotation has no effect."
                )
            else:
                theta = np.deg2rad(rotation)
                k     = _AXIS_UNIT_VEC[rotation_about]
                c, s  = np.cos(theta), np.sin(theta)
                # Rodrigues' rotation formula — k is a unit vector.
                base_normal = (
                    base_normal * c
                    + np.cross(k, base_normal) * s
                    + k * np.dot(k, base_normal) * (1.0 - c)
                )

        if origin is None:
            origin_arr = np.zeros(3)
        else:
            origin_arr = np.asarray(origin, dtype=float)
        if origin_arr.shape != (3,):
            raise ValueError(
                f"origin must be a length-3 vector; got shape {origin_arr.shape}"
            )

        point = origin_arr + offset * base_normal

        self._model._log(
            f"add_axis_cutting_plane(axis={axis!r}, offset={offset}, "
            f"origin={tuple(origin_arr)}, rotation={rotation}, "
            f"rotation_about={rotation_about!r}) "
            f"-> point={tuple(point)}, normal={tuple(base_normal)}"
        )

        return self.add_cutting_plane(
            point=point,
            normal_vector=base_normal,
            label=label,
            sync=sync,
        )

    # ------------------------------------------------------------------
    # Cutting operations
    # ------------------------------------------------------------------

    def _collect_volume_tags(self) -> list[Tag]:
        """Return every registered 3-D entity tag from the registry."""
        return [
            tag for (dim, tag) in self._model._registry.keys()
            if dim == 3
        ]

    def _resolve_label_to_tags(self, label: str) -> list[Tag]:
        """Find every dim=3 entity in the registry whose ``label``
        field matches *label*.

        Returns an empty list when no match is found — the caller
        decides whether to raise.
        """
        return [
            tag
            for (dim, tag), entry in self._model._registry.items()
            if dim == 3 and entry.get('label') == label
        ]

    def _normalize_solid_input(
        self,
        solid: Tag | str | list[Tag | str] | None,
        collector,
    ) -> list[Tag]:
        """Coerce the ``solid`` argument into a concrete list of tags.

        Accepts:

        * ``None`` — every registered volume (via ``collector()``).
        * ``int`` — a single tag, wrapped into a one-element list.
        * ``str`` — a registry label; resolved to every dim=3 entity
          whose ``label`` field matches.  This is the key feature
          for chained slicing: after a slice the original tag is
          consumed but the label can be propagated to the fragments,
          so the next slice can still say ``slice("shaft", ...)``.
        * ``list`` — a mix of ints and/or strings, each resolved
          individually and concatenated.
        """
        if solid is None:
            tags = collector()
        elif isinstance(solid, str):
            tags = self._resolve_label_to_tags(solid)
        elif isinstance(solid, int):
            tags = [int(solid)]
        elif isinstance(solid, (list, tuple)):
            tags = []
            for item in solid:
                if isinstance(item, str):
                    tags.extend(self._resolve_label_to_tags(item))
                else:
                    tags.append(int(item))
        else:
            tags = [int(solid)]
        if not tags:
            raise ValueError(
                "no solids to cut — pass an explicit tag list, a "
                "label string, or register at least one volume "
                "before calling the cut"
            )
        return tags

    def cut_by_surface(
        self,
        solid          : Tag | str | list[Tag | str] | None,
        surface        : Tag,
        *,
        keep_surface   : bool = True,
        remove_original: bool = True,
        label          : str | None = None,
        sync           : bool = True,
    ) -> list[Tag]:
        """
        Split one or more solids with an arbitrary cutting surface.

        Uses OCC's ``fragment`` operation under the hood, which splits
        every input shape at its intersections and keeps **all**
        resulting sub-shapes.  Unlike :meth:`cut_by_plane`, this method
        does not classify the output pieces — callers that need
        "above/below" semantics should use :meth:`cut_by_plane` (which
        delegates here and adds the classification step).

        Parameters
        ----------
        solid : Tag, list[Tag], or None
            Volume(s) to cut.  When ``None``, every registered volume
            in the model is cut against the surface.
        surface : Tag
            The cutting surface.  Can be any registered 2-D entity —
            a plane from :meth:`add_cutting_plane`, a STEP-imported
            trimmed surface, a Coons patch, etc.
        keep_surface : bool, default True
            Leave the (now-trimmed) surface in the model after the
            cut.  Useful when you want to mesh the cut interface as a
            shared face for conformal ties.  Set to ``False`` to
            delete it.
        remove_original : bool, default True
            Consume the original solid(s) so only the cut pieces
            remain.  When ``False``, OCC keeps the originals alongside
            the pieces, which usually produces overlapping geometry
            and is rarely what you want.
        label : str, optional
            Label applied to every new volume fragment in the
            registry.  Pass ``None`` to leave the fragments unlabelled.
        sync : bool, default True
            Synchronise the OCC kernel after the cut.

        Returns
        -------
        list[Tag]
            Solid tags of the fragments produced by the cut, in the
            order OCC returns them.  An empty list means the cut
            produced nothing new (shouldn't happen unless the surface
            misses every input solid entirely).

        Example
        -------
        ::

            box   = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            plane = g.model.geometry.add_axis_cutting_plane('z', offset=0.5)
            pieces = g.model.geometry.cut_by_surface(box, plane)
        """
        solid_tags = self._normalize_solid_input(solid, self._collect_volume_tags)

        out_dimtags, _ = gmsh.model.occ.fragment(
            [(3, int(t)) for t in solid_tags],
            [(2, int(surface))],
            removeObject=remove_original,
            removeTool=not keep_surface,
        )

        new_volume_tags: list[Tag] = [
            int(t) for (d, t) in out_dimtags if d == 3
        ]

        # Collect the original labels from consumed solids BEFORE
        # removing them from the registry.  Fragments inherit the
        # original label so that label-based lookups (e.g.
        # ``slice("shaft", axis='z', ...)`` after a previous slice)
        # still resolve to the surviving pieces.
        inherited_label = label
        if inherited_label is None and remove_original:
            original_labels = {
                entry.get('label')
                for t in solid_tags
                for entry in [self._model._registry.get((3, int(t)), {})]
                if entry.get('label')
                and entry.get('label') != f"{entry.get('kind', '')}_{t}"
            }
            # If all consumed solids share the same user-supplied label,
            # propagate it.  Mixed labels -> don't guess, leave unlabeled.
            if len(original_labels) == 1:
                inherited_label = original_labels.pop()

        if remove_original:
            for t in solid_tags:
                self._model._registry.pop((3, int(t)), None)

        for t in new_volume_tags:
            self._model._register(3, t, inherited_label, 'cut_fragment')

        # The trimmed cutting surface may have been split into multiple
        # faces; keep whichever 2-D entities came out so they stay
        # addressable.
        if keep_surface:
            surviving_surfaces = [
                int(t) for (d, t) in out_dimtags if d == 2
            ]
            for t in surviving_surfaces:
                if (2, t) not in self._model._registry:
                    self._model._register(2, t, None, 'cut_interface')

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"cut_by_surface(solids={solid_tags}, surface={int(surface)}) "
            f"-> {len(new_volume_tags)} volume fragment(s): {new_volume_tags}"
        )
        return new_volume_tags

    def cut_by_plane(
        self,
        solid          : Tag | str | list[Tag | str] | None,
        plane          : Tag,
        *,
        keep_plane     : bool = True,
        remove_original: bool = True,
        above_direction: list[float] | ndarray | None = None,
        label_above    : str | None = None,
        label_below    : str | None = None,
        sync           : bool = True,
    ) -> tuple[list[Tag], list[Tag]]:
        """
        Split one or more solids with a plane and classify the
        resulting pieces by which side of the plane they sit on.

        Thin wrapper around :meth:`cut_by_surface` that additionally
        computes which fragments are "above" (same side as the plane
        normal) vs "below" the plane.  The normal direction is
        resolved from, in order of priority:

        1. An explicit ``above_direction`` argument.
        2. The ``normal`` and ``point`` stashed in the registry by
           :meth:`add_cutting_plane` / :meth:`add_axis_cutting_plane`.
        3. ``gmsh.model.getNormal`` sampled at the parametric centre
           of the plane surface.

        Parameters
        ----------
        solid : Tag, list[Tag], or None
            Volume(s) to cut.  ``None`` = every registered volume.
        plane : Tag
            Planar surface to cut with.  Ideally built by
            :meth:`add_cutting_plane` so its normal and point are in
            the registry; other planar surfaces work too but require
            an explicit ``above_direction`` or fall back to querying
            Gmsh.
        keep_plane : bool, default True
            Leave the trimmed plane in the model as a registered
            surface (useful for meshing the cut interface).
        remove_original : bool, default True
            Consume the original solid(s).
        above_direction : array-like of 3 floats, optional
            Override the plane's normal direction.  Pieces whose
            centroid dotted with this vector (relative to the plane
            point) is positive are classified as "above".
        label_above, label_below : str, optional
            Labels applied to the above / below fragment solids.
        sync : bool, default True
            Synchronise the OCC kernel after the cut.

        Returns
        -------
        tuple[list[Tag], list[Tag]]
            ``(above_tags, below_tags)`` — solid tags on each side of
            the plane, classified by the sign of
            ``(centroid - plane_point) · normal``.

        Example
        -------
        ::

            col = g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
            pl  = g.model.geometry.add_axis_cutting_plane('z', offset=1.5)

            top, bot = g.model.geometry.cut_by_plane(
                col, pl,
                label_above="col_upper", label_below="col_lower",
            )
        """
        plane_tag = int(plane)

        # Resolve the plane's defining point and normal.
        entry = self._model._registry.get((2, plane_tag))
        stashed_normal = entry.get('normal') if entry else None
        stashed_point  = entry.get('point')  if entry else None

        if above_direction is not None:
            normal = np.asarray(above_direction, dtype=float)
            norm_len = float(np.linalg.norm(normal))
            if norm_len == 0.0:
                raise ValueError("above_direction must be non-zero")
            normal = normal / norm_len
            # Need a point on the plane for the dot-product
            # classification.  Prefer the stashed one; otherwise ask
            # Gmsh for a vertex of the plane.
            if stashed_point is not None:
                point = np.asarray(stashed_point, dtype=float)
            else:
                point = self._any_point_on_surface(plane_tag)
        elif stashed_normal is not None and stashed_point is not None:
            normal = np.asarray(stashed_normal, dtype=float)
            point  = np.asarray(stashed_point,  dtype=float)
        else:
            # Fall back to gmsh.model.getNormal at the parametric
            # midpoint.  Requires a synced model.
            gmsh.model.occ.synchronize()
            try:
                nxyz = gmsh.model.getNormal(plane_tag, [0.5, 0.5])
            except Exception as exc:
                raise ValueError(
                    f"cut_by_plane: plane tag {plane_tag} has no registry "
                    f"normal and Gmsh could not compute one — pass "
                    f"above_direction=... explicitly. ({exc})"
                ) from exc
            normal = np.asarray(nxyz, dtype=float)
            norm_len = float(np.linalg.norm(normal))
            if norm_len == 0.0:
                raise ValueError(
                    f"cut_by_plane: plane tag {plane_tag} returned a zero "
                    f"normal from Gmsh; pass above_direction=..."
                )
            normal = normal / norm_len
            point = self._any_point_on_surface(plane_tag)

        # Perform the actual cut via the general surface method.
        fragments = self.cut_by_surface(
            solid,
            plane_tag,
            keep_surface=keep_plane,
            remove_original=remove_original,
            label=None,          # we re-label by side below
            sync=False,          # single sync at the end of this method
        )

        # Classify each fragment by the sign of (centroid - p) . n.
        above_tags: list[Tag] = []
        below_tags: list[Tag] = []
        for t in fragments:
            com = np.asarray(
                gmsh.model.occ.getCenterOfMass(3, int(t)),
                dtype=float,
            )
            signed = float(np.dot(com - point, normal))
            if signed >= 0.0:
                above_tags.append(t)
                if label_above is not None:
                    entry = self._model._registry.get((3, t))
                    if entry is not None:
                        entry['label'] = label_above
            else:
                below_tags.append(t)
                if label_below is not None:
                    entry = self._model._registry.get((3, t))
                    if entry is not None:
                        entry['label'] = label_below

        if not above_tags or not below_tags:
            self._model._log(
                f"cut_by_plane: WARNING plane {plane_tag} produced "
                f"only one side ({len(above_tags)} above, "
                f"{len(below_tags)} below) — the plane may not "
                f"intersect the solid(s)"
            )

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"cut_by_plane(plane={plane_tag}) -> "
            f"above={above_tags}, below={below_tags}"
        )
        return above_tags, below_tags

    def _any_point_on_surface(self, surface_tag: Tag) -> ndarray:
        """Return a point in space known to lie on ``surface_tag``.

        Used by :meth:`cut_by_plane` when the plane's defining point
        is not in the registry — we ask OCC for the centre of mass
        of the surface, which always lies on the surface for planar
        faces (and is a reasonable proxy for curved ones).
        """
        com = gmsh.model.occ.getCenterOfMass(2, int(surface_tag))
        return np.asarray(com, dtype=float)

    # ------------------------------------------------------------------
    # Slice (atomic cut + cleanup)
    # ------------------------------------------------------------------

    def slice(
        self,
        solid   : Tag | str | list[Tag | str] | None = None,
        *,
        axis    : Literal['x', 'y', 'z'],
        offset  : float = 0.0,
        classify: bool = False,
        label   : str | None = None,
        sync    : bool = True,
    ) -> list[Tag] | tuple[list[Tag], list[Tag]]:
        """
        Slice solids at an axis-aligned plane in one atomic call.

        Internally creates a temporary cutting plane, fragments the
        solids, removes the cutting plane (and any trimmed surfaces
        it left behind), and returns the volume fragments.  No
        orphaned geometry is left in the model.

        Parameters
        ----------
        solid : Tag, list[Tag], or None
            Volume(s) to slice.  ``None`` slices every registered
            volume in the model.
        axis : {'x', 'y', 'z'}
            Axis the plane is **normal to**.  ``'z'`` slices with
            a horizontal XY-plane, etc.
        offset : float, default 0.0
            Signed distance along the axis from the origin.
        classify : bool, default False
            When True, returns ``(positive_side, negative_side)``
            classified by the plane's normal direction (the positive
            axis direction).  When False (default), returns all
            fragments as a flat list.
        label : str, optional
            Label applied to every fragment in the registry.
        sync : bool, default True
            Synchronise the OCC kernel after the operation.

        Returns
        -------
        list[Tag]
            All volume fragments (when ``classify=False``).
        tuple[list[Tag], list[Tag]]
            ``(positive_side, negative_side)`` fragments classified
            by which side of the plane each piece's centroid sits on
            (when ``classify=True``).

        Example
        -------
        ::

            # Slice a box at y = 0.5
            box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            pieces = g.model.geometry.slice(box, axis='y', offset=0.5)

            # Slice and classify
            top, bot = g.model.geometry.slice(
                box, axis='z', offset=0.5, classify=True,
            )

            # Slice all volumes at x = 0
            g.model.geometry.slice(axis='x', offset=0.0)
        """
        # Build the cutting plane (deferred sync — we sync once at
        # the end).
        plane_tag = self.add_axis_cutting_plane(
            axis, offset=offset, sync=False,
        )

        if classify:
            above, below = self.cut_by_plane(
                solid, plane_tag,
                keep_plane=False,
                label_above=label,
                label_below=label,
                sync=False,
            )
            result: list[Tag] | tuple[list[Tag], list[Tag]] = (above, below)
        else:
            fragments = self.cut_by_surface(
                solid, plane_tag,
                keep_surface=False,
                label=label,
                sync=False,
            )
            result = fragments

        # Clean up any surviving cutting-plane surfaces and their
        # sub-entities (edges, points) that OCC may have left behind
        # after the fragment operation.  The plane tag itself was
        # consumed by fragment (remove_tool=True via keep_surface=
        # False), but fragment can produce trimmed remnants with
        # fresh tags.  Walk the 2-D entities and remove any that
        # were registered as 'cutting_plane' or 'cut_interface'.
        for dt in list(self._model._registry.keys()):
            if dt[0] != 2:
                continue
            entry = self._model._registry.get(dt)
            if entry and entry.get('kind') in ('cutting_plane', 'cut_interface'):
                try:
                    gmsh.model.occ.remove([dt], recursive=True)
                except Exception:
                    pass
                self._model._registry.pop(dt, None)

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"slice(axis={axis!r}, offset={offset}, classify={classify})"
        )
        return result

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
        self._model._log(f"add_box(origin=({x},{y},{z}), size=({dx},{dy},{dz})) -> tag {tag}")
        return self._model._register(3, tag, label, 'box')

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
        self._model._log(f"add_sphere(centre=({cx},{cy},{cz}), r={radius}) -> tag {tag}")
        return self._model._register(3, tag, label, 'sphere')

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
        self._model._log(f"add_cylinder(base=({x},{y},{z}), axis=({dx},{dy},{dz}), r={radius}) -> tag {tag}")
        return self._model._register(3, tag, label, 'cylinder')

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
        self._model._log(f"add_cone(base=({x},{y},{z}), r1={r1}, r2={r2}) -> tag {tag}")
        return self._model._register(3, tag, label, 'cone')

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
        self._model._log(f"add_torus(centre=({cx},{cy},{cz}), R={r1}, r={r2}) -> tag {tag}")
        return self._model._register(3, tag, label, 'torus')

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
        self._model._log(f"add_wedge(origin=({x},{y},{z}), size=({dx},{dy},{dz}), ltx={ltx}) -> tag {tag}")
        return self._model._register(3, tag, label, 'wedge')
