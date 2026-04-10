from __future__ import annotations

import math
from typing import TYPE_CHECKING

import gmsh

from ._helpers import Tag, DimTag, TagsLike

if TYPE_CHECKING:
    from .Model import Model


class _TransformsMixin:
    """Transform and extrusion/revolution methods extracted from Model."""

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
            f"rotate {math.degrees(angle):.2f}\u00b0 about axis=({ax},{ay},{az}) "
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
        self._log(f"copy \u2192 new tags {new_tags}")
        return new_tags

    # ------------------------------------------------------------------
    # Extrusion / Revolution  (generative sweep ops)
    # ------------------------------------------------------------------

    def extrude(
        self,
        tags: TagsLike,
        dx: float, dy: float, dz: float,
        *,
        dim          : int             = 2,
        num_elements : list[int] | None = None,
        heights      : list[float] | None = None,
        recombine    : bool            = False,
        sync         : bool            = True,
    ) -> list[DimTag]:
        """
        Linear extrusion \u2014 sweeps entities along (dx, dy, dz).

        Creates new geometry one dimension up: point \u2192 curve,
        curve \u2192 surface, surface \u2192 volume.

        Parameters
        ----------
        tags : entities to extrude.
        dx, dy, dz : extrusion vector.
        dim : default dimension for bare integer tags (default 2).
        num_elements : structured layer counts, e.g. ``[10]`` for
            10 layers.  Empty list (default) = unstructured.
        heights : relative heights per layer, e.g. ``[0.3, 0.7]``.
            Must sum to 1.0 when provided.  Empty = uniform layers.
        recombine : if True, produce hex/quad elements instead of
            tet/tri (requires structured layers).
        sync : synchronise OCC kernel after extrusion (default True).

        Returns
        -------
        list[DimTag]
            All generated (dim, tag) pairs.  For a surface \u2192 volume
            extrusion the list contains the top face, the volume, and
            the lateral faces \u2014 index into it to assign physical groups.

        Example
        -------
        ::

            surf = g.model.add_plane_surface(loop)
            out  = g.model.extrude(surf, 0, 0, 3.0, num_elements=[10])
            # out[0] = (2, top_face), out[1] = (3, volume), ...
        """
        dt = self._as_dimtags(tags, dim)
        ne = num_elements if num_elements is not None else []
        ht = heights if heights is not None else []
        result: list[tuple[int, int]] = gmsh.model.occ.extrude(
            dt, dx, dy, dz,
            numElements=ne,
            heights=ht,
            recombine=recombine,
        )
        if sync:
            gmsh.model.occ.synchronize()
        for d, t in result:
            self._register(d, t, None, 'extrude')
        self._log(
            f"extrude({dt}, ({dx},{dy},{dz})) \u2192 {len(result)} entities"
        )
        return result

    def revolve(
        self,
        tags  : TagsLike,
        angle : float,
        *,
        x : float = 0.0, y : float = 0.0, z : float = 0.0,
        ax: float = 0.0, ay: float = 0.0, az: float = 1.0,
        dim          : int             = 2,
        num_elements : list[int] | None = None,
        heights      : list[float] | None = None,
        recombine    : bool            = False,
        sync         : bool            = True,
    ) -> list[DimTag]:
        """
        Revolution \u2014 sweeps entities around an axis.

        Parameters
        ----------
        tags : entities to revolve.
        angle : sweep angle in radians (2\u03c0 for full revolution).
        x, y, z : point on the rotation axis.
        ax, ay, az : direction vector of the rotation axis.
        dim : default dimension for bare integer tags (default 2).
        num_elements, heights, recombine : same as :meth:`extrude`.
        sync : synchronise OCC kernel (default True).

        Returns
        -------
        list[DimTag]
            All generated (dim, tag) pairs.

        Example
        -------
        ::

            # Revolve a cross-section 360\u00b0 around the Y axis
            out = g.model.revolve(profile, 2 * math.pi, ay=1)
        """
        dt = self._as_dimtags(tags, dim)
        ne = num_elements if num_elements is not None else []
        ht = heights if heights is not None else []
        result: list[tuple[int, int]] = gmsh.model.occ.revolve(
            dt, x, y, z, ax, ay, az, angle,
            numElements=ne,
            heights=ht,
            recombine=recombine,
        )
        if sync:
            gmsh.model.occ.synchronize()
        for d, t in result:
            self._register(d, t, None, 'revolve')
        self._log(
            f"revolve({dt}, angle={math.degrees(angle):.1f}\u00b0, "
            f"axis=({ax},{ay},{az}) through ({x},{y},{z})) "
            f"\u2192 {len(result)} entities"
        )
        return result

    # ------------------------------------------------------------------
    # Sweep along a path  (OCC pipe)
    # ------------------------------------------------------------------

    def sweep(
        self,
        profiles : TagsLike,
        path     : Tag,
        *,
        dim       : int        = 2,
        trihedron : str        = "DiscreteTrihedron",
        label     : str | None = None,
        sync      : bool       = True,
    ) -> list[DimTag]:
        """
        Sweep one or more profile entities along an arbitrary wire.

        This is the "constant-section sweep" operation: a single profile
        (point, curve, or surface) is translated along *path*, generating
        geometry one dimension up — point → curve, curve → surface,
        surface → volume.  Unlike :meth:`extrude` the path does not have
        to be a straight line: it can be any OCC wire built from lines,
        arcs, splines, or a mix, assembled via
        :meth:`~Model.add_wire`.

        Parameters
        ----------
        profiles : entity or entities to sweep.  For a solid you
            normally pass a plane surface.
        path : tag of an OCC wire to sweep along (use
            :meth:`~Model.add_wire` to build it).  A curve_loop can be
            used for closed paths.
        dim : default dimension for bare integer tags in *profiles*
            (default 2).
        trihedron : how the profile frame is transported along the
            path.  One of ``"DiscreteTrihedron"`` (default),
            ``"CorrectedFrenet"``, ``"Fixed"``, ``"Frenet"``,
            ``"ConstantNormal"``, ``"Darboux"``, ``"GuideAC"``,
            ``"GuidePlan"``, ``"GuideACWithContact"``,
            ``"GuidePlanWithContact"``.  Most structural workflows
            want the default; use ``"Frenet"`` for smooth curves
            without inflection and ``"Fixed"`` to keep the profile's
            orientation constant in world space.
        label : optional label applied to the highest-dimension
            survivor of the sweep (the volume for a surface sweep).
        sync : synchronise the OCC kernel after the call (default
            True).

        Returns
        -------
        list[DimTag]
            All generated ``(dim, tag)`` pairs.  Index into the list
            to grab the volume, the lateral faces, or the end caps and
            assign them to physical groups.

        Example
        -------
        ::

            section = g.model.add_plane_surface(loop, label="I_section")
            path    = g.model.add_wire([arc1, line1, arc2], label="beam_path")
            out     = g.model.sweep(section, path, label="curved_beam")
        """
        dt = self._as_dimtags(profiles, dim)
        result: list[tuple[int, int]] = gmsh.model.occ.addPipe(
            dt, int(path), trihedron=trihedron,
        )
        if sync:
            gmsh.model.occ.synchronize()

        if result and label is not None:
            max_dim = max(d for d, _ in result)
            for d, t in result:
                lbl = label if d == max_dim else None
                self._register(d, t, lbl, 'sweep')
        else:
            for d, t in result:
                self._register(d, t, None, 'sweep')

        self._log(
            f"sweep({dt}, path={path}, trihedron={trihedron!r}) "
            f"→ {len(result)} entities"
        )
        return result

    # ------------------------------------------------------------------
    # Variable-section sweep  (OCC thru-sections / loft)
    # ------------------------------------------------------------------

    def thru_sections(
        self,
        wires : list[Tag],
        *,
        make_solid     : bool       = True,
        make_ruled     : bool       = False,
        max_degree     : int        = -1,
        continuity     : str        = "",
        parametrization: str        = "",
        smoothing      : bool       = False,
        label          : str | None = None,
        sync           : bool       = True,
    ) -> list[DimTag]:
        """
        Variable-section sweep — loft a volume (or surface shell)
        through an ordered list of wires.

        This is the right operation when the cross-section *changes*
        along the sweep: a tapered column, a transition piece between
        two different flange shapes, a blended nozzle.  Each wire
        defines one intermediate section; OCC builds a smooth surface
        that interpolates between them and (optionally) caps the ends
        to produce a solid.

        All wires should be topologically similar (same number of
        sub-curves in the same order) for reliable lofting.  Open wires
        produce a skin; closed wires with ``make_solid=True`` produce a
        solid.

        Parameters
        ----------
        wires : ordered list of wire tags (build each one with
            :meth:`~Model.add_wire`).  At least two wires are required.
        make_solid : if True (default), cap the ends and return a
            solid; if False, return only the skinned surface(s).
        make_ruled : if True, force the lateral faces to be ruled
            surfaces (linear interpolation between adjacent sections).
        max_degree : maximum degree of the resulting surface
            (``-1`` = OCC default).
        continuity : ``"C0"``, ``"G1"``, ``"C1"``, ``"G2"``, ``"C2"``,
            ``"C3"``, or ``"CN"`` (``""`` = OCC default).
        parametrization : ``"ChordLength"``, ``"Centripetal"``, or
            ``"IsoParametric"`` (``""`` = OCC default).
        smoothing : if True, apply a smoothing pass to the resulting
            surface.
        label : optional label applied to the highest-dimension
            survivor (the volume when ``make_solid=True``).
        sync : synchronise the OCC kernel after the call (default
            True).

        Returns
        -------
        list[DimTag]
            All generated ``(dim, tag)`` pairs.

        Example
        -------
        ::

            w_base = g.model.add_wire([lb1, lb2, lb3, lb4])
            w_top  = g.model.add_wire([lt1, lt2, lt3, lt4])
            out    = g.model.thru_sections(
                [w_base, w_top],
                make_solid=True,
                label="tapered_column",
            )
        """
        if len(wires) < 2:
            raise ValueError(
                "thru_sections requires at least two wires (got "
                f"{len(wires)})."
            )
        result: list[tuple[int, int]] = gmsh.model.occ.addThruSections(
            list(wires),
            makeSolid      = make_solid,
            makeRuled      = make_ruled,
            maxDegree      = max_degree,
            continuity     = continuity,
            parametrization= parametrization,
            smoothing      = smoothing,
        )
        if sync:
            gmsh.model.occ.synchronize()

        if result and label is not None:
            max_dim = max(d for d, _ in result)
            for d, t in result:
                lbl = label if d == max_dim else None
                self._register(d, t, lbl, 'thru_sections')
        else:
            for d, t in result:
                self._register(d, t, None, 'thru_sections')

        self._log(
            f"thru_sections(wires={list(wires)}, solid={make_solid}) "
            f"→ {len(result)} entities"
        )
        return result
