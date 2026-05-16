"""
_Structured — transfinite / recombine / smoothing / compound control.

Accessed via ``g.mesh.structured``.  Owns the "structured meshing"
knobs: transfinite constraints, recombination into quads/hexes,
Laplacian smoothing, and compound merging.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from .Mesh import Mesh


from apeGmsh._types import DimTag


class _Structured:
    """Transfinite constraints, recombination, smoothing, compounds."""

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # Transfinite constraints
    # ------------------------------------------------------------------

    def _resolve(self, tag, dim: int) -> list[int]:
        """Resolve a flexible ref (int, str, dim-tag, or list) to tags.

        Delegates to the central :func:`resolve_to_tags` helper so all
        structured methods accept the same ref shapes used elsewhere in
        the API.
        """
        from apeGmsh.core._helpers import resolve_to_tags
        return resolve_to_tags(tag, dim=dim, session=self._mesh._parent)

    def set_transfinite_curve(
        self,
        tag,
        n_nodes  : int,
        *,
        mesh_type: str   = "Progression",
        coef     : float = 1.0,
    ) -> "_Structured":
        """Force a curve to be meshed with a deterministic node count and distribution.

        A "transfinite" curve has its nodes placed by formula rather than by
        the unstructured mesher.  This is the building block for structured
        meshes: when every bounding curve of a surface is transfinite, the
        surface itself can be made transfinite (a structured grid); same for
        volumes built from transfinite surfaces.

        Parameters
        ----------
        tag :
            Curve identifier.  Accepts an int tag, a label string, a
            physical-group name, a ``(1, tag)`` dimtag, or a list of any
            of these (constraint applied to each).  Works with
            :meth:`Selection.tags() <apeGmsh.core._selection.Selection.tags>`
            output.
        n_nodes : int
            Number of nodes along the curve (≥ 2).  ``n_nodes - 1``
            elements are produced.
        mesh_type : str, default ``"Progression"``
            Node distribution rule.  One of:

            - ``"Progression"`` — geometric progression.  ``coef`` is
              the ratio between successive intervals.
              ``coef = 1`` ⇒ uniform; ``coef > 1`` clusters nodes
              toward the curve's end point; ``coef < 1`` clusters
              toward the start.
            - ``"Bump"`` — symmetric biasing.  ``coef > 1`` clusters
              toward the middle; ``coef < 1`` clusters toward both
              ends; ``coef = 1`` ⇒ uniform.
            - ``"Beta"`` — Beta-distribution biasing (Gmsh ≥ 4.10).
        coef : float, default ``1.0``
            Distribution parameter — meaning depends on ``mesh_type``
            (see above).  Use ``1.0`` for a uniform distribution.

        Returns
        -------
        _Structured
            ``self`` for chaining.

        Notes
        -----
        The constraint is only honored if the curve is part of a
        transfinite surface (and the surface part of a transfinite volume,
        for 3-D meshes).  A transfinite curve in isolation will simply
        seed the unstructured mesher with that node count.

        Curve **direction** matters for non-uniform distributions: which
        end is "start" vs "end" follows the curve's intrinsic orientation
        in Gmsh.  Reverse the orientation (or pass ``coef = 1/r`` instead
        of ``r``) to flip the clustering.

        Examples
        --------
        Uniform 11-node spacing on a single curve::

            m.mesh.structured.set_transfinite_curve(curve_tag, n_nodes=11)

        Geometric refinement toward the curve's end (boundary-layer style)::

            m.mesh.structured.set_transfinite_curve(
                curve_tag, n_nodes=21,
                mesh_type="Progression", coef=1.1,
            )

        Symmetric refinement toward the middle (capture a feature at mid-span)::

            m.mesh.structured.set_transfinite_curve(
                curve_tag, n_nodes=21,
                mesh_type="Bump", coef=0.25,
            )

        Pass a Selection's tags to constrain many curves at once::

            edges = m.model.queries.select("box", dim=1)
            m.mesh.structured.set_transfinite_curve(
                edges.parallel_to("z").tags(),
                n_nodes=21,
            )
        """
        for t in self._resolve(tag, dim=1):
            gmsh.model.mesh.setTransfiniteCurve(t, n_nodes,
                                                 meshType=mesh_type, coef=coef)
            self._mesh._directives.append({
                'kind': 'transfinite_curve', 'tag': t,
                'n_nodes': n_nodes, 'mesh_type': mesh_type, 'coef': coef,
            })
            self._mesh._log(
                f"set_transfinite_curve(tag={t}, n={n_nodes}, "
                f"type={mesh_type!r}, coef={coef})"
            )
        return self

    def set_transfinite_surface(
        self,
        tag,
        *,
        arrangement: str            = "Left",
        corners    : list[int] | None = None,
    ) -> "_Structured":
        """Force a surface to be meshed as a structured grid.

        A transfinite surface has its interior nodes laid out by transfinite
        interpolation between its bounding curves.  Combined with transfinite
        curves on every bounding edge, this produces a fully structured
        surface mesh (triangles by default, quads with
        :meth:`set_recombine`).

        Parameters
        ----------
        tag :
            Surface identifier.  Accepts an int tag, label string,
            physical-group name, ``(2, tag)`` dimtag, or list of any.
        arrangement : str, default ``"Left"``
            Diagonal direction for the structured triangles.  Ignored
            once the surface is recombined to quads.  Values:

            - ``"Left"``        all diagonals slant the same way
            - ``"Right"``       all diagonals slant the other way
            - ``"AlternateLeft"`` alternating pattern, starting Left
            - ``"AlternateRight"`` alternating pattern, starting Right
        corners : list[int] | None, default ``None``
            Tags of the 3 or 4 corner points defining the structured
            topology.  Required when the surface has **more than 4
            bounding curves** (e.g. after a face split) or when Gmsh
            can't auto-detect the corners.  Pass ``None`` for a clean
            3- or 4-sided face.

        Returns
        -------
        _Structured
            ``self`` for chaining.

        Prerequisites
        -------------
        Every bounding curve of the surface must already be transfinite
        (see :meth:`set_transfinite_curve`).  Opposite edges must have
        matching ``n_nodes`` (or the mesher will fail at generation time
        with a "transfinite surface: inconsistent number of nodes" error).

        Examples
        --------
        Clean rectangle::

            m.mesh.structured.set_transfinite_surface(face_tag)

        Surface with 5+ bounding curves — pick the 4 logical corners::

            m.mesh.structured.set_transfinite_surface(
                face_tag, corners=[p1, p2, p3, p4],
            )

        Apply to every horizontal face of a layer::

            faces = m.model.queries.select("layer_1", dim=2)
            m.mesh.structured.set_transfinite_surface(
                faces.normal_along("z").tags(),
            )

        Notes
        -----
        For a quad/hex mesh, follow with :meth:`set_recombine` on the
        same surface.  The all-in-one helper :meth:`set_transfinite_box`
        does this automatically for clean hex volumes.
        """
        for t in self._resolve(tag, dim=2):
            gmsh.model.mesh.setTransfiniteSurface(t, arrangement=arrangement,
                                                   cornerTags=corners or [])
            self._mesh._directives.append({
                'kind': 'transfinite_surface', 'tag': t,
                'arrangement': arrangement,
                'corners': corners or [],
            })
            self._mesh._log(
                f"set_transfinite_surface(tag={t}, "
                f"arrangement={arrangement!r})"
            )
        return self

    def set_transfinite_volume(
        self,
        tag,
        *,
        corners: list[int] | None = None,
    ) -> "_Structured":
        """Force a volume to be meshed as a structured grid.

        A transfinite volume has its interior nodes laid out by transfinite
        interpolation between its bounding surfaces.  Combined with
        :meth:`set_recombine` on each face, this produces a pure hex mesh.

        Parameters
        ----------
        tag :
            Volume identifier.  Accepts an int tag, label string,
            physical-group name, ``(3, tag)`` dimtag, or list of any.
        corners : list[int] | None, default ``None``
            Tags of the 6 (prism) or 8 (hex) corner points that define
            the structured topology.  Required when the volume has
            irregular face counts or when Gmsh can't auto-detect the
            corners.  Pass ``None`` for a clean 5- or 6-faced volume.

        Returns
        -------
        _Structured
            ``self`` for chaining.

        Prerequisites
        -------------
        - Every bounding surface must already be transfinite
          (:meth:`set_transfinite_surface`).
        - Opposite surfaces must have matching node counts on their
          shared edges.

        Examples
        --------
        Single hex (after edges and faces are already transfinite)::

            m.mesh.structured.set_transfinite_volume(vol_tag)

        Notes
        -----
        For the common case of "transfinite + recombine + hex on every
        face of a clean box," use :meth:`set_transfinite_box` instead —
        it sets the constraints on edges, faces, and volume in one call.
        """
        for t in self._resolve(tag, dim=3):
            gmsh.model.mesh.setTransfiniteVolume(t, cornerTags=corners or [])
            self._mesh._directives.append({
                'kind': 'transfinite_volume', 'tag': t,
                'corners': corners or [],
            })
            self._mesh._log(f"set_transfinite_volume(tag={t})")
        return self

    def set_transfinite_automatic(
        self,
        dim_tags    : list[DimTag] | None = None,
        *,
        corner_angle: float = 2.35,
        recombine   : bool  = True,
    ) -> "_Structured":
        """Auto-detect transfinite-compatible entities and constrain them.

        Walks each surface and volume in ``dim_tags`` (or the entire model
        if ``None``).  A face counts as "transfinite-compatible" if it
        has 3 or 4 corners whose angles are within ``corner_angle`` of a
        flat angle (π radians).  Compatible faces and the volumes built
        from them get transfinite + (optionally) recombine constraints
        applied automatically.

        Useful as a fallback after boolean operations leave you with a
        mix of clean and split faces — :meth:`set_transfinite_box`
        would fail on the split faces, but ``automatic`` simply skips
        them.

        Parameters
        ----------
        dim_tags : list[(dim, tag)] | None, default ``None``
            Restrict the search to these entities.  ``None`` ⇒ walk
            every entity in the model.
        corner_angle : float, default ``2.35``
            Threshold angle in **radians** for the "is this a corner?"
            test.  The default ≈ 135° is Gmsh's own — vertices whose
            interior angle deviates from π (180°) by less than this
            tolerance are not counted as corners.  Reduce for stricter
            corner detection; raise to admit more rounded transitions.
        recombine : bool, default ``True``
            Recombine detected faces into quads (and volumes into hexes).
            Set ``False`` for a transfinite tet mesh.

        Returns
        -------
        _Structured
            ``self`` for chaining.

        Examples
        --------
        Mesh-everything-it-can fallback after a boolean op::

            m.model.boolean.fragment("box_a", "box_b")
            m.mesh.structured.set_transfinite_automatic()
            m.mesh.generation.generate(dim=3)

        Restrict to a subset (only the volumes you care about)::

            m.mesh.structured.set_transfinite_automatic(
                dim_tags=[(3, t) for t in vol_tags],
            )
        """
        gmsh.model.mesh.setTransfiniteAutomatic(
            dimTags=dim_tags or [],
            cornerAngle=corner_angle,
            recombine=recombine,
        )
        self._mesh._directives.append({
            'kind': 'transfinite_automatic',
            'dim_tags': dim_tags or [],
            'corner_angle': corner_angle,
            'recombine': recombine,
        })
        self._mesh._log(
            f"set_transfinite_automatic("
            f"corner_angle={math.degrees(corner_angle):.1f}°, "
            f"recombine={recombine})"
        )
        return self

    def set_transfinite_box(
        self,
        vol,
        *,
        size: float | None = None,
        n   : int | None   = None,
        recombine: bool    = True,
    ) -> "_Structured":
        """Apply transfinite + recombine constraints to a clean hex volume.

        Captures the full "structured hex" setup in one call.  Walks the
        volume's bounding curves, assigns a node count per edge, marks
        every bounding surface as transfinite (and recombined to quads
        when ``recombine=True``), then marks the volume itself as
        transfinite.

        Parameters
        ----------
        vol :
            Volume identifier.  Accepts an int tag, a label string, a
            physical-group name, or a ``(3, tag)`` dimtag.  If it
            resolves to multiple volumes, all are constrained.
        size : float, optional
            Target element edge length.  Node count per edge is
            ``round(edge_length / size) + 1`` (clamped to a minimum
            of 2 nodes).  Provides isotropic sizing — each edge gets
            its own ``n_nodes`` based on its length.
        n : int, optional
            Uniform node count on **every** edge of the volume.
            Overrides per-edge length-based sizing.
        recombine : bool, default ``True``
            Recombine each face to quads (gives a hex mesh).  Set
            ``False`` for a transfinite tet mesh.

        Returns
        -------
        _Structured
            ``self`` for chaining.

        Raises
        ------
        ValueError
            If neither ``size`` nor ``n`` is given, or both are.

        Requirements
        ------------
        The volume must be **hex-decomposable**: exactly 5 or 6 faces,
        each a 3- or 4-sided patch.  After :meth:`boolean.fragment`
        operations, volumes may end up with split faces and stop being
        transfinite-compatible — in that case use
        :meth:`set_transfinite_automatic` instead, which silently
        skips incompatible faces.

        Examples
        --------
        Uniform mesh — same ``n`` per edge regardless of edge length::

            m.mesh.structured.set_transfinite_box("box", n=11)

        Length-based sizing — denser mesh on longer edges::

            m.mesh.structured.set_transfinite_box("box", size=0.5)

        Per-axis control — use the lower-level methods instead::

            edges = m.model.queries.select("box", dim=1)
            m.mesh.structured.set_transfinite_curve(
                edges.parallel_to("x").tags(), n_nodes=11)
            m.mesh.structured.set_transfinite_curve(
                edges.parallel_to("y").tags(), n_nodes=11)
            m.mesh.structured.set_transfinite_curve(
                edges.parallel_to("z").tags(), n_nodes=21)
            # then surfaces + volume via set_transfinite_box(recombine=...)
        """
        if (size is None) == (n is None):
            raise ValueError("Pass exactly one of size= or n=.")

        # Resolve volume → list of dim=3 tags
        tags = self._resolve(vol, dim=3)

        for vtag in tags:
            edges = self._mesh._parent.model.queries.boundary_curves(vtag)
            faces = self._mesh._parent.model.queries.boundary(vtag, oriented=False)

            for _, ctag in edges:
                if n is not None:
                    n_edge = n
                else:
                    bb = self._mesh._parent.model.queries.bounding_box(ctag, dim=1)
                    L  = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
                    n_edge = max(2, round(L / size) + 1)
                self.set_transfinite_curve(ctag, n_edge)

            for _, stag in faces:
                self.set_transfinite_surface(stag)
                if recombine:
                    self.set_recombine(stag, dim=2)

            self.set_transfinite_volume(vtag)

        self._mesh._log(
            f"set_transfinite_box(vol={vol!r}, size={size}, n={n}, "
            f"recombine={recombine}) — applied to {len(tags)} volume(s)"
        )
        return self

    def set_transfinite_by_physical(
        self,
        name : str,
        *,
        dim  : int,
        **kwargs,
    ) -> "_Structured":
        """
        Deprecated.  ``set_transfinite_curve/surface/volume`` already
        accept a label or physical-group name directly — pass it as
        ``tag``.

        Example
        -------
        ::

            # old
            g.mesh.structured.set_transfinite_by_physical("flange", dim=2,
                                                          arrangement="Left")
            # new
            g.mesh.structured.set_transfinite_surface("flange",
                                                      arrangement="Left")
        """
        import warnings
        warnings.warn(
            "set_transfinite_by_physical is deprecated; "
            "set_transfinite_curve/surface/volume already accept a "
            "physical-group name as tag.",
            DeprecationWarning,
            stacklevel=2,
        )
        if dim == 1:
            return self.set_transfinite_curve(name, **kwargs)
        if dim == 2:
            return self.set_transfinite_surface(name, **kwargs)
        if dim == 3:
            return self.set_transfinite_volume(name, **kwargs)
        raise ValueError(
            f"set_transfinite_by_physical: dim must be 1, 2, or 3, got {dim!r}"
        )

    # ------------------------------------------------------------------
    # Recombination
    # ------------------------------------------------------------------

    def set_recombine(
        self,
        tag,
        *,
        dim  : int   = 2,
        angle: float = 45.0,
    ) -> "_Structured":
        """Request quad recombination. ``tag`` accepts int, label, or PG name."""
        for t in self._resolve(tag, dim=dim):
            gmsh.model.mesh.setRecombine(dim, t, angle)
            self._mesh._directives.append({
                'kind': 'recombine', 'dim': dim, 'tag': t, 'angle': angle,
            })
            self._mesh._log(f"set_recombine(dim={dim}, tag={t}, angle={angle}°)")
        return self

    def recombine(self) -> "_Structured":
        """Globally recombine all triangular elements into quads."""
        gmsh.model.mesh.recombine()
        self._mesh._log("recombine()")
        return self

    def set_recombine_by_physical(
        self,
        name : str,
        *,
        dim  : int = 2,
        angle: float = 45.0,
    ) -> "_Structured":
        """Deprecated.  ``set_recombine`` accepts a PG name directly."""
        import warnings
        warnings.warn(
            "set_recombine_by_physical is deprecated; pass the "
            "physical-group name to set_recombine() as tag.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.set_recombine(name, dim=dim, angle=angle)

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def set_smoothing(self, tag, val: int, *, dim: int = 2) -> "_Structured":
        """Set smoothing passes. ``tag`` accepts int, label, or PG name."""
        for t in self._resolve(tag, dim=dim):
            gmsh.model.mesh.setSmoothing(dim, t, val)
            self._mesh._directives.append({
                'kind': 'smoothing', 'dim': dim, 'tag': t, 'val': val,
            })
            self._mesh._log(f"set_smoothing(dim={dim}, tag={t}, val={val})")
        return self

    def set_smoothing_by_physical(
        self,
        name: str,
        val : int,
        *,
        dim : int = 2,
    ) -> "_Structured":
        """Deprecated.  ``set_smoothing`` accepts a PG name directly."""
        import warnings
        warnings.warn(
            "set_smoothing_by_physical is deprecated; pass the "
            "physical-group name to set_smoothing() as tag.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.set_smoothing(name, val, dim=dim)

    # ------------------------------------------------------------------
    # Compound + constraint removal
    # ------------------------------------------------------------------

    def set_compound(self, dim: int, tags) -> "_Structured":
        """Merge entities so they are meshed together as a single compound.

        ``tags`` accepts int, label/PG name, ``(dim, tag)`` tuple, or a
        list of any mix.
        """
        resolved = self._resolve(tags, dim=dim)
        gmsh.model.mesh.setCompound(dim, resolved)
        self._mesh._log(f"set_compound(dim={dim}, tags={resolved})")
        return self

    def remove_constraints(self, dim_tags=None) -> "_Structured":
        """Remove all meshing constraints from the given (or all) entities.

        ``dim_tags`` accepts any flexible-ref form (int, label/PG name,
        ``(dim, tag)``, or list thereof).  ``None`` clears every
        entity in the model.
        """
        if dim_tags is None:
            dts: list[DimTag] = []
        else:
            from apeGmsh.core._helpers import resolve_to_dimtags
            dts = resolve_to_dimtags(
                dim_tags, default_dim=3, session=self._mesh._parent,
            )
        gmsh.model.mesh.removeConstraints(dimTags=dts)
        self._mesh._log(f"remove_constraints(dim_tags={dim_tags})")
        return self
