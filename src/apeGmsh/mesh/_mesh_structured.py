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
        """
        Set a transfinite constraint on a curve.

        ``tag`` accepts an int, a label string, or a PG name.
        If it resolves to multiple curves, the constraint is
        applied to each.
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
        """
        Set a transfinite constraint on a surface.

        ``tag`` accepts an int, a label string, or a PG name.
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
        """Set a transfinite constraint on a volume.

        ``tag`` accepts an int, a label string, or a PG name.
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
        """
        Let gmsh automatically detect and set transfinite constraints
        on compatible 3- and 4-sided surfaces/volumes.
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
        """
        Apply transfinite + recombine constraints to a clean 6-face hex
        volume — captures the full unstructured-to-hex setup in one call.

        Walks the volume's bounding curves and assigns a node count
        per edge, marks every bounding surface as transfinite (and
        recombined to quads if ``recombine=True``), then marks the
        volume as transfinite.

        Parameters
        ----------
        vol :
            int tag, label string, PG name, or ``(3, tag)`` tuple.
        size :
            Target element size — node count per edge is
            ``round(length / size) + 1``.
        n :
            Uniform node count on every edge (overrides ``size``).
        recombine :
            Recombine each face to quads.  Default True (gives a hex mesh).
            Set False for a transfinite tet mesh.

        Notes
        -----
        Requires the volume to be hex-decomposable — exactly 6 faces,
        each a 4-sided patch.  After ``fragment()`` operations, volumes
        may end up with split faces and stop being transfinite-compatible;
        in that case use ``set_transfinite_automatic()`` instead.

        Example
        -------
        ::

            m.mesh.structured.set_transfinite_box('box', size=0.5)
            m.mesh.structured.set_transfinite_box(vol_tag, n=11)
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
