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


DimTag = tuple[int, int]


class _Structured:
    """Transfinite constraints, recombination, smoothing, compounds."""

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # Transfinite constraints
    # ------------------------------------------------------------------

    def _resolve(self, tag, dim: int) -> list[int]:
        """Resolve a tag/label/PG ref to concrete tags."""
        from apeGmsh.core._helpers import resolve_to_tags
        if isinstance(tag, str):
            return resolve_to_tags(tag, dim=dim, session=self._mesh._parent)
        return [int(tag)]

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

    def set_transfinite_by_physical(
        self,
        name : str,
        *,
        dim  : int,
        **kwargs,
    ) -> "_Structured":
        """
        Apply the appropriate ``set_transfinite_*`` call to every
        entity in a physical group.
        """
        tags = self._mesh._resolve_physical(name, dim)
        self._mesh._log(
            f"set_transfinite_by_physical(name={name!r}, dim={dim}, tags={tags})"
        )
        if dim == 1:
            for t in tags:
                self.set_transfinite_curve(t, **kwargs)
        elif dim == 2:
            for t in tags:
                self.set_transfinite_surface(t, **kwargs)
        elif dim == 3:
            for t in tags:
                self.set_transfinite_volume(t, **kwargs)
        else:
            raise ValueError(
                f"set_transfinite_by_physical: dim must be 1, 2, or 3, got {dim!r}"
            )
        return self

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
        """Apply :meth:`set_recombine` to every entity in a physical group."""
        tags = self._mesh._resolve_physical(name, dim)
        self._mesh._log(
            f"set_recombine_by_physical(name={name!r}, dim={dim}, tags={tags})"
        )
        for t in tags:
            self.set_recombine(t, dim=dim, angle=angle)
        return self

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
        """Apply :meth:`set_smoothing` to every entity in a physical group."""
        tags = self._mesh._resolve_physical(name, dim)
        self._mesh._log(
            f"set_smoothing_by_physical(name={name!r}, dim={dim}, "
            f"tags={tags}, val={val})"
        )
        for t in tags:
            self.set_smoothing(t, val, dim=dim)
        return self

    # ------------------------------------------------------------------
    # Compound + constraint removal
    # ------------------------------------------------------------------

    def set_compound(self, dim: int, tags: list[int]) -> "_Structured":
        """Merge entities so they are meshed together as a single compound."""
        gmsh.model.mesh.setCompound(dim, tags)
        self._mesh._log(f"set_compound(dim={dim}, tags={tags})")
        return self

    def remove_constraints(
        self,
        dim_tags: list[DimTag] | None = None,
    ) -> "_Structured":
        """Remove all meshing constraints from the given (or all) entities."""
        gmsh.model.mesh.removeConstraints(dimTags=dim_tags or [])
        self._mesh._log("remove_constraints()")
        return self
