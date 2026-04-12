"""
_Editing — mesh mutation, embedding, periodicity, and STL import.

Accessed via ``g.mesh.editing``.  Owns every operation that changes
mesh topology or embeds lower-dim entities, plus the STL → discrete
→ geometry pipeline.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from .Mesh import Mesh


DimTag = tuple[int, int]


class _Editing:
    """Mesh mutation, embedding, periodicity, STL import."""

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        tags,
        in_tag,
        *,
        dim   : int = 0,
        in_dim: int = 3,
    ) -> "_Editing":
        """
        Embed lower-dimensional entities inside a higher-dimensional
        entity so the mesh is conforming along them.

        Parameters accept int tags, label/PG strings, or lists thereof.

        Example
        -------
        ::

            g.mesh.editing.embed("crack_surf", "body", dim=2, in_dim=3)
            g.mesh.editing.embed([p1, p2, p3], surf_tag, dim=0, in_dim=2)
        """
        from apeGmsh.core._helpers import resolve_to_tags
        tag_list = resolve_to_tags(tags, dim=dim, session=self._mesh._parent)
        in_tags = resolve_to_tags(in_tag, dim=in_dim, session=self._mesh._parent)
        in_tag_resolved = in_tags[0]
        gmsh.model.mesh.embed(dim, tag_list, in_dim, in_tag_resolved)
        self._mesh._log(
            f"embed(dim={dim}, tags={tag_list}, "
            f"in_dim={in_dim}, in_tag={in_tag})"
        )
        return self

    # ------------------------------------------------------------------
    # Periodicity
    # ------------------------------------------------------------------

    def set_periodic(
        self,
        tags       : list[int],
        master_tags: list[int],
        transform  : list[float],
        *,
        dim        : int = 2,
    ) -> "_Editing":
        """
        Declare periodic mesh correspondence between entities.

        Parameters
        ----------
        tags        : slave entity tags
        master_tags : master entity tags
        transform   : 16-element row-major 4×4 affine matrix mapping
                      master → slave coordinates
        dim         : entity dimension (1 = curves, 2 = surfaces)
        """
        gmsh.model.mesh.setPeriodic(dim, tags, master_tags, transform)
        self._mesh._log(
            f"set_periodic(dim={dim}, tags={tags}, master={master_tags})"
        )
        return self

    # ------------------------------------------------------------------
    # STL / discrete geometry
    # ------------------------------------------------------------------

    def import_stl(self) -> "_Editing":
        """
        Classify an STL mesh previously loaded into the gmsh model via
        ``gmsh.merge`` as a discrete surface mesh.
        """
        gmsh.model.mesh.importStl()
        self._mesh._log("import_stl()")
        return self

    def classify_surfaces(
        self,
        angle              : float,
        *,
        boundary           : bool  = True,
        for_reparametrization: bool = False,
        curve_angle        : float = math.pi,
        export_discrete    : bool  = True,
    ) -> "_Editing":
        """
        Partition a discrete STL mesh into surface patches based on
        dihedral angle.
        """
        gmsh.model.mesh.classifySurfaces(
            angle,
            boundary=boundary,
            forReparametrization=for_reparametrization,
            curveAngle=curve_angle,
            exportDiscrete=export_discrete,
        )
        self._mesh._log(
            f"classify_surfaces(angle={math.degrees(angle):.1f}°, "
            f"boundary={boundary})"
        )
        return self

    def create_geometry(
        self,
        dim_tags: list[DimTag] | None = None,
    ) -> "_Editing":
        """
        Create a proper CAD-like geometry from classified discrete surfaces.
        Must be called after ``classify_surfaces``.
        """
        gmsh.model.mesh.createGeometry(dimTags=dim_tags or [])
        self._mesh._log("create_geometry()")
        return self

    # ------------------------------------------------------------------
    # Mesh editing
    # ------------------------------------------------------------------

    def clear(self, dim_tags: list[DimTag] | None = None) -> "_Editing":
        """Clear mesh data (nodes + elements)."""
        gmsh.model.mesh.clear(dimTags=dim_tags or [])
        self._mesh._log(f"clear(dim_tags={dim_tags})")
        return self

    def reverse(self, dim_tags: list[DimTag] | None = None) -> "_Editing":
        """Reverse the orientation of mesh elements in the given entities."""
        gmsh.model.mesh.reverse(dimTags=dim_tags or [])
        self._mesh._log("reverse()")
        return self

    def relocate_nodes(self, *, dim: int = -1, tag: int = -1) -> "_Editing":
        """Project mesh nodes back onto their underlying geometry."""
        gmsh.model.mesh.relocateNodes(dim=dim, tag=tag)
        self._mesh._log(f"relocate_nodes(dim={dim}, tag={tag})")
        return self

    def remove_duplicate_nodes(self, verbose: bool = True) -> "_Editing":
        """
        Merge nodes that share the same position within tolerance.

        Parameters
        ----------
        verbose : if True (default), print how many nodes were merged.
        """
        before = len(gmsh.model.mesh.getNodes()[0])
        gmsh.model.mesh.removeDuplicateNodes()
        after  = len(gmsh.model.mesh.getNodes()[0])
        removed = before - after
        if verbose:
            if removed > 0:
                print(f"remove_duplicate_nodes: merged {removed} "
                      f"node(s) ({before} → {after})")
            else:
                print(f"remove_duplicate_nodes: no duplicates found "
                      f"({before} nodes unchanged)")
        self._mesh._log(f"remove_duplicate_nodes() removed={removed}")
        return self

    def remove_duplicate_elements(self, verbose: bool = True) -> "_Editing":
        """Remove elements with identical node connectivity."""
        def _count() -> int:
            _, tags, _ = gmsh.model.mesh.getElements()
            return sum(len(t) for t in tags)

        before = _count()
        gmsh.model.mesh.removeDuplicateElements()
        after  = _count()
        removed = before - after
        if verbose:
            if removed > 0:
                print(f"remove_duplicate_elements: removed {removed} "
                      f"element(s) ({before} → {after})")
            else:
                print(f"remove_duplicate_elements: no duplicates found "
                      f"({before} elements unchanged)")
        self._mesh._log(f"remove_duplicate_elements() removed={removed}")
        return self

    def affine_transform(
        self,
        matrix  : list[float],
        dim_tags: list[DimTag] | None = None,
    ) -> "_Editing":
        """
        Apply an affine transformation to mesh nodes (12 coefficients,
        row-major 4×3 matrix — translation in last column).
        """
        gmsh.model.mesh.affineTransform(matrix, dimTags=dim_tags or [])
        self._mesh._log("affine_transform()")
        return self
