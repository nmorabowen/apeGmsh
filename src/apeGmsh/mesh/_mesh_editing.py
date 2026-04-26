"""
_Editing — mesh mutation, embedding, periodicity, and STL import.

Accessed via ``g.mesh.editing``.  Owns every operation that changes
mesh topology or embeds lower-dim entities, plus the STL -> discrete
-> geometry pipeline.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from .Mesh import Mesh


from apeGmsh._types import DimTag


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
        tags,
        master_tags,
        transform  : list[float],
        *,
        dim        : int = 2,
    ) -> "_Editing":
        """
        Declare periodic mesh correspondence between entities.

        Parameters
        ----------
        tags        : slave entity reference(s) — int, label, PG name,
                      ``(dim, tag)`` tuple, or list of any mix.
        master_tags : master entity reference(s) — same flexible form.
        transform   : 16-element row-major 4×4 affine matrix mapping
                      master -> slave coordinates
        dim         : entity dimension (1 = curves, 2 = surfaces)
        """
        from apeGmsh.core._helpers import resolve_to_tags
        slave_resolved = resolve_to_tags(
            tags, dim=dim, session=self._mesh._parent,
        )
        master_resolved = resolve_to_tags(
            master_tags, dim=dim, session=self._mesh._parent,
        )
        if len(slave_resolved) != len(master_resolved):
            raise ValueError(
                f"set_periodic: slave/master count mismatch — "
                f"slaves={slave_resolved} ({len(slave_resolved)}), "
                f"masters={master_resolved} ({len(master_resolved)}). "
                f"Each slave needs exactly one master under the same "
                f"transform."
            )
        gmsh.model.mesh.setPeriodic(
            dim, slave_resolved, master_resolved, transform,
        )
        self._mesh._log(
            f"set_periodic(dim={dim}, tags={slave_resolved}, "
            f"master={master_resolved})"
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

    def clear(self, dim_tags=None) -> "_Editing":
        """Clear mesh data (nodes + elements).

        ``dim_tags`` accepts any flexible-ref form — int, label/PG name,
        ``(dim, tag)``, or a list mixing those — resolved via
        :func:`resolve_to_dimtags` (default_dim=3).  ``None`` (the
        default) clears every entity in the model.

        Example
        -------
        ::

            g.mesh.editing.clear()                  # clear everything
            g.mesh.editing.clear("col.body")        # clear a labelled volume
            g.mesh.editing.clear([(2, 5), "fillet"]) # mixed refs
        """
        if dim_tags is None:
            dts: list[DimTag] = []
        else:
            from apeGmsh.core._helpers import resolve_to_dimtags
            dts = resolve_to_dimtags(
                dim_tags, default_dim=3, session=self._mesh._parent,
            )
        gmsh.model.mesh.clear(dimTags=dts)
        self._mesh._log(f"clear(dim_tags={dim_tags})")
        return self

    def reverse(self, dim_tags=None) -> "_Editing":
        """Reverse the orientation of mesh elements in the given entities.

        ``dim_tags`` accepts any flexible-ref form (int, label/PG name,
        ``(dim, tag)``, or list thereof).  ``None`` reverses every
        entity in the model.

        Example
        -------
        ::

            g.mesh.editing.reverse("inverted_face")
            g.mesh.editing.reverse([(2, 5), (2, 6)])
        """
        if dim_tags is None:
            dts: list[DimTag] = []
        else:
            from apeGmsh.core._helpers import resolve_to_dimtags
            dts = resolve_to_dimtags(
                dim_tags, default_dim=3, session=self._mesh._parent,
            )
        gmsh.model.mesh.reverse(dimTags=dts)
        self._mesh._log(f"reverse(dim_tags={dim_tags})")
        return self

    def relocate_nodes(self, *, dim: int = -1, tag=-1) -> "_Editing":
        """Project mesh nodes back onto their underlying geometry.

        ``tag`` accepts an int, label/PG name, ``(dim, tag)``, or a
        list mixing those.  Because gmsh's ``relocateNodes`` operates
        on a single entity at a time, when a reference resolves to
        multiple entities the wrapper iterates and calls gmsh once
        per resolved ``(dim, tag)``.

        ``tag=-1`` (the default) relocates nodes for every entity in
        the model; ``dim`` is forwarded to gmsh in that case.

        Example
        -------
        ::

            g.mesh.editing.relocate_nodes()                 # all entities
            g.mesh.editing.relocate_nodes(tag="col.faces")  # whole label
            g.mesh.editing.relocate_nodes(tag=(2, 5))
        """
        if tag == -1:
            gmsh.model.mesh.relocateNodes(dim=dim, tag=-1)
            self._mesh._log(f"relocate_nodes(dim={dim}, tag=-1)")
            return self

        from apeGmsh.core._helpers import resolve_to_dimtags
        default_dim = dim if dim != -1 else 3
        dts = resolve_to_dimtags(
            tag, default_dim=default_dim, session=self._mesh._parent,
        )
        for d, t in dts:
            gmsh.model.mesh.relocateNodes(dim=d, tag=t)
        self._mesh._log(f"relocate_nodes(resolved={dts})")
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
                      f"node(s) ({before} -> {after})")
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
                      f"element(s) ({before} -> {after})")
            else:
                print(f"remove_duplicate_elements: no duplicates found "
                      f"({before} elements unchanged)")
        self._mesh._log(f"remove_duplicate_elements() removed={removed}")
        return self

    def affine_transform(
        self,
        matrix  : list[float],
        dim_tags=None,
    ) -> "_Editing":
        """
        Apply an affine transformation to mesh nodes (12 coefficients,
        row-major 4x3 matrix — translation in last column).

        ``dim_tags`` accepts any flexible-ref form (int, label/PG name,
        ``(dim, tag)``, or list thereof).  ``None`` transforms every
        entity in the model.

        Example
        -------
        ::

            identity = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0]
            g.mesh.editing.affine_transform(identity, "col.body")
        """
        if dim_tags is None:
            dts: list[DimTag] = []
        else:
            from apeGmsh.core._helpers import resolve_to_dimtags
            dts = resolve_to_dimtags(
                dim_tags, default_dim=3, session=self._mesh._parent,
            )
        gmsh.model.mesh.affineTransform(matrix, dimTags=dts)
        self._mesh._log(f"affine_transform(dim_tags={dim_tags})")
        return self
