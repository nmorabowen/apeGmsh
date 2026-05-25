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
import numpy as np

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

    def remove_duplicate_nodes(self) -> "_Editing":
        """
        Merge nodes that share the same position within tolerance.

        Node removal is **always** announced on stdout — there is no
        silent mode. Deleting nodes from a meshed model is a
        destructive operation; the visibility floor is intentional so
        that an unexpected dedup never hides in a long pipeline log.
        """
        before = len(gmsh.model.mesh.getNodes()[0])
        gmsh.model.mesh.removeDuplicateNodes()
        after  = len(gmsh.model.mesh.getNodes()[0])
        removed = before - after
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

    def crack(
        self,
        physical_group: str,
        *,
        dim          : int                              = 1,
        open_boundary: str | None                       = None,
        normal       : tuple[float, float, float] | None = None,
        side_labels  : tuple[str, str] | bool           = True,
    ) -> "_Editing":
        """
        Duplicate mesh nodes along a physical group to create a crack.

        Wraps Gmsh's built-in ``Crack`` plugin
        (``gmsh.plugin.run("Crack")``).  After meshing, the plugin
        walks the elements on one side of ``physical_group`` and
        reconnects them to a freshly duplicated set of nodes — the
        crack is therefore a discontinuity in the mesh, not in the
        geometry.

        By default the plugin keeps the **boundary vertices of the
        crack curve shared** (e.g. the crack tip in fracture
        mechanics).  Naming a sub-region of those boundary vertices
        in ``open_boundary`` overrides the default and **duplicates
        them too** — that is how you model a crack mouth that opens
        onto a free surface.

        Must be called **after** ``g.mesh.generation.generate(...)``
        — the plugin operates on the mesh, not the geometry.

        Parameters
        ----------
        physical_group : str
            Name of the physical group containing the crack
            curves (``dim=1``) or surfaces (``dim=2``).  Create it
            ahead of time via ``g.physical.add_curve(..., name=...)``
            or ``g.physical.add_surface(..., name=...)``.
        dim : int
            Dimension of the crack itself.  ``1`` for a 1-D crack
            in a 2-D mesh; ``2`` for a 2-D crack in a 3-D mesh.
        open_boundary : str, optional
            Name of a physical group, **one dimension lower than**
            ``dim``, naming the crack-curve boundary vertices that
            should *also* be duplicated.  Use this for the crack
            mouth (where the crack reaches a free surface).  Leave
            ``None`` for an interior crack so every boundary vertex
            (including the tip) stays shared.
        normal : (nx, ny, nz), optional
            Hint vector forwarded to the plugin to disambiguate the
            two sides of the crack when topology alone is not enough.
            Almost never needed for clean transfinite or unstructured
            meshes.
        side_labels : tuple[str, str] or bool, default True
            Post-plugin, the crack is owned by **two** distinct face
            entities (the original entity plus a new one created by the
            plugin to host the duplicated side).  This argument
            controls whether they get named physical groups attached:

              * ``True`` (default) — auto-derive
                ``f"{physical_group}_normal"`` and
                ``f"{physical_group}_inverted"`` and add them as
                physical groups, one per face entity.
              * ``(normal_name, inverted_name)`` tuple — use these
                explicit names instead.
              * ``False`` — skip side labeling (legacy behaviour).

            **Convention.**  ``<pg>_normal`` is the face entity whose
            adjacent volume elements sit on the side the *original*
            surface normal points toward; ``<pg>_inverted`` is the
            face on the opposite side.  This is computed at runtime
            from the signed distance between an adjacent tet's
            centroid and the crack plane along that normal — it does
            not assume the plugin's "original vs new" mapping is
            stable.  Only supported for ``dim=2`` cracks in 3D meshes.

        Returns
        -------
        _Editing  (self, for chaining)

        Example
        -------
        Edge crack reaching the bottom edge — duplicate the mouth,
        keep the tip shared::

            g.physical.add_curve([crack_curve], name="Crack")
            g.physical.add_point([base_point],   name="CrackBase")
            g.mesh.generation.generate(dim=2)
            g.mesh.editing.crack(
                "Crack", dim=1, open_boundary="CrackBase",
            )
        """
        physical = getattr(self._mesh._parent, 'physical', None)
        if physical is None:
            raise RuntimeError(
                "crack: session has no 'physical' composite — "
                "physical groups are required to invoke the Crack plugin."
            )

        crack_pg_tag = physical.get_tag(dim, physical_group)
        if crack_pg_tag is None:
            raise KeyError(
                f"crack: no physical group named {physical_group!r} at "
                f"dim={dim}.  Create it first via "
                f"g.physical.add_curve(..., name={physical_group!r}) "
                f"(or add_surface for dim=2)."
            )

        open_pg_tag = 0  # plugin default — no open boundary
        if open_boundary is not None:
            open_dim = dim - 1
            open_pg_tag = physical.get_tag(open_dim, open_boundary)
            if open_pg_tag is None:
                raise KeyError(
                    f"crack: no physical group named {open_boundary!r} "
                    f"at dim={open_dim}.  The open boundary lives one "
                    f"dimension lower than the crack itself."
                )

        # Snapshot pre-plugin state for side-labeling.  We capture
        # the source surface's analytic OCC normal *before* the plugin
        # runs because the plugin produces a discrete (mesh-only)
        # surface for the duplicated side that has no parameterisation
        # — only the original entity is OCC-backed and queryable via
        # gmsh.model.getNormal.
        do_side_labels = (
            side_labels is not False and dim == 2
        )
        pre_ents: set[int] = set()
        src_ents: list[int] = []
        ref_origin: np.ndarray | None = None
        ref_normal: np.ndarray | None = None
        if do_side_labels:
            pre_ents = {
                int(t) for d_, t in gmsh.model.getEntities(dim) if d_ == dim
            }
            src_ents = sorted(
                int(e) for e in
                gmsh.model.getEntitiesForPhysicalGroup(dim, crack_pg_tag)
            )
            if len(src_ents) < 1:
                raise RuntimeError(
                    f"crack: source PG {physical_group!r} resolves to "
                    f"no entities; side_labels= cannot be applied."
                )
            src_tag = src_ents[0]
            nrm = gmsh.model.getNormal(src_tag, [0.5, 0.5])
            n_arr = np.asarray(nrm, dtype=float)
            if float(np.linalg.norm(n_arr)) < 1e-12:
                raise RuntimeError(
                    f"crack: source surface {src_tag} returned a "
                    f"degenerate analytic normal — pass "
                    f"side_labels=False or a normal= hint."
                )
            ref_origin = np.asarray(
                gmsh.model.occ.getCenterOfMass(2, src_tag), dtype=float,
            )
            ref_normal = n_arr / float(np.linalg.norm(n_arr))

        gmsh.plugin.setNumber("Crack", "Dimension", float(dim))
        gmsh.plugin.setNumber("Crack", "PhysicalGroup", float(crack_pg_tag))
        gmsh.plugin.setNumber(
            "Crack", "OpenBoundaryPhysicalGroup", float(open_pg_tag),
        )
        if normal is not None:
            nx, ny, nz = normal
            gmsh.plugin.setNumber("Crack", "NormalX", float(nx))
            gmsh.plugin.setNumber("Crack", "NormalY", float(ny))
            gmsh.plugin.setNumber("Crack", "NormalZ", float(nz))

        gmsh.plugin.run("Crack")

        if do_side_labels:
            post_ents = {
                int(t) for d_, t in gmsh.model.getEntities(dim) if d_ == dim
            }
            new_ents = sorted(post_ents - pre_ents)
            if len(new_ents) != 1:
                raise RuntimeError(
                    f"crack: expected exactly 1 new face entity, got "
                    f"{len(new_ents)} (src_ents={src_ents}).  "
                    f"side_labels= cannot be applied unambiguously; pass "
                    f"side_labels=False to skip auto-labeling."
                )

            normal_name, inverted_name = (
                (f"{physical_group}_normal",
                 f"{physical_group}_inverted")
                if side_labels is True
                else side_labels
            )

            new_tag  = new_ents[0]
            orig_tag = src_ents[0]
            # The plugin reconnects exactly one side; the original and
            # new entities are guaranteed to lie on opposite sides, so
            # we only need to probe one of them.  ref_origin/ref_normal
            # are the source surface's analytic plane.
            assert ref_origin is not None and ref_normal is not None
            new_side = self._classify_face_side(
                new_tag, ref_origin, ref_normal,
            )
            normal_tag   = new_tag  if new_side > 0 else orig_tag
            inverted_tag = orig_tag if new_side > 0 else new_tag

            physical.add_surface([normal_tag],   name=normal_name)
            physical.add_surface([inverted_tag], name=inverted_name)

            self._mesh._log(
                f"crack: side labels {normal_name!r}->entity {normal_tag}, "
                f"{inverted_name!r}->entity {inverted_tag}"
            )

        self._mesh._log(
            f"crack(physical_group={physical_group!r}, dim={dim}, "
            f"open_boundary={open_boundary!r}) "
            f"-> crack_pg_tag={crack_pg_tag}, "
            f"open_pg_tag={open_pg_tag}"
        )
        return self

    @staticmethod
    def _classify_face_side(
        face_tag : int,
        origin   : np.ndarray,
        normal   : np.ndarray,
    ) -> int:
        """
        Classify which side of an external reference plane (defined by
        ``origin`` and ``normal``) a face entity's adjacent volume tets
        sit on.

        Returns +1 if the adjacent-tet centroid lies on the +normal
        half-space, -1 otherwise.
        """
        etypes, _, enodes = gmsh.model.mesh.getElements(2, face_tag)
        face_node_set: set[int] = set()
        for etype, conn in zip(etypes, enodes):
            if int(etype) != 2:           # gmsh elem type 2 = 3-node tri
                continue
            face_node_set.update(int(n) for n in conn)
        if not face_node_set:
            raise RuntimeError(
                f"crack: face entity (dim=2, tag={face_tag}) has no "
                f"3-node triangles; cannot classify side."
            )

        for vd, vt in gmsh.model.getEntities(3):
            etypes3, _, enodes3 = gmsh.model.mesh.getElements(vd, vt)
            for etype3, conn3 in zip(etypes3, enodes3):
                if int(etype3) != 4:      # gmsh elem type 4 = 4-node tet
                    continue
                tets = np.asarray(conn3, dtype=int).reshape(-1, 4)
                for tet in tets:
                    if sum(1 for n in tet if int(n) in face_node_set) == 3:
                        pts = np.array([
                            gmsh.model.mesh.getNode(int(nid))[0]
                            for nid in tet
                        ])
                        centroid = pts.mean(axis=0)
                        signed = float(np.dot(centroid - origin, normal))
                        return 1 if signed > 0 else -1

        raise RuntimeError(
            f"crack: no adjacent volume tet found for face entity "
            f"(dim=2, tag={face_tag})."
        )

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
