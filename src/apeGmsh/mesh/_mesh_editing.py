"""
_Editing — mesh mutation, embedding, periodicity, and STL import.

Accessed via ``g.mesh.editing``.  Owns every operation that changes
mesh topology or embeds lower-dim entities, plus the STL -> discrete
-> geometry pipeline.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Literal

import gmsh
import numpy as np

if TYPE_CHECKING:
    from .Mesh import Mesh


from apeGmsh._types import DimTag


#: Gmsh element-type codes for line elements (curve dim=1).
#: Line4 (type 26, cubic edges from order-3 meshes) is deferred per
#: ADR 0037 §Future work; the dim != 1 / Line4 guard in
#: :meth:`_Editing.split_higher_order_lines` will route it once the
#: generalisation lands.
_GMSH_LINE2 = 1   # 2-node line
_GMSH_LINE3 = 8   # 3-node line (second-order)


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

    def split_higher_order_lines(
        self,
        physical_group: str | Iterable[str],
        *,
        policy: Literal["forbid", "split", "constrain"],
        dim: int = 1,
    ) -> "_Editing":
        """
        Demote higher-order line elements (Line3) on the named PG(s) to
        the 1st-order Line2 elements that OpenSees beam-columns require.

        2nd-order continuum meshes (quadratic shells, solid tet10s, etc.)
        propagate their order to every line entity in the same Gmsh
        model — even line entities the user intended as frame elements
        get meshed as Line3 (Gmsh type 8, three nodes ``(i, j, mid)``).
        OpenSees beam-columns are strictly 2-node 1st-order, so the
        bridge raises at ``_check_two_nodes`` whenever it sees a Line3
        in a frame PG.  This method is the broker-side resolution:
        rewrite the mesh so the bridge sees only Line2 on the named
        PG(s), keeping the continuum side untouched.

        Three policies pick what happens to the mid-side node:

        * ``"split"`` — each Line3 ``(i, j, mid)`` is replaced by two
          Line2 elements ``(i, mid)`` and ``(mid, j)`` on the SAME dim=1
          entity (PG membership tracks at entity level — no rebinding).
          The mid-node becomes a real FE node with its own DOFs.
          Exact subdivision for prismatic elastic frames; for distributed-
          plasticity frames the integration-point count doubles (each
          sub-element gets its own N-IP rule), which can shift hinge
          locations under softening / cyclic degradation.  Concentrated-
          plasticity integration rules (``HingeRadau``, ``HingeRadauTwo``,
          ``HingeMidpoint``, ``HingeEndpoint``) are structurally
          incompatible with this policy — splitting puts the calibrated
          end-region hinges in the wrong places.  Use ``"forbid"`` if
          you must lock in 1st-order lines, or fall back to "this PG
          stays 1st-order in the mesh"; ``"constrain"`` is reserved
          but not implemented this round.
        * ``"constrain"`` — RESERVED but NOT IMPLEMENTED this round.
          The kinematically clean answer (mid-node interpolated linearly
          from i and j via a 2-master/1-slave constraint) requires an
          OpenSees primitive that does not exist today:
          ``ASDEmbeddedNodeElement`` accepts exactly 3 or 4 retained
          nodes (ADR 0036), so a Line2 master pair (2 nodes) cannot be
          expressed.  A future round paired with upstream OpenSees work
          (new element class accepting N-retained-node coupling, or a
          new MP_Constraint primitive) will land this policy.  Raises
          ``NotImplementedError`` until then.
        * ``"forbid"`` — fail-loud if the named PG(s) contain any Line3
          element.  Use as a build-time invariant lock when you must
          guarantee a PG remained 1st-order through meshing.

        Call timing: AFTER ``g.mesh.generation.generate(...)`` (the
        method operates on the live mesh), BEFORE
        ``g.mesh.queries.get_fem_data(...)`` (the snapshot must see the
        rewritten topology).  Never call inside a stage block — the
        mesh edit is global and must complete before the bridge builds.

        Parameters
        ----------
        physical_group
            PG name (``str``) or iterable of PG names (``list``/
            ``tuple``).  Required — there is no "split everything"
            mode, callers must name what they're rewriting.  Each name
            must already exist as a physical group of ``dim``; unknown
            names raise ``KeyError`` naming the missing one.
        policy
            ``"forbid"`` / ``"split"`` / ``"constrain"`` — see above.
            Required kwarg; no default in the spirit of fail-loud:
            destructive mesh mutation should never happen by accident.
        dim
            Dimension of the higher-order lines.  Currently only ``1``
            is supported; ``dim != 1`` raises ``NotImplementedError``.
            Future work (line4, cubic edges from order-3 meshes) lands
            via the same kwarg.

        Returns
        -------
        _Editing  (self, for chaining)

        Raises
        ------
        NotImplementedError
            ``policy="constrain"`` (deferred to a follow-up paired with
            upstream OpenSees work — ADR 0036 future track).
            ``dim != 1`` (line4 / cubic lines are out of scope this
            round; structure permits adding them with the same kwarg).
        RuntimeError
            ``policy="forbid"`` and any named PG contains at least one
            Line3 element.  Message names the offending PG and count.
        KeyError
            A named PG does not exist at ``dim``.
        ValueError
            ``policy`` is not one of the three accepted values.

        Example
        -------
        ::

            # Quadratic shell + frame model.  The 2nd-order shell mesh
            # forces every line entity to Line3; the frame PG needs
            # to come back to Line2 before the bridge sees it.
            g.mesh.generation.generate(dim=3)
            g.mesh.editing.split_higher_order_lines(
                "Beams", policy="split",
            )
            fem = g.mesh.queries.get_fem_data(dim=3)
            ops = apeSees(fem)
            ops.element.elasticBeamColumn(pg="Beams", ...)  # works now
        """
        if policy not in ("forbid", "split", "constrain"):
            raise ValueError(
                f"split_higher_order_lines: policy must be 'forbid', "
                f"'split', or 'constrain'; got {policy!r}."
            )
        if dim != 1:
            raise NotImplementedError(
                f"split_higher_order_lines: dim={dim} is not supported "
                "this round (only dim=1 / Line3 is implemented).  Cubic "
                "lines (Line4, Gmsh type 26) from order-3 meshes are "
                "deferred to a follow-up that generalizes this verb to "
                "higher-order lines."
            )
        if policy == "constrain":
            raise NotImplementedError(
                "split_higher_order_lines: policy='constrain' is "
                "reserved but not implemented this round.  The "
                "kinematically clean linear-interp mid-node constraint "
                "requires an OpenSees primitive that does not exist "
                "today (ASDEmbeddedNodeElement needs 3-4 retained "
                "nodes per ADR 0036; a Line2 master pair has 2).  Use "
                "policy='split' or policy='forbid' this round, or "
                "track the follow-up that pairs this work with "
                "upstream OpenSees changes."
            )

        physical = getattr(self._mesh._parent, "physical", None)
        if physical is None:
            raise RuntimeError(
                "split_higher_order_lines: session has no 'physical' "
                "composite — physical groups are required to name what "
                "to rewrite."
            )

        if isinstance(physical_group, str):
            pg_names: tuple[str, ...] = (physical_group,)
        else:
            pg_names = tuple(physical_group)
        if not pg_names:
            raise ValueError(
                "split_higher_order_lines: physical_group must be a "
                "non-empty PG name or iterable of names."
            )

        # Resolve every PG to its entity list before doing any
        # mutation, so we fail loud on unknown names before touching
        # the mesh.
        pg_entities: list[tuple[str, list[int]]] = []
        for name in pg_names:
            pg_tag = physical.get_tag(dim, name)
            if pg_tag is None:
                raise KeyError(
                    f"split_higher_order_lines: no physical group named "
                    f"{name!r} at dim={dim}.  Create it first via "
                    f"g.physical.add_curve(..., name={name!r})."
                )
            ents = [
                int(e)
                for e in gmsh.model.getEntitiesForPhysicalGroup(
                    dim, pg_tag,
                )
            ]
            pg_entities.append((name, ents))

        # For policy="forbid", scan every entity for Line3 elements
        # and raise on first hit.  For "split", do the surgery
        # entity-by-entity.
        total_split = 0
        for name, ents in pg_entities:
            n_line3_in_pg = 0
            for ent in ents:
                line3_tags, line3_conn = self._gather_line3(ent)
                if not line3_tags:
                    continue
                n_line3_in_pg += len(line3_tags)
                if policy == "forbid":
                    continue   # accumulate then raise at end of PG
                # policy == "split"
                self._replace_line3_with_line2_pair(
                    ent, line3_tags, line3_conn,
                )
                total_split += len(line3_tags)
            if policy == "forbid" and n_line3_in_pg > 0:
                raise RuntimeError(
                    f"split_higher_order_lines: PG {name!r} (dim={dim}) "
                    f"contains {n_line3_in_pg} Line3 element(s); "
                    f"policy='forbid' refuses to rewrite the mesh.  "
                    "Re-mesh the source curve(s) at order 1, or call "
                    "this method with policy='split' to demote them."
                )

        if policy == "split":
            if total_split > 0:
                print(
                    f"split_higher_order_lines: demoted {total_split} "
                    f"Line3 -> 2x Line2 across "
                    f"{len(pg_names)} PG(s)"
                )
            else:
                print(
                    f"split_higher_order_lines: no Line3 elements found "
                    f"on PG(s) {list(pg_names)} — no-op"
                )
        self._mesh._log(
            f"split_higher_order_lines(pg={list(pg_names)}, "
            f"policy={policy!r}, dim={dim}) -> split={total_split}"
        )
        return self

    @staticmethod
    def _gather_line3(
        entity_tag: int,
    ) -> tuple[list[int], list[tuple[int, int, int]]]:
        """Walk ``entity_tag`` (dim=1) and return ``(line3_tags,
        line3_conn)`` for every Line3 (Gmsh type 8) element on it.

        Returns empty lists when the entity has no Line3 elements (all
        Line2, or no mesh).
        """
        types, tags, nodes = gmsh.model.mesh.getElements(
            dim=1, tag=entity_tag,
        )
        out_tags: list[int] = []
        out_conn: list[tuple[int, int, int]] = []
        for etype, ttags, tnodes in zip(types, tags, nodes):
            if int(etype) != _GMSH_LINE3:
                continue
            # tnodes is a flat array of length 3 * len(ttags) — three
            # node tags per Line3 in the order (i, j, mid).
            flat = [int(n) for n in tnodes]
            for k, line_tag in enumerate(ttags):
                base = 3 * k
                out_tags.append(int(line_tag))
                out_conn.append((flat[base], flat[base + 1], flat[base + 2]))
        return out_tags, out_conn

    @staticmethod
    def _replace_line3_with_line2_pair(
        entity_tag: int,
        line3_tags: list[int],
        line3_conn: list[tuple[int, int, int]],
    ) -> None:
        """Surgically rewrite ``line3_tags`` on ``entity_tag`` into
        2× Line2 pairs ``(i, mid)`` + ``(mid, j)``.

        The mid-side node is preserved as a real FE node (it already
        existed as the side-node of each parent Line3).  PG membership
        tracks at the entity level, so the new Line2s automatically
        inherit the parent PG affiliation.  Fresh element tags are
        allocated above the current ``maxElementTag`` so no collision
        is possible.
        """
        if not line3_tags:
            return
        gmsh.model.mesh.removeElements(1, entity_tag, line3_tags)

        # Allocate 2N fresh tags above the current max.
        base = int(gmsh.model.mesh.getMaxElementTag()) + 1
        new_tags: list[int] = []
        new_nodes: list[int] = []
        for k, (n_i, n_j, n_mid) in enumerate(line3_conn):
            new_tags.append(base + 2 * k)       # tag for (i, mid)
            new_tags.append(base + 2 * k + 1)   # tag for (mid, j)
            new_nodes.extend([n_i, n_mid, n_mid, n_j])

        gmsh.model.mesh.addElements(
            1, entity_tag,
            elementTypes=[_GMSH_LINE2],
            elementTags=[new_tags],
            nodeTags=[new_nodes],
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
