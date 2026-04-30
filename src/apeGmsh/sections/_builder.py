"""
SectionsBuilder — composite for building sections directly in session.
=====================================================================

Accessed via ``g.sections``.  Builds parametric cross-section
geometry directly in the current apeGmsh session — no Part
intermediary, no STEP round-trip, no sidecar.  Labels are created
natively in the session's label system.

Returns an :class:`Instance` with ``.labels`` accessor so the user
can address sub-regions by name.

Usage::

    with apeGmsh("frame") as g:
        col = g.sections.W_solid(
            bf=150, tf=20, h=300, tw=10, length=2000, label="col",
        )
        col.labels.web           # -> "col.web"
        col.labels.start_face    # -> "col.start_face"
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from apeGmsh._types import SessionProtocol as _SessionBase
    from apeGmsh.core._parts_registry import Instance


from apeGmsh._logging import _HasLogging
from apeGmsh.core._section_placement import apply_placement
from ._classify import (
    classify_end_faces,
    classify_w_volumes,
    classify_w_outer_faces,
    classify_tee_outer_faces,
    classify_angle_outer_faces,
)


class _PrefixedLabels:
    """Proxy that prepends ``prefix.`` to every :meth:`add` call.

    Lets the standalone classify helpers (which use bare names like
    ``"body"``, ``"start_face"``) work inside a shared session where
    labels must be namespaced per instance.
    """
    __slots__ = ("_labels", "_prefix")

    def __init__(self, labels, prefix: str) -> None:
        self._labels = labels
        self._prefix = prefix

    def add(self, dim, tags, name: str) -> None:
        self._labels.add(dim, tags, name=f"{self._prefix}.{name}")


class SectionsBuilder(_HasLogging):
    """Direct in-session section builder (``g.sections``)."""

    _log_prefix = "Sections"

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent

    def _build_section(
        self,
        build_fn,
        label: str,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
        lc: float = 1e22,
        *,
        anchor=None,
        align=None,
        length: float | None = None,
    ) -> "Instance":
        """Shared helper: snapshot entities before/after the build
        function, register the delta as an Instance, apply transforms.
        """
        from apeGmsh.core._parts_registry import Instance

        parts = getattr(self._parent, 'parts', None)

        # Check instance label uniqueness BEFORE building geometry.
        # If we build first and check later, a duplicate label leaves
        # orphaned geometry + labels in the session with no cleanup.
        if parts is not None and label in parts._instances:
            raise ValueError(
                f"Section label '{label}' already exists in the "
                f"session. Use a different label."
            )

        # Snapshot current entities
        before: dict[int, set[int]] = {}
        for d in range(4):
            before[d] = {t for _, t in gmsh.model.getEntities(d)}

        # Run the build function — it creates geometry + labels
        # directly in the current session
        build_fn()

        # Compute the delta
        after: dict[int, set[int]] = {}
        for d in range(4):
            after[d] = {t for _, t in gmsh.model.getEntities(d)}

        entities: dict[int, list[int]] = {}
        for d in range(4):
            new_tags = sorted(after[d] - before[d])
            if new_tags:
                entities[d] = new_tags

        # OCC operations (add_rectangle, extrude, fragment, …) create
        # BRep points with lc=0.  If MeshSizeFromPoints is on (default),
        # those zeros override set_global_size and cause
        # "Wrong mesh element size lc = 0" errors.
        # When lc is the default 1e22, section points impose no size
        # constraint — set_global_size governs.  When the user passes a
        # specific value, that becomes the section's target element size.
        new_pts = entities.get(0, [])
        if new_pts:
            gmsh.model.mesh.setSize([(0, t) for t in new_pts], lc)

        # Compose anchor/align (local-frame placement) AND the user's
        # translate/rotate into one affine matrix and apply via a
        # single OCC call.  Each separate sync would renumber boundary
        # sub-topology and force its own snap/restore cycle — composing
        # avoids that entirely.  ``affected`` is the build_fn's entity
        # delta so PG handling stays scoped to this section in a
        # shared session.
        needs_placement = (
            anchor is not None or align is not None
            or translate != (0.0, 0.0, 0.0) or rotate is not None
        )
        if entities and needs_placement:
            top_dim = max(entities)
            placement_dimtags = [(top_dim, t) for t in entities[top_dim]]
            affected: list[tuple[int, int]] = [
                (int(d), int(t))
                for d, tags in entities.items()
                for t in tags
            ]
            apply_placement(
                anchor if anchor is not None else "start",
                align if align is not None else "z",
                length=length,
                user_translate=translate,
                user_rotate=rotate,
                dimtags=placement_dimtags,
                affected=affected,
            )

        # Harvest labels created during the build
        labels_comp = getattr(self._parent, 'labels', None)
        label_names: list[str] = []
        if labels_comp is not None:
            for lbl_name in labels_comp.get_all():
                # Only include labels that were just created (they
                # reference tags in our delta)
                try:
                    lbl_tags = labels_comp.entities(lbl_name)
                    all_new = set()
                    for d_tags in entities.values():
                        all_new.update(d_tags)
                    if any(t in all_new for t in lbl_tags):
                        label_names.append(lbl_name)
                except Exception:
                    pass

        # Create umbrella label for the entire section instance
        # (mirrors _parts_registry._import_cad).
        if labels_comp is not None and entities:
            top_dim = max(entities)
            try:
                labels_comp.add(top_dim, entities[top_dim], name=label)
                label_names.append(label)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Umbrella label creation failed for "
                    f"{label!r} (dim={top_dim}): {exc}",
                    stacklevel=2,
                )

        # Compute bbox
        bbox = None
        if parts is not None:
            dimtags_all = [
                (d, t) for d, tags in entities.items() for t in tags
            ]
            bbox = parts._compute_bbox(dimtags_all)

        # Register as an Instance in parts registry
        if parts is not None:
            parts._counter += 1

        inst = Instance(
            label=label,
            part_name=label,
            entities=entities,
            translate=translate,
            rotate=rotate,
            bbox=bbox,
            label_names=label_names,
        )

        if parts is not None:
            parts._register_instance(inst)

        self._log(
            f"built {label!r}: {sum(len(v) for v in entities.values())} "
            f"entities, labels={label_names}"
        )
        return inst

    # ------------------------------------------------------------------
    # W-shape solid
    # ------------------------------------------------------------------

    def W_solid(
        self,
        bf: float,
        tf: float,
        h: float,
        tw: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "W_solid",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a W-shape solid directly in the current session.

        Same geometry as :func:`apeGmsh.sections.W_solid` but
        without a Part intermediary.  Returns an Instance with
        ``.labels`` accessor.

        Parameters
        ----------
        anchor : str or (x, y, z), default ``"start"``
            Re-origin the section in its local frame before optional
            ``align`` and before the user's ``translate``/``rotate``.
            See :func:`apeGmsh.core._section_placement.compute_anchor_offset`.
        align : str or (ax, ay, az), default ``"z"``
            Reorient the local +Z axis to a world direction.
            See :func:`apeGmsh.core._section_placement.compute_alignment_rotation`.
        lc : float
            Target element size for this section's BRep points.
            Default ``1e22`` imposes no constraint — element size
            is governed by :meth:`set_global_size` alone.

        Example
        -------
        ::

            with apeGmsh("frame") as g:
                col = g.sections.W_solid(
                    bf=150, tf=20, h=300, tw=10, length=2000,
                    label="col", lc=50,
                )
                g.mesh.sizing.set_global_size(100)
                g.mesh.generation.generate(3)
        """
        def _build():
            geo = self._parent.model.geometry
            boo = self._parent.model.boolean
            tr  = self._parent.model.transforms

            total_h = 2 * tf + h
            outer  = geo.add_rectangle(x=-bf/2, y=-total_h/2, z=0, dx=bf, dy=total_h)
            void_l = geo.add_rectangle(x=-bf/2, y=-h/2, z=0, dx=bf/2-tw/2, dy=h)
            void_r = geo.add_rectangle(x=tw/2,  y=-h/2, z=0, dx=bf/2-tw/2, dy=h)
            boo.cut(outer, [void_l, void_r], dim=2)

            surfs = gmsh.model.getEntities(2)
            if surfs:
                tr.extrude(surfs[0], 0, 0, length)

            geo.slice(axis='x', offset=-tw/2)
            geo.slice(axis='x', offset=tw/2)
            geo.slice(axis='y', offset=h/2)
            geo.slice(axis='y', offset=-h/2)

            # Label volumes by structural role
            labels = self._parent.labels
            # Prefix labels with the instance label
            top_tags, bot_tags, web_tags = [], [], []
            for _, tag in gmsh.model.getEntities(3):
                com = gmsh.model.occ.getCenterOfMass(3, tag)
                if com[1] > h/2:
                    top_tags.append(tag)
                elif com[1] < -h/2:
                    bot_tags.append(tag)
                else:
                    web_tags.append(tag)
            if top_tags:
                labels.add(3, top_tags, name=f"{label}.top_flange")
            if bot_tags:
                labels.add(3, bot_tags, name=f"{label}.bottom_flange")
            if web_tags:
                labels.add(3, web_tags, name=f"{label}.web")

            # End faces
            start_tags, end_tags = [], []
            for _, tag in gmsh.model.getEntities(2):
                try:
                    com = gmsh.model.occ.getCenterOfMass(2, tag)
                except Exception:
                    continue
                if abs(com[2]) < 1e-3:
                    start_tags.append(tag)
                elif abs(com[2] - length) < 1e-3:
                    end_tags.append(tag)
            if start_tags:
                labels.add(2, start_tags, name=f"{label}.start_face")
            if end_tags:
                labels.add(2, end_tags, name=f"{label}.end_face")

        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )

    # ------------------------------------------------------------------
    # Rectangular solid
    # ------------------------------------------------------------------

    def rect_solid(
        self,
        b: float,
        h: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "rect",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a solid rectangular bar directly in the session.

        Parameters
        ----------
        lc : float
            Target element size for this section's BRep points.
            Default ``1e22`` imposes no constraint.
        """
        def _build():
            tag = self._parent.model.geometry.add_box(
                -b/2, -h/2, 0, b, h, length,
            )
            self._parent.labels.add(3, [tag], name=f"{label}.body")
        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )

    # ------------------------------------------------------------------
    # Hollow rectangular (HSS)
    # ------------------------------------------------------------------

    def rect_hollow(
        self,
        b: float,
        h: float,
        t: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "rect_hollow",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a hollow rectangular tube (HSS) directly in the session.

        Parameters
        ----------
        b : float
            Outer width (X-direction).
        h : float
            Outer height (Y-direction).
        t : float
            Wall thickness.
        length : float
            Extrusion length (Z-direction).
        lc : float
            Target element size. Default ``1e22`` imposes no constraint.
        """
        def _build():
            geo = self._parent.model.geometry
            outer = geo.add_box(-b/2, -h/2, 0, b, h, length)
            inner = geo.add_box(-b/2 + t, -h/2 + t, 0, b - 2*t, h - 2*t, length)
            self._parent.model.boolean.cut(outer, [inner])
            lbl = _PrefixedLabels(self._parent.labels, label)
            for _, tag in gmsh.model.getEntities(3):
                lbl.add(3, [tag], name="body")
                break
            classify_end_faces(length, lbl)

        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )

    # ------------------------------------------------------------------
    # Circular pipe (solid)
    # ------------------------------------------------------------------

    def pipe_solid(
        self,
        r: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "pipe_solid",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a solid circular bar directly in the session.

        Parameters
        ----------
        r : float
            Radius.
        length : float
            Extrusion length (Z-direction).
        lc : float
            Target element size. Default ``1e22`` imposes no constraint.
        """
        def _build():
            tag = self._parent.model.geometry.add_cylinder(0, 0, 0, 0, 0, length, r)
            lbl = _PrefixedLabels(self._parent.labels, label)
            lbl.add(3, [tag], name="body")
            classify_end_faces(length, lbl)

        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )

    # ------------------------------------------------------------------
    # Hollow circular pipe
    # ------------------------------------------------------------------

    def pipe_hollow(
        self,
        r_outer: float,
        t: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "pipe_hollow",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a hollow circular pipe directly in the session.

        Parameters
        ----------
        r_outer : float
            Outer radius.
        t : float
            Wall thickness.
        length : float
            Extrusion length (Z-direction).
        lc : float
            Target element size. Default ``1e22`` imposes no constraint.
        """
        def _build():
            geo = self._parent.model.geometry
            outer = geo.add_cylinder(0, 0, 0, 0, 0, length, r_outer)
            inner = geo.add_cylinder(0, 0, 0, 0, 0, length, r_outer - t)
            self._parent.model.boolean.cut(outer, [inner])
            lbl = _PrefixedLabels(self._parent.labels, label)
            for _, tag in gmsh.model.getEntities(3):
                lbl.add(3, [tag], name="body")
                break
            classify_end_faces(length, lbl)

        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )

    # ------------------------------------------------------------------
    # L-shape (angle)
    # ------------------------------------------------------------------

    def angle_solid(
        self,
        b: float,
        h: float,
        t: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "angle_solid",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build an L-shape (angle) directly in the session.

        Parameters
        ----------
        b : float
            Horizontal leg width (X-direction).
        h : float
            Vertical leg height (Y-direction).
        t : float
            Thickness of both legs.
        length : float
            Extrusion length (Z-direction).
        lc : float
            Target element size. Default ``1e22`` imposes no constraint.
        """
        def _build():
            geo = self._parent.model.geometry
            boo = self._parent.model.boolean
            tr  = self._parent.model.transforms

            h_leg = geo.add_rectangle(x=0, y=0, z=0, dx=b, dy=t)
            v_leg = geo.add_rectangle(x=0, y=0, z=0, dx=t, dy=h)
            boo.fuse([h_leg], [v_leg], dim=2)

            surfs = gmsh.model.getEntities(2)
            if surfs:
                tr.extrude(surfs[0], 0, 0, length)

            geo.slice(axis='x', offset=t)
            geo.slice(axis='y', offset=t)

            lbl = _PrefixedLabels(self._parent.labels, label)
            h_tags, v_tags = [], []
            for _, tag in gmsh.model.getEntities(3):
                com = gmsh.model.occ.getCenterOfMass(3, tag)
                if com[1] < t:
                    h_tags.append(tag)
                else:
                    v_tags.append(tag)
            if h_tags:
                lbl.add(3, h_tags, name="horizontal_leg")
            if v_tags:
                lbl.add(3, v_tags, name="vertical_leg")
            classify_end_faces(length, lbl)
            classify_angle_outer_faces(lbl)

        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )

    # ------------------------------------------------------------------
    # C-shape (channel)
    # ------------------------------------------------------------------

    def channel_solid(
        self,
        bf: float,
        tf: float,
        h: float,
        tw: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "channel_solid",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a C-shape (channel) directly in the session.

        Parameters
        ----------
        bf : float
            Flange width.
        tf : float
            Flange thickness.
        h : float
            Clear web height (between flanges).
        tw : float
            Web thickness.
        length : float
            Extrusion length (Z-direction).
        lc : float
            Target element size. Default ``1e22`` imposes no constraint.
        """
        def _build():
            geo = self._parent.model.geometry
            boo = self._parent.model.boolean
            tr  = self._parent.model.transforms

            total_h = h + 2 * tf
            outer = geo.add_rectangle(x=0, y=-total_h/2, z=0, dx=bf, dy=total_h)
            void  = geo.add_rectangle(x=tw, y=-h/2, z=0, dx=bf - tw, dy=h)
            boo.cut(outer, [void], dim=2)

            surfs = gmsh.model.getEntities(2)
            if surfs:
                tr.extrude(surfs[0], 0, 0, length)

            geo.slice(axis='x', offset=tw)
            geo.slice(axis='y', offset=h/2)
            geo.slice(axis='y', offset=-h/2)

            lbl = _PrefixedLabels(self._parent.labels, label)
            classify_w_volumes(h, tw, tf, bf, lbl)
            classify_end_faces(length, lbl)
            classify_w_outer_faces(h, tf, lbl)

        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )

    # ------------------------------------------------------------------
    # T-shape (tee)
    # ------------------------------------------------------------------

    def tee_solid(
        self,
        bf: float,
        tf: float,
        h: float,
        tw: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "tee_solid",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a T-shape (tee) directly in the session.

        Parameters
        ----------
        bf : float
            Flange width.
        tf : float
            Flange thickness.
        h : float
            Stem height.
        tw : float
            Stem thickness.
        length : float
            Extrusion length (Z-direction).
        lc : float
            Target element size. Default ``1e22`` imposes no constraint.
        """
        def _build():
            geo = self._parent.model.geometry
            boo = self._parent.model.boolean
            tr  = self._parent.model.transforms

            flange = geo.add_rectangle(x=-bf/2, y=0, z=0, dx=bf, dy=tf)
            stem   = geo.add_rectangle(x=-tw/2, y=-h, z=0, dx=tw, dy=h)
            boo.fuse([flange], [stem], dim=2)

            surfs = gmsh.model.getEntities(2)
            if surfs:
                tr.extrude(surfs[0], 0, 0, length)

            geo.slice(axis='x', offset=-tw/2)
            geo.slice(axis='x', offset=tw/2)
            geo.slice(axis='y', offset=0)

            lbl = _PrefixedLabels(self._parent.labels, label)
            flange_tags, stem_tags = [], []
            for _, tag in gmsh.model.getEntities(3):
                com = gmsh.model.occ.getCenterOfMass(3, tag)
                if com[1] >= 0:
                    flange_tags.append(tag)
                else:
                    stem_tags.append(tag)
            if flange_tags:
                lbl.add(3, flange_tags, name="flange")
            if stem_tags:
                lbl.add(3, stem_tags, name="stem")
            classify_end_faces(length, lbl)
            classify_tee_outer_faces(h, tf, lbl)

        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )

    # ------------------------------------------------------------------
    # W-shape shell (mid-surfaces)
    # ------------------------------------------------------------------

    def W_shell(
        self,
        bf: float,
        tf: float,
        h: float,
        tw: float,
        length: float,
        *,
        anchor="start",
        align="z",
        label: str = "W_shell",
        lc: float = 1e22,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a W-shape as 3 mid-surface shell rectangles.

        Parameters
        ----------
        lc : float
            Target element size for this section's BRep points.
            Default ``1e22`` imposes no constraint.
        """
        def _build():
            from apeGmsh.sections.shell import _build_rect_surface
            geo = self._parent.model.geometry
            y_top = h/2 + tf/2
            y_bot = -(h/2 + tf/2)

            _build_rect_surface(
                geo,
                -bf/2, y_top, 0,  bf/2, y_top, 0,
                bf/2, y_top, length,  -bf/2, y_top, length,
                label=f"{label}.top_flange",
            )
            _build_rect_surface(
                geo,
                -bf/2, y_bot, 0,  bf/2, y_bot, 0,
                bf/2, y_bot, length,  -bf/2, y_bot, length,
                label=f"{label}.bottom_flange",
            )
            _build_rect_surface(
                geo,
                0, -h/2, 0,  0, h/2, 0,
                0, h/2, length,  0, -h/2, length,
                label=f"{label}.web",
            )
            self._parent.model.sync()

        return self._build_section(
            _build, label, translate, rotate, lc=lc,
            anchor=anchor, align=align, length=length,
        )
