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
        col.labels.web           # → "col.web"
        col.labels.start_face    # → "col.start_face"
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase
    from apeGmsh.core._parts_registry import Instance


class SectionsBuilder:
    """Direct in-session section builder (``g.sections``)."""

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[Sections] {msg}")

    def _build_section(
        self,
        build_fn,
        label: str,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Shared helper: snapshot entities before/after the build
        function, register the delta as an Instance, apply transforms.
        """
        from apeGmsh.core._parts_registry import Instance

        parts = getattr(self._parent, 'parts', None)

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

        # Apply transforms to the new entities (top-dim only)
        if entities:
            top_dim = max(entities)
            transform_dimtags = [(top_dim, t) for t in entities[top_dim]]
            dx, dy, dz = translate
            if rotate is not None:
                if len(rotate) == 4:
                    angle, ax, ay, az = rotate
                    gmsh.model.occ.rotate(
                        transform_dimtags, 0, 0, 0, ax, ay, az, angle,
                    )
                elif len(rotate) == 7:
                    angle, ax, ay, az, cx, cy, cz = rotate
                    gmsh.model.occ.rotate(
                        transform_dimtags, cx, cy, cz, ax, ay, az, angle,
                    )
                gmsh.model.occ.synchronize()
            if dx != 0.0 or dy != 0.0 or dz != 0.0:
                gmsh.model.occ.translate(transform_dimtags, dx, dy, dz)
                gmsh.model.occ.synchronize()

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

        # Compute bbox
        bbox = None
        if parts is not None:
            dimtags_all = [
                (d, t) for d, tags in entities.items() for t in tags
            ]
            bbox = parts._compute_bbox(dimtags_all)

        # Register as an Instance in parts registry
        if parts is not None:
            if label in parts._instances:
                raise ValueError(f"Section label '{label}' already exists.")
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
            parts._instances[label] = inst

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
        label: str = "W_solid",
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a W-shape solid directly in the current session.

        Same geometry as :func:`apeGmsh.sections.W_solid` but
        without a Part intermediary.  Returns an Instance with
        ``.labels`` accessor.

        Example
        -------
        ::

            with apeGmsh("frame") as g:
                col = g.sections.W_solid(
                    bf=150, tf=20, h=300, tw=10, length=2000,
                    label="col",
                )
                g.mesh.structured.set_transfinite_automatic()
                g.mesh.sizing.set_global_size(50)
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

        return self._build_section(_build, label, translate, rotate)

    # ------------------------------------------------------------------
    # Rectangular solid
    # ------------------------------------------------------------------

    def rect_solid(
        self,
        b: float,
        h: float,
        length: float,
        *,
        label: str = "rect",
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a solid rectangular bar directly in the session."""
        def _build():
            tag = self._parent.model.geometry.add_box(
                -b/2, -h/2, 0, b, h, length,
            )
            self._parent.labels.add(3, [tag], name=f"{label}.body")
        return self._build_section(_build, label, translate, rotate)

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
        label: str = "W_shell",
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
    ) -> "Instance":
        """Build a W-shape as 3 mid-surface shell rectangles."""
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

        return self._build_section(_build, label, translate, rotate)
