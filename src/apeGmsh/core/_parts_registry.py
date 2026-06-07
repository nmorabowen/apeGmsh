"""
PartsRegistry — Instance management for multi-part workflows.

Replaces Assembly.py's instance storage, CAD import, fragmentation,
and node/face mapping.  Registered as ``g.parts``.

Four entry points for creating instances:

* ``with g.parts.part("beam"):`` — context manager, diff-based tracking
* ``g.parts.register("slab", [(3, tag)])`` — manual tagging
* ``g.parts.add(part_obj, label="col")`` — import a saved Part
* ``g.parts.import_step("file.step", label="col")`` — import CAD file

Usage::

    g = apeGmsh("bridge")
    g.begin()

    with g.parts.part("beam"):
        g.model.geometry.add_box(0, 0, 0, 1, 0.5, 10)

    g.parts.import_step("slab.step", label="slab", translate=(0, 0, 10))
    g.parts.fragment_all()

    # Build node map for constraint resolution
    fem = g.mesh.queries.get_fem_data(dim=3)
    nm  = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np

from ._parts_fragmentation import _PartsFragmentationMixin

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase
    from apeGmsh.core.Part import Part
    from apeGmsh.core._instance_edit import InstanceEdit

from apeGmsh._types import DimTag


# ---------------------------------------------------------------------------
# Instance dataclass  (same fields as Assembly's Instance)
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """Bookkeeping record for one part placement.

    Attributes
    ----------
    label       : unique name inside the session
    part_name   : name of the source Part or file stem
    file_path   : CAD file that was imported (None for inline parts)
    entities    : ``{dim: [tag, ...]}`` — updated in-place by fragment
    translate   : applied translation (dx, dy, dz)
    rotate      : applied rotation (angle_rad, ax, ay, az[, cx, cy, cz])
    properties  : arbitrary user metadata
    bbox        : axis-aligned bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
    label_names : label names created for this instance (Tier 1
                  naming, e.g. ``["col_A.shaft", "col_A.top"]``).
                  Populated by ``_import_cad`` when the Part's CAD
                  file has a ``.apegmsh.json`` sidecar carrying
                  label definitions.  These are NOT solver-facing
                  physical groups — use ``g.labels.entities(name)``
                  to resolve entity tags, and
                  ``g.labels.promote_to_physical(name)`` to create
                  a solver PG when ready.
    """
    label: str
    part_name: str
    file_path: Path | None = None
    entities: dict[int, list[int]] = field(default_factory=dict)
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate: tuple[float, ...] | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    bbox: tuple[float, float, float, float, float, float] | None = None
    label_names: list[str] = field(default_factory=list)
    labels: "_InstanceLabels" = field(init=False, repr=False)
    edit: "InstanceEdit" = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'labels', _InstanceLabels(self))
        # ``edit`` is attached by the registry at registration time —
        # see ``PartsRegistry._register_instance``.  Defaults to None
        # for unregistered Instances (e.g. during construction).


class _InstanceLabels:
    """Attribute-access helper for Part labels on an Instance.

    Accessed via ``inst.labels``.  Each Part label becomes an
    attribute that returns the prefixed label string ready to
    pass to any method::

        inst = g.parts.add(column, label="col")
        inst.labels.web            # -> "col.web"
        inst.labels.top_flange     # -> "col.top_flange"
        inst.labels.start_face     # -> "col.start_face"

    Typos raise ``AttributeError`` with the list of available
    labels.  Combined with the shared entity resolver, the user
    never types a raw label string.
    """

    __slots__ = ('_inst',)

    def __init__(self, inst: Instance) -> None:
        object.__setattr__(self, '_inst', inst)

    def __getattr__(self, name: str) -> str:
        inst = object.__getattribute__(self, '_inst')
        prefixed = f"{inst.label}.{name}"
        if prefixed in inst.label_names:
            return prefixed
        available = [
            n.split('.', 1)[1] for n in inst.label_names if '.' in n
        ]
        raise AttributeError(
            f"Instance '{inst.label}' has no label '{name}'. "
            f"Available: {available}"
        )

    def __dir__(self) -> list[str]:
        """Enable IDE autocomplete for available labels."""
        inst = object.__getattribute__(self, '_inst')
        return [
            n.split('.', 1)[1] for n in inst.label_names if '.' in n
        ]

    def __repr__(self) -> str:
        inst = object.__getattribute__(self, '_inst')
        names = [n.split('.', 1)[1] for n in inst.label_names if '.' in n]
        return f"InstanceLabels({names})"


# ---------------------------------------------------------------------------
# PartsRegistry composite
# ---------------------------------------------------------------------------

class PartsRegistry(_PartsFragmentationMixin):
    """Instance management composite — registered as ``g.parts``."""

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent
        self._instances: dict[str, Instance] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def instances(self) -> dict[str, Instance]:
        """Read-only view of all instances."""
        return dict(self._instances)

    # ------------------------------------------------------------------
    # Internal: instance registration with edit-composite wiring
    # ------------------------------------------------------------------

    def _register_instance(self, inst: Instance) -> Instance:
        """Add ``inst`` to the registry and attach an ``InstanceEdit``.

        All entry points (context manager, register, add, import_step,
        SectionsBuilder) funnel through here so every Instance has
        ``inst.edit`` wired up at registration time.
        """
        # Phase 3B.2d / ADR 0038 — parts registration is build-phase
        # only; the chain-phase broker carries its own immutable
        # ``parts.maps`` snapshot.
        from ._compose_errors import chain_phase_guard
        chain_phase_guard(
            self._parent, f"g.parts.<register>({inst.label})"
        )
        from ._instance_edit import InstanceEdit

        self._instances[inst.label] = inst
        object.__setattr__(inst, 'edit', InstanceEdit(inst, self))
        return inst

    # ------------------------------------------------------------------
    # Entry point 1: Context manager (inline geometry)
    # ------------------------------------------------------------------

    @contextmanager
    def part(self, label: str):
        """Track entities created inside the block as a named part.

        Yields the label string.  After the block, any entities that
        exist now but didn't before are stored as an Instance.

        Example::

            with g.parts.part("beam"):
                g.model.geometry.add_box(0, 0, 0, 1, 0.5, 10)
        """
        if label in self._instances:
            raise ValueError(f"Part label '{label}' already exists.")

        before = {d: set(t for _, t in gmsh.model.getEntities(d)) for d in range(4)}
        yield label
        after = {d: set(t for _, t in gmsh.model.getEntities(d)) for d in range(4)}

        entities: dict[int, list[int]] = {}
        for d in range(4):
            new_tags = sorted(after[d] - before[d])
            if new_tags:
                entities[d] = new_tags

        dimtags = [(d, t) for d, tags in entities.items() for t in tags]
        inst = Instance(
            label=label,
            part_name=label,
            entities=entities,
            bbox=self._compute_bbox(dimtags) if dimtags else None,
        )
        self._register_instance(inst)

    # ------------------------------------------------------------------
    # Entry point 2: Manual registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        dimtags: list[DimTag] | None = None,
        *,
        label: str | None = None,
        pg: str | None = None,
        dim: int | None = None,
    ) -> Instance:
        """Tag existing entities under a part name.

        Exactly one of ``dimtags``, ``label``, or ``pg`` must be given.

        Parameters
        ----------
        name : str
            Unique part name.
        dimtags : list of (dim, tag), optional
            Entities to assign directly.  Also accepted positionally
            as the second argument.
        label : str, optional
            Name of an apeGmsh label (``g.labels``) whose entities
            should be adopted.
        pg : str, optional
            Name of a physical group (``g.physical``) whose entities
            should be adopted.
        dim : int, optional
            Forwarded to ``g.labels.entities(label, dim=dim)`` when
            using ``label=`` and the label spans multiple dimensions.

        Returns
        -------
        Instance
        """
        provided = sum(x is not None for x in (dimtags, label, pg))
        if provided != 1:
            raise TypeError(
                "register() requires exactly one of dimtags=, label=, "
                f"or pg= (got {provided})."
            )

        if label is not None:
            labels_comp = self._parent.labels
            if dim is not None:
                tags = labels_comp.entities(label, dim=dim)
                resolved: list[DimTag] = [(dim, int(t)) for t in tags]
            else:
                # Raises ValueError on multi-dim, KeyError on missing
                labels_comp.entities(label)
                resolved = []
                for d in range(4):
                    try:
                        d_tags = labels_comp.entities(label, dim=d)
                    except KeyError:
                        continue
                    resolved = [(d, int(t)) for t in d_tags]
                    break
        elif pg is not None:
            physical = self._parent.physical
            resolved = []
            for d in range(4):
                pg_tag = physical.get_tag(d, pg)
                if pg_tag is None:
                    continue
                resolved.extend(
                    (d, int(t)) for t in physical.get_entities(d, pg_tag)
                )
            if not resolved:
                raise KeyError(f"No physical group named {pg!r}.")
            pg_dims = {d for d, _ in resolved}
            if len(pg_dims) > 1:
                raise ValueError(
                    f"Physical group {pg!r} exists at multiple "
                    f"dimensions {sorted(pg_dims)}. Multi-dimensional "
                    f"physical groups are not supported."
                )
        else:
            resolved = [(int(d), int(t)) for d, t in dimtags]

        if name in self._instances:
            raise ValueError(f"Part label '{name}' already exists.")

        # Ownership check — each entity can belong to at most one part
        for d, t in resolved:
            for existing_label, existing_inst in self._instances.items():
                if t in existing_inst.entities.get(d, []):
                    raise ValueError(
                        f"Entity (dim={d}, tag={t}) already belongs to "
                        f"part '{existing_label}'. Remove it first."
                    )

        entities: dict[int, list[int]] = {}
        for d, t in resolved:
            entities.setdefault(d, []).append(t)

        inst = Instance(
            label=name,
            part_name=name,
            entities=entities,
            bbox=self._compute_bbox(resolved) if resolved else None,
        )
        self._register_instance(inst)
        return inst

    # ------------------------------------------------------------------
    # Entry point 3: Adopt existing model geometry
    # ------------------------------------------------------------------

    def from_model(
        self,
        label: str,
        *,
        dim: int | None = None,
        tags: list[int] | None = None,
    ) -> Instance:
        """Adopt entities already in the Gmsh session as a named part.

        Useful after ``g.model.io.load_step()`` or ``g.model.io.load_iges()``
        when you want the imported geometry tracked for constraints
        and fragmentation.

        Parameters
        ----------
        label : str
            Part name.
        dim : int, optional
            Dimension to adopt.  If None, adopts all dimensions.
        tags : list[int], optional
            Specific entity tags to adopt.  If None, adopts all
            **untracked** entities (not already assigned to a part).

        Returns
        -------
        Instance

        Examples
        --------
        ::

            # Load geometry, then adopt it
            g.model.io.load_step("bracket.step")
            g.parts.from_model("bracket")

            # Adopt only specific volumes
            g.parts.from_model("slab", dim=3, tags=[1, 2])
        """
        if label in self._instances:
            raise ValueError(f"Part label '{label}' already exists.")

        # Collect already-tracked tags per dim
        tracked: dict[int, set[int]] = {}
        for inst in self._instances.values():
            for d, ts in inst.entities.items():
                tracked.setdefault(d, set()).update(ts)

        # Determine which dims to scan
        dims = [dim] if dim is not None else list(range(4))

        entities: dict[int, list[int]] = {}
        for d in dims:
            all_tags_d = [t for _, t in gmsh.model.getEntities(d)]
            if tags is not None:
                # User specified exact tags — use them
                adopted = [t for t in all_tags_d if t in tags]
            else:
                # Adopt untracked entities
                adopted = [t for t in all_tags_d if t not in tracked.get(d, set())]
            if adopted:
                entities[d] = sorted(adopted)

        if not entities:
            import warnings
            warnings.warn(
                f"No entities to adopt for part '{label}'.  "
                f"All entities are already tracked or the session is empty.",
                stacklevel=2,
            )

        dimtags = [(d, t) for d, ts in entities.items() for t in ts]
        inst = Instance(
            label=label,
            part_name=label,
            entities=entities,
            bbox=self._compute_bbox(dimtags) if dimtags else None,
        )
        self._register_instance(inst)
        return inst

    # ------------------------------------------------------------------
    # Entry point 4: Import a Part object
    # ------------------------------------------------------------------

    def add(
        self,
        part: "Part",
        *,
        label: str | None = None,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
        highest_dim_only: bool = True,
    ) -> Instance:
        """Import a saved Part into the session.

        Parameters
        ----------
        part : Part
            Must have been ``save()``-d to disk.
        label : str, optional
            Auto-generated as ``"{part.name}_1"`` if omitted.
        translate, rotate : placement transforms.
        highest_dim_only : keep only highest-dim entities from the CAD.
        """
        if not part.has_file:
            hint = (
                "Call part.save('file.step') explicitly"
                if not getattr(part, "_auto_persist", True)
                else
                "Exit the Part's `with` block (or call part.end()) "
                "before calling parts.add(part) so auto-persist can "
                "write the tempfile, OR call part.save('file.step') "
                "explicitly"
            )
            raise FileNotFoundError(
                f"Part '{part.name}' has no file to import.  {hint}."
            )
        if label is None:
            self._counter += 1
            label = f"{part.name}_{self._counter}"
        # part.has_file was checked above; this implies file_path
        # is not None. Narrow the type for mypy.
        assert part.file_path is not None
        return self._import_cad(
            file_path=part.file_path,
            label=label,
            part_name=part.name,
            translate=translate,
            rotate=rotate,
            highest_dim_only=highest_dim_only,
            properties=dict(part.properties),
        )

    # ------------------------------------------------------------------
    # Entry point 5b: Parametric plane-wave box (soil + absorbing skin)
    # ------------------------------------------------------------------

    def add_plane_wave_box(
        self,
        *,
        x: tuple[float, int],
        y: tuple[float, int],
        z,
        skin_thickness=None,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_z_deg: float = 0.0,
        name: str | None = None,
        names: dict[str, str] | None = None,
        apply_transfinite: bool = True,
    ):
        """Build a structured soil box wrapped by an ASDAbsorbingBoundary skin.

        A *plane-wave box* is an axis-aligned structured soil box plus a
        one-element-thick absorbing **offset shell** on its five truncation
        faces (the local ``+Z`` top is the free surface and is never shelled).
        Soil + shell form one rectangular block; the shell is decomposed into
        face / vertical-edge / bottom-edge / bottom-corner regions, each tagged
        with its OpenSees ``btype``.  The companion bridge element
        (``ASDAbsorbingBoundary3D``) fans out one element per skin-region hex.

        Built directly in the live session (no Part/STEP round-trip); pairs with
        — but does not use — :meth:`add_DRM_box`.  See ADR 0054.

        Parameters
        ----------
        x, y : (size, n_elements)
            Lateral soil extent (symmetric, centred) and element count.
        z : (depth, n_elements)
            Vertical soil extent (downward, free surface at the top) and element
            count.  A ``list`` of layers (stratigraphy) is rejected in this slice.
        skin_thickness : float | (tx, ty, tz) | None
            Absorbing-skin thickness.  ``None`` (default) matches the adjacent
            soil element size per face.
        center : (cx, cy, cz)
            World location of the soil top-face centre (free surface).
        rotation_z_deg : float
            Must be ``0.0`` in this slice (rotation is a later slice).
        name, names, apply_transfinite :
            PG-name prefix, per-PG override dict, and transfinite toggle —
            mirroring :meth:`add_DRM_box`.

        Returns
        -------
        AbsorbingSkinResult
            PG names (``soil_pg``, ``skin_pgs`` by btype, ``skin_all_pg``,
            ``bottom_pgs``, ``free_surface_pg``), ``axes``, and placement.

        Example
        -------
        ::

            res = g.parts.add_plane_wave_box(
                x=(605, 22), y=(605, 20), z=(420, 16),
            )
            g.mesh.generation.generate(dim=3)
            # res.skin_pgs["L"], res.skin_all_pg, res.bottom_pgs ...
        """
        from apeGmsh.parts.plane_wave_box import build_plane_wave_box

        return build_plane_wave_box(
            self._parent,
            x=x, y=y, z=z,
            skin_thickness=skin_thickness,
            center=center,
            rotation_z_deg=rotation_z_deg,
            name=name,
            names=names,
            apply_transfinite=apply_transfinite,
        )

    # ------------------------------------------------------------------
    # Entry point 5: Parametric DRM-box primitive
    # ------------------------------------------------------------------

    def add_DRM_box(
        self,
        *,
        x_inner: tuple[float, int],
        x_layer: tuple[float, int],
        x_outer: tuple[float, int],
        y_inner: tuple[float, int],
        y_layer: tuple[float, int],
        y_outer: tuple[float, int],
        z_top: tuple[float, int],
        z_mid: tuple[float, int],
        z_bottom: tuple[float, int],
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_z_deg: float = 0.0,
        name: str | None = None,
        names: dict[str, str] | None = None,
        apply_transfinite: bool = True,
        tag_line_pgs: bool = True,
    ):
        """Build, place, and tag a Domain-Reduction-Method soil box.

        A DRM box is a layered solid with three concentric regions
        per lateral axis (inner core | transition layer | outer
        absorbing layer) and a downward Z stack (top | mid | bottom).
        The classic symmetric case has ``5 * 5 * 3 = 75`` axis-aligned
        hex sub-volumes, each meshed structured-hex with per-region
        element counts.

        ``center=(0, 0, 0)`` puts the top-face centre of the inner
        box at the origin (free-surface convention).  Rotation is
        applied CCW about ``+Z`` at ``center``; the rotated frame
        survives every step (volume PGs, line PGs, transfinite
        cascade) because we classify by world-coords transformed
        back to the local frame.

        Parameters
        ----------
        x_inner, x_layer, x_outer, y_inner, y_layer, y_outer :
            ``(size, n_elements)`` tuples — symmetric layered lateral
            axes.  Each segment's element count drives the
            transfinite cascade.
        z_top, z_mid, z_bottom :
            ``(size, n_elements)`` tuples — downward Z stack with the
            free surface at ``z = 0`` (inner-box top).
        center :
            World-coordinate location for the top-face centre of the
            inner box.
        rotation_z_deg :
            CCW rotation about ``+Z`` applied at ``center``, in
            degrees.
        name :
            Instance label and default PG prefix.  When ``None``,
            uses ``"drm_box"``.  PGs default to ``inner_box`` /
            ``transition_box`` / ``outer_box`` (and the matching
            ``lines_*`` curves); when ``name`` is given they become
            ``{name}_inner_box`` etc.
        names :
            Per-PG override dict.  Keys: ``inner_pg``, ``transition_pg``,
            ``outer_pg``, ``line_pg_<region>_<axis>`` (e.g.
            ``line_pg_inner_x``, ``line_pg_top_z``).  Each override
            replaces the entire PG name (the ``name`` prefix is
            ignored for that key).
        apply_transfinite :
            When True (default), apply the structured-hex transfinite
            cascade to every sub-volume using the per-region element
            counts in ``axis_x`` / ``axis_y`` / ``axis_z``.
        tag_line_pgs :
            When True (default), tag axis-parallel edges by region
            into curve PGs ``lines_{region}_{axis}``.  When False,
            ``result.line_pgs`` is empty.

        Returns
        -------
        DRMBoxResult
            Frozen summary with PG names, Axis1D descriptors, the
            applied ``center`` and ``rotation_z`` (in radians).

        Example
        -------
        ::

            res = g.parts.add_DRM_box(
                x_inner=(605, 10), x_layer=(10, 1), x_outer=(20, 2),
                y_inner=(605, 10), y_layer=(10, 1), y_outer=(20, 2),
                z_top=(50, 5), z_mid=(50, 5), z_bottom=(200, 20),
                center=(0, 0, 0),
            )
            g.mesh.generation.generate(dim=3)
            # res.inner_pg == "inner_box", res.transition_pg == "transition_box",
            # res.outer_pg == "outer_box"
        """
        import math
        import numpy as np

        from apeGmsh.parts.drm_box import DRMBox, DRMBoxResult

        instance_label = name or "drm_box"

        # Resolve PG names — ``name`` acts as a prefix unless the user
        # supplied ``names`` overrides.  When ``name is None`` we keep
        # the bare defaults so the simple case stays terse.
        prefix = f"{name}_" if name else ""
        pg_defaults = {
            "inner_pg":      f"{prefix}inner_box",
            "transition_pg": f"{prefix}transition_box",
            "outer_pg":      f"{prefix}outer_box",
        }
        line_pg_defaults: dict[str, str] = {}
        # Lateral axes carry 3 regions; Z carries 3 regions of its
        # own.  The dict keys mirror the spec example
        # ``{'inner_x', 'layer_x', 'top_z'}``.
        for region in ("inner", "layer", "outer"):
            for axis in ("x", "y"):
                line_pg_defaults[f"{region}_{axis}"] = (
                    f"{prefix}lines_{region}_{axis}"
                )
        for region in ("top", "mid", "bottom"):
            line_pg_defaults[f"{region}_z"] = (
                f"{prefix}lines_{region}_z"
            )

        overrides = dict(names or {})
        for k, default in pg_defaults.items():
            if k in overrides:
                pg_defaults[k] = str(overrides[k])
        # Line-PG override keys look like ``line_pg_inner_x``.
        for key in list(line_pg_defaults):
            override_key = f"line_pg_{key}"
            if override_key in overrides:
                line_pg_defaults[key] = str(overrides[override_key])

        # ── Build the DRM-box Part in its own session ────────────────
        # ``Part.begin()`` calls ``gmsh.model.add(part.name)``, which
        # makes the Part's model the current gmsh model.  The Part's
        # ``end()`` decrements the gmsh refcount but does NOT switch
        # the current model back, so without an explicit
        # ``setCurrent`` here ``self.add(drm)`` would importShapes into
        # the Part's model (doubling the volume count in the live
        # session).  Snapshot the assembly's model name first and
        # restore it after the Part's ``with`` block exits.
        assembly_model_name = self._parent.name
        drm = DRMBox(
            x_inner=x_inner, x_layer=x_layer, x_outer=x_outer,
            y_inner=y_inner, y_layer=y_layer, y_outer=y_outer,
            z_top=z_top, z_mid=z_mid, z_bottom=z_bottom,
            name=f"_drm_part_{instance_label}",
        )
        with drm:
            drm.build()
        gmsh.model.setCurrent(assembly_model_name)
        # ``drm`` now has an auto-persisted STEP tempfile.

        theta = math.radians(float(rotation_z_deg))
        rotate_arg: tuple[float, ...] | None
        if abs(theta) > 1e-15:
            # OCC rotate at world origin — then translate.  This
            # matches ``_apply_transforms``: rotate first about
            # axis through (0, 0, 0), then translate by ``center``.
            rotate_arg = (theta, 0.0, 0.0, 1.0)
        else:
            rotate_arg = None

        inst = self.add(
            drm,
            label=instance_label,
            translate=center,
            rotate=rotate_arg,
        )

        # Release the Part's tempfile — we've imported the geometry
        # and no longer need the on-disk STEP.  ``drm`` is otherwise
        # garbage-collected at function exit, but cleanup() here
        # avoids waiting for GC.
        drm.cleanup()

        # ── Classify each sub-volume in the local frame ─────────────
        cx, cy, cz = (float(v) for v in center)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def to_local(world_xyz):
            """Inverse of: rotate CCW about +Z at origin, then translate by center."""
            wx, wy, wz = world_xyz
            # subtract translation
            dx, dy, dz = wx - cx, wy - cy, wz - cz
            # inverse rotation
            lx = cos_t * dx + sin_t * dy
            ly = -sin_t * dx + cos_t * dy
            lz = dz
            return lx, ly, lz

        # Volume-PG classifier — matches the canonical DRM layout
        # the user's notebook expressed via in_box selection:
        #
        #   * ``inner_box`` = the single inner-inner-top sub-volume
        #     (the geometric "inner box" where the embedded structure
        #     lives).
        #   * ``transition_box`` = the layer-bounded AABB
        #     ``[-x_LL,+x_LL] x [-y_LL,+y_LL] x [-(z_top+z_mid), 0]``
        #     minus the inner box.  i.e. sub-vols whose lateral region
        #     is ``inner`` or ``layer`` AND whose Z region is ``top`` or
        #     ``mid``, with the single ``inner`` cell carved out.
        #   * ``outer_box`` = everything else — the absorbing region,
        #     including the inner-inner-mid / inner-inner-bottom
        #     sub-vols below the structure (per the user's geometric
        #     AABB rule, those z layers are not inside the transition
        #     shell).
        inner_vols: list[int] = []
        transition_vols: list[int] = []
        outer_vols: list[int] = []
        # ``per_class_counts`` ⇒ list of (vol_tag, nx, ny, nz)
        per_vol_counts: list[tuple[int, int, int, int]] = []

        for vtag in inst.entities.get(3, []):
            com_world = self._parent.model.queries.center_of_mass(
                int(vtag), dim=3,
            )
            lx, ly, lz = to_local(com_world)
            rx = drm.axis_x.region_of(lx)
            ry = drm.axis_y.region_of(ly)
            rz = drm.axis_z.region_of(lz)
            nx = drm.axis_x.count_for(lx)
            ny = drm.axis_y.count_for(ly)
            nz = drm.axis_z.count_for(lz)
            per_vol_counts.append((int(vtag), nx, ny, nz))

            is_inner = (rx == "inner" and ry == "inner" and rz == "top")
            inside_transition_bbox = (
                rx in ("inner", "layer")
                and ry in ("inner", "layer")
                and rz in ("top", "mid")
            )
            if is_inner:
                inner_vols.append(int(vtag))
            elif inside_transition_bbox:
                transition_vols.append(int(vtag))
            else:
                outer_vols.append(int(vtag))

        physical = self._parent.physical
        if inner_vols:
            physical.add(3, inner_vols, name=pg_defaults["inner_pg"])
        if transition_vols:
            physical.add(3, transition_vols, name=pg_defaults["transition_pg"])
        if outer_vols:
            physical.add(3, outer_vols, name=pg_defaults["outer_pg"])

        # ── Optional: transfinite cascade per sub-volume ────────────
        if apply_transfinite:
            structured = self._parent.mesh.structured
            for vtag, nx, ny, nz in per_vol_counts:
                # Tuple form ``n=(nx, ny, nz)`` orders by principal
                # axis (closest-global-axis), so it is rotation-safe
                # by construction — the dict form would require
                # global-axis-aligned edges and raise here.  Axis1D
                # stores element counts; ``set_transfinite`` takes
                # node counts (``n_nodes - 1`` elements per curve).
                structured.set_transfinite(
                    (3, vtag),
                    n=(nx + 1, ny + 1, nz + 1),
                    recombine=True,
                )

        # ── Optional: line PGs per (region, axis) ───────────────────
        line_pgs_out: dict[str, str] = {}
        if tag_line_pgs:
            from apeGmsh.parts.drm_box import classify_drm_box_lines

            all_curves = [int(t) for _d, t in gmsh.model.getEntities(1)]
            classified = classify_drm_box_lines(
                axis_x=drm.axis_x,
                axis_y=drm.axis_y,
                axis_z=drm.axis_z,
                center=(cx, cy, cz),
                rotation_z=theta,
                line_pg_names=line_pg_defaults,
                curve_tags=all_curves,
            )
            # Invert line_pg_defaults so we can pair back to region keys
            # for the result's ``line_pgs`` dict.
            name_to_key = {v: k for k, v in line_pg_defaults.items()}
            for pg_name, edge_tags in classified.items():
                physical.add(1, edge_tags, name=pg_name)
                line_pgs_out[name_to_key[pg_name]] = pg_name

        # ── Stash rebuild-required state on the Instance ────────────
        # The line-PG classifier is a pure function of (axes, center,
        # rotation, curve_tags), so a future boolean that mutates the
        # box can drop the stale PGs and replay the classifier against
        # the post-cut curves.  We persist: the axis construction
        # params (so Axis1D can be rebuilt), the line-PG name map
        # (so we know which PGs are owned by this Part), the center,
        # and rotation_z.  See ``rebuild_drm_box_line_pgs`` in
        # ``apeGmsh.parts.drm_box``.
        inst.properties.setdefault("drm_box", {}).update({
            "line_pgs": dict(line_pgs_out),
            "center": (cx, cy, cz),
            "rotation_z": float(theta),
        })

        return DRMBoxResult(
            inner_pg=pg_defaults["inner_pg"],
            transition_pg=pg_defaults["transition_pg"],
            outer_pg=pg_defaults["outer_pg"],
            line_pgs=line_pgs_out,
            axes={
                "x": drm.axis_x,
                "y": drm.axis_y,
                "z": drm.axis_z,
            },
            center=(cx, cy, cz),
            rotation_z=float(theta),
        )

    # ------------------------------------------------------------------
    # Entry point 4: Import a STEP/IGES file
    # ------------------------------------------------------------------

    def import_step(
        self,
        file_path: str | Path,
        *,
        label: str | None = None,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, ...] | None = None,
        highest_dim_only: bool = True,
        heal: bool | float | str = False,
        dedupe: bool | float = False,
        properties: dict[str, Any] | None = None,
    ) -> Instance:
        """Import a STEP or IGES file as a named instance.

        Parameters
        ----------
        file_path : path
            STEP (.step, .stp) or IGES (.iges, .igs) file.
        label : str, optional
            Auto-generated from file stem if omitted.
        translate, rotate : placement transforms.
        heal : bool, float, or "auto"
            Heal the imported CAD immediately after import — same
            semantics as :meth:`g.model.io.load_step <_IO.load_step>`:
            ``True`` / ``"auto"`` use a scale-aware tolerance, a float
            overrides, ``False`` (default) imports raw and emits a
            :class:`WarnGeomImportHealth` advisory if slivers are found.
            Best-effort for sidecar-carrying parts (healing renumbers,
            so anchors rebind against the healed geometry).
        dedupe : bool or float
            Merge coincident entities after import (and after heal).
        properties : arbitrary metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CAD file not found: {file_path}")
        if label is None:
            self._counter += 1
            label = f"{file_path.stem}_{self._counter}"
        return self._import_cad(
            file_path=file_path,
            label=label,
            part_name=file_path.stem,
            translate=translate,
            rotate=rotate,
            highest_dim_only=highest_dim_only,
            heal=heal,
            dedupe=dedupe,
            properties=properties or {},
        )

    # ------------------------------------------------------------------
    # Fragment
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Node / face maps (for constraint resolution)
    # ------------------------------------------------------------------

    def build_node_map(
        self,
        node_tags: np.ndarray,
        node_coords: np.ndarray,
    ) -> dict[str, set[int]]:
        """Partition mesh nodes by instance bounding box.

        Returns ``{label: {node_tag, ...}}``.
        """
        tags = np.asarray(node_tags)
        coords = np.asarray(node_coords).reshape(-1, 3)
        return {
            label: self._nodes_in_bbox(tags, coords, inst.bbox)
            for label, inst in self._instances.items()
        }

    def build_face_map(
        self,
        node_map: dict[str, set[int]],
    ) -> dict[str, np.ndarray]:
        """Partition surface elements by instance node ownership.

        Returns ``{label: face_connectivity_array}``.
        """
        faces = self._collect_surface_faces()
        if faces.size == 0:
            return {label: np.empty((0, 0), dtype=int)
                    for label in self._instances}

        out: dict[str, np.ndarray] = {}
        for label, nodes in node_map.items():
            if not nodes:
                out[label] = np.empty((0, faces.shape[1]), dtype=int)
                continue
            mask = np.all(np.isin(faces, list(nodes)), axis=1)
            out[label] = faces[mask]
        return out

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, label: str) -> Instance:
        """Return the Instance registered under ``label``.

        Useful when you didn't store the return value of
        :meth:`add` / :meth:`import_step` and want to access an
        Instance later — e.g. to apply ``inst.edit.*`` transforms::

            g.parts.add(beam, label="b1")
            g.parts.get("b1").edit.translate(0, 0, 50)

        Raises
        ------
        KeyError
            If no instance is registered under ``label``.  The error
            message lists the available labels so you can spot a typo.
        """
        if label not in self._instances:
            available = sorted(self._instances)
            raise KeyError(
                f"No instance labeled {label!r}.  "
                f"Available: {available}"
            )
        return self._instances[label]

    def labels(self) -> list[str]:
        """Return all instance labels in insertion order."""
        return list(self._instances.keys())

    def rename(self, old_label: str, new_label: str) -> None:
        """Rename an instance.

        Raises
        ------
        KeyError   if *old_label* does not exist.
        ValueError if *new_label* already exists.
        """
        if old_label not in self._instances:
            raise KeyError(f"No part '{old_label}'.")
        if new_label in self._instances:
            raise ValueError(f"Part '{new_label}' already exists.")
        inst = self._instances.pop(old_label)
        inst.label = new_label
        self._instances[new_label] = inst

    def delete(self, label: str) -> None:
        """Remove an instance from the registry.

        The entities remain in the Gmsh session — they become
        "untracked" and will appear under the Untracked group
        in the viewer's Parts tab.

        Raises
        ------
        KeyError if *label* does not exist.
        """
        if label not in self._instances:
            raise KeyError(f"No part '{label}'.")
        self._instances.pop(label)

    # ------------------------------------------------------------------
    # Private: CAD import (shared by add() and import_step())
    # ------------------------------------------------------------------

    def _import_cad(
        self,
        file_path: Path,
        label: str,
        part_name: str,
        translate: tuple[float, float, float],
        rotate: tuple[float, ...] | None,
        highest_dim_only: bool,
        heal: bool | float | str = False,
        dedupe: bool | float = False,
        properties: dict[str, Any] | None = None,
    ) -> Instance:
        """Import CAD geometry, apply transforms, store instance."""
        if label in self._instances:
            raise ValueError(f"Part label '{label}' already exists.")

        # Snapshot pre-import entities so heal / dedupe (which renumber)
        # can re-derive the surviving imported set — mirrors
        # ``_IO._import_shapes``.
        will_mutate = bool(heal) or bool(dedupe)
        snapshot: dict[int, set[int]] = (
            {d: {t for _, t in gmsh.model.getEntities(d)} for d in range(4)}
            if will_mutate else {}
        )

        raw = gmsh.model.occ.importShapes(
            str(file_path), highestDimOnly=highest_dim_only,
        )
        gmsh.model.occ.synchronize()

        if heal:
            from ._model_io import _model_bbox_diag, _suggested_heal_tolerance
            if heal is True or heal == "auto":
                heal_tol = _suggested_heal_tolerance(_model_bbox_diag())
            else:
                heal_tol = float(heal)
            if raw:
                self._parent.model.io.heal_shapes(
                    list(raw), tolerance=heal_tol, sync=True,
                )
        if dedupe:
            dedupe_tol = None if dedupe is True else float(dedupe)
            self._parent.model.queries.remove_duplicates(
                tolerance=dedupe_tol, sync=True,
            )
        if will_mutate:
            raw = [
                (d, t)
                for d in range(4)
                for t in sorted(
                    {t for _, t in gmsh.model.getEntities(d)}
                    - snapshot.get(d, set())
                )
            ]
        elif not heal:
            # Raw import: surface a non-mutating health advisory so the
            # assembly path gets the same signal as g.model.io.load_step.
            self._parent.model.io.diagnose(warn=True)

        # ``importShapes`` returns a flat list with every sub-entity at
        # every dimension, and the same lower-dim tags appear multiple
        # times because shared edges/points belong to several faces.
        # Deduplicate per dim as we collect.
        entities: dict[int, list[int]] = {}
        seen: dict[int, set[int]] = {}
        for dim, tag in raw:
            tag_set = seen.setdefault(dim, set())
            if tag in tag_set:
                continue
            tag_set.add(tag)
            entities.setdefault(dim, []).append(tag)

        # Flat list of EVERY entity we imported — used for bbox and
        # anchor rebinding (both want the full set).
        dimtags_all = [(d, t) for d, tags in entities.items() for t in tags]

        # OCC transforms propagate through sub-topology automatically:
        # translating a volume moves its surfaces, edges, and vertices
        # in one operation.  Passing the full ``dimtags_all`` list to
        # ``translate`` raises "OpenCASCADE transform changed the
        # number of shapes" because the lower-dim sub-shapes try to
        # transform twice.  Use only the highest-dim entities as the
        # transform handles.
        top_dim = max(entities) if entities else -1
        if top_dim >= 0:
            transform_dimtags = [(top_dim, t) for t in entities[top_dim]]
        else:
            transform_dimtags = []
        self._apply_transforms(transform_dimtags, translate, rotate)
        dx, dy, dz = translate

        # Rebind labels from the sidecar (if present).
        # For each label defined in the Part, re-create it as a
        # label PG (Tier 1) in the Assembly with an instance-scoped
        # name: "{instance_label}.{pg_name}".  These are NOT user-
        # facing physical groups — the user promotes them when ready.
        label_names: list[str] = []
        # The umbrella-label block below (around the final ``if
        # labels_comp is not None and top_dim >= 0:``) runs even when
        # there is no sidecar payload, so ``labels_comp`` must be
        # defined before the ``if payload is not None`` branch.
        labels_comp = getattr(self._parent, 'labels', None)
        if isinstance(file_path, Path):
            from ._part_anchors import read_sidecar, rebind_physical_groups
            payload = read_sidecar(file_path)
            if payload is not None:
                anchors = payload.get('anchors', [])
                # For rebinding, we need ALL entities in the model
                # (including sub-entities like surfaces of volumes)
                # because the sidecar may carry anchors at any dim.
                # ``entities`` from importShapes may be incomplete
                # when ``highest_dim_only=True`` was used.
                all_model_entities: dict[int, list[int]] = {}
                for d in range(4):
                    tags_at_d = [
                        t for _, t in gmsh.model.getEntities(d)
                    ]
                    if tags_at_d:
                        all_model_entities[d] = tags_at_d
                pg_matches = rebind_physical_groups(
                    anchors=anchors,
                    imported_entities=all_model_entities,
                    translate=(dx, dy, dz),
                    rotate=rotate,
                    gmsh_module=gmsh,
                )
                if labels_comp is not None and pg_matches:
                    for pg_name, dimtags in pg_matches.items():
                        prefixed = f"{label}.{pg_name}"
                        by_dim: dict[int, list[int]] = {}
                        for d, t in dimtags:
                            by_dim.setdefault(d, []).append(t)
                        for d, tags in by_dim.items():
                            try:
                                labels_comp.add(d, tags, name=prefixed)
                                label_names.append(prefixed)
                            except Exception as exc:
                                import warnings
                                warnings.warn(
                                    f"Label rebinding failed for "
                                    f"{prefixed!r} (dim={d}): {exc}",
                                    stacklevel=2,
                                )

        # Create umbrella label for the entire Part instance.
        # This allows ``fem.nodes.get(label="column")`` to return
        # all nodes of the part, not just a sub-component.
        if labels_comp is not None and top_dim >= 0:
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

        inst = Instance(
            label=label,
            part_name=part_name,
            file_path=file_path.resolve() if isinstance(file_path, Path) else file_path,
            entities=entities,
            translate=(dx, dy, dz),
            rotate=rotate,
            properties=properties or {},
            bbox=self._compute_bbox(dimtags_all),
            label_names=label_names,
        )
        self._register_instance(inst)
        return inst

    # ------------------------------------------------------------------
    # Private: transforms (DRY — used by _import_cad)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transforms(
        dimtags: list[DimTag],
        translate: tuple[float, float, float],
        rotate: tuple[float, ...] | None,
    ) -> None:
        """Apply rotation then translation to dimtags."""
        if not dimtags:
            return
        if rotate is not None:
            if len(rotate) == 4:
                angle, ax, ay, az = rotate
                cx = cy = cz = 0.0
            elif len(rotate) == 7:
                angle, ax, ay, az, cx, cy, cz = rotate
            else:
                raise ValueError(
                    "rotate must be (angle, ax, ay, az) or "
                    "(angle, ax, ay, az, cx, cy, cz)"
                )
            gmsh.model.occ.rotate(dimtags, cx, cy, cz, ax, ay, az, angle)
            gmsh.model.occ.synchronize()

        dx, dy, dz = translate
        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            gmsh.model.occ.translate(dimtags, dx, dy, dz)
            gmsh.model.occ.synchronize()

    # ------------------------------------------------------------------
    # Private: spatial helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bbox(
        dimtags: list[DimTag],
    ) -> tuple[float, float, float, float, float, float] | None:
        """Compute the AABB of a set of entities."""
        if not dimtags:
            return None
        xmin = ymin = zmin = float("inf")
        xmax = ymax = zmax = float("-inf")
        for dim, tag in dimtags:
            try:
                bb = gmsh.model.getBoundingBox(dim, tag)
                xmin = min(xmin, bb[0])
                ymin = min(ymin, bb[1])
                zmin = min(zmin, bb[2])
                xmax = max(xmax, bb[3])
                ymax = max(ymax, bb[4])
                zmax = max(zmax, bb[5])
            except Exception:
                # Entity may lack a valid bbox (e.g. degenerate edge).
                # Skipping is safe — if ALL fail, the method returns None.
                pass
        if xmin == float("inf"):
            return None
        return (xmin, ymin, zmin, xmax, ymax, zmax)

    @staticmethod
    def _nodes_in_bbox(
        node_tags: np.ndarray,
        node_coords: np.ndarray,
        bbox: tuple[float, float, float, float, float, float] | None,
    ) -> set[int]:
        """Return node tags inside a bounding box (with tolerance)."""
        if bbox is None or len(node_tags) == 0:
            return set(int(t) for t in node_tags)
        mins = np.array(bbox[:3], dtype=float)
        maxs = np.array(bbox[3:], dtype=float)
        span = max(float((maxs - mins).max()), 1.0)
        tol = 1e-6 * span
        mask = np.all(
            (node_coords >= (mins - tol)) & (node_coords <= (maxs + tol)),
            axis=1,
        )
        return set(int(t) for t in node_tags[mask])

    def _get_nodes_for_entities(
        self,
        entities: list[DimTag] | None,
    ) -> set[int]:
        """Collect mesh node tags for the given geometric entities."""
        if not entities:
            return set()
        tags: set[int] = set()
        for dim, tag in entities:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    dim=int(dim), tag=int(tag),
                    includeBoundary=True,
                    returnParametricCoord=False,
                )
                tags.update(int(t) for t in nt)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Could not extract nodes for entity "
                    f"({dim}, {tag}): {exc}",
                    stacklevel=2,
                )
        return tags

    def _collect_surface_faces(
        self,
        entities: list[DimTag] | None = None,
    ) -> np.ndarray:
        """Collect surface element connectivity as a rectangular array."""
        if entities is None:
            surface_ents = list(gmsh.model.getEntities(2))
        else:
            surface_ents = []
            for dim, tag in entities:
                if dim == 2:
                    surface_ents.append((2, tag))
                elif dim == 3:
                    for bd, bt in gmsh.model.getBoundary(
                        [(dim, tag)], oriented=False,
                    ):
                        if bd == 2:
                            surface_ents.append((bd, bt))

        blocks: list[np.ndarray] = []
        npe: int | None = None
        for _, tag in surface_ents:
            etypes, _, node_tags = gmsh.model.mesh.getElements(dim=2, tag=tag)
            for etype, enodes in zip(etypes, node_tags):
                if len(enodes) == 0:
                    continue
                _, _, _, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
                if npe is None:
                    npe = int(n_nodes)
                elif npe != int(n_nodes):
                    raise ValueError(
                        "Mixed surface element types not supported in "
                        "automatic face extraction."
                    )
                blocks.append(np.array(enodes, dtype=int).reshape(-1, int(n_nodes)))

        if not blocks:
            return np.empty((0, 0), dtype=int)
        return np.vstack(blocks)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._instances)
        return f"<PartsRegistry {n} instance{'s' if n != 1 else ''}>"
