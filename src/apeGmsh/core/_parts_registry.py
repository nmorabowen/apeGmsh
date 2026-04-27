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
    edit: "InstanceEdit | None" = field(init=False, repr=False, default=None)

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
            resolved = [(int(d), int(t))
                        for d, t in self._parent.physical.dim_tags(pg)]
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
        properties: dict[str, Any] | None = None,
    ) -> Instance:
        """Import CAD geometry, apply transforms, store instance."""
        if label in self._instances:
            raise ValueError(f"Part label '{label}' already exists.")

        raw = gmsh.model.occ.importShapes(
            str(file_path), highestDimOnly=highest_dim_only,
        )
        gmsh.model.occ.synchronize()

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
