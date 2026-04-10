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

    g = pyGmsh("bridge")
    g.begin()

    with g.parts.part("beam"):
        g.model.add_box(0, 0, 0, 1, 0.5, 10)

    g.parts.import_step("slab.step", label="slab", translate=(0, 0, 10))
    g.parts.fragment_all()

    # Build node map for constraint resolution
    fem = g.mesh.get_fem_data(dim=3)
    nm  = g.parts.build_node_map(fem.node_ids, fem.node_coords)
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase
    from pyGmsh.core.Part import Part

DimTag = tuple[int, int]


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
    """
    label: str
    part_name: str
    file_path: Path | None = None
    entities: dict[int, list[int]] = field(default_factory=dict)
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate: tuple[float, ...] | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    bbox: tuple[float, float, float, float, float, float] | None = None


# ---------------------------------------------------------------------------
# PartsRegistry composite
# ---------------------------------------------------------------------------

class PartsRegistry:
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
    # Entry point 1: Context manager (inline geometry)
    # ------------------------------------------------------------------

    @contextmanager
    def part(self, label: str):
        """Track entities created inside the block as a named part.

        Yields the label string.  After the block, any entities that
        exist now but didn't before are stored as an Instance.

        Example::

            with g.parts.part("beam"):
                g.model.add_box(0, 0, 0, 1, 0.5, 10)
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
        self._instances[label] = inst

    # ------------------------------------------------------------------
    # Entry point 2: Manual registration
    # ------------------------------------------------------------------

    def register(self, label: str, dimtags: list[DimTag]) -> Instance:
        """Tag existing entities under a part label.

        Parameters
        ----------
        label : str
            Unique part name.
        dimtags : list of (dim, tag)
            Entities to assign.

        Returns
        -------
        Instance
        """
        if label in self._instances:
            raise ValueError(f"Part label '{label}' already exists.")

        # Ownership check — each entity can belong to at most one part
        for dim, tag in dimtags:
            for existing_label, existing_inst in self._instances.items():
                if int(tag) in existing_inst.entities.get(int(dim), []):
                    raise ValueError(
                        f"Entity (dim={dim}, tag={tag}) already belongs to "
                        f"part '{existing_label}'. Remove it first."
                    )

        entities: dict[int, list[int]] = {}
        for dim, tag in dimtags:
            entities.setdefault(int(dim), []).append(int(tag))

        inst = Instance(
            label=label,
            part_name=label,
            entities=entities,
            bbox=self._compute_bbox(dimtags) if dimtags else None,
        )
        self._instances[label] = inst
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

        Useful after ``g.model.load_step()`` or ``g.model.load_iges()``
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
            g.model.load_step("bracket.step")
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
        self._instances[label] = inst
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
            raise FileNotFoundError(
                f"Part '{part.name}' has not been saved.  "
                f"Call part.save('file.step') first."
            )
        if label is None:
            self._counter += 1
            label = f"{part.name}_{self._counter}"
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

    def fragment_all(self, *, dim: int | None = None) -> list[int]:
        """Fragment all entities so interfaces become conformal.

        Updates each Instance.entities in-place with post-fragment tags.

        Parameters
        ----------
        dim : int, optional
            Target dimension.  Auto-detects highest present if None.

        Returns
        -------
        list[int]
            Tags of all surviving entities at the target dimension.
        """
        if dim is None:
            for d in (3, 2, 1):
                if gmsh.model.getEntities(d):
                    dim = d
                    break
            else:
                raise RuntimeError("No entities found.")

        all_ents = gmsh.model.getEntities(dim)

        # Warn about untracked entities
        tracked = set()
        for inst in self._instances.values():
            for t in inst.entities.get(dim, []):
                tracked.add(t)
        all_tags = set(t for _, t in all_ents)
        orphans = all_tags - tracked
        if orphans:
            import warnings
            warnings.warn(
                f"{len(orphans)} entities at dim={dim} are not tracked "
                f"by any part (tags: {sorted(orphans)}).  They will "
                f"participate in fragmentation but won't be remapped.  "
                f"Use g.parts.register() or g.parts.from_model() to "
                f"adopt them.",
                stacklevel=2,
            )

        if len(all_ents) < 2:
            return [t for _, t in all_ents]

        obj = [all_ents[0]]
        tool = list(all_ents[1:])
        input_ents = obj + tool

        result, result_map = gmsh.model.occ.fragment(
            obj, tool, removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()

        # old-tag → new-tags mapping
        old_to_new: dict[int, list[int]] = {}
        for old_dt, new_dts in zip(input_ents, result_map):
            old_dim, old_tag = old_dt
            if old_dim == dim:
                old_to_new[old_tag] = [t for d, t in new_dts if d == dim]

        # Update instance entities in-place
        for inst in self._instances.values():
            old_tags = inst.entities.get(dim, [])
            new_tags: list[int] = []
            for ot in old_tags:
                new_tags.extend(old_to_new.get(ot, [ot]))
            inst.entities[dim] = new_tags

        return [t for _, t in result]

    def fragment_pair(
        self,
        label_a: str,
        label_b: str,
        *,
        dim: int | None = None,
    ) -> list[int]:
        """Fragment only two instances against each other.

        Returns
        -------
        list[int]
            Surviving entity tags at the target dimension.
        """
        inst_a = self._instances[label_a]
        inst_b = self._instances[label_b]

        if dim is None:
            for d in (3, 2, 1):
                if d in inst_a.entities and d in inst_b.entities:
                    dim = d
                    break
            else:
                raise RuntimeError(
                    f"No common dimension between '{label_a}' and '{label_b}'."
                )

        obj = [(dim, t) for t in inst_a.entities.get(dim, [])]
        tool = [(dim, t) for t in inst_b.entities.get(dim, [])]

        result, _ = gmsh.model.occ.fragment(
            obj, tool, removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()
        return [t for _, t in result]

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
    # Convenience
    # ------------------------------------------------------------------

    def add_physical_groups(self, dim: int | None = None) -> dict[str, int]:
        """Create one physical group per instance.

        Returns ``{label: pg_tag}``.
        """
        if dim is None:
            for d in (3, 2, 1, 0):
                if any(d in inst.entities for inst in self._instances.values()):
                    dim = d
                    break
            else:
                raise RuntimeError("No entities in any instance.")

        groups: dict[str, int] = {}
        for label, inst in self._instances.items():
            tags = inst.entities.get(dim, [])
            if tags:
                pg = gmsh.model.addPhysicalGroup(dim, tags, name=label)
                groups[label] = pg
        return groups

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, label: str) -> Instance:
        """Return an Instance by label."""
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

        entities: dict[int, list[int]] = {}
        for dim, tag in raw:
            entities.setdefault(dim, []).append(tag)

        dimtags = [(d, t) for d, tags in entities.items() for t in tags]
        self._apply_transforms(dimtags, translate, rotate)
        dx, dy, dz = translate

        inst = Instance(
            label=label,
            part_name=part_name,
            file_path=file_path.resolve() if isinstance(file_path, Path) else file_path,
            entities=entities,
            translate=(dx, dy, dz),
            rotate=rotate,
            properties=properties or {},
            bbox=self._compute_bbox(dimtags),
        )
        self._instances[label] = inst
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
                xmin = min(xmin, bb[0]); ymin = min(ymin, bb[1]); zmin = min(zmin, bb[2])
                xmax = max(xmax, bb[3]); ymax = max(ymax, bb[4]); zmax = max(zmax, bb[5])
            except Exception:
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
            except Exception:
                pass
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
