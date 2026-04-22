# pyGmsh v2 — Final Implementation Plan

## Architecture Principles

1. **pyGmsh IS the assembly.** Single runtime, all features always available.
2. **Part is optional.** Standalone geometry builder with its own Gmsh session. Exports STEP.
3. **Named sets at every level.** Parts (semantic), physical groups (geometry), mesh selections (mesh) — all produce the same data shapes.
4. **One entity tracking system.** Model's `_registry` is the single source of truth. No proxy classes, no parallel tracking.
5. **Multiple definition methods.** Context manager, explicit registration, file import, Part import — all feed the same underlying data structure.
6. **Constraints read from any named set.** Physical groups, mesh selections, or part labels — the resolver doesn't care.
7. **Assembly deleted.** No backward compatibility.

## Target Composite Layout

```python
g = apeGmsh()
g.begin()

# Geometry
g.model          # OCC geometry API + _registry (unchanged)
g.inspect        # geometry introspection (unchanged)

# Parts (NEW — replaces Assembly instance management)
g.parts          # label -> entity_tags mapping
                 # .import_step(), .add(), .register()
                 # .fragment_all(), .fragment_pair()
                 # with g.model.part("col"): context manager

# Mesh
g.mesh           # generation, sizing, renumbering (unchanged)
g.physical       # pre-mesh named sets on geometry (unchanged)
g.mesh_selection # post-mesh named sets on nodes/elements (NEW)

# Constraints
g.constraints    # define + resolve kinematic interactions (NEW)
                 # reads from physical, mesh_selection, or parts
                 # can create new nodes/elements

# Solvers & Export
g.opensees       # OpenSees builder (unchanged API)
g.g2o            # gmsh2opensees bridge (unchanged)
g.partition      # Metis partitioning (unchanged)
g.view           # Gmsh post-processing views (unchanged)
g.plot           # Matplotlib visualization (unchanged)
g.loader         # .msh file loading (unchanged)

# FEM Data Broker
fem = g.mesh.queries.get_fem_data(dim=2)
fem.physical         # PhysicalGroupSet snapshot (unchanged)
fem.mesh_selection   # MeshSelectionStore snapshot (NEW)
fem.constraints      # ConstraintStore snapshot (NEW, Phase 4)
```

---

## Phase 1: MeshSelectionSet

**Goal:** Add `g.mesh_selection` without touching existing code.

### New Files

#### `src/pyGmsh/mesh/_mesh_filters.py` (~300 lines)

Pure spatial query functions. No Gmsh dependency. Fully testable in isolation.

```python
"""Pure spatial filters operating on numpy arrays."""

import numpy as np
from numpy import ndarray

BBox = tuple[float, float, float, float, float, float]

def nodes_on_plane(
    node_ids: ndarray, node_coords: ndarray,
    axis: str, value: float, atol: float,
) -> ndarray:
    """Boolean mask: nodes within atol of the plane axis=value."""

def nodes_in_box(
    node_ids: ndarray, node_coords: ndarray,
    bbox: BBox,
) -> ndarray:
    """Boolean mask: nodes inside bounding box [xmin,ymin,zmin,xmax,ymax,zmax]."""

def nodes_in_sphere(
    node_ids: ndarray, node_coords: ndarray,
    center: tuple[float,float,float], radius: float,
) -> ndarray:
    """Boolean mask: nodes within radius of center."""

def nodes_nearest(
    node_ids: ndarray, node_coords: ndarray,
    point: tuple[float,float,float], count: int,
) -> ndarray:
    """Indices of the count nearest nodes to point."""

def elements_in_box(
    elem_ids: ndarray, connectivity: ndarray,
    node_id_to_coord: dict[int, ndarray],
    bbox: BBox,
) -> ndarray:
    """Boolean mask: elements whose centroid falls inside bbox."""

def elements_on_plane(
    elem_ids: ndarray, connectivity: ndarray,
    node_id_to_coord: dict[int, ndarray],
    axis: str, value: float, atol: float,
) -> ndarray:
    """Boolean mask: elements with ALL nodes within atol of plane."""

def element_centroids(
    connectivity: ndarray,
    node_id_to_coord: dict[int, ndarray],
) -> ndarray:
    """Compute centroid (mean of node coords) per element. Returns (E, 3)."""

def boundary_nodes_of(connectivity: ndarray) -> ndarray:
    """Node tags that appear on the boundary of the element set.
    Boundary = nodes belonging to fewer elements than interior nodes.
    Implementation: count node occurrences, boundary nodes appear in
    only one element along at least one face."""
```

#### `src/pyGmsh/mesh/MeshSelectionSet.py` (~600 lines)

Two classes: live composite + immutable snapshot.

```python
"""Post-mesh selection sets — complementary to PhysicalGroups."""

from __future__ import annotations
from typing import TYPE_CHECKING
import gmsh
import numpy as np
import pandas as pd
from numpy import ndarray
from . import _mesh_filters as mf

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

Tag    = int
DimTag = tuple[int, int]


class MeshSelectionSet:
    """
    Post-mesh selection composite — g.mesh_selection.

    Identity: (dim, tag) + optional name, same as PhysicalGroups.
        dim=0 -> node set
        dim=1 -> 1D element set
        dim=2 -> 2D element set
        dim=3 -> 3D element set

    All mutating methods return self for chaining.
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent
        # {(dim, tag): {'name': str,
        #               'node_ids': ndarray,
        #               'node_coords': ndarray,
        #               'element_ids': ndarray | None,
        #               'connectivity': ndarray | None}}
        self._sets: dict[DimTag, dict] = {}
        self._next_tag: dict[int, int] = {0: 1, 1: 1, 2: 1, 3: 1}

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[MeshSelection] {msg}")

    def _auto_tag(self, dim: int) -> int:
        tag = self._next_tag[dim]
        self._next_tag[dim] = tag + 1
        return tag

    def _get_all_nodes(self) -> tuple[ndarray, ndarray]:
        """Fetch all mesh nodes from live Gmsh session."""
        raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
        return (
            np.array(raw_tags, dtype=np.int64),
            np.array(raw_coords).reshape(-1, 3),
        )

    def _get_all_elements(self, dim: int) -> tuple[ndarray, ndarray, ndarray]:
        """Fetch all elements of given dimension from live Gmsh session."""
        etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(dim=dim)
        # ... flatten into (elem_ids, connectivity, elem_types)
        # same pattern as _fem_extract.extract_raw()

    # ── Creation ─────────────────────────────────────────

    def add(
        self, dim: int, tags: list[int], *,
        name: str = "", tag: int = -1,
    ) -> Tag:
        """Add a selection set from explicit node/element tags.
        dim=0: tags are node IDs. dim>=1: tags are element IDs."""
        assigned_tag = tag if tag > 0 else self._auto_tag(dim)
        # Fetch coords for nodes, or connectivity for elements
        # Store in self._sets[(dim, assigned_tag)]
        if name:
            self._sets[(dim, assigned_tag)]['name'] = name
        self._log(f"add(dim={dim}, n_tags={len(tags)}) -> tag={assigned_tag}")
        return assigned_tag

    def add_nodes(
        self, *, name: str = "", tag: int = -1,
        on_plane: tuple[str, float, float] | None = None,
        in_box: tuple[float,...] | None = None,
        in_sphere: tuple[tuple[float,float,float], float] | None = None,
        nearest_to: tuple[float, float, float] | None = None,
        count: int = 1,
        predicate = None,
    ) -> Tag:
        """Create a node set (dim=0) from spatial queries.
        Filters are AND-combined."""

    def add_elements(
        self, dim: int = 2, *, name: str = "", tag: int = -1,
        in_box: tuple[float,...] | None = None,
        on_plane: tuple[str, float, float] | None = None,
        by_type: int | None = None,
        predicate = None,
    ) -> Tag:
        """Create an element set from spatial queries."""

    # ── Naming ───────────────────────────────────────────

    def set_name(self, dim: int, tag: int, name: str) -> MeshSelectionSet:
        ...
        return self

    def remove_name(self, name: str) -> MeshSelectionSet:
        ...
        return self

    # ── Removal ──────────────────────────────────────────

    def remove(self, dim_tags: list[DimTag]) -> MeshSelectionSet:
        for dt in dim_tags:
            self._sets.pop(dt, None)
        return self

    def remove_all(self) -> MeshSelectionSet:
        self._sets.clear()
        return self

    # ── Queries ──────────────────────────────────────────

    def get_all(self, dim: int = -1) -> list[DimTag]:
        if dim == -1:
            return sorted(self._sets.keys())
        return sorted(k for k in self._sets if k[0] == dim)

    def get_entities(self, dim: int, tag: int) -> list[int]:
        """Returns node tags (dim=0) or element tags (dim>=1)."""
        info = self._sets[(dim, tag)]
        if dim == 0:
            return info['node_ids'].tolist()
        return info['element_ids'].tolist()

    def get_name(self, dim: int, tag: int) -> str:
        return self._sets[(dim, tag)].get('name', '')

    def get_tag(self, dim: int, name: str) -> int | None:
        for (d, t), info in self._sets.items():
            if d == dim and info.get('name', '') == name:
                return t
        return None

    def summary(self) -> pd.DataFrame:
        """DataFrame indexed by (dim, tag) with name, n_nodes, n_elements."""

    # ── Mesh data (same shape as PhysicalGroups) ─────────

    def get_nodes(self, dim: int, tag: int) -> dict:
        """{'tags': ndarray(N,), 'coords': ndarray(N, 3)}"""
        info = self._sets[(dim, tag)]
        return {
            'tags': np.asarray(info['node_ids']).astype(object),
            'coords': np.asarray(info['node_coords'], dtype=np.float64),
        }

    def get_elements(self, dim: int, tag: int) -> dict:
        """{'element_ids': ndarray(E,), 'connectivity': ndarray(E, npe)}"""
        info = self._sets[(dim, tag)]
        return {
            'element_ids': np.asarray(info['element_ids']).astype(object),
            'connectivity': np.asarray(info['connectivity']).astype(object),
        }

    # ── Set algebra ──────────────────────────────────────

    def union(self, dim: int, tag_a: int, tag_b: int, *,
              name: str = "", tag: int = -1) -> int:
        """A | B -> new set."""

    def intersection(self, dim: int, tag_a: int, tag_b: int, *,
                     name: str = "", tag: int = -1) -> int:
        """A & B -> new set."""

    def difference(self, dim: int, tag_a: int, tag_b: int, *,
                   name: str = "", tag: int = -1) -> int:
        """A - B -> new set."""

    # ── Bridge ───────────────────────────────────────────

    def from_physical(self, dim: int, name_or_tag,
                      *, ms_name: str = "", ms_tag: int = -1) -> int:
        """Import a physical group as a mesh selection set."""

    # ── Snapshot ─────────────────────────────────────────

    def _snapshot(self) -> MeshSelectionStore:
        """Deep copy of all sets as an immutable store for FEMData."""
        import copy
        return MeshSelectionStore(copy.deepcopy(self._sets))

    # ── Dunder ───────────────────────────────────────────

    def __repr__(self) -> str:
        return f"MeshSelectionSet(n_sets={len(self._sets)})"


class MeshSelectionStore:
    """
    Immutable snapshot of mesh selections for FEMData.
    Mirrors PhysicalGroupSet API exactly.
    """

    def __init__(self, sets: dict[DimTag, dict]) -> None:
        self._sets = sets

    def get_all(self, dim: int = -1) -> list[DimTag]: ...
    def get_name(self, dim: int, tag: int) -> str: ...
    def get_tag(self, dim: int, name: str) -> int | None: ...
    def get_nodes(self, dim: int, tag: int) -> dict: ...
    def get_elements(self, dim: int, tag: int) -> dict: ...
    def summary(self) -> pd.DataFrame: ...

    def __len__(self) -> int:
        return len(self._sets)

    def __repr__(self) -> str:
        return f"MeshSelectionStore({len(self._sets)} sets)"
```

### Modified Files

#### `src/pyGmsh/_core.py`
```python
# Already has mesh_selection in _COMPOSITES (previously added).
# Verify it's wired and add static type declaration.
```

#### `src/pyGmsh/mesh/FEMData.py`
```python
# Add field to FEMData dataclass:
mesh_selection: MeshSelectionStore = field(
    repr=False,
    default_factory=lambda: MeshSelectionStore({}),
)
```

#### `src/pyGmsh/mesh/_fem_extract.py`
```python
# Modify build_fem_data signature:
def build_fem_data(dim=2, *, mesh_selection_composite=None):
    ...existing code...

    # After physical group extraction:
    ms_store = MeshSelectionStore({})
    if mesh_selection_composite is not None:
        ms_store = mesh_selection_composite._snapshot()

    return FEMData(..., mesh_selection=ms_store)
```

#### `src/pyGmsh/mesh/Mesh.py`
```python
# In get_fem_data(), pass mesh_selection composite:
def get_fem_data(self, dim=2):
    ms_composite = getattr(self._parent, 'mesh_selection', None)
    return _fem_extract.build_fem_data(dim=dim, mesh_selection_composite=ms_composite)
```

#### `src/pyGmsh/mesh/__init__.py`
```python
from .MeshSelectionSet import MeshSelectionSet, MeshSelectionStore
# Add to __all__
```

### Tests

```python
# tests/test_mesh_selection_set.py

# 1. _mesh_filters tests (pure numpy, no Gmsh)
#    - nodes_on_plane with known geometry
#    - nodes_in_box boundary conditions
#    - nodes_nearest with ties
#    - element_centroids correctness

# 2. MeshSelectionSet integration tests (need Gmsh)
#    - add() with explicit tags
#    - add_nodes() with spatial queries
#    - add_elements() with spatial queries
#    - get_nodes() / get_elements() shape matches PhysicalGroups
#    - set algebra (union, intersection, difference)
#    - from_physical() bridge
#    - summary() DataFrame structure
#    - name/tag lookup

# 3. MeshSelectionStore snapshot tests
#    - _snapshot() produces independent copy
#    - Snapshot API matches PhysicalGroupSet API
#    - FEMData.mesh_selection populated after get_fem_data()
```

### Validation Checklist
- [ ] All existing tests pass (zero changes to existing code paths)
- [ ] `g.mesh_selection.add_nodes()` returns same shape as `g.physical.get_nodes()`
- [ ] `g.mesh_selection.add_elements()` returns same shape as `g.physical.get_elements()`
- [ ] `fem.mesh_selection` populated after `g.mesh.queries.get_fem_data()`
- [ ] Set algebra produces correct results
- [ ] `from_physical()` bridge works

---

## Phase 2: Constraints Composite

**Goal:** Extract constraint factory methods from Assembly into standalone `g.constraints`. `ConstraintResolver` and all dataclasses in `solvers/Constraints.py` stay UNTOUCHED.

### New Files

#### `src/pyGmsh/core/ConstraintsComposite.py` (~400 lines)

```python
"""Constraint composite — define and resolve kinematic interactions."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from ..solvers.Constraints import (
    ConstraintDef, ConstraintRecord, ConstraintResolver,
    EqualDOFDef, RigidLinkDef, PenaltyDef,
    RigidDiaphragmDef, RigidBodyDef, KinematicCouplingDef,
    TieDef, DistributingCouplingDef, EmbeddedDef,
    TiedContactDef, MortarDef,
)

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

class ConstraintsComposite:
    """
    Constraint composite — g.constraints.

    Two-stage pipeline:
    1. Define intent (pre or post mesh): equal_dof(), tie(), rigid_link(), ...
    2. Resolve to concrete node pairs: resolve()

    Master/slave references can be:
    - Part label (str) -> resolved via g.parts
    - Physical group name (str) -> resolved via g.physical
    - Mesh selection name (str) -> resolved via g.mesh_selection
    - Explicit node set (dict) -> {'node_ids': ndarray, 'node_coords': ndarray}
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent
        self.defs: list[ConstraintDef] = []
        self.records: list[ConstraintRecord] = []

    # ── Node resolution ────────────────────────────────
    def _resolve_nodes(self, ref) -> set[int]:
        """Resolve a reference to a set of node tags.
        ref can be: str (part label, pg name, ms name) or dict."""
        # 1. Check g.parts (if exists)
        # 2. Check g.physical (by name)
        # 3. Check g.mesh_selection (by name)
        # 4. If dict, use directly

    def _resolve_faces(self, ref) -> np.ndarray:
        """Resolve a reference to face connectivity array."""

    # ── Level 1-4 factory methods ──────────────────────
    # (Same signatures as Assembly, but master/slave are
    #  flexible references instead of instance labels)

    def equal_dof(self, master, slave, *, dofs=None, ...) -> EqualDOFDef: ...
    def rigid_link(self, master, slave, *, link_type="beam", ...) -> RigidLinkDef: ...
    def penalty(self, master, slave, *, stiffness=1e10, ...) -> PenaltyDef: ...
    def rigid_diaphragm(self, master, slave, *, ...) -> RigidDiaphragmDef: ...
    def rigid_body(self, master, slave, *, ...) -> RigidBodyDef: ...
    def kinematic_coupling(self, master, slave, *, ...) -> KinematicCouplingDef: ...
    def tie(self, master, slave, *, ...) -> TieDef: ...
    def distributing_coupling(self, master, slave, *, ...) -> DistributingCouplingDef: ...
    def embedded(self, host, embedded, *, ...) -> EmbeddedDef: ...
    def tied_contact(self, master, slave, *, ...) -> TiedContactDef: ...
    def mortar(self, master, slave, *, ...) -> MortarDef: ...

    # ── Resolution ─────────────────────────────────────
    def resolve(
        self,
        node_tags=None, node_coords=None,
        elem_tags=None, connectivity=None,
    ) -> list[ConstraintRecord]:
        """Resolve all definitions to concrete records.
        If node_tags not provided, extracts from live Gmsh session."""
        # Uses ConstraintResolver (unchanged)
        # Resolution logic extracted from Assembly.resolve_constraints()

    # ── Queries ────────────────────────────────────────
    def list_defs(self) -> list[dict]: ...
    def list_records(self) -> list[dict]: ...
    def summary(self) -> pd.DataFrame: ...
```

### Modified Files

#### `src/pyGmsh/_core.py`
```python
# Add to _COMPOSITES:
("constraints", ".core.ConstraintsComposite", "ConstraintsComposite", False),
```

### Validation Checklist
- [ ] `solvers/Constraints.py` untouched (zero diff)
- [ ] All constraint types definable via `g.constraints`
- [ ] Resolution produces same records as Assembly did
- [ ] Master/slave resolution works with physical groups, mesh selections

---

## Phase 3: Parts Registry + Delete Assembly

**Goal:** Move instance management into pyGmsh. Delete Assembly.

### New Files

#### `src/pyGmsh/core/_parts_registry.py` (~350 lines)

```python
"""Parts registry — manages part instances within a pyGmsh session."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import gmsh
import numpy as np

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase
    from apeGmsh.core.Part import Part

@dataclass
class Instance:
    """Bookkeeping record for one imported part."""
    label: str
    part_name: str
    file_path: Path | None
    entities: dict[int, list[int]] = field(default_factory=dict)
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate: tuple[float, ...] | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    bbox: tuple[float, ...] | None = None


class PartsRegistry:
    """
    Parts registry composite — g.parts.

    Four ways to define parts (all feed the same _instances dict):

    A. Context manager:
        with g.model.part("column"):
            g.model.geometry.add_cylinder(...)

    B. Explicit registration:
        vol = g.model.geometry.add_box(...)
        g.parts.register("slab", [(3, vol)])

    C. Import from file:
        g.parts.import_step("column.step", label="col_1")

    D. Import from Part object:
        g.parts.add(col_part, label="col_1", translate=(0,0,0))
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent
        self._instances: dict[str, Instance] = {}
        self._counter: int = 0

    # ── A: Context manager on g.model ──────────────────
    @contextmanager
    def part(self, label: str):
        """Tag all entities created inside block with label.
        Usage: with g.parts.part("col"): g.model.geometry.add_cylinder(...)"""
        pre = set(gmsh.model.getEntities(-1))
        yield
        gmsh.model.occ.synchronize()
        post = set(gmsh.model.getEntities(-1))
        new_entities = post - pre
        entities: dict[int, list[int]] = {}
        for dim, tag in new_entities:
            entities.setdefault(dim, []).append(tag)
        self._register_instance(label, entities=entities)

    # ── B: Explicit registration ───────────────────────
    def register(self, label: str, dim_tags: list[tuple[int,int]]) -> Instance:
        """Associate existing entities with a part label."""
        entities: dict[int, list[int]] = {}
        for dim, tag in dim_tags:
            entities.setdefault(dim, []).append(tag)
        return self._register_instance(label, entities=entities)

    # ── C: Import from STEP/IGES file ──────────────────
    def import_step(
        self, path, *, label=None,
        translate=(0.0, 0.0, 0.0), rotate=None,
        highest_dim_only=True,
    ) -> Instance:
        """Import a CAD file as a named instance."""
        # Extracted from Assembly.add_file()
        # Same logic: importShapes -> rotate -> translate -> store

    # ── D: Import from Part object ─────────────────────
    def add(
        self, part: Part, *, label=None,
        translate=(0.0, 0.0, 0.0), rotate=None,
        highest_dim_only=True,
    ) -> Instance:
        """Import a Part object (must be saved to disk)."""
        # Extracted from Assembly.add_part()

    # ── Fragment ───────────────────────────────────────
    def fragment_all(self, *, dim=None) -> list[int]:
        """Fragment all entities for conformal mesh."""
        # Extracted from Assembly.fragment_all()
        # Updates self._instances entity tags after fragment

    def fragment_pair(self, label_a, label_b, *, dim=None) -> list[int]:
        """Fragment two specific instances."""
        # Extracted from Assembly.fragment_pair()

    # ── Instance queries ───────────────────────────────
    def get(self, label: str) -> Instance: ...
    def list(self) -> list[str]: ...
    def summary(self) -> pd.DataFrame: ...

    # ── Node/face mapping (used by constraints) ────────
    def build_node_map(self, node_tags, node_coords) -> dict[str, set[int]]:
        """Infer instance node ownership from bounding boxes."""
        # Extracted from Assembly._build_instance_node_map()

    def build_face_map(self, node_map) -> dict[str, np.ndarray]:
        """Infer per-instance surface faces."""
        # Extracted from Assembly._build_instance_face_map()

    # ── Convenience ────────────────────────────────────
    def add_physical_groups(self, dim=None) -> dict[str, int]:
        """Create one physical group per instance."""
        # Extracted from Assembly.add_physical_groups_from_instances()

    # ── Internal ───────────────────────────────────────
    def _register_instance(self, label, **kwargs) -> Instance:
        if label in self._instances:
            raise ValueError(f"Part label '{label}' already exists.")
        inst = Instance(label=label, **kwargs)
        self._instances[label] = inst
        return inst
```

### Modified Files

#### `src/pyGmsh/_core.py`
- Add `parts` composite (PartsRegistry)
- Add convenience delegates: `g.create_part()` -> `g.parts.part()` context manager
- Add `g.fragment()` -> `g.parts.fragment_all()`

#### `src/pyGmsh/__init__.py`
- Remove `from apeGmsh.core.Assembly import Assembly`
- Remove Assembly from `__all__`
- Add `Instance`, `PartsRegistry` exports

#### `src/pyGmsh/core/__init__.py`
- Remove Assembly import

### Deleted Files

#### `src/pyGmsh/core/Assembly.py` — DELETED

### Updated Files

#### `tests/test_library_contracts.py`
- Remove Assembly references, test pyGmsh directly

#### Example notebooks
- Update any Assembly usage to pyGmsh + g.parts

### Validation Checklist
- [ ] Assembly.py deleted
- [ ] `g.parts.import_step()` produces same Instance as Assembly.add_file()
- [ ] `g.parts.fragment_all()` produces same results as Assembly.fragment_all()
- [ ] Context manager tracks entities correctly
- [ ] Constraint resolution works with parts labels

---

## Phase 4: Constraint-Mesh Integration

**Goal:** Constraints can create new nodes/elements, auto-registered in mesh_selection.

### Changes

#### `src/pyGmsh/core/ConstraintsComposite.py`
- `resolve()` returns augmented data: original mesh + new nodes/elements
- New nodes get auto-registered as mesh selections:
  - `"constraint_{name}_nodes"` for duplicated nodes
  - `"constraint_{name}_elements"` for zero-length elements
- Node ID assignment: continues from max existing ID

#### `src/pyGmsh/mesh/FEMData.py`
- Add optional constraints snapshot

#### `src/pyGmsh/mesh/_fem_extract.py`
- `build_fem_data()` accepts constraint augmentation data
- Augmented arrays: original + constraint-created nodes/elements

### Validation Checklist
- [ ] Constraint-created nodes have valid contiguous IDs
- [ ] Created nodes appear in `fem.mesh_selection`
- [ ] FEMData arrays include augmented nodes/elements
- [ ] OpenSees can consume augmented FEMData

---

## Phase 5: Viewer Integration

**Goal:** Visualize mesh selections in g.view, g.plot, and interactive viewers.

### Changes
- `g.view.add_selection_highlight(name)` — Gmsh view colored by selection membership
- `g.plot.mesh()` supports `highlight_selection="name"` parameter
- Interactive viewers show selection sets as toggleable overlays

---

## Implementation Order & Dependencies

```
Phase 1: MeshSelectionSet
   |  standalone, zero risk
   v
Phase 2: Constraints composite
   |  needs Phase 1 for mesh_selection resolution
   v
Phase 3: Parts registry + delete Assembly
   |  needs Phase 2 (constraints moved out of Assembly)
   |  THIS IS THE ONLY BREAKING PHASE
   v
Phase 4: Constraint-mesh integration
   |  needs all three above
   v
Phase 5: Viewer integration
      needs Phase 1 (MeshSelectionSet)
```

Phase 5 can run in parallel with Phases 2-4 since it only depends on Phase 1.

## Files Summary

| File | Phase | Action | ~Lines |
|------|-------|--------|--------|
| `mesh/_mesh_filters.py` | 1 | NEW | 300 |
| `mesh/MeshSelectionSet.py` | 1 | NEW | 600 |
| `mesh/FEMData.py` | 1, 4 | MODIFY | +50 |
| `mesh/_fem_extract.py` | 1 | MODIFY | +20 |
| `mesh/Mesh.py` | 1 | MODIFY | +5 |
| `mesh/__init__.py` | 1 | MODIFY | +5 |
| `_core.py` | 1, 2, 3 | MODIFY | +50 |
| `core/ConstraintsComposite.py` | 2 | NEW | 400 |
| `core/_parts_registry.py` | 3 | NEW | 350 |
| `core/Assembly.py` | 3 | DELETE | -1244 |
| `core/__init__.py` | 3 | MODIFY | -2 |
| `__init__.py` | 3 | MODIFY | +5 |
| `solvers/Constraints.py` | — | UNTOUCHED | 0 |
| `core/Part.py` | — | UNTOUCHED | 0 |
| Tests | 1-4 | NEW | ~1000 |

**Net: ~2780 new, ~1244 deleted = ~1536 net growth**
