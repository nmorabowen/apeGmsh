# MeshSelectionSet Implementation Plan

## Overview

A new post-mesh selection system complementary to `PhysicalGroups`, using the same `(dim, tag) + name` identity contract. Lives as a top-level composite `g.mesh_selection` alongside `g.physical`.

## Design Decisions

- **Identity**: `(dim, tag)` + optional name — mirrors `PhysicalGroups`
  - `dim=0` → node set
  - `dim=1` → 1D element set (line elements)
  - `dim=2` → 2D element set (tri/quad)
  - `dim=3` → 3D element set (tet/hex)
- **Storage**: Immutable numpy array snapshots (like `PhysicalGroupSet`)
- **Namespace**: Independent from `PhysicalGroups` (no collision risk)
- **API verbs**: Mirror `PhysicalGroups` — `add`, `get_nodes`, `get_elements`, `get_name`, `get_tag`, `summary`
- **FEMData integration**: `FEMData` carries both `fem.nodes.physical` (PhysicalGroupSet) and `fem.mesh_selection` (MeshSelectionStore) as parallel snapshots

## Architecture

```
PRE-MESH                              POST-MESH
  g.physical ─────────────────────┐
  (dim, tag) + name               │
  add_surface([s1], name="slab")  │
                                  ▼
  g.model.selection ──────> g.mesh.generation.generate() ──────> g.mesh_selection
  geometry queries            ▲                       (dim, tag) + name
  .to_physical()              │                       add_nodes(on_plane=...)
                              │                       add_elements(in_box=...)
                              │                       .from_physical("slab")
                              │
                         FEMData (broker)
                         ├── fem.physical      → PhysicalGroupSet (snapshot)
                         └── fem.mesh_selection → MeshSelectionStore (snapshot)
                              │
                              │  Same output shape:
                              │  get_nodes()    → {'tags', 'coords'}
                              │  get_elements() → {'element_ids', 'connectivity'}
                              │
                         g.opensees / any solver
```

## New Files

### 1. `src/pyGmsh/mesh/MeshSelectionSet.py` (~600-800 lines)

Three classes:

#### `MeshSelectionSet` — Composite (lives on `g.mesh_selection`)

```python
class MeshSelectionSet:
    """
    Post-mesh selection composite — complementary to PhysicalGroups.

    Uses (dim, tag) + name identity, same as PhysicalGroups:
      dim=0 → node set
      dim=1 → 1D element set
      dim=2 → 2D element set
      dim=3 → 3D element set
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent
        self._sets: dict[tuple[int, int], dict] = {}
        # Internal: {(dim, tag): {'name': str,
        #                         'node_ids': ndarray,
        #                         'node_coords': ndarray,
        #                         'element_ids': ndarray (optional),
        #                         'connectivity': ndarray (optional)}}

    # ── Creation (mirrors g.physical.add) ──────────────────────

    def add(self, dim: int, tags: list[int], *, name: str = "", tag: int = -1) -> int:
        """Add a mesh selection set from explicit node/element tags.
        dim=0: tags are node IDs. dim>=1: tags are element IDs."""

    def add_nodes(self, *, name: str = "", tag: int = -1, **filters) -> int:
        """Create a node set (dim=0) from spatial queries.
        Filters: on_plane, in_box, in_sphere, nearest_to, predicate."""

    def add_elements(self, dim: int = 2, *, name: str = "", tag: int = -1, **filters) -> int:
        """Create an element set from spatial queries.
        Filters: in_box, on_plane, by_type, predicate."""

    # ── Naming (mirrors g.physical) ────────────────────────────

    def set_name(self, dim: int, tag: int, name: str) -> MeshSelectionSet: ...
    def remove_name(self, name: str) -> MeshSelectionSet: ...

    # ── Removal ────────────────────────────────────────────────

    def remove(self, dim_tags: list[DimTag]) -> MeshSelectionSet: ...
    def remove_all(self) -> MeshSelectionSet: ...

    # ── Queries (mirrors g.physical) ───────────────────────────

    def get_all(self, dim: int = -1) -> list[DimTag]: ...
    def get_entities(self, dim: int, tag: int) -> list[int]: ...
        # Returns node tags (dim=0) or element tags (dim>=1)
    def get_name(self, dim: int, tag: int) -> str: ...
    def get_tag(self, dim: int, name: str) -> int | None: ...
    def summary(self) -> pd.DataFrame: ...

    # ── Mesh data (same output as PhysicalGroups) ──────────────

    def get_nodes(self, dim: int, tag: int) -> dict:
        """{'tags': ndarray(N,), 'coords': ndarray(N, 3)}"""

    def get_elements(self, dim: int, tag: int) -> dict:
        """{'element_ids': ndarray(E,), 'connectivity': ndarray(E, npe)}"""

    # ── Set algebra ────────────────────────────────────────────

    def union(self, dim: int, tag_a: int, tag_b: int, *,
              name: str = "", tag: int = -1) -> int: ...
    def intersection(self, dim: int, tag_a: int, tag_b: int, *,
                     name: str = "", tag: int = -1) -> int: ...
    def difference(self, dim: int, tag_a: int, tag_b: int, *,
                   name: str = "", tag: int = -1) -> int: ...

    # ── Bridges ────────────────────────────────────────────────

    def from_physical(self, dim: int, name_or_tag: str | int, *,
                      ms_name: str = "", ms_tag: int = -1) -> int:
        """Import a physical group as a mesh selection set."""

    def to_physical(self, dim: int, tag: int, *,
                    pg_name: str = "") -> int:
        """Promote a mesh selection (dim>=1 element set) to a physical group.
        Only works if elements map back to geometric entities."""

    # ── Snapshot (for FEMData) ─────────────────────────────────

    def _snapshot(self) -> MeshSelectionStore:
        """Return an immutable copy of all sets for FEMData."""
```

#### `MeshSelectionStore` — Immutable Snapshot (lives on `fem.mesh_selection`)

Mirrors `PhysicalGroupSet` exactly:

```python
class MeshSelectionStore:
    """Snapshot of mesh selections captured at get_fem_data() time.
    Accessed via fem.mesh_selection. Same API as PhysicalGroupSet."""

    def __init__(self, sets: dict[tuple[int, int], dict]) -> None: ...

    def get_all(self, dim: int = -1) -> list[DimTag]: ...
    def get_name(self, dim: int, tag: int) -> str: ...
    def get_tag(self, dim: int, name: str) -> int | None: ...
    def get_nodes(self, dim: int, tag: int) -> dict: ...
    def get_elements(self, dim: int, tag: int) -> dict: ...
    def summary(self) -> pd.DataFrame: ...
```

### 2. `src/pyGmsh/mesh/_mesh_filters.py` (~300-400 lines)

Spatial filter engine — pure functions, no Gmsh dependency:

```python
def nodes_on_plane(node_ids, node_coords, axis, value, atol) -> mask
def nodes_in_box(node_ids, node_coords, bbox) -> mask
def nodes_in_sphere(node_ids, node_coords, center, radius) -> mask
def nodes_nearest(node_ids, node_coords, point, count) -> mask
def elements_in_box(elem_ids, connectivity, node_coords, bbox) -> mask
    # by centroid
def elements_on_plane(elem_ids, connectivity, node_coords, axis, value, atol) -> mask
    # all nodes on plane
def element_centroids(connectivity, node_coords) -> ndarray(E, 3)
def boundary_nodes_of(elem_ids, connectivity) -> ndarray
    # nodes that appear in only one element (boundary detection)
```

## Modified Files

### 3. `src/pyGmsh/_core.py` — Register composite

```python
_COMPOSITES = (
    ...existing...
    ("mesh_selection", ".mesh.MeshSelectionSet", "MeshSelectionSet", False),  # ADD
)

# Static type declaration
mesh_selection: MeshSelectionSet  # ADD
```

### 4. `src/pyGmsh/mesh/FEMData.py` — Add mesh_selection to FEMData

```python
@dataclass
class FEMData:
    ...existing fields...
    mesh_selection: MeshSelectionStore = field(repr=False, default_factory=lambda: MeshSelectionStore({}))
```

### 5. `src/pyGmsh/mesh/_fem_extract.py` — Snapshot mesh selections during build

In `build_fem_data()`, after building `PhysicalGroupSet`, also snapshot the mesh selections:

```python
def build_fem_data(dim=2, mesh_selection_composite=None):
    ...existing...
    physical = PhysicalGroupSet(extract_physical_groups())

    # Snapshot mesh selections if available
    ms_store = MeshSelectionStore({})
    if mesh_selection_composite is not None:
        ms_store = mesh_selection_composite._snapshot()

    return FEMData(..., mesh_selection=ms_store)
```

### 6. `src/pyGmsh/mesh/__init__.py` — Export new classes

```python
from .MeshSelectionSet import MeshSelectionSet, MeshSelectionStore
```

## Usage Examples

```python
g = apeGmsh(model_name="bridge")
g.begin()

# Geometry + physical groups (pre-mesh, as today)
slab = g.model.geometry.add_rectangle(0, 0, 0, 20, 10)
g.physical.add_surface([slab], name="slab")
g.physical.add_curve([1, 3], name="supports")

# Mesh
g.mesh.generation.generate(2)

# Mesh selections (post-mesh)
g.mesh_selection.add_nodes(on_plane=("z", 0.0, 1e-3), name="base_nodes")
g.mesh_selection.add_nodes(nearest_to=(10.0, 5.0, 0.0), count=1, name="load_point")
g.mesh_selection.add_elements(dim=2, in_box=[5, 2, -1, 15, 8, 1], name="center_zone")

# Explicit tag-based
g.mesh_selection.add(dim=0, tags=[101, 102, 103], name="sensor_nodes")

# Set algebra
g.mesh_selection.difference(
    dim=0, tag_a=1, tag_b=2,  # base_nodes minus load_point
    name="distributed_base"
)

# Bridge from physical
g.mesh_selection.from_physical(dim=1, name_or_tag="supports", ms_name="support_nodes")

# Query — identical shape to g.physical.get_nodes()
nodes = g.mesh_selection.get_nodes(dim=0, tag=1)
# → {'tags': ndarray(N,), 'coords': ndarray(N, 3)}

elems = g.mesh_selection.get_elements(dim=2, tag=3)
# → {'element_ids': ndarray(E,), 'connectivity': ndarray(E, npe)}

# FEM broker — both sources available
fem = g.mesh.queries.get_fem_data(dim=2)
fem.physical.get_nodes(0, 1)          # from physical groups
fem.mesh_selection.get_nodes(0, 1)    # from mesh selections

# OpenSees — consumes either
g.opensees.elements.fix(source="physical", name="supports", dofs=[1,1,1,0,0,0])
g.opensees.elements.fix(source="mesh_selection", name="base_nodes", dofs=[1,1,1,0,0,0])

g.mesh_selection.summary()
# DataFrame indexed by (dim, tag):
#           name              n_nodes  n_elems
# (0, 1)   base_nodes         42       0
# (0, 2)   load_point          1       0
# (0, 3)   sensor_nodes        3       0
# (2, 1)   center_zone         0      28
```

## Implementation Sequence

1. **`_mesh_filters.py`** — pure spatial filter functions (testable without Gmsh)
2. **`MeshSelectionSet.py`** — `MeshSelectionSet` + `MeshSelectionStore` classes
3. **`_core.py`** — register `g.mesh_selection` composite
4. **`FEMData.py`** — add `mesh_selection: MeshSelectionStore` field
5. **`_fem_extract.py`** — snapshot mesh selections into `FEMData`
6. **`__init__.py`** — export new classes
7. **Tests** — spatial queries, set algebra, snapshot, output shape parity
8. **Viewer integration** (optional) — highlight selections in `g.view` / `g.plot`

## Key Invariants

- `g.mesh_selection.get_nodes()` returns **exactly the same dict shape** as `g.physical.get_nodes()`
- `g.mesh_selection.get_elements()` returns **exactly the same dict shape** as `g.physical.get_elements()`
- `MeshSelectionStore` mirrors `PhysicalGroupSet` API 1:1
- Both live on `FEMData` as parallel attributes: `fem.physical` and `fem.mesh_selection`
- `(dim, tag)` identity is independent between the two systems — no collision
- All stored arrays are immutable snapshots
