# pyGmsh v2 — Unified Architecture Implementation Plan

> **Status:** Delivered — v2 architecture (no Assembly class, parts registry, unified composites) is what shipped through v1.0+. Historical reference.

## Vision

pyGmsh becomes the single full-featured runtime. Whether you model one body or assemble ten, you use the same object with all composites available. Assembly is removed. Part stays as an optional standalone geometry builder.

## Current State

```
pyGmsh (_core.py)     → composites: model, mesh, physical, partition, view, opensees, plot, inspect, loader
Assembly (Assembly.py) → same composites + instances + constraints + fragment
Part (Part.py)         → composites: model, inspect, plot
```

**Problems:**
- Constraints only available on Assembly
- MeshSelectionSet doesn't exist yet
- Single-body models can't use constraints
- Two entry points (pyGmsh vs Assembly) with nearly identical composites

## Target State

```
pyGmsh (_core.py)     → all composites + parts registry + constraints + mesh_selection
Part (Part.py)         → unchanged (optional geometry builder)
Assembly.py            → DELETED
```

## Files Affected

### DELETED
- `src/pyGmsh/core/Assembly.py` — absorbed into pyGmsh

### NEW (4 files)
- `src/pyGmsh/mesh/MeshSelectionSet.py` — MeshSelectionSet composite + MeshSelectionStore snapshot
- `src/pyGmsh/mesh/_mesh_filters.py` — spatial query functions
- `src/pyGmsh/core/Constraints.py` — constraint composite (extracted from Assembly)
- `src/pyGmsh/core/_parts_registry.py` — inline part creation + import + instance tracking

### MODIFIED (6 files)
- `src/pyGmsh/_core.py` — absorb Assembly functionality, add new composites
- `src/pyGmsh/_session.py` — add parts registry support to base class
- `src/pyGmsh/mesh/FEMData.py` — add mesh_selection + constraints fields
- `src/pyGmsh/mesh/_fem_extract.py` — snapshot mesh selections during build
- `src/pyGmsh/mesh/__init__.py` — export new classes
- `src/pyGmsh/__init__.py` — remove Assembly, export new classes

### TEST FILES
- `tests/test_mesh_selection_set.py` — new
- `tests/test_constraints_composite.py` — new
- `tests/test_parts_registry.py` — new
- `tests/test_library_contracts.py` — update (remove Assembly references)

---

## Implementation Phases

### Phase 1: MeshSelectionSet (non-breaking, additive only)

**Goal:** Add `g.mesh_selection` composite without touching anything else.

**Files:**

1. **NEW `src/pyGmsh/mesh/_mesh_filters.py`** (~300 lines)

   Pure functions, no Gmsh dependency. Easy to test in isolation.

   ```python
   def nodes_on_plane(node_ids, node_coords, axis, value, atol) -> ndarray[bool]
   def nodes_in_box(node_ids, node_coords, bbox) -> ndarray[bool]
   def nodes_in_sphere(node_ids, node_coords, center, radius) -> ndarray[bool]
   def nodes_nearest(node_ids, node_coords, point, count) -> ndarray[int]
   def elements_in_box(elem_ids, connectivity, node_coords_lookup, bbox) -> ndarray[bool]
   def elements_on_plane(elem_ids, connectivity, node_coords_lookup, axis, value, atol) -> ndarray[bool]
   def element_centroids(connectivity, node_coords_lookup) -> ndarray  # (E, 3)
   def boundary_nodes_of(connectivity) -> ndarray[int]
   ```

2. **NEW `src/pyGmsh/mesh/MeshSelectionSet.py`** (~600 lines)

   Two classes:

   **`MeshSelectionSet`** — live composite on `g.mesh_selection`
   ```python
   class MeshSelectionSet:
       def __init__(self, parent: _SessionBase) -> None:
           self._parent = parent
           self._sets: dict[tuple[int, int], dict] = {}
           self._next_tag: dict[int, int] = {0: 1, 1: 1, 2: 1, 3: 1}

       # ── Creation (mirrors g.physical) ─────────────
       def add(self, dim: int, tags: list[int], *, name="", tag=-1) -> int
       def add_nodes(self, *, name="", tag=-1, **filters) -> int
       def add_elements(self, dim=2, *, name="", tag=-1, **filters) -> int

       # ── Naming ────────────────────────────────────
       def set_name(self, dim: int, tag: int, name: str) -> MeshSelectionSet
       def remove_name(self, name: str) -> MeshSelectionSet

       # ── Removal ───────────────────────────────────
       def remove(self, dim_tags: list[DimTag]) -> MeshSelectionSet
       def remove_all(self) -> MeshSelectionSet

       # ── Queries (mirrors g.physical) ──────────────
       def get_all(self, dim=-1) -> list[DimTag]
       def get_entities(self, dim: int, tag: int) -> list[int]
       def get_name(self, dim: int, tag: int) -> str
       def get_tag(self, dim: int, name: str) -> int | None
       def summary(self) -> pd.DataFrame

       # ── Mesh data (same shape as PhysicalGroups) ──
       def get_nodes(self, dim: int, tag: int) -> dict
       def get_elements(self, dim: int, tag: int) -> dict

       # ── Set algebra ───────────────────────────────
       def union(self, dim, tag_a, tag_b, *, name="", tag=-1) -> int
       def intersection(self, dim, tag_a, tag_b, *, name="", tag=-1) -> int
       def difference(self, dim, tag_a, tag_b, *, name="", tag=-1) -> int

       # ── Bridge ────────────────────────────────────
       def from_physical(self, dim, name_or_tag, *, ms_name="", ms_tag=-1) -> int

       # ── Snapshot (for FEMData) ────────────────────
       def _snapshot(self) -> MeshSelectionStore
   ```

   **`MeshSelectionStore`** — immutable snapshot on `fem.mesh_selection`
   ```python
   class MeshSelectionStore:
       """Mirrors PhysicalGroupSet API exactly."""
       def __init__(self, sets: dict[tuple[int, int], dict]) -> None
       def get_all(self, dim=-1) -> list[DimTag]
       def get_name(self, dim, tag) -> str
       def get_tag(self, dim, name) -> int | None
       def get_nodes(self, dim, tag) -> dict    # {'tags', 'coords'}
       def get_elements(self, dim, tag) -> dict  # {'element_ids', 'connectivity'}
       def summary(self) -> pd.DataFrame
   ```

3. **MODIFY `src/pyGmsh/_core.py`** — add to _COMPOSITES:
   ```python
   ("mesh_selection", ".mesh.MeshSelectionSet", "MeshSelectionSet", False),
   ```
   Note: this line already exists in the current file (was added previously).

4. **MODIFY `src/pyGmsh/mesh/FEMData.py`** — add field:
   ```python
   mesh_selection: MeshSelectionStore = field(repr=False, default_factory=lambda: MeshSelectionStore({}))
   ```

5. **MODIFY `src/pyGmsh/mesh/_fem_extract.py`** — in `build_fem_data()`:
   ```python
   def build_fem_data(dim=2, mesh_selection_composite=None):
       ...
       ms_store = MeshSelectionStore({})
       if mesh_selection_composite is not None:
           ms_store = mesh_selection_composite._snapshot()
       return FEMData(..., mesh_selection=ms_store)
   ```

6. **MODIFY `src/pyGmsh/mesh/__init__.py`** — add exports

7. **NEW `tests/test_mesh_selection_set.py`**

**Validation:** All existing tests pass. New tests pass. `g.physical` unchanged.

---

### Phase 2: Constraints Composite (extract from Assembly)

**Goal:** Create `g.constraints` as a standalone composite. The `ConstraintResolver` and all constraint dataclasses in `solvers/Constraints.py` stay untouched — we're only extracting the factory methods and resolution orchestration from Assembly.

**Files:**

1. **NEW `src/pyGmsh/core/Constraints.py`** (~400 lines)

   Extracted from Assembly.py lines 731-1155 (factory methods + resolve_constraints).

   ```python
   class Constraints:
       """Constraint composite — define + resolve kinematic interactions."""

       def __init__(self, parent: _SessionBase) -> None:
           self._parent = parent
           self.defs: list[ConstraintDef] = []
           self.records: list[ConstraintRecord] = []

       # ── Level 1: Node-to-Node ────────────
       def equal_dof(self, master, slave, *, dofs=None, tolerance=1e-6, name=None)
       def rigid_link(self, master, slave, *, link_type="beam", ...)
       def penalty(self, master, slave, *, stiffness=1e10, ...)

       # ── Level 2: Node-to-Group ───────────
       def rigid_diaphragm(self, master, slave, *, master_point, ...)
       def rigid_body(self, master, slave, *, master_point, ...)
       def kinematic_coupling(self, master, slave, *, master_point, ...)

       # ── Level 3: Node-to-Surface ─────────
       def tie(self, master, slave, *, tolerance=1.0, ...)
       def distributing_coupling(self, master, slave, *, ...)
       def embedded(self, host, embedded, *, ...)

       # ── Level 4: Surface-to-Surface ──────
       def tied_contact(self, master, slave, *, ...)
       def mortar(self, master, slave, *, ...)

       # ── Resolution ───────────────────────
       def resolve(self, node_tags, node_coords, ...) -> list[ConstraintRecord]
   ```

   **Key change in `master`/`slave` arguments:**

   Currently these are instance labels (strings). In the new API, they become
   flexible references that can be:
   - A part label (string) → resolves via parts registry
   - A physical group name (string prefixed with "pg:") → resolves via g.physical
   - A mesh selection name (string prefixed with "ms:") → resolves via g.mesh_selection
   - A dict with explicit node_tags → used directly

   Or simpler: `master` and `slave` accept either:
   - `str` → resolved as part label (backward compat) or physical group name
   - `dict` → {'node_ids': ndarray, 'node_coords': ndarray} directly

   The resolver logic (`_resolve_constraint_nodes`, `_resolve_constraint_faces`)
   moves from Assembly into this class.

2. **MODIFY `src/pyGmsh/_core.py`** — add to _COMPOSITES:
   ```python
   ("constraints", ".core.Constraints", "Constraints", False),
   ```

3. **NEW `tests/test_constraints_composite.py`**

**Validation:** ConstraintResolver + all dataclasses untouched. Tests pass.

---

### Phase 3: Parts Registry (absorb Assembly)

**Goal:** Move instance management into pyGmsh. Create parts inline or import from STEP.

**Files:**

1. **NEW `src/pyGmsh/core/_parts_registry.py`** (~350 lines)

   Extracted from Assembly.py: Instance dataclass + add_part + add_file +
   fragment_all + fragment_pair + instance node/face map building.

   ```python
   @dataclass
   class Instance:
       label: str
       part_name: str
       file_path: Path | None
       entities: dict[int, list[int]]
       translate: tuple[float, float, float]
       rotate: tuple[float, ...] | None
       properties: dict[str, Any]
       bbox: tuple[float, ...] | None

   class PartsRegistry:
       """Manages parts/instances within a pyGmsh session."""

       def __init__(self, parent: _SessionBase) -> None:
           self._parent = parent
           self.instances: dict[str, Instance] = {}
           self._counter: int = 0

       # ── Part management ───────────────────
       def create_part(self, label: str) -> InlinePart:
           """Create a part inline — returns a geometry proxy."""

       def import_part(self, path, *, label=None, translate=(0,0,0), rotate=None) -> Instance:
           """Import STEP/IGES file as a named instance."""

       def add_part(self, part: Part, *, label=None, translate=(0,0,0), rotate=None) -> Instance:
           """Import a standalone Part object."""

       # ── Fragment ──────────────────────────
       def fragment_all(self, *, dim=None) -> list[int]
       def fragment_pair(self, label_a, label_b, *, dim=None) -> list[int]

       # ── Instance queries ──────────────────
       def get_instance(self, label) -> Instance
       def list_instances(self) -> list[str]

       # ── Node/face mapping (for constraints) ──
       def build_instance_node_map(self, node_tags, node_coords) -> dict[str, set[int]]
       def build_instance_face_map(self, instance_node_map) -> dict[str, ndarray]

   class InlinePart:
       """Lightweight proxy — delegates geometry to parent's g.model,
       tracks which entities belong to this part label."""

       def __init__(self, label: str, parent_model: Model) -> None:
           self._label = label
           self._model = parent_model
           self._entities: list[DimTag] = []

       def add_box(self, ...) -> int:
           tag = self._model.add_box(...)
           self._entities.append((3, tag))
           return tag
       # ... delegates for other geometry methods
   ```

2. **MODIFY `src/pyGmsh/_core.py`**:
   - Add PartsRegistry as a composite or direct attribute
   - Add convenience methods: `g.create_part()`, `g.import_part()`, `g.fragment()`
   - These delegate to `self._parts_registry`

3. **MODIFY `src/pyGmsh/__init__.py`**:
   - Remove `from apeGmsh.core.Assembly import Assembly`
   - Remove Assembly from `__all__`
   - Add new exports

4. **DELETE `src/pyGmsh/core/Assembly.py`**

5. **MODIFY `src/pyGmsh/core/__init__.py`**:
   - Remove Assembly import

6. **MODIFY `tests/test_library_contracts.py`**:
   - Update Assembly references to pyGmsh

7. **Update examples** that use Assembly → use pyGmsh directly

**Validation:** All example notebooks updated. Tests pass.

---

### Phase 4: Constraint-Mesh Integration (node/element creation)

**Goal:** Allow constraints to create new mesh nodes and elements during resolution.

This is the payoff phase — constraints can now:
- Duplicate nodes at interfaces
- Create zero-length elements
- Create Lagrange multiplier nodes
- Register all created entities as mesh selections automatically

**Files:**

1. **MODIFY `src/pyGmsh/core/Constraints.py`**:
   - `resolve()` returns augmented mesh data (new nodes, new elements)
   - Created entities auto-registered in `g.mesh_selection`

2. **MODIFY `src/pyGmsh/mesh/FEMData.py`**:
   - Add constraint data to FEMData snapshot
   - Augmented node/element arrays include constraint-created entities

3. **MODIFY `src/pyGmsh/mesh/_fem_extract.py`**:
   - `build_fem_data()` accepts constraint augmentation

**Validation:** Full pipeline test with constraint node creation.

---

### Phase 5: Viewer Integration

**Goal:** MeshSelectionSet selections visualizable in g.view / g.plot.

1. **MODIFY `src/pyGmsh/mesh/View.py`** — highlight methods for selections
2. **MODIFY `src/pyGmsh/viz/Plot.py`** — highlight methods for selections
3. **MODIFY viewers** — selection visualization in interactive viewers

---

## Dependency Graph (implementation order)

```
Phase 1: MeshSelectionSet     ← standalone, no dependencies
    │
Phase 2: Constraints composite ← needs MeshSelectionSet for resolution
    │
Phase 3: Parts Registry       ← needs Constraints to move from Assembly
    │                            Assembly deleted here
Phase 4: Constraint-Mesh      ← needs all three above
    │
Phase 5: Viewer Integration   ← needs MeshSelectionSet
```

## Risk Mitigation

- **Phase 1 is fully additive** — zero changes to existing code paths
- **Phase 2 extracts, doesn't rewrite** — ConstraintResolver untouched
- **Phase 3 is the only breaking change** — Assembly deletion
  - Since Assembly is being deprecated, no backward compatibility needed
  - All examples updated in this phase
- **Phase 4 extends, doesn't modify** — new capability on top of stable foundation
- **Each phase has its own test suite** — run after each phase

## Summary of Changes by File

| File | Phase | Action | Lines |
|------|-------|--------|-------|
| `mesh/_mesh_filters.py` | 1 | NEW | ~300 |
| `mesh/MeshSelectionSet.py` | 1 | NEW | ~600 |
| `mesh/FEMData.py` | 1,4 | MODIFY | ~50 |
| `mesh/_fem_extract.py` | 1 | MODIFY | ~20 |
| `mesh/__init__.py` | 1 | MODIFY | ~5 |
| `_core.py` | 1,2,3 | MODIFY | ~100 |
| `core/Constraints.py` | 2 | NEW | ~400 |
| `core/_parts_registry.py` | 3 | NEW | ~350 |
| `core/Assembly.py` | 3 | DELETE | -1244 |
| `core/__init__.py` | 3 | MODIFY | ~5 |
| `__init__.py` | 3 | MODIFY | ~10 |
| `solvers/Constraints.py` | — | UNTOUCHED | 0 |
| `core/Part.py` | — | UNTOUCHED | 0 |
| Tests | 1-4 | NEW/MODIFY | ~1000 |

**Net delta:** ~1800 new lines, ~1244 deleted (Assembly) = ~550 net growth
