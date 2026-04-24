# Changelog

## v1.0.4 — Low-level booleans preserve Instances + accept label/PG refs

### FIXED

- `g.model.boolean.{fuse,cut,intersect,fragment}` now keep
  `Instance.entities` consistent when called directly on tags that
  happen to belong to a tracked Instance.  Previously the remap only
  ran inside `g.parts.fragment_all` / `fragment_pair` / `fuse_group`,
  so a low-level boolean left Instance entries pointing at consumed
  tags.  The remap-from-result walk has been extracted into
  `PartsRegistry._remap_from_result` and every OCC boolean call site
  (both `_bool_op` and the Parts-level methods) now routes through
  that single implementation.

### ADDED

- `g.model.boolean.*` accepts label names and user physical-group
  names in `objects=` / `tools=`, matching the input shape of
  `g.physical.add`.  Strings resolve via the shared resolver: label
  (Tier 1) first, then user PG (Tier 2).  Raw tags, dimtags, and
  mixed lists still work.

### INTERNAL

- New `resolve_to_dimtags` helper in `apeGmsh.core._helpers` —
  companion to `resolve_to_tags` that emits `(dim, tag)` pairs.
  Handles labels / PGs that span multiple dimensions without the
  caller having to coerce a single dim.
- Plan B (`Instance.entities` as a computed label-backed property)
  was weighed against this conservative fix and deferred; see
  `internal_docs/plan_instance_computed_view.md` for the signals that
  would trigger revisiting it.

---

## v1.0.0 — Clean Architecture (breaking)

v1.0 bundles two breaking changes: the package rename and the Model
composition refactor. A full find-replace migration guide is at
[`internal_docs/MIGRATION_v1.md`](internal_docs/MIGRATION_v1.md).

### BREAKING

- **Package renamed**: `pyGmsh` → `apeGmsh`
  - `from pyGmsh import pyGmsh` → `from apeGmsh import apeGmsh`
  - `class pyGmsh(_SessionBase)` → `class apeGmsh(_SessionBase)`
  - Companion app `apeGmshViewer` keeps its name (only its internal
    imports of our theme module were updated)

- **Model methods split into five sub-composites** (composition replaces
  mixin inheritance):
  - `g.model.geometry.*` — 19 primitive builders (add_point, add_line,
    add_box, add_cylinder, etc.)
  - `g.model.boolean.*` — fuse, cut, intersect, fragment
  - `g.model.transforms.*` — translate, rotate, scale, mirror, copy,
    extrude, revolve, sweep, thru_sections
  - `g.model.io.*` — load/save STEP, IGES, DXF, MSH, heal_shapes
  - `g.model.queries.*` — bounding_box, center_of_mass, mass, boundary,
    adjacencies, entities_in_bounding_box, remove, remove_duplicates,
    make_conformal, registry
  - `g.model.sync()`, `g.model.viewer()`, `g.model.gui()`,
    `g.model.launch_picker()`, `g.model.selection` stay flat on Model

- **Rename `g.mass` → `g.masses`** for consistency with the other
  plural composites (`g.loads`, `g.parts`, `g.physical`, `g.constraints`,
  `g.mesh_selection`)
  - `fem.mass` → `fem.masses`
  - Class names (`MassesComposite`, `MassSet`, `MassDef`, `MassRecord`)
    unchanged

- **Removed legacy aliases**:
  - `g.initialize()` / `g.finalize()` → use `g.begin()` / `g.end()`
  - `g._initialized` → use `g.is_active`
  - `g.model_name` → use `g.name`

- **Removed deprecated methods**:
  - `g.model.viewer_fast()` / `g.mesh.viewer_fast()` → use `viewer()`
    (always fast now)
  - `g.parts.add_physical_groups()` → explicit
    `g.physical.add_volume(inst.entities[3], name=...)`
  - `g.opensees.add_nodal_load()` → use `g.loads.point()` in a
    `g.loads.pattern()` block
  - `g.mesh_selection.add_nodes(nearest_to=...)` → `closest_to=`

- **Removed convenience delegates on the session**:
  - `g.remove_duplicates()` → `g.model.queries.remove_duplicates()`
  - `g.make_conformal()` → `g.model.queries.make_conformal()`

- **Removed property on `_SessionBase`**:
  - `_parent.model_name` → `_parent.name`

### FIXED

- Pylance / static analyzers no longer lose track of Model methods.
  Composition makes every method statically discoverable through
  the sub-composite classes (`_Geometry`, `_Boolean`, `_Transforms`,
  `_IO`, `_Queries`), each of which is a concrete class with explicit
  methods. No MRO walking across 5 mixin files.

### INTERNAL

- `_GeometryMixin` → `_Geometry`
- `_BooleanMixin` → `_Boolean`
- `_TransformsMixin` → `_Transforms`
- `_IOMixin` → `_IO`
- `_QueriesMixin` → `_Queries`

Each sub-composite now takes a `model` reference in `__init__` and
accesses Model state via `self._model._log(...)`,
`self._model._register(...)`, `self._model._as_dimtags(...)`,
`self._model._registry`, instead of inheriting state.

### MIGRATION

See [`internal_docs/MIGRATION_v1.md`](internal_docs/MIGRATION_v1.md) for the complete
find-replace table and an automated migration script.

**v0.3.0** is the last `pyGmsh` release (pre-rename safety tag).
**v0.3.1** is the transitional `apeGmsh` release (rename only).
**v1.0.0** is the new architecture (composition + cleanups).

---

## v0.3.1 — Package rename (transitional)

- Renamed `pyGmsh` package directory to `apeGmsh/` on disk
- All internal imports, class name, config entries updated
- Examples and docs still reference old API (deferred to v1.0)
- Safety release — rename is isolated from the architectural
  refactor that follows in v1.0

## v0.3.0 — Last pyGmsh release (safety tag)

Safety checkpoint before the package rename and v1.0 refactor. This
is the final tag under the `pyGmsh` name. If you need the old API,
pin to this version.

---

## v0.2.x — Loads, Masses, Viewer overlays

- New `g.loads` composite with pattern context managers
- New `g.mass` composite (renamed to `g.masses` in v1.0)
- Read-only Loads/Mass tabs in the model viewer
- `fem.loads` / `fem.mass` auto-resolved by `get_fem_data()`

## v0.2.0 — Composites architecture

- Assembly absorbed into `apeGmsh` as composites:
  - `g.parts` (PartsRegistry)
  - `g.constraints` (ConstraintsComposite)
- MeshSelectionSet + \_mesh\_filters spatial query engine
- Viewer rebuild: BRep / mesh viewers unified around EntityRegistry,
  PickEngine, ColorManager, VisibilityManager
- Catppuccin Mocha theme across all viewers
