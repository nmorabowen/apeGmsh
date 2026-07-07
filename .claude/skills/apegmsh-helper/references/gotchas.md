# Gotchas — anti-patterns & easily-missed pitfalls
<!-- skill-freshness: verified against apeGmsh main@8eeda7a3 (2026-07-06) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

Read this when a build "should work" but doesn't, or before writing
constraint / selection / Results code from memory. The other references
cover the happy path; this file is the ❌→✅ list and the subtle traps
that aren't obvious from the API surface.

## Anti-patterns (❌ → ✅)

### ❌ `equal_dof` for non-matching meshes → ✅ use `tie`
`equal_dof` needs co-located nodes. `tie` uses shape-function
interpolation for non-matching interfaces.

### ❌ `g.mesh.generate()` → ✅ `g.mesh.generation.generate()`
No shortcut methods on parent composites. The sub-composite prefix is
required everywhere (`g.model.geometry.add_box`, not `g.model.add_box`).

### ❌ Skipping `make_conformal` after STEP import
Touching bodies need `remove_duplicates` + `make_conformal` for shared
interfaces, or the meshes won't share nodes at the interface.

### ❌ Storing tags in local dicts → ✅ use `g.physical`
Physical groups are the only way to address mesh subsets by name across
the pipeline. Local dicts don't survive `fragment_all()` (OCC renumbers).

### ❌ `fem.node_ids` (old API) → ✅ `fem.nodes.ids`
The FEMData API was rebuilt as composites. Old flat attributes no longer
exist — use `fem.nodes.*` and `fem.elements.*`.

### ❌ Hand-stitching spatial filters → ✅ the `.select()` chain
Don't manually `np.isin` / intersect results, and don't assume the
entity-family `g.model.select().in_box` matches a point-family box. Use
the one canonical chain: `fem.nodes.select(pg=...).in_box(...).on_plane(...)`,
with set algebra via `| & - ^`. The old single-shot `fem.*.get/.resolve`
/ `g.mesh_selection.add_nodes` accessors were **removed** — `.select()`
is the only accessor (its no-arg / `pg=` seed covers the simple case too).

### ❌ `results.elements.gauss.select(...)` — not shipped
Results sub-composites (`gauss` / `fibers` / `layers` / `line_stations` /
`springs`) have **no** `.select()`. Use their existing
`.in_box / .nearest_to / .on_plane` helpers. (Mesh-selection name-seed,
by contrast, **is** shipped: `g.mesh_selection.select(name="my_set")`.)

### ❌ Reaching for `g.opensees` / `g.opensees.ingest`
Both are **gone**. The entry point is `from apeGmsh.opensees import apeSees`.
`apeSees` does **not** ingest `g.loads` / `g.masses` — re-declare those
explicitly on the bridge (see `opensees-bridge.md`). MP constraints **DO**
emit automatically from `g.constraints.*`.

### ❌ Expecting `BindError` (it's gone) → ✅ read `lineage.warnings`
`BindError` was DELETED in the three-broker refactor. Use
`results.lineage.warnings` (warn-not-raise per ADR 0021 INV-2) or opt into
loud-fail via `results.lineage.assert_clean()`. See `results.md`.

### ❌ `Results.from_native(path)` without `model=`
`model=` is **required** (TypeError otherwise). Canonical Composed-file
pattern: `Results.from_native(path, model=OpenSeesModel.from_h5(path, fem_root="/model"))`.
Same for `Results.from_mpco(model_h5=)` and `Results.from_recorders(model=)`.
See `results.md`.

### ❌ `Results.viewer(model_h5=...)`
The kwarg was removed — the viewer reads `results.model` directly. CLI:
`python -m apeGmsh.viewers run.h5` auto-resolves native Composed files;
`--model-h5 PATH` is required ONLY for `.mpco` files (sibling pointer).

### ❌ `h5_reader.materials()` returning a dict
The dict-style accessors were removed. Today's `model.materials()` returns
`list[MaterialRecord]`; `model.sections()` returns
`list[SectionSimpleRecord | SectionComplexRecord]`; etc. Use
`materials_by_family()` for the family-keyed view.

### ❌ `Viscous` in a `section.Aggregator` → ✅ put it on a `ZeroLength` `-mat`
A rate-dependent material (`Viscous` / `ViscousDamper` / `Maxwell`) is silently
inert inside `section.Aggregator → ZeroLengthSection` — neither passes a strain
rate, so it yields **zero** damping force. `Aggregator` now **fails loud** on
one. Put the dashpot directly on a `ZeroLength` `(material, dof)` pair (parallel
an elastic spring on the same DOF for a non-singular static tangent).

### ⚠️ `ZeroLengthSection` Rayleigh defaults **ON** (opposite of `ZeroLength`)
`zeroLengthSection` initialises `-doRayleigh 1` in OpenSees, so the primitive
mirrors that: `do_rayleigh` defaults **`True`** (plain `ZeroLength` defaults
`False`). It always emits the flag explicitly, so `do_rayleigh=False` actually
disables it. Don't assume the two elements share a default.

## Pitfalls not covered in the other references

### `remove_duplicates` tolerance is unit-dependent
mm models: `tolerance=1e-3`. Metre models: `tolerance=1e-6`. Picking the
metre tolerance on a mm model silently fails to merge coincident nodes.

### `in_box` is half-open `[lo, hi)` (point family)
Every point-family `.select().in_box(...)` (and the retained
`filter_set(in_box=)`) excludes a coordinate / centroid lying exactly on
the **upper** box face by default (matches the results side; adjacent
boxes don't double-count). Pass `inclusive=True` for the closed box
`[lo, hi]`. The **entity** family (`g.model.select().in_box`) has **no**
`inclusive=` knob — passing it raises `TypeError`; use `.on_plane(...)` /
`.crossing_plane(spec, mode=...)` for an exact predicate.

### Selection-v2 capability gaps (ADR-0017)
v2's mandate was unification, not capability reduction. Two removed
surfaces are *incomplete unification*, not WONTFIX:
- **Gap 1 — geometry → named mesh-selection**: capability is **INTACT**
  via the 2-call route — pre-mesh `g.model.select(...).to_physical(name)`
  then post-mesh `g.mesh_selection.from_physical(dim, name, ms_name=)`
  (or `g.mesh_selection.add(dim, ids, name=)`). Only the one-call ergonomic
  was lost.
- **Gap 2 — declarative filter grammar** (`select_*(labels=/kinds=/
  *_range=/predicate=/exclude_tags=)`): a unique-capability loss; a
  v2-native `EntitySelection` successor is **owed/planned**
  (`docs/plans/selection-gaps-v3.md`). Until it ships, use the viewer-pick
  `viz.Selection.filter()` or a manual predicate over
  `g.model.select(...).result()`.

Source of truth: `docs/plans/selection-unification-v2.md` §6 + ADR 0016/0017.
