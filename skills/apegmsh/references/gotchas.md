# Gotchas — anti-patterns & easily-missed pitfalls
<!-- skill-freshness: verified against apeGmsh main@56cd64ec (2026-08-04) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

Read this when a build "should work" but doesn't, or before writing
constraint / selection / Results code from memory. The other references
cover the happy path; this file is the ❌→✅ list and the subtle traps
that aren't obvious from the API surface.

## Anti-patterns (❌ → ✅)

### ❌ `equal_dof` for non-matching meshes → ✅ use `tie`
`equal_dof` needs co-located nodes. `tie` uses shape-function
interpolation for non-matching interfaces.

### The constraint/load family fails LOUD now (silent-failures program, Aug 2026)
Don't be surprised by new exceptions where old code was silent — each one
is replacing a converged-but-wrong answer:
- **from_h5/compose sessions**: misspelled `bc`/load/mass targets raise
  `KeyError`; `g.displacements.*`, distributed loads/masses, `contact`,
  `g.embed`, `g.reinforce`, `g.decouple_node` raise `ChainPhaseError`
  (declare them in the part session before saving). Live sessions keep the
  lenient defer-to-re-extraction path.
- **tie/tied_contact resolving 0 records** raises `ValueError` (check the
  tolerance vs the interface gap); a partial projection warns with counts.
- **`p.from_model(case)` matching zero records** raises `BridgeError` at
  emit (`allow_empty=True` to permit); pre-2.26.1 h5 files flattened
  displacement case names to `'default'` — re-save from the source session.
- **tie/embedded `stiffness` defaults to `"auto"`** (K = 1e3·E_host·L_char
  at emit); a numeric 1e18 (old files) warns; Lagrange + absolute
  `NormDispIncr` warns (multiplier DOFs make tight tolerances unreachable).

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
`apeSees` does **not** auto-emit `g.loads` / `g.displacements` / `g.masses` —
loads and prescribed displacements are **opt-in** via `p.from_model(case)`
inside a bridge pattern (ADR 0051); masses go through explicit `ops.mass(...)`
or `ops.mass_from_model()` (see `opensees-bridge.md`). MP constraints **DO**
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

### ❌ OS-capturing the human's desktop to "look" → ✅ visors
Agents do not screenshot monitors or the live Qt window (that is the
human's desk: other apps, secrets, the wrong display). ✅ Write
`.apegmsh/visors/*.png` with `results.render` / `fem.render` (ADR 0094
offscreen stills) and labeled `results.plot.*`, then read those files.
`viewer()` stays the human surface.

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

### ❌ static integrator + `ops.analysis.Transient()` → ✅ match the slot
OpenSees keeps separate static / transient integrator slots and **silently
substitutes a default** when the one an `analysis` reads is empty —
`Newmark(0.5,0.25)` for `Transient`, `LoadControl(1,1,1,1)` for `Static`.
The warning is suppressible, so a suppressed run says nothing at all. A
`LadrunoArcLength` (a `StaticIntegrator`) under `Transient` is discarded
and you get a plausible curve that never passes the limit point. No bridge
guard yet — ADR 0082 G1. Slot table + the `LadrunoDynamicRelaxation`
trap (quasi-static method, *transient* by dispatch) in
[opensees-bridge.md](opensees-bridge.md).

### ❌ `Newton(tangent="initial")` → ✅ `ModifiedNewton(tangent="initial")`
Same algorithm, same iterates — but `Newton` re-forms and re-factorizes the
unchanged `K₀` on *every* iteration (`formTangent` is inside the loop, and
its `zeroA()` kills the fork's factorization reuse). `ModifiedNewton` forms
it once per step. Worst on softening / arc-length runs. See
[opensees-bridge.md](opensees-bridge.md); ADR 0082 G2.

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
  v2-native `EntitySelection` successor is **owed/planned** (ADR 0017).
  Until it ships, use the viewer-pick `viz.Selection.filter()` or a
  manual predicate over `g.model.select(...).result()`.

Source of truth: ADR 0016/0017 + the published selection page
<https://nmorabowen.github.io/apeGmsh/concepts/selection/>.

## Section analyzer (ADR 0078) — the three traps

Full reference: `section-properties.md`. The ones that bite:

### Overlap + fragment double-covers material PGs
Authoring an inner face **inside** an outer one and calling
`fragment_pair` puts the overlap piece in BOTH PGs → the analyzer's
exact-cover gate raises. ✅ Carve first
(`g.model.boolean.cut(outer.entities[2], inner.entities[2], dim=2,
remove_tool=False)`), THEN `fragment_pair` for conformity.

### Coincident edges ≠ shared edges → "disconnected" warping raise
Two hand-authored regions whose edges merely coincide mesh as separate
parts (`warping()` raises with the component count under the default
`disconnected="raise"` — that raise is the bug-catcher working). ✅
Build each shared line ONCE and reference it in both curve loops, or
author overlap-free faces + `fragment_pair`.

### Linear elements warp badly / composite accessors raise
`warping()` on tri3/quad4 fires `SectionAccuracyWarning` — ✅
`g.mesh.generation.set_order(2)` before `get_fem_data`. And on a
composite, `sec.geometric().Ixx_c` raising `CompositeSectionError` is
the naming law, not a bug — read `EIxx_c` or `transformed(e_ref=...)`
(ratios `rx/ry/r11/r22`, `alpha_x/alpha_y` stay valid everywhere).

## Multi-part ties (ADR 0085/0086) — the ones that cost a run

### Mixed element order needs compose, and `Part` cannot help
`set_order` is global per gmsh session → hex20 ribs + hex8 covers is
impossible in ONE session, and a `Part` has no mesh composite (deliberate —
ADR 0085; the docstring that said otherwise was stale for months). ✅ One
full `apeGmsh` session per part → `g.save()` → `Assembly` /
`from_h5`+`compose`. See `compose.md` §"Independent meshes per part".

### Collocation tie across an order mismatch over-constrains
`tie()` (default `method="collocation"`) pins quad8 slave nodes onto the
bilinear master field — the quadratic surface can't deform (measured on the
Cerro Lindo fuse: rib-root fixity a = 0.195 vs 0.230, K +7.7 %). ✅
`tie(..., method="mortar", enforce="equation")` — flat coincident interface
required, needs the Lagrange handler + an unsymmetric system, refuses tri6
SLAVE facets (swap sides), curved edges (midsides must sit at edge midpoints)
and overlapping master facets (coverage counts multiplicity, so two layers of
master surface can mask an untied slave strip — it gets *extrapolated*, and
`Σw = 1` plus a linear patch both stay clean, which is why it must be refused
up front). Every degenerate case is a hard `MortarTieError`
— if it resolved, it bonded. Subtlety: on NESTED aligned grids dual mortar
provably equals collocation; the difference (and the fix) appears when slave
facets straddle master face boundaries.

### Extract parts with dim=None or ties refuse
`tied_contact` and `tie(method="mortar")` resolve faces from the broker's
dim-2 element groups — a part saved from `get_fem_data(dim=3)` has none and
the chain-phase router raises `ValueError("re-extract with dim=None")`.
✅ Always `get_fem_data(dim=None)` before `g.save()` on tie-bearing parts.
