# ADR 0059 — Mesh recipes: one-call unstructured / structured meshing (`g.mesh.recipe`)

**Status:** Proposed (2026-06-11). Adds a high-level orchestration tier
over the existing `g.mesh.sizing` / `g.mesh.field` / `g.mesh.structured`
/ `g.mesh.generation` verbs. Session-side (mesh composite) — no bridge,
no Emitter Protocol, no schema change.

## Context

Meshing in apeGmsh today is a set of granular, composable verbs. That is
the right *power* surface, but even an easy model needs two to six calls
plus knowledge of several non-obvious footguns:

1. **CAD point-`lc` override.** STEP/IGES imports bake per-point
   characteristic lengths; `Mesh.MeshSizeFromPoints` is on by default,
   so they silently override `set_global_size`. The fix
   (`set_size_sources(from_points=False, ...)`) is documented but easy
   to not know.
2. **Per-region sizing bleeds.** `g.mesh.sizing.set_size(target, size)`
   walks the target down to its BRep *points*. Shared corner points
   belong to both regions, so a fine region softens its coarse neighbor
   near every interface — and `Mesh.MeshSizeMax` is global, so there is
   no per-region ceiling at all.
3. **Recombined-structured next to unstructured-tet cannot conform.** A
   transfinite + recombined volume puts **quads** on its boundary faces;
   a conformal tet neighbor needs **triangles** there. Gmsh has no
   robust automatic pyramid transition, so this fails late, at
   `generate()`, with a cryptic boundary-recovery error (or worse,
   produces a broken mesh caught only at FEM extraction).
4. **`set_transfinite` skip leaves orphan sizing.** The unified
   dispatcher (the current top structured tier,
   `_mesh_structured.py::set_transfinite`) warn-skips
   non-hex-decomposable entities — but the skipped entity is then meshed
   with whatever global sizing happens to be set, which may be wildly
   different from the structured spec next door.

The *structured* side already has most of a ladder: granular
(`set_transfinite_curve/surface/volume`) → `set_transfinite_box` →
`set_transfinite` (unified dispatcher, warn+skip). What is missing is a
top rung that also handles sizing fallback, generation, and the
mixed-interface guard. The *unstructured* side has no high-level rung at
all — the user composes `set_size_sources` + `set_global_size` /
`set_size` / fields by hand.

The goal stated by the user: **a recipe** — "set min and max element
size and output a mesh, for the whole model or parts; we still have the
fine tune, but for easy models we have an easy method."

## Decision

### 1. New sub-composite `g.mesh.recipe` with two verbs

`_Recipe` in `src/apeGmsh/mesh/_mesh_recipe.py`, the eighth `g.mesh`
sub-composite (alongside `generation`, `sizing`, `field`, `structured`,
`editing`, `queries`, `partitioning`). It is **pure orchestration**:
every action delegates to existing sizing / field / structured /
generation verbs and is recorded through the same `_directives` channel.
No new persistence, no schema bump, no `FEMData` impact, chain-phase
guard inherited from the delegated verbs.

```python
g.mesh.recipe.unstructured(
    target=None, *,                 # None = whole model; label/PG/part/
                                    #   Selection/dimtags otherwise
    max_size=None,                  # None → bbox-diagonal heuristic (§3)
    min_size=0.0,                   # 0.0 = no floor (matches set_global_size)
    dim=None,                       # generate dim; None → highest dim present
    generate=None,                  # None → auto by scope (§2)
)

g.mesh.recipe.structured(
    target=None, *,
    size=None, n=None,              # exactly one; same grammar as
                                    #   set_transfinite (scalar/tuple/dict)
    recombine=True,
    fallback="unstructured",        # "unstructured" | "warn" | "strict" (§4)
    dim=None,
    generate=None,
)
```

`target` resolves through the standard `resolve_to_dimtags` chain
(label → physical group → part), same as every other mesh verb.

### 2. `generate` semantics — auto by scope

`generate=None` (default) resolves to:

- **`True` when `target is None`** — the whole-model one-liner produces
  a mesh, full stop.
- **`False` when targeted** — region recipes are *declarations* that
  compose; the user calls several, then one
  `g.mesh.generation.generate(dim=...)` (or passes `generate=True` on
  the last recipe).

Explicit `True`/`False` always wins. This avoids the footgun where the
first of several region recipes generates prematurely, while keeping the
easy path a genuine single call:

```python
# easy model — one line
g.mesh.recipe.unstructured(min_size=0.2, max_size=1.0)

# mixed model — compose, then generate once
g.mesh.recipe.structured("soil_block", size=2.0, recombine=False)
g.mesh.recipe.unstructured("tunnel_liner", max_size=0.4)
g.mesh.generation.generate(dim=3)
```

### 3. Unstructured recipe

**Whole-model** (`target=None`):

1. `set_size_sources(from_points=False, from_curvature=False)` — bakes
   away footgun 1. (Opting back in = don't use the recipe, or call
   `set_size_sources` after it.)
2. `set_global_size(max_size, min_size)`.
3. `generate(dim)` per §2.

When `max_size=None`, derive it from the model bounding-box diagonal:
`max_size = diag / 20` (calibration is Open Q2). This makes
`g.mesh.recipe.unstructured()` with **zero arguments** produce something
sane on any model.

**Targeted** (`target=...`): per-region sizing is implemented with
**fields, not point-`lc`** (footgun 2):

- Each call builds a gmsh `Constant` field (`VolumesList` /
  `SurfacesList` = the resolved target entities, `VIn = max_size`,
  `VOut = 1e22`) via the existing `g.mesh.field` raw surface.
- The recipe owns a **single `Min` combiner field** per session,
  registered as the background mesh; repeated recipe calls append their
  `Constant` fields to it. The global `MeshSizeMin`/`MeshSizeMax` band
  is widened as needed so it does not clamp the fields.
- `min_size > 0.0` on a targeted call lowers the global floor (the
  background-field mechanism carries one desired size per point; the
  floor is global by gmsh construction — documented, not hidden).

### 4. Structured recipe

Delegates the constraint cascade to
`g.mesh.structured.set_transfinite(target, n=|size=, recombine=)` —
same sizing grammar (scalar / per-axis dict / per-principal-axis tuple),
same volumes-first ordering, same corner clustering. The recipe adds
what the dispatcher deliberately does not do:

**`fallback=` governs non-decomposable entities** (closes footgun 4):

- `"unstructured"` (default): a skipped volume/surface gets the §3
  region treatment at an **equivalent size** — `size` form: that size
  directly; `n` form: a size derived per entity from its characteristic
  edge length `/ (n - 1)`. The warning still fires, now stating the
  fallback size. *Recipe philosophy: you always get a mesh, and the
  incompatible region is tets at the size you asked for.*
- `"warn"`: exactly today's `set_transfinite` behavior (warn + skip,
  global sizing applies).
- `"strict"`: raise `MeshRecipeError` listing the offending entities
  and why each failed (cluster counts, from the existing cascade
  diagnostics).

Then `generate(dim)` per §2.

### 5. Mixed-interface guard — the genuinely new check

At **recipe generate time** (whole-model recipes, or a targeted recipe
with `generate=True`):

1. Classify each volume: *recombined-structured* (transfinite + recombine
   constraints applied) vs *everything else*.
2. Find boundary faces shared between the two classes
   (`g.model.queries` adjacency walk).
3. **Fail loud** (`MeshRecipeError`) naming the shared faces and the two
   remediations: `recombine=False` on the structured side (transfinite
   prisms/tets conform to triangles), or make the neighbor structured
   too.

The guard lives **only inside the recipe path**. The raw
`g.mesh.generation.generate()` is untouched — the PR #378 lesson
(open-world validation auto-wired into `generate` broke 63 legitimate
raw-gmsh models) applies verbatim: recipes are closed-world over what
*they* declared; raw users keep raw semantics. A standalone
`g.mesh.recipe.check()` exposing the same guard for raw-path users is
Open Q5.

### 6. Escape hatches unchanged

Recipes **layer over** the granular verbs, never replace them. Granular
calls compose before or after a recipe: a `Bump` bias on one edge after
`structured(...)`, a `threshold` refinement field that Min-combines with
the recipe's region fields, an explicit `set_transfinite_surface`
corner list where auto-detection fails. The recipe is the floor of
effort, not the ceiling of capability.

## Rationale

- **A fourth tier completes the existing ladder.** `set_transfinite`
  already proved the pattern (dispatcher over granular verbs,
  warn-skip); the recipe adds the three things a *recipe* needs that a
  *dispatcher* must not assume: sizing fallback, generation, and the
  interface guard.
- **The footguns are knowledge, not flexibility.** Nobody *wants*
  CAD point-`lc` silently overriding their band, or a fine region
  bleeding into a coarse one. Baking the fixes into the recipe removes
  required knowledge without removing any capability.
- **`Constant`+`Min` fields are gmsh's intended mechanism** for "desired
  size at a point, per region"; point-`lc` is a corner-only
  approximation that interpolates across interfaces. The field path
  costs ~30 lines via the existing raw `g.mesh.field` surface.
- **Fail-loud beats gmsh's late failure** on mixed quad/tri interfaces:
  the recipe knows *intent* (which regions were declared structured),
  so it can diagnose at declaration-aggregation time what gmsh can only
  trip over mid-generation.

**Alternatives rejected:**

- *Methods on `g.mesh.generation`* — mixes one-shot orchestration into a
  thin passthrough composite; `recipe` keeps the "composites split by
  concern" rule.
- *Quality presets* (`"coarse"/"medium"/"fine"`) — speculative; min/max
  plus the bbox default already covers the easy path. Revisit only on
  demand.
- *Automatic pyramid transition meshing* — gmsh has no robust support;
  out of scope. The guard converts the wall into a clear error instead.
- *Auto-wiring the interface guard into raw `generate()`* — violates
  the open-world lesson (PR #378 revert); recipes-only.

## Consequences

- New module `_mesh_recipe.py` + `g.mesh.recipe` attribute; **no schema
  bump, no persistence change** (delegated verbs already record
  `_directives`).
- The recipe **owns the background-field slot** (one `Min` combiner).
  Interaction with a user-authored `set_background` is Open Q3.
- `MeshRecipeError` joins the typed mesh/geometry error surface.
- Docs: `guide_meshing.md` gains a "recipes first" opening section
  (easy path before the granular tour); skill cheatsheet gains the
  `g.mesh.recipe` rows.
- `set_transfinite` remains the targeted power tool; `recipe.structured`
  is a strict superset wrapper (fallback + generate + guard) and shares
  its cascade code — no duplicated transfinite logic.

## Open questions

1. **`generate=None` auto rule** — is scope-dependent default too
   magic? Alternative: always `generate=True` and document the
   `generate=False` compose pattern. Lean: keep auto — the targeted
   premature-generate footgun is worse than the implicitness.
2. **Default `max_size` calibration** — `diag/20` proposed; validate
   against the worked-example models (tunnel twin, staged-gravity-ssi)
   before freezing.
3. **User-authored background field** — fold into the recipe's `Min`
   combiner (lean: fold + log) vs raise on collision.
4. **`n`-form fallback size derivation** — per-entity characteristic
   edge length `/(n-1)`: use the entity's own edges or the
   cluster-mean of the originally-specified axis?
5. **Expose the guard standalone** as `g.mesh.recipe.check()` for
   raw-path users? Lean: yes — trivial once the guard exists.

## Related

- `src/apeGmsh/mesh/_mesh_structured.py` — `set_transfinite` (the
  cascade this wraps), `set_transfinite_automatic`,
  `set_transfinite_box`.
- `src/apeGmsh/mesh/_mesh_sizing.py` — `set_size_sources` (CAD
  point-`lc` override), `set_global_size`, the point-walk `set_size`
  this ADR routes around for regions.
- `src/apeGmsh/mesh/_mesh_field.py` — raw field surface the region
  sizing builds on (`add("Constant")` / `minimum` / `set_background`).
- PR #378 revert (`8fb3e658`) — the never-auto-wire-open-world-checks
  precedent governing §5.
- `internal_docs/guide_meshing.md` — doc surface to update on
  acceptance.
