# ADR 0097 — Void tools on the modeling composite: role, not type

**Status:** Proposed (2026-08-15). Session-side geometry/boolean
authoring — no Emitter Protocol, no H5 schema bump, no OpenSees
change. Chain-phase freeze (ADR 0038) already covers every `add_*` /
boolean / transform this ADR composes over.

## Context

Structural models routinely need *openings*: ducts through a wall,
a tunnel through a continuum box, a hole in a shell, a tapered void
lofted between two sections. apeGmsh already has the OCC operation
that makes a hole — `g.model.boolean.cut` (A − B, dim 2 or 3), with
label/PG survival via `pg_preserved()` and Part remapping. It also
already has the bodies that *become* tools: solid primitives
(`add_box` / `add_cylinder` / …), faces (`add_rectangle` /
`add_plane_surface`), `geometry.sweep` (polyline path + profile, with
interior-path cleanup), `transforms.sweep` (OCC `addPipe` on a wire),
and `transforms.thru_sections` (loft).

What it does not have:

1. **`add_polyline`** — named in the `add_arch` error text, absent
   from the API. Authors currently hand-roll points → lines → wire.
2. **Vertex fillet / chamfer** on those joints. OCC `addPipe` kinks
   or fails at sharp path corners; loft wires must stay
   topologically similar. Rounding selected vertices is how you
   control that flow *before* the solid exists.
3. **A void *role*.** `_register` stores only `kind`. A tool volume
   left in the kernel is meshed as structure. There is no way to
   mark “this body is emptiness” and fail loud at `generate()` if it
   was never subtracted.
4. **A two-phase apply.** The requested workflow is: author empty
   geometry (primitives, sweeps, lofts), *then* establish the
   boolean. Immediate `cut(host, freshly_created_cylinder)` works
   today but does not batch, and does not protect against forgetting.

`cut_by_surface` / `cut_by_plane` are the *split-and-keep-pieces*
family. They are the wrong operation for an opening and stay out of
this ADR.

The design fork: a first-class `Void` solid (parallel
`Void.add_box` catalog) versus extra functionality on the existing
solid/surface creation path.

## Decision

### 1. Void is a role on an OCC entity, not a type

OCC has no empty body. A hole is an ordinary volume (or a dim-2
face) used as the *tool* of `boolean.cut`. A parallel Void primitive
catalog would fork every `add_*`, confuse labels/PGs, and still
mesh as material if left in the kernel.

v1 threads `as_void: bool = False` through `_add_solid` (every
solid primitive) and the surface tails used as 2-D tools
(`add_rectangle`, `add_plane_surface`). `_register` gains an
optional `role=` ; `role="void"` is stored next to `kind` in
`model._metadata`. No `g.model.voids` composite, no `Void` class.

### 2. Apply is `boolean.cut` (subtract)

Openings subtract. Splits stay on `cut_by_*`.

```python
g.model.geometry.add_cylinder(..., as_void=True, label="duct")
g.model.boolean.cut("wall", "duct")            # existing
g.model.boolean.apply_voids(host="wall")       # all role=void tools vs host
```

`apply_voids` resolves the host, collects every live `_metadata`
entry with `role=="void"` at the host's dimension, and delegates to
`_bool_op("cut", ...)`. Tools are consumed (`remove_tool=True`);
host labels survive through the existing remap. Dim-2 openings use
the same role + `cut(..., dim=2)` — the path section profiles
already take.

### 3. Two-phase authoring; unapplied voids fail loud

Author tools, then apply. One-shot helpers are sugar over the same
path. `validate_pre_mesh` (already invoked by `generate`) raises
`GeometryValidationError` naming leftover void labels/tags. It does
**not** silently delete them — an unapplied hole must be obvious.

### 4. Polyline + vertex fillet/chamfer is the missing authoring layer

```python
curves = g.model.geometry.add_polyline(
    [(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)],
    closed=True,
    fillet={1: 0.15, 3: 0.15},   # vertex index → radius
    chamfer={2: 0.05},            # vertex index → setback
    label="opening_profile",
)
```

Returns the ordered **persistent curve tags** (lines + inserted
arcs). Closed profiles become a plane surface via existing
`add_curve_loop` + `add_plane_surface`. Open polylines become a
sweep path via existing `add_wire` (transient; never labelled —
ADR-pinned by the current `add_wire` contract).

Fillet/chamfer v1 is **polyline vertex** construction: trim both
adjacent segments, insert a circular arc (fillet) or a straight
setback line (chamfer). Math lives in `core/_polyline.py` (no gmsh,
reusable by tests). Same-vertex fillet+chamfer raises; setback
exceeding the adjacent segment (or the sum of both ends) raises;
open-polyline endpoints cannot be treated. Interior vertices whose
turning angle exceeds 30° with no treatment emit
`WarnGeomSharpPolylineCorner` (OCC pipe risk). Do **not** call OCC
volume `fillet`/`chamfer` in v1 — that rounds solid edges *after*
the body exists and does not fix path kinks before the pipe.

### 5. Void-sweep and void-loft are orchestration, not a third sweep

v1 covers both patterns, sharing the polyline/fillet builder:

- `g.model.geometry.add_void_sweep(profile, path, …)` — profile is
  an existing face or a closed curve-tag list; path is a curve-tag
  list (delegates to `geometry.sweep` so the interior path is
  cleaned) or a wire tag (delegates to `transforms.sweep`). Marks
  the highest-dimension survivor `role="void"`.
- `g.model.geometry.add_void_loft(sections, …)` — two or more
  section curve-tag lists (or wire tags); delegates to
  `thru_sections`. Matching sub-curve counts after fillet are
  required when both sections are curve lists; mismatch raises with
  a fix-it (apply the same vertex map to every section). Marks the
  loft solid `role="void"`.

Do not add a third pipe implementation.

## Alternatives rejected

- **First-class Void solid.** Duplicates primitives; OCC still
  stores a volume; meshing would treat it as material.
- **`g.model.voids` composite with a full primitive catalog.** Same
  duplication. A later thin alias is allowed if `as_void=` feels
  noisy; not v1.
- **Immediate-only cut with no role.** Fights two-phase authoring
  and leaves leftover tools meshable as structure.
- **OCC `fillet` on the void volume.** Different feature. Useful
  later; does not fix path kinks *before* the pipe.

## Consequences

- Authors mark tools with `as_void=True` (or the sweep/loft
  wrappers) and apply before `generate()`. Forgetting is a loud
  pre-mesh error, not a spurious solid in the snapshot.
- Polyline fillets are the supported way to keep a swept void's
  Frenet frame well-defined at corners.
- No schema / Protocol / viewer change. `kind` in `_metadata`
  stays the OCC-construction name; `role` is additive and ignored
  by everything that only reads `kind`.

## Open items

- OCC 3-D edge fillet/chamfer on an already-built solid.
- A thin `g.model.voids` alias if the flag on every `add_*` proves
  noisy in scripts.
- Dual-rail (guide-curve) sweep. v1 is path+profile and
  thru-sections only.
