# ADR 0050 — Dimension-indexed loads + a `g.displacements` composite

**Status:** Proposed (2026-05-31). Restructures the `g.loads` authoring
surface and splits prescribed motion into a new sibling composite. Sibling
in spirit to [ADR 0049](0049-user-declared-nodes.md) (new `g.nodes`
authoring composite) — same "authoring composite resolves into `fem`"
pattern. Forces the bridge-side emit half tracked as **LOAD-1** /
**DOC-1** in `internal_docs/todo_apesees.md`.

## Context

`g.loads` today is a **flat list of eight methods** that mixes two
unrelated axes:

- *dimension-named*: `line`, `surface`
- *semantics-named*: `point`, `point_closest`, `gravity`, `body`,
  `face_load`, `face_sp`

The sibling `g.masses` composite is already **dimension-indexed**
(`point` / `line` / `surface` / `volume`). `g.loads` is the outlier, and
the flat surface buries real structure:

1. Every `LoadDef` is in fact **typed by the dimension of its target**
   (the resolver carries an `expected_dim`: point = any, line = 1,
   surface = 2, volume = 3). The dimension spine exists; the API hides it.
2. `surface` overloads a `normal=True/False` **boolean** that silently
   re-interprets what `magnitude`/`direction` mean — the single most
   confusing knob in the loads API.
3. `face_sp` is **not a load** — it is a prescribed displacement (an `sp`
   in an OpenSees load pattern). It is mis-filed under loads.
4. `gravity` is filed as volume-only, but self-weight is genuinely
   **cross-dimensional**: a beam (ρ·A·g per length), a shell (ρ·t·g per
   area), and a solid (ρ·g per volume) all have self-weight.

Separately, **LOAD-1**: the apeSees bridge currently *silently ignores*
all `g.loads` declarations (the documented `body_force=` workaround
bypasses the tributary/consistent choice and is not equivalent). The
bridge-emit half was never wired. This ADR's cross-dim gravity **cannot**
be expressed without building that half, so the restructure is the
forcing function to fix LOAD-1 properly rather than paper over it.

## Decision

### 1. Four single-purpose composites, all dimension-indexed

| Composite | Owns | Dimension spine |
| --- | --- | --- |
| `g.loads` | **force** (concentrated, distributed, body, self-weight) | point / line / surface / volume |
| `g.displacements` *(new)* | **prescribed motion** (non-homogeneous `sp`) | point / surface |
| `g.constraints` | relations + **homogeneous** fixes | (existing) |
| `g.masses` | inertia | point / line / surface / volume |

### 2. `g.loads` surface (hard rename)

```python
# point (dim 0) — namespace, force/moment × named/coordinate
g.loads.point.force(target, F)
g.loads.point.moment(target, M)
g.loads.point.force_closest(xyz, F)
g.loads.point.moment_closest(xyz, M)

# line (dim 1) — single verb → plain callable
g.loads.line(target, ..., reduction=, target_form=)

# surface (dim 2) — namespace; three per-area fields + one resultant
g.loads.surface.pressure(target, p, ...)        # scalar × face normal (out-of-plane)
g.loads.surface.traction(target, vec, ...)      # free GLOBAL vector / area
g.loads.surface.shear(target, vec, ...)         # strict IN-plane (tangential)
g.loads.surface.force_resultant_center_mass(target, force=, moment=)

# volume (dim 3) — single verb → plain callable
g.loads.volume(target, force_per_volume=, reduction=, target_form=)

# self-weight — cross-dimensional, top-level
g.loads.gravity(target, g=(0,0,-9.81), density=None, reduction=, target_form=)
```

**Single-verb dimensions (`line`, `volume`) are plain callables**;
multi-verb dimensions (`point`, `surface`) are namespaces. This mirrors
`g.masses` (flat dim methods) where there is one verb, and only adds
nesting where there is genuinely more than one.

### 3. Surface: kill the `normal=` boolean → three explicit verbs

- `pressure` — scalar along the per-face normal (orientation-following).
- `traction` — a free vector per area in **global** coordinates,
  independent of face orientation (the common "slab live load is always
  `(0,0,-w)`" case).
- `shear` — strict **in-plane** (tangential) traction. Input is a global
  reference vector **projected onto each face's tangent plane** at resolve
  time (normal component subtracted), so one call works across a
  faceted/curved surface with no per-face basis from the user. A face
  where the projection vanishes (load is purely normal there) is
  **fail-loud**.

`pressure` / `traction` / `shear` are per-area **fields**;
`force_resultant_center_mass` is a **lumped resultant** (total force/moment
at the area centroid, distributed to nodes — the old `face_load`). The
name is deliberately verbose to flag "resultant, not a distribution."

### 4. Reduction is a property of *field* loads only

**If it is a per-length / per-area / per-volume field, it carries
`reduction` (tributary | consistent) and `target_form` (nodal | element).
If it is a point or a resultant, it does not.**

| Verb | `reduction` | `target_form` |
| --- | :---: | :---: |
| `point.*` | — | — |
| `line` | ✅ | ✅ |
| `surface.pressure` / `traction` / `shear` | ✅ | ✅ |
| `surface.force_resultant_center_mass` | — | — |
| `volume` | ✅ | ✅ |
| `gravity` | ✅ | ✅ |

`reduction="consistent"` on volume/gravity is **behaviorally identical to
tributary for tet4/hex8** (same per-node share for a constant body force);
it only diverges for higher-order (tet10/hex20). Documented as such — not
pretended to change linear-element results.

### 5. Nodal is the default; element form is the load-bearing escape hatch

`target_form="nodal"` reduces the field to `NodalLoadRecord`s mesh-side.
`target_form="element"` emits an `ElementLoadRecord` (`beamUniform`,
`surfacePressure`, `bodyForce`) and defers integration to OpenSees
`eleLoad`. Element form is **not optional** in two cases:

1. **Frame-element distributed loads** — lumping a UDL on a beam to its
   end nodes loses the member's **fixed-end moments** (wrong internal
   moment diagram). `eleLoad beamUniform` is the structurally correct path.
2. **Cross-dim gravity on beams/shells** — ρ·A and ρ·t need the section
   (area / thickness), which lives in the **bridge**, not mesh-side. So
   dim-1/dim-2 gravity *must* emit as element `bodyForce` for the bridge
   to expand. Only dim-3 gravity (needs ρ only) reduces nodal mesh-side.

### 6. `g.displacements` — new composite

Prescribed displacement is pattern-bound (a time-varying `sp` under a load
pattern), distinct from `g.constraints.bc` (a permanent homogeneous fix).

```python
g.displacements.point(target, dofs=, values=, pattern=)
g.displacements.surface(target, disp_xyz=, rot_xyz=, ...)   # was face_sp
```

**Ownership rule for a zero prescribed displacement** (the degenerate
overlap with `bc`): `g.constraints.bc` owns permanent homogeneous fixes;
`g.displacements` owns anything with a nonzero or time-varying value. A
zero authored through `g.displacements` is allowed (a pattern-bound hold)
but is the user's explicit choice, not a silent alias for `bc`.

## Rationale

- **The dimension spine already exists** in the resolver; this ADR exposes
  it instead of hiding it, and unifies `g.loads` with the already
  dimension-indexed `g.masses`. One mental model across both composites.
- **The `normal=` boolean was the actual complexity**, not the physics.
  Three named verbs are each unambiguous; the boolean overloaded the
  meaning of two other arguments.
- **`face_sp` was a category error.** Force and prescribed motion are
  different physics with different OpenSees emit (`load` vs `sp`); separate
  composites stop the leak.
- **Cross-dim gravity is the honest model of self-weight** and the
  forcing function for LOAD-1 — it makes the bridge-emit half unavoidable,
  so we build it correctly instead of shipping the `body_force=` workaround
  forever.

## Consequences

- **Persistence/round-trip is largely untouched.** The underlying
  `LoadDef` dataclasses and their **`kind` strings** (`point`, `line`,
  `surface`, `gravity`, `body`, `face_load`, `face_sp`) are the serialized
  identity and the resolver-dispatch key. The rename is an **authoring-
  surface** change; `model.h5` round-trip and the resolver dispatch table
  stay stable. New behavior adds: a surface `mode`
  (`pressure`/`traction`/`shear`) replacing the `normal` bool, and a
  cross-dim flavor on `GravityLoadDef`.
- **Hard rename breaks** existing scripts, the apegmsh-helper skill, and
  every loads doc the day it lands — by explicit decision (no deprecation
  shim). Migration is mechanical (old → new method map in the plan).
- **`g.displacements` is a new session composite**, resolving into
  `fem.nodes` like `g.loads` / `g.nodes` — same authoring pattern as
  [ADR 0049](0049-user-declared-nodes.md).
- **LOAD-1 closes** as part of P4/P5: the bridge stops silently ignoring
  `g.loads` and emits the resolved records (nodal `load` / `sp`, element
  `eleLoad`). **DOC-1** (guide ⇄ skill contradiction on auto-emit) and
  **DOC-2** (namespace docstrings) reconcile against the new, true
  behavior.
- **`surface.shear` is genuinely new physics** (in-plane projection), not
  a rename — the only verb without a one-to-one old counterpart.

## Open questions

1. **`force_resultant_center_mass` length.** Accepted as the explicit
   name; revisit if a shorter unambiguous form emerges. It is the area
   centroid, not the mass center — name is by user choice, documented.
2. **`shear` input form.** Global vector projected per face (decided), but:
   reject vs. warn when the user's vector is *mostly* normal (small
   tangential residual → noisy direction)? Lean: fail-loud below a
   tangential-fraction floor.
3. **`g.displacements` def types.** Reuse `FaceSPDef` (kind `face_sp`) for
   v1, or mint dedicated displacement defs? Lean: reuse for v1 (keeps the
   resolver/dispatch stable), revisit if the point variant needs its own.
4. **`density=None` source for gravity.** Bridge reads ρ from the assigned
   material/section. Needs the section-introspection seam that LOAD-1's
   bridge half introduces — confirm one path serves both gravity and the
   general element-form emit.

## Related

- [ADR 0049](0049-user-declared-nodes.md) — new `g.nodes` authoring
  composite; same "composite → `fem`" pattern this ADR follows for
  `g.displacements`.
- `internal_docs/todo_apesees.md` — **LOAD-1** (silent-ignore bridge half,
  closed here), **LOAD-2** (body_force docs), **DOC-1** (auto-emit
  contradiction), **DOC-2** (namespace docstrings).
- `internal_docs/plan_loads_displacements_restructure.md` — phased
  implementation plan for this ADR.
- `g.masses` (`core/MassesComposite.py`) — the dimension-indexed precedent
  this ADR aligns `g.loads` with.
