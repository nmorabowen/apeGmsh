# Plan — Dimension-indexed loads + `g.displacements` split

Implementation plan for [ADR 0050](../src/apeGmsh/opensees/architecture/decisions/0050-dimension-indexed-loads-and-displacements.md).
Closes **LOAD-1** and reconciles **DOC-1 / DOC-2** from
`todo_apesees.md`.

**Guiding invariant:** the `LoadDef` dataclasses and their `kind` strings
are the serialized identity and the resolver-dispatch key. Keep them
stable wherever possible — most phases are an **authoring-surface** rewrite
over an unchanged resolver, so `model.h5` round-trip and the bridge
dispatch are unaffected until P3/P4 deliberately extend them.

## Old → new method map (migration reference)

| Old (`g.loads.*`) | New |
| --- | --- |
| `point(force_xyz=)` | `point.force(target, F)` |
| `point(moment_xyz=)` | `point.moment(target, M)` |
| `point_closest(force_xyz=)` | `point.force_closest(xyz, F)` |
| `point_closest(moment_xyz=)` | `point.moment_closest(xyz, M)` |
| `line(...)` | `line(...)` *(unchanged signature)* |
| `surface(normal=True)` | `surface.pressure(...)` |
| `surface(normal=False, direction=)` | `surface.traction(...)` |
| *(none)* | `surface.shear(...)` **(new physics)** |
| `face_load(...)` | `surface.force_resultant_center_mass(...)` |
| `body(...)` | `volume(...)` |
| `gravity(...)` *(volume-only)* | `gravity(...)` *(cross-dim)* |
| `face_sp(...)` | `g.displacements.surface(...)` |

## Phases

### P1 — `g.loads` authoring rewrite (pure rename, no behavior change) ✅ SHIPPED
The keystone, zero new physics. Landed on branch `guppi/apesees-todo-list`
(commits 05d1f7e9 engine+tests+skill, 385ef734 docs sweep). Deferred from
P1 scope: `SurfaceLoadDef.normal: bool` → `mode` field stays a **bool** until
P3 (a bool can't carry the third `shear` state; keeping it preserves the
"resolver untouched" invariant). `face_sp` stays on `g.loads` until P2.

1. Build the `point` and `surface` namespace objects + plain-callable
   `line` / `volume`. Each verb constructs the **same existing `LoadDef`**
   with the same `kind`. `surface.pressure/traction` map to the old
   `normal=True/False` internally; `force_resultant_center_mass` →
   `FaceLoadDef`.
   → verify: every old call has a new call producing a byte-identical
   `LoadDef` (parametrized equivalence test, old-vs-new).
2. Replace `SurfaceLoadDef.normal: bool` with `mode:
   Literal["pressure","traction","shear"]`; `pressure`→normal,
   `traction`→vector. Keep `kind="surface"`.
   → verify: resolver dispatch unchanged for pressure/traction;
   `model.h5` round-trip of a surface load is stable.
3. Delete the flat methods (hard rename). Update apegmsh-helper skill +
   all `guide_loads.md` / `guide_*` examples.
   → verify: `pytest tests/` loads suite green; grep shows no
   `g.loads.point(` / `.face_load(` / `.body(` survivors in docs/skill.

### P2 — `g.displacements` composite ✅ SHIPPED
Landed: `DisplacementsComposite` (`surface` = relocated face_sp,
`point` = new `PointSPDef`), `resolve_point_sp`, fem_factory + pre-mesh
wiring, face_sp removed from `g.loads`. Target resolution **delegates**
to `g.loads` (a shared mixin is the future refactor, not done). Docs +
2 notebooks swept. Decision: reused `FaceSPDef` (kind `face_sp`) for the
surface verb per the plan (resolver dispatch stays stable).
4. New `DisplacementsComposite` (sibling registration on the session,
   mirroring `g.loads`). v1 reuses `FaceSPDef` (kind `face_sp`); add a
   `point` variant (prescribed `sp` at a node).
   → verify: `face_sp` resolution path unchanged (same records); a
   prescribed-disp E2E emits OpenSees `sp` under the right pattern.
5. Remove `face_sp` from `g.loads`. Document the `bc` vs `displacements`
   ownership rule (zero/permanent → `bc`; nonzero/time-varying →
   `displacements`).
   → verify: a zero `g.displacements` call is allowed; docs state the rule.

### P3 — `surface.shear` (new in-plane physics)
6. Resolver: per-face tangent-plane **projection** of the global reference
   vector (subtract normal component); fail-loud when the tangential
   residual falls below a floor.
   → verify: flat face in a tilted plane → shear lies in-plane, magnitude
   preserved; purely-normal input → raises.
7. `consistent` + `element` (`surfacePressure`-class) forms for shear.
   → verify: tributary vs consistent on a quad8 shear field; element-form
   record shape.

### P4 — cross-dim gravity + the LOAD-1 bridge half
The meaty one — needs the bridge.

8. `GravityLoadDef` accepts dim 1/2/3 targets. dim-3 reduces nodal
   mesh-side (ρ only). dim-1/2 emit **element `bodyForce`** carrying `g` +
   `density` (or `None` → bridge reads section).
   → verify: dim-3 gravity nodal totals == ρ·V·g; dim-1/2 produce element
   records (no nodal lumping attempted without a section).
9. **Bridge emit (LOAD-1 close):** apeSees stops silently ignoring
   `g.loads`. Wire the resolved record stream → OpenSees:
   nodal `load`, element `eleLoad` (`beamUniform` / `surfacePressure` /
   `bodyForce`). Section-introspection seam supplies A/t/ρ for element-form
   gravity.
   → verify: a beam under `g.loads.line(target_form="element")` shows
   correct fixed-end moments in results; a beam/shell `gravity` self-weight
   reaction == ρ·A·L·g / ρ·t·A·g; **a `g.loads.point.force` actually moves
   the model** (the LOAD-1 regression).

### P5 — docs reconciliation (DOC-1 / DOC-2)
10. Rewrite `guide_loads.md` + `guide_opensees.md` against the *true*
    post-P4 emit behavior (kills the DOC-1 auto-emit contradiction). Add a
    2-D `body_force`/volume example (LOAD-2). Fill the `ops.*` namespace
    method docstrings (DOC-2).
    → verify: skill ⇄ guide say the same thing about emit; `todo_apesees.md`
    marks LOAD-1/DOC-1 DONE (LOAD-2/DOC-2 as addressed).

## Sequencing notes

- **P1 is shippable alone** (rename only) and unblocks the skill/doc churn
  early. P2 is independent of P1's internals.
- **P4 depends on the bridge section-introspection seam.** If that seam is
  larger than expected, dim-3 gravity (step 8, mesh-side) can ship before
  the dim-1/2 element-form half (step 9).
- **P3 `surface.shear`** is fully independent — can land any time after P1.
- Each phase is its own PR off `main` (per the PR-base rule), not stacked.
