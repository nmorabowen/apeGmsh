# Future: Derived-Field Computation

**Status:** future · **Estimated cost:** ~1–2 weeks · **Depends on:** plan 04
**Discussed in:** [`../diagram-catalog-comparison.md`](../diagram-catalog-comparison.md)

## Goal

Compute and visualize derived fields from recorded components on the fly: Von Mises,
max shear, principal stresses, yield ratio, damage indicators. Tiered: a registry of
built-in named invariants now, optional expression engine later.

## Why

- Today: every field the user wants to plot must be **explicitly recorded** in the
  OpenSees declaration. There's no way to color by `sqrt(s_xx² + s_yy² - s_xx·s_yy + 3·s_xy²)`
  (Von Mises) on the fly.
- This is **THE biggest single FEM-postprocessor gap.** Abaqus CAE, STKO, Femap all do
  this natively. The "barebones" complaint about `results.viewer` is largely this.

## Mechanism

Two-tier approach:

- **Tier 1 (now):** named invariant registry. Functions like `stress_vm`,
  `stress_max_shear`, `stress_principal_1/2/3`, `yield_ratio`, etc. Each is a pure
  numpy function: `tensor_slab (N,6) → scalar (N,)`. Registry maps name → function.
- **Tier 2 (later):** expression strings via `numexpr` or `vtkProgrammableFilter`. Defer
  until Tier 1 proves the pattern.

In `ContourStyle.component`, allow either a recorded component name OR an invariant
name. The selector layer routes to the right code path.

## Files

- New: `src/apeGmsh/results/_invariants.py` — pure numpy functions.
- Modify: `src/apeGmsh/viewers/diagrams/_kind_catalog.py` — surface invariants in the
  Component combo when the parent tensor is recorded.
- Modify: `src/apeGmsh/viewers/diagrams/_contour.py` — route invariant names to the
  invariant computer instead of a raw slab read.

## Risks

- **Tensor conventions must be FEM-correct.** Sign conventions, plane-stress vs full 3D,
  shell vs solid layer ordering, OpenSees vs STKO storage. Document and test against
  reference cases.
- **Performance.** Computing Von Mises across N cells × T stages is `O(N·T·6)`. For
  large analyses, cache per-step on first compute.
- **Yield ratio** requires a yield-stress reference; comes from material, not from the
  recorder. Need a material lookup mechanism.

## Done criteria

- "Von Mises stress" appears as a component option in Contour diagrams whenever the
  recorded stress tensor is present.
- Principal stresses (σ1, σ2, σ3) appear as options when the tensor has 6 components.
- Yield ratio appears when both stress tensor and material yield stress are available.
- Unit tests against analytical references.

## Out of scope (Tier 2)

- Full expression engine.
- User-saved custom invariants via session JSON.
- Damage models specific to particular constitutive laws — keep generic for v1.
