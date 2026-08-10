# ADR 0091 — Bernstein-aware consistent load reduction (`basis=` on field loads)

**Status:** Accepted (2026-08-10). Extends the ADR 0050 authoring surface
and the ADR 0051 record contract; changes no bridge behavior. Trigger:
the TIMs T2 incident (2026-08-10) on the Ladruno fork's Bézier elements.

## Context

`reduction="consistent"` integrates a distributed load against the
**Lagrange** shape functions of the gmsh element
(`_kernel/_consistent_quadrature.py`). For a quadratic tri6 face under
uniform `q` that gives corners ~0 and midsides `q·A/3` — correct when the
target element's DOFs are **nodal values** (TenNodeTetrahedron,
SixNodeTri).

The Ladruno fork's `BezierTet10` / `BezierTri6` interpret the same gmsh
connectivity as Bernstein **control values** (straight-sided v1: control
points coincide with the gmsh nodes, connectivity verbatim — see
`opensees/element/solid.py` and `apeGmsh._basis`). Applying the
Lagrange-consistent vector to control values represents a strongly
oscillatory traction: the resultant is exact (partition of unity), but
locally it spikes. On a 3775-element strip-footing deck this drove
near-surface DruckerPrager Gauss points into apex/tension and the run
diverged at first yield, while TenNodeTetrahedron on the identical mesh
converged (TIMs T2). The correct vector is `f_a = ∫ t·B_a dΓ` with the
Bernstein basis `B` in the same Gauss loop; for a quadratic Bernstein
simplex every basis function integrates to `measure/n` (`A/6` per tri6
face node, `L/3` per line3 node, `V/10` per tet10 node), so uniform `q`
maps to **equal** control-point loads. With that vector the same deck
converges the full surcharge in one step.

**Where could the reduction learn the target element family?** Nowhere,
by design: loads resolve session-side into `NodalLoadRecord`s at
`get_fem_data()` time (ADR 0050), while the element family is chosen
later on the bridge (`ops.element.BezierTet10(pg=...)`, ADR 0051 opt-in
`p.from_model`). A post-hoc remap on the bridge is impossible — the
Lagrange→Bernstein map mixes nodes *within a face*, and the per-node
records have already accumulated across faces. So the seam is
**authoring-side**, like `reduction=` itself.

## Decision

1. **`basis="lagrange" | "bernstein"`** on the field-load verbs whose
   consistent quadrature it changes: `g.loads.line`,
   `g.loads.surface.pressure/traction/shear`. Default `"lagrange"`
   (existing behavior, bit-identical). Stored on `LoadDef.basis`; defs
   are not persisted, so no schema change.
2. **Quadrature branch** in `_consistent_quadrature.py`: Bernstein
   line3 + tri6 shape functions (values *and* derivatives — the
   Jacobian uses the element's own Bernstein geometry map, identical to
   Lagrange on the straight-sided meshes v1 validates). Node-order
   contract, **verified against the fork sources** (`BezierTet10.cpp`
   `shapeFunctions`, `BezierTri6.cpp`): gmsh connectivity verbatim,
   corner `a` ↦ `L_a²`, midside of edge `(a,b)` ↦ `2·L_a·L_b`. The
   face-local assignment is orientation-independent (each face node's
   Bernstein function is the trace of its volumetric one), so the tri6
   surface-mesh node order needs no permutation.
3. **Fail-loud everywhere else**: `basis="bernstein"` with
   `reduction="tributary"` or `target_form="element"` raises at
   authoring (`_add_def`); a quad face (quad4/8/9) raises
   `NotImplementedError` (the fork Bézier family is simplex-only); an
   unknown basis string raises. Degree-1 targets (line2/tri3) are
   accepted — the bases coincide.
4. **Gravity / volume loads get no knob**: the equal split the
   tributary/consistent path already emits **is** the exact
   Bernstein-consistent vector for a constant body force
   (`∫B_a dV = V/10`, `∫B_a dA = A/6`). Documented on
   `resolve_gravity_consistent`. (For higher-order *Lagrange* elements
   the equal split remains the known resultant-exact approximation.)

## Audit — other distributed→nodal conversions (T2 follow-through)

| Path | Bézier status |
| --- | --- |
| `LoadResolver.element_volume` (gravity/volume loads) | **Fixed here**: 10/20/27-node rows fell to the bounding-box volume (~6x overshoot on a tet10) — the isoparametric `∫\|J\|dξ` path `MassResolver` already had is now mirrored. Straight-sided Bézier = affine ⇒ exact. |
| Gravity/volume equal split | Bernstein-exact (see §4). |
| `g.masses` HRZ lumping (`MassResolver`, `_hrz.py`) | HRZ weights use **Lagrange** `∫N_I²`; the Bernstein-HRZ weights differ (tri6: corners 1/5 vs edge 2/15 of `ρA`). Both are positive, total-preserving diagonal approximations — no oscillatory-traction cliff, and the fork elements compute their own consistent/lumped mass from `rho` (`-cMass`), which is the recommended route for Bézier regions. **Documented, not changed.** |
| `surface.force_resultant_center_mass` / `face_sp` least-norm | Resultants, not fields — any statically equivalent split is admissible; equal split on control values preserves the resultant by partition of unity. No change. |
| `integrate_edge_scaled` (varying line loads) | Carries `basis=` too — the scalar field is sampled at the Gauss points and integrated against the selected basis. |

## Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| Bridge-side auto-remap when the pattern imports onto Bézier elements | Impossible post-accumulation (the basis change mixes nodes within a face; records have summed across faces). Re-resolving on the bridge would need mesh-side face connectivity + the resolver — a layering inversion of ADR 0050/0051. |
| Tag records with their basis and *warn* at `build()` on a mismatch | Adds a `NodalLoadRecord` field (schema churn) for a heuristic warning that cannot see defs → deferred; the fail-loud authoring validation plus docs carry the weight. Revisit if a second incident shows the knob being forgotten. |
| Auto-detect: session-side elements know their gmsh type | The gmsh type (tet10) is the same for both targets — the *solver element* decides Lagrange vs Bernstein, and the session never knows it. |

## Consequences

- Default path is bit-identical; only opt-in `basis="bernstein"` changes
  numbers. No H5 schema change (defs are session-only).
- `loads.summary()` gains a `basis` column.
- Live gate: `tests/opensees/integration_ladruno/
  test_bezier_consistent_loads_live.py` — the Bernstein vector is the
  exact uniform-compression patch test on BezierTet10 (`u_z = -qL/E`
  at every top control point, rtol 1e-6) and the Lagrange vector on the
  same mesh visibly oscillates (the T2 mechanism, reproduced clean).
- Unit gate: `tests/test_consistent_reduction.py` — equal `A/6` / `L/3`
  Bernstein weights, unchanged Lagrange patterns, simplex-only refusal,
  authoring validation, and the tet10 isoparametric volume fix.

## Related

- [ADR 0050](0050-dimension-indexed-loads-and-displacements.md) — the
  authoring surface this extends (reduction as a field-load property).
- [ADR 0051](0051-bridge-load-consumption.md) — why the bridge cannot
  branch (records are the contract; loads are opt-in imports).
- `apeGmsh._basis` — the pinned Bernstein node-order contract shared
  with the `.ladruno` reader.
- Ladruno fork `SRC/element/bezierTetrahedron/BezierTet10.cpp`,
  `SRC/element/bezierTriangle/BezierTri6.cpp` — the verified element
  conventions.
