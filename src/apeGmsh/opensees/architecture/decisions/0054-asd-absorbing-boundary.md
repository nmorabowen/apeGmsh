# ADR 0054 — ASDAbsorbingBoundary as an extruded skin + staged absorbing flip

**Status:** Proposed (2026-06-07). Not implemented. Supersedes the working
assumption (reached in discussion, then **refuted by an adversarial source
review**) that the absorbing layer could be the reinterpreted outermost ring of
the soil box. Grounded in the fork guide
`Ladruno_implementation/absorbing_boundaries_and_pml_guide.md` and direct reads
of `SRC/element/absorbentBoundaries/ASDAbsorbingBoundary{2D,3D}.cpp/.h`. Pairs
with the existing `DRMBox` part (`parts/drm_box.py`) and the SSI/staged ADR set
([0028](0028-initial-stress-via-parameter-ramping.md) /
[0029](0029-staged-analysis-context-manager.md) /
[0030](0030-stage-bound-topology-activation.md) /
[0034](0034-stage-bound-bcs-and-recorders.md)). First slice: **3D only**
(`ASDAbsorbingBoundary3D`); 2D follows.

## Context

`ASDAbsorbingBoundary2D` (4-node quad) / `ASDAbsorbingBoundary3D` (8-node brick)
are ASDEA's production wave-absorbing boundary elements for soil–structure
interaction. They are a **one-element-thick layer** wrapping the truncation
surface, combining Lysmer–Kuhlemeyer dashpots with an enforced **free-field
column** (1D site response + spring/mass), and they are **staged**: stage 0
(`Stage_StaticConstraint`) holds the boundary with a penalty during gravity;
stage 1 (`Stage_Absorbing`) activates the dashpots + free-field. The flip is
live, one-way (`setParameter -val 1 -ele <tags> stage`). The canonical SSI model
is interior → DRM ring (injects the incident field) → absorbing layer just
outside (lets the scattered field out).

Source-verified element grammar:

```
element ASDAbsorbingBoundary2D $tag $n1..$n4  $G $v $rho $thickness $btype  <-fx $ts> <-fy $ts>
element ASDAbsorbingBoundary3D $tag $n1..$n8  $G $v $rho            $btype  <-fx $ts> <-fy $ts> <-fz $ts>
```

- Raw doubles `G, v, rho` — **not** a material tag (`3D:499-504`). `thickness`
  is 2D-only.
- `$btype` = OR-combinable face string `{B,L,R,F,K}` (3D) / `{B,L,R}` (2D)
  (`3D:507-513`).
- `-fx/-fy/-fz` time series are parsed and consumed **only on bottom (`B`)
  elements** (`3D:526`, `addBaseActions` early-returns otherwise `3D:2546`); they
  inject a base **velocity** history (`3D:2550-2582`), not a force.
- The element **self-sorts** its nodes from coordinates in `setDomain`
  (`3D:741-821`), so there is no input winding-order contract.
- Standard solid DOFs (ndf ≥ 3 in 3D, ≥ 2 in 2D).

## Decision

Implement the absorbing boundary as an **extruded extra skin** outside an intact
axis-aligned structured box, emit one element per skin cell with a
construction-derived `btype`, supply `G/v/rho` as raw floats, and flip the stage
via an **explicit** staged verb.

1. **Geometry: build a real extra skin; never reinterpret the soil ring.** A
   geometry-time offset-shell core adds a one-element layer outside the
   axis-aligned box's truncation faces (excluding the local +Z free surface),
   sharing inner nodes with the soil and meshing conformally. Two entry points
   ride it: a turnkey **`PlainWaveBox`** (`add_plane_wave_box`) and a
   bring-your-own-box **`add_absorbing_shell`**. **DRMBox is not involved** — it
   serves the Domain Reduction Method (a separate facility that *pairs* with the
   skin in a model: DRM injects, the skin absorbs). The soil box stays fully
   intact. See [the AB-1 plan](../../../../../internal_docs/plan_absorbing_skin_ab1.md).
2. **btype by construction, one PG per combo.** The extruder knows which face(s)
   each skin cell came from, so it tags each cell with the exact topological
   `btype` (face `L/R/F/K/B`, vertical edge `LF/LK/RF/RK`, bottom edge
   `BL/BR/BF/BK`, bottom corner `BLF/BLK/BRF/BRK`) and emits **one physical group
   per distinct btype**. Each PG fans out to a vanilla frozen
   `ASDAbsorbingBoundary3D` element with a fixed `btype: str` — preserving the
   homogeneous "one PG → one element type, identical args" fan-out the bridge
   already supports. The base input series attach to **every btype containing
   `B`** (`B`, `BL`, `BR`, `BF`, `BK`, `BLF`, …), one series per active direction
   — only the wired directions are emitted (the STKO reference uses `-fx` alone
   for 1D horizontal input).
3. **Material: both modes, collapse to raw floats, no dependency.** Accept
   `material=ElasticIsotropic(...)` (derive `G = E/(2(1+ν))`, reuse ν, ρ at
   construction) **or** raw `G=/v=/rho=`. The derived values are stored as plain
   floats on the frozen spec; the material is **not** returned from
   `dependencies()` (it would emit a stray orphan `nDMaterial` or trip the
   reachability guard, since the element emits raw doubles).
4. **Stage flip: explicit `s.activate_absorbing()`.** A staged record emits the
   STKO-confirmed `parameter`-object idiom — `parameter <pid>` →
   `addToParameter <pid> element <eid> stage` for every tracked absorbing element
   → `updateParameter <pid> 1` → `remove parameter <pid>` — once, in the dynamic
   stage, after the gravity stages' `loadConst`. This is the **exact plumbing
   apeGmsh's `s.initial_stress` already emits** (`ParameterRecord` /
   `AddToParameterRecord`), so reuse is direct; the equivalent
   `setParameter -val 1 -ele <tags> stage` one-liner also works. The flip is
   emitted **per partition** (each rank flips its owned elements — STKO does
   exactly this), and the user never types a tag. Auto-detection is rejected
   (see below).
5. **Fail-loud guards.** Refuse non-axis-aligned / non-Z-rotated boxes and
   skewed/degenerate skin cells; warn on high-aspect-ratio skin cells.

## Why not reinterpret the outer ring (the refuted alternative)

We initially favored reinterpreting the box's outermost element ring as absorbing
(no new geometry). An adversarial source review killed it: the element is a
**half-space surrogate**, not a soil hex.

- **Zero gravity weight/mass in stage 0.** `getMass` returns zero unless
  absorbing (`3D:938`); `addInertiaLoadToUnbalance` is a no-op (`3D:947-951`).
  Reinterpreting the outer soil ring **deletes its self-weight and K0
  contribution from the gravity solve → corrupted in-situ stress at the
  boundary**.
- **The element owns the fiction of soil beyond it.** Nodes split soil-side
  `{2,3,6,7}` vs free-field-exterior `{0,1,4,5}` (`3D:192-193`); the free-field
  column treats the element's own volume as exterior soil (`addMff` `3D:1812`,
  `addKff` `3D:1948-2014`). Reinterpreting shifts the effective soil column
  inward one element and discards the real ring's continuum stiffness/mass.
- **It eats the DRM buffer** between the interior and the absorbing layer.

STKO extrudes an extra skin for exactly this reason: the full continuum domain
must stay intact to produce the correct gravity/K0 field and give DRM a clean
straddling ring. Extruding is also the *easier* path to get right — btype falls
out by construction instead of from a fragile centroid/plane classifier (region
labels can't even distinguish `L` from `R`; `Axis1D` labels both ±X as
`"outer"`).

## Source-grounded gotchas the implementation must honor

- **btype is the cell's topological role, with no default branch.** The 17-case
  constraint blocks (`3D:1524-1619`), dashpot selection (`LKselectPairs`
  `3D:264-290`), and free-field mapping all branch on the exact `m_boundary`
  value; an illegal/wrong combo parses, sorts, runs, and **silently** applies
  wrong/no constraints. Edge/corner cells must carry the OR-combined btype, never
  a single dominant face.
- **Axis-aligned + non-degenerate is an unstated contract.** 3D
  `handleDistortion` (`3D:376-469`) silently idealizes a skewed hex into an
  axis-aligned brick (changing the synthesized free-field column / dashpot
  areas); a singular Jacobian hard-`exit(-1)`s (`3D:396-399`); 2D has no
  distortion handling at all. → fail-loud guards.
- **Stage flip ordering.** `updateStage` (`3D:1487-1499`) bakes the current
  displacement (`m_U0`) and frozen stage-0 reactions (`m_R0`) at flip time, and
  re-injects `m_R0` every transient step (`addRReactions` `3D:1786-1798`). The
  flip must happen **after** gravity convergence + `loadConst`, **before** the
  transient, on **all** absorbing tags together. Any other value or a second
  flip `exit(-1)`s.
- **Free-field column damping rides global Rayleigh.** `addCff` reads
  `alphaM/betaK` (`3D:2315-2326`). Do **not** zero-Rayleigh the absorbing region
  or the free-field columns lose internal damping (the L-K dashpots still act).
- **`-fx/-fy/-fz` are base velocities** on `B*` cells only — document units.

## Validation against a real STKO export

Cross-checked against an STKO-exported deck (`waveletExample`: a 3D SSI model,
1936 `ASDAbsorbingBoundary3D` elements, 2-partition Mumps/KrylovNewton implicit
run with a base wavelet). Every load-bearing decision is confirmed:

- **Extra extruded skin, not reinterpret.** STKO generates *new* nodes for the
  layer (ids 8132+), including nodes *below* the soil-box bottom; each element
  lists **4 existing soil nodes + 4 new outer nodes**. The soil box is left
  intact. This is the definitive confirmation that reinterpret-the-ring is wrong.
- **btype scheme is exactly the predicted set.** Observed distribution: faces
  `L`(320) `R`(320) `F`(352) `K`(352); vertical edges `LF/LK/RF/RK`(16 each);
  bottom face `B`(440); bottom edges `BL`(20) `BR`(20) `BF`(22) `BK`(22); bottom
  corners `BLF/BLK/BRF/BRK`(1 each). **No top** — free surface excluded.
- **Raw `G v rho`, G = shear modulus from the soil elastic.** Elements carry
  `1351.03 0.262 2.4e-09`; the soil is `nDMaterial ElasticIsotropic 21 3400.0
  0.262 2.4e-09`. ν and ρ pass straight through; `G ≈ E/(2(1+ν))`. Uniform across
  all absorbing elements → fits the homogeneous per-PG fan-out.
- **Base input on all B-containing cells, one direction.** All 528 bottom-side
  elements (every btype with `B`) carry `-fx 5`; `timeSeries Path 5` is the
  wavelet (a velocity history, `-factor 9810`). Only `-fx` is wired (1D). The
  `pattern UniformExcitation` lines exist but sit **after `exit`** — dead code;
  the live seismic input is the absorbing-element base action. (Ties into the
  Ricker/Path primitives from the wavelet work.)
- **Stage flip = the initial-stress plumbing, per partition.**
  `parameter -2001` → `addToParameter -2001 element $eid stage` (looped over the
  rank's absorbing element list) → `updateParameter -2001 1` → `remove parameter
  -2001`, guarded per `process_id`, emitted *between* the static stages
  (`loadConst -time 0.0; wipeAnalysis`) and the transient. Confirms: explicit,
  all-together, ordered after gravity, and per-rank.
- **Rayleigh is assigned to the absorbing region.** `region 7 ... -rayleigh
  1.4265 0.0 1.72e-4 0.0` — so the free-field column damping (`addCff`) is fed by
  Rayleigh exactly as the gotcha warns; the region must not be zero-Rayleigh.
- **Implicit run** (Mumps + KrylovNewton + LoadControl/Newmark) — the ASD
  element's primary supported path.

Net: no decision in this ADR needs reversal; the refinements folded in above
(base input on every `B*` cell + one-series-per-direction; the `parameter`-object
flip idiom; per-partition flip is the real idiom, not merely a deferral).

## Slice plan

- **AB-1 (mesh):** geometry-time offset-shell core (one-element shell on the 5
  truncation faces, local +Z excluded, conformal node-sharing, per-cell btype by
  grid position, one PG per combo, fail-loud axis-alignment/quality guards) +
  two entry points: turnkey `add_plane_wave_box` (`PlainWaveBox`) and
  bring-your-own-box `add_absorbing_shell`. The naive per-quad-extrusion is wrong
  (leaves edge gaps; STKO has `LF`/`BLF` cells) — see
  [the AB-1 plan](../../../../../internal_docs/plan_absorbing_skin_ab1.md).
- **AB-2 (bridge):** ✅ **DONE.** `ASDAbsorbingBoundary3D` frozen `Element`
  (`opensees/element/absorbing.py`) — raw `G/v/rho`, fixed `btype` (illegal/
  opposite/repeated letters rejected), optional `-fx/-fy/-fz` guarded to bottom
  PGs. Facades `ops.element.ASDAbsorbingBoundary3D` (material= derives
  `G=E/2(1+ν)`, read-not-emitted, no dependency / or raw `G/v/rho`) and
  `ops.element.absorbing_boundary(skin=…)` (fans over every btype PG, base series
  on bottom only). `_ELEM_REGISTRY` entry (`mat_family="none"`, `ndf_ok={3}`) for
  ADR-0048 inference. 25 unit tests + an end-to-end deck test reproducing the
  closed-form tally; primitives+parts 1482/1482 green.
- **AB-3 (staging):** ✅ **DONE.** `s.activate_absorbing(pg=|elements=)`
  (`_StageBuilder`) → `ActivateAbsorbingRecord` → `emit_activate_absorbing`
  emits the one-shot `parameter`/`addToParameter ... stage`/`updateParameter 1`/
  `remove parameter` flip, after the stage analysis chain and before `analyze`,
  per partition (reuses the initial-stress `fem_eid→ops_tag` map + per-rank
  filtering). New `flip_element_stage` emitter method (Tcl/py/live/recording;
  H5 no-op). 8 unit + a staged e2e test; integration+unit 2321/2321 green.
- **AB-4:** end-to-end plane-wave example (`PlainWaveBox` + base series +
  staged gravity→flip→transient). DRMBox is **not** modified (separate facility).
- **AB-5:** 2D (`ASDAbsorbingBoundary2D`, with `thickness`).

## Open / deferred

- 2D distortion has no source-side guard — apeGmsh must guard it entirely.
- H5 round-trip of the absorbing elements + the tracked flip set (treat like any
  ElementRecord; the flip is a staged record).
- Cross-partition flip fan-out is part of AB-3 (per-rank `parameter`/`updateParameter`,
  per the STKO reference), not a deferral — but the partitioned skin-extrusion
  path (AB-1 across a partition cut) may land in a later slice.
- STKO-byte-parity of mesh topology is explicitly **not** a goal; equivalence is
  physical, not file-level.
