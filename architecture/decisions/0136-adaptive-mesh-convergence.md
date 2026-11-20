# ADR 0136 — Adaptive mesh convergence: error estimate, size field, fresh-session rebuild

**Status:** Proposed (2026-11-18). Design only, no code. Drafted on 2026-09-02
after a red/blue adversarial review, left uncommitted, and re-verified against
`origin/main` `f1789a06` on 2026-11-18 before filing (see "Review log").

**Owner:** nmora

**Sibling:** the fork's ADR 236 (WP-236,
`Ladruno_implementation/236_ladruno_state_transfer_adaptivity_adr.md`):
material history get/set and the commit primitive that Level B needs. This
ADR owns the apeGmsh half: an error estimator on `Results`, a size field on
`g.mesh`, and a driver that reruns a user's `build()` on a fresh session per
iteration.

**Builds on** ADRs 0041 (the chain-phase freeze, which stays), 0059 (mesh
recipes), 0104 (`ops.strategy.Substep`), 0115 (read-back), 0117 (Assembly),
0121 (the freeze lock and the engine policy), 0122 and 0131 (`.ladruno`),
0132 (the variants manifest) and 0133 (`fem_hash` pairing). **Supersedes**
nothing.

**Numbering.** The September draft was numbered 0103 and named a fork
"ADR-88". Both numbers were taken while it sat uncommitted: 0103 is the
SANISAND deck contract and the fork's 88 is the H5DRM higher-order work. The
fork's MPM ADR (`232_ladruno_mpm_general_driver_adr.md`) cites that draft by
file name; this ADR and fork ADR 236 are what it meant.

## Context

Nothing on `main` estimates discretization error, runs a refinement study,
interpolates a field from one mesh to another, or builds a size field from
results. A search of `src/`, `scripts/`, `tests/` and `architecture/` on
2026-11-18 found only step-size adaptivity (ADR 0104). The one test that
asserts an error shrinks under refinement is
`tests/test_loads_line_callable_magnitude.py::test_tributary_midpoint_undershoots_then_converges`.
The concrete guide tells the user to regularize or the "results are mesh
garbage" (`docs/guides/nonlinear_concrete_solver.md`, §3), and
`architecture/_DEFERRED.md` (the `HostProjector` RFC) names a refinement
study as its trigger. Every mesh-convergence claim a deck makes today is made
by hand.

The program targets four problem classes (owner, 2026-09-02): RC damage, SSI
wave propagation, contact, and soil failure with foundation uplift. The
failure-locus probes of the TIMs macroelement calibration are the motivating
use: many pushovers on one geometry, each of which should be mesh-converged.

### What already exists

- **Declarations that survive a rebuild.** Loads, constraints, masses, rebar
  and labels are geometry-level defs, resolved to node tags inside
  `src/apeGmsh/mesh/_fem_factory.py::_from_gmsh`. Labels are entity-level
  physical groups (`src/apeGmsh/core/Labels.py`, module docstring).
  `ConstraintsComposite.resolve_interfaces` reads live gmsh. A script re-run
  on a new mesh re-resolves all of it.
- **Gauss-point results.** `src/apeGmsh/results/_slabs.py::GaussSlab` holds
  integration-point values with `element_index` and `natural_coords`
  (normalized to gmsh's reference frame at the reader, #2228). The slab is
  read from `.ladruno`, the only results format since ADRs 0122 and 0131.
  `results/_gauss_extrapolation.py::extrapolate_gauss_slab_to_nodes` is a
  nodal recovery; `results/_gauss_world_coords.py::compute_global_coords`
  gives world coordinates; `fem/_shape_functions.py::compute_jacobian_dets`
  and `fem/_quadrature.py` give Jacobians and reference rules.
- **Size fields.** `mesh/_mesh_field.py::FieldHelper` wraps any gmsh field.
  `mesh/_mesh_recipe.py::_Recipe._refresh_min_field` folds a user background
  field into the recipe's Min combiner. bamg (anisotropic 2D from a tensor
  view) and mmg3d meshed out of the box in gmsh 4.15 (probed 2026-09-02; not
  re-probed).
- **Point location.** `_kernel/geometry/_inverse_map.py::locate_point` is a
  guarded inverse isoparametric map for tri3, quad4, tet4 and hex8.
- **Iteration bookkeeping.** `results/user/_variants.py::write_variants`
  writes the `<study>.variants.json` manifest of ADR 0132, and ADR 0133
  pairs each `.ladruno` to its model archive by `fem_hash`.

### What does not exist, and what the review found

- **No transfer between meshes.** `results/_composites.py::_nearest_node_id`
  and `::_nearest_element_id` are nearest-only.
- **A session cannot be remeshed, and this ADR does not change that.**
  - Every mesh verb calls `core/_compose_errors.py::chain_phase_guard`, which
    raises `ChainPhaseError` once the session has a broker
    (`_session.py::_SessionBase._check_chain_phase`).
  - Lifting that guard would not be enough. Only declarations bump the
    broker's freshness counter (`_core.py::apeGmsh._bump_fem_counter`, called
    from `core/_declarations.py`), so a regenerated mesh would return the old
    broker.
  - Declarations made after the first `get_fem_data()` are stored as
    tag-resolved records on the broker, not as defs
    (`_kernel/resolvers/_chain_phase_router.py::try_chain_phase_route`). They
    would be lost on a remesh.
- **A model-data view cannot be a background field.** Probed on gmsh 4.15
  (2026-09-02, twice, independently): after `mesh.clear()` a view written
  with `addModelData` raises "Cannot use view based on current mesh for
  background mesh", and `view.probe` returns nothing. A list-data view
  refined 66 to 559 elements and still probed. `mesh/View.py::View.add_element_scalar`
  writes model data and has no list-data writer.
- **The slab cannot integrate an energy norm by itself.**
  - `GaussSlab` has no weights and no Jacobians.
  - `fem/_quadrature.py` holds reference rules only; nothing maps an
    OpenSees element class and its Gauss-point count to a rule.
  - The recovery is a corner-only, unweighted mean.
  - `opensees/model_data.py::ModelData` excludes materials. The material
    link is positional in `opensees/_internal/typed_records.py::ElementRecord`,
    and plane stress or strain is a field on the element
    (`opensees/element/solid.py`, `plane_type`).
- **Two declarations shift under a rebuild.**
  `mesh/_fem_factory.py::_resolve_decoupled_nodes` writes the minted tag back
  on the def, and `LoadsComposite._resolve_point_closest` with a tolerance
  can pick a different node count.
- **No rerun harness.** The September draft leaned on the `sensitivity/`
  package. It was deleted in #1493. The driver below has no precedent in the
  tree.

### What the fork has decided since September

- **Regularization of softening soil is not coming as a wrapper.** The
  fork's ADR 90 measured the generic rate-dependent wrapper this program had
  asked for. Its recommendation is disclosure-only: the localization width
  does not converge and is set by the imperfection field, not by a
  relaxation time. The load at a fixed settlement does converge under
  refinement without regularization (7.80, 5.28, 2.86 % across three
  meshes). The owner's September choice, "softening with regularization
  first", is overtaken by that evidence; D5 follows ADR 90.
- **Large deformation is the MPM program's.** The fork's WP-232 and ADR 0134
  here make MPM the route for runout, penetration and large-strain footings.
  Deformed-boundary remeshing is listed there as a later phase.
- **apeGmsh is "a seeder, not a runtime remesher"** (fork ADR-60a, repeated
  in ADR 0134). This ADR keeps to it: apeGmsh never touches a running domain.

## Decision

There are two levels and one loop.

- **Level A** reruns the whole load history on each new mesh. It is a
  mesh-convergence runner; it does not adapt mid-history.
- **Level B** restarts on the new mesh from an injected nodal and
  integration-point state, and continues. It plugs into the same loop at one
  step. It depends on fork ADR 236.

Neither level mutates a running OpenSees domain.

### D1 — The model is a function, and each iteration gets a fresh session

The user writes two callables:

- `build(g, size_field=None)` authors geometry, labels, sizing,
  declarations and the mesh on the session it is given;
- `run(fem, k)` emits and runs the deck and returns a `Results` (through
  `ops.results(path)`, ADR 0115 D3). Its analysis may use
  `ops.strategy.Substep` (ADR 0104).

The driver opens a new `apeGmsh` session per iteration, calls `build`, takes
the broker, calls `run`, estimates, and builds the next size field. No
session is remeshed, so ADR 0041 is untouched and every declaration
re-resolves. Nothing is added to Studio or the Qt viewers (ADR 0112 D8).

### D2 — The size field is a list-data view, folded into the recipe's Min

`g.mesh.adapt.size_field(...)` returns a plain `SizeField`: points from the
old mesh, with a target size at each. Applying it:

1. writes a **list-data** view (scalar per element type, or tensor data for
   bamg anisotropy);
2. adds a `PostView` field on it;
3. sets it as background, then makes the recipe refresh its Min field.

Step 3 needs care. `_refresh_min_field` runs only from
`_Recipe._add_region_field`, so a background set after the last recipe
region would replace the Min field instead of joining it. The helper
triggers the refresh itself.

The sizing law is `h_new = h_old · (η_target / η_e)^(1/p)`, clamped to
`[h_min, h_max]`, under a total element budget. It refines only; coarsening
is not in this ADR.

### D3 — The estimator runs on `Results`

`results.error.zz(step)` returns a Zienkiewicz–Zhu energy-norm error per
element:

`η_e² = Σ_gp w · |J| · (σ* − σ_h)ᵀ D⁻¹ (σ* − σ_h)`

- `σ_h` comes from the `GaussSlab`, and `σ*` from the existing nodal
  recovery.
- `w` comes from a new table that maps an element class and its Gauss-point
  count to a rule in `fem/_quadrature.py`; `|J|` from `compute_jacobian_dets`.
- `D` is rebuilt per element from the `/opensees/materials` zone
  (`emitter/h5_reader.py::H5Model.materials`) and the element's `plane_type`.

Version 1 covers elastic isotropic materials on linear elements and refuses
everything else.

`results.error.indicator(field, kind=...)` ranks elements by the gradient or
the rate of a recorded field, such as plastic strain. It drives refinement
near collapse, where the energy norm of an elastic estimator means little.

SSI wave problems need no estimator. Their size field is known in advance
(about a tenth of the shortest wavelength) and goes straight into D2.

### D4 — Level B: a transfer bundle

`results.transfer.to(fem_new)` returns nodal displacements (and velocities
and accelerations for a transient) at the new nodes, and a history vector at
each new integration point.

- Nodal fields are interpolated with the old element's shape functions,
  located with `locate_point`.
- **History is copied from the nearest old integration point.** A state
  committed at a real Gauss point is admissible. A nodal average of damage,
  hardening variables, nested yield surfaces or crack planes is either
  inadmissible or undefined. Smoothing is an opt-in per variable.

The bridge emits the bundle as fork commands. Those are new rows in the
existing `"domain"` family of `emitter/verbs.py::_ROWS`, beside
`set_node_vel` and `set_node_accel`. They are emitted into decks, because
ADR 0121 D2 keeps live runs to unstaged models.

### D5 — Scope by problem class

| Class | Level A | Level B |
|---|---|---|
| Elastic and SSI waves | size field in advance (D3) | not needed |
| Plasticity without softening (J2, Drucker–Prager, SANISAND family) | estimator before yield, indicator near collapse | after fork ADR 236 S1–S2 |
| Softening soil | converges on the load at a fixed settlement; the band width is **disclosed, not converged** (fork ADR 90) | not in this ADR |
| RC damage with crack-band regularization | `h_min` is the band width (INV-5) | not in this ADR: the fork's concrete state is not a vector |
| Footing uplift and rocking | soil rebuilt each iteration, footing instanced | contact state is reset, not transferred (fork ADR 236 D4) |

The footing in an uplift model is an Assembly instance of a saved model file
(ADR 0117). It is never remeshed. Only the soil file is rebuilt, and the
assembly-level ties and contact are re-resolved against it.

The size field is built on the reference geometry. Remeshing a deformed
boundary belongs to the MPM program.

### D6 — Each iteration is a variant

Iteration `k` writes its model archive and its `.ladruno`, and the driver
adds one row to `<study>.variants.json`: `label`, `path`, and `params` with
`k`, the global error, the element count and the target (ADR 0132 D8). The
results pair to their archive by `fem_hash` (ADR 0133). No new H5 zone or
attribute is added.

## Invariants

- **INV-1 — one mesh per session.** The driver never calls a mesh verb on a
  session that has a broker. `ChainPhaseError` stays the guard.
- **INV-2 — the driver proves the declarations re-resolved.** After each
  iteration it compares, against the previous one, the pair count of every
  interface and tie and the tributary sum of every distributed load. A
  change fails. A `g.mesh_selection` in `build` is refused, and a
  `g.decouple_node` tag may not be carried across iterations.
- **INV-3 — list data only.** A model-data view handed to D2 raises before
  gmsh does, and the recipe's Min field is never replaced.
- **INV-4 — pairing.** The results of iteration `k` open only against
  archive `k` (ADR 0133).
- **INV-5 — the floor and the budget are required arguments.** For a
  crack-band material the floor may not go below the band width.
- **INV-6 — the estimator refuses what it cannot integrate**: an element
  class with no rule in the table, a material with no `D`, mixed plane
  types, a quadratic element in version 1. The bounding-box fallback of
  `compute_global_coords` is for plotting only.
- **INV-7 — Level B is gated by identity.** A bundle is trusted only after a
  same-mesh round trip reproduces the uninterrupted run (fork ADR 236
  INV-1). The re-equilibration step reports the plastic strain it added, and
  the driver gates on that number.

## Governance (ADR 0121)

- **The freeze lock.** `tests/freeze_census.py` holds four registries. S1 to
  S3 grow none of them: `results.error`, `g.mesh.adapt` and the driver are
  not primitives, constraint kinds, MCP tools or emitter verb families. The
  driver is not exposed as an MCP tool.
- **S5's rows** join the existing `"domain"` family, so they grow no
  registry either. Each is a fork-only member under D3 of ADR 0121: refused
  on a stock build with an error that names the fork, pinned by
  `pytest.raises`, and Warranted only when its own test passes in
  `live-fork` on the build `FORK_PIN` names. That needs fork ADR 236 S1 to
  ship and the pin to move.

## Slices

1. **S0 — this ADR and fork ADR 236** (documents only).
2. **S1 — `results.error.zz` and `indicator`.** Adds the element-to-rule
   table and a `D`-per-element helper.
   Gates: a uniform-stress patch gives zero error to 1e-12; on the Kirsch
   plate with a hole the effectivity index is between 0.8 and 1.2 on three
   uniform meshes, in plane stress and plane strain. If it falls below 0.7
   on quad4, the recovery is replaced by a patch recovery before S3.
3. **S2 — `g.mesh.adapt.size_field`**, the list-data writer and the Min
   refresh.
   Gates: a prescribed size halves the element size in a region within
   20 %; the model-data path raises; the tensor path gives a median aspect
   ratio above 5 in the prescribed band.
4. **S3 — the driver, Level A.** This is the first slice a user can use.
   Gates: Kirsch reaches the target in at most 4 iterations; the L-shaped
   domain converges near `N^-0.5`, against `N^-0.27` uniform; a plane-strain
   strip footing on a Tresca soil reaches `N_c = 2 + π` within 1 % with
   fewer elements than the uniform mesh of the same error; a small model
   with one interface and one tie passes INV-2. The fixtures are written for
   this slice in `build(g)` form. The TIMs strip-footing gate
   (`tests/opensees/integration_ladruno/test_tims_strip_footing_gate.py`) and
   the `ssi_frame_wall` viewer bench are not reusable: one reads an external
   mesh bundle, the other is a gmsh-direct bench.
5. **S4 — fork ADR 236 S1–S2** (not this repo), then a `FORK_PIN` bump.
6. **S5 — `results.transfer.to` and the `"domain"` rows.**
   Gate: a J2 notched bar remeshed at 60 % of its history ends within 2 %
   of a fine-mesh reference, with the INV-7 numbers reported.
7. **S6 — the transient variant.**
   Gate: on an elastic bar wave the energy jump across the remesh is under
   1 % and the arrival time is unchanged.
8. **Later, each under its own decision:** footing uplift at Level B,
   coarsening, concrete state, contact slip history.

## Alternatives rejected

- **An in-session `g.mesh.remesh()`.** Blocked three ways (Context), and it
  would need amendments to ADRs 0021, 0041 and 0117.
  `g.mesh.editing.import_discrete` (ADR 0125, unmerged) imports a mesh into
  a live session before the broker exists; it does not reopen this.
- **Mutating the running domain.** The fork's `repartition()` is a stub,
  `remove element` leaves elemental loads behind, and save/restore is keyed
  by tag (fork ADR 236, Context).
- **Averaging history to nodes and sampling it back**, as the default
  transfer (D4).
- **Refining one labelled region and tying it to the rest with a mortar tie
  (ADR 0086).** It adds the tie's own consistency error inside the region of
  interest, and a footing mechanism moves through the whole soil. It remains
  available to a user by hand.
- **A regularization wrapper for softening soil.** Measured and declined by
  the fork (ADR 90).

## Consequences

- A deck can state how much its answer moved under refinement, with a gate
  behind the number.
- Level A costs one full run per iteration. For a nonlinear history that
  takes hours, four to six iterations take a day. That is the case for
  Level B, and the reason Level B is gated rather than skipped.
- A script that wants adaptivity has to be written as `build(g, size_field)`.
  The docs get one page for it.
- The element-to-rule table and the `D` helper are the first constitutive
  knowledge in `results/`. They are elastic-only and named so.
- For softening soil the driver reports a band width with no claim that it
  has converged.

## Review log

**2026-09-02, red and blue teams on the brainstorm.**

- Falsified: the in-session remesh; a model-data view as background field;
  "stress and equivalent plastic strain is the state" (the fork's J2 carries
  eight backstresses and damage, its concrete about 40 doubles, ASDConcrete3D
  crack-plane objects); "re-equilibrate at frozen load" as a defined
  operation.
- Confirmed: restart on a new mesh instead of mutating the domain; Level A
  before Level B; 2D first.

**2026-11-18, re-verification before filing.**

- Every pointer was re-read on `origin/main` `f1789a06`. Twenty had moved
  and are now cited as `path::symbol`.
- Three claims were wrong by November and are corrected above: the rerun
  harness (`sensitivity/`, deleted in #1493), the compose re-merge (compose
  v1 was deleted by ADR 0117), and the embedded-node resolver as the
  inverse-map precedent (it is closed-form; `locate_point` is the Newton
  one).
- Two fixtures the draft named as gates are not reusable (S3).
- The governance note was rewritten: the draft argued against ADR 0102 D2,
  which ADR 0121 D4 has since replaced.
- The soil scope was rewritten against fork ADR 90 and the MPM program.

## Open questions

1. Is Level A alone worth S1 to S3 before the fork work starts? The
   proposal says yes: it is the convergence gate TIMs decks lack today.
2. Should the driver live in `apeGmsh.adapt` or under `results/`? Proposed:
   its own package, because it owns sessions.
3. Does the Kirsch gate hold on the fork's Bézier elements, whose Gauss
   coordinates use a different basis? Not measured.
