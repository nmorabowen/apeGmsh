# ADR 0064 — Ladruno recorder region filter (`-R`)

**Status:** Accepted (2026-06-18; implemented RF0–RF3, full `tests/opensees`
suite green — 4623 passed). Closes the first of the two region-related deferrals
carried in the `Ladruno` docstring since recorder-plan L1
(`src/apeGmsh/opensees/recorder.py:766`). Mirrors the MPCO filter machinery
ratified by [0024](0024-emitter-protocol-widen-region.md) (the `Emitter.region`
widening) and made partition-aware by [0027](0027-cross-partition-mp-constraints.md)
(INV-4, the per-rank region pass). No OpenSees change and no H5 schema bump —
the fork recorder already parses `-R`. Implementation plan:
`internal_docs/plan_ladruno_region_filter.md`.

**Shipped.** RF0 extracted `FilterableRecorder` (selectors + `has_filter` /
`resolve_filter_ids` / `materialize` / `_validate_filter`); MPCO subclasses it
behaviour-preserving (existing MPCO region tests + golden partition tests green).
RF1 made `Ladruno(FilterableRecorder)` emit `-R` before `-G energy`, with the
`region + energy` fail-loud guard and the `nodes=/nodes_pg=/elements=/elements_pg=`
authoring surface on `ops.recorder.Ladruno`. RF2 widened the three INV-4
`isinstance(…, MPCO)` region gates to `FilterableRecorder` (the internal
`_MPCOFilterPlan` / `_plan_partitioned_mpco_recorders` / `_emit_mpco_filter_
regions_for_rank` names were **kept** — no external callers, so renaming was
unwarranted churn; only the type annotations widened). RF3 added staged
(`s.recorder`) + `model.h5` archival coverage. One contract adjustment surfaced:
the cross-family primitive contract requires `dependencies` directly in each
concrete class's `__dict__`, so it stays defined on MPCO and Ladruno (not lifted
to the base).

**Follow-up shipped — per-region energy (§4, the reserved landing).** PE0–PE3
flipped the `region + energy` fail-loud guard into the promised
`-G energy $region_tag`. The C++ grammar was re-read and the deployed build
(`5266a708`) **run-verified**: `recorder ladruno … -E axialForce -R 10 -T nsteps
1 -G energy 10` writes **both** `RESULTS/ON_DOMAIN/energyBalance` (whole-model)
and `RESULTS/ON_REGIONS/energyBalance` (per-region) — confirming the fork emits
both balances additively. The §4 premise was corrected along the way: the fork's
`-G energy <tag>` energy-region list is **independent** of `-R` (orthogonal, not
"reuse the filter tag because they must match"). We still couple them — energy
reuses the value-filter's already-allocated region tag — because that reuse is
exactly what makes per-region energy ride the existing flat / staged /
partitioned region plumbing with zero new wiring. So `energy=True` + a filter now
emits `-G energy $tag` over that region; `energy=True` alone stays whole-model.
The consumer side (`Results.energy(region=<tag>)` reading `ON_REGIONS`) was
already built.

**Follow-up shipped — decoupled energy region (`energy_pg=`, EPG).** The
remaining coupling (energy could only be recorded over the *value-filter*
region) is now lifted: `energy_pg=` records energy over an **independent**
region, orthogonal to the `-R` value filter (run-verified on the deployed build:
`-E axialForce -R 5 -T nsteps 1 -G energy 10` — value channels over region 5,
energy over region 10 — writes both `ON_DOMAIN` and `ON_REGIONS`). A Ladruno can
now carry up to two auto-emitted regions (the value filter + the energy region);
the partition planner (`_MPCOFilterPlan` → a list of `_RegionEmit`) and per-rank
emit were generalized from one region per recorder to N, so the energy region
gets its own per-rank INV-4 fan-out. `energy_pg` takes precedence over the
coupled `energy=True` form and implies energy recording. The only residual
deferral is a **UX** one (shared by all auto-region recorders, not energy-
specific): discovering the bridge-allocated region tag from a PG name to pass to
`Results.energy(region=…)` — today via the reader's "recorded region tags: […]"
error and `/opensees/regions`.

## Context

The `Ladruno` recorder (`recorder ladruno fname.ladruno ...`) is the fork's
canonical HDF5 sink — value channels (`-N` nodal, `-E` element), cadence
(`-T dt|nsteps`), and a whole-model energy balance (`-G energy`). It is
**whole-model only**. To record a sub-model — a single physical group of
elements, a foundation, a pier — a user has to fall back to the `MPCO`
recorder, which carries the full filter surface (`nodes=`, `nodes_pg=`,
`elements=`, `elements_pg=`) and auto-emits an OpenSees `region` the recorder
line references via `-R $tag`.

The `Ladruno` docstring frames `-R` as deferred because *"the canonical
`.ladruno` is self-sufficient and the common case is whole-model recording."*
That framing is a **scoping decision, not a capability gap.** We
source-verified the fork recorder: `MPCORecorderLadruno.cpp` (worktree
`OpenSees_Compile/mpco-ladruno-wt`) parses `-R <regionTag>` at lines 1654–1682,
pulls `region->getNodes()` / `region->getElements()` from
`domain->getRegion(rtag)` into its node/element sets, filters
`writeModelNodes()` / `writeModelElements()` (lines 457–524) to that set, and
records the tag into HDF5 `MODEL/SETS` for self-description (lines 684–757).
**The recorder has supported `-R` all along; only the apeGmsh bridge withholds
it.**

So the work is a **bridge-only port** of an already-proven pattern. The cost is
not designing region emission (MPCO did that) — it is doing the port *without
copy-pasting* ~200 lines of filter machinery into a second recorder that is
explicitly defined to track MPCO "to 1e-12" on its shared channels. Two near
-identical copies of `resolve_filter_ids` / `materialize` / the asymmetric
-filter guards is a standing drift hazard.

There is a second seam. The partitioned-emit path (ADR 0027 INV-4) plans and
emits filter regions per rank through `_plan_partitioned_mpco_recorders`
(`apesees.py:4098`) and `_emit_mpco_filter_regions_for_rank` (`apesees.py:4148`),
both gated on `isinstance(p, MPCO)` (`apesees.py:4126`). A region-filtered
recorder that works in a flat deck but silently degrades in a partitioned run
(the common HPC / SSI case) would be a sharp footgun. Region support for
Ladruno therefore *must* reach the partition planner, not just `_emit`.

A third seam is the energy channel. Ladruno's `-G energy` must stay the **last**
token: the fork's `-G` parser eagerly consumes trailing region-tag integers and
cannot rewind past a following flag, so `-G energy -T nsteps 10` is a parse
error (run-verified, recorded at `recorder.py:761`). Any `-R $tag` we add must
sit **before** `-G energy`. And `-G energy <regionTag...>` is itself the *second*
deferred item (per-region energy) — distinct from `-R` (region-filtered value
channels) but sharing the same region tag, which shapes how the two deferrals
compose.

## Decision

Add region filtering to `Ladruno` by **extracting the MPCO filter machinery
into a shared `FilterableRecorder` base** that both `MPCO` and `Ladruno`
inherit, and **broadening the partition planner's `isinstance(…, MPCO)` gates
to the base**. The fork recorder is unchanged; emission gains `-R $tag` ahead of
the energy flag.

1. **Shared base — `FilterableRecorder(Recorder)`.** Lift the four selector
   fields (`nodes`, `nodes_pg`, `elements`, `elements_pg`), the private
   `_region_tag`, `has_filter()`, `resolve_filter_ids()`, and the region-emitting
   `materialize()` out of `MPCO` into a frozen, `kw_only`, `slots` base
   dataclass. The asymmetric-filter guards (node-only + `elem_responses`;
   element-only + `nodal_responses`) move to a shared `_validate_filter()`
   helper the subclasses call from `__post_init__`, with the recorder kind
   ("MPCO" / "Ladruno") parameterised into the message. `MPCO` and `Ladruno`
   become thin subclasses: each owns its construction validation, its emitted
   recorder-kind token, and its own `_emit`. The refactor is **behaviour
   -preserving for MPCO** — its emitted deck stays byte-identical (the golden
   contract for the refactor step).

2. **`Ladruno` gains the four selectors + `-R`.** `Ladruno` inherits the
   selectors and `materialize()` unchanged. Its `_emit` appends `["-R",
   self._region_tag]` when `_region_tag is not None`, positioned **after** `-T`
   and **before** the trailing `-G energy`:

   ```
   recorder ladruno fname.ladruno [-N ...] [-E ...] [-T dt|nsteps v] [-R $tag] [-G energy]
   ```

   The same `NotImplementedError` defence-in-depth guard MPCO uses (reaching
   `_emit` with an unmaterialised `*_pg=` means the bridge was bypassed) carries
   over.

3. **Partition path widened to the base.** Change the three `isinstance(…,
   MPCO)` gates that drive INV-4 region planning/emission
   (`_plan_partitioned_mpco_recorders` `apesees.py:4126`, and the two
   per-element / per-node owner predicates at `apesees.py:3348` / `:3376`) to
   `isinstance(…, FilterableRecorder)`. Because the per-rank machinery already
   operates purely through `has_filter()` / `resolve_filter_ids()` /
   `_region_tag`-injection — all now defined on the base — a filter-bearing
   `Ladruno` rides the identical per-rank intersection and region-emit code with
   no new branch. The internal `_MPCOFilterPlan` dataclass / `_plan_partitioned_
   mpco_recorders` / `_emit_mpco_filter_regions_for_rank` names are **kept** (the
   grep confirmed no external callers; only docstring) — their `materialised_spec`
   annotation widens to `FilterableRecorder`. Renaming was rejected as
   unjustified churn on internal symbols.

4. **`region=` + `energy=True` → per-region energy.** *(Originally fail-loud;
   the follow-up below shipped the per-region form — see the "Follow-up shipped"
   note in the status block.)* A region filter scopes
   the **value** channels (`-N`/`-E`) via `-R`; `-G energy` (no trailing tag)
   is **whole-model**. The clean per-region form is `-G energy $tag`. The
   original slice **raised `ValueError`** on a filter + `energy=True` because the
   C++ `-G energy <regionTag>` path was not yet run-verified. The follow-up
   (PE0–PE3) verified it on the deployed fork build and flipped the
   guard into `-G energy $region_tag` reusing the tag `materialize()` already
   allocated.

5. **No new schema, no Emitter widening.** `Emitter.region` already exists (ADR
   0024). `Ladruno` region records round-trip through the same `MODEL/SETS`
   self-description MPCO uses — no H5 zone-schema bump.

## Alternatives considered

- **Copy MPCO's filter machinery verbatim into `Ladruno` (the L1 plan's
  literal suggestion).** Rejected. ~200 lines duplicated across two recorders
  that are *contractually* parallel ("to 1e-12") is the exact drift hazard a
  shared base removes; and it does **nothing** for the partition path — the
  `isinstance(…, MPCO)` gates would still exclude Ladruno, shipping a recorder
  that works flat and silently degrades partitioned. More code, worse coverage.

- **Make `Ladruno` subclass `MPCO` directly.** Rejected on is-a grounds. A
  `Ladruno` is not a kind of `MPCO` (different recorder token, different output
  extension, an extra `energy` channel MPCO's `__post_init__` knows nothing
  about). It would also make every `isinstance(p, MPCO)` site across the bridge
  *accidentally* match Ladruno — convenient for the three partition gates,
  surprising everywhere else. A sibling base makes the shared surface explicit
  and the broadening intentional.

- **Flat-only region support; defer partitioned to a follow-up.** Rejected.
  Partitioned runs are the dominant Ladruno use case (HPC, SSI). Because the
  per-rank machinery already runs entirely off the base-class contract,
  widening three `isinstance` checks is nearly free — deferring it would ship a
  documented footgun to save almost nothing.

- **Allow `region=` + `energy=True`, emitting whole-model energy.** Rejected
  for now as a silent semantic mismatch (filtered stresses, unfiltered energy,
  one recorder). Fail-loud is the house style and is cheaply reversible once
  per-region energy lands.

## Consequences

- **Positive.** `Ladruno` reaches MPCO parity on region targeting with the same
  authoring surface (`nodes_pg=` / `elements_pg=` / `nodes=` / `elements=`),
  flat **and** partitioned. The shared base removes the two-copy drift risk and
  *retro-hardens* MPCO (one tested code path, not two). Net line count falls
  versus the copy alternative.

- **Cost / risk.** The base-extraction is a behaviour-preserving refactor of a
  load-bearing, partition-critical recorder. The mitigation is a golden-deck
  gate: MPCO's emitted Tcl/py must be byte-identical before and after the
  refactor, verified before Ladruno touches anything (plan step RF0). The
  partition gates are widened only after the base lands and MPCO is green.

- **Per-region energy — SHIPPED (PE + EPG follow-ups).** `energy=True` + a
  filter emits `-G energy $region_tag` (coupled, PE); `energy_pg=` emits
  `-G energy $energyTag` over an **independent** region (decoupled, EPG), with
  its own per-rank INV-4 fan-out — both run-verified on the deployed fork build.
  The reader (`Results.energy(region=<tag>)`) was already in place. Residual
  deferral is a **UX** item only (shared by all auto-region recorders, not
  energy-specific): discovering the bridge-allocated region tag from a PG name to
  pass to `Results.energy(region=…)` — today it surfaces via the reader's
  "recorded region tags: […]" error and `/opensees/regions`.

- **Staged path.** `s.recorder` (ADR 0034) drives the same `emit_recorder_spec`
  → `materialize` → `_emit` pipeline, so a region-filtered `Ladruno` inside an
  `ops.stage` materialises its region in the correct stage with no extra wiring
  (covered by plan RF3, flat). Under **partitioning**, a stage-bound filtered
  recorder emits one whole-resolved `region` line **once in global scope** (one
  shared tag) rather than the per-rank INV-4 intersection — and this is
  **correct, not a bug** (investigated 2026-06-18): OpenSees `MeshRegion`
  silently filters region members to each rank's local domain
  (`MeshRegion::setElements/setNodes` guard every tag with `if (theEle != 0)`,
  no abort, no warning — source- and run-verified), so every rank's region
  becomes its owned intersection and the per-rank `.ladruno`/`.mpco` files merge
  by the single shared tag. The non-staged path's per-rank INV-4 emission is a
  **deck-size optimization** (keeps each rank's region line O(model/np) vs. an
  O(model) global line — the [ADR 0061](0061-per-rank-deck-emission.md)
  resident-text concern), *not* a correctness requirement; the staged path
  simply forgoes that optimization. Flat-staged, global-partitioned, and
  staged-partitioned filtering all produce identical, correct results.
