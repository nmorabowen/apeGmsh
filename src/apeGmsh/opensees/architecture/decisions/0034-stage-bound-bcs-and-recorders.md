# ADR 0034 — Stage-bound BCs and recorders (Phase SSI-2.D)

**Status:** Accepted (Phase SSI-2.D, May 2026). Shipped in three PRs:
[#323](https://github.com/nmorabowen/apeGmsh/pull/323) (PR-A —
validators + `StageRecord` slots),
[#324](https://github.com/nmorabowen/apeGmsh/pull/324) (PR-B —
`s.fix` / `s.mass` builders + emit),
[#326](https://github.com/nmorabowen/apeGmsh/pull/326) (PR-C —
`s.region` / `s.recorder` builders + emit + V4). Extends the SSI
ADR set ([0028](0028-initial-stress-via-parameter-ramping.md) /
[0029](0029-staged-analysis-context-manager.md) /
[0030](0030-stage-bound-topology-activation.md) /
[0031](0031-ssi-convenience-helpers.md)). No new `Emitter` Protocol
methods.

## Context

[ADR 0029](0029-staged-analysis-context-manager.md) ships staged
analysis with per-stage analysis chains and `s.activate(pgs=[...])`
for topology that comes online mid-analysis. The H1 hardening
validator (#312) then refused every `apeSees.fix` / `apeSees.mass` /
`apeSees.region` whose target resolved to a stage-bound node — those
directives would emit in the global pre-stage block, reference a
not-yet-existent OpenSees node, and crash at deck parse time. H1's
error message pointed the user at "a future phase that supports
stage-bound BCs."

Phase SSI-2.D is that future phase. It lifts the refusal by emitting
stage-bound BCs (and recorders) INSIDE the stage block, alongside
the stage's topology and before the stage's `domain_change` (for
BCs) or after the stage's analysis chain and before its `analyze`
(for recorders).

The lift surfaced four sub-decisions that aren't obvious:

1. **PUSH vs PULL builder asymmetry.** `s.add(initial_stress)`
   already exists and uses PULL (the record was constructed by
   `ops.initial_stress(...)` which allocates parameter tags + ramp
   procs upfront — those side effects MUST happen before the stage
   context opens). Should `s.fix` / `s.mass` / `s.region` /
   `s.recorder` follow the same PULL pattern, or PUSH directly into
   the stage's pool?
2. **Validator surface.** H1 was the only ownership-tier guard;
   adding stage-bound BCs creates new failure modes (stage N's BC
   targeting stage M > N, cross-tier duplicates, region name
   collisions, recorder targets that don't exist at the recorder's
   parse position). How many validators, sharing what helpers?
3. **Recorder claiming.** Stage-bound recorders need to NOT emit in
   the global post-element recorder pass. How does the bridge know
   which recorders to skip without breaking the existing
   `bridge._primitives` topological-order machinery?
4. **Region tag identity under MP.** Stage-bound regions whose
   members span multiple ranks need ALL contributing ranks to agree
   on the SAME OpenSees region tag — but the tag is local to each
   stage so two stages with regions named identically (which V3
   refuses anyway) wouldn't reuse the same scalar.

A four-agent cross-check against the actual OpenSees C++ source
([SRC/domain/](https://github.com/OpenSees/OpenSees/tree/master/SRC/domain),
[SRC/analysis/numberer/](https://github.com/OpenSees/OpenSees/tree/master/SRC/analysis/numberer),
[SRC/recorder/](https://github.com/OpenSees/OpenSees/tree/master/SRC/recorder))
preceded the implementation; several initially-plausible designs
were ruled out by the cross-check. The most consequential findings
are folded into the decisions below.

## Decision

### 1. PUSH for `s.fix` / `s.mass` / `s.region`; PULL for `s.recorder`

`s.fix(*, pg=None, nodes=None, dofs)` constructs a `FixRecord`
directly inside the stage's `_fix_records` list. Same for
`s.mass(...)` → `_mass_records` and `s.region(name=..., pg=, nodes=)`
→ `_region_records`. No bridge-side registration step — these
dataclasses are inert.

`s.recorder(spec)` is PULL: `spec` must already be in
`bridge._primitives` (i.e. constructed via `ops.recorder.Node(...)`
/ `Element(...)` / `MPCO(...)`, which allocates a tag at
construction). The builder adds `id(spec)` to
`bridge._stage_claimed_recorder_ids` and appends `spec` to
`stage._recorder_specs`.

**Why the asymmetry stands:** the difference reflects which records
carry registration side effects.

| Record | Side effects at construction | Storage choice |
|---|---|---|
| `FixRecord` / `MassRecord` / `RegionAssignmentRecord` | None (inert dataclasses) | PUSH on stage directly |
| `InitialStressRecord` | Allocates parameter tags + step-hook ramp proc names | PULL — `apeSees.initial_stress` registers globally first; `s.add` moves into stage |
| `Recorder` | Allocates a primitive tag (used by the recorder emit) | PULL — `ops.recorder.X` registers globally first; `s.recorder` claims into stage |

A "uniform PUSH everywhere" alternative would force `apeSees.fix`
and friends to ALSO check whether they're being called inside a
stage context (via the open `_open_stage_builder` slot) — leaking
the stage state into every BC-registering call site. A "uniform PULL
everywhere" alternative would force every BC to go through
`apeSees.fix(...)` → `s.add(...)` even though the record has no
global-side effect to preserve.

### 2. Five-validator orchestrator

`BuiltModel._run_staged_bc_validators(node_owner_stage,
element_owner_stage)` is the single entry point, called from both
`_emit_flat` and `_emit_partitioned` when `stage_records` is
non-empty. Each validator is a no-op when its scope is empty:

- **H1** (PR #312, refactored in PR #323) — global pool targets
  stage-bound nodes. **#323 bug fix:** the partitioned path
  previously skipped H1 entirely; now invoked from both emit paths.
- **V1** — stage N's BC targets a node owned by stage M > N.
- **V2** — duplicate `(node, DOF)` fix or duplicate `(node)` mass
  across global + per-stage tiers. OpenSees rejects duplicate SP
  constraints
  ([Domain.cpp:589-605](https://github.com/OpenSees/OpenSees/blob/master/SRC/domain/domain/Domain.cpp))
  and silently overwrites mass on `setMass` — refuse explicitly so
  the physics change isn't silent.
- **V3** — region `name=` collision across scopes. OpenSees
  `Domain::addRegion` silently appends on duplicate tag
  ([Domain.cpp:2679-2697](https://github.com/OpenSees/OpenSees/blob/master/SRC/domain/domain/Domain.cpp));
  `getRegion` returns only the first match — silent data loss.
- **V4** — stage N's recorder targets a node OR an element owned by
  stage M > N. Same ownership-tier rule as V1 applied to recorder
  selectors (`Node.pg/nodes`, `Element.pg/elements`,
  `MPCO.nodes_pg/nodes/elements_pg/elements`).

H1 and V1 share `_collect_ownership_offenders` and
`_render_offender_line` helpers so error messages stay consistent.
V2 / V3 have different shapes and use bespoke iteration; the
orchestrator runs them in fixed order so error-message stability
holds across versions.

**V4 scope clarification:** the original Red critique framed V4 as
"stage-bound recorder may not reference a global region declared in
a later-emitted scope (parse-time cache rule)." That framing is
moot for the current apeGmsh API — no `Recorder` subclass exposes a
`region_name=` parameter. V4 ships as V1 extended to recorder
pg/nodes/elements selectors, which is the actual ownership concern.

### 3. Recorder claiming via `_stage_claimed_recorder_ids`

The bridge holds `_stage_claimed_recorder_ids: set[int]`. When
`s.recorder(spec)` fires, `id(spec)` lands in the set and `spec`
stays in `_primitives` so its allocated tag remains discoverable
via `tag_for[id(p)]`.

`BuiltModel._claimed_recorder_ids()` derives the same set from
`stage_records` at emit time (each `StageRecord.recorder_specs`
contributes its members). Both global recorder emit loops (flat and
partitioned) check `id(p) in claimed_ids` and skip claimed specs.
The stage emit pass invokes `emit_recorder_spec(spec, emitter,
tag_for[id(spec)], ...)` inside the stage block.

**Why not remove from `_primitives` instead:** removing a recorder
from `_primitives` would also remove it from the
topological-order pass and from `tag_for`. Claiming keeps the
primitive discoverable (tooling, introspection) while changing only
where it emits.

### 4. Per-stage `region_tag_cache` keyed by name

Inside `_emit_stages_partitioned`, each stage iteration constructs
a FRESH `stage_region_tag_cache: dict[str, int]`. The cache is
shared across the per-rank loop FOR THIS STAGE: the first rank to
contribute owned members for a given region name allocates the tag
via `tags.allocate("region")`; subsequent ranks read the same scalar
from the cache and emit it on their own `region $tag -node ...`
line. Result: all contributing ranks for a stage's named region
agree on one tag.

Cache lifetime is per-stage, not per-model, because V3 refuses same
`name=` across stages — so two stages with conceptually-similar
regions must already use distinct names (`lining_r_stage2` vs
`lining_r_stage3`), and tag-cache sharing across stages would
either be wrong (collisions) or unnecessary (no shared names).

Stage-bound region tags are disjoint from global region tags by
construction: both flows allocate from the same monotonic
`TagAllocator.allocate("region")` counter, so the counter handles
disjointness automatically. This matters because OpenSees
`Domain::addRegion` silently appends on duplicate tag — relying on
OpenSees to reject would mean silent bugs.

### 5. Unified `domain_change` gate + empty-bracket skip

The `domain_change` directive emits ONCE per stage, gated on
`has_activation OR has_bcs` where `has_bcs = bool(stage.fix_records
or stage.mass_records or stage.region_records)`. The original
SSI-2.B gate fired on topology only; SSI-2.D widens it because
stage-bound BCs without topology activation still need a barrier
(a stage that just adds mass to globally-emitted nodes still needs
OpenSees to rebuild its analysis tables before the stage's chain
binds).

Under MP, per-rank `partition_open(K) / partition_close` brackets
are SKIPPED on ranks with no content (no owned topology AND no
owned BCs AND no owned region members). Without the skip, the Py
emitter would produce an empty `if getPID() == K:` block — a
Python `SyntaxError`. The Tcl emitter tolerates empty blocks but
the symmetry rule is preserved for diff/merge stability.

## Source-side basis (cross-check verdicts)

Five OpenSees-source claims were verified against
`C:\Users\nmora\Github\OpenSees_Compile\OpenSees\SRC\` (the four-
agent cross-check that preceded implementation). Highlights:

| Claim | Verdict | Source citation |
|---|---|---|
| `Domain::addNode/addElement/addSP_Constraint` set the domain dirty flag internally | CONFIRMED | Domain.cpp:485, 511, 626 |
| `Domain::addRegion` and `Domain::setMass` do NOT set the dirty flag | REFUTED my original claim — they skip the flag | Domain.cpp:2679-2697, 3876-3883 |
| `Domain::addRegion` silently appends on duplicate tag | CONFIRMED — `getRegion` returns first match | Domain.cpp:2700-2707 |
| Recorders cache region members at TCL PARSE TIME, not at `setDomain()` | CONFIRMED — much stricter than initially thought | TclRecorderCommands.cpp:276, 1331+ |
| `ParallelNumberer` uses Channel point-to-point, NOT MPI_Allreduce, and does NOT check the dirty flag | CONFIRMED — global `domain_change` is a semantic marker, not a parallel-sync barrier | ParallelNumberer.cpp:146, 303-304 |
| `Domain::addSP_Constraint` REJECTS duplicate `(node, DOF)` with an error | CONFIRMED | Domain.cpp:589-605 |
| Node mass is reassembled fresh each `formTangent` from `Node::getMass()` | CONFIRMED | Node.cpp:1205, 1269; Newmark.cpp:284, 294 |

The cross-check overruled three initially-plausible designs:
1. "Defensive `domain_change` after every region/mass call to be
   safe" — overruled. The next chain bind post-`wipeAnalysis` reads
   Domain state fresh; one barrier per stage is sufficient.
2. "Recorders can be moved out of `_primitives` instead of claimed"
   — overruled. Moving breaks tag discovery; claiming preserves it.
3. "Empty `partition_open(K)` brackets on every rank for parallel
   symmetry" — overruled by Py emitter `SyntaxError`. Skipping
   non-contributing ranks is the correct path.

## Consequences

**Positive:**

- Stage-bound BCs lift the H1 workaround ("keep the BC on a
  globally-emitted node"). Geotechnical decks can now express
  "anchor the new lining at its stage-bound base" naturally.
- Stage-bound recorders capture per-stage state correctly. The
  recorder line emits AFTER the chain (so it sees bound state) and
  BEFORE `analyze` (so it captures the stage's analyze steps).
- The four-validator surface catches every ownership-tier failure
  mode at build time with clear offender lists — users don't see
  OpenSees parse errors on stage-bound mistakes.
- Tag determinism extends to stage-bound regions under MP: the
  per-stage cache + monotonic allocator guarantee disjoint tags
  with no chance of OpenSees-side silent overwrite.

**Negative / accepted limitations:**

- Stage-bound BCs / recorders are APPEND-ONLY across stages. A
  stage cannot release a prior stage's fix via `s.fix` or zero out
  a prior stage's mass via `s.mass`. `remove sp` / `zero mass`
  verbs are deferred (see
  [_DEFERRED.md](../_DEFERRED.md) §"`remove sp` / mass-zero-out
  across stages").
- Stage-bound MPCO with `nodes_pg=` / `elements_pg=` filters works
  but doesn't reuse the cross-rank tag-identity infrastructure for
  filter regions. Defer until a real consumer needs it
  ([_DEFERRED.md](../_DEFERRED.md) §"MPCO recorders with filters
  under stages").
- Cross-stage region union (one OpenSees region whose members span
  multiple stages) is refused by V3. Lifting would require either
  OpenSees `region` extension (impossible — `MeshRegion`
  membership is immutable post-construction:
  [MeshRegion.cpp:82-85](https://github.com/OpenSees/OpenSees/tree/master/SRC/domain/region))
  or deferred emit of the unified region to the last contributing
  stage (loses recorder coverage for earlier stages). Likely won't
  lift; the workaround (per-stage named regions + client-side
  aggregation) is correct OpenSees usage.

## Alternatives considered

- **Stage-bound BCs as global `_StageBuilder` kwargs** — e.g.
  `s = ops.stage(name, fix=[...], mass=[...])`. Rejected: nested
  data structures hide the per-record validation surface and don't
  compose with the existing verb-per-line `_StageBuilder` shape.
- **Recorder claiming via subclass discriminator** — define
  `StageBoundRecorder(Recorder)` and have the global emit skip via
  `isinstance` check. Rejected: forces a parallel type hierarchy
  for what's purely a scope discriminator.
- **V4 as a recorder-references-region-by-name check** — original
  Red framing. Rejected because no apeGmsh `Recorder` subclass
  exposes a `region_name=` parameter; V4 is V1 extended to
  recorder selectors instead.

## File map

| Concern | Source |
|---|---|
| `StageRecord.{fix,mass,region,recorder_specs}` dataclass fields | [`_internal/build.py`](../../_internal/build.py) `class StageRecord` |
| Builder methods | [`apesees.py`](../../apesees.py) `_StageBuilder.{fix,mass,region,recorder}` |
| Emit slot (flat) | [`apesees.py`](../../apesees.py) `BuiltModel._emit_stages_flat` |
| Emit slot (partitioned + per-stage region_tag_cache) | [`apesees.py`](../../apesees.py) `BuiltModel._emit_stages_partitioned` |
| Region emit helpers | [`apesees.py`](../../apesees.py) `_emit_stage_regions`, `_emit_stage_regions_partitioned` |
| Recorder claiming | [`apesees.py`](../../apesees.py) `apeSees._stage_claimed_recorder_ids`, `BuiltModel._claimed_recorder_ids` |
| Validator orchestrator | [`apesees.py`](../../apesees.py) `BuiltModel._run_staged_bc_validators` |
| V1 / V2 / V3 / V4 + offender helpers | [`apesees.py`](../../apesees.py) `_validate_stage_bound_node_targets`, `_validate_no_duplicate_fix_mass_across_tiers`, `_validate_region_scope_invariants`, `_validate_stage_bound_recorder_targets`, `_collect_ownership_offenders`, `_render_offender_line` |
| Recorder target resolvers (V4) | [`apesees.py`](../../apesees.py) `_recorder_node_targets`, `_recorder_element_targets`, `_build_fem_eid_owner_stage_map` |
| Introspection symmetry | [`apesees.py`](../../apesees.py) `apeSees.all_fix_records`, `all_mass_records`, `all_region_records`, `all_recorder_specs` |

## Test map

See [staged-analysis.md](../staged-analysis.md) §"Test map" for the
full SSI-2.D test surface. Phase SSI-2.D added:

- [`tests/opensees/unit/test_stage_bound_validators.py`](../../../../../tests/opensees/unit/test_stage_bound_validators.py) — V1 / V2 / V3.
- [`tests/opensees/unit/test_stage_bound_fix_mass.py`](../../../../../tests/opensees/unit/test_stage_bound_fix_mass.py) — `s.fix` / `s.mass` + flat emit + introspection.
- [`tests/opensees/integration/test_emit_partitioned_stage_bound_bcs.py`](../../../../../tests/opensees/integration/test_emit_partitioned_stage_bound_bcs.py) — partitioned fix/mass + empty-bracket skip + unified `domain_change` gate.
- [`tests/opensees/unit/test_stage_bound_region_recorder.py`](../../../../../tests/opensees/unit/test_stage_bound_region_recorder.py) — `s.region` / `s.recorder` + claiming + V4 + introspection.
- [`tests/opensees/integration/test_emit_partitioned_stage_bound_regions.py`](../../../../../tests/opensees/integration/test_emit_partitioned_stage_bound_regions.py) — per-stage region_tag_cache + cross-rank tag identity + tag disjointness.

## Cross-references

- [staged-analysis.md](../staged-analysis.md) — internals doc; per-
  stage emit pipelines, validator orchestrator, deck layout.
- [api-design.md](../api-design.md) §"Staged analysis" — user-facing
  surface for `s.fix` / `s.mass` / `s.region` / `s.recorder`.
- [ADR 0029](0029-staged-analysis-context-manager.md) — original
  `_StageBuilder` context-manager design that this ADR extends.
- [ADR 0030](0030-stage-bound-topology-activation.md) — topology
  activation; stage-bound BCs reuse the same ownership-tier
  computation (`compute_stage_ownership`).
- [ADR 0027](0027-cross-partition-mp-constraints.md) §"Regions
  interaction" — INV-4, the existing region per-rank fan-out
  convention that the per-stage region emit mirrors.
- [_DEFERRED.md](../_DEFERRED.md) §"Staged-analysis follow-ups" —
  `remove sp` / mass-zero-out, MPCO with filters under stages,
  cross-stage region union.
