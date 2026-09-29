# ADR 0030 — Stage-bound topology activation via `s.activate(pgs=)` (Phase SSI-2.B)

**Status:** Accepted (Phase SSI-2.B, May 2026). Third of the SSI
four-ADR set ([0028](0028-initial-stress-via-parameter-ramping.md)
/ [0029](0029-staged-analysis-context-manager.md) /
[0030](0030-stage-bound-topology-activation.md) /
[0031](0031-ssi-convenience-helpers.md)). Widens the `Emitter`
Protocol with one runtime-state method (`domain_change`).

## Context

[ADR 0029](0029-staged-analysis-context-manager.md) ships
multi-stage analysis but assumes the **full topology** exists at
the start of the model. Real geotechnical workflows are
incremental:

- Stage 1 (in-situ): rock mass alone — lining elements don't exist
  yet.
- Stage 2 (excavation): rock mass minus tunnel zone — that's a
  removal, not an addition, but typically modelled by activating
  excavation-soft elements that ramp the in-situ stress back to
  zero in the excavated region (covered by the SSI-1 ramp).
- Stage 3 (lining installation): rock mass + shotcrete shell + ribs
  — **new elements come online** mid-analysis.
- Stage 4 (loading): everything from stage 3, no new topology.

The OpenSees mechanism for adding elements mid-analysis is to emit
them inside the stage block (after the prior stage's analyze, before
the current stage's analyze) and then issue `domainChange` so
OpenSees rebuilds its renumbered DOF map. Without `domainChange`,
the newly-added elements participate in the next analyze loop with
stale numbering — typically a silent convergence collapse, sometimes
a hard segfault.

Pre-SSI-2.B, the apeSees bridge had no way to express "this element
PG comes online in stage N". All elements emit in the global
pre-stage block. Users wanting staged construction had to:

1. Declare every element globally, then `ops.remove element <tag>`
   between stages and re-emit them. Loses the typed surface and
   the live / Tcl / Py emitter parity.
2. Build N separate `apeSees(fem)` instances (one per stage) and
   concatenate the resulting Tcl files. Each instance allocates
   tags independently → cross-stage tag identity is lost → the
   SSI-1 `addToParameter` fan-out and #314's recorder pg=
   translation can't reach the right tag in the wrong instance.

Neither composes with the "one model, one emit" contract.

## Decision

### A new `_StageBuilder` verb `s.activate(pgs=[...])`

```python
with ops.stage(name="excavate") as s:
    s.activate(pgs=["Lining"])
    s.analysis(test=..., algorithm=..., ...)
    s.run(n_increments=20, dt=0.05)
```

`s.activate(pgs=...)` marks element-PG names as **owned by this
stage**. Elements whose `pg=` matches any activated PG emit their
`element` lines **inside this stage's block** (between `stage_open`
and `domain_change`), not in the global pre-stage emit. Nodes
referenced exclusively by stage-bound elements move into the stage
block too; nodes shared with global elements stay global.

The verb may be called multiple times per stage; PGs accumulate as
a set. The same PG activated in two different stages raises
`BridgeError` at build time — first-stage-wins is unsafe because
the user clearly meant something different.

`s.activate(pgs=...)` is **optional** on every stage. A stage with
no activated PGs is the canonical SSI-1 "ramp in-situ stress on
existing topology" stage (the `insitu` stage in the running
example).

### Widen the `Emitter` Protocol — one method

```python
class Emitter(Protocol):
    # ... existing methods unchanged ...

    def domain_change(self) -> None: ...
```

Per-emitter shape:

| Emitter | `domain_change()` |
|---|---|
| `TclEmitter` | `domainChange` at outer indent |
| `PyEmitter` | `ops.domainChange()` |
| `LiveOpsEmitter` | Forward to `self._ops.domainChange()`. Reachable only from a non-staged user-driven call (the staged path raises `NotImplementedError` in `stage_open` / `stage_close`); included for completeness. |
| `H5Emitter` | **No-op** — domain renumbering is runtime state, not a model-definition change. The archived flat topology is what downstream readers consume; they don't need to replay per-stage `domainChange` calls. |
| `RecordingEmitter` | Capture `("domain_change", (), {})` |

### Ownership computation

A new helper `compute_stage_ownership(stage_records, elements, fem)`
in [`_internal/build.py:1830-1902`](../_internal/build.py) returns
two maps:

- `element_owner: dict[id(spec), stage_index]` — element-primitive
  identity → owning stage. Primitives not in any stage's
  `activated_pgs` are absent (global emit).
- `node_owner: dict[fem_node_id, stage_index]` — FEM node id →
  owning stage. Computed via two-pass logic:

  1. Walk every element spec; for each element's PG fan-out, add
     its referenced node ids to either `global_nodes` (if the
     element is global) or `node_stages[nid]` (set of stages
     referencing the node).
  2. A node referenced by **any** globally-emitted element stays
     global (absent from `node_owner`).
  3. A node referenced **only** by stage-bound elements is owned by
     the **lowest** stage index that references it (`min(stages)`).

Three corollaries follow from this:

- **Global wins.** A node shared between a global element and any
  stage's element stays global. OpenSees has the node from the
  start; the stage's element references it directly without needing
  the node to come online mid-analysis.
- **Lowest-index wins for stage-shared nodes.** When the same node
  is referenced only by stage-bound elements across multiple
  stages, the first stage that emits it owns it. The emit order
  matches `stage_records` registration order (= `with` block exit
  order = lexical order, given M4-validated no-nesting).
- **Per-PG exclusivity raises loud.** Activating the same PG in two
  stages is rejected at `compute_stage_ownership` with `BridgeError
  ("PG {pg!r} is activated by another stage ...")`. Users who want
  cross-stage-shared elements split the PG first.

### `domain_change` is emitted conditionally

`_emit_stages_flat` emits `domain_change()` for a stage **only if**
that stage added at least one node or element to the deck. A
"pure analysis" stage (no `s.activate(...)`) emits the analysis
chain + analyze + stage_close, no `domain_change` line. Without the
conditional, the deck would emit a useless `domainChange` per
stage, polluting the diff for the common SSI-1-only ramp case.

### Element-tag pre-allocation across stages

The most subtle invariant of this ADR. Element tags are
**pre-allocated once** by `allocate_element_tags(elements, fem,
tags)` BEFORE any stage emits. The same `fem_eid → ops_tag` map is
shared across the global block and every stage's block.

Why this matters:

- SSI-1's `addToParameter` fan-out
  ([`_internal/build.py::emit_initial_stress_addtoparameter`](../_internal/build.py))
  indexes by `fem_eid_to_ops_tag`. If a stage's elements were tag-
  allocated lazily inside the stage block, the SSI-1 `addToParameter`
  call would resolve a stale `fem_eid → ops_tag` mapping for any
  cross-stage targets — silent miswiring.
- #314's `Element` recorder pg= translation similarly indexes by
  the same map. Same drift risk.

By pre-allocating upfront, every stage's element tags are
deterministic before any emit happens. The cost is the in-memory
plan structure (per-spec tuple of `(eid, conn, ele_tag)`); the
benefit is tag-identity safety across the entire deck.

## Invariants

- **INV-1.** `Emitter.domain_change` is on the Protocol; every
  existing and future emitter implements it. Live forwards;
  Tcl/Py emit the literal line; H5 + Recording behave per the
  matrix above.
- **INV-2.** A PG activated by stage K cannot also be activated by
  stage L. `compute_stage_ownership` raises `BridgeError`. First-
  write semantics are unsafe — the user clearly meant something
  different.
- **INV-3.** A node referenced by any global element stays global.
  Lowest-stage-index ownership applies only to nodes referenced
  exclusively by stage-bound elements.
- **INV-4.** Element tags are pre-allocated **once** across the
  whole deck before any stage emits. The same `fem_eid →
  ops_tag` map is shared between the global block, every stage's
  block, every SSI-1 `addToParameter` fan-out, and every
  recorder's pg= translation. Cross-stage tag drift would silently
  miswire SSI-1 ramps and recorder targets.
- **INV-5.** `domain_change` emits **only when the stage added
  topology**. A pure-analysis stage (no `s.activate(...)`) emits
  no `domain_change` line — keeps the deck diff clean.
- **INV-6.** The H5 emitter's `domain_change` is a no-op
  intentionally. Domain renumbering is runtime state; the archived
  flat topology is what downstream readers consume. No schema bump.
- **INV-7.** `domain_change` runs **after** the stage's element
  block but **before** the stage's analysis-chain emit. Without
  INV-7 the new chain would bind to the stale DOF map and the
  analyze would diverge.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Per-element activation** (`s.activate(elements=[...])` taking explicit FEM eids) | Loses the "address mesh subsets by name" idiom that apeSees inherits from the broker. The user already named the lining region as a PG when they declared `g.physical.add_volume(..., name="Lining")`; asking them to re-list FEM eids reverses that. Plus the eids depend on mesh renumbering, which the SSI workflow does at `g.mesh.partitioning.renumber(...)` time — fragile coupling. |
| **Re-allocate element tags per stage** | Saves the upfront allocation pass. Breaks INV-4 — SSI-1 `addToParameter` fan-out and #314's recorder pg= translation both index by `fem_eid → ops_tag`; lazy allocation produces drift that's near-impossible to debug because the recorder file silently writes only the time column when its `-ele <tag>` argument resolves to a not-yet-emitted element. |
| **Emit `domainChange` unconditionally per stage** | Pollutes the deck diff for the common SSI-1-only case (a `domainChange` line per stage that didn't add topology). The conditional emit (INV-5) is one boolean check; cheap. |
| **Make `domain_change` an attribute of the `StageRecord`** (e.g. `s.run(..., rebuild_dofs=True)`) | Hides the OpenSees mechanism behind a user-facing toggle. The user shouldn't need to know whether `domainChange` is required for their stage; the build pipeline can detect it from `owned_nodes or owned_specs`. The conditional emit keeps the verb out of the user surface. |
| **First-write-wins for cross-stage PG activation** | Silent miscompiled deck. If the user wrote `s1.activate(pgs=["Lining"])` and later `s2.activate(pgs=["Lining"])`, the second activation would be a no-op and the deck would emit lining elements only in stage 1 — opposite of intent. Loud refusal (INV-2) is correct. |
| **Highest-index wins for node ownership** (instead of lowest) | Equivalent functionally — the node just appears in a different stage. Lowest-index matches the common reading "the first stage that needs this node emits it"; the alternative requires explaining why a node a user expects in stage 2 actually appears in stage 4. |
| **Drop `domain_change` and rely on OpenSees auto-detecting topology changes** | OpenSees does not auto-detect — `domainChange` is the explicit signal. Without it, the next stage's analyze runs against the stale renumbering and typically diverges or segfaults. INV-7 is non-negotiable. |

## Consequences

**Positive:**

- Closes the staged-construction prerequisite for the Cerro Lindo
  tunnel migration. Lining elements declared via the standard
  `ops.element.Brick(pg="Lining", ...)` come online at the right
  stage automatically; users do not need to interact with
  `domainChange` directly.
- Composes with SSI-1's ramps and SSI-2.A's per-stage analysis
  chains. The SSI-1 `addToParameter` fan-out finds the lining
  elements' tags by the shared pre-allocated map; the recorder
  pg= translation (#314) lands recorder targets on the right
  OpenSees element regardless of which stage emitted it.
- Build-time validation surfaces wrong models loudly. Double-
  activation of a PG raises with a clear "PG {pg!r} is activated
  by another stage" message; users do not need to inspect the
  emitted deck to discover the conflict.

**Negative:**

- One more Protocol method. Every concrete emitter (current +
  future) implements it. H5 + Live are minimal additions.
- The conditional `domain_change` emit (INV-5) is one extra
  bookkeeping branch in `_emit_stages_flat` and
  `_emit_stages_partitioned`. The alternative (unconditional
  emit) was rejected for diff cleanliness, but the conditional
  adds a small cognitive cost when reading the build pipeline.

## Cross-references

- ADR [0028](0028-initial-stress-via-parameter-ramping.md) —
  the initial-stress mechanism whose `addToParameter` fan-out
  shares the pre-allocated tag map this ADR establishes (INV-4).
- ADR [0029](0029-staged-analysis-context-manager.md) — the
  staged-analysis context manager that hosts `s.activate(...)`.
- ADR [0027](0027-cross-partition-mp-constraints.md) — the
  cross-partition policy whose per-rank fan-out the Phase SSI-2.C
  follow-on (PR #315) threads through the per-stage activation in
  `_emit_stages_partitioned`.
- [staged-analysis.md](../staged-analysis.md) §"Ownership
  computation (Phase SSI-2.B)" — the rules walkthrough.
- [api-design.md](../api-design.md) §"Staged analysis" — the user
  surface for `s.activate(pgs=...)`.
- [emitter.md](../emitter.md) §"Phase SSI-2 — staged emit" — the
  per-emitter `domain_change` matrix.
- `_internal/build.py::compute_stage_ownership` — the ownership
  computation this ADR specifies.
- `_internal/build.py::allocate_element_tags` — the upfront tag
  allocation this ADR's INV-4 depends on.
- `tests/opensees/unit/test_stage_activation.py` — node + element
  routing coverage, duplicate-PG and global-shared-node rules.
- `tests/opensees/subprocess/test_stage_activation_subprocess.py`
  — Tcl + Py subprocess smoke for the topology-activation path.
