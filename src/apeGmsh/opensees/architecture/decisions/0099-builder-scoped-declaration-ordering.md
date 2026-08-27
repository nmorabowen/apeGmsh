# ADR 0099 — Builder-scoped declarations must follow the last model re-issue

**Status:** Proposed (2026-08-18) — filed off a blocking defect report from
**Cerro Lindo Hito-3 (Rock–Frame Interaction RevA)**, whose emitted deck dies
at `pattern Plain 1 1`. Everything below was **measured this session**: the
survival table and the fix ordering are both results of running probe decks
through `C:\Program Files\Ladruno\OpenSees\bin\OpenSees.exe` on Ladruno build
`25a0647f`, not readings of the source. The reporter's proposed fix is
insufficient and is refuted below.

## Context

`model BasicBuilder` is not a state tweak — it runs a destructor.
`specifyModelBuilder` (fork `SRC/modelbuilder/tcl/myCommands.cpp:85`) does
`delete theBuilder` before constructing the replacement, and
`~TclModelBuilder()` (fork `SRC/modelbuilder/tcl/TclModelBuilder.cpp:681`)
purges a set of **process-global** registries on the way out. The `Domain` is
a file-static that is reused across builders, which is why nodes and elements
survive a re-issue unharmed.

That last fact is exactly what hid this defect. The apeGmsh docstring at
`src/apeGmsh/opensees/_internal/build.py:2094` says re-issuing `model basic`
"does NOT wipe the domain". That is **true**, and it is the premise the
bracket was designed against — but the domain is not the whole model. The
registries the destructor purges live outside it.

Measured survival across one `model BasicBuilder` re-issue:

| declaration | survives a `model BasicBuilder` re-issue |
|---|---|
| `node`, `uniaxialMaterial`, `nDMaterial`, `section` | yes |
| analysis chain (`constraints` … `analysis`) | yes |
| `timeSeries` | **NO** — `TimeSeries *getTimeSeries(int tag) - none found with tag: N` |
| `geomTransf` | **NO** — `CrdTransf *getCrdTransf(int tag) - none found with tag: N` |
| `beamIntegration` | **NO** — `BeamIntegrationRule - none found with tag: N` |
| `damping` | **NO** — `Damping *getDamping(int tag) - none found with tag: N`, **and the command does not return an error, it only warns** |

The destructor also clears `frictionModel`, `limitCurve`, `damageModel` and
`hystereticBackbone`. None of those four is reachable from apeGmsh today — no
primitive, no namespace, no `Emitter` Protocol method — so the **live**
builder-scoped surface is exactly the four rows above. They are named here
because a future friction-bearing or backbone-carrying authoring surface joins
this set the day it lands, and whoever adds it needs to know that.

### Where apeGmsh puts the bracket, and why the bracket is right

`open_builder_ndf_bracket` / `close_builder_ndf_bracket`
(`_internal/build.py:2085` / `:2112`) wrap each gated element block for the
classes in `_BUILDER_NDF_GATED` (`_element_capabilities.py:818`): `quad`,
`tri6n`, `LadrunoQuad`, `LadrunoCST`, `LadrunoLST`, `LadrunoUP`.

That table is **correct** and the bracket is **necessary**. `OPS_LadrunoLST`
gates on `ndm != 2 || ndf != 2` (`OPS_LadrunoLST.cpp:46`) reading the BUILDER
state — per-node `-ndf` does not satisfy it — so a mixed-ndf model (ndf=3
envelope because of beams, soil nodes inferred ndf=2 per ADR 0048/0049) dies
at the first such element without the bracket. And a `geomTransf` cannot be
created under an ndf=2 envelope, so the envelope restore is equally load-
bearing. Neither half can be dropped. **The defect is purely ordering.**

### All four emit paths order builder-scoped declarations ahead of the bracket

- `_emit_flat` — `apesees.py:1506-1525` (pre_element pass, then
  `emit_transform_specs`), elements at `1530+`, first bracket at `1551`.
- `_emit_split` — `apesees.py:1806-1821`, then the `_emit_element_subset`
  bracket at `1991`. *(As-found. Fixed by S6 — the hoist moves the gated
  fragments' `source` lines, not their element lines; see the
  implementation plan.)*
- `_emit_partitioned` — `apesees.py:2688-2716`, then the
  `emit_element_spec_partitioned` bracket at `_internal/build.py:7220`.
- **H5 replay** — `_internal/compose.py:432-458`. *(As-found. Fixed by
  S4 — replay is the one path here that could be hoisted rather than only
  made loud; see the implementation plan.)*

Plus the staged case, which is the worst of the five: a stage-activated gated
element brackets at `apesees.py:2149`, and for stage index ≥ 2 that bracket
fires **after** the global declarations, **after** the `pattern` blocks, and
**after** one or more completed `analyze` calls. The wipe there is not a
line-order problem at all; it lands mid-history.

### One gated element is self-wiping

`FourNodeQuad` carries `damp: Damping | None` (`element/solid.py:386`),
returns it from `dependencies()` (`solid.py:408-411`), and emits `-damp $tag`
(`solid.py:431`). So `quad(damp=…)` under a mixed-ndf envelope declares its
damping, **destroys it with its own bracket**, and then references the dead
tag. The other five gated classes depend only on an `NDMaterial`, which
survives — `quad` is the sole self-wiping member today.

> **Correction (S4, measured).** This paragraph originally said the deck
> "runs and reports a **silently undamped** answer", on the strength of the
> warn-only row in the survival table above. That is wrong, and it was the
> strongest-sounding justification for INV-3, so it matters. Re-measured on
> the binary, `element quad 101 … -damp 5` against a purged tag **aborts**:
> `WARNING damping not found` / `FourNodeQuad element: 101` / `while
> executing "element quad …"`. The warn-only property belongs to the
> *lookup*, not to element construction — and `damping`'s *add* hard-errors
> too (`MapOfTaggedObjects::addComponent - … similar tag exists`). INV-3
> still stands, but on the honest ground that no ordering can satisfy it and
> the alternative is a deck that dies at the element line, not one that lies
> about damping.

### What RevA hits, and why the reported fix is not enough

RevA's deck dies at `pattern Plain 1 1` (deck line 32332) against a
`timeSeries` declared at lines 1971-1973. The reporter proposed deferring
`timeSeries` past the element pass. That is **insufficient**: deck line 1978
is `geomTransf Corotational 1`, one line above the bracket at 1979, so
deferring only the time series moves the failure to the S2 arch's transform,
and after that to its `beamIntegration`. Fixing this one kind at a time walks
the failure down the table. And the `damping` case is strictly worse than the
reported one — no error, so the deck runs to completion and answers wrong.

## Decision

Builder-scoped declarations get an **ordering contract**, stated over the deck
text rather than over any one emitter.

- **INV-1 — no builder-scoped declaration may precede the last
  `model BasicBuilder` line in an emitted deck.** Every `timeSeries`,
  `geomTransf`, `beamIntegration` and `damping` line lands after the final
  builder re-issue, whatever emitted it and whatever emitted the re-issue.
  This is a property of the deck, so it is checkable on the deck.

- **INV-2 — the builder-scoped kind set lives in exactly ONE table in code**,
  annotated with the fork source line it was audited against, in the same
  style `_BUILDER_NDF_GATED` already uses. There is no second copy in a
  docstring, a validator, or a test. The set is a fork property that will
  change when `~TclModelBuilder()` changes; one table means one line to
  re-audit, and the four currently-unreachable kinds
  (`frictionModel`, `limitCurve`, `damageModel`, `hystereticBackbone`) are
  recorded there as dormant rather than omitted.

- **INV-3 — a gated element may not depend on a builder-scoped primitive.**
  Violation is a **loud emit-time error, never a silent reorder**: the
  dependency is inside the bracket by construction, so no ordering of the deck
  can satisfy both it and INV-1. `quad(damp=…)` violates this today and must
  raise. (The escape is authoring-side — drop the per-element damping, or use
  a non-gated element — not a smarter emitter.)

- **INV-4 — an emit path that cannot satisfy INV-1 refuses to emit** rather
  than producing a deck that dies late or runs wrong. A refusal at emit time
  is cheap; the alternative is the RevA failure (30k lines later, pointing at
  a `pattern` that is not the problem) or the `damping` failure (no message at
  all).

## Implementation plan (slices, each PR-able)

1. **S1 — the table and the check.** `_BUILDER_SCOPED_KINDS` in
   `_element_capabilities.py` (INV-2), next to `_BUILDER_NDF_GATED` and
   annotated with `TclModelBuilder.cpp:681`; plus
   `validate_builder_scope_ordering(...)` in `_internal/build.py`, following
   the house `validate_*` pattern already used by
   `validate_node_ndf_element_compat` / `validate_ladruno_up_solver` /
   `validate_load_basis_vs_elements`, called from `emit()` and from
   `compose.py`. This is a **static check over the spec lists and the emit
   plan** — no runtime ledger, no line-by-line deck scan.

   *Recorded because it will be re-proposed:* the obvious alternative — a
   delegating wrapper `Emitter` that observes every call and orders them — is
   **rejected**, and not on taste. `set_tag_resolver` / `set_element_nodes` /
   `set_current_fem_element_id` (`_internal/tag_resolution.py:109` / `:158` /
   `:313`) stash state **on the emitter object via `setattr`**, and the
   backends read it back off `self`. A wrapper interposed in that path severs
   the channel **silently** — the primitives would emit with stale or missing
   node tags rather than raising. The seam is not wrappable as it stands.

2. **S2 — the hoist, `_emit_flat` only.** Gated element blocks are emitted as
   **one contiguous bracketed run**, placed after the nodes and the surviving
   declarations (materials, sections, analysis chain) and **before** the
   builder-scoped declarations, `emit_transform_specs`, and the ungated
   elements. This works precisely because everything the five non-`quad` gated
   classes need — an `NDMaterial` — survives the bracket (see the table).

   **Element tag allocation is unaffected.** `allocate_element_tags` runs over
   the spec list, not over emit order, so the `fem_eid → ops_tag` identity is
   preserved exactly; only line positions move.

   **Verified on the real binary, not assumed.** A deck with this exact
   ordering — hoisted `LadrunoLST` bracket, then `section` / `geomTransf` /
   `beamIntegration` / `timeSeries`, then `dispBeamColumn`, then `pattern` —
   ran through `OpenSees.exe` `25a0647f` and every object built.

3. **S3 — tests + `repros/repro4_builder_scoped_wipe.py`.** The repro carries
   the mechanism end-to-end (mixed-ndf gated block + each of the four kinds),
   so a fork change to `~TclModelBuilder()` shows up as a failing repro rather
   than as a field report.

4. **S4 — the H5 / deck-replay hoist + two legacy refusals.** Replay walks
   ALREADY-RESOLVED records, so tags are fixed and reordering only moves
   deck lines — this is the one deferred path that could be genuinely
   *fixed* rather than only made loud.

   *S4a — the hoist.* `needs_builder_ndf_bracket_for_token` (the token core
   of the existing spec predicate — `element_builder_ndf` already resolves
   both spellings, so they cannot disagree); `_replay_into` partitions its
   elements on it and emits the gated run at a new **step 4b**, above the
   four declarations. Two conditions gate the partition, and both matter:
   the predicate is **envelope-aware** (a gated class under a matching
   envelope needs no bracket and must not move, or every ndf=2 quad deck
   reorders for nothing), and the hoist is skipped when the deck carries
   **no builder-scoped declaration at all** (INV-1 then holds vacuously, so
   reordering would be pure churn). Gatedness is resolved once per DISTINCT
   token — per-element evaluation measured **+13-15%** on emit of an
   ungated-only 200k-element deck, a cost landing precisely on the models
   the hoist cannot help.

   `build('h5')` **opts out** (`deck_ordering=False`): `H5Emitter.model`
   only stores ndm/ndf, so the bracket is never persisted as a line and the
   hoist buys that path nothing — while skipping it keeps the
   archive-rewrite fixed point by construction rather than by argument.
   `build('live')` deliberately does **not** opt out: measured, openseespy
   runs no `TclModelBuilder` destructor (a `timeSeries` and a `geomTransf`
   declared before an in-process `model` re-issue both survive), so live
   never had this defect — the hoist is kept only so the ordering holds for
   any backend rather than resting on one backend's teardown behaviour.

   *S4b — the refusals.* `validate_builder_scope_replay`, a record-shaped
   sibling of `validate_builder_scope_ordering` (replay has type tokens and
   flat arg tails, not specs with `dependencies()`). Two arms: **INV-3**, a
   gated record referencing a builder-scoped declaration through its arg
   tail (`quad ... -damp N`); and **INV-4**, a stage-owned gated element,
   guarded in `_replay_staged_into` and NOT in `_replay_into` — the H5
   re-emit path replays a staged archive through the flat helper and must
   stay able to rewrite one. The element arg-tail flags (`-damp`,
   `-fx/-fy/-fz`) become a **third column on `_BUILDER_SCOPED_KINDS`**
   rather than a literal at the call site, keeping INV-2's one-table rule.

   Both arms are **pre-S1 legacy defence**: today's bridge refuses to write
   either archive. Note the INV-3 severity is smaller than the ADR's
   authoring-side case — measured post-S4a, that deck *aborts* at the
   element line rather than running silently undamped, so the guard buys an
   emit-time error in place of a runtime abort, not a silent-wrong rescue.

5. **S5 — the partitioned hoist.** The `_emit_flat` pass replicated on
   `_emit_partitioned`: one extra `if {[getPID] == K}` guard per rank that
   OWNS a gated element, carrying the nodes those elements need and then
   the bracketed blocks, placed between the surviving declarations and the
   builder-scoped ones. Ranks owning none get no block — an empty guard is
   dead weight in Tcl and a syntax error in the Python emitter. The INV-4
   refusal is lifted for `partitioned` and kept for `partitioned per_rank`
   (see below).

   **Gated on both conditions, so unhoisted decks are byte-identical.**
   With no builder-scoped declaration INV-1 holds vacuously; with no
   rank-OWNED gated row no bracket ever fires. In either case the
   pre-element pass stays one topo-ordered loop and the deck does not move
   a byte — measured on three variants (nothing gated / nothing scoped /
   neither), byte-identical against the pre-S5 emitter.

   **Element tag allocation is unaffected**, by the same argument S2
   recorded: `allocate_element_tags` (and the per-rank bucketing that
   reads it) moves above the pre-element pass so the hoisted pass has a
   plan, and `TagAllocator` is per-kind, so the element and `geomTransf`
   counters do not interact. The staged branch has always allocated
   there.

   **Verified on the real binary, per rank.** A 2-rank mixed-ndf deck —
   quads on rank 0, frame on rank 1, all four scoped kinds global — ran
   through `OpenSees.exe` once per rank (a `proc getPID` driver
   pre-empting the deck's own shim) and matched the flat reference to
   every printed digit on both. Neutering the hoist reproduces the
   pre-S5 split exactly: rank 0 dies at `pattern Plain` with all four
   kinds listed as INV-1 offenders, rank 1 still passes. That arm is
   `repros/repro4_builder_scoped_wipe.py --partitioned`; the S4a replay
   is `--h5`.

   > **Correction (measured, post-S5).** As shipped, the scoped half of
   > the gate read `pre_element` alone — and `GeomTransf` is pre-binned
   > into the separate `transforms` list by `emit` step 3, so it never
   > reaches `pre_element`. A partitioned deck whose **only**
   > builder-scoped declaration is a `geomTransf` therefore skipped the
   > hoist and emitted the INV-1 violation this slice exists to prevent:
   > measured on a 3-rank fixture (gated `FourNodeQuad` +
   > `elasticBeamColumn`, no beamIntegration / timeSeries / damping),
   > rank 0's stream carries `geomTransf Linear 1` above the last of its
   > three `model` lines while ranks 1 and 2 are clean — the rank-local
   > asymmetry again, undetected because the S5 fixture bundles all four
   > kinds and the other three *do* reach `pre_element`. The gate now
   > counts `transforms` alongside `pre_element`. `pre_scoped` itself
   > stays `pre_element`-only: it is the hold-back list for step 1c, and
   > the transform fan-out already emits below the hoist unconditionally.
   > Both gates still matter — nothing gated means no bracket fires, so a
   > transform-only deck with no gated row keeps its shape byte for byte.
   > Pinned by `test_s5_transf_only_deck_hoists_on_every_rank` and
   > `test_s5_transf_only_deck_does_not_move_without_a_gated_row`.
   >
   > The general lesson, which the next emit path should inherit: a gate
   > over "is any builder-scoped declaration present" must read every
   > list the pre-bin splits primitives into, not the one that happens to
   > hold most of them.

6. **S6 — the split hoist.** The measured design below ("Split IS
   hoistable"), implemented as layout reuse: the hoist moves each gated
   module's **`source` line**, never its element lines.
   `_emit_split`'s driver-pre splits in two — the surviving definitions,
   then a HOISTED band of gated-module fragments, then the
   builder-scoped declarations + the `geomTransf` fan-out — and the
   Tcl / Py writers place the hoisted band's `source` lines between the
   two halves.  A module owning only gated (and scoped-independent)
   elements hoists wholesale: its fragment is exactly what the ADR 0043
   band would have written, sourced earlier.  A module owning BOTH a
   gated element and a builder-scoped-dependent one — dependence
   recognised through the one table, `builder_scoped_kind` over
   `dependencies()`, per INV-2 — emits as the two ordered fragments the
   measurement predicted: `<m>_gated` carries ALL the module's nodes
   plus the bracketed blocks, `<m>_rest` the remaining elements + fix +
   mass in the ADR 0043 band, so `_rest` never forward-references a
   node.  (S5 hoisted only the NEEDED nodes into its rank block; here
   the whole node set rides `_gated` — a fragment is a file, and
   splitting a module's node list across two files buys nothing but a
   harder diff.)

   **Gated on both conditions, so unhoisted decks are byte-identical**
   — with one deliberate widening over S5's gate: scoped-present counts
   the `transforms` list alongside `pre_element`, because
   `emit_transform_specs` is itself a builder-scoped declaration pass
   and GeomTransf primitives are pre-binned OUT of `pre_element`.
   *Measured while implementing:* S5's gate reads only `pre_element`,
   so a partitioned deck whose only scoped declarations are geomTransf
   lines (gated quads + `elasticBeamColumn`, no beamIntegration /
   timeSeries / damping) skips the S5 hoist and emits a rank-0 stream
   with `geomTransf Linear 1` above its bracket — an INV-1 violation
   S5's fixtures never constructed, because every frame there carried a
   beamIntegration.  Fixed separately; see the post-S5 correction above.
   Byte-identity measured on six no-hoist variants (no elements at all,
   scoped-only frame, mixed-ndf nothing-gated, mixed-ndf
   nothing-scoped; tcl + py): driver and every fragment byte-identical
   against the pre-S6 emitter.

   **Element tag allocation is unaffected**, by the S2/S5 argument:
   `allocate_element_tags` moved above the driver-pre pass so the
   hoisted band has a plan; allocation emits nothing, and TagAllocator
   is per-kind, so the element / geomTransf counters do not interact.

   **Verified on the real binary.** A mixed-ndf split deck — two gated
   quads (module `Soil`), a damped `dispBeamColumn` (module `Frame`),
   all four scoped kinds — loads clean, zero INV-1 offenders in the
   executed stream (the driver with each `source` line inlined), and
   matches the flat reference deck to every printed digit:
   `nodeDisp 5` = `0.00000000000000000010 -0.00477430555555555594`,
   `nodeDisp 9` = `0.00000000000000000000 -0.13333333333333358128
   -0.10000000000000019984` — identical strings on both decks, i.e.
   bit-identical doubles.  The one-module boundary variant
   (`Mix_gated` / `Mix_rest`, the damped beam in the SAME module as the
   quads) matches identically.  Neutering the hoist reproduces the
   pre-S6 failure exactly: all four kinds flagged as INV-1 offenders
   and the driver dying at `pattern Plain` (driver line 19) against the
   purged `timeSeries`.  That arm is
   `repros/repro4_builder_scoped_wipe.py --split`.

   *Rejected:* reordering in the WRITER — leave the emit order alone
   and move only the `source` lines at write time.  The writer would
   then need its own copy of the gate and of the gated/dependent
   partition (a second table by another name, against INV-2), and the
   emitted buffer would stop reading in execution order.  The emit-side
   restructure keeps the buffer in execution order — the same shape S2
   and S5 left `_emit_flat` / `_emit_partitioned` in — and the writers
   stay pure slicers over a `hoisted` band `_SplitLayout` now carries.

7. **S7 — the staged replay.** The stage-activated gated-element case,
   the one path hoisting cannot reach: the bracket fires INSIDE the
   stage block — after the global declarations, after the pattern
   blocks, and (for any stage past the first) after completed `analyze`
   calls.  Implemented exactly under the constraints the self-healing
   rejection below records: scoped to the STAGED paths alone
   (`_emit_stages_flat` step 3 + `_replay_staged_into`'s stage loop),
   fired only after a bracket actually emitted in that stage, and gated
   on the runtime that actually purges — never a property of
   `close_builder_ndf_bracket`.

   *Hoist first, replay only what's left.*  Within the stage block the
   bracket-needing specs emit FIRST, then the replay
   (`replay_builder_scoped_declarations`), then the ungated stage-owned
   elements — a stage-activated `dispBeamColumn` resolves `geomTransf`
   / `beamIntegration` by tag on its own element line and must see the
   replayed registry.  A staged model whose gated elements are all
   GLOBAL keeps taking the S2 / S4a hoist: no stage bracket fires and
   nothing is replayed.  Armed on the same two conditions as every
   0099 hoist (a stage-owned bracket-needing element AND a
   builder-scoped declaration for it to destroy), so every other
   staged deck does not move a byte, and element tags come from the
   same pre-allocated plan — only line positions move.

   *The replay content.*  Bridge side: the 5b `pre_scoped` primitives
   re-emit through their own deterministic `_emit`; the transform
   fan-out cannot be re-RUN (it allocates per-vecxz tags), so
   `emit_transform_specs` grows a `replay_log` capture of the emitted
   lines that the replay re-drives verbatim.  Replay side: the four
   record lists re-emit in the step 5–7b order.

   *The backend gate, measured — and narrower than the rejection
   paragraph below guessed.*  That paragraph said "gate it on the deck
   backends"; measurement says gate it on the **Tcl deck alone**.  On
   the fork's in-process module — which runs BOTH `build('live')` and
   the emitted **py deck** — `ops.model()` purges nothing (a
   `timeSeries` / `geomTransf` / `beamIntegration` declared before the
   re-issue all survive), and re-declaring any of them hard-errors
   (`MapOfTaggedObjects::addComponent - … similar tag exists`).  A
   replay on the py deck would therefore kill the deck it exists to
   save.  The gate is an emitter attribute
   (`TclEmitter.model_reissue_purges = True`; everyone else takes the
   `getattr` default `False`).  The py deck still gets the bracket and
   the within-stage gated-first ordering — harmless there, and the
   ordering then holds for any backend, matching the S4a live decision.

   **Tag identity, measured on the binary** — the questions the
   deferral paragraph left open, each probed as a mutated replay
   against a bracket-free reference (Ladruno `25a0647f`, LoadControl /
   Newmark, 16 printed digits):

   - A `pattern` built before the bracket holds a **private copy** of
     its series: replaying `timeSeries Constant 1 -factor 0.0` over a
     purged `Linear 1` mid-LoadControl changes nothing — 16/16 digits
     against the reference.  It does not re-resolve by tag.
   - A `dispBeamColumn` built before the bracket holds
     **construction-time copies** of its `CrdTransf` and
     `BeamIntegration`: replaying `Corotational` + `Legendre` under
     the original tags (with axial + lateral load, where the transform
     class matters) changes nothing — 16/16 digits.
   - `region -damp` **copies the damping into the elements at the
     region line**: replaying a 10x-zeta `damping Uniform` under the
     same tag mid-transient changes nothing — 16/16 digits.
   - The purge ALONE also changes nothing for already-constructed
     objects (16/16 digits with no replay at all): only future by-tag
     lookups die, which is exactly the failure class the replay
     serves.

   So a re-declaration cannot retro-write integrated history, and every
   post-bracket by-tag reference (`pattern`, later `element` lines,
   `region -damp`) resolves the replayed — identical — declarations.

   **Acceptance, measured on the real binary.**  A staged two-stage
   mixed-ndf model — global frame + all four scoped kinds + a stage-1
   pattern and completed analyze; gated quad soil activated at stage 2
   with a stage-2 pattern resolving the series AFTER the bracket —
   matches the same model's py deck run under the in-process module
   (the form that never needed the replay, since that runtime never
   purges) to **all 16 printed digits** on every probed displacement
   component, for both the forward tcl deck and the
   `h5 → from_h5 → build('tcl')` round-trip.  Stripping the replayed
   lines from the accepted deck reproduces the RevA failure mid-stage
   (`TimeSeries *getTimeSeries(int tag) - none found with tag: 1` at
   `pattern Plain 2 1`) — `repros/repro4_builder_scoped_wipe.py
   --staged` carries both probes.

   *What stays refused, plainly.*  Partitioned staged gated — the
   per-rank stage machinery has no replay, so
   `validate_builder_scope_ordering`'s staged arm now keys on
   `path != "flat"` — plus split and `per_rank` as before.  INV-3 is
   untouched: a stage-owned `quad … -damp N` record still refuses,
   because the purge lands at bracket OPEN, before the element line
   parses, so no stage-close replay can save it (measured — the deck
   aborts at the element line); the arg-tail scan moved into
   `_replay_staged_into` because the global `_replay_into` scan skips
   stage-owned records.  S4b's INV-4 arm (and its `stage_owned_tags` /
   `scoped_present` parameters) is deleted — the case it refused is now
   emitted correctly, and the write-time staged refusal in
   `validate_builder_scope_ordering` is lifted for the flat path, so a
   stage-owned gated element now archives to h5 and replays.

**Deferred, plainly.** `_emit_split` and `_emit_partitioned` **keep the
defect** after S2 — they are not fixed there, they are made **loud** by
INV-4. *(S5 has since fixed the default partitioned path and S6 the
split path; what remains refused is `per_rank=True` alone — its span
writer slices recorded guard spans out of a finished buffer post-emit
and cannot reorder them yet.  The section below is what S5 and S6 were
built from.)* And the staged bracket **cannot be fixed by hoisting at all**: a
stage-activated gated element brackets after the pattern blocks and after
completed `analyze` calls, so there is no earlier position to hoist to.
That case needs a **replay of the builder-scoped declarations at bracket
close**, which is a different mechanism with its own tag-identity questions.
*(S7 has since shipped that replay on the flat staged path — Tcl deck
only, with the tag-identity questions answered by measurement above.
What remains refused after S7: split, `per_rank`, and a stage-activated
gated element on the PARTITIONED staged path.)*

### How the two deferred paths should actually be fixed (measured, S4)

Recorded because the obvious reading of the paragraph above — that split and
partitioned are the same problem as staged — is **wrong**, and it cost a
round of design before an adversarial probe measured it.

- **Partitioned is the EASY case, not a sibling of split.** Default
  partitioned emit is **one file** with `if {[getPID] == K}` brace guards;
  the builder-scoped declarations sit global, *outside* every guard. A brace
  is not a file boundary, so the S2 hoist replicates directly: one extra
  rank-guard block carrying that rank's gated elements and their nodes,
  placed above the declarations. Measured exact against the flat reference on
  both ranks. Note also that the damage is **rank-local** — a rank owning no
  gated element never executes the bracket and runs fine today, so the
  current failure is non-deterministic in `np`. Only `per_rank=True` produces
  the file-per-rank layout that resembles split. **Implemented in S5**,
  including the rank-local asymmetry as a regression pin: the fixture puts
  the gated block on one rank of three, so a hoist that fixed only the
  file-wide reading would still fail it.

- **Split IS hoistable — you move the `source` line, not the nodes.** The
  claim that hoisting a gated element would drag its nodes out of its
  fragment is false: hoisting the gated module's `source` statement above the
  declarations carries the nodes along *inside* the fragment, which stays
  byte-identical. Measured: loads clean, matches the flat reference to 12
  digits. The genuine boundary is narrower — it fails only when a **single
  module** carries both a gated element and a builder-scoped-dependent
  element, and even then that module can emit as **two ordered fragments**
  (`A_gated` / `A_rest`), a shape ADR 0061's `per_rank` writer already
  produces. So this is layout reuse, not new machinery. **Implemented in
  S6**, including the boundary as a regression pin: the one-module
  fixture carries the gated quads and the transf-dependent beam in the
  same module, so a hoist that only moved whole fragments would make the
  beam forward-reference its transform and fail it.

- **Do NOT make the bracket self-healing.** The tempting unification —
  `close_builder_ndf_bracket` re-declaring what it destroyed — is
  **rejected on measurement**. Re-declaring a tag that was *not* purged
  hard-errors for all four kinds (`MapOfTaggedObjects::addComponent - …
  similar tag exists`), and `ops.model()` under openseespy purges nothing, so
  on `build('live')` the **second** heal collides — while split and
  partitioned always carry M ≥ 2 brackets. It would fail on live for exactly
  the model class it exists to fix, and it converts a deck-ordering property
  into a backend-conditional one. If a heal is ever wanted for the staged
  case, scope it there alone, gate it on the deck backends, and gate it on a
  bracket having actually fired — never as a property of the bracket itself.
  Even in staged, hoist every gated element that is not stage-activated first;
  only truly stage-activated ones need the replay.  *(S7 has since shipped
  exactly that heal, under exactly these constraints — and measured the
  backend gate NARROWER than "the deck backends": the emitted py deck runs
  under the same in-process module as live, which purges nothing and
  hard-errors on re-declaration, so the replay is gated on the Tcl deck
  alone, via `TclEmitter.model_reissue_purges`.)*

The replay mechanism itself is **sound where it is needed** (measured): an
element built before a purge keeps working — `FourNodeQuad` stores
`(*damping).getCopy()` per integration point, so a damped quad stays damped
across a purge (0.0239 vs 0.0535 undamped, identical to 6 s.f. before and
after) — a `timeSeries` already bound to a `pattern` is a private copy that
re-declaration neither orphans nor mutates, and 100 declare/bracket/replay
cycles are bit-identical.

Note what that leaves, once S4 lands. S1's write-time refusal is narrower
than it first reads: it refuses split and partitioned outright, but refuses
a staged model only when a gated element is **stage-OWNED**
(`build.py`, `staged = [g for g in gated if id(g) in element_owner_stage]`)
— a staged model whose gated element is activated **globally** writes to h5
without complaint. So the post-S1 archive set carrying a gated element *and*
a builder-scoped declaration is **flat archives plus globally-activated
staged ones**, not flat alone.

S4 still covers all of them, but for a mechanical reason rather than by
elimination: `_replay_staged_into` delegates its global prefix to
`_replay_into`, so the S4a hoist fires there too. *Measured* — such an
archive replays with zero INV-1 offenders (`element quad` above
`geomTransf`) and runs clean on the binary. The two S4b refusals remain
**pre-S1 defence**: today's bridge still cannot write a stage-OWNED gated
element or a `quad ... -damp`. *(S7 has since deleted the S4b INV-4 arm
and lifted the write-time staged refusal for the flat path — a
stage-OWNED gated element now archives to h5 and replays through the
stage-close mechanism; only the INV-3 refusal remains, its arg-tail scan
moved into `_replay_staged_into` for the stage-owned records the global
scan skips.)*

## Consequences

- **Decks that previously emitted and silently ran wrong now raise.** The
  affected class is `per_rank=True` emit, or staged emit with
  a **stage-owned** gated element (a globally-activated one is not
  refused), plus any builder-scoped declaration. *(Default partitioned
  emit was in this set until S5 hoisted it, and split until S6 moved the
  gated fragments' `source` lines; `per_rank` stays because its span
  writer reorders nothing yet — the remaining fix is S6's source-line
  move applied post-emit.  S7 removed FLAT staged emit with a
  stage-owned gated element from the set too — it now emits with the
  stage-close replay; the staged refusal survives only on the
  partitioned path.)* This is a **deliberate breaking
  change**, and it is strictly better than the status quo: those decks either
  died 30k lines downstream pointing at the wrong command, or — in the
  `damping` case, which only warns — converged on a wrong answer that nothing
  flagged. A refusal is the smallest honest output for a model we cannot emit
  correctly yet.
- **INV-1 is amended for the staged Tcl deck (S7).** A stage-activated
  gated element's bracket is a `model` line that legitimately FOLLOWS
  the global builder-scoped declarations; the invariant there becomes:
  every builder-scoped declaration purged by the last `model` line is
  **re-declared after it**, same lines, same tags. Non-staged decks keep
  the original reading unchanged.
- **The flat path's deck line ORDER changes.** Same objects, same tags, same
  results; different positions. Any test pinning absolute deck line numbers
  moves. Content-based and order-insensitive pins are unaffected. The same
  holds for the partitioned path after S5 and for the split driver after
  S6 (fragment `source` positions move; a wholesale-hoisted fragment's
  own bytes do not), and only for decks that carry both a
  bracket-needing gated element and a builder-scoped declaration —
  every other partitioned or split deck is byte-unchanged.
- **The replayed deck's line ORDER changes too** (`build('tcl')` / `('py')`
  / `('live')`), and only for decks that carry both a bracket-needing gated
  element and at least one builder-scoped declaration — a 20-deck corpus
  byte-diffed 18/20 identical across S4a, the 2 that moved being exactly the
  intended hoist. `build('h5')` is byte-unchanged by construction.
- `quad(damp=…)` under a mixed-ndf envelope becomes an emit-time error
  (INV-3). It is not a regression: that combination has never produced a
  damped model. It dies at the element line (measured — see the correction
  above; it does NOT run undamped), so this trades a failure 30k lines
  downstream for one at emit.
- The bracket itself is untouched. `_BUILDER_NDF_GATED` stays as it is, the
  fork gate it encodes stays as it is, and nothing about mixed-ndf authoring
  changes. Only where the four builder-scoped kinds land relative to the last
  re-issue.
- One docstring premise gets corrected rather than deleted:
  `_internal/build.py:2094` is right that the domain survives, and that
  sentence stays — with the registries the destructor *does* purge named
  beside it, so the next reader cannot draw the conclusion this defect was
  built on.
