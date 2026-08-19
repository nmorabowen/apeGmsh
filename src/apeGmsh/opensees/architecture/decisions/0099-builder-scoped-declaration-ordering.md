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
  bracket at `1991`.
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

**Deferred, plainly.** `_emit_split` and `_emit_partitioned` **keep the
defect** after S2 — they are not fixed there, they are made **loud** by
INV-4. *(S5 has since fixed the default partitioned path; what remains
refused is `_emit_split` and `per_rank=True`, the two file-per-fragment
layouts — see the section below, which is what S5 was built from.)* And the staged bracket **cannot be fixed by hoisting at all**: a
stage-activated gated element brackets after the pattern blocks and after
completed `analyze` calls, so there is no earlier position to hoist to.
That case needs a **replay of the builder-scoped declarations at bracket
close**, which is a different mechanism with its own tag-identity questions.
Future work, not a rider on this ADR.

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
  produces. So this is layout reuse, not new machinery.

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
  only truly stage-activated ones need the replay.

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
element or a `quad ... -damp`.

## Consequences

- **Decks that previously emitted and silently ran wrong now raise.** The
  affected class is split emit, `per_rank=True` emit, or staged emit with
  a **stage-owned** gated element (a globally-activated one is not
  refused), plus any builder-scoped declaration. *(Default partitioned
  emit was in this set until S5 hoisted it; `per_rank` stays because it
  turns each rank guard into a separate FILE, which is the `split`
  problem, not the partitioned one.)* This is a **deliberate breaking
  change**, and it is strictly better than the status quo: those decks either
  died 30k lines downstream pointing at the wrong command, or — in the
  `damping` case, which only warns — converged on a wrong answer that nothing
  flagged. A refusal is the smallest honest output for a model we cannot emit
  correctly yet.
- **The flat path's deck line ORDER changes.** Same objects, same tags, same
  results; different positions. Any test pinning absolute deck line numbers
  moves. Content-based and order-insensitive pins are unaffected. The same
  holds for the partitioned path after S5, and only for decks that carry
  both a bracket-needing gated element and a builder-scoped declaration —
  every other partitioned deck is byte-unchanged.
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
