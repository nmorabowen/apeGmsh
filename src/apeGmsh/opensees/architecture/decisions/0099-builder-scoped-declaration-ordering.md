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
tag. Because the `damping` lookup only warns, the deck runs and reports a
**silently undamped** answer. The other five gated classes depend only on an
`NDMaterial`, which survives — `quad` is the sole self-wiping member today.

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

**Deferred, plainly.** `_emit_split` and `_emit_partitioned` **keep the
defect** after S2 — they are not fixed here, they are made **loud** by
INV-4. And the staged bracket **cannot be fixed by hoisting at all**: a
stage-activated gated element brackets after the pattern blocks and after
completed `analyze` calls, so there is no earlier position to hoist to.
That case needs a **replay of the builder-scoped declarations at bracket
close**, which is a different mechanism with its own tag-identity questions.
Future work, not a rider on this ADR.

Note what that leaves, once S4 lands: because S1 refuses split / partitioned
/ staged **at write time**, the only archive a post-S1 apeGmsh can produce
carrying a gated element *and* a builder-scoped declaration is a **flat**
one — and flat replay is exactly what S4 fixes. S4 therefore covers 100% of
post-S1 archives, and its two refusals exist for **pre-S1 data only**.

## Consequences

- **Decks that previously emitted and silently ran wrong now raise.** The
  affected class is split / partitioned / staged emit, with a gated element,
  plus any builder-scoped declaration. This is a **deliberate breaking
  change**, and it is strictly better than the status quo: those decks either
  died 30k lines downstream pointing at the wrong command, or — in the
  `damping` case, which only warns — converged on a wrong answer that nothing
  flagged. A refusal is the smallest honest output for a model we cannot emit
  correctly yet.
- **The flat path's deck line ORDER changes.** Same objects, same tags, same
  results; different positions. Any test pinning absolute deck line numbers
  moves. Content-based and order-insensitive pins are unaffected.
- **The replayed deck's line ORDER changes too** (`build('tcl')` / `('py')`
  / `('live')`), and only for decks that carry both a bracket-needing gated
  element and at least one builder-scoped declaration — a 20-deck corpus
  byte-diffed 18/20 identical across S4a, the 2 that moved being exactly the
  intended hoist. `build('h5')` is byte-unchanged by construction.
- `quad(damp=…)` under a mixed-ndf envelope becomes an emit-time error
  (INV-3). It is not a regression: that combination has never produced a
  damped model, it produced an undamped one without saying so.
- The bracket itself is untouched. `_BUILDER_NDF_GATED` stays as it is, the
  fork gate it encodes stays as it is, and nothing about mixed-ndf authoring
  changes. Only where the four builder-scoped kinds land relative to the last
  re-issue.
- One docstring premise gets corrected rather than deleted:
  `_internal/build.py:2094` is right that the domain survives, and that
  sentence stays — with the registries the destructor *does* purge named
  beside it, so the next reader cannot draw the conclusion this defect was
  built on.
