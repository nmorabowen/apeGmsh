# ADR 0113 — The IMPL-EX time driver, the adaptive transient stage, and the dTime trap

**Status:** Proposed (2026-10-03). Slice 1 (D1–D4 and D6) is implemented on
branch `feat/implex-time-driver`: `ops.implex_time(mode=...)` in
[`analysis/implex.py`](../../src/apeGmsh/opensees/analysis/implex.py) and
[`_internal/implex.py`](../../src/apeGmsh/opensees/_internal/implex.py),
three Emitter verbs, and the staged flat and partitioned emit paths. D5 and
D7 are decided here and not implemented. No schema bump: the declaration
does not round-trip H5, and the H5 emitter refuses a model that drives.
It stays Proposed until the 2-rank OpenSeesMP smoke in
`repros/adr0113_implex_mpi_smoke/` passes (D3, "Per-rank attachment").

**Completes:** [ADR 0104](0104-substep-run-to-criterion-controller.md) D6 for
the *transient* case (D5 below): STKO's adaptive time step as a typed spec
whose loop is emitted into the deck. ADR 0104's displacement-controlled
`Substep` stays in-process.

**Came from:** the San Ramon STKO → apeGmsh port (research repo
`Epistemic Uncertanty`, `models/apegmsh/`), which needed all of this and wrote
it as text patches on the emitted deck: `sanramon/staging.py`
(`prelude_lines`, `proc sr_implex_dt`, `adaptive_loop_lines`,
`finalize_deck`), `sanramon/model.py::insert_excitation`, recipe
`recipes/1D.md` §2.6 and §4.6, gate G20 in
`validation/driver_equivalence_1D.md`, and the bridge gaps listed in
`FINDINGS.md` ("apeGmsh bridge gaps hit"). The translator (ADR 0111, rules
C12–C14 in `internal_docs/stko_translator_rules.md`) refuses or leaves as data
the same features.

## Context

### What ASDConcrete does with time

ASDConcrete3D and ASDConcrete1D read a time increment `dtime_n` for two
things: the IMPL-EX extrapolation ratio `dtime_n / dtime_n_commit * implexAlpha`
and the viscous factor `eta / (eta + dtime_n)`. Source, OpenSees fork at
`fd87e396d` (2026-09-29), `SRC/material/nD/ASDConcrete3DMaterial.cpp`:

- `setTrialStrain` (l. 1624): `if (!dtime_is_user_defined) dtime_n = ops_Dt;`
  — OpenSees' own increment, current minus committed domain time.
- `updateParameter` (l. 2101–2111): the parameters `dTime`, `dTimeCommit`,
  `dTimeInitial` (ids 2000–2002) set `dtime_n`, `dtime_n_commit`, `dtime_0`
  **and set `dtime_is_user_defined = true`**. Only `revertToStart`
  (l. 1809) clears it. One write, by any route, and the material stops
  following `ops_Dt` for the rest of the run.
- `commitState` copies `dtime_n_commit = dtime_n` (l. 1771);
  `revertToLastCommit` restores `dtime_n = dtime_n_commit` (l. 1787).
- the time factor (l. 2333–2335) and the IMPL-EX error control with
  `-implexControl tol red` (l. 1662–1683; the step is refused while
  `dtime_n >= red * dtime_0`). The 1-D twin has the same logic
  (`ASDConcrete1DMaterial.cpp` l. 1008, 1339).

STKO writes all three before **every attempted increment** of every stage
(`STKO_DT_UTIL_OnBeforeAnalyze`, `setParameter -val $dt -ele $e dTime` per
target, and `dTimeCommit` / `dTimeInitial` while the stage's increment
counter is 1). So the IMPL-EX ratio restarts at 1 at every stage start, and
`dTime` always equals the increment being attempted.

### The trap

A deck that writes `dTime*` once and then steps with another increment runs
on a stale `dTime`: the materials extrapolate and regularize with the old
step and nothing reports it. The translator's rule C13 writes the reset once
per static stage through `s.update_parameter`, which is right for those
stages (constant increment) and wrong for whatever runs next unless that
writes `dTime` too. The translator says so in prose ("the time-history
driver must set `dTime` before every transient step"); nothing enforces it.

### What the port measured (G20, serial)

`validation/driver_equivalence_1D.md` (a **serial** run), one 1D wall panel (ASDShellQ4 →
LayeredShell → PlateFromPlaneStress → ASDConcrete3D) plus one fiber column
(forceBeamColumn → FiberSection3d → ASDConcrete1D), gravity, hold, and 1600
adaptive transient steps with five failed attempts:

1. The deck-patch driver (persistent `parameter` / `addToParameter` /
   `updateParameter`) and STKO's own `setParameter -val -ele` loop give
   **bitwise identical** results: 35 recorder files, the step log, and a
   per-attempt `dTime` probe. One call per constant-increment stage equals
   STKO's call per increment.
2. `updateParameter` reaches every concrete point: all 216 shell points and
   1280 column fibers read back the stage's increment at every stage start.
   PlateRebar → Hysteretic and the column's torsion answer −1 and are
   skipped.
3. Switching the driver off moves the response (column 3.7e-5 at the hold
   stage start, O(1) in the shells' drilling moments because of
   `-drillingNL`, F2 below).
4. MPI: these points come from **reading the source**, not from G20, which
   ran serially. `addToParameter` on an element the rank does not hold
   prints "no objects were able to identify parameter"
   (`Parameter::addComponent`, a warning, not a Tcl error). A persistent
   parameter keeps raw pointers to the materials collected at
   `addToParameter` time, so it must be built after the elements and
   breaks if they are removed. Under OpenSeesSP the master copies are not
   the analysed materials. The only MPI run so far is the port's G21
   esmeralda smoke, which exercised its runtime `getEleTags` form, not the
   bridge's static per-rank form (D3).

### What else the port had to patch

`FINDINGS.md`: no `UniformExcitation` inside a stage (lines inserted by
`insert_excitation`); no adaptive transient loop (the bridge's fixed loop
replaced by a port of STKO's template); plus four MPI/reader gaps listed in
D8.

## Decision

### D1 — A model-wide typed declaration, `ops.implex_time(mode=...)`

```python
ops.implex_time()            # mode="stko"
```

`ImplexTime(mode)` is a frozen spec, not a registered primitive (nothing to
tag or order), like a `Ladder`. It is model-wide because the parameters are
persistent across stages and the trap is a cross-stage property. Modes:

| mode | meaning |
|---|---|
| `"stko"` | STKO's driver (D3). Shipped. |
| `"off"` | no driver, and **nothing may write** `dTime*`: the materials follow `ops_Dt` for the whole run. Declaring it makes that choice explicit (the port's `twin_implex_off` handshake) and lets the bridge refuse a deck that contradicts it. |
| `"follow"` | follow `ops_Dt` but restart the IMPL-EX ratio at a stage start. Needs fork parameter F1 (D8); refused at construction until it exists. |
| undeclared | no driver; the trap check (D6) runs. |

*Alternatives.* (a) A per-stage verb (`s.implex_dt(...)`): rejected, the
parameter graph and the trap span stages. (b) Keep `s.update_parameter` per
stage (today's translator): correct only for constant-increment stages,
cannot serve an adaptive loop, and creates and removes one parameter per
stage. (c) Emit STKO's `setParameter -val -ele` loop: G20 shows identical
results, but each call builds a temporary `ElementStateParameter` per
element per attempt; the persistent route builds the graph once.

### D2 — Targets come from the material graph, never from ids

A target is an element spec whose `dependencies()` closure (charter P5, the
same edges the emit order uses) reaches a material that reads `dTime`:
ASDConcrete3D / ASDConcrete1D with `implex=True` **or** `eta > 0`, and
ASDSteel1D with `implex=True` (same `dtime_is_user_defined` switch:
`ASDSteel1DMaterial.cpp` l. 2152, `setParameter` l. 2590–2598,
`updateParameter` l. 2614–2622; it has no `eta`). That is STKO's C13
predicate restricted to the types the bridge has; `DamageTC1D/3D` and
`ASDBondSlip` join `reads_dtime` when typed. Every row of a target spec is a
target. A driver with no target is refused (it would update nothing), and so
is a target spec whose physical group has no elements. The propagation chains G20 verified are the shell and fiber chains
above; an element type whose `setParameter` does not forward to its
materials would print OpenSees' "no objects" warning (follow-up: a `live`
read-back test per chain).

### D3 — Emission: a prelude, then one call per stage

After every global element and before the first stage banner (where the
port put it):

```tcl
# apeSees IMPL-EX time driver (ADR 0113): persistent dTime / dTimeCommit / dTimeInitial parameters
parameter 1
parameter 2
parameter 3
proc _apesees_implex_dt {dt first} {
    if {$first} {
        updateParameter 2 $dt
        updateParameter 3 $dt
    }
    updateParameter 1 $dt
}
foreach _apesees_e [list \
    11 12 21 22] {
    addToParameter 1 element $_apesees_e dTime
    addToParameter 2 element $_apesees_e dTimeCommit
    addToParameter 3 element $_apesees_e dTimeInitial
}
```

Parameter tags come from the bridge's allocator (P4). Then in every stage,
immediately before its analyze loop (after the chain, the stage patterns,
the recorders, `reset` and the velocity zeroing, where STKO's
`OnBeforeAnalyze` hook sits): `_apesees_implex_dt <increment> 1`. It must
follow `reset`, because `reset` reverts the materials to their start state
and clears `dtime_is_user_defined` (ASDConcrete3D l. 1809). The `py` deck
has the same `def _apesees_implex_dt(dt, first)` and calls. The live emitter
**refuses** in its prelude verb: staged live runs are refused at
`stage_open` anyway, and the prelude comes first, so without its own refusal
it would already have created parameters in the live domain. The H5 emitter
**refuses** (no store; a replay would run without the driver), as it already
does for `s.update_parameter`.

*Why one call per stage is enough in slice 1.* Each stage steps with one
known increment: `Static` + `LoadControl(dlam)` (`ops_Dt = dlam`; with
`num_iter` only when `min_lam == max_lam == dlam`, since
`LoadControl::newStep` clamps every increment to `[min, max]`), or
`Transient` + `run(dt=)`. With `dTime` constant within
the stage, `commitState` / `revertToLastCommit` keep `dtime_n` equal to it,
so a write per increment changes nothing (G20 point 1). Any other stage
(`VariableTransient`, `DisplacementControl`, arc length, adaptive
`LoadControl`) is refused with a pointer to D5, whose loop calls the proc
before every attempt with `[expr {$increment == 1}]`. The proc signature is
already that per-attempt form.

*Per-rank attachment.* On a partitioned deck the declaration and the proc
are global and each rank's `addToParameter` loop sits in its
`if {[getPID] == K}` block with the targets it owns, from the ADR 0027
ownership map every other per-rank emit uses. The port intersected with
`getEleTags` at run time instead. On a bridge-partitioned deck the two
should be the same set, because the same map decides which `element` lines
go into each rank's block. The golden test checks this without trusting the
map: it reads each rank's elements from the `element` lines in that rank's
`getPID` blocks, feeds them to the port's prelude as `getEleTags`, and
requires the same command log. The static form was chosen because it follows
ADR 0027, carries only each rank's ids into its block (and into its fragment
with `per_rank=True`), and extends to stage-activated elements, which a
prelude-time `getEleTags` cannot see.

**Evidence still owed:** the static form has not run under OpenSeesMP. A
2-rank smoke deck emitted by the bridge, with a read-back of each target's
`dTime dTimeCommit dTimeInitial` (response 4000) after every stage and a
checker, is in `repros/adr0113_implex_mpi_smoke/`. Its serial half passed
locally, and so did a driver-off negative control. This ADR moves to
Accepted only after the 2-rank run passes.

### D4 — MPI and topology limits

- **OpenSeesSP is not supported, and not refused.** The bridge emits
  OpenSeesMP decks; under SP the parameters would bind master copies
  (context point 4). Whether a deck will run under SP cannot be known at
  emit, and the emitted Tcl has no reliable SP test, so this limit is
  **documented** (here, in the `ops.implex_time` docstring and in the
  changelog), not refused.
The other two limits are refused at emit:

- **Stage-activated targets are refused** in slice 1 (the parameters are
  built once, from the elements in the domain before the first stage).
  Follow-up: attach them inside their stage, after the stage's elements.
- **Removing a target is refused:** OpenSees cannot detach an object from a
  `Parameter`, so the next driver call would write through a dangling
  pointer. Removing a non-target element is fine.

### D5 — The adaptive transient stage (decided, not implemented)

`ops.strategy.AdaptiveTime(desired_iter, max_iter, max_factor=1.0,
min_factor=1e-6, max_factor_increment=1.5, min_factor_increment=1e-6)`,
passed as `s.run(n_increments=N, dt=dt, strategy=AdaptiveTime(...))` on a
`Transient` stage: duration `N * dt`, initial increment `dt`. The emitted
loop is STKO's `template_trans_rev.tcl` logic in apeSees names (the port's
`adaptive_loop_lines`, our own Tcl): clip the last increment to the
duration; before every attempt the before-attempt hooks
(`_apesees_implex_dt $dt [expr {$inc == 1}]`, D7's error control);
after a converged step one row `increment dt time iterations norm` in an
optional `step_log=` file on rank 0 (brace-quoted path; **opt-in**,
default `None`, so decks stay byte-stable unless a log is asked for) and
`factor *= min(max_factor_increment, desired / max(iters, 1))`, capped;
after a failure `factor *= max(min_factor_increment, desired / max_iter)`;
below `min_factor` the deck **errors** (the #587 fail-loud contract, never a
partial run). Decisions inside it:

- `max_iter` must equal the stage test's `max_iter` (the failure factor
  assumes the test gave up there; STKO writes the test with twice
  `desired_iter`). A mismatch is refused.
- `Ladder` + `AdaptiveTime` is refused in v1 (one owner of the retry);
  composition is a follow-up.
- STKO's `template_trans_rev.tcl` l. 17 reads
  `set min_factor_increment __min_factor__`, while `AnalysesCommand.py`
  substitutes a separate `__min_factor_incr__` placeholder (l. 215–217).
  So the exported loop appears to use `min_factor` as its minimum factor
  increment. This is read from the installed STKO 2026 template, not
  observed in a deck: the San Ramon decks set both to 1e-6, so they cannot
  tell. The spec keeps the two fields separate and documents the STKO
  reading.
- A static adaptive variant must re-issue `integrator LoadControl $dt`
  before each attempt (ADR 0104 D1's lesson); it comes after the transient.
- H5: refused until ADR 0057 Phase C.

**Excitation inside a stage.** `s.uniform_excitation(direction=, accel=series,
factor=...)` creates a stage-owned `UniformExcitation` (claimed like
`s.pattern`'s Plain, so the global pattern pass skips it), emitted after the
chain and before the recorders, outside any per-rank block (every rank needs
it). It is **removed at its stage's close** (`remove loadPattern $tag`)
instead of being frozen by `loadConst`: `EarthquakePattern::applyLoad`
stops advancing `currentTime` once `setLoadConst` ran
(`EarthquakePattern.cpp` l. 79–81, `LoadPattern.cpp` l. 433), so a frozen
excitation would keep applying its last ground acceleration as a constant
inertia load in every later stage. The staged validators that assume every
stage pattern is a `Plain` get a type dispatch.

### D6 — The dTime trap is refused at emit

Run on every emit path before anything is written
(`validate_implex_time`):

- `mode="stko"`: no `s.update_parameter` may write `dTime*` (one owner).
- `mode="off"`: no `s.update_parameter` may write `dTime*`.
- undeclared: from the first stage that writes any `dTime*`, that stage and
  every later one must write `dTime`, equal (to 1e-12 relative) to its own
  increment, on every element switched so far; a later stage whose increment is not one known number is
  refused. A partial write (only `dTimeCommit`) counts as a writer: it
  already switched the material off `ops_Dt`.

The translator's C13 reset (all three, equal to the stage increment, in every
static stage) passes; a translated deck with a later stage that does not
reset is now refused instead of running on a stale `dTime`. The check is by
parameter name, so any element parameter named `dTime*` is held to it;
nothing else in the bridge uses those names.

The check also tracks **which elements** each write reaches. An element is
switched off `ops_Dt` by any `dTime*` write to its `pg` or `elements`; the
`material=` form addresses elements too. Every later stage's `dTime` writes
must cover the union of switched elements so far, and the refusal names the
uncovered ones. Without this, a stage that rewrites `dTime` on the column
only would pass while the wall, switched earlier, steps on a stale value.
That was the review's counterexample, now a test.

If a deliberate `dTime` different from the increment is ever needed, add an
explicit `mode="manual"` rather than a waiver.

### D7 — Typed IMPL-EX options and the error control (decided, not implemented)

- `ASDConcrete3D` / `ASDConcrete1D` gain `implex_alpha: float = 1.0`
  (`-implexAlpha`, emitted only when ≠ 1, so existing decks stay
  byte-identical; refused without `implex=True`) and
  `implex_control: ImplexControl | None` (`-implexControl tol red`, the
  material-level check). STKO always writes `-implex -implexAlpha a` and has
  the material-level check commented out ("Don't use implexCheckError... a
  problem in OpenSeesMP when a material fails",
  `physical_properties/materials/nD/ASDConcrete3D.py` l. 11, 587–597), so
  `implex_control` gets a deck warning under partitioned emit.
- STKO's `ImplexAutoErrorControlActivate` becomes
  `ops.implex_time(..., error_control=ImplexErrorControl(tolerance=0.05,
  time_reduction_limit=0.01, error="max"|"average"))`: an after-attempt hook
  of D5's loop that reads `implexError` / `avgImplexError` through a
  parameter on one local target, reduces across ranks (STKO's `send`/`recv`
  ring), and when the error exceeds the tolerance and `dt >= limit *
  initial dt`, **reduces the next factor**. It does not reject the step
  already committed; that is what STKO's template does with
  `STKO_VAR_afterAnalyze_done` (`template_trans_rev.tcl`), and it is kept.
  One difference: `GlobalParameters` lives in an anonymous namespace in
  *each* of `ASDConcrete3DMaterial.cpp` (l. 307) and
  `ASDConcrete1DMaterial.cpp` (l. 100), so the 3-D and 1-D maxima are
  separate singletons. STKO reads the first target only and so misses the
  fibers when that target is a shell; the bridge reads one 3-D and one 1-D
  target per rank when both exist.
- The translator then maps `implexAlpha ≠ 1` and
  `ImplexAutoErrorControlActivate` before an analysis instead of refusing
  them (ADR 0111 D3), and emits its time-history stage on D5.

### D8 — Follow-ups and fork dependencies

Bridge gaps from the port (`FINDINGS.md`), not designed here:

1. `system Mumps` is refused on an unpartitioned deck (the port partitioned
   into 4 to get it).
2. Per-stage profiler files (`s.profile`, `<stage>.h5`) collide across MPI
   ranks: the report name needs the rank.
3. The Ladruno reader refuses to merge partitioned energy balances.
4. Ranks with no fixed nodes write empty reaction files that the multi-part
   reader rejects.
5. Stage-activated IMPL-EX targets (D4); a `live` read-back test of
   `dTime` per propagation chain (D2); H5 persistence of the driver and of
   D5 (ADR 0057 Phase C).

OpenSees fork work that would simplify D1–D3 (not apeGmsh work):

- **F1.** An ASDConcrete parameter that resets `dtime_n_commit` (the IMPL-EX
  ratio) **without** setting `dtime_is_user_defined`, so a stage start can
  restart the ratio while the material keeps following `ops_Dt`. With it,
  `mode="follow"` needs no per-attempt driver at all.
- **F2.** `ASDShellQ4 -drillingNL`: a per-component size floor in the
  drilling-damage computation ("Compute drilling damages",
  `ASDShellQ4.cpp`), which today turns round-off in components near zero
  into O(1) drilling damage; G20's driver-off run moved by O(1) in the
  shells for a 1e-16 change in `dt`. Every San Ramon shell uses it, so any
  round-off-level perturbation (rank count, solver, binary) can separate two
  runs by more than round-off.

## Decisions on the slice-1 open questions (2026-10-03)

- **Non-staged models.** `mode="stko"` stays refused on a flat deck. One
  analysis with one increment already gives ratio 1 through `ops_Dt`, so the
  driver would add nothing. `mode="off"` is allowed anywhere.
- **Step log (D5).** Opt-in (`step_log=None` by default).
- **Translator migration.** `build_conditions` keeps its C13 per-stage
  `s.update_parameter` reset, which passes D6, until three things land:
  D5, the in-stage excitation, and `reads_dtime` parity with C13's
  `_DTIME_TYPES` (ASDSteel1D is in; DamageTC1D/3D and ASDBondSlip are not
  typed). Then it moves to `ops.implex_time()`. The move is gated by a test
  that the translator's `implex_dt_targets` equal the bridge's targets on the
  1A–1D documents.

## PR breakdown

Every PR targets `main` (`gh pr create --base main`); none is stacked on
another PR's branch. Each one lands before the next is opened or rebased.

1. **This ADR and slice 1:** D1–D4 and D6, the review fixes, the skill docs,
   and the MPI smoke deck (`feat/implex-time-driver`). It merges only after
   the 2-rank smoke passes; the status line moves to Accepted in that PR.
2. **D5a, stage excitation:** `s.uniform_excitation` with removal at its
   stage's close.
3. **D5b, adaptive transient:** `ops.strategy.AdaptiveTime`, the emitted
   loop, the per-attempt IMPL-EX call and the opt-in step log.
4. **D7a, typed IMPL-EX options:** `implex_alpha` and `implex_control` on
   ASDConcrete3D/1D; the translator maps `implexAlpha`.
5. **D7b, error control:** `ImplexErrorControl` as an after-attempt hook;
   the translator maps `ImplexAutoErrorControlActivate`.
6. **Translator migration:** emit its time-history stage on 2–5 and replace
   the C13 resets with `ops.implex_time()`, gated by the target-set parity
   test.
7. **The D8 bridge gaps:** one PR each.

## Consequences

**Positive**

- STKO's IMPL-EX hook is a one-line declaration whose targets cannot drift
  from the materials, emitted the same way on serial and partitioned decks.
- The trap that the translator and the port documented in prose now fails
  at emit, naming the stage.
- The port's deck patches (`finalize_deck`) can be deleted once D5 lands:
  the prelude, the per-stage calls, the excitation and the loop are all
  bridge output.

**Negative / accepted**

- Slice 1 refuses every stage without one known increment, so the San Ramon
  transient (adaptive) still needs D5 before it leaves the patch.
- A model that drives cannot be archived to H5 until Phase C.
- The trap check is by name; a non-ASDConcrete parameter called `dTime`
  would be held to it.

## Tests (slice 1)

`tests/opensees/unit/test_implex_time_driver.py` covers:

- the modes;
- targets from the graph: the wall and column are in, the elastic slab is
  out, `eta` alone counts, and a steel-only ASDSteel1D column counts;
- the Tcl prelude and proc, line for line;
- one call per stage, immediately before the analyze loop and after `reset`;
- 20-id wrapping;
- both tag modes, `fem` and the default `sequential`;
- the py deck, serial and partitioned (compiled);
- the emit order;
- per-rank targets, with a global `first` call, on a partitioned stub;
- the H5 refusal and the live refusal;
- every D4 and D6 refusal, including the subset-coverage counterexample and
  the unknown-increment branch;
- a **golden replay**: the port's `prelude_lines` output and our prelude are executed in a
plain Tcl interpreter (`tkinter.Tcl`, no OpenSees) with `parameter` /
`addToParameter` / `updateParameter` stubbed to a log, and must issue the same
command sequence for the same driver calls, serially and per rank (our
`getPID` block vs their `getEleTags` intersection, where each rank's
`getEleTags` is read from the deck's own `element` lines in that rank's
blocks).

## Cross-references

- [ADR 0027](0027-cross-partition-mp-constraints.md) — per-rank ownership the targets use.
- [ADR 0051](0051-bridge-load-consumption.md) — stage-scoped patterns (D5's
  excitation).
- [ADR 0057](0057-solution-strategy-ladder.md) — deck-is-authoritative; Phase C.
- [ADR 0104](0104-substep-run-to-criterion-controller.md) — D6, closed here for
  the transient case.
- [ADR 0111](0111-stko-translator.md) — the translator's C13 reset and its
  refusals of `implexAlpha` and `ImplexAutoErrorControlActivate`.
