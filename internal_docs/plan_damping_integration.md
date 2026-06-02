# Plan — Damping definition on the apeSees bridge (ADR 0053)

Implementation plan for [ADR 0053](../src/apeGmsh/opensees/architecture/decisions/0053-damping-definition.md).
Establishes the `ops.damping` namespace across all four OpenSees damping
mechanisms. Closes the gap that today's `-doRayleigh` element flags have
**nothing to opt into** — there is no `rayleigh` command on the bridge.

**Guiding invariant:** damping is a *domain-level* concern (sibling of
`fix` / `mass` / `region`), never part of the analysis chain. The new
surface is one namespace `ops.damping`; the existing per-element
`do_rayleigh` flags are **untouched** — D1 only gives them a coefficient
source. The node-only `RegionAssignmentRecord` is **not overloaded**; D2
adds a parallel `-ele` membership route.

**Verify gotcha:** bare `python -c "import apeGmsh"` resolves the MAIN-repo
editable install (no worktree changes). Verify via `pytest`
(`pythonpath=["src"]` → worktree) using the `opensees_venv` python
(`C:\Users\nmora\venv\opensees_venv\Scripts\python.exe`). All work here is
deck-level (Tcl/py emit) — assert on captured deck lines, mirroring
`tests/opensees/unit/test_emitter_*` and `test_analysis.py`. No viewer, no
GPU.

**Source of truth for grammar:** ADR 0053 §"Source cross-check" — every
emitted line must match a cited `OpenSees_Compile` parser signature. Do not
invent flags.

## Scope (ADR 0053)

- **In:** `ops.damping` namespace; every member a lowercase declaration verb
  with a uniform `on=` scope (required/optional/forbidden per form, resolved
  at emit — **no `assign`, no user-held object**); global + region-scoped
  Rayleigh (raw + ratio helper + `stiffness=` switch); region `-ele`
  membership route; modal damping (`eigen`-bundled); four object-backed forms
  (`uniform`/`sec_stif`/`urd`/`urd_beta`) + universal `region -damp`;
  `on=`-list reuse (one object → many attachments); named-primitive alias;
  flat `ops.eigen` verb (D4) that `modal` requires + reuses.
- **Out / deferred:** staged damping (`s.damping.*`) → D5; element-flag
  `-damp` attach → D3b (secondary, capability-gated); `ops.damping.eigen`
  standalone → OQ-2; non-Rayleigh `do_rayleigh` flag plumbing on elements
  beyond the existing four primitives.

## Phases

### D1 — Global Rayleigh (`ops.damping.rayleigh`, raw + ratio) ✅ shippable alone
The minimal change that makes the existing `-doRayleigh` flags meaningful.
1. Add `_DampingNS(self)` → `self.damping` at `apesees.py:4039`-adjacent.
2. `ops.damping.rayleigh(...)` with `on=None` (global) only this slice:
   record a `RayleighRecord(alpha_m, beta_k, beta_k_init, beta_k_comm, on=None)`;
   resolve to `rayleigh αM βK βK0 βKc` (all four, post-build) on every emitter
   (`tcl` / `py` / `live` / `h5` / `recording`).
3. Ratio helper: when `ratio=`/`f_i=`/`f_j=` given (XOR with raw), compute
   α = 2ξω_iω_j/(ω_i+ω_j), β = 2ξ/(ω_i+ω_j) (ω = 2πf); place β by
   `stiffness=` switch — **default `initial`→beta_k_init** (fork: prefer βK0
   for nonlinear), `current→beta_k`, `committed→beta_k_comm`. Fail loud if
   both raw and ratio supplied, or ratio form missing any of ratio/f_i/f_j.
→ verify: deck contains exactly one `rayleigh` line with 4 numeric args;
raw form round-trips byte-for-byte; a known (ξ=0.05, f=1, f=10) case yields
the hand-computed α/β within tolerance and lands β in the slot the
`stiffness=` switch names; `pytest tests/opensees` green; a Truss with
`do_rayleigh=True` + a `rayleigh` command produces a runnable deck.

### D2 — Region-scoped Rayleigh (`on=`, `-ele` membership)
1. Add an element-membership region route: extend `RayleighRecord` with a
   resolved `on` target list; `on=` PG name(s) → `expand_pg_to_elements`
   fan-out (reuse the `s.remove_element` path); resolve to one
   `region $tag -ele e1 e2 … -rayleigh αM βK βK0 βKc` per target.
2. `on=` accepts a single PG/region name, a list of them, or explicit
   element/node tags. PG/element targets → fresh `-ele` region; a name that
   matches an existing region → attach `-rayleigh` to its command (no second
   region); node tags → `-node` region (+ warn-or-error per OQ-1 when β ≠ 0).
3. Region-tag allocation reuses the global + per-stage INV-4 convention.
4. **Overwrite order + guard** (fork-verified: element Rayleigh OVERWRITES,
   not additive): resolver emits the global `rayleigh` line **before** any
   region-rayleigh, so "region refines global"; emit a warning when an `on=`
   target overlaps elements already covered by a global `rayleigh`.
→ verify: `on="x"` emits one `region $tag -ele … -rayleigh …` with the body's
element tags; a list `on=["a","b"]` emits one region-rayleigh per target;
re-using an existing region name attaches to the same tag (no duplicate);
node target + non-zero β triggers the OQ-1 **hard error**; global emitted
before region; overlap triggers the overwrite warning; global vs region
paths don't collide on tags; deck runs.

### D3 — Object-backed forms (`ops.damping.uniform/sec_stif/urd/urd_beta`)
Lowercase declaration verbs (per §1) that resolve to a tagged object **plus**
its `region -damp` attachment. `on=` is **required** (no global `-damp`).
1. New `Damping(Primitive)` frozen subclass per type; registered on
   `_primitives` for topological tag allocation (like `section`); optional
   `name=` via the named-primitive alias channel.
2. Resolve to `damping Uniform $tag ζ f1 f2 [flags]` / `SecStif $tag β` /
   `URD $tag N f1 z1 … [flags]` / `URDbeta $tag N fc1 b1 …`. `uniform`'s
   `ratio=` passes the **physical ζ** verbatim (OpenSees doubles internally —
   document, do not pre-divide). URD/urd_beta enforce N ≥ 2.
2a. **Time-window kwargs** (all four types): `activate_time` →
   `-activateTime`, `deactivate_time` → `-deactivateTime`, `factor`
   (an `ops.timeSeries.*` object, added as a dependency so its line emits
   first) → **`-factor`** (NOT `-fact`: Tcl accepts both but openseespy only
   `-factor`, `OpenSeesDampingCommands.cpp:71`). These window *when the object
   dissipates energy* — the "no damping during gravity" lever.
3. For each target in `on=`, emit `region $tag -ele … -damp $obj.tag`. A list
   `on=[...]` → ONE `damping` object reused across N region-damp lines (the
   reuse path); fail loud if `on=` is absent.
4. H5: persist Damping primitives under a new `/opensees/...` sub-tree;
   **bump bridge `SCHEMA_VERSION`**; `OpenSeesModel._replay_into` replays
   them. Reader accepts own-minor + one-below per ADR 0023.
→ verify: each type resolves to its cited grammar; the `damping` line is
dependency-ordered before its region `-damp`; `on=[a,b]` yields one object +
two region-damp lines sharing the tag; absent `on=` fails loud; named alias
resolves; `activate_time`/`deactivate_time` emit the time flags and `factor=`
emits **`-factor`** (asserted in BOTH tcl and py decks; `-fact` never
emitted) with the timeSeries line ordered first; H5 round-trip reconstructs
object + attachments + time-window; schema-version guard rejects an over-new
minor.

### D3b — Element-flag `-damp` attach (secondary, fixed allow-list)
Optional `damp=` kwarg on the **known supporting** element primitives only —
fork-verified allow-list: `ElasticBeam2d/3d`, `DispBeamColumn`,
`ForceBeamColumn`, `Brick`, `FourNodeQuad`, the Shell family, `ZeroLength`
(every other element's base `setDamping` just warns). Emit `element … -damp
$tag`. Fail loud on any primitive not on the allow-list.
→ verify: a supported element emits `-damp`; an unsupported one raises a
clear capability error naming the element type; no silent drop.

### D4 — `ops.eigen` (flat) + modal damping (`ops.damping.modal`)
1. New flat verb `ops.eigen(n, *, solver="genBandArpack")` → `eigen <solver>
   N` (sibling of `ops.fix`/`ops.mass`, NOT in `ops.damping`; general-purpose
   modal analysis). Records an `EigenRecord(n, solver)`.
2. `ops.damping.modal(ratios)` → `modalDamping f1 [f2 …]`. **Requires** a
   prior `ops.eigen(...)` → fail loud if absent. Scalar `ratios` → uniform;
   sequence → per-mode, `len == eigen.n` or fail loud. `on=` **forbidden** →
   fail loud if given (domain-wide).
3. **No `modal_q`** — `modalDampingQ` is a fork-verified anti-damping bug
   (wrong sign, Δt-independent; "use modalDamping, never modalDampingQ").
   The bridge does not author it (OQ-3 resolved).
4. Ordering: resolver emits `eigen` then `modalDamping`, after the model is
   built and before any `analyze`; fail loud if no transient analysis follows.
→ verify: `ops.eigen(5)` emits `eigen 5`; `modal` without a prior eigen fails
loud; deck orders `eigen N` before `modalDamping`; uniform vs per-mode forms;
length mismatch vs `eigen.n` fails loud; `on=` rejected; runs on a small
modal model; **no `modalDampingQ` is ever emitted.**

### D5 — Staged damping (`s.damping.*`)
Mirror `s.region` / `s.fix`: stage-bound `s.damping.rayleigh(..., on=)` /
`s.damping.uniform(..., on=)` pools resolving inside the owning stage's
block (same lowercase verbs + `on=` as the flat path). Per-stage modal
damping (eigen + `wipeAnalysis` interaction) gets its own design note before
code.
→ verify: stage-bound Rayleigh emits inside the stage block (not globally);
two stages with same-named damping regions get distinct tags (V3); global +
stage damping coexist; staged deck runs.

## Resolved questions (all closed — nothing blocks D1)

- **OQ-1** `on=` node tags + non-zero β → **HARD ERROR** (fork-confirmed:
  nodes carry αM only, β is meaningless).
- **OQ-2** → **flat `ops.eigen`** added (outside `ops.damping`); `modal`
  requires + reuses it. No `ops.damping.eigen`.
- **OQ-3** → **dropped `modal_q`** (`modalDampingQ` anti-damping bug, fork
  document-only decision).
- **OQ-4** ratio-helper `stiffness=` default → **`initial`** (βK0), per the
  fork's nonlinear guidance; `current`/`committed` explicit opt-ins.

## Docs / skill reconciliation (after D1–D4 land)

- `internal_docs/guide_opensees.md` — add a Damping section.
- `skills/apegmsh/references/opensees-bridge.md` — currently has **zero**
  damping mention; add `ops.damping.*` once shipped (sync via `sync_skill.py`
  to the `apegmsh-helper` mirror).
- `CHANGELOG.md` Unreleased — one entry per shipped slice.
