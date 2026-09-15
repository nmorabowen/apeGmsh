# apeGmsh ← Ladruno: adoption guide for the 2026-09-14/15 fork batch

Five fork work packages landed as pull requests on 2026-09-14/15. This guide
says, per feature: **what changed in the fork**, the **exact tokens / response
names / widths / return codes**, **what apeGmsh must change and where** (with
`file:line` into this repo), a **verification recipe**, and a **"nothing to
do"** list.

It is the apeGmsh-side sibling of the fork's own two batch guides —
`Ladruno_implementation/ladruno_apegmsh_adoption_guide_2026-09-07.md` (PR #823)
and `..._2026-09-08.md` (PR #828) — written here rather than there because
every adoption point below is an apeGmsh file.

> ### STATUS — all five fork PRs are OPEN, none merged
>
> Every token, response width and return code below was read from the PR
> **diffs and tests at these heads**, not from a merged `ladruno` tip. Do not
> ship an emitter change that depends on them until the PR merges and
> `ops.ladrunoBuild()` is a descendant of the merge commit.
>
> | WP | PR | head at writing | state | what it is |
> |---|---|---|---|---|
> | WP-99 / F7 | [#838](https://github.com/nmorabowen/OpenSees/pull/838) | `75140bf2e` | **DRAFT** | IMPL-EX commit-time refusal latch; `implexRefusals` 4 → 6; `analyze()` → `-4` |
> | WP-100 / F8 | [#836](https://github.com/nmorabowen/OpenSees/pull/836) | `b4fb89d64` | open | ASD Drucker-Prager apex classification under dilatant flow |
> | WP-101 / F9 | [#839](https://github.com/nmorabowen/OpenSees/pull/839) | `a1a072e77` | open | `LadrunoKinematicCoupling -alUpdate`; K_t band + conditioning guard |
> | WP-102 / F10 | [#837](https://github.com/nmorabowen/OpenSees/pull/837) | `6f79e7943` | open | **diagnosis only** — the IMPL-EX self-weight wall |
> | WP-103 | [#840](https://github.com/nmorabowen/OpenSees/pull/840) | `90a2bc599` | open | `OPS_GetStringFromAll` never filled the buffer under classic Tcl |
>
> Base branch for all five: `ladruno`, cut from `9c2f964ea`.

**Two of the five change nothing apeGmsh emits and everything apeGmsh reads.**
F7 widens a response apeGmsh already parses by name and adds a new `analyze()`
failure code apeGmsh's substep ladder will currently retry forever. WP-103 fixes
a bug that silently discarded `-k` / `-dof` / `-host` on **every Tcl deck
apeGmsh has ever emitted**. Read §1 and §5 first.

---

## 1. F7 — the IMPL-EX commit-time refusal latch (#838)

### 1.1 What changed in the fork

Until WP-99 a `LadrunoSANISAND` companion that hit `-maxSubsteps` **at
`commitState`** could not refuse: `Domain::commit()` is `elePtr->commitState();`
with the return value dropped, for every element, fork or vanilla. The material
committed the partially integrated state and the analysis walked on reporting
every step converged — measured on the TIMs plane-strain strip (9 720 Gauss
points) as **25.9 M capped commits and a straight-line load–settlement curve to
2 674 kPa**, all steps "converged".

Two mechanisms now stop it, and review round 1 measured why both are needed:

1. **The commit is aborted, element-independently.** The material declares the
   refusal out of band (`ladrunoNoteCommitRefusal()`, a header-only counter in
   `SRC/material/LadrunoMaterialStatus.h`); `Domain::commit()` checks it after
   the element loop, prints one line, and **returns `-1` before
   `committedTime`/`dT` and before the recorder loop**.
2. **The instance latches.** Trial restored from the committed state,
   `ManzariDafalias::commitState()` skipped, and every later `setTrialStrain`
   on that integration point returns `LADRUNO_MATERIAL_REFUSED` (`-33086`).
   Sticky; cleared **only** by `revertToStart()`, never by
   `revertToLastCommit()`. Shipped on the wire (`Vector(33) → Vector(34)`).

Mechanism 2 alone was measured insufficient: two stacked `stdBrick` (a DISCARD
element) with `algorithm Linear` ran **20 further accepted steps with
`analyze() == 0`**, the refusing element frozen as a rigid inclusion.

### 1.2 The exact return-code chain — `analyze()` returns `-4`

| step | file | result |
|---|---|---|
| `Domain::commit()` | `SRC/domain/domain/Domain.cpp` (after the element loop) | `-1` |
| `AnalysisModel::commitDomain()` | `:656-659` | `-2` |
| `IncrementalIntegrator::commit()` | `:254` | `-2` |
| `StaticAnalysis::analyze()` | `:213-222` | revert, **`-4`** |
| `DirectIntegrationAnalysis::analyze()` | `:259-267` | revert, **`-4`** |
| `VariableTimeStepDirectIntegrationAnalysis::analyze()` | `:137-140` | `result = -4`, revert-and-subdivide — **the retry is refused too** |

The fork's own evidence run:

```
Domain::commit() - 8 integration point(s) REFUSED this commit ...
WARNING: AnalysisModel::commitDomain - Domain::commit() failed
StaticAnalysis::analyze() - the Integrator failed to commit at step: 0 ...
OpenSees > analyze failed, returned: -4 error flag
step  rc   implexRefusals [total, d2, ctl, companion, commitLatched, latched]
  0   -4   [8, 0, 0, 8, 1,   8]
  1   -4   [8, 0, 0, 8, 1,  32]
  ...
  7   -4   [8, 0, 0, 8, 1, 176]
```

**What `-4` means for a driver, precisely.** *That* Gauss point committed
nothing — but `Domain::commit()` walks **nodes first, then elements**, and every
integration point is its own material object, so by the time one refuses the
nodes and its sibling points **have already committed** (fork-measured
`[1,0,0,1]` latched across one `LadrunoQuad`). The model state at that commit is
**inconsistent, not merely un-advanced**. A `-4` is therefore **not a
subdivision signal**: the latch is sticky, every later step also returns `-4`,
and the run must be restarted from the last good checkpoint. The recoverable
alternative is `-implexControl`, which catches the same cap one phase earlier at
the **trial**, where the step it belongs to fails and a retry at a smaller
increment is meaningful.

### 1.3 `implexRefusals` widened 4 → 6

| slot | name | meaning |
|---|---|---|
| 0 | `implexRefusals_total` | total **GENUINE** refusals = `[1]+[2]+[3]` |
| 1 | `implexRefusals_signChange` | D2 sign change |
| 2 | `implexRefusals_control` | `-implexControl` past tolerance |
| 3 | `implexRefusals_companion` | companion hit `-maxSubsteps` |
| 4 | `implexRefusals_commitLatched` | **this integration point's** commit-refusal latch, 0/1 — the **only per-instance slot** |
| 5 | `implexRefusals_latched` | POST-latch refusals, process-wide; fires **once per Newton iteration per point** and is deliberately **not** folded into `[0]` |

All six are emitted as real `ResponseType` component names (`XML` header and
`.ladruno` `COMP_NAMES`). Slots 0–3 keep their meaning exactly; the change that
bites is the **width**. Python contract in the fork guide:

```python
refusals = ops.eleResponse(eleTag, "material", intPtNum, "implexRefusals")
n_total, n_signchange, n_control, n_companion, commit_latched, n_latched = refusals
```

Also new: a banner line `IMPL-EX commit-time refusal latch — no silent partial
commits`, and a 10-per-process warning budget naming each latched tag
(`WARNING LadrunoSANISAND tag <t>: REFUSING every further update (-33086). ...`).
The `-implexControl` trial-time refusal is **unchanged**.

### 1.4 What apeGmsh must adopt

**(a) The `implexRefusals` reader is hard-coded 4 wide — in two tables.**

- `src/apeGmsh/results/readers/_ladruno_element_io.py:313-316` —
  `_MATERIAL_BUCKET_TOKENS["implexrefusals"]`, the **by-position** canonical
  names (`implex_refusals_total`, `..._sign_change`, `..._control`,
  `..._companion`).
- `src/apeGmsh/results/readers/_ladruno_element_io.py:401-404` —
  `_MATERIAL_BUCKET_EXPECTED_NAMES["implexrefusals"]`, the wire-spelling check
  (`implexRefusals_total`, `implexRefusals_signChange`, `implexRefusals_control`,
  `implexRefusals_companion`).

The failure mode is **silent data loss, not a warning**. `_block_canonicals`
(`_ladruno_element_io.py:759-771`) resolves a material bucket by position **only
when `len(names) == len(b.comp_names)`**; a 6-column block against a 4-name
table falls through to `continuum_canonical(name)` on the labels, which returns
`None` for every `implexRefusals_*` spelling — so the whole bucket lands in
`dropped` and **disappears from `gauss_available` / the returned fields**.
`_name_mismatch` (`:774-798`) also returns `None` on a width mismatch, so there
is not even a mismatch warning.

Fix: extend **both** tuples to six, in slot order, mirroring the
`implexdetail` entries just above them. Add
`implex_refusals_commit_latched` and `implex_refusals_latched` to the
by-position table and `implexRefusals_commitLatched` /
`implexRefusals_latched` to the expected-names table.

Then update the pinned assertion at
`tests/test_ladruno_stress_zz_ready.py:188`, which asserts the 4-tuple for
`material_bucket_canonicals("material.implexRefusals")`.

`src/apeGmsh/opensees/recorder.py:361` lists `implexRefusals` in
`_MATERIAL_ONLY_ELEM_TOKENS` — **no change**, the token spelling is unchanged and
the recorder passes `-E` tokens verbatim.

> **Slot 4 is per-instance, slots 0/1/2/3/5 are process-wide.** Any aggregation
> apeGmsh does over Gauss points must not sum slot 0 (it is already a
> process-wide ledger, identical at every GP), and must not treat slot 5 as a
> refusal count — it fires once per Newton iteration per point (fork-measured
> 244 post-latch against 4 genuine cap hits before the split). Slot 4 is the
> only one it is meaningful to reduce over GPs (`max`, or a count of latched
> points).

**(b) `analyze()` returning `-4` must abort, not subdivide.**

Two families of call site, and they behave differently.

**Already correct — the staged drivers abort on any non-zero rc:**
`src/apeGmsh/opensees/apesees.py:2968` and `:4738` (`if rc != 0: raise
BridgeError(...)`), the emitted Python deck
(`src/apeGmsh/opensees/emitter/py.py`, `if ops.analyze(1) != 0: raise
SystemExit(...)`) and the emitted Tcl deck
(`src/apeGmsh/opensees/emitter/tcl.py`, `if {[analyze 1] != 0} { error ... }`).
A `-4` stops these the way it should; they just will not *say* why.

**Needs the new code — everything that retries, escalates or flattens:**

- `src/apeGmsh/opensees/analysis/strategy.py:443` — `if driver.analyze(ds_use)
  != 0:` → `nsub += 1; ds *= 0.5` and keep going until the subdivision budget or
  the `ds_min` floor. Against a latched material this burns the whole budget on
  a step that can never succeed and then reports `budget` / `floor` as if it
  were a mechanical verdict.
- `src/apeGmsh/opensees/analysis/strategy.py:198` —
  `OpenSeesPyDriver.analyze` returns the raw `int` (this is the seam where a
  `-4` can be recognised).
- `src/apeGmsh/opensees/emitter/live.py:853-890` — `LiveEmitter.analyze`;
  the hook-wrapped loop breaks on the first non-zero `r` and returns it
  (`:884-889`), and the un-hooked path returns `ops.analyze(steps)` verbatim
  (`:866-870`). Both already **propagate** `-4` — nothing swallows it.
- `src/apeGmsh/opensees/emitter/live.py:891-...` — `_analyze_ladder`, the rung
  walk: escalates rungs on a failed `analyze(1)` and returns the failing rc on
  exhaustion. A `-4` will walk every rung first, pointlessly.
- `src/apeGmsh/interop/solve.py:149` — `return o.analyze(1) == 0`, which
  collapses `-4` to `False` with no distinction from ordinary non-convergence.
- `src/apeGmsh/sections/_mc.py` — the moment-curvature driver
  (`if ops.analyze(1) != 0: raise MomentCurvatureError(...)`, then a
  `complete = False; break` path); and `src/apeGmsh/sensitivity/driver.py`,
  which **discards** the rc entirely.
- `src/apeGmsh/opensees/apesees.py` — the two `analyze(steps=1,
  dt=_DTCR_PRIME_DT)` explicit **prime** calls (near `:10582` and `:10656`)
  discard their rc, and the flat live path returns
  `int(live_emitter.analyze(...))` to the caller unchecked.

**No site anywhere in `src/` distinguishes a negative rc from a positive one
today** — every check is `!= 0` / `== 0`. That was harmless while every rc meant
"this step did not converge"; `-4` is the first rc that means "the model is
inconsistent".

Recommended shape (guide only — **not implemented here**): one shared predicate,
e.g. `LADRUNO_ABORT_RCS = (-4,)` plus the existing material sentinel `-33086`,
consulted by the substep controller and the rung ladder **before** subdividing
or escalating, raising a named error that says *the model state is inconsistent;
restart from the last good checkpoint*. Both `-4` and `-33086` are
non-retryable; every other non-zero rc keeps today's behaviour.

> **`-33086` is not new** and is already the documented retry signal in the
> 2026-09-07 fork guide §5 ("a refused step returns `analyze` rc `= -33086`
> … and is the retry-with-smaller-step signal"). That remains true for the
> **trial-time** refusal under `-implexControl`. What F7 adds is a *second*,
> **non**-retryable code, `-4`, for the commit-time one. If apeGmsh implements
> only one of the two, implement `-4`.

**(c) The refusal-propagation roster apeGmsh encodes is too narrow, and the
gate built on it is now partly obsolete.**

F7's second half is an **audit**: the fork's four shipped `opserr` texts claimed
the refusal is acted on "today `LadrunoBrick`" and listed `QuadUP` among the
discarders — both halves wrong. The full audit is now done, every `.cpp` under
`SRC/element` mentioning both `setTrialStrain` and `NDMaterial`, classified by
what its own `update()` does with the code:

| class | count |
|---|---|
| FORWARD (a nonzero reaches the return of `update()`) | **26** |
| SENTINEL-only (`LadrunoBrick`, per ADR-33/34) | **1** |
| DISCARD | **25** |
| total NDMaterial hosts | **52** |

The **single authoritative copy** is the table **"Element refusal roster"** in
the fork's `Ladruno_implementation/LEDGER_quirks.md`; every `opserr` string and
guide list is now explicitly non-exhaustive and points there (a second closed
copy is exactly how the previous wrong one survived in four documents at once).
Named forwarders include `LadrunoBrick20`, `LadrunoQuad`/`CST`/`LST`,
`BezierTet10`/`Tri6`, `TenNodeTetrahedron`, `LadrunoUP`, `Twenty_Node_Brick`,
`ConstantPressureVolumeQuad`, `Tri31`, `NineNodeQuad`, `EightNodeQuad`,
`SixNodeTri`, and **the whole vanilla u-p family** (`FourNodeQuadUP`,
`BBarFourNodeQuadUP`, `Nine_Four_Node_QuadUP`, `Twenty_Eight_Node_BrickUP`).
Named discarders include `Brick` (= `stdBrick`), `BbarBrick`, `BrickUP`,
`BBarBrickUP`, `SSPquad`/`SSPquadUP`, `SSPbrick`/`SSPbrickUP`,
`LadrunoSolidShell`, `FourNodeTetrahedron`, `EnhancedQuad`, `NineNodeMixedQuad`.

apeGmsh encodes this in two places, both built on the **old, wrong** claim:

- `src/apeGmsh/opensees/_element_capabilities.py` — `_ElemSpec.
  propagates_material_refusal` (`:133`) is set on **exactly three** elements:
  `TenNodeTetrahedron` `True` (`:305`), `stdBrick` `False` (`:315`),
  `LadrunoBrick` `True` (`:346`). Every other element is `None` = "unmeasured",
  and the ADR 0105 D4 gate never warns on `None`
  (`element_propagates_material_refusal`, `:793-803`).
- `src/apeGmsh/opensees/_internal/build.py:4069` —
  `_REFUSAL_PROPAGATING_ELEMENTS = frozenset({"LadrunoBrick"})`, the ALLOW-list
  behind `validate_sanisand_substep_cap` (`:4095-4127`), which **raises**
  `BridgeError` when a `max_substeps`-capped `LadrunoSANISAND` reaches anything
  else. Its message says *"Only LadrunoBrick propagates on every path today"*.

Two consequences:

1. **The ALLOW-list is too narrow and internally inconsistent.** It excludes
   `TenNodeTetrahedron`, which apeGmsh's *own* capability table marks `True`
   (`_element_capabilities.py:305`) — so a capped SANISAND on a tet is refused
   at build time for a reason apeGmsh itself does not believe. With the fork's
   roster in hand, populate `propagates_material_refusal` from it and key the
   gate on `element_propagates_material_refusal(...) is False` (the way
   `validate_asdplastic_host` at `build.py:4145-4182` already does) instead of
   on a one-element ALLOW-list.
2. **The gate's premise is now half-answered by the fork.** The failure it
   guards — *"a partially integrated stress accepted as converged"* — is the
   **commit-time** case, and since F7 `Domain::commit()` aborts that
   **element-independently**: a discarding element can no longer swallow it. The
   **trial-time** refusal (`-implexControl`, `-33086` out of `setTrialStrain`)
   still needs a forwarding element, and `validate_asdplastic_host`'s warning
   (`ASDPlasticHostWarning`, `build.py:4129-4143` — *"`strict_convergence` … is
   therefore inert on such a host"*) is entirely about the trial path and stays
   exactly as true as before. Split the wording accordingly rather than dropping
   either gate: **commit-time refusals are now everyone's; trial-time refusals
   are still the element's.**
   Keep `LadrunoBrick`'s special case visible while you are there — it is
   **SENTINEL-only**, the one element that filters for exactly
   `LADRUNO_MATERIAL_REFUSED` per ADR-33/34, not a plain forwarder.

**(d) Recorder output is not written for the aborted commit.** The early return
in `Domain::commit()` skips the recorder loop and the `commitTag` bump
deliberately, so an aborted step leaves **no trace** in the `.ladruno` file. A
reader that infers "the run reached step N" from the recorder's row count will
under-count by exactly the aborted step. State it wherever apeGmsh documents
step/row correspondence.

### 1.5 Verification recipe

Mirror the fork's own gate, `tests/test_ladrunoQuad_sanisand_implex_commit_refusal.py`
(5 cases) and `tests/test_ladruno_sanisand_implex.py`:

- `test_implexrefusals_carries_the_commitlatched_slot` — assert
  `len(ops.eleResponse(1, 'material', 1, 'implexRefusals')) == 6` and that
  `[0] != [1]+[2]+[3]+[5]` (slot 0 is genuine-only).
- `test_quad_commit_time_companion_refusal_latches_and_stops_the_run` and
  `test_brick_commit_time_companion_refusal_latches_and_stops_the_run` — starve
  a deck with a low `-maxSubsteps` and assert `ops.analyze(1) == -4` **and that
  every subsequent step also returns `-4`**.
- `test_discarding_element_still_stops_the_run_via_domain_commit` — the same on
  `stdBrick`, the canonical DISCARD element: this is the case the latch alone
  could not stop.
- `test_latch_is_cleared_only_by_reverttostart` — `revertToLastCommit()` does
  **not** clear it.
- `test_implexcontrol_refuses_the_same_cap_at_trial_and_does_not_latch` — the
  recoverable path; slot 4 stays `0`.

On the apeGmsh side the cheap gate is a reader unit test: hand
`_ladruno_element_io` a synthetic 6-column `implexRefusals` block with the six
`COMP_NAMES` and assert the six canonical names come back — today the bucket is
dropped instead.

### 1.6 Nothing to do

- No emitter change. No new token, no changed token, no changed default.
- `-implexControl` semantics unchanged.
- Slots 0–3 keep their meaning and spelling; existing readers of `[3]`
  (the companion bucket — see §4) stay correct.
- `implexDetail` (6) and `implexGuards` (7) are untouched by this WP.

---

## 2. F8 — ASD Drucker-Prager apex classification under dilatant flow (#836)

### 2.1 What changed in the fork

`wp/94f` (#832) had fixed the zero-dilatancy apex misclassification by
**unioning** ADR-97's elastic-metric apex test with the yield function's
Euclidean `check_apex_region`. A union keeps the **wider** region, and the exact
slope `K·etabar/G` overtakes the Euclidean `eta` as soon as

```
etabar > eta·G/K
```

On the ADR-95 deck (`G/K = 0.10345`, `eta = 0.4457`) the crossover is
`etabar = 0.0461` — **psi ≈ 2.3°**. So the union was wrong for essentially any
**dilatant** deck, not only for associated flow:

| leg | Euclidean slope `eta` | exact slope `K·etabar/G` | union keeps | verdict |
|---|---|---|---|---|
| psi = 0 | 0.4457 | 0 | the exact one | correct |
| psi ≈ 2.3° | 0.4457 | 0.4457 | either | the crossover |
| psi ≈ phi/2 | 0.4457 | 2.1545 | the Euclidean one | **4.8× too wide** |
| psi = phi (associated) | 0.4457 | 4.3089 | the Euclidean one | **≈10× too wide** |

Every trial in the wedge `eta·q ≤ p − p_apex < (K·etabar/G)·q` was
**apex-projected although its correct return is to the cone flank** — committing
`sigma_apex` with **no deviator** and, under `tangent_type Continuum`, a **zero
tangent** — and **no refusal was issued**. The failure is silent and presents as
"the element walls while still hardening".

**The fix.** For yield functions declaring the `yf_apex_elastic_metric` trait
(**Drucker-Prager only**) the elastic-metric test **replaces** the Euclidean one.
At `etabar = 0` the exact region strictly contains the Euclidean one, so
wp/94f's zero-dilatancy result is untouched **by construction**. Every other
yield function still calls `check_apex_region`.

Round 1 added two more changes inside the same trait scope:

- a **deviator-flip guard after the flank Newton**: narrowing the region routes
  near-boundary trials into the flank map, whose `dPhi/dlambda` carries the
  pinned vanilla `df/dk = −1` cohesion term that `f` does not contain; with
  cohesion **softening** that Newton *converges* (`rc = 0`, `|f| ~ 1e-7`, no
  exhaustion) onto a **sign-flipped deviator**. The guard rejects
  `dot(r_ret, r_tr) < 0`, measured on the **relative** deviator
  `r = dev(sigma) − alpha` (round 2 caught a first version testing the
  *absolute* deviator, which refused an admissible return when `alpha` was
  antiparallel to the trial deviator), with a `||r_ret|| > tol_yf` floor so it
  does not fire on round-off at a return landing exactly on the vertex.
- `cp_apex_region()` now measures its flip on the **relative** deviator
  `s − alpha` for the opted-in family (it used the raw deviator before).

**Pinned, not fixed:** `DruckerPrager_YF::apex_stress()` still ignores the back
stress `alpha` entirely (recorded in the fork's `LEDGER_quirks`; identical
before and after F8).

### 2.2 Measured effect

Same mesh, same deck, ADR-95 R3 gate, 200 `LadrunoBrick -formulation bbar`,
`system Pardiso`, exact `q_u = 138.907 kPa`:

| leg | build | `q_max` | ratio | mode | failed/subdiv | **relaxed** | wall s |
|---|---|---|---|---|---|---|---|
| UW associated (reference) | `9c2f964ea` | 268.75 | 1.9348 | TARGET | 0 / 0 | 0 / 329 | 63 |
| ASD associated **PRE-fix** | `9c2f964ea` | 226.51 | 1.6307 | BUDGET | 898 / 81 | 256 / 560 | 894 |
| ASD associated **POST-fix** | `3324485f7` | 268.38 | 1.9321 | BUDGET | 1528 / 81 | **638 / 687** | 1122 |
| ASD psi = 0 PRE and POST | both | 150.71 | 1.0850 | TARGET | 0 / 0 | 0 / 329 | 72 / 107 |

The two implementations now follow the **same path**, not merely the same peak:
worst deviation **0.075 %** over the whole common range.

> **Read the `relaxed` column before quoting the agreement.** The ASD associated
> leg needed the push ladder's **third rung** (`KrylovNewton` at 10× the
> `NormUnbalance` tolerance, 60 iterations) on **638 of its 687 converged steps
> (92.9 %)**, where UW associated and both psi = 0 legs needed **zero**. Part of
> the asymmetry is deck, not material (the ASD decks carry
> `strict_convergence 1` / `n_max_iterations 100`; vanilla `DruckerPrager` has
> no equivalent switch, so a step ASD *refuses* is one UW silently accepts).
> **Nothing measured explains why the ASD path costs more Newton work on the
> same cone.**

The psi = 0 leg is **byte-identical across the fix** (329 rows, 0 differing).

### 2.3 What apeGmsh must adopt

**Nothing in the emitters.** No token, no parameter, no default moves. The two
Drucker-Prager emit sites stay exactly as they are:

- `src/apeGmsh/opensees/material/nd.py:1234` —
  `"DruckerPrager_YF": frozenset({"DP_xi_c", "DP_eta"})`
- `src/apeGmsh/opensees/material/nd.py:1242` —
  `"DruckerPrager_PF": frozenset({"DP_etabar"})`

and the vanilla `DruckerPrager` class (`nd.py:227-390`) is not touched by this
WP at all (F8 is `ASDPlasticMaterial3D` only).

What apeGmsh **must record**, because it changes what a user should believe
about existing results:

1. **A minimum-build note for dilatant ASD Drucker-Prager.** `nd.py` already
   carries this pattern: `DP_ADR95_MIN_FORK_BUILD = "61b3efa04"` (`nd.py:223`),
   `ASDP_MIN_FORK_BUILD = "bbf657d49"` (`nd.py:1355`),
   `ASDP_CLOSEST_POINT_MIN_BUILD = "7e93e4381"` (`nd.py:1364`),
   `SANISAND_IMPLEX_FACTOR_MIN_BUILD = "179da6ffb"` (`nd.py:1378`) — all
   documented, not enforced, because a bare hash cannot prove ancestry. Add the
   same kind of constant for the F8 merge commit once #836 lands, and cite it in
   the `DruckerPrager_PF` docstring: **`DP_etabar > DP_eta·G/K` on a build
   before it silently apex-projects a wedge of trials.**
2. **Any stored ASD Drucker-Prager result with `DP_etabar > DP_eta·G/K` from a
   build before the fix is suspect** — not because the load path looks wrong
   (pre-fix it was within 5.17 % worst / 0.141 % at its terminal point) but
   because the *iteration* is destroyed: an apex projection is a different,
   non-smooth map with a zero `Continuum` tangent at exactly the Gauss points the
   mechanism forms around. The symptom is a controller death, not a wrong
   number. **"The load path looks fine" is not evidence that the return map is.**
3. Cross-link from the existing `internal_docs/guide_ladruno_adr95_druckerprager_fix.md`
   and `internal_docs/guide_ladruno_asdp_closest_point.md`, and from ADR
   `src/apeGmsh/opensees/architecture/decisions/0105-asdplastic-deck-contract-after-ladruno-adr94.md`,
   which is where the apeGmsh-side ASD deck contract lives.
4. If apeGmsh's harness surfaces a "relaxed rung" count (it has the machinery —
   `strategy.py`'s rung ladder records `strategy_events`), **report it next to
   any ASD-vs-UW agreement figure**, per the fork's own discipline above.

### 2.4 Verification recipe

Fork gates to mirror, all on `ASDPlasticMaterial3D`:

- `tests/test_f8_asd_dp_associated_apex.py` — **12/12** in 0.71 s, `zone_a`; it
  is **1 failed / 3 passed on `9c2f964ea`**, so it genuinely gates the fix. The
  discriminating assertion is at one Gauss point, associated, ADR-95 cone
  (`eta = 0.445749`, `xi_c = 0.115470`, `p_apex = 0.2590471`), trial at
  `(p − p_apex)/q = 2.0`: pre-fix commits `(p, q) = (0.2590471, 0)`, post-fix the
  closed-form cone return `(−0.5314874, 0.3523800)`. Closed form:
  `dgamma = f_tr/(G + K·eta·etabar)`, `q_ret = q_tr − G·dgamma`,
  `p_ret = p_tr − K·etabar·dgamma`.
- `tests/test_adr94f_asd_apex_fallback.py` (4/4) — wp/94f's own cases unchanged.
- `tests/test_adr97_p4_inertness.py` (10/10) — `Backward_Euler` still
  byte-identical on all 23 baseline decks.
- `tests/test_r3_prandtl_asd_associated.py` — the slow tier (`--runslow`, ~20 min),
  all three legs in-session.

On the apeGmsh side: re-run any regression deck whose `DP_etabar` exceeds
`DP_eta·G/K` and diff the curve; a `psi = 0` deck must be **bit-identical**.

### 2.5 Nothing to do

- Emitters, parameter names, defaults: unchanged.
- Yield functions other than Drucker-Prager: byte-identical (the trait is
  opt-in and only `DruckerPrager_YF` declares it).
- `psi = 0` (`DP_etabar = 0`) decks: byte-identical — no golden file moves.
- `Closest_Point` / `tangent_type Algorithmic` (ADR-97): untouched.

---

## 3. F9 — `LadrunoKinematicCoupling -alUpdate` and the K_t rule (#839)

### 3.1 The new token

```
element LadrunoKinematicCoupling $tag $refNode $N $s1..$sN [-dof c1..cK]
        [-k {Kt|auto}] [-kAlpha a] [-host eleTag] [-kr Kr]
        [-enforce {penalty|al}] [-alUpdate {commit|iter}]
        [-bipenalty {-dtcr dt | -wcap beta}] [-absolute]
```

- Token spelling is **exact and case-sensitive**: `-alUpdate`. Values are
  **exactly** `commit` or `iter` (no aliases, no case folding —
  `OPS_LadrunoKinematicCoupling.cpp`, the `strcmp(mode, "iter")` /
  `strcmp(mode, "commit")` pair). Anything else: `WARNING
  LadrunoKinematicCoupling: unknown -alUpdate '<x>' (want iter|commit)` and the
  element is **not created** (the parser returns `0`).
- **Default is `commit`** — the pre-existing cadence. A deck that never emits
  `-alUpdate` is byte-identical to one built before the token existed.
- `-alUpdate` **without** `-enforce al` is a warning and is ignored:
  `": -alUpdate has no effect without -enforce al (the penalty formulation
  carries no multiplier); ignored"`, and `alUpdate` is forced back to `0`.
- `-alUpdate iter` emits a **parse-time echo** of its validity window, and is
  **refused at the first `update()`** unless the active algorithm is **full
  Newton** and the active static integrator is **`LoadControl`** (read via
  `OPS_GetAlgorithm` / `OPS_GetStaticIntegrator` / `OPS_GetTransientIntegrator`).
  On refusal `update()` returns `-1` and the analysis fails loudly. The guard is
  **silent while no analysis exists** (a null algorithm pointer means "no
  opinion"), because `Domain::addElement` calls `update()` at declaration time.

Measured on this PR's own 2×2×2 elastic fixture, **before** the refusal guard —
this is why `iter` is not the default (cells are **failed steps**):

| algorithm | `LoadControl` (1 step) | `DisplacementControl` (5 steps) |
|---|---|---|
| Newton | 0 | **5** |
| ModifiedNewton | **1** | **5** |
| KrylovNewton | **1** | **5** |
| BFGS | **1** | **5** |
| Broyden | **1** | **5** |

`update()` advances `λ` **before** the force is formed, so the tie force carries
`λ_k + 2·D·g(u_k)` against a tangent that linearises a single `D·g` — the
residual is **not a function of `u`** and every secant / accelerated /
re-solving method is fed `(δu, δr)` pairs describing no Jacobian.

### 3.2 The within-step route that works — the ADR-41 augmentation sweep

**No new hook, no new virtual, no registry.** `Domain::commit()` runs every
element's `commitState()` during an ADR-41 held-load sweep
(`Domain::contactAugmenting` only suppresses the recorder loop and the
`commitTag` bump), and with `-alUpdate commit` that `commitState` **is** the
outer Uzawa update `λ += D g`. Each held-load `analyze()` is an inner solve at
**fixed `λ`**, so the residual is well posed and every algorithm works.

```python
ops.element('LadrunoKinematicCoupling', 1, ref, n, *skin,
            '-dof', 1, 2, 3, '-k', 1.0e6, '-enforce', 'al')   # -alUpdate commit = default

ops.integrator('LoadControl', 1.0)
ops.analyze(1)                                   # the real step (any algorithm/integrator)

ops.ladrunoBeginAugment()                        # recorders + commitTag frozen
try:
    ops.integrator('LoadControl', 0.0)           # HOLD the load, whatever drove the step
    for _ in range(10):
        ops.analyze(1)                           # inner solve at FIXED lambda ...
        if ops.eleResponse(1, 'constraintViolation')[0] < tol:
            break                                # ... then commitState does lambda += D g
finally:
    ops.ladrunoEndAugment()
```

Measured at `K_t = 1e6` (penalty floor `1.63e-2`), **20/20 cells converge**:

| driving integrator | algorithms | passes | final `max|g|/push` |
|---|---|---|---|
| `LoadControl` | Newton, ModifiedNewton, KrylovNewton, BFGS, Broyden | 5 | `6.14e-11` |
| `DisplacementControl` | Newton, ModifiedNewton, KrylovNewton, BFGS, Broyden | 4 | `9.53e-10` |

Three traps the fork measured and that any apeGmsh helper **must** encode:

1. **`try/finally` is mandatory.** While the flag is on, `Domain::commit()`
   fires **no recorders** and bumps **no commitTag**. A forgotten
   `ladrunoEndAugment` therefore **does not fail** — the next ordinary step
   returns `ok = 0`, time advances `1 → 2`, and **every recorder sample from
   then on is silently lost** (fork-measured: recorder file **empty**). Since
   WP-101 a second `ladrunoBeginAugment` without an intervening `End`
   **warns**, and the flag is cleared by `Domain::clearAll()` (`wipe`) and by
   `wipeAnalysis()` — but **inside one analysis nothing else will tell you**.
2. **Use `LoadControl 0.0` for the held passes whatever drove the real step.**
   A zero-increment `DisplacementControl` is degenerate: vanilla's
   `dLambda = −dUabar/dUahat` is unbounded at a zero increment
   (`DisplacementControl.cpp:332`; fork-measured load factor `1.0 → 377.19` in
   one step on their fixture, `−1.6e38` on the reviewer's, **and the step often
   still returns `ok = 0`**).
3. **The held passes hold the LOAD, not the control DOF.** `LoadControl 0.0`
   freezes the load factor, not a `DisplacementControl` target — so under a
   displacement-driven step the control DOF is **released** and drifts
   (fork-measured **1.08 %** of the push while the gap fell `1.63e-2 → 9.5e-10`).
   Physically right; read the displacement back **after** `ladrunoEndAugment`
   rather than assuming the target.

### 3.3 `K_t`: the selection band and the conditioning warning

Rigidity error is **exactly** `err = c/K_t` — measured `err·K_t = 1.663e4`,
constant to five digits over six decades:

| `K_t` | `1e6` | `1e7` | `1e8` | `5e9` | `1e12` |
|---|---|---|---|---|---|
| `err = max\|g\|/push` | `1.63e-2` | `1.66e-3` | `1.66e-4` | `3.33e-6` | `1.66e-8` |

Against that, conditioning degrades linearly in `K_t`. On the **TIMs strip,
101 583 DOF**: at `1e12` **Pardiso limped on perturbed pivots and then died and
SuperLU failed its first factorisation**; at `5e9` the same leg ran to a clean
plateau (`6.7e-7`). The TIMs `1e12 → 5.7e-8` figure is already **off the `1/K`
line, on the round-off floor**.

**Recommended band: `1e2 … 1e4 × k_host`.** A once-only warning now fires from
`resolveAutoKt()` (not the parser — it needs `host->getInitialStiff()`, and at
parse time the host may not be in the domain nor `setDomain`'d) when a **numeric
`-k`** is given **together with a named `-host`** above `1e6 × k_host`:

```
WARNING LadrunoKinematicCoupling 999: -k 1e+12 is 1.26061e+08x the -host element's stiffness
scale (7932.69). Above ~1e6x the penalty block dominates the global matrix and the solve goes
ill-conditioned (perturbed pivots / failed factorisation) while the rigidity error stops
improving. Recommended band: 1e2..1e4x the host diagonal; use -enforce al to tighten the tie
at a MODERATE K_t instead. See LadrunoKinematicCoupling_guide.md section 3.1
```

It fires **with `-host` regardless of declaration order**: `resolveAutoKt()` no
longer latches `ktResolved` on a failed host lookup, so a coupling declared
*before* its host is retried (previously `-k auto` silently stayed at `1e12`
instead of `7.93e6` and the warning could never fire).

**`-k auto -host` cannot hold a rigid footing.** It is a **conditioning**
control (`K_u = a·max_i|K_host(i,i)|`, default `a = 1e3`), pinning `K_t` to the
host's own order — exactly the regime where the residual gap is largest. TIMs
measured `1.4e-4`; the Zone-A gate reproduces the mechanism at
`K_t ≈ 7.9e6 → 2.1e-3`. Pinned by `test_k_auto_host_cannot_hold_a_rigid_footing`.

### 3.4 Other behaviour changes in #839

- **`revertToLastCommit` rolls `λ` back.** It was a bare `return 0`; `λ` is now
  snapshotted at `commitState` (`lambdaCommitted`) and restored, so a
  failed/retried step no longer inherits the multipliers of a discarded trial.
- **Wire format: header `Vector(20) → Vector(21)`, version `hdr(19) = 2`**, with
  `hdr(20) = alUpdate` and `lambdaCommitted` in the payload (`3·nGap`). A
  **newer** version is now refused rather than mis-read. Relevant only if
  apeGmsh saves/restores an `FE_Datastore`.
- **Explicit-integrator warning, now correctly gated.** `-enforce al` is
  refused-by-consequence under an **explicit** integrator (`-bipenalty` is
  dropped when `-enforce al` is given, leaving a massless tied DOF with no mass
  source; measured under `CentralDifferenceLadruno`: `penalty -bipenalty` → `0`,
  `penalty` alone → `-2`, `al -bipenalty` → `-2`). The warning is now gated on an
  **enumerated explicit class-tag set** read from `SRC/classTags.h`
  (`CentralDifference` 5, `CentralDifferenceAlternative` 17,
  `CentralDifferenceNoDamping` 18, `ExplicitDifference` 55, `ExplicitBathe`
  33000, `ExplicitDifferenceStatic` 33001, `ExplicitBatheLNVD` 33002,
  `CentralDifferenceLadruno` 33003, `CentralDifferenceSMS` 33007,
  `CentralDifferenceSMSConsistent` 33008, `ExplicitBatheSMS` 33009–33012); an
  **unrecognised** transient tag is treated as implicit. **Implicit transient is
  fine** — `Newmark 0.5 0.25` + Newton + `-enforce al` is measured 0/10 failed,
  `maxGap = 1.510e-08`, and emits **no** warning. The first cut warned there too
  and was wrong.
- **Class tag unchanged** (`ELE_TAG 33012`); no new element.

### 3.5 What apeGmsh must adopt

**(a) A `-alUpdate` field on the coupling control.** The emitter is
`CouplingControl.emit_flags` at
`src/apeGmsh/_kernel/_coupling_control.py:158-194` — an order-independent flag
tail, defaults elided. `-enforce` is emitted at `:186-187`
(`if self.enforce != "penalty": out += ["-enforce", self.enforce]`). Add an
`al_update: str | None = None` field to the dataclass (`:19-...`, the field
block near `enforce`, documented in the flag table at `:31-57`) and emit it
immediately after `-enforce`:

```python
if self.al_update is not None:
    out += ["-alUpdate", self.al_update]
```

with `__post_init__` validation mirroring the fork's two rules: value in
`("commit", "iter")`, and `al_update is not None` requires `enforce == "al"`
(the fork warns-and-ignores; apeGmsh should refuse, per its existing habit of
raising rather than emitting a flag the fork will drop). `None` omits the token
and inherits the fork default `commit` — byte-identical to today's decks.

The emit site that consumes this tail for RBE2 is
`src/apeGmsh/opensees/_internal/build.py:7113-7161`
(`emitter.element("LadrunoKinematicCoupling", ele_tag, *args)` at `:7160`, with
`-dof` at `:7157-7158` and the control flags at `:7159`). **No change needed
there** — the tail is opaque to it.

> **Scope check.** `CouplingControl` is shared with
> `LadrunoDistributingCoupling` (RBE3) and, via `EmbeddedNodeControl`, with
> `LadrunoEmbeddedNode`. `-alUpdate` is a **`LadrunoKinematicCoupling`-only**
> token in #839. Either gate the field on the RBE2 emit path, or accept that a
> user setting it on an RBE3 record produces a flag the RBE3 parser will reject.
> Prefer the gate.

**(b) An augment-sweep helper.** There is none today
(`grep -rn "ladrunoBeginAugment" src` → no hits). The `try/finally` shape and
the `LoadControl 0.0` rule in §3.2 are exactly the kind of thing that belongs in
a context manager on the live emitter, e.g.

```python
with live.augment(element=tag, tol=1e-8, max_passes=10):
    ...
```

wrapping `ops.ladrunoBeginAugment()` / `ops.ladrunoEndAugment()` in `finally`,
re-issuing `integrator('LoadControl', 0.0)` itself, polling
`eleResponse(tag, 'constraintViolation')[0]`, and **restoring the caller's
integrator afterwards**. It must also refuse to nest (the fork warns on a double
`Begin`; a context manager can make that unreachable).

Gate it behind the existing build check — `live.py:416-430` already reads
`ops.ladrunoBuild` and has the "stock openseespy, or a fork build predating
`ladrunoBuild`" path; `live.py:149` already lists `LadrunoKinematicCoupling` in
the fork-only element set, and `live.py:388` is the existing
`APEGMSH_OPENSEES_BIN` guidance string.

**(c) A K_t sizing note, not a new helper.** apeGmsh already has the analogue:
`AUTO_STIFFNESS_ALPHA = 1.0e3` (`src/apeGmsh/opensees/_internal/build.py:7167`)
and `make_auto_stiffness_resolver` (`:7170-...`), which computes
`K = AUTO_STIFFNESS_ALPHA · E_host · L_char` because *"apeGmsh never assembles
stiffness matrices, so this deliberately estimates the host diagonal scale from
material + geometry rather than reading `max|K(i,i)|` the way the fork's C++
`k="auto"` does"*. `α = 1e3` sits **inside** the fork's newly measured
`1e2…1e4 × k_host` band, so **no value changes**. What to add is the docstring
line: this estimator targets **conditioning**, is **not** a rigidity setting, and
**cannot hold a rigid footing** — for that, use `-enforce al` at a moderate `K_t`
plus the §3.2 sweep. Also worth a note that a hand-set `k` above
`1e6 × k_host` will now draw a fork warning whenever a `-host` is named.

**(d) There is no reader for the coupling's `lambda` / `constraintViolation`
responses.** `src/apeGmsh/opensees/_response_catalog.py` carries no
`LadrunoKinematicCoupling` entry at all, and nothing in `src/` reads a
`lambda` / `lambdaCommitted` element response. The augment-sweep helper in (b)
needs `eleResponse(tag, 'constraintViolation')[0]` as its loop condition, so it
is the first consumer — either call `ops.eleResponse` directly from the helper
(simplest, live-only) or add a catalog entry if the sweep's `λ` history should
reach `Results`. `lambdaCommitted` is new in #839 and is **wire-only** (the
`FE_Datastore` payload); there is no response token for it.

**(e) `-dof` on the coupling is already emitted** (`build.py:7157-7158`) and the
2026-09-07 fork guide item 3 already required it for u-p slaves. **No change.**

### 3.6 Verification recipe

Fork gate: `tests/test_ladrunoKinematicCoupling_element.py`, **55 passed** (19
pre-existing + 36 new). The cases worth mirroring on the apeGmsh side:

- `test_al_update_commit_reproduces_penalty_within_a_step` — the default pin;
  an apeGmsh deck with `enforce="al"` and no `al_update` must be byte-identical
  to today's.
- `test_augment_sweep_closes_constraint` (10 cells) and
  `test_enforcement_sweep` (12 cells) — the sweep helper's acceptance.
- `test_iter_cadence_refused` / `test_iter_echo_at_parse` — `al_update="iter"`
  outside full-Newton + `LoadControl` must fail loudly, never silently succeed.
- `test_high_kt_against_host_warns` — a numeric `k` above `1e6 × k_host` with a
  `host` set emits the §3.3 warning.
- `test_al_under_newmark_is_silent_and_works` — implicit transient + `al` must
  emit **no** warning.
- `test_double_begin_augment_warns` / `test_wipe_analysis_clears_augment_flag` —
  the helper's nesting refusal and the recorder-file-non-empty assertion.
- `test_k_auto_host_cannot_hold_a_rigid_footing` — the documentation claim.

apeGmsh-side cheap gates: an emission test asserting `-alUpdate` lands in the
flag tail exactly once and only with `-enforce al`; and a `ValueError` test for
`al_update="iter"` with `enforce="penalty"`.

### 3.7 Nothing to do

- Default behaviour: unchanged. Decks that never set `al_update` are
  byte-identical.
- `ELE_TAG 33012` unchanged; no class-tag work.
- `-dof`, `-k`, `-kAlpha`, `-host`, `-kr`, `-enforce`, `-bipenalty`,
  `-absolute`: spelling and semantics unchanged.
- `AUTO_STIFFNESS_ALPHA = 1e3` needs **no numeric change** — it is inside the
  measured band.
- Finite-rotation transport is unchanged (the fork's §6.x scope note stands): the
  rigid-footing driver is still geometrically **linear**.

---

## 4. F10 — the IMPL-EX self-weight wall: a diagnosis, not a feature (#837)

**No C++, no banner, no flag default moved.** Engine: the pinned release build
of `ladruno` tip `9c2f964`, which **predates #838** — which matters for the
recommendation.

### 4.1 The finding

On a plane-strain strip footing on **self-weight** SANISAND under
`-implex -implexControl`, the run walls on the step floor at `s/B = 0.0085` with
**724 refusals, all in the `control` bucket**. Three sentences:

1. **`-implexControl` is what stops this deck.** Removing it — bare `-implex`,
   same doubling controller, everything else identical (**leg N**) — reaches
   `s/B = 0.0500` in **104 steps, 0 subdivisions, 0 failed attempts, 58 s**, with
   the refusal ledger at `0/0/0/0` and **`implexRefusals[3]` (companion)
   explicitly `0`**.
2. **With the control on, the refusal COUNT is set by the harness's growth
   rule** (×2 → 724 refusals, ×1.25 → 253, ×1.0 → 6), because `implexError` is
   first order in the step while `-implexControl` bounds it absolutely. A
   co-factor of the control, not an independent cause.
3. **The FLOOR seizure itself is the control's own `implexPrimed` bare `> 0.0`
   test** (`LadrunoSANISAND.cpp:2905`): Gauss points whose committed plastic
   history is `1e-12 … 1e-21` forfeit the un-primed exemption and are refused on
   an error that **does not decay with `dt`** (0.2243 at `|dt| = 4e-5` → 0.2143
   at `2e-5` — a halved step moves the error 4.5 %).

`-implexFactor controlIter` is **not** the tool here (+5 % reach for 2.9× wall).
`reductionLimit = 0.5` is "turn the control off after one halving", and at the
shipped `0.01` the material floor `reductionLimit·|dt0| = 2e-7` m **is** the
harness `DS_MIN` exactly, which is why it never fires. `-implexGuard off` (+85 %)
removes the P2-2 protection ADR-93's seat needs — recorded, not recommended.

### 4.2 What apeGmsh must adopt — harness guidance, not code

1. **Run bare `-implex` and watch `implexRefusals[3]`.** For a self-weight deck,
   run without `-implexControl`, read the **companion** bucket at the end of
   every leg, and **do not call the result a capacity**. On a build predating
   #838 a capped companion commit is otherwise silent; once #838 lands the run
   aborts with `-4` instead (§1). The bucket was `0` on legs B, C, D, E, K, L, N
   and N1 and `≤ 42` anywhere in the campaign.
   - In apeGmsh terms: emit `implex=True, implex_control=None`
     (`src/apeGmsh/opensees/material/nd.py:990-992`), record
     `material.implexRefusals` (`src/apeGmsh/opensees/recorder.py:361`), and
     assert `implex_refusals_companion == 0` at the end of the leg. **This is
     the concrete reason §1.4(a) matters**: with the 4-wide table the whole
     bucket is dropped and that assertion cannot be written.
   - **And it collides with the substep-cap gate.** The companion bucket only
     ever climbs under a `-maxSubsteps` cap, and
     `validate_sanisand_substep_cap` (`src/apeGmsh/opensees/_internal/build.py:4095-4127`)
     **raises** on a capped SANISAND under anything but `LadrunoBrick`. The
     F10 recipe on a `LadrunoQuad` / u-p / tet deck is therefore unreachable
     from apeGmsh today — see checklist items 9a/9b.
2. **Scope the claim honestly.** Leg N is a **termination** result. It says
   nothing about ADR-92 §8's **accuracy** claim, which the engine itself prints
   on every control-off run (`LadrunoSANISAND.cpp:2029-2031`): *"P0 measured
   IMPL-EX unusable from `d_eps = 5e-4` at `p0 = 5 kPa`, so at a low-confinement
   corner the control is a requirement, not an option"*. **This deck sits inside
   that range** — minimum `p'` **6.374 kPa = 1.27× the corner**, leg N's strain
   increment crosses `5e-4` at `s/B = 0.0012` and runs at **2.6–4× the corner**
   to the target, with **no implicit anchor past `s/B = 0.00227`**. No general
   "the control is not a low-confinement requirement" is claimed anywhere, and
   apeGmsh must not imply one in a docstring.
3. **Pin the growth factor if the control is wanted.** `×1.0` (0.0265, 6
   refusals) or `tol = 0.5` (reaches the target, 18 refusals, agrees with the
   control-off arm to **0.395 % mean / 1.625 % max** over `0.002 ≤ s/B ≤ 0.05`)
   — i.e. **`tol 0.5` makes the control nearly inert on this deck**. `×1.25` is
   not enough. apeGmsh's growth/subdivision policy lives in
   `src/apeGmsh/opensees/analysis/strategy.py` (the `ds *= 0.5` subdivision at
   `:443`, the `ds_min` floor and the `budget`); a `growth` knob pinned to 1.0
   is the apeGmsh-side expression of this.
4. **The `relaxed` column is part of the result.** Same discipline as §2.2: a
   leg's rung-escalation count belongs next to its reach. `strategy.py` already
   records `strategy_events`; surface the count.
5. **Concurrent-writer guard.** The fork campaign tore a results CSV by
   launching the same leg twice and had to add an `F10_FORCE=1` single-process
   guard. Any apeGmsh harness that writes one CSV per leg needs the same
   (a lock file, or a refusal when the output exists).

### 4.3 Verification recipe

There is no new fork test — this WP shipped a note
(`Ladruno_implementation/92b_implex_selfweight_wall_note.md`), a testbed
(`Ladruno_files/testbed/hypo_bearing/adr92_f10/`), six `LEDGER_quirks` entries
and an ADR-92 log entry. The apeGmsh-side check is the leg-N shape: a
self-weight SANISAND push with `implex=True, implex_control=None` that
**terminates at the target with `implex_refusals_companion == 0`**, recorded and
asserted rather than eyeballed.

### 4.4 Nothing to do

- No token, no default, no response change.
- `-implexFactor` (ADR 92 P2-9, already adopted at `nd.py:992` with
  `SANISAND_IMPLEX_FACTOR_MIN_BUILD = "179da6ffb"` at `nd.py:1378`) needs no
  change; `controlIter` simply is not the tool for this deck.
- `-implexGuard`, `-implexControl` parser and semantics: unchanged.

---

## 5. WP-103 — `OPS_GetStringFromAll` never filled the buffer under classic Tcl (#840)

### 5.1 What was broken

`SRC/api/elementAPI_TCL.cpp:493` was:

```cpp
extern "C" const char* OPS_GetStringFromAll(char *buffer, int len)
{ return OPS_GetString(); }          // `buffer` is never touched
```

`elementAPI.h:209` documents the function as *"does a strcpy"*, and the
**openseespy** backend (`PythonModule::getStringFromAll`) does exactly that. The
classic-Tcl backend did not — so every caller written in the family idiom

```cpp
char tok[64];
OPS_GetStringFromAll(tok, sizeof(tok));
if (strcmp(tok, "auto") == 0) ...
```

read **uninitialised stack** under `OpenSees.exe` / SP / MP, and the option was
**silently lost**. **31 call sites**, including
`OPS_Ladruno{DistributingCoupling,EmbeddedNode,EmbeddedRebar,KinematicCoupling,UP}.cpp`
and `Parallel3DMaterial` / `Series3DMaterial`.

Observed pre-fix (`9c2f964`), with **nondeterministic** garbage tokens across runs
(`''`, `'0����'`, `'����o'`, `' ����'`, `'��no'`):

```
WARNING LadrunoKinematicCoupling: -k wants a number or 'auto', got '0����'
WARNING LadrunoKinematicCoupling: -dof needs at least one component
WARNING LadrunoEmbeddedNode: nHost must be >= 1 (or use -host eleTag); got '��no'
```

Three symptoms, not one:

- **`-k <number>` lost** — the numeric branch never reached.
- **`-dof` broken a second way** — its greedy reader `strtol`s the same
  uninitialised buffer, so the component list came out **empty** and the element
  was **refused outright**.
- **`LadrunoEmbeddedNode` / `LadrunoEmbeddedRebar` explicit-host form unusable**
  — the token choosing between `-host <eleTag>` and `<nHost> h1..hN` goes
  through the buffer, and the explicit branch `atoi`s garbage.
- **`-k auto` never matched** — the sentinel the idiom exists to detect. Post-fix
  it reaches the real requirement: *"-k auto requires a representative -host
  element"*.

Post-fix, the same deck: `print -ele` shows **`K_t: 1e+06`** (the `-k 1.0e6` that
used to be lost) and **`nGap: 6`** for `-dof 1 2 3` against **`12`** for the
default 6-component list.

### 5.2 The fix, and why the return value did not change

```cpp
const char* res = OPS_GetString();   // Everything's a string in Tcl
if (buffer == 0 || len <= 0) return res;
if (res == 0) { buffer[0] = '\0'; return 0; }
strncpy(buffer, res, (size_t)(len - 1));
buffer[len - 1] = '\0';
return buffer;
```

**Out-of-args still returns `0`** — exactly what `OPS_GetString()` returns in
that translation unit — so no existing null-checking caller changes behaviour;
the buffer is emptied so a caller that ignores the return reads `""` rather than
garbage. `SRC/runtime/parsing/InterpreterAPI.cpp:140` carries the identical
defect and was deliberately **not** touched (`SRC/runtime` is never built in this
fork).

### 5.3 What apeGmsh must adopt

**The openseespy path was never affected** — `PythonModule::getStringFromAll`
always did the strcpy. So every apeGmsh **live** run and every **openseespy deck**
apeGmsh emits (`src/apeGmsh/opensees/emitter/py.py`,
`src/apeGmsh/opensees/emitter/live.py`) needs **nothing**.

**apeGmsh does have a Tcl emitter**, and it is affected:
`src/apeGmsh/opensees/emitter/tcl.py` (`TclEmitter.element` at `:770`,
`.nDMaterial` at `:724`), selected via `target="tcl"` at
`src/apeGmsh/opensees/apesees.py:10995-11003` and instantiated at `:10791`.
`LadrunoKinematicCoupling` has **no dedicated Tcl method** — it goes through the
generic `TclEmitter.element(...)`, so the deck carries whatever `build.py`
composes: the coupling flag tail from `_coupling_control.py:158-194` and the
`-dof` list at `build.py:7157-7158`. The two elements #840 names explicitly
**do** have dedicated methods and are affected the same way:
`TclEmitter.embedded_rebar` (`tcl.py:633-642`, emitting `element
LadrunoEmbeddedRebar`) and `TclEmitter.embedded_node` (`tcl.py:659-665`,
emitting `element LadrunoEmbeddedNode`) — both pass args pre-built by the
bridge, including the `-host` / `nHost h1..hN` choice that #840 restores.

**Consequence, stated plainly: every Tcl deck apeGmsh has emitted with a numeric
`k`, a `dofs` list, or an explicit `host` on the coupling / embedded-node /
embedded-rebar family silently ran with those options DISCARDED on
`OpenSees.exe` / SP / MP.** A `-k 1e6` deck ran at the fork default `1e12`; a
`-dof 1 2 3` deck was refused outright or tied all six components.

Actions:

1. **No emitter code change** — the tokens are already correct. What changes is
   the **minimum fork build for the Tcl target**. Add a documented constant in
   the same style as `nd.py:223 / :1355 / :1364 / :1378` (e.g.
   `TCL_COUPLING_TOKENS_MIN_BUILD`) pointing at the #840 merge commit, and cite
   it from `tcl.py`'s module docstring and from the coupling docstrings in
   `_coupling_control.py`.
2. **Re-run, do not re-read, any Tcl-target validation** that used `k`, `dofs` or
   `host` on the coupling family. Results are not merely imprecise — the
   constraint layout differed (`nGap: 6` vs `12`).
3. If apeGmsh's `doctor` (`src/apeGmsh/doctor.py:653-712`, which already resolves
   `APEGMSH_OPENSEES_BIN` and reports on the fork binary) is extended, a
   one-line Tcl probe of `-k 1.0e6` + `print -ele` is a cheap, decisive check —
   this is exactly what the fork's own gate does.

### 5.4 Verification recipe

The fork's gate is `tests/test_wp103_getstringfromall_tcl.py` driving
`tests/tcl/wp103_getstringfromall.tcl` through `OpenSees.exe`, following the
`tests/test_ladruno_solver_queries_tcl.py` (#729) shell-out precedent. It is
proven **fail-before / pass-after** by pointing it at each binary through
`LADRUNO_TCL_EXE` — the pre-fix binary *is* the mutant (`assert 5 >= 15` on
`9c2f964`, `1 passed` on the fix). The decisive assertions are `K_t: 1e+06` in
`print -ele` and `nGap: 6` vs `12`.

apeGmsh-side: emit a small Tcl deck through `TclEmitter` with
`CouplingControl(k=1.0e6, kr=..., enforce="al")` and `dofs=[1,2,3]`, run it on
the fork's `OpenSees.exe`, and assert the same two readbacks.

### 5.5 Nothing to do

- openseespy emitters, the live runner, the `APEGMSH_OPENSEES_BIN` seam: nothing.
- Token spellings: nothing changed; the tokens were always right, the fork just
  could not read them.
- `-enforce`, `-absolute`, `-bipenalty`: these do **not** go through
  `OPS_GetStringFromAll` on the affected branches and were never lost.

---

## 6. Consolidated adoption checklist

Tick these on the apeGmsh side. **"Code"** = an apeGmsh source change is
required; **"Doc"** = a docstring / ADR / guide line; **"None"** = verified
nothing to do.

| # | § | Item | Where | Kind |
|---|---|---|---|---|
| 1 | 1.4(a) | Widen `_MATERIAL_BUCKET_TOKENS["implexrefusals"]` 4 → 6 (`implex_refusals_commit_latched`, `implex_refusals_latched`) | `src/apeGmsh/results/readers/_ladruno_element_io.py:313-316` | **Code** |
| 2 | 1.4(a) | Widen `_MATERIAL_BUCKET_EXPECTED_NAMES["implexrefusals"]` 4 → 6 (`implexRefusals_commitLatched`, `implexRefusals_latched`) | `.../_ladruno_element_io.py:401-404` | **Code** |
| 3 | 1.4(a) | Update the pinned 4-tuple assertion | `tests/test_ladruno_stress_zz_ready.py:188` | **Code** |
| 4 | 1.4(a) | Reader unit test: a 6-column `implexRefusals` block resolves (today it is silently **dropped**, not warned) | `tests/results/readers/test_ladruno_reader.py` | **Code** |
| 5 | 1.4(a) | Document that slot 4 is the **only** per-instance slot and slot 5 is per-Newton-iteration, so neither is summable like slots 1–3 | `_ladruno_element_io.py` docstring | Doc |
| 6 | 1.4(b) | Treat `analyze()` rc `-4` as **non-retryable abort**: stop subdividing, stop escalating rungs, raise "restart from the last good checkpoint" | `src/apeGmsh/opensees/analysis/strategy.py:443`; `.../strategy.py:198`; `src/apeGmsh/opensees/emitter/live.py:891-...` (`_analyze_ladder`) | **Code** |
| 7 | 1.4(b) | Same for `src/apeGmsh/interop/solve.py:149`, which collapses every rc to a bool | `interop/solve.py:149` | **Code** |
| 8 | 1.4(b) | Keep `-33086` as the **retryable** trial-time signal; only `-4` is the abort | wherever (6) lands | Doc |
| 9a | 1.4(c) | Populate `_ElemSpec.propagates_material_refusal` from the fork's 52-element roster (only 3 elements carry a value today; the rest are `None` = never warned about) | `src/apeGmsh/opensees/_element_capabilities.py:133`, `:305`, `:315`, `:346` | **Code** |
| 9b | 1.4(c) | Retire the one-element ALLOW-list; key `validate_sanisand_substep_cap` on `element_propagates_material_refusal(...) is False`, as `validate_asdplastic_host` already does. It currently refuses `TenNodeTetrahedron`, which apeGmsh's own table marks `True` | `src/apeGmsh/opensees/_internal/build.py:4069`, `:4095-4127` | **Code** |
| 9c | 1.4(c) | Re-word both gates: **commit**-time refusals are now element-independent (fork `Domain::commit()`); **trial**-time refusals still need a forwarding element | `build.py:4095-4127`, `:4129-4182` | Doc |
| 9d | 1.4(d) | Note that an aborted commit writes **no** recorder row and bumps no `commitTag` | recorder / results docs | Doc |
| 10 | 2.3 | Add an F8 minimum-build constant + cite it from the `DruckerPrager_PF` docs: `DP_etabar > DP_eta·G/K` is unsafe on older builds | `src/apeGmsh/opensees/material/nd.py:1242` (+ a constant beside `nd.py:1355-1378`) | Doc |
| 11 | 2.3 | Cross-link F8 from the existing ASD docs / ADR 0105 | `internal_docs/guide_ladruno_adr95_druckerprager_fix.md`, `internal_docs/guide_ladruno_asdp_closest_point.md`, `src/apeGmsh/opensees/architecture/decisions/0105-*.md` | Doc |
| 12 | 2.3 | Report the **relaxed-rung count** next to any ASD-vs-UW agreement figure | `src/apeGmsh/opensees/analysis/strategy.py` (`strategy_events`) | Doc |
| 13 | 3.5(a) | Add `al_update: Literal["commit","iter"] \| None = None`, emit `["-alUpdate", v]` after `-enforce`, validate `iter`/`commit` and require `enforce == "al"` | `src/apeGmsh/_kernel/_coupling_control.py:158-194` (emit) and the field/flag table at `:19-57` | **Code** |
| 14 | 3.5(a) | Gate the new field to the **RBE2** path — `-alUpdate` is `LadrunoKinematicCoupling`-only, but `CouplingControl` is shared with RBE3 / `EmbeddedNodeControl` | `_coupling_control.py`, `build.py:7113-7161` | **Code** |
| 15 | 3.5(b) | Augment-sweep context manager: `ladrunoBeginAugment` / `LoadControl 0.0` / poll `constraintViolation` / `finally: ladrunoEndAugment`, restore the caller's integrator, refuse nesting | `src/apeGmsh/opensees/emitter/live.py` (beside `:149` fork-only set, gated by `:416-430`) | **Code** |
| 16 | 3.5(c) | Docstring: `AUTO_STIFFNESS_ALPHA = 1e3` is a **conditioning** control inside the fork's measured `1e2…1e4 × k_host` band, **not** a rigidity setting; it cannot hold a rigid footing | `src/apeGmsh/opensees/_internal/build.py:7163-7190` | Doc |
| 17 | 3.5(d) | The sweep helper is the first consumer of `eleResponse(tag, 'constraintViolation')`; `_response_catalog.py` has **no** coupling entry — decide whether λ history reaches `Results` or stays live-only | `src/apeGmsh/opensees/_response_catalog.py` | **Code** |
| 17b | 3.5 | Note the wire-format bump (`hdr 20 → 21`, version 2) for any `FE_Datastore` save/restore | coupling docs | Doc |
| 18 | 4.2 | Harness: for self-weight SANISAND, emit bare `-implex` (no `-implexControl`), record `material.implexRefusals`, assert `implex_refusals_companion == 0`, and **never call the result a capacity** — depends on items 1–2 | `nd.py:990-992`, `recorder.py:361`, harness | **Code** |
| 19 | 4.2 | Do **not** state a general "the control is not a low-confinement requirement"; the deck sits at 1.27× the P0 corner | harness docs | Doc |
| 20 | 4.2 | Pin the growth factor (×1.0) or `tol 0.5` when the control is wanted; `×1.25` is not enough | `src/apeGmsh/opensees/analysis/strategy.py` | Doc |
| 21 | 4.2 | Concurrent-writer guard on per-leg CSV output | harness | **Code** |
| 22 | 5.3 | Tcl target: add a documented minimum-build constant for #840 and cite it; **re-run** any Tcl-target validation that used `k` / `dofs` / `host` on the coupling family | `src/apeGmsh/opensees/emitter/tcl.py`, `_coupling_control.py` | Doc + re-run |
| 23 | — | Emitters for F7, F8, F10, WP-103 | — | **None** |
| 24 | — | `-dof` already emitted explicitly for u-p slaves (2026-09-07 guide item 3) | `build.py:7157-7158` | **None** |
| 25 | — | `implexRefusals` token spelling / recorder plumbing | `recorder.py:361` | **None** |
| 26 | — | `implexDetail` (6), `implexGuards` (7), `psi`, `yieldDistance` | `_ladruno_element_io.py:308-312`, `:325-...` | **None** |
| 27 | — | Vanilla `DruckerPrager` (`nd.py:227-390`) — F8 is `ASDPlasticMaterial3D` only | `nd.py` | **None** |
| 28 | — | `AUTO_STIFFNESS_ALPHA` numeric value (already inside the band) | `build.py:7167` | **None** |
| 29 | — | openseespy emitters / live runner / `APEGMSH_OPENSEES_BIN` seam for WP-103 | `emitter/py.py`, `emitter/live.py:238-295` | **None** |

**Blocking order.** Items 1–4 **and** 9a/9b gate item 18: you cannot assert on a
bucket the reader drops, and you cannot set the `max_substeps` that makes the
companion bucket move without tripping the substep-cap gate on any element but
`LadrunoBrick`. Item 6 gates any campaign run on a post-#838 build. Item 22 gates
every Tcl-target result already on disk.

---

## 7. What could not be verified here

- **All five PRs are open; none is merged.** Every claim above is read from the
  PR diffs and tests at the heads in the status box, not from a `ladruno` tip. A
  rebase or a review round could still move a token.
- **No fork binary was built or run for this guide.** Every measurement quoted
  (reach, wall times, gap norms, refusal counts, the 4-vs-6 slot evidence run) is
  the fork's own, reproduced verbatim from the PR bodies.
- **No apeGmsh code was changed and no apeGmsh test was run.** The reader
  drop-on-width-mismatch behaviour in §1.4(a) is read from
  `_block_canonicals` / `_name_mismatch` (`_ladruno_element_io.py:759-798`), not
  executed.
- **Line numbers are against `origin/main` at `dfa52e51`** (the branch point for
  this document). The repo's own working checkout was 47 commits behind
  `origin/main` at writing; re-verify before quoting a line number from a
  different tree.
- **#838's own "not verified" list stands**: no parallel/MP run, no mutation
  gate, Zone-A not run, the TIMs strip's 25.9 M / 2 674 kPa figures are quoted
  and not re-measured, and the `VariableTimeStepDirectIntegrationAnalysis` `-4`
  path is read from source rather than exercised by a test.
- **#839's own "not verified" list stands**: no MPI run of an AL coupling
  (the `lambda` probe is serial `FE_Datastore` only), the `iter` failure numbers
  for a **nonlinear** host are the reviewer's, and the TIMs `K_t` numbers
  (`1.4e-4`, `5.7e-8`, `6.7e-7`, the 101 583-DOF solver failures) are quoted from
  the campaign, not re-measured.
- **#836 did not re-run the 20-minute deck legs after round 1.**
- **`ops.ladrunoBuild()` is a CONFIGURE-time stamp and lags** (fork
  `CMakeLists.txt:200-207` captures the hash in `execute_process` at configure
  time; an incremental `build.bat` does not re-configure). It **understates**
  what was built. Do not use it alone to prove a binary contains any of the
  above — use a behavioural probe (e.g. `len(implexRefusals) == 6`).
