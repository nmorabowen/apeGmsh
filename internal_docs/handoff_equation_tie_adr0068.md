# Handoff — tied-interface `enforce=` routes (ADR 0068)

Status: **P0–P4 shipped + Open item 5 closed**, all on `main` (PRs
[#697](https://github.com/nmorabowen/apeGmsh/pull/697) →
[#701](https://github.com/nmorabowen/apeGmsh/pull/701) →
[#702](https://github.com/nmorabowen/apeGmsh/pull/702)). Every
*silent-correctness* gap from the adversarial review is closed. Remaining
P5 items are convenience/scope, not correctness.

`g.constraints.tie` / `tied_contact` now choose **how** a non-matching
interface is coupled via `enforce=`. This is the apeGmsh analogue of
LS-DYNA `*CONTACT_TIED_SURFACE_TO_SURFACE`: the penalty route is the
offset-penalty variant; the equation route is the constrained (kinematic)
variant. Origin: an OpenSees-H5DRM-vs-LS-DYNA-SSI review — the equation
route is what lets a **non-matching soil-structure interface run under
explicit DRM** (penalty ties shrink the critical Δt; the projection-handled
equation tie does not).

Design rationale: `src/apeGmsh/opensees/architecture/decisions/0068-equation-constraint-tied-interface.md`.

---

## The three `enforce=` routes

All three reuse the **same** geometry — `resolve_tie` Newton-projects each
slave node onto the closest master face and evaluates the face shape
functions; only the emission differs.

| `enforce=` | emits | knobs | enforcement |
|---|---|---|---|
| `"penalty"` *(default)* | `ASDEmbeddedNodeElement` | `stiffness` / `stiffness_p` / `rotational` / `pressure` | penalty element (handler-independent); 3/4-node host only |
| `"penalty_al"` | `LadrunoEmbeddedNode` (fork) | `control=CouplingControl(...)` | penalty + augmented-Lagrange + bipenalty; any host arity; **translations only** |
| `"equation"` | `equationConstraint` (`EQ_Constraint`) | — | **exact**, handler-enforced; any host arity; **translations only** |

```python
from apeGmsh._kernel._coupling_control import CouplingControl

g.constraints.tie("struct", "soil", enforce="penalty")                 # default, unchanged
g.constraints.tie("struct", "soil", enforce="equation")                # exact kinematic tie
g.constraints.tie("struct", "soil", enforce="penalty_al",
                  control=CouplingControl(k="auto", host=<fem_eid>,
                                          enforce="al"))                # fork penalty+AL
```

**Constraint math (equation route):** per tied DOF `d`,
`u_d(slave) = Σ wᵢ·u_d(masterᵢ)`, emitted as
`equationConstraint slave d 1.0  m₁ d -w₁  m₂ d -w₂ …` (the OpenSees
sum-to-zero form `cCoef·cDOF + Σ rCoef·rDOF = 0`, `Ccr = -rci/cc`).

## Handler selection (equation route)

`EQ_Constraint` is **upstream** OpenSees (Yuli Huang, ~May 2025). It is
enforced by `Lagrange` / `Penalty` / **`LadrunoProjection`** (fork) — **not**
by `Transformation` or `Auto` (neither iterates `getEQs()`; verified in the
fork C++). The bridge handles this automatically:

- **Non-staged:** `_maybe_auto_emit_constraint_handler` auto-emits
  `Lagrange` (implicit) when an equation tie is present, **never**
  `Transformation`, and **fails loud** if the user declared
  `Transformation` *or* `Auto` (INV-4).
- **Staged:** `_validate_staged_eq_handlers` runs in both
  `_emit_stages_flat` and `_emit_stages_partitioned` and **fails loud per
  stage** whose `stage.constraints` isn't `Lagrange`/`Penalty`/
  `LadrunoProjection` (incl. none → default `Plain`). (Open item 5.)
- **Explicit transient:** declare `ops.constraints.LadrunoProjection()`
  yourself (Δt-neutral, momentum-conserving). Auto-detecting implicit vs
  explicit is Open item 1 — until then the auto-emit warns.

## Persistence

`enforce` round-trips through `model.h5` (neutral schema **2.14.0**):
`enforce` column on the interpolation lane, `sr_enforce` (uint8 0/1/2) on
the surface-coupling lane, both presence-probed (2.13.x files decode as
`"penalty"`). `CouplingControl` already round-trips via the `cpl_*` /
`sr_cpl_*` columns, so `penalty_al` needed **no** schema work.

**The equation tie is fully recoverable from `model.h5` — Open item 4
RESOLVED with no schema bump.** The interpolation lane persists not just
`enforce` but the projection `weights` (and `slave_node`/`master_nodes`/
`dofs`) — everything `_emit_equation_tie` needs. So an equation-tied archive
round-trips via `FEMData.from_h5` → `apeSees(fem).tcl()/py()/run()` (the
forward emit re-runs `_emit_one_interpolation` → `_emit_equation_tie`),
exactly like g.embed / contact / reinforce. The H5 *deck* emitter now no-ops
`equationConstraint` **silently** (mirroring those siblings); the old
`H5EquationConstraintDeviationWarning` is **dormant** (retained for
back-compat). A dedicated `/opensees` deck-zone record (standalone deck
replay) stays the shared low-priority follow-on with reinforce/contact/embed.

## Where things live

| Concern | File |
|---|---|
| `enforce=` / `control=` params | `core/ConstraintsComposite.py` (`tie`, `tied_contact`) |
| defs + validation | `_kernel/defs/constraints.py` (`TieDef`/`TiedContactDef`, `_validate_tie_enforce`) |
| record field | `_kernel/records/_constraints.py` (`InterpolationRecord.enforce` / `.control`) |
| resolver threading | `_kernel/resolvers/_constraint_resolver/_resolver.py` |
| emission | `opensees/_internal/build.py` (`_emit_one_interpolation`, `_emit_equation_tie`, `_emit_penalty_al_tie`) |
| Protocol method | `opensees/emitter/base.py` `equationConstraint` + all 5 emitters |
| handler primitive | `opensees/analysis/constraint_handler.py` (`LadrunoProjection`) |
| handler guards | `opensees/apesees.py` (`_maybe_auto_emit_constraint_handler`, `_validate_staged_eq_handlers`) — **on class `BuiltModel`, not `apeSees`** |
| H5 codec | `mesh/_record_h5.py` + `mesh/_femdata_h5_io.py` (`enforce`/`sr_enforce`) |
| tests | `tests/test_equation_tie_emission.py`, `tests/test_penalty_al_tie_emission.py`, `tests/test_staged_eq_handler_guard.py` |

## Adversarial review — fixed

4-lens review (partitioned / persistence / handler-matrix / math-edge):
zero-weight masters silently dropped the whole tie (OpenSees rejects any
zero `rcoef`, `EQ_Constraint.cpp:98` → filter them); `Auto` handler
unguarded; partitioned planner mis-routed EQ (now fail-loud — cross-rank is
Open item 2); `tied_contact` "bidirectional" docstring was false; the
persistence drop above; and a bytes-vs-`str` decode bug (use `_str()`).

## Testing

Run on the **fork** interpreter (it has `equationConstraint` + the fork
elements/handler):

```
$env:PYTHONPATH = "<repo>\src"; $env:LADRUNO_OPENSEES_QUIET = "1"
C:\Users\nmora\venv\opensees_venv\Scripts\python.exe -m pytest `
  tests/test_equation_tie_emission.py `
  tests/test_penalty_al_tie_emission.py `
  tests/test_staged_eq_handler_guard.py -q
```

`test_equation_tie_emission.py` includes a **live openseespy static solve**
proving `u(slave)=Σ wᵢ·u(mᵢ)` to 1e-9 (sign convention end-to-end on the
fork build) + an H5 round-trip. The system `Python311` has numpy/pytest but
**no** openseespy, so live tests skip there. Static gates (ruff + mypy
ratchet, baseline 0) cover `src/apeGmsh/opensees` only.

## Tie-force recovery — SHIPPED (P5 item 2)

The interface force a non-matching equation tie carries (≈ LS-DYNA
`*DATABASE_NCFORC`) is now recoverable two ways. Both reuse the fork OpenSees
tie force `f = M(a_raw − a_proj)` (the `LadrunoProjection` handler's projection
constraint force, ADR-30 P3/P4, already on `ladruno`); the apeGmsh side was a
thin pair of wrappers + one reader-map entry.

- **Live query** — `apeSees.ladruno_projection_tie_force(node, dof)` (1-based
  `dof`) returns the tie force from the last projection step of a prior **live**
  `analyze(...)`. Delegates to the fork-gated
  `LiveOpsEmitter.ladruno_projection_tie_force` (mirrors `critical_time_step`).
  `BridgeError` before any live run; `RuntimeError` on a stock build. Works
  implicit *and* explicit.

  ```python
  ops = apeSees(fem)
  ...                                       # build + LadrunoProjection handler
  ops.analyze(steps=1, dt=1e-3)
  f = ops.ladruno_projection_tie_force(slave_node, 1)
  ```

- **Recorder readback** — emission already worked via the passthrough
  `ops.recorder.Ladruno(nodal_responses=("constraintTieForce",))`; the only gap
  was reading it back. Closed by one `_NODAL_RESULT_NAME_MAP` entry
  (`CONSTRAINT_TIE_FORCE` → `constraint_tie_force`,
  `results/readers/_mpco_translation.py`) so a `.ladruno` file is read via
  `results.nodes.get(component="constraint_tie_force_x")` (components
  `TFx,TFy,TFz`; vec3/node; **explicit-only** — scattered each commit by
  `CentralDifferenceLadruno`).

Tests: `tests/test_tie_force_helper.py`. The readback-mapping + `BridgeError`-gate
legs run anywhere; the live legs (query == exact `F·m₂/(m₁+m₂)`, wrong-handler
guard, recorder round-trip via `LadrunoReader`) need an openseespy **built from
`ladruno` ≥ ADR-30 P3/P4** and skip cleanly otherwise (the `opensees_venv`
openseespy + the local `dist` build as of 2026-06-18 predate the command —
verified absent via `hasattr` / `strings`; the commands themselves are confirmed
on `origin/ladruno`). Files: `opensees/emitter/live.py`,
`opensees/apesees.py`, `results/readers/_mpco_translation.py`. Docs:
`guide_constraints.md` §"Tie-force recovery". Not done: the declarative canonical
token (`capture(nodes="constraint_tie_force")`) — deferred to avoid the shared
`_MPCO_NODE_TOKENS` MPCO-leak risk; emission stays the documented passthrough.

## Handler implicit/explicit auto-detect — SHIPPED (P5, Open item 1)

`BuiltModel._maybe_auto_emit_constraint_handler` now keys the EQ-tie handler
auto-emit off the registered integrator (shared `_is_explicit_integrator`, the
same classifier `apeSees._check_explicit_solver_compat` uses — refactored to one
source of truth). When an `enforce="equation"` tie is present and no handler was
declared: **explicit** integrator → auto-emit fork **`LadrunoProjection`**
(Δt-neutral); **implicit / none** → **`Lagrange`** (exact). Declaring a handler
is the override (respected as before); INV-4 still fail-louds on
`Transformation`/`Auto` + equation tie; a *soft* warning now fires for
user-declared `Lagrange` + explicit integrator (massless multiplier DOFs break
an explicit mass solve). Files: `opensees/apesees.py`. Tests:
`tests/test_constraint_emission_phase7b.py` (`TestEquationTieHandlerAutoDetect`).
No `tie_handler=` kwarg was needed — the explicit-handler path is the override.

## Cross-partition EQ replication — SHIPPED (P5, Open item 2)

`_plan_rank_constraints` (`opensees/_internal/build.py`) no longer fail-louds on
an `enforce="equation"` tie under partitioned emit. It now replicates the tie on
EVERY rank that owns the slave OR any master (the `rigidDiaphragm` rule, not the
single-canonical-host-rank element rule the penalty tie uses): a new
`_RankConstraintPlan.equation_records` lane collects such ties per rank, each
owning rank ghost-declares the foreign slave/masters first (existing
`_add_foreign_or_phantom`), and `emit_mp_constraints_partitioned` emits them via
`_emit_surface_couplings_for_rank` → `_emit_one_interpolation` → `_emit_equation_tie`
(no element tag, so replication is tag-stream-neutral; rows byte-identical across
ranks). Handler side is covered by Open item 1 (auto-emit picks
`LadrunoProjection`/`Lagrange`). Tests:
`tests/opensees/integration/test_emit_partitioned_replicate_on_both.py`
(`test_cross_rank_equationConstraint_replicates_on_owning_ranks` +
`test_equationConstraint_single_owning_rank_no_spurious_replication`).
Emission logic is unit-verified without MPI; a live OpenSeesMP run is the final
confirmation when a multi-rank fork build is available.

## Remaining (P5 — none are correctness gaps)

1. **DRM integration test** *(capstone)* — non-matching soil/structure
   under explicit DRM (ADR 0066); the real use case end-to-end. Needs an
   openseespy **built from current `ladruno`** (the local build is stale) and a
   full SSI model.
