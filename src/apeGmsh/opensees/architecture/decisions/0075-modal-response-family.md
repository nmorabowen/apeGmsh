# ADR 0075 — Consuming the Ladruno modal-analysis family

**Status:** Accepted (2026-07-13). Umbrella ADR for the apeGmsh
consumption of the fork's modal family (fork ADRs 43 / 44 / 45 / 46);
ships across five PR slices, each referencing this document.

## Context

The Ladruno fork's modal-analysis program (fork ADR 45 umbrella) is
complete on `ladruno`:

- **Fork ADR 44 — `LadrunoModalResponse` (classTag 33024).**
  `modalResponseHistory` (exact piecewise-linear modal-superposition
  transient — commits one domain step per station, so ordinary
  recorders capture the history), `responseSpectrumAnalysis -combine`
  (native CQC / SRSS / ABS / TenPercent over the per-mode modal
  displacements), `frequencyResponse` / `steadyStateDynamics` (modal
  FRF sweep, rows returned to the interpreter), and `randomResponse`
  (stationary PSD → RMS on the same FRF). Two excitation channels
  everywhere: uniform base acceleration (`-baseAccel -dir`) and a
  nodal-force pattern (`-load`). Fork PRs #537/#539/#544/#546/#552/
  #553/#555.
- **Fork ADR 43 — FEAST eigensolver.** `eigen -feast fmin fmax`
  band-targeted eigensolve (returns *all* modes in the band — the
  count is not an input) plus a `-certify` Sturm/inertia completeness
  certificate; MP-parallel capable. Fork PRs #515/#517/#524–#532.
- **Fork ADR 46 — `complexEigen` (classTag 33019).** Complex /
  state-space modal for non-classically damped models: per-mode true
  damping ratios ζ_k, damped frequencies ω_d,k, phased mode shapes.
  Fork PRs #506–#514.

Before this ADR the bridge consumed none of it, and `modalProperties`
(upstream `DomainModalProperties`, the prerequisite state for every
ADR-44 command) had no apeGmsh surface at all.

## Decision

### Two-tier surface, following existing precedent

**Tier 1 — Emitter-Protocol widening** (ADR 0025 ceremony; all five
concrete emitters) for commands that belong in decks — they commit
domain state and/or are the MP-parallel path:

| Protocol method | OpenSees command | fork-only? |
|---|---|---|
| `modal_properties(*, unorm=False, out=None) -> dict` | `modalProperties [-unorm] [-file $out]` | no (upstream) |
| `modal_response_history(...)` *(PR 2)* | `modalResponseHistory ...` | yes |
| `response_spectrum_analysis(...)` *(PR 2)* | `responseSpectrumAnalysis ... -combine ...` | `-combine` is fork-only |
| `eigen_feast(f_min, f_max, *, certify=False) -> list[float]` *(PR 4)* | `eigen -feast $fmin $fmax [-certify]` | yes |

Emission contract per emitter (mirrors `eigen`): live executes and
returns the real value; Tcl / py emit the single command line and
return a vacant default; H5 **no-ops** (runtime retrieval, not model
definition — no schema bump, same rationale as `eigen`); recording
captures `(name, args, kwargs)`.

The live `modal_properties` passes `-return` to get the properties
dict; the deck emitters do **not** emit `-return` (an
interpreter-return concern with no meaning in a deck).

`eigen_feast` is a **separate Protocol method**, not an `eigen`
overload: the band form has no a-priori mode count, which breaks
`eigen`'s `num_modes` contract.

**Classic-Tcl caveat for `-feast`**: the fork wires `-feast` into the
interpreter/openseespy `eigen` parser only — the classic
`OpenSees.exe` / `OpenSeesMP.exe` exes (`SRC/tcl/commands.cpp`) do
**not** parse it, so a Tcl deck carrying `eigen -feast ...` fails at
parse time on those binaries. The deck target for `eigen_feast` is
openseespy decks until the fork adds classic parity (the other
family commands ARE wired into classic Tcl).

**Tier 2 — live-emitter-only methods** (the `profiler` /
`critical_time_step` tier — `getattr(self._ops, name, None)` gate +
friendly "requires the Ladruno fork" `RuntimeError`) for commands
whose value *is* the interpreter return: `frequency_response`,
`steady_state_dynamics`, `random_response` *(PR 3)*, `complex_eigen`
*(PR 5)*. No deck story in v1 — the three sweep drivers additionally
pass `out=` through to the fork's `-out <file>` for users who want an
artifact on disk (`complex_eigen` has no `-out` at either end; its
disk artifact is the recorder route below).

### Bridge drivers auto-issue prerequisites

Every ADR-44 driver on `apeSees` builds a **fresh** domain
(`build()` → `LiveOpsEmitter(wipe=True)` → `bm.emit`), so a "prior"
`apeSees.eigen()` cannot satisfy the fork commands' eigen +
modalProperties prerequisite — it ran on a different domain. Each
driver therefore takes a required `num_modes` kwarg and issues
`emitter.eigen(num_modes, solver=solver)` →
`emitter.modal_properties()` → the command itself (precedent:
`_emit_modal_damping`, ADR 0053 D4, which bundles eigen +
modalDamping the same way).

### Explicit per-call damping — no coupling to `ops.damping`

The modal-response commands take at most one of `damp=` /
`rayleigh=(a0, a1)` / `modal_damp=[ξ1, ..]`, validated by the shared
`_damping_channel_args` helper (`analysis/modal.py`) and rendered to
the verbatim fork flags. The channel is **required** (exactly one)
on `modal_response_history` and the three sweeps; on
`response_spectrum_analysis` it is **optional** (SRSS/ABS/TenPercent
need none; CQC requires one) and `rayleigh=` is **not offered** —
the fork RSA parser accepts only `-damp`/`-modalDamp` and silently
drops unknown flags. Ratios must be `>= 0` (the fork refuses
negatives on four of five parsers; RSA does not, and a mixed-sign
CQC list silently zeros the combined field — the bridge closes that
hole). Damping here is an **analysis input** of the
modal-superposition post-processor, not a model property — deriving
it from ADR 0053 `ops.damping` declarations was considered and
rejected (silent coupling between a model declaration and a
post-processing analysis; the fork itself warns that `getDamp()`-
invisible damping like `modalDamping` does not flow into these
commands anyway).

### Handle → tag resolution

Excitation channels reference registered primitives: `base_accel=` /
`series=` / `input_psd=` take `ops.timeSeries.*` handles, `load=`
takes an `ops.pattern.Plain` handle. Tags are allocated at
registration (`apeSees.tag_for(prim)`), so drivers resolve handles
before emit and fail loud (`BridgeError`) on unregistered handles.
A registered timeSeries **emits even when no pattern references it**
(pinned by test) — the `-baseAccel` / `-inputPSD` channels need no
dummy pattern.

### Staged models refused

All new drivers copy the `eigen` guard: any registered stage →
`NotImplementedError` (live staged execution is unsupported, Phase
SSI-2.A). Per-stage modal analysis stays deferred with the ADR 0053
per-stage-modal item.

### Result surfaces (frozen dataclasses, `EigenResult` style)

- `ModalPropertiesResult` (PR 1, `analysis/modal.py`) — eigenvalues +
  the raw `-return` dict + derived ω/f/T + component-keyed accessors
  (`participation_factors("MX")`, `mass_ratios`,
  `cumulative_mass_ratios` — percent, as OpenSees returns them) +
  lazy `mode_shape`. Dict keys follow the `printDict` layout
  (`DomainModalProperties.cpp`): components `MX/MY/MZ/RMX/RMY/RMZ`
  (2-D: `MX/MY/RMZ`).
- `ModalHistoryResult` / `ResponseSpectrumResult` (PR 2) — lazy
  readers over the committed domain state (`_live` back-reference,
  same documented staleness caveat as `EigenResult`).
- `FrequencyResponseResult` / `SteadyStateResult` /
  `RandomResponseResult` (PR 3) — **eager** (no `_live`): the sweep
  values are fully returned by the command.
- `EigenResult` is reused unchanged by `eigen_feast` (PR 4).
- `ComplexEigenResult` (PR 5) — parses the flat 7-per-mode list
  `[ω0, ω_d, ζ, Re λ, Im λ, kind, resid]` into arrays.

### Fork gating

Live tier-2 methods gate on the missing openseespy attribute.
`eigen_feast` rides the stock `ops.eigen` symbol, so its driver
pre-checks `capabilities().has_fork` instead. `modalProperties` is
upstream — no gate. `_assert_fork_if_required` (OpenSeesTarget
`require_fork=True`) applies to all drivers as usual.

## Rejected alternatives

- **Analysis-primitive modelling** (`ops.analysis.ModalHistory(...)`)
  — same rejection as ADR 0025: these are one-shot commands that
  return values and take no `analysis <Type>` chain; forcing the
  primitive mold means special-casing emit order and return surfaces.
- **Auto-deriving damping from `ops.damping`** — see above.
- **A Protocol method per tier-2 command** — the FRF/random values
  are interpreter returns; a Tcl deck line without `-out` is a no-op
  for the user, and with `-out` it is expressible via the live
  driver's `out=` passthrough. Widen later if a deck use case appears
  (second-use-case trigger, ADR 0025 consequence style).

## Invariants

- **INV-1** — Protocol methods exist on all five emitters; the
  `test_emitter_protocol.py` representative sweep pins each widening.
- **INV-2** — H5 no-ops on every method this ADR adds; no schema
  bump anywhere in the family.
- **INV-3** — ADR-44 drivers emit `eigen` → `modalProperties` →
  command, in that order, on the same live emitter.
- **INV-4** — damping-channel validation (at-most-one-of, required
  where the command demands it, every ratio `>= 0`) happens at the
  bridge (before any emission), shared via `_damping_channel_args`.
- **INV-5** — Tcl / py emit the same logical command for the same
  driver arguments (bit-identical-emit, ADR 0008).
- **INV-6** — all result dataclasses are frozen; `_live`-holding
  results carry the ADR 0025 INV-4 staleness contract (lazy reads,
  no detection).

## Cross-references

- ADR 0025 — `Emitter.eigen` widening (the ceremony + driver-tier
  precedent this ADR follows).
- ADR 0053 — damping definition (D4 bundled eigen+modalDamping is
  the prerequisite-auto-issue precedent; the no-coupling decision
  lives against its declaration surface).
- Fork ADRs 43 / 44 / 45 / 46 and
  `Ladruno_implementation/LadrunoModalResponse_guide.md` /
  `LadrunoComplexEigen_guide.md` — the authoritative command
  surfaces and validation oracles.
