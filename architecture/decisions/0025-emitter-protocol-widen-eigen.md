# ADR 0025 — `Emitter.eigen` Protocol widening for one-shot modal solves

**Status:** Accepted (Cerro Lindo regression review, late-May 2026).
Closes feature F from the apeGmsh feature-requests backlog and widens
the `Emitter` Protocol (the explicit "architecture event" the
Protocol's header documents).

## Context

Modal analysis is core OpenSees functionality: `ops.eigen(solver,
num_modes)` returns a list of eigenvalues `λ_i = ω_i²`; mode shapes
are then queried via `ops.nodeEigenvector(node_tag, mode_idx)`.
Modal verification of large 3D shell models is the standard way to
sanity-check assembled stiffness and mass before stepping into a
non-linear pushover or transient run.

Before this ADR, the apeGmsh bridge had **no way** to drive an
`eigen` solve. The `analysis/` namespace exposed Static / Transient /
VariableTransient as registered primitives — three flag-only
analysis-type singletons that emit `analysis <Type>` and feed a
stepped `analyze N [dt]` driver on the bridge. Eigen does not fit
this shape:

- Eigen does **not** require a preceding `analysis <Type>` command.
  It needs only the assembled stiffness and mass matrices — no
  constraint handler, no numberer, no system, no test, no algorithm,
  no integrator, no analysis-type primitive.
- Eigen is a **one-shot** call, not a stepped driver. It does not
  consume the `analyze N [dt]` shape; the eigenvalues are returned
  directly from the same call that issues the solve.
- Eigen **returns values** to the user. The Static / Transient
  primitives return nothing — they only configure the OpenSees
  Analysis subclass that the next `analyze` invocation will use.

The downstream pain was concrete: the Cerro Lindo cimbra calibration
Phase 2 needs to compare a 3D shell `Mcr` (lateral-torsional buckling
moment) against an analytical Pi-Trahair Vlasov result, and the
cleanest path is a modal eigenvalue extraction. The fallback (push to
snap-through) takes ~8000 non-linear steps per design point and
mixes plasticity with stability — slow, and the mode discrimination
is poor.

The header of `opensees/emitter/base.py` documents that widening the
Protocol is *"an architecture event"*: every concrete emitter
(`TclEmitter`, `PyEmitter`, `LiveOpsEmitter`, `H5Emitter`,
`RecordingEmitter`) must grow the new method. This ADR is that event.
ADR 0024 (the `region()` widening) is the immediate precedent — same
shape of ceremony, smaller blast radius.

## Decision

### Widen the `Emitter` Protocol — one new method

```python
class Emitter(Protocol):
    # ... existing methods unchanged ...

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]: ...
```

`solver` is the verbatim OpenSees flag token — one of
`-genBandArpack` (default), `-symmBandLapack`, `-fullGenLapack`,
`-frequency`, `-standard`. Unrecognized tokens are passed through to
OpenSees, which raises at runtime (fail-loud; no whitelist
maintenance burden).

Implementation on all five concrete emitters:

| Emitter | `eigen(num_modes, *, solver)` |
|---|---|
| `TclEmitter` | Append `eigen $solver $numModes` (single line). Returns `[]`. |
| `PyEmitter` | Append `ops.eigen($solver, $numModes)` line. Returns `[]`. |
| `LiveOpsEmitter` | Call `ops.eigen(solver, num_modes)` directly on `openseespy.opensees`. Returns the eigenvalue list cast to `list[float]`. |
| `H5Emitter` | No-op. Eigen is a runtime retrieval, not part of the model definition the H5 archive captures. Returns `[]`. |
| `RecordingEmitter` | Append `("eigen", (), {"num_modes": …, "solver": …})` to `self.calls`. Returns `[]`. |

The asymmetric return contract mirrors `analyze`: the live emitter
returns the meaningful value; the others return a vacant default
(`0` for `analyze`, `[]` for `eigen`).

### Bridge method on `apeSees` — not a registered primitive

```python
class apeSees:
    def eigen(
        self,
        num_modes: int,
        *,
        solver: str = "-genBandArpack",
    ) -> EigenResult:
        ...
```

`apeSees.eigen` sits at the same architectural tier as
`apeSees.analyze`, `apeSees.tcl`, `apeSees.py`, `apeSees.run`,
`apeSees.h5` — bridge driver methods, not typed primitives. The
implementation builds a `BuiltModel`, drives a `LiveOpsEmitter`
through it (model + nodes + elements + bcs + mass), then issues the
single `eigen` call and wraps the eigenvalues in an `EigenResult`.

Unlike `apeSees.analyze`, **no analysis-chain pre-check** runs:
eigen does not consume constraints / numberer / system / test /
algorithm / integrator / analysis. The pre-check would be wrong here.

### `EigenResult` dataclass

```python
@dataclass(frozen=True, slots=True)
class EigenResult:
    eigenvalues: np.ndarray            # λ_i = ω_i²
    _live: LiveOpsEmitter              # kept for mode_shape() queries

    @property
    def omega(self) -> np.ndarray: ...     # √λ
    @property
    def freq(self) -> np.ndarray: ...      # ω / (2π)
    @property
    def periods(self) -> np.ndarray: ...   # 1 / f

    def mode_shape(self, node, mode) -> np.ndarray:
        # Lazy: ops.nodeEigenvector(tag, mode) at call time.
        ...
```

Eigenvectors are not eagerly fetched — they live in openseespy's
domain state until `mode_shape()` queries them. Calling
`mode_shape()` after another `apeSees.eigen(...)` or `ops.wipe()`
returns whatever openseespy currently holds; no staleness detection.

## Rejected alternatives

### Fold eigen into the Static/Transient/Analysis primitive pattern

Would require: a new `Eigen(Analysis)` dataclass that emits nothing
in `_emit` (because there is no `analysis Eigen` Tcl command) but
gets specially recognised by `BuiltModel.emit` to issue the actual
`eigen` call at the end of the build. The user would then call
`ops.analysis.Eigen(num_modes=5)` and somehow retrieve eigenvalues
back from `apeSees.run()` or `apeSees.analyze()`.

**Rejected because** the asymmetry is genuine OpenSees behavior, not
architectural drift: eigen has no `analysis` directive, no stepping,
and returns values. Forcing it into the `Static` mold means special-
casing it in three places (the primitive's `_emit`, the build
pipeline's emission order, and the bridge's return surface) and
producing the wrong mental model for users (eigen is not a "stepped
analysis you configure once and run repeatedly"). Modelling it
correctly once — as a bridge driver method — is cleaner.

### Probe-and-fallback / capability detection of the bound OpenSees binary

Considered as a way to gracefully handle binaries where eigen is
compiled out. Rejected on two grounds:

1. **Violates fail-loud.** The Cerro Lindo regression review (this
   ADR's parent) reaffirmed that silent capability fallbacks
   (`InitialStressMaterial` → `Steel02(sig_init)`) are toxic — they
   change the emitted Tcl text without the user asking, breaking the
   "Tcl emit and `ops.run()` bit-identical under the same build hash"
   invariant.
2. **Wrong layer.** The bridge does not adapt to its target binary;
   the binary is expected to satisfy the bridge's vocabulary, and
   regressions in that vocabulary are surfaced via live smoke tests
   pinned to a build hash. Capability detection at module load time
   would couple emit-time decisions to runtime probes — a deeper
   commitment than the problem warrants.

### Auto-inject the `eigen` line into `apeSees.tcl(path)` decks

Considered for symmetry with how the user might want a Tcl deck that
includes `eigen` for offline runs. Rejected because `apeSees.analyze`
also does not auto-inject into Tcl decks today — `analyze` is a
"I want values now" call. Eigen has the same shape. Forcing
auto-injection on `eigen` only would break the symmetry.

The user retains full control: they can drive a `TclEmitter` end-to-
end themselves and call `emitter.eigen(5)` on it to get the line in
their deck.

## Invariants

- **INV-1** — `Emitter.eigen` is on the Protocol; every existing and
  future emitter implements it. The `test_emitter_protocol.py`
  representative-method sweep pins this.
- **INV-2** — `apeSees.eigen` is a bridge driver method, not a
  registered primitive. No `eigen` instance appears in
  `BuiltModel.primitives`; no tag is allocated; no entry appears in
  the `_KIND_BY_FAMILY` dispatch.
- **INV-3** — `apeSees.eigen` does **not** invoke
  `_check_analysis_chain_for_analyze`. Eigen requires no analysis
  chain.
- **INV-4** — `EigenResult` is frozen. Once returned, the eigenvalue
  array is immutable; `mode_shape()` reads from openseespy state
  lazily and does not mutate the result.
- **INV-5** — Tcl / Py / live emit the same logical command for the
  same `(num_modes, solver)` pair. The bit-identical-emit invariant
  carries over from ADR 0008 (three emit targets).
- **INV-6** — `num_modes < 1` is refused at the bridge with
  `ValueError`. No bare or zero-mode `eigen` line reaches an emitter.

## Consequences

- Unblocks modal verification on the Cerro Lindo cimbra Phase 2 work
  and on every future apeGmsh model that needs a sanity-check eigen
  solve before non-linear stepping.
- The `EigenResult` surface is deliberately minimal — eigenvalues +
  derived ω / f / T + lazy `mode_shape()`. Modal mass / participation
  factors, Rayleigh damping helpers, Results-broker integration, H5
  persistence, and ResultsViewer mode-shape plots are **all deferred**
  to a second-use-case trigger. Add what the next caller asks for;
  do not pre-build.
- The H5 emitter is a no-op for `eigen` (the call is a runtime
  retrieval, not a model-definition declaration). No `model.h5`
  schema bump is required; per-zone schema versioning (ADR 0023) is
  unchanged. The `SCHEMA_VERSION` remains `2.9.0`.
- The Tcl and Py emitters do append the `eigen` line when called
  directly, so users who drive a `TclEmitter` / `PyEmitter` manually
  (outside `apeSees`) can still produce decks that include modal
  extraction. The default `apeSees.tcl(path)` / `apeSees.py(path)`
  flow does not auto-inject `eigen` — symmetric with `analyze`.
- The `eigen` Protocol method is general-purpose; future use cases
  beyond bridge-driven modal solves (e.g. an `EigenResult.to_h5(...)`
  surface, or a domain-capture spec for streaming mode shapes) can
  reuse the same Protocol entry point. This ADR closes the Protocol
  widening, not the set of consumers.

## Cross-references

- ADR 0008 — Three emit targets via Emitter Protocol (the bit-
  identical-emit invariant this ADR preserves).
- ADR 0022 — MP constraint emission fan-out (precedent for Protocol
  widening with the same ceremony).
- ADR 0023 — Per-zone schema versioning (the policy this ADR does
  *not* need to invoke — no schema bump required).
- ADR 0024 — `Emitter.region` Protocol widening (immediate
  precedent — same shape of ceremony, larger blast radius).
