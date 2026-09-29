# ADR 0031 — SSI convenience helpers: `convergence_confinement` + `imposed_displacement` (Phase SSI-3)

**Status:** Accepted (Phase SSI-3, May 2026). Fourth of the SSI
four-ADR set ([0028](0028-initial-stress-via-parameter-ramping.md)
/ [0029](0029-staged-analysis-context-manager.md) /
[0030](0030-stage-bound-topology-activation.md) /
[0031](0031-ssi-convenience-helpers.md)). **Does not widen the
`Emitter` Protocol** — both helpers compose over existing
primitives.

## Context

[ADR 0028](0028-initial-stress-via-parameter-ramping.md) shipped
`ops.initial_stress(...)` for the general in-situ stress install.
The Cerro Lindo tunnel migration drives two additional SSI use
cases that are *expressible* as compositions of existing primitives
but read poorly:

### Use case 1 — convergence-confinement relaxation

The tunnelling-mechanics canonical pattern: ramp a single stress
component on a boundary region to a fraction (`λ`) of the in-situ
target over `N` analyze steps, then stop. Used in
convergence-confinement analyses to model partial face advance
(`λ = 0.3` for "30% deconfinement applied" etc.). STKO reference:
[`SSI/Interaccion/analysis_steps.tcl:19753-19767`](file:///C:/Users/nmora/seadrive_root/nmb/My%20Libraries/Cerro%20Lindo/Modelos/SSI/Interaccion/analysis_steps.tcl)
emits a single-component `_stressCtrl_11` proc:

```tcl
proc _stressCtrl_11 {} {
    global _stressCtrl_11
    if {![info exists _stressCtrl_11(count)]} {
        set _stressCtrl_11(count) 0
        set _stressCtrl_11(XX) 0.0
    }
    set _stressCtrl_11(count) [expr {$_stressCtrl_11(count) + 1}]
    set _stressCtrl_factor [expr {$_stressCtrl_11(count) / 100.0}]
    if {$_stressCtrl_factor > 1.0} { set _stressCtrl_factor 1.0 }
    set _stressCtrl_current [expr -6300.0 * $_stressCtrl_factor]
    set _stressCtrl_incr [expr $_stressCtrl_current - $_stressCtrl_11(XX)]
    updateParameter 4 $_stressCtrl_incr
    set _stressCtrl_11(XX) $_stressCtrl_current
}
lappend STKO_VAR_OnBeforeAnalyze_CustomFunctions _stressCtrl_11
```

This is expressible via `ops.initial_stress(...)` already:

```python
ops.initial_stress(
    name="rock_relax_50",
    pg="Rock",
    sigma_xx=-6300.0, sigma_yy=0.0, sigma_zz=0.0,
    ramp_steps=100,
    lambda_install=0.5,         # the 50% partial install
)
```

But the kwarg names read awkwardly for the convergence-confinement
context — `lambda_install` reads as "fraction of permanent
installation" rather than the more natural "fraction of target
deconfinement". And `ramp_steps` is fine but doesn't match the
canonical tunnelling-spec phrasing `n_steps`.

### Use case 2 — fault-slip / support-settlement imposed displacement

Prescribed displacements on a node set, ramped via a time-series
factor. STKO reference: the fault-slip pattern at
[`SSI/Interaccion y Falla/analysis_steps.tcl:22832-23253`](file:///C:/Users/nmora/seadrive_root/nmb/My%20Libraries/Cerro%20Lindo/Modelos/SSI/Interaccion%20y%20Falla/analysis_steps.tcl)
emits:

```tcl
pattern Plain 16 1 -fact 0.001 {
    if {$STKO_VAR_process_id == 5} {
        sp 195 1 -1.0
        sp 195 2 -4.0
        ...
    }
}
```

This is expressible via the existing `ops.pattern.Plain(...)` +
`p.sp(...)` primitives:

```python
ts = ops.timeSeries.Linear(factor=0.001)
plain = ops.pattern.Plain(series=ts)
with plain:
    for node in node_ids:
        plain.sp(node=node, dof=1, value=-1.0)
        plain.sp(node=node, dof=2, value=-4.0)
```

— but every fault-slip / support-settlement caller writes the same
five-line boilerplate, and the `pattern_factor` vs.
`Linear(factor=)` choice (the apeSees `pattern Plain` primitive
does not carry a `-fact` arg, so the factor is folded into the time
series) is not obvious from the API surface.

Both use cases are repeated enough in the SSI literature and in the
Cerro Lindo decks to earn their own typed entry points. The
question for this ADR: do they earn **separate bridge methods**, or
should users compose them by hand?

## Decision

Ship two thin bridge methods that wrap existing primitives:

### `apeSees.convergence_confinement(...)` — wraps `initial_stress`

```python
def convergence_confinement(
    self, *,
    name: str,
    pg: str | None = None,
    elements: Iterable[int] | None = None,
    sigma_xx: float = 0.0,
    sigma_yy: float = 0.0,
    sigma_zz: float = 0.0,
    lambda_target: float,
    n_steps: int,
) -> InitialStressRecord:
    if sigma_xx == 0.0 and sigma_yy == 0.0 and sigma_zz == 0.0:
        raise ValueError(
            "apeSees.convergence_confinement: at least one of "
            "sigma_xx / sigma_yy / sigma_zz must be non-zero."
        )
    return self.initial_stress(
        name=name, pg=pg, elements=elements,
        sigma_xx=sigma_xx, sigma_yy=sigma_yy, sigma_zz=sigma_zz,
        ramp_steps=n_steps,
        lambda_install=lambda_target,
    )
```

Two cosmetic renames:

- `lambda_target` (vs. `lambda_install`) — reads more naturally for
  relaxation / confinement contexts than for installation.
- `n_steps` (vs. `ramp_steps`) — matches the tunnelling-spec
  phrasing.

Plus one extra validation: at least one of `sigma_xx / sigma_yy /
sigma_zz` must be non-zero. The vanilla `initial_stress` allows all-
zero because the call-site validation only enforces XOR on
`pg`/`elements`; for the convergence-confinement use case, all-zero
is a typo (the proc would emit but never advance any parameter).

Returns the underlying `InitialStressRecord` so callers can pass it
to a stage block via `s.add(record)`. Same record type, same
fan-out, same `addToParameter` mechanism. The deck output for a
single-component call (`sigma_xx=-6300.0, sigma_yy=0.0, sigma_zz=0.0`)
still emits **three** parameters with two 0.0-target updates per
step — that's [ADR 0028 INV-3](0028-initial-stress-via-parameter-ramping.md)
("Always three parameter declarations per record"); this helper does
not relax it.

### `apeSees.imposed_displacement(...)` — wraps `pattern.Plain` + `p.sp`

```python
def imposed_displacement(
    self, *,
    pg: str | None = None,
    nodes: Iterable[int] | None = None,
    ux: float | None = None,
    uy: float | None = None,
    uz: float | None = None,
    pattern_factor: float = 1.0,
    series: "TimeSeries | None" = None,
) -> "Plain":
    # ... validations ...
    if series is None:
        series = self.timeSeries.Linear(factor=float(pattern_factor))
    plain = self.pattern.Plain(series=series)
    dof_values = ((1, ux), (2, uy), (3, uz))
    with plain:
        if pg is not None:
            for dof, value in dof_values:
                if value is None:
                    continue
                plain.sp(pg=pg, dof=dof, value=float(value))
        else:
            for node in nodes:
                for dof, value in dof_values:
                    if value is None:
                        continue
                    plain.sp(node=int(node), dof=dof, value=float(value))
    return plain
```

Validations (build-time loud):

| Condition | Outcome |
|---|---|
| `(pg is None) == (nodes is None)` | `ValueError` |
| All of `ux` / `uy` / `uz` are `None` | `ValueError` |
| `pattern_factor == 0.0` | `ValueError` (inert pattern == typo) |
| `uz=` on `ndf=2` model (or any DOF index > `ndf`) | `ValueError` at declaration time (red-team H3, post-merge hardening) |

Where STKO uses `pattern Plain N tsTag -fact F { sp ... }`, this
helper folds `F` into an auto-created `Linear(factor=F)` time
series. Numerically identical (`value × F × t`), simpler API. Pass
an explicit pre-registered `series=` to override; `pattern_factor`
is then ignored.

Limitations (documented in the docstring + in [api-design.md](../api-design.md)
§"Imposed displacement"):

- **Scalar broadcast only.** Every targeted node gets the same
  scalar per DOF. For different values per node, the user calls
  `imposed_displacement` multiple times with disjoint `nodes=` lists
  or builds the pattern manually via `ops.pattern.Plain(...)` +
  per-node `p.sp(...)` calls.
- **Global pattern.** The returned `Plain` is registered globally;
  if used inside a staged deck it fires in every stage's analyze
  loop. Gate via the time-series shape if that is not desired.

## Invariants

- **INV-1.** Neither helper widens the `Emitter` Protocol. Both
  compose over existing primitives — `initial_stress` for SSI-1's
  ramp, `pattern.Plain` + `sp` for the imposed-displacement
  pattern. The Protocol's "architecture event" ceremony does not
  apply.
- **INV-2.** `convergence_confinement` returns the underlying
  `InitialStressRecord`. The return identity matches what
  `initial_stress` would have returned — `s.add(record)` works the
  same way whether the record came from `initial_stress` or
  `convergence_confinement`.
- **INV-3.** `imposed_displacement` returns the registered `Plain`
  pattern. Callers wanting to add more `sp` directives to the same
  pattern can reuse the returned handle via
  `with returned_plain: returned_plain.sp(...)`. The pattern is
  already registered with the bridge so no re-registration is
  needed.
- **INV-4.** All-zero stress targets are refused by
  `convergence_confinement` (validation absent from
  `initial_stress`). The convergence-confinement use case
  specifically intends to drive at least one component; all-zero
  is a typo that would emit a no-op proc.
- **INV-5.** `pattern_factor == 0.0` is refused by
  `imposed_displacement`. A zero factor produces an inert pattern
  that's almost certainly a typo (a deliberate factor-zero pattern
  is expressible by building `Plain` manually).
- **INV-6.** DOF index validation against `ndf` (red-team H3).
  `imposed_displacement(uz=...)` on an `ndf=2` model raises at
  declaration time, not at OpenSees parse time. Without INV-6 the
  emitted `sp NODE 3 VALUE` line would produce a less-helpful
  parse error during subprocess execution.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Don't ship either helper; users compose** | Both use cases are repeated enough in SSI literature and in the Cerro Lindo decks to justify named entry points. The composition recipe is non-obvious for `imposed_displacement` (the `-fact F` → `Linear(factor=F)` translation is easy to get wrong); shipping the helper documents the right pattern by example. |
| **Drop the cosmetic renames** — reuse `lambda_install` / `ramp_steps` on `convergence_confinement` | Loses the readability win that's the entire reason for the helper. The renames are cosmetic at the Python level but load-bearing at the docs / grep level — a user reading `lambda_target=0.5, n_steps=100` instantly recognises convergence-confinement; `lambda_install=0.5, ramp_steps=100` doesn't carry the same signal. |
| **Make `convergence_confinement` allow `sigma_*=0.0` everywhere** (mirror `initial_stress`) | `initial_stress` allows it because the validation focuses on the XOR contract; a zero-stress install is silly but not strictly wrong. For convergence-confinement specifically, all-zero is a typo — the entire point is to drive at least one component. Loud refusal (INV-4) catches the typo at the call site. |
| **Drop `imposed_displacement` in favour of `ops.fault_slip` / `ops.support_settlement`** | Both are special cases of the same mechanism (`pattern Plain` + `sp`). Splitting them adds API surface without adding capability. One verb covers both. |
| **Auto-detect DOF count from a model parameter** (let `uz=` silently go through on ndf=2 and let OpenSees error at parse time) | Worse error UX. OpenSees' parse error doesn't name the apeSees declaration; the validation (INV-6) catches it at the right layer. |
| **Make `convergence_confinement` and `imposed_displacement` namespace methods** (e.g. `ops.ssi.convergence_confinement(...)`) | Inconsistent with the rest of the apeSees surface — `ops.fix`, `ops.mass`, `ops.initial_stress`, `ops.region`, `ops.analyze`, `ops.eigen` are all flat methods. Adding a namespace just for two SSI helpers fragments the surface. |
| **Promote both helpers to ADR-grade architecture decisions individually** | The four-ADR set was structured around Protocol-widening events (SSI-1, SSI-2.A, SSI-2.B). SSI-3 doesn't widen the Protocol — it's two thin wrappers. Spending an ADR on each helper would inflate the architectural surface; combining them in one short ADR reflects the actual decision weight. |

## Consequences

**Positive:**

- Two repeated SSI patterns earn typed entry points. The Cerro
  Lindo decks become easier to scan: `ops.convergence_confinement(...)`
  vs. `ops.initial_stress(..., lambda_install=...)` immediately
  signals the use case.
- The `Linear(factor=F)` ↔ `pattern Plain N tsTag -fact F`
  translation is documented by example, not folklore. Users do not
  have to discover that the apeSees `Plain` pattern primitive
  doesn't carry a `-fact` arg — the helper folds the factor in
  correctly.
- DOF-index validation against `ndf` (INV-6) catches a common
  authoring mistake at the right layer.

**Negative:**

- Two more bridge methods on a surface that's already wide. Both
  return the underlying primitive (`InitialStressRecord` /
  `Plain`), so the typed result is still composable with
  `s.add(record)` / `with plain: ...`; the cost is the discoverability
  overhead.
- `convergence_confinement` reproduces [ADR 0028
  INV-3](0028-initial-stress-via-parameter-ramping.md)'s "always
  three parameters" divergence from STKO. A user reading the
  emitted deck for a single-component `convergence_confinement` call
  still sees three `parameter` declarations and three
  `updateParameter` lines per step. Documented in
  [api-design.md](../api-design.md) §"Initial-stress injection"
  and in this ADR's body; not relaxed by the helper.
- The `imposed_displacement` scalar-broadcast limitation hides the
  per-node-varying case behind a "build it manually" workaround.
  Users with non-trivial fault-slip distributions (varying slip
  amplitude along the fault trace) drop to `ops.pattern.Plain(...)`
  manually. Acceptable as a v1 limitation given that the typical
  Cerro Lindo cases are uniform-slip; could be lifted in a follow-up.

## Cross-references

- ADR [0028](0028-initial-stress-via-parameter-ramping.md) — the
  SSI-1 mechanism `convergence_confinement` wraps. INV-3 there
  documents the "always three parameters" choice this helper does
  not relax.
- ADR [0029](0029-staged-analysis-context-manager.md) — the
  staged-analysis context manager that consumes the
  `InitialStressRecord` returned by `convergence_confinement` via
  `s.add(record)`.
- ADR [0005](0005-patterns-explicit.md) — the explicit context-
  manager pattern shape that `imposed_displacement` builds on
  (it returns the registered `Plain` so callers can re-enter the
  context).
- ADR [0007](0007-time-series-separated-from-pattern.md) — the
  time-series-separate-from-pattern split that
  `imposed_displacement`'s `Linear(factor=)` folding respects.
- [api-design.md](../api-design.md) §"Initial-stress injection" /
  §"Imposed displacement" — the user-surface walkthroughs.
- [staged-analysis.md](../staged-analysis.md) — context for how the
  `InitialStressRecord` returned by `convergence_confinement`
  participates in per-stage emission.
- `SSI/Interaccion/analysis_steps.tcl:19753-19767` — the STKO
  reference proc this helper produces equivalent behaviour to.
- `SSI/Interaccion y Falla/analysis_steps.tcl:22832-23253` — the
  STKO reference fault-slip pattern this helper produces equivalent
  behaviour to.
- `tests/opensees/unit/test_phase3_helpers.py` — coverage of both
  helpers' validation paths and emitted-pattern shape.
- `tests/opensees/subprocess/test_phase3_subprocess.py` — Tcl + Py
  subprocess smoke for both helpers end-to-end.
