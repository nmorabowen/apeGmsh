# ADR 0101 — SANISAND materials + `s.update_material_stage`

**Status:** Accepted. Adds `ops.nDMaterial.ManzariDafalias` and
`ops.nDMaterial.SAniSandMS` in
[`material/nd.py`](../../material/nd.py), and a stage-bound
`s.update_material_stage(...)` mutator on the staged context manager
in [`apesees.py`](../../apesees.py). Bumps the OpenSees zone H5
schema to 2.21.0.

## Context

[`_DEFERRED.md`](../_DEFERRED.md) carried an entry for
`updateMaterialStage` — the staged-gravity idiom every soil-plasticity
deck needs (build elastic, solve gravity, flip to plastic, push) — but
that entry's own point 4 explained why it could not land alone: "a
typed staging mutator over materials apeGmsh cannot yet *declare*
would be typed at one end and stringly at the other." `nDMaterial
ManzariDafalias` was not wrapped at all, and `updateMaterialStage`
only does anything useful against a material that actually honours
staged plasticity. The two halves — the typed SANISAND materials and
the typed stage mutator — had to land together or neither one would
be trustworthy.

This also mattered more than it used to: on Ladruno engines built
from 2026-08 (fork PR #714), `ManzariDafalias`'s static `mElastFlag`
default flipped from `1` to `0` — the constructor now starts
**elastic**, not elastoplastic. A deck that never called
`updateMaterialStage` used to get plasticity by accident; on the new
default it silently stays elastic for the whole run. Older builds
still start elastoplastic. Either way, a deck that cares should set
the stage explicitly rather than lean on the constructor default.

## Decision

### Parameter surface

Both materials are typed, frozen, `kw_only` dataclasses in
`material/nd.py`, mirroring their Tcl signatures:

```
nDMaterial ManzariDafalias $tag $G0 $nu $e_init $Mc $c $lambda_c $e0 \
    $ksi $P_atm $m $h0 $Ch $nb $A0 $nd $z_max $cz $Rho \
    <$IntScheme $TanType $JacoType $TolF $TolR>

nDMaterial SAniSandMS $tag $G0 $nu $e_init $Mc $c $lambda_c $e0 $ksi \
    $P_atm $m $h0 $Ch $nb $A0 $nd $zeta $mu0 $beta $Rho \
    <$IntScheme $TanType $JacoType $TolF $TolR>
```

`ManzariDafalias` takes 18 required doubles (`G0 nu e_init Mc c
lambda_c e0 ksi P_atm m h0 Ch nb A0 nd z_max cz rho`) plus the optional
tail `int_scheme=1 tan_type=0 jaco_type=1 tol_f=1e-7 tol_r=1e-7`.
`SAniSandMS` shares the first 15 required doubles, then substitutes
the memory-surface trio `zeta mu0 beta` for the fabric-dilatancy pair
`z_max cz`, for 19 required doubles, plus the same five-slot tail with
different defaults (`int_scheme=3 tan_type=2`).

The tail emits **all-or-nothing**: every optional at its default emits
only the required block; changing any one emits all five. Both C++
parsers read the tail positionally into a fixed-size array, so a
partial tail would misalign every downstream field. Verified against
the live engine — 18/23 args accepted for `ManzariDafalias` and 19/24
for `SAniSandMS`, in all four combinations (required-only /
required+tail, both materials).

### The four fork-notes findings, and how each is answered

1. **Static `mElastFlag` (both materials start elastic on 2026-08+
   Ladruno builds).** Answered by making `materials=` on
   `s.update_material_stage` a *sequence*, not a single handle — see
   below — and by documenting the rule in
   `internal_docs/guide_opensees.md`: because the flag is
   process-wide, every SANISAND material that needs plasticity in a
   given stage must be listed in the same call.

2. **`ManzariDafalias.int_scheme` 3 (RungeKutta4) or 5 (ForwardEuler)
   integrate with no error control.** Their adaptive-substep code is
   dead and the yield-drift correction is commented out in the C++. A
   reported triaxial characterisation came out 31–46% too strong on
   scheme 3 before the mismatch was caught. `__post_init__` still
   accepts these values (they run) but raises
   `SanisandIntegrationWarning` pointing at `int_scheme=1`
   (ModifiedEuler) or `45` (RungeKutta45, Sloan) instead.

3. **`SAniSandMS.int_scheme` outside `{1, 3}` is unsafe at the process
   level, not just numerically.** `SAniSandMS.cpp:1240-1254` sends
   0, 4, 6, 7, 8, 9 into branches that call `exit(0)` — killing the
   Python process with no traceback, no exception to catch. 2
   (BackwardEuler) prints "Implicit integration not available yet" and
   silently integrates nothing; 5 prints "does not work" and falls
   through to RK4. Because a warning cannot protect against a process
   `exit(0)`, `__post_init__` raises `ValueError` for anything outside
   `{1, 3}`, hard-stopping before the material ever reaches the
   emitter.

4. **`ssp` element formulations pair unsafely with either SANISAND
   material.** `ssp` derives its stabilization stiffness from the
   material's initial tangent. `ManzariDafalias::initialize()`
   (`ManzariDafalias.cpp:826-864`) evaluates that tangent via
   `GetElasticModuli` at a hard-coded reference stress `p = P_atm`
   (`:855`), regardless of the model's real confinement — at low or
   high confinement the stabilization is scaled from the wrong
   reference. `SAniSandMS` is a separate implementation (not a
   `ManzariDafalias` subclass) but its own `initialize()`
   (`SAniSandMS.cpp:1010-1013`) hard-codes the same `p = P_atm`
   reference, so the pairing is unsafe there too. This is a known
   upstream `ssp` defect (TIMs report item 9) with no apeGmsh-side
   fix, so `SSP_UNSAFE_MATERIAL_CLASSES` in
   [`_element_capabilities.py`](../../_element_capabilities.py) and
   `_warn_if_ssp_unsafe` in
   [`element/solid.py`](../../element/solid.py) only **warn** — a
   `LadrunoBrick`/`LadrunoQuad` with `formulation="ssp"` and a SANISAND
   material still builds, it just warns.

### `s.update_material_stage` is stage-bound, not top-level

```python
with ops.stage("gravity") as s:
    ...
with ops.stage("push") as s:
    s.update_material_stage(materials=[soil], stage=1)
```

There is no `apeSees.update_material_stage` — it exists only on the
`Stage` object returned by `with ops.stage(...) as s`, mirroring
`s.remove_sp` / `s.remove_element` / `s.activate_absorbing` (the
Phase SSI-2.E between-stage Domain mutators family, ADR 0034's
V1–V5 validator discipline). `updateMaterialStage` is, mechanically,
exactly that kind of call: a line that only makes sense inside a
staged deck, ordered relative to a specific stage's element
activation and BC changes. Emission order within a stage is: element
activation → `remove_sp`/`remove_element` → the
`update_material_stage` lines → the stage's new `fix`/`mass`/`region`
lines — so a stage can release, re-fix, and flip in the right order.

`materials=` takes a sequence deliberately.
`ManzariDafalias::mElastFlag` is a `static` field — one flag shared by
every instance in the process, reset by any constructor call. Two
SANISAND materials cannot sit at different stages; the enforced
practice is to pass every material tag that needs the flip in one
call, every time. Only `stage=0` (elastic) or `stage=1` (elastoplastic)
are accepted; the handler additionally calls `Elastic2Plastic()` on 1.

Emits one `updateMaterialStage -material $tag -stage $stage` line per
material, in the order given — no `-parameter` form.

### Validation: call-site + build-time (V7)

Two layers:

- **Call-site**: `s.update_material_stage` rejects an unregistered
  handle (`BridgeError`) and rejects any material whose
  `type(...).__name__` is not in `STAGED_MATERIAL_CLASSES` — a
  `frozenset({"ManzariDafalias", "SAniSandMS"})` in `apesees.py`. This
  is the named extension point: flipping an `ElasticIsotropic` is a
  loud, immediate `BridgeError` instead of a silent no-op, and the set
  is exactly where a future `LadrunoSANISAND` or the
  `PressureIndepMultiYield`/`PM4Sand` family gets added once typed.

- **Build-time (V7)**: `_validate_material_stage_targets` in
  `apesees.py`. It exists because of a measured runtime fact:
  `updateMaterialStage` resolves through the **Domain's live
  elements**, not the material registry.
  `MaterialStageParameter::setDomain()` walks the Domain's elements
  looking for the material tag; on a miss it prints
  `MaterialStageParameter::setDomain() - no effect with material tag
  N`, the command still reports success, and the flip silently does
  nothing. A material declared but never attached to a live element —
  or attached only to elements a *later* stage activates — hits this
  exactly. V7 walks each `update_material_stage_records` target,
  computes the earliest stage at which some element using that
  material tag is live, and raises `BridgeError` (with up to 10
  offending targets named) if the flip's stage is earlier than that.
  Known gap, already noted in the validator's own docstring:
  `s.remove_element` is not modelled — a deck that removes every
  element using a material and then flips that material still passes
  V7 and silently no-ops at run time. Not worth the per-stage live-set
  machinery until a real deck hits it.

### `RO` variants excluded

`ManzariDafaliasRO` (and the `SAniSandMS` equivalent) are not typed
here. `ManzariDafaliasRO` is registered only in the Tcl builder
(`TclModelBuilderNDMaterialCommand.cpp:591`), never in the openseespy
command map — so a typed primitive for it would emit fine to `.tcl`
but fail on the `py`, `live`, and `run` targets (ADR 0008's four
uniform emit targets). Adding it would break INV-4 (Protocol shape
uniform across emitters) for exactly one material. Out of scope until
the fork exposes it to openseespy.

### H5 schema 2.20.0 → 2.21.0 (additive)

Each stage group gains an optional `(N, 2)` int64
`update_material_stage` dataset — one row per
`(material_tag, stage)` pair, in call order. Replay is wired through
[`_internal/compose.py::_replay_staged_into`](../../_internal/compose.py)
(the `from_h5` path) and both the flat and partitioned emit paths.
Per ADR 0023's two-version reader window, 2.20.x files simply lack the
dataset and replay nothing new; 2.21.x readers default to no
recorded flips when the dataset is absent.

### Partitioning

The `updateMaterialStage` line replicates into every MPI rank's
bracket — each rank is its own process with its own `static
mElastFlag`, so the flip has to be issued rank-locally. The H5 capture
still dedupes the authored call, so the same deck hashes identically
whether partitioned or not (same discipline as the rest of the
Phase SSI-2.E / ADR 0055 partitioned-staged work).

## Consequences

**Positive.**
- Staged soil-plasticity decks (gravity elastic → flip → push) are now
  buildable through typed primitives end to end — material, stage
  flip, H5 round trip — instead of a raw-`ops` escape hatch.
- The two schemes that kill the Python process outright
  (`SAniSandMS.int_scheme` in `{0,4,6,7,8,9}`) are unreachable through
  the typed API; a user can still hit them via raw `ops`, but not by
  accident through `ops.nDMaterial.SAniSandMS(...)`.
- V7 converts a silent, "reports success but does nothing" OpenSees
  runtime behaviour into a build-time `BridgeError` naming the
  offending stage and material tag.

**Neutral / forward-looking.**
- `STAGED_MATERIAL_CLASSES` and `SSP_UNSAFE_MATERIAL_CLASSES` are both
  small, named frozensets specifically so the `PressureIndepMultiYield`
  / `PM4Sand` family (still deferred — see `_DEFERRED.md`) and a future
  `LadrunoSANISAND` can join with a one-line addition, not a design
  change.
- `SAniSandMS.tol_r` is typed but currently unreachable: any non-default
  value raises `NotImplementedError`, because `SAniSandMS.cpp:134`
  computes `numData = numArgs - 19` although the command carries 19
  doubles plus the tag, so `TolR` is never consumed by the vanilla
  parser. The keyword stays so a future parser fix (`LadrunoSANISAND`)
  doesn't need an API change.

**Breaking — none.** Both materials and the mutator are new API
surface; nothing pre-existing changes shape. The H5 bump is additive
per ADR 0023.

## Cross-references

- [ADR 0008](0008-three-emit-targets.md) — the Emitter Protocol
  uniformity (INV-4) that excludes the `RO` variants.
- [ADR 0023](0023-per-zone-schema-versioning.md) — two-version reader
  window discipline for the 2.21.0 bump.
- [ADR 0029](0029-staged-analysis-context-manager.md) — the staged
  context manager `s.update_material_stage` extends.
- [ADR 0034](0034-stage-bound-bcs-and-recorders.md) —
  the V1–V5 stage-bound mutator validator family V7 joins.
- [ADR 0055](0055-staged-h5-archival.md) — staged H5 archival
  (flat + partitioned) that the `update_material_stage` dataset rides.
- [ADR 0035](0035-asd-embedded-node-element-option-exposure.md) — the
  precedent this ADR follows for exposing a C++ class's positional
  optionals as typed kwargs with an all-or-nothing tail.
- [`_DEFERRED.md`](../_DEFERRED.md) — the `updateMaterialStage` entry
  this ADR closes, and the still-open `PressureIndepMultiYield` /
  `PM4Sand` entry it does not.
