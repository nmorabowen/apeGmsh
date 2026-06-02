# ADR 0053 — Damping definition on the apeSees bridge

**Status:** Proposed (2026-06-01). Not implemented. Phased plan lives in
`internal_docs/plan_damping_integration.md` (slices D1–D5). Source-grounded
against the upstream OpenSees tree at
`C:\Users\nmora\Github\OpenSees_Compile\OpenSees` (file:line citations
inline below) **and** the fork's source+test-verified reference
`Ladruno_implementation/12_damping_channels.md` (8 passed / 1 xfail battery),
which settled several behaviours not visible from the code alone. Introduces
one new bridge namespace `ops.damping` plus one flat verb `ops.eigen`. All
four design questions are **resolved** (see end). No H5 schema bump in D1–D2
(Rayleigh records replay verbatim through the existing region/raw-command
paths); the tagged `Damping` primitive in D3 adds a new `/opensees/...`
sub-tree and **will** bump the bridge `SCHEMA_VERSION`.

**Namespace boundary — `ops.damping` owns 4 of OpenSees' 6 damping channels.**
The fork reference enumerates six: (1) element Rayleigh, (2) nodal Rayleigh,
(3) modal, (4) `Damping` objects, (5) material/dashpot (`Viscous`,
`ViscousDamper`, Maxwell…), (6) numerical/algorithmic (HHT-α, generalized-α,
bulk viscosity). `ops.damping` owns **1–4**. Channel 5 is a *material* and
stays in `ops.uniaxialMaterial.*` (it enters `R(u̇)`, not `C` — and is the
idiomatic way to damp a `zeroLength`). Channel 6 is an *integrator* property
and stays in `ops.integrator.*` (HHT, the fork `ExplicitBatheLNVD` FLAC
alpha, etc.). Naming this boundary keeps "damping" from implying one namespace
owns all of it.

## Context

The apeSees bridge has **no way to define damping coefficients today.**
The only damping-related surface that exists is:

- A per-element `do_rayleigh: bool` opt-in marker on `Truss`, `CorotTruss`,
  `ZeroLength`, `ZeroLengthSection` that emits `-doRayleigh 1`
  (`element/truss.py:85,108`, `element/zero_length.py:122,155`).
- The fork-only `ExplicitBatheLNVD` integrator's FLAC local non-viscous
  damping `alpha` (`analysis/integrator.py:447`).
- Optional per-material `eta` on `ElasticMaterial` (`material/uniaxial.py:591`).

There is **no `rayleigh` command, no region-scoped `-rayleigh`, no
`modalDamping`, and no `damping` (Damping-object) factory** anywhere in the
bridge. The `region(name=, pg=/nodes=)` verb (`apesees.py:4605`) exists but
emits only `region $tag -node …` for recorder filtering — its docstring
*claims* it is "useful for damping assignments," but it carries no damping
flags.

The consequence is upside-down: elements carry `-doRayleigh` opt-in flags,
but **there is nothing to opt into** — you cannot set the Rayleigh
coefficients the flags select for. Damping is effectively undefined on the
bridge at every scope above a single material.

This ADR establishes how damping is authored across all four OpenSees
mechanisms the user requires: global Rayleigh, region-scoped Rayleigh, modal
damping, and tagged Damping objects (`-damp`).

### Source cross-check (verified OpenSees command grammar)

All signatures below were read from the upstream parser source, not guessed:

- **Global Rayleigh** — `rayleigh αM βK βK0 βKc`. All four positional and
  **required** (`OPS_rayleighDamping`,
  `SRC/interpreter/OpenSeesMiscCommands.cpp:119`, guard
  `OPS_GetNumRemainingInputArgs() < 4` at :120; applies via
  `theDomain->setRayleighDampingFactors(...)` at :178). Carries an optional
  `-ele|-node tag…` filter (:141) — so even "global" Rayleigh can target
  explicit tags without a region.
- **Region** — `region $tag <-ele …|-eleRange e1 e2|-eleOnly …|-node …|`
  `-nodeOnly …|-nodeRange n1 n2> <-rayleigh αM βK βK0 βKc> <-damp $tag>`
  (`OPS_MeshRegion`, `SRC/interpreter/OpenSeesMiscCommands.cpp:784`).
  `-rayleigh` takes exactly 4 (:951), `-damp` takes exactly 1 tag (:976)
  resolved via `OPS_getDamping` and attached with `theRegion->setDamping(...)`
  (:1040). **Both Rayleigh and Damping-objects attach at the region.**
- **Modal damping** — `modalDamping f1 [f2 … fN]` and `modalDampingQ
  q1 [q2 …]` (`OpenSeesCommands.cpp:3215` / :3276). One factor → uniform
  across all modes; N factors → per-mode. **Both require a prior `eigen`
  call** (guard at :3226 / :3287). `modalDamping` reads inputs as damping
  *ratios* (ζ); `modalDampingQ` reads them as *quality factors* (Q = 1/2ζ).
- **`eigen`** — `eigen [<-solver>] N` (`OpenSeesCommands.cpp:2115`), solver
  options first, `numModes` last; default `genBandArpack`, generalized.
- **Damping-object factory** — `damping <Type> $tag …`
  (`SRC/interpreter/OpenSeesDampingCommands.cpp:394`). The registered types
  are exactly **four**:
  - `Uniform $tag $zeta $freq1 $freq2 [-activateTime ta] [-deactivateTime td]
    [-factor tsTag]` — note the stored coefficient is `zeta*2.0` internally
    (`OPS_UniformDamping:86`); the user still passes the **physical target
    ratio ζ**, OpenSees applies the factor of two. Not a footgun to correct,
    a convention to document.
  - `SecStif $tag $beta` (alias `SecStiff`) `[-activateTime …] [-deactivateTime …]
    [-factor …]`.
  - `URD $tag $N $f1 $z1 … $fN $zN [-tol] [-maxiter] [-factor] [-prttag] …`
    (N ≥ 2).
  - `URDbeta $tag $N $fc1 $b1 … [-factor] …` (N ≥ 2).
  - **There is no `damping Rayleigh` object** — the earlier guess that one
    exists is wrong. Rayleigh is reached only through the `rayleigh` command
    and `region -rayleigh`.
  - **Time-window flags** (all four types): `-activateTime $ta` (default 0.0),
    `-deactivateTime $td` (default 1e20), and a TimeSeries scale factor.
    These control *when the object dissipates energy* — the lever for "no
    damping during the gravity stage." **Tcl/openseespy divergence on the
    factor flag**: the Tcl parser accepts both `-fact` and `-factor`
    (`TclDampingCommand.cpp:120`), but the openseespy parser accepts **only
    `-factor`** (`OpenSeesDampingCommands.cpp:71`). The bridge therefore emits
    **`-factor`** on both backends (the fork reference's `-fact` is Tcl-only
    and would break an openseespy deck). `-activateTime`/`-deactivateTime`
    spell the same on both paths.
- **Element `-damp`** — `element <type> … -damp $tag`. **Per-element, not
  generic**: each element factory must parse it explicitly (present on
  Brick, DispBeamColumn2d/3d, ElasticBeam2d, …; absent on most). By contrast
  **`region -damp` is universal** — it works for any element the region owns.

### Three facts that shape the design

1. **The existing `-doRayleigh` flags are inert until D1 ships.** A global
   `rayleigh` command is the minimal change that gives them meaning.
2. **Region-scoped Rayleigh needs ELEMENT membership.** βK is
   stiffness-proportional, so the region must own `-ele`. Today's
   `RegionAssignmentRecord` is **node-only** (`_internal/build.py:642`,
   fields `name/pg/nodes`). Region damping therefore cannot reuse the
   node-only region path as-is — it needs a `-ele` membership route (pg →
   element fan-out, which `expand_pg_to_elements` already provides for
   `s.remove_element`).
3. **Prefer `region -damp` over element `-damp` for attaching Damping
   objects** — region attachment is universal; the element flag is supported
   by only a handful of element types and would need a per-primitive audit.

## Decision

### 1. One `ops.damping` namespace

Damping is a **domain-level** concern (sibling of `fix` / `mass` / `region`),
**not** part of the analysis chain (`constraints/numberer/system/test/`
`algorithm/integrator/analysis`). It is exposed as a single new typed
namespace `self.damping = _DampingNS(self)` instantiated alongside the other
namespaces (`apesees.py:4030–4039`). This was the ratified surface choice:
one discoverable home, consistent with `timeSeries` / `recorder`.

**Every member is a declaration, resolved at emit time** — uniform with the
rest of the bridge (`fix` / `mass` / `region` / `element.*` are all recorded
and rendered at `build`, nothing applies at the call). There is **no
`assign` step and no user-held object**: each method records a damping
declaration, and the emit-time resolver renders it to the right deck lines.
All members are lowercase declaration verbs:

```python
ops.eigen(n, solver=)                  # → eigen N   (flat verb, NOT in ops.damping)
ops.damping.rayleigh(...)              # → rayleigh OR region -rayleigh
ops.damping.modal(ratios=)             # → modalDamping (needs a prior ops.eigen)
ops.damping.uniform(..., on=)          # → damping Uniform  $t … + region -damp $t
ops.damping.sec_stif(..., on=)         # → damping SecStif  $t … + region -damp $t
ops.damping.urd(..., on=)              # → damping URD      $t … + region -damp $t
ops.damping.urd_beta(..., on=)         # → damping URDbeta  $t … + region -damp $t
```

There is **no `modal_q`** — its OpenSees command `modalDampingQ` is a
verified upstream anti-damping bug (§3). The user wanting Q can raw-emit; the
bridge will not author a wrong-signed line.

**The one uniform scope kwarg is `on=`** (a PG/region name, or a list of
them, or explicit element/node tags). It is the *only* thing that differs
across forms — and only in whether it is required, optional, or forbidden,
which is per-method validation, not a different API shape:

| Form | `on=` rule | Rendered |
|---|---|---|
| `rayleigh` | **optional** | absent → global `rayleigh αM βK βK0 βKc`; present → `region $t -ele … -rayleigh …` |
| `modal` | **forbidden** | always domain-wide (`modalDamping` has no region scope in OpenSees); fail loud if given |
| `uniform` / `sec_stif` / `urd` / `urd_beta` | **required** | `damping <Type> $t …` + `region $t -ele … -damp $t`; fail loud if absent (no global `-damp` exists) |

**Combine rules the resolver must respect** (from the fork battery — these
are not visible in the command grammar): element Rayleigh is **OVERWRITE per
element**, not additive — a `region -rayleigh` on an element *replaces* any
global `rayleigh` it had (verified ζ≈0.06, not 0.08). So the resolver emits
the global `rayleigh` **first**, then region forms, giving the intuitive
"region refines global" order; and it **warns** when an `on=` target overlaps
an element already covered by a global `rayleigh`, since a user expecting
summation gets replacement. By contrast `modalDamping` and `Damping` objects
are **ADDITIVE** on top of Rayleigh (easy to over-damp — a docs caution, not
a guard).

**Why no `assign`, and why this is faithful:** on the bridge, a damping
object never needs a Python handle — the emit-time resolver allocates its
tag and attaches it. The earlier "define then `assign`" two-step leaked an
OpenSees *runtime* distinction (`rayleigh` is a command; `damping <Type>` is
a tagged object) into the *authoring* surface, where everything is already a
deferred declaration. Collapsing it removes the asymmetry **and** an OpenSees
footgun: `rayleigh` is order-sensitive at run time (it binds whatever is in
the domain when the line executes), but as a declaration the user never
places the line — the resolver emits it at the correct point (§5 handles the
per-stage case). There is still no `damping Rayleigh` object (verified: only
Uniform / SecStif / URD / URDbeta), so `rayleigh`/`modal` simply resolve to
commands and the four type-verbs resolve to object-plus-attach; the user
sees one consistent "declare a damping rule" surface either way.

### 2. `ops.damping.rayleigh` — raw OR ratio, global OR scoped

```python
ops.damping.rayleigh(
    *,
    # raw form
    alpha_m: float | None = None,
    beta_k: float | None = None,
    beta_k_init: float = 0.0,
    beta_k_comm: float = 0.0,
    # ratio form (XOR with raw)
    ratio: float | None = None,
    f_i: float | None = None,          # Hz
    f_j: float | None = None,          # Hz
    stiffness: Literal["initial", "current", "committed"] = "initial",
    # scope — optional for rayleigh (absent = global)
    on: str | Iterable[str | int] | None = None,
) -> None
```

- **Raw form** passes the four coefficients straight through (the two
  `*_init` / `*_comm` default to 0.0 — the common case — but emit all four
  because OpenSees requires four positional args).
- **Ratio form** owns the classic two-target Rayleigh fit. With
  ω = 2πf and a single target ratio ξ at frequencies f_i, f_j:

  α = 2ξ·ω_i·ω_j / (ω_i + ω_j),  β = 2ξ / (ω_i + ω_j)

  α lands in `alpha_m`; β lands in the slot named by `stiffness` —
  `initial → beta_k_init`, `current → beta_k`, `committed → beta_k_comm`.
  **This is the one genuine engineering default, and it is `initial`**
  (βK0). The fork reference is explicit: for nonlinear runs prefer βK0,
  because the current tangent βK can vanish or go negative on softening and
  destabilize the run (and material yielding makes βK·K drift). apeGmsh is
  nonlinear/SSI-heavy, so `initial` is the safe default; `current` and
  `committed` remain explicit opt-ins. The switch is always explicit in the
  sense that the choice changes nonlinear results materially — the helper
  documents which slot it filled.
- **`on=`** is optional here (rayleigh is the one form that can be global).
  Absent → global `rayleigh αM βK βK0 βKc`. A PG/region name (or a list of
  them) → one `region $t -ele … -rayleigh …` per target, with `-ele`
  membership because βK is stiffness-proportional. Passing node tags with a
  non-zero β trips the OQ-1 guard (a node-only region silently drops
  stiffness-proportional damping in OpenSees).

### 3. `ops.eigen` (new flat verb) + `ops.damping.modal` (no `modal_q`)

`eigen` is general-purpose (modal analysis output, not only damping), so it
is a **flat bridge verb** — `ops.eigen(...)`, a sibling of `ops.fix` /
`ops.mass`, **not** a member of `ops.damping`:

```python
ops.eigen(n: int, *, solver: str = "genBandArpack") -> None   # → eigen <solver> N
```

Modal damping then **reuses** it instead of bundling its own copy:

```python
ops.damping.modal(ratios: float | Sequence[float]) -> None    # → modalDamping …
```

`modal` **requires a prior `ops.eigen(...)`** and fails loud if none was
declared (a clear "modal damping needs ops.eigen(N) first" — replacing the
old bundling's forgetting-protection with an explicit guard, and keeping the
eigen analysis as a single source of truth so it is never emitted twice). A
scalar `ratios` → one uniform factor; a sequence → per-mode, whose length
**must equal the declared eigen `n`** or the call fails loud. `on=` is
**forbidden** — `modalDamping` is domain-wide in OpenSees with no region
scope, so passing it fails loud. The resolver orders `eigen` → `modalDamping`
in a post-build / pre-`analyze` block.

**No `modal_q`.** The fork battery verified that `modalDampingQ` (the
force-only path, `inclMatrix=false`) applies damping with the **wrong sign** —
in free vibration it *amplifies* the response (measured ζ ≈ −ζ_target),
Δt-independent, under both Newton and Linear; `modalDampingQ(−ζ)` damps
correctly at +ζ. The fork decision (2026-06-01) is document-only: "use
`modalDamping`, never `modalDampingQ`." So the bridge **does not author
`modalDampingQ`** — a user who needs Q can raw-emit and own the risk. (See
OQ-3, now resolved.)

**Scope/over-damp cautions** (docs, not guards): modal damping only damps the
modes you computed — `eigen 3` + `modalDamping 0.05` leaves modes 4+ at zero
modal damping (high-frequency FE content is undamped by this channel, often
mopped up with a little stiffness Rayleigh); and modal damping is **ADDITIVE**
with any Rayleigh you set, so combining them sums (easy to over-damp).

Staged modal damping (per-stage eigen) is **out of scope for D4** and
deferred to D5 — modal damping interacts with `wipeAnalysis` and the
per-stage chain in ways that need their own design pass.

### 4. Object-backed forms — `uniform` / `sec_stif` / `urd` / `urd_beta`

These four are lowercase declaration verbs like every other member (§1), but
they resolve to a **tagged object plus an attachment**: a `Damping`
declaration (rendered as `damping <Type> $tag …`, tag allocated at emit) and,
for each target in `on=`, a `region $tag -ele … -damp $tag` line. `on=` is
**required** — there is no global `-damp`, so a damping object with no target
is meaningless and fails loud. The user never holds the tag or calls a
separate attach step:

```python
ops.damping.uniform(ratio=0.03, f_lo=0.5, f_hi=10.0, on="Soil")
# → damping Uniform $t 0.03 0.5 10.0   +   region $t -ele … -damp $t

ops.damping.uniform(ratio=0.03, f_lo=0.5, f_hi=10.0, on=["Soil", "Backfill"])
# → ONE damping Uniform $t …   +   region $t -ele … -damp $t  PER target
#   (the list form is the reuse path — one object, many attachments, no
#    repeated params, no handle to thread)
```

`uniform`'s `ratio=` is the **physical target ζ** — OpenSees applies the
internal factor of two; the bridge does not pre-divide. A `name=` is still
accepted for the named-primitive alias channel / deck readability even
though the user never references the object directly. These objects are
**ADDITIVE** with Rayleigh (independent channels), but a `region -damp`
overwrites an element's prior `-damp`.

**Time-window control — when the object dissipates energy.** Unlike Rayleigh
(which simply exists once set), a `Damping` object dissipates **from t = 0
unless windowed**, and that is almost always wrong for a staged model: it
would eat energy during the quasi-static gravity stage and corrupt the
initial state before the dynamic phase begins. All four types carry the same
three optional levers, exposed as plain kwargs:

```python
ops.damping.uniform(
    ratio=0.03, f_lo=0.5, f_hi=10.0, on="Soil",
    activate_time=t_dyn,        # -activateTime  : off until the shaking starts
    deactivate_time=None,       # -deactivateTime: default never (1e20)
    factor=ts,                  # -factor $tsTag : an ops.timeSeries.* object
)
```

`factor=` takes an `ops.timeSeries.*` object (the bridge resolves its tag and
adds it as a dependency, so the `timeSeries` line is emitted first). **The
bridge emits `-factor`, never `-fact`** — the latter is Tcl-only and would
break an openseespy deck (source cross-check above). A **staged-aware** sugar
— activate at a named stage boundary instead of an absolute time — is a **D5
horizon** (it rides on the staging work), not D3; the absolute-time kwargs
ship in D3 and already cover the gravity-then-dynamic case.

Element-flag attachment (`-damp` on a specific supported element, in place of
the region) is a **secondary D3b path**. The supporting element set is known
from the fork reference — `ElasticBeam2d/3d`, `DispBeamColumn`,
`ForceBeamColumn`, `Brick`, `FourNodeQuad`, the Shell family, `ZeroLength`
(any other element's base `setDamping` just warns) — so D3b's capability gate
is a fixed allow-list, not an audit, and fails loud on an unsupported
element.

### 5. Staging

Stage-bound damping (`s.damping.rayleigh(...)`, `s.damping.uniform(..., on=)`)
mirrors the existing `s.region` / `s.fix` pools and resolves inside the
owning stage's block. It is **deferred to D5**, after the flat (non-staged)
path is proven in D1–D4. The flat `ops.damping.*` verbs work in non-staged
decks from D1. The declaration model pays off here: the same verbs work at
both scopes, and the resolver — not the user — places each line correctly
relative to the stage's `domain_change` barrier.

## Consequences

- The inert `-doRayleigh` element flags become meaningful the moment D1
  ships — no change to the flags themselves.
- `region` keeps its current node-only identity job; D2 adds an **element**
  membership route used by damping, rather than overloading the existing
  node path. The two coexist (a region can carry `-node` for a recorder and
  `-ele` for βK).
- D1–D2 need no H5 schema bump (raw `rayleigh` lines and region records
  replay verbatim). D3's tagged `Damping` primitive adds a persisted
  sub-tree and bumps the bridge `SCHEMA_VERSION`.
- One namespace, one uniform shape: every member is a declaration with an
  `on=` scope (required / optional / forbidden per form). No `assign` verb,
  no user-held object, no immediate-vs-deferred asymmetry. The `on=`-list
  form is the reuse path (one object → many attachments). The resolver, not
  the user, places each line — which removes the `rayleigh` order-sensitivity
  footgun and lets the same verbs serve staged decks (§5).
- The ratio helper and the `stiffness=` switch put the only real engineering
  decision in the user's hands explicitly instead of burying it.
- Element Rayleigh **overwrites** per element (fork-verified). The resolver
  emits global `rayleigh` before region forms ("region refines global") and
  **warns** on global-vs-`on=` element overlap, so the OpenSees overwrite
  rule never silently surprises a user expecting summation.

## Resolved questions

1. **`on=` node tags + non-zero β — HARD ERROR.** The fork reference confirms
   a node carries *no stiffness*, so nodal Rayleigh stores `alphaM` only — a β
   on a node-only target is meaningless, not just lossy. β ≠ 0 with node-only
   `on=` raises; αM-only on nodes is fine.
2. **~~Standalone eigen~~ — RESOLVED.** Add a flat `ops.eigen(n, solver=)`
   verb (outside `ops.damping`, since eigen is general-purpose modal
   analysis); `ops.damping.modal` requires and reuses it (§3). No
   `ops.damping.eigen`.
3. **~~`modalDampingQ` surface~~ — RESOLVED.** Dropped. `modalDampingQ` is a
   fork-verified anti-damping bug (§3); the bridge will not author it. No
   `modal_q`.
4. **~~Ratio-helper default stiffness~~ — RESOLVED.** Default flips to
   `"initial"` (βK0), per the fork's nonlinear guidance; `current` /
   `committed` stay explicit opt-ins (§2).

All four are now closed; no open questions block D1.
