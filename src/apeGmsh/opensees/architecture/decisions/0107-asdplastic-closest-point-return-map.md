# ADR 0107 — The ASDPlasticMaterial3D closest-point return map (fork ADR-97)

**Status:** Accepted (2026-09-08). Verified live: the `ladruno_fork`
battery is **4/4 green against fork build `ff47275fd`** (the `origin/ladruno`
tip, which contains the ADR-97 closeout `7e93e4381`), and skips cleanly on
an older backend. Measurements in "What was measured" below.

**Extends:** [ADR 0105](0105-asdplastic-deck-contract-after-ladruno-adr94.md)
— same file, same `__post_init__`, same `Begin_Integration_Options` block.
0105's D3 built the client-side enum validation of `integration_method` /
`tangent_type` against the fork's post-ADR-94 token lists; this ADR extends
those two tables by one value each and adds the one pairing rule the fork
introduced with them. Sibling of
[ADR 0103](0103-sanisand-integrator-deck-contract.md) — same shape again: a
fork integrator change becomes an apeGmsh deck contract.

**Fork-side sources** (`Ladruno_implementation/` of the fork, branch
`ladruno`): `97_ladruno_asdp_closest_point_adr.md` (plan, decisions D1–D6,
implementation log), `reviews/adr97_verdict.md` (§0 verdict, §3 the P6
mesh-scale measurement and its default-flip recommendation, §4 owner items,
§5 residual risks), `LEDGER_implementations.md`, `LEDGER_quirks.md`, and the
parser itself, `SRC/material/nD/ASDPlasticMaterial3D/
OPS_AllASDPlasticMaterial3Ds.cpp`. Fork PR stack #817 → #819 → #824 → #825
→ #827 → #829 → #826 → the P7 closeout. apeGmsh working notes:
`internal_docs/guide_ladruno_asdp_closest_point.md`.

## Context

`Backward_Euler` — apeGmsh's only integrator until now, and the default of
all three D5 helpers — is an Ortiz–Simo cutting-plane map. **No tangent the
fork ever shipped for it is the tangent of that map.** Measured on the fork
against a central difference of the material's own committed response:
`Continuum` (apeGmsh's default since ADR 0105) 57 % off, `Secant` 80 %,
`Elastic` 103 %. Every `MohrCoulombSoil` deck we emit has therefore been
running a global Newton on a tangent that is merely *close*, which is why
those decks lose steps at large increments rather than merely iterating more.

Fork ADR-97 does not fix that tangent. It adds a **second, fully implicit
integrator** — `Closest_Point`, a true closest-point projection — together
with its own exact consistent tangent, `Algorithmic`. The pair is strictly
opt-in on the fork: `Backward_Euler`'s behaviour is untouched (fork D1,
pinned at 23 decks / 282 committed-stress rows), so **every deck apeGmsh
emits today is byte-identical on the new fork build.**

Three fork facts shape what reaches this codebase:

1. **`Algorithmic` is refused with any integrator but `Closest_Point`**
   (fork D2). This is a static fact about the deck's own two strings.
2. **`Closest_Point` covers 23 of the fork's 46 registered
   specializations** (fork D3) — VonMises, Drucker–Prager (apex included),
   Mohr-Coulomb, MohrCoulombTensionCutoff and Hoek-Brown, and only where
   the yield function and the plastic-flow potential are the SAME family.
   A mixed pairing, `StiffSoil` and `RoundedMohrCoulomb` are refused. This
   is *not* a fact about the deck's strings: it depends on compile-time
   markers inside the fork's templated
   `ASDPlasticMaterial3D<YF,PF,EL,IV>` instantiations.
3. **The four explicit integrators are now refused by default** (fork D5)
   unless the deck also passes `experimental_integrator 1`. Orthogonal to
   family support; it applies to a build regardless of which material.

## Decision

### D1 — Extend the two token tables; keep the defaults

`_ASDP_INTEGRATION_METHODS` gains `Closest_Point`, `_ASDP_TANGENT_TYPES`
gains `Algorithmic`. The methods table is additionally **split** into
`_ASDP_IMPLICIT_INTEGRATION_METHODS` (`Backward_Euler`, `Closest_Point`)
and `_ASDP_EXPLICIT_INTEGRATION_METHODS` (the other four), because the
implicit/explicit distinction is now load-bearing in two places rather
than being a synonym for "is or is not `Backward_Euler`".

**Defaults do not move.** All three D5 helpers keep
`integration_method="Backward_Euler"` and `tangent_type="Continuum"`. The
fork's own P6 measurement recommends flipping the fork's shipped default
(24000-DOF strip footing: `Closest_Point` 12/12 push steps in 78 s against
`Backward_Euler`/`Secant` 7/12 in 504 s, at equal-or-fewer Newton
iterations), but that flip is explicitly the fork owner's decision in a
separate PR and is **not** part of ADR-97's warrant. apeGmsh flipping ITS
default first would silently change behaviour for every existing
`MohrCoulombSoil` caller on a fork build that has not made that decision
either — and on a *pre*-ADR-97 build it would break every deck outright.
The preferred configuration is documented in the helper docstrings as
guidance for new models, which is where a recommendation belongs.

### D2 — Check the pairing rule, do NOT reimplement the support matrix

`tangent_type="Algorithmic"` without `integration_method="Closest_Point"`
raises `ValueError` in `__post_init__`, before any fork process is
touched. It is checked here because it is decidable from the deck's own
two strings, it is cheap, and the failure is otherwise a fork abort in the
middle of a long build.

Everything else in fork D3 is **deliberately not duplicated**: which
YF/PF/EL/IV combinations support `Closest_Point`, `StiffSoil`,
`RoundedMohrCoulomb`, mixed pairings. A Python table of the 23 supported
specializations would have to be kept in sync with a C++ template registry
it cannot introspect, and the fork's parser already fails loud naming the
exact YF/PF/IV and citing ADR-97 D3 — strictly more information than a
mirrored table could carry. Same reasoning as ADR 0105 D8. All three D5
helpers build matched pairs, so all three are inside the supported set by
construction; the risk is confined to hand-built generic decks, which is
exactly where the fork's message is the better oracle.

### D3 — The explicit-integrator warning names the gate

`ASDPlasticIntegrationWarning` is no longer "this is not
`Backward_Euler`"; it is "this is one of the four EXPLICIT methods". Two
consequences. `Closest_Point` no longer warns — it is a supported map, and
warning on it would train users to ignore the warning that matters. And
the message names `experimental_integrator` and
`ASDP_CLOSEST_POINT_MIN_BUILD`, because on a post-ADR-97 build the deck is
no longer merely unsupported, it is **refused**.

**The warning is NOT silenced by the opt-in**, and the message explicitly
says not to add that token unconditionally. An adversarial probe caught
the first draft doing the opposite, and it was a trap: `experimental_
integrator` is an unknown token to every parser before
`ASDP_CLOSEST_POINT_MIN_BUILD`, where ADR-94's fail-loud parser **rejects
the whole `nDMaterial` command over it**. So the draft advised a token
that breaks the deck on the build most users actually have, and then went
quiet once they took the advice. Being explicit — carrying no active
yield-drift correction — is a property of the METHOD, not of whether the
fork accepts the deck, so it warns every time.

No typed `experimental_integrator` field is added — no apeGmsh consumer
asks for the four gated methods, and the generic primitive's pass-through
already carries it for anyone who does (CLAUDE.md §2).

### D3a — A duplicated integration option is refused

D2 is this file's first CROSS-FIELD rule, and it was bypassable:
validation reads `dict(self.integration_options)` while `_emit` iterates
the sequence and writes every pair, so
`(("tangent_type", "Algorithmic"), ("tangent_type", "Secant"))` validated
as `Secant` and still emitted `Algorithmic` — the exact pairing D2 exists
to refuse — with no warning. The asymmetry pre-dates this ADR but was
harmless while every key was validated independently. A repeated option is
ambiguous in the deck regardless, so `__post_init__` now refuses it.

### D4 — `ASDP_CLOSEST_POINT_MIN_BUILD`, documented and not enforced

`"7e93e4381"`, alongside `ASDP_MIN_FORK_BUILD`. Not enforced, for the same
reason as its neighbour: a bare hash cannot prove ancestry. It is safe to
leave unenforced here because an older parser does **not** silently ignore
the token — `Closest_Point` is an unknown *value*, so the ADR-94 parser
aborts the `nDMaterial` command naming it. A deck built against a stale
backend fails loud rather than quietly running `Backward_Euler`. The
constant's job is to make the skip messages and the docstrings say which
build a user needs.

### D5 — `cp_iterations` is readable, and its shape was observed

The fork's new per-Gauss-point response (the local-Newton iteration count
of the last `Closest_Point` solve) needed one reader change. The recorder
passes any token through unaltered, but `_MATERIAL_BUCKET_TOKENS` in
`results/readers/_ladruno_element_io.py` is a deliberately closed table
(ADR 0105 Amendment 1), so an unmapped `material.<token>` bucket was
dropped with a `GaussColumnDroppedWarning`.

The entry was **observed, not guessed**, by recording a real
`Closest_Point` deck to a `.ladruno` on build `ff47275fd` and reading the
raw HDF5: the bucket token is literally `material.cp_iterations`, and it
is a SCALAR per Gauss point — `NUM_COMP` 1, `MULTIPLICITY` 1,
`FIBER_ID` -1, `SECTION_TAG` -1, one column per `GAUSS_ID`, `DATA` shaped
`(steps, elements, n_gauss)` against `stress`'s `(steps, elements,
6*n_gauss)`. It therefore maps to a single canonical name, `cp_iterations`
(following the table's `dp_cohesion` precedent: a name that collides with
no reader-derived quantity keeps its own spelling).

Semantics pinned live: **0 while the Gauss point is elastic, 1 once
Mohr-Coulomb yields** — the fork documents up to 5 for Hoek-Brown's curved
surface. The value is undefined under `Backward_Euler`, and the fork still
writes the bucket there, so a value read off a non-`Closest_Point` deck is
meaningless rather than absent. That caveat is documentation, not a gate:
apeGmsh cannot tell from the file which integrator produced it.

## What was measured

Live on fork build `ff47275fd` (the `origin/ladruno` tip; ADR-97 closeout
`7e93e4381` is an ancestor), via
`tests/opensees/integration_ladruno/test_asdplastic_closest_point_live.py`,
4/4 green:

- **`Closest_Point` is ~4 orders of magnitude tighter.** On the shared
  20-step deviatoric leg its committed states sit at `max|f_MC|` =
  **8.24e-13** against `Backward_Euler`'s **1.03e-08** — at the *same*
  global-Newton iteration count (20 vs 20).
- **The two maps agree to 0.82 %,** and only on one component. `sxx` and
  `szz` match to ~1e-11 relative; `syy` differs by 26.7 because
  `Closest_Point` pins `sxx == syy` exactly — the true
  triaxial-compression corner — where `Backward_Euler` splits them through
  its `MC_ds=1e-5` rounding. Compared per component against that
  component's own magnitude: against the strength scale `c*cos(phi)`
  (86.6) the same 26.7 would read as 31 %, which is why the test does not
  normalise that way.
- **`cp_iterations` reads 0 while elastic and 1 once MC yields,** recorded
  through a real `.ladruno` and read back as a named Gauss component.

**The one finding a caller will actually hit** — and it is ours, not the
fork's. apeGmsh defaults `strict_convergence=True` (ADR 0105 D2). Under
`Closest_Point` that **refuses this leg at step 1 of 20** (`analyze()`
returns `-3`): the fork checks a TRIAL state's yield residual against the
ABSOLUTE `f_absolute_tol`, and a mid-Newton trial reaches `f = 1.37e-06`
against the `1e-06` default. At kPa soil scale that threshold is ~1.6e-08
*relative* — tight enough that the more accurate map trips it where the
coarser one happens not to, even though `Closest_Point`'s committed states
are four orders of magnitude better. Pinned in
`case_strict_refuses_closest_point` so it is not rediscovered as a mystery
`-3`. No default was changed in response: the remedy is per-deck
(`f_relative_tol`, or strict off), and picking one for every caller is not
this ADR's warrant.

Skip behaviour is verified in both directions: green on `ff47275fd`,
cleanly skipped on `1652f945c` with the running build named in the message.
The gate is a capability probe, not a build-hash comparison, so it arms
itself on the first supporting backend.

## Consequences

- No signature changes; `_api_index.json` does not move (verified).
- No golden deck moves: a deck that does not ask for the new tokens emits
  byte-identically.
- `results/readers/_ladruno_element_io.py` gains one
  `_MATERIAL_BUCKET_TOKENS` entry (D5). A `.ladruno` carrying a
  `material.cp_iterations` bucket previously raised
  `GaussColumnDroppedWarning` and hid the column; it now surfaces as the
  Gauss component `cp_iterations`. Nothing else in the reader changes.
- A duplicated `integration_options` name is now a `ValueError` (D3a). No
  emitted deck could have relied on it: the fork sees every repeat too.
- One pre-existing test changed —
  `test_backward_euler_and_every_valid_token_are_silent` looped over
  `_ASDP_TANGENT_TYPES` under `Backward_Euler`, which D2 now refuses for
  one member. It becomes
  `test_implicit_integrators_and_every_valid_token_are_silent`,
  parameterised over both implicit maps and skipping the one pair D2
  covers directly.
- Open, inherited from the fork's own verdict §4 and affecting
  `Backward_Euler` decks only (i.e. everything we emit today): Hoek-Brown's
  flow direction collapses to a Tresca potential so `HB_mb_psi` is inert
  and flow is exactly non-dilatant; Mohr-Coulomb's analytic Lode-angle
  gradient is wrong at `MC_ds = 0` (our helpers default `ds=1e-5`, the
  better-behaved branch, by accident); Drucker–Prager cohesion hardening is
  inert. These are fork follow-ups, not apeGmsh work, and none is fixed by
  this ADR.
