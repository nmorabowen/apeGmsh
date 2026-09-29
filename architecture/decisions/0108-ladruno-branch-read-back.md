# ADR 0108 — The `ladrunoBranch` read-back contract (fork ADR-95)

**Status:** Accepted (2026-09-08). Partially verified live: the deck, the
capability probe, the recorder's refusal and the reader's silence all ran
against the installed fork build `1652f945c`, which is **pre-fix** — see
"What was measured". The five post-fix assertions are written and gated;
they have not been run to green, and cannot be until the venv's engine is
rebuilt at or after `61b3efa04`.

**Sibling of** [ADR 0103](0103-sanisand-integrator-deck-contract.md) and
[ADR 0105](0105-asdplastic-deck-contract-after-ladruno-adr94.md) — same
shape a third time: a fork change becomes an apeGmsh **read-back**
contract. Where 0103 and 0107 turned fork integrator changes into deck
contracts, this one turns a fork *diagnostic* into a contract about what
eight unnamed columns mean.

**Fork-side sources** (branch `ladruno`, tip `9c2f964ea`):
`Ladruno_implementation/95_prandtl_reissner_quadratic_root_cause.md`,
`95_prandtl_reissner_campaign_report.md`,
`_adr95_corner_tangent_review.md`, `LEDGER_quirks.md` (the ADR-95 rows).
Fix commit `31322a47a`, merged as **`61b3efa04`**; the ASD-DP apex sibling
merged as **`67474aeb7`** (fork ADR-94 wp/94f, PR #832). apeGmsh working
notes: `internal_docs/guide_ladruno_adr95_druckerprager_fix.md`.

## Context

The vanilla UW `DruckerPrager` had a dead branch in its two-surface return
map: the tension-cutoff residual row was never assembled, so a Gauss point
crossing the cutoff kept an **unreturned** stress and a pathological
consistent tangent. On the fork's Prandtl–Reissner strip-footing deck that
killed every quadratic element on the step floor at 30–77 % of the collapse
load while the linear b-bar hex plateaued at the exact answer — a false
collapse that reads as a mesh or material problem.

The fork repaired the map and, with it, added a per-Gauss-point read-only
material response, `ladrunoBranch`, forwarded by `LadrunoBrick`,
`LadrunoBrick20`, `BezierTet10` and `TenNodeTetrahedron`:

```
eleResponse <ele> material <gp> ladrunoBranch
  -> [branch, gamma0, gamma1, f1_trial, f2_trial, forcedAccept, I1, detAmin]
```

`branch` is 0 elastic / 1 cone / 2 tension cutoff / 3 corner. `detAmin` is
the minimum over ~200 directions of `det(n · C_ep · n)` normalised by
`(2G)³`, so it is O(1) in the elastic range and O(−0.06) at an ordinary
yielded point.

The **recorder/file half was already adopted** ahead of this ADR (the
`61b3efa04` build floor, the recorder's refusal of the bare token, and the
eight by-position reader column names). What was outstanding is everything
that turns those columns into something a user can act on: the two derived
fields the fork's own campaign actually used, a capability probe, a live
gate, and this contract.

## Decision

### D1 — The response layout is the contract, and it is by POSITION

`DruckerPrager::setResponse` returns a bare `MaterialResponse(this, 95,
Vector(8))` with **no ResponseType**, so the `.ladruno` file always writes
`C1..C8`. The eight names in `_MATERIAL_BUCKET_TOKENS["ladrunobranch"]` —
`dp_branch`, `dp_gamma_cone`, `dp_gamma_cutoff`, `dp_f1_trial`,
`dp_f2_trial`, `dp_forced_accept`, `dp_i1`, `dp_det_a_min` — are pinned to
the fill site `DruckerPrager::getLadrunoBranch()` and are the **only**
authority on what those columns mean. Exactly the `implexGuards` situation
(ADR 92 P2-9).

Consequence: if the fork ever reorders the vector, or grows it, nothing in
the file says so. Two guards exist for that. `material_bucket_expected_names`
deliberately carries **no** entry for this token (there is no real spelling
to compare against, so a fabricated expectation could only false-positive),
and instead the width is checked at the live seam —
`probe_ladruno_branch` raises rather than answering on any reply that is
neither empty nor exactly eight long.

The names keep the house `<material>_<quantity>` shape rather than the
fork's raw `branch` / `gamma0` / `I1`. `branch` and `I1` are far too
generic for a flat Gauss-component namespace shared with `stress_*`,
`state_parameter` and `implex_*`; `dp_` says whose branch it is, which is
the whole of D3 below.

### D2 — The **material-bucket** seam, not `RESPONSE_CATALOG`

`ladrunoBranch` is a MATERIAL response. It reaches the file as one
`material.ladrunoBranch` bucket **already split per Gauss point**
(`gauss_id >= 0`), which `_block_canonicals` names by position from
`_MATERIAL_BUCKET_TOKENS`.

It does **not** go through `RESPONSE_CATALOG` / `resolve_generic_gauss_blocks`.
That pair is the ELEMENT-response seam: it exists to split the single
element-level `C1..Cn` block an untagged *element* response produces into
one block per Gauss point, using the catalog's `(class, int_rule, token)`
layout. A material bucket never presents as that single element-level
block, so a catalog entry for this token would be read by nobody.

**Rejected:** adding `("LadrunoBrick", Hex_GL_2, "ladrunoBranch")` and
siblings to `RESPONSE_CATALOG` anyway. Besides being dead on the file path,
it is actively hazardous: `_class_int_rule` scans **all** tokens for a
class and returns `None` when more than one non-`Custom` rule appears, so a
`ladrunoBranch` row at a rule that does not match the class's existing
`stress` row would silently drop *every* element of that class from
DomainCapture — including its stress. The catalog is the right seam only if
and when the live capture path is generalised (see D5).

### D3 — Two censuses, and only one of them is DruckerPrager's

`results.elements.gauss` gains `tension_census()` and `corner_census()`,
returning a `GaussCensus` — count, `examined`, the matching values, and the
`element_index` / `natural_coords` pair that `GaussSlab` carries, so
`global_coords(fem)` is the same reconstruction.

- **`tension_census()`** counts Gauss points with `mean_stress >= 0`, built
  from the ordinary `stress` response through the existing `mean_stress`
  derived scalar. It is therefore **material-agnostic** — deliberately, and
  it is the one the fork's campaign leaned on: the first tensile Gauss
  points beside a footing edge, which only quadratic elements resolve, are
  where every material on that deck gets into trouble.
- **`corner_census()`** counts `dp_branch == 3`, and is
  UW-DruckerPrager-only.

`>= 0`, not `> 0`: a point returned exactly to the apex sits at `I1 = T`
and is the whole point of the census.

Both take the standard `pg=` / `label=` / `selection=` / `ids=` / `time=` /
`stage=` selectors and are evaluated at the LAST step of the selected
slice — a census is a "how many, and where" at one instant. `examined`
travels beside `count` because a census of 12 is a different fact at 40
Gauss points than at 40 000.

**A missing component is refused, not counted as zero.** The reader answers
an absent Gauss component with an EMPTY slab; a census over that would
report `count = 0`, which reads as "no corner points" when the truth is
"never recorded" — and for `dp_branch` the second is exactly what a
pre-`61b3efa04` engine produces. `_census` checks `available_components()`
first and raises naming the component and the build floor.

### D4 — The empty reply IS the capability probe

`probe_ladruno_branch(ops, element_tag, gauss_point=1)` lands in
`opensees/_element_capabilities.py`. It calls the response on a live domain
and answers `True` only on the documented eight floats.

An engine older than `61b3efa04` — and every stock openseespy — parses a
`DruckerPrager` deck **identically** and answers `[]` with no exception.
That same engine also gets the return map wrong, so the empty list is not
merely "no diagnostic": it is "do not believe this run's collapse load on
quadratic solid elements". It is the cheapest probe there is, and unlike a
build-hash comparison it cannot be fooled by a stale `opensees.pyd`.

The probe knows the **width only**, never the eight names. Those live on
the read side (`results/readers/_ladruno_element_io.py`), and the
dependency runs results → opensees; duplicating the names into the bridge
to satisfy a probe would invert it and create a second place to get the
slot order wrong.

`LADRUNO_BRANCH_ELEMENT_CLASSES` records the four classes whose
`setResponse` forwards `material <gp> <token>` to the NDMaterial. Aiming
the probe anywhere else answers `[]` for reasons that have nothing to do
with the build, and the caller would misread it as an old engine.

**The branch codes are UW-DruckerPrager's, not a shared vocabulary.** No
other material writes this token; a `dp_branch` column on a results file is
evidence about *that* material and nothing else, and 0/1/2/3 must not be
read as a generic "plasticity state" on anything else. `ASDPlasticMaterial3D`
+ `DruckerPrager_YF` has its own, later floor (`67474aeb7`) for its apex
classification and exposes **no** response token at all, so it cannot be
probed this way — compare `get_backend_build()` instead.

### D5 — The LIVE capture path stays out of scope, with its seam named

`ops.eleResponse(e, "material", gp, "ladrunoBranch")` works today on a
post-fix build; what does not exist is a way to capture it through
`DomainCapture` into a `Results` without a file recorder.

That gap is real and is deliberately **not** closed here. `DomainCapture`'s
per-material routing is hard-coded to one token —
`needs_per_material_strain(class_name, catalog_token)` is
`catalog_token == "strain" and class_name in PER_MATERIAL_STRAIN_CLASSES` —
and generalising it needs, at minimum: the eight names in
`_vocabulary.ALL_CANONICAL` and in the capture spec's `gauss` category, a
routing pair in `_GAUSS_PREFIX_TO_KEYWORD` / `_KEYWORD_TO_CATALOG_TOKEN`,
`RESPONSE_CATALOG` rows at each class's **existing** `int_rule` (D2), and a
`per_material_token` on `_GaussClassGroup` because `_query_element` reuses
the catalog token as the material token. It also cannot be verified without
a post-`61b3efa04` build. That is its own slice, and it has one known trap
already: `LadrunoBrick20 -formulation uri` has 8 material points against
the catalog's 27, so the per-GP loop would run off the end — loudly, but it
would need handling.

Nothing in this ADR is blocked on it: the file route already carries every
column, and the censuses read from `Results`.

## What was measured

Fork build **`1652f945c`** (`ops.ladrunoBuild()`), i.e. **pre-`61b3efa04`**.
All of the following ran; the post-fix half could not.

- **The pre-fix defect, directly.** The ADR's own live-gate deck — one
  `LadrunoBrick` + `DruckerPrager` (φ_txc 20°, ψ = 0, σ_y = 0.2, ν = 0.45),
  every one of the 24 DOFs prescribed as a uniform hydrostatic expansion —
  ran six load steps to convergence and ended at **σ = 2.692 hydrostatic,
  I1 = 8.08**, against a tension cutoff of `√(2/3)·σ_y/ρ = 0.4487`. The
  stress was never returned to the cutoff: that *is* the dead branch, at
  18× the cutoff.
- **The probe.** `eleResponse(e, "material", 1, "ladrunoBranch")` → `[]`;
  `probe_ladruno_branch` → `False`.
- **The recorder.** `LadrunoRecorder warning: -E material.ladrunoBranch : 1
  of 1 LadrunoBrick element(s) returned no response`, then "matched no
  element in the model; nothing will be recorded for it" — and the
  `.ladruno` carries no `dp_branch` column, only the 18 stress-derived
  ones.
- **Both halves agree.** `test_pre_fix_engine_is_self_consistent` runs on
  ANY fork build and asserts the probe and the file tell the same story; it
  passes here on the "no" side.
- **The gate has teeth.** Mutating `probe_ladruno_branch` to answer `True`
  on an empty reply kills **6 of 6** tests in the live gate, including the
  one that passes today. (The first pass of the elastic-step test survived
  that mutation by comparing two empty arrays; it now pins the slab width
  first.)
- **Unit coverage, engine-free.** 9 census tests over a hand-written native
  results file (slot order, the `>= 0` boundary, selectors, the
  missing-component refusal, `global_coords` parity with the slab) and 11
  probe cases. 21 passed / 5 skipped across the three new files.

**Not measured, and it must be said plainly:** every assertion behind
`_require_post_fix` — the eight-float layout, `branch ∈ {2, 3}` with
`I1 ≈ T` on a driven tensile point, `branch == 0` with `detAmin > 0` on an
elastic one, the reader-vs-`eleResponse` slot check, and the census
cross-check. The `T = √(2/3)·σ_y/ρ` value is the guide's formula, not an
observation. Rebuilding the fork at `9c2f964ea` and re-running
`tests/opensees/integration_ladruno/test_ladruno_dp_branch_live.py` is the
close-out.

One prediction is worth recording so the first post-fix run can falsify it:
with zero deviator this cutoff plane **is** the cone apex, so a purely
hydrostatic tensile path should report branch **3** (corner), not 2. The
gate accepts either, and the census cross-check compares against the
engine's own sweep rather than a hard-coded count, so it stays honest
whichever the fork's active-set bookkeeping picks.

## Consequences

- **No signature changes to any `ops.*` / `g.*` primitive**;
  `_api_index.json` does not move. `DruckerPrager`'s arguments, validation
  and emitted line are untouched, as they were for the recorder half.
- **New public surface:** `apeGmsh.results.GaussCensus`, and
  `results.elements.gauss.tension_census()` / `.corner_census()`.
  `probe_ladruno_branch` / `LADRUNO_BRANCH_WIDTH` /
  `LADRUNO_BRANCH_ELEMENT_CLASSES` are bridge-internal
  (`opensees._element_capabilities`), documented in
  `docs/concepts/backend-capabilities.md` alongside the two build floors.
- **`tension_census` needs `stress` recorded**, nothing more, and works on
  any material and any build. Only `corner_census` is fork- and
  material-gated.
- **The census is one instant, not a history.** A per-step count would be
  cheap but was not asked for and is not built; `time=` selects which
  instant. The fork's own budgeting note stands either way: a per-Gauss
  census at every station multiplies the wall by ~5–8×, so sample at
  stations, not at steps.
- **Out of scope, already documented, no code owed:** the ADR-95 deck rules
  (Bernstein-consistent loads on Bézier elements per ADR 0091, prefer the
  b-bar variants, an unsymmetric solver for non-associated tangents) and
  the SANISAND guidance (IMPL-EX, `maxSubsteps` 1000, `-Pmin`) covered by
  ADR 0103.
- **Open:** D5's live-capture slice, and the live close-out above. Both
  need an engine at or after `61b3efa04`; the venv's is not.
