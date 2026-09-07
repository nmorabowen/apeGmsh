# apeGmsh ← Ladruno ASDPlasticMaterial3D: the fork-side change request

> **STATUS (2026-09-07).** Two asks, neither filed yet. apeGmsh ADR 0105 is
> Accepted against fork build `3622d6214` with both worked around on the
> apeGmsh side (a hand-maintained schema table pinned by a 46-combination
> fixture; a hand-copied refused-integrator list). Nothing here is
> implemented on the fork; this page is the request.

Fork ADR-94 made the `ASDPlasticMaterial3D` parser fail loud, which is what
apeGmsh wanted — a typo'd `MC_phi` no longer runs at φ = 0. It also moved two
facts apeGmsh now has to carry by hand: the per-combination parameter schema
and the list of refused integrators. Both drift silently when the fork
changes, and the tripwires apeGmsh keeps (the unit test over the 46
registered combinations, the live battery) catch the drift only after it
has broken a deck. The bar for an ask is the same as in
`contact_2d_fork_asks.md`: "apeGmsh cannot get this any other way".

Sources are the fork's `OPS_AllASDPlasticMaterial3Ds.cpp` (the `list` verb at
`:161`, the parameter check at `:429-445`, the required-parameter check at
`:642-676`, the refusals at `:550-585`), the component headers'
`using parameters_t` tuples, and `_adr94_inventory.md` §(b).

## A. The change request

### Ask 1 — `nDMaterial ASDPlasticMaterial3D <tag> list -params`

**The problem.** The `list` verb prints the four type strings of every
registered combination and nothing else. The parameter names each
combination requires are known only to the C++ templates
(`getParameterNames()` on an instantiated material), so a deck author — or
apeGmsh — cannot discover them without instantiating and reading the
refusal message. apeGmsh carries the schema as a table
(`_ASDP_PARAMS_BY_COMPONENT` in `material/nd.py`), read once from the
component headers, composed as `EL ∪ YF ∪ PF ∪ (one set per IV hardening
policy) ∪ {MassDensity, InitialP0}`. It was verified against the
parser's own `getParameterNames()` for the 43 combinations it covers (the
StiffSoil family is left to the fork), but only at one build; a new
component, a renamed parameter, or a policy that gains a parameter is
invisible until a deck is refused.

**The ask.** A `-params` (or `-schema`) flag on the `list` verb that prints,
per registered combination, the four type strings AND the parameter names
`getParameterNames()` returns, with the two optional names marked — for
example:

```
YF = MohrCoulomb_YF
PF = MohrCoulomb_PF
EL = LinearIsotropic3D_EL
IV = BackStress(NullHardeningTensorFunction):
PARAMS = MC_phi MC_c MC_ds MC_psi YoungsModulus PoissonsRatio [MassDensity] [InitialP0]
```

One block per combination, on `cout` like the rest of `list`, so
`python -c "ops.nDMaterial('ASDPlasticMaterial3D', 1, 'list', '-params')"`
captured through a pipe is a complete schema dump. apeGmsh would then
GENERATE and pin `_ASDP_PARAMS_BY_COMPONENT` from that output instead of
maintaining it, and the fixture would carry the names, not only the types.

**Cost on the fork.** The names already exist — the refusal path prints
them (`:436-440`) — so this is the `list` loop calling the same
`for_each_in_tuple(getParameterNames(), …)` on each factory instance.
De-duplicating repeated names (see the observation below) is optional but
would make the output a clean set.

### Ask 2 — a Python-visible constant for the refused-integrator list

**The problem.** `Backward_Euler_LineSearch` and
`Runge_Kutta_45_Error_Control_old` are refused by name inside the parser
(`:550-585`) with a reason string each. apeGmsh refuses them client-side
(`_ASDP_REFUSED_INTEGRATION_METHODS`, with the fork's reasons copied) so a
deck fails at build time rather than at run time. If the fork refuses a
third integrator, apeGmsh keeps emitting it and the deck fails one step
later, in the fork's words, not apeGmsh's.

**The ask.** Expose the list from Python — any of:

- a module attribute on the `.pyd`, e.g.
  `opensees.ASDPLASTIC_REFUSED_INTEGRATION_METHODS = ("Backward_Euler_LineSearch", …)`;
- a verb, `nDMaterial ASDPlasticMaterial3D <tag> list -integrators`, printing the
  accepted and the refused tokens with their reasons;
- the same through `ops.ladrunoBuild()`'s sibling query surface, if one is
  planned.

apeGmsh would probe it once per session (the way it probes
`criticalTimeStep` for the backend name) and fall back to its own copy on
older builds.

**Cost on the fork.** The tokens and reasons are string literals in one
function; the constant is a table those branches read from.

## B. Not asked, on purpose

- **`stdBrick` / `BrickUP` / `QuadUP` swallowing material refusals (B2).**
  The fork left it by design and pins it (`test_R2_strict_convergence_is_a_
  noop_on_stdbrick`). apeGmsh keeps its build-time warning
  (`validate_asdplastic_host`, ADR 0105 D4) until the fork decides
  otherwise; it is not asking the fork to change vanilla elements.
- **Corner return on the Hoek–Brown / tension-cutoff composites.** The
  fork records it as its own follow-up (wp/94d docstring); apeGmsh's
  battery documents the behaviour (strict on: the corner step is refused;
  strict off: the plateau is committed) rather than asking for a fix.

## C. Observations from the apeGmsh battery (not asks)

- The required-parameter message lists a name once per DECLARING
  component: a deck missing `MC_phi` is refused with
  `2 required model parameter(s) were never given a value: MC_phi, MC_phi`
  (`MohrCoulomb_YF` and `MohrCoulomb_PF` both declare it). Cosmetic; the
  count is misleading.
- A bare `-E pstrain` (or any other material-level token) records nothing,
  silently — no warning, no bucket. The `material.pstrain` spelling works
  and is now the only one apeGmsh emits. The recorder's "silent-drop
  diagnostics" counter (`num_request_answers`) could name the token in a
  warning when it stays at zero.
