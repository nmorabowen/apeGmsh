# ADR 0086 — Integral-mortar mesh-ties on composed assemblies: a weight *method* on `tie`, computed in apeGmsh, emitted as equation constraints

**Status:** Accepted (2026-08-04) — signed off same day. D1
`method="mortar"` and D3 flat-coincident v1 as recommended; D2
resolved as **"both"**: the numpy port ships now (this ADR's scope),
and extending the fork's `LadrunoMortarKernel`/`LadrunoTie` to
quad8/tri6 is a named follow-up work item on the fork side so the two
kernels can cross-check long-term. tri6 SLAVE facets are refused in
v1 (the classic dual-basis degeneracy: tri6 corner shape functions
integrate to zero over the parent triangle, so the per-facet rowsum
scaling divides by zero — quad8 corners integrate to −A/12 ≠ 0 and
are fine); tri6 is accepted on the master side.

The request: make an integral-mortar mesh-tie usable on an assembly
built with `apeGmsh.from_h5(host)` + `g.compose(...)` — because mixed
element ORDER across parts requires compose (`set_order` is global per
gmsh session), and the measured cost of collocation across an order
mismatch is real: hex20 ribs tied node-to-face onto hex8 plates
over-constrain the rib root (a = 0.195 vs 0.230 conformal reference,
K +7.7 %), and the only current fix — everything hex20 and fine —
costs +53 % solve time.

## The finding that reshapes the design

The request assumed "the fork already has the kernel — this is
plumbing on the apeGmsh side, not new mechanics." **Source-verified:
that premise fails for the driving use case.** The fork's `LadrunoTie
-mortar` (and the `contactSurface`/`contact -mortar -tie` route, and
the `LadrunoMortarKernel` beneath both) accept **tri3 and quad4 facets
only**:

* command parser: `ltReadFacets` rejects `nps ∉ {3,4}` before the
  generator runs (fork `LadrunoTie.cpp:1191-1193`);
* generator: collocation `LadrunoTie.cpp:348-351`, mortar `:592-595`;
* kernel: fixed-size `double X[4][3]`, `D[4][4]`, `M[4][4]`
  (`LadrunoMortarKernel.h:83-90`);
* contact route: same skip at `LadrunoContactHandler.cpp:559-561`.

A hex20 rib face is quad8. Neither side of the rib interface can be
expressed to the fork's mortar today — this is not a documented
deferral, it is out of scope there (no mention of quad8/tri6/midside
anywhere in `LadrunoTie.cpp`, the kernel, or fork ADR-62). Any design
that emits the fork's mortar commands either (a) restricts the feature
to linear meshes — useless for the problem that motivated it — or
(b) sub-facets quadratic faces into linear tris, degrading the slave
basis and adding a fork-only runtime dependency and a sixth+seventh
emitter Protocol method (ADR 0068 INV-2 makes every new emit verb a
six-emitter event).

## The second decisive constraint — handler exclusivity

The `contact` subsystem route (ADR 0073) is architecturally closed for
this use case regardless of facet support: `contact` forces the
`LadrunoContact` handler, which **cannot enforce `enforce="equation"`
ties or any MP constraint** (`apesees.py:5101-5119`; fork
`LadrunoContactHandler.cpp:415-421` warns and does not enforce
non-trivial EQ rows). The fuse model ties its other five interfaces
with `enforce="equation"` — a mortar bond expressed as contact could
never coexist with them in one analysis. `contact -mortar -tie` is
also penalty/ALM at run time (gap driven to `augTol`, never zero,
`dt_cr`-polluting, `-soft` refused with `-tie`) — the wrong mechanics
for a permanent weld. `contact` stays what ADR 0073 made it: genuine
gap/friction interaction. **A permanent bond is a tie.** (This answers
design question 1 in the direction the requester leaned, with the
stronger reason.)

## What mortar actually needs — and what the broker already carries

The fork's own mortar (`LadrunoTie::generateMortar`,
`LadrunoTie.cpp:585-907`) proves the data contract: facet connectivity
for both sides + nodal coordinates; the normal is **derived from facet
winding** and summed (`:644-649`), `-outward` only overrides, and a
cancelling normal (closed/curved surface) is a **named refusal**
(`:652-656`), not a silent zero. With `-dual` it condenses per facet to
a diagonal slave matrix and emits **per-slave-DOF `EQ_Constraint` rows
`u_s = Σ P_Ik · u_m_k`** at command time (`:891-895`) — the same
record family apeGmsh's `enforce="equation"` route already emits
(ADR 0068), enforced by the same handlers (Lagrange implicit /
LadrunoProjection explicit; Transformation unsupported).

On the apeGmsh side, the chain-phase router already resolves
`TiedContactDef` from **both-side face connectivity** pulled from the
broker's dim-2 element groups (`FEMDataSource.boundary_faces_for`,
`_chain_phase_router.py:480-481`) — pure numpy, no gmsh. A mortar tie
needs *exactly this data*: master facets + slave facets + coords.

**Conclusion: a dual-basis mortar tie is not a fourth `enforce=`
route and not a contact. It is a second way of computing the weights
of the `InterpolationRecord` that `resolve_tie` already produces.**
ADR 0068's invariant — "all three coupling targets consume the same
`InterpolationRecord`; only the emission differs" (`0068:73-74`) —
survives intact: collocation evaluates master shape functions at a
projected point; dual mortar evaluates `P = D_dual⁻¹ M` from overlap
integrals. Same record, same three `enforce=` emissions, same
handlers, same h5 schema (the `enforce` utf8 column is untouched; no
`_SR_ENFORCE_CODE` change), same chain-phase machinery.

## Decision (proposed)

**D1 — API: `method=` on `tie`, not `enforce="mortar"`, not
`contact`.**

```python
g.constraints.tie(master_label, slave_label,
                  method="mortar",          # "collocation" (default) | "mortar"
                  dofs=[1, 2, 3],
                  enforce="equation", ...)  # the three ADR 0068 routes, unchanged
```

`enforce=` selects *how the record is enforced* (element vs constraint
layer — ADR 0068's axis); `method=` selects *how its weights are
computed*. The two compose: `method="mortar", enforce="equation"` is
the calibrated production combination; the penalty routes accept the
same records unchanged (allowed, tested only via equation in v1).
Rejected spellings: `enforce="mortar"` conflates the axes (a mortar
tie still needs an enforcement; agent-verified asymmetries: `outward`,
face-face inputs, and `ngp` have no home in the enforce vocabulary,
and `_validate_tie_enforce`'s knob matrix would tangle);
`contact(formulation="mortar", tie=True)` is handler-exclusive with
the equation ties the same model needs (above). `tied_contact` keeps
its bidirectional-collocation meaning; `mortar()` stays the deprecated
ADR 0073 alias it already is.

**D2 — the weights are computed in apeGmsh (pure numpy), not by
emitting the fork's `LadrunoTie` command.** The kernel is a direct
port of the fork's verified algorithm (`LadrunoMortarKernel.h`:
Sutherland–Hodgman clip of slave∩master on the auxiliary plane,
fan-triangulate, Gauss order 4; `D_IJ += wJ N_I N_J`,
`M_IK += wJ N_I φ_K`; per-facet dual condensation
`A e = diag(rowsum Dᵉ)(Dᵉ)⁻¹`; the same four guards — coverage,
conforming-gap, near-zero-row, partition-of-unity `|ΣP−1| ≤ 1e-6`) —
**extended to tri6/quad8 slave and master facets**, which is the
entire point. The fork's oracles are the cross-check: linear patch
exact to ~1e-15 with dual (fork P2.1: lumped D errs 0.28, dual 2e-15 —
`-dual` is not optional, it is the only sane variant, so apeGmsh does
not expose a toggle). Emission is the **existing** equation route —
zero new emitter Protocol methods, decks work on any build
(`equationConstraint`), fork-only-ness applies only where it already
did (the explicit LadrunoProjection handler).

Why not emit `LadrunoTie -mortar` even for the linear-facet subset:
quad8 is the driving case (blocked, above); a new command means two
new Protocol verbs across six emitters (ADR 0068 INV-2); generation at
solver command time makes the tie invisible to the broker (no records
to inspect/count/persist — and "the tie resolves a non-zero number of
records" is acceptance criterion #1); and the fork requires the
command AFTER elements/masses exist (BLOCKER-2 mass scan,
`LadrunoTie.cpp:421-425`), an ordering constraint apeGmsh's deck
builder would have to thread. The fork kernel remains the reference
implementation and its numbers the regression targets.

**D3 — normals: derived from master facet winding, hard error when
ambiguous.** v1 scope is the tie case: **flat, coincident (gap ≤
tolerance), convex tri3/tri6/quad4/quad8 facets**. The accumulated
unit normal must be coherent (min per-facet dot with the mean ≥ 0.99)
or the resolver raises with the fork's own remedy text (supply
`outward=`, which `TieDef` gains as an optional field used only by
`method="mortar"`). Warped facets beyond a flatness tolerance and
non-convex facets are hard errors (the fork silently accepts warped
slave facets with ~0.7 % area bias — we refuse instead; loosen later
if a real model needs it). No silent zero-force path exists by
construction: zero resolved records is already fail-loud through
`Assembly.materialize`, and `_route_tie`'s empty-record silent return
gains a `method="mortar"` warning sibling.

**D4 — the silent-drop defect gets its fix here.** `contact()` never
calls `_add_def` — in chain phase nothing happens at all: no route
attempt, no counter bump, the def sits inert on `contact_defs` and the
model silently loses the interface (`ConstraintsComposite.py:399-418`
+ the `_mesh_queries.py:198-212` short-circuit). `contact()` gains a
chain-phase guard that raises `ChainPhaseError` naming the remedy
("contact needs a live gmsh session — declare it in the part session
before saving, or use tie(method='mortar') for a permanent bond").
That is the narrow fix; routing full contact through chain phase stays
out of scope (it genuinely needs geometry).

## Implementation plan (slices, each PR-able)

1. **S1 — the kernel.** `_kernel/resolvers/_mortar.py`: flat-facet
   dual-mortar integrator (clip, triangulate, Gauss; tri3/tri6/quad4/
   quad8 shape functions — the tri6/quad8 functions already exist in
   the collocation resolver; D/M assembly; per-facet dual
   condensation; the four guards). Pure numpy, no session. Unit tests
   against analytic overlaps + the fork oracle numbers (partition of
   unity ≤ 1e-6, linear patch ~machine-exact, lumped-vs-dual 0.28 vs
   2e-15 reproduction on the fork's own patch geometry).
2. **S2 — the def + record plumbing.** `TieDef.method` ("collocation"
   default | "mortar") + optional `outward`; validation matrix
   (mortar ⇒ dofs translations-only v1, `tolerance` reinterpreted as
   the coincidence gap tolerance, penalty-only knobs unchanged);
   `InterpolationRecord` untouched (weights are weights). H5: nothing
   — additive `method` provenance column only if inspection demands
   it (default: skip; the record is self-describing).
3. **S3 — resolution.** `ConstraintResolver.resolve_tie_mortar(defn,
   master_faces, slave_faces)` (build + chain share it, like
   `resolve_tie`). Build phase: slave faces via the existing
   `_collect_surface_faces` walk. Chain phase: `_route_tie` branches
   on `defn.method` and pulls `boundary_faces_for` for BOTH sides
   (the `TiedContactDef` pattern; inherits the "re-extract with
   `dim=None`" hard error).
4. **S4 — fail-louds.** D3's normal/flatness/convexity errors; D4's
   `contact()` chain-phase `ChainPhaseError`.
5. **S5 — surface plumbing.** `Assembly.couple` passes `method=`
   through `**options` (already does — verify + test);
   `_validate_tie_enforce` untouched.
6. **S6 — acceptance.** Extends `tests/test_meshable_part_route.py`'s
   rig: two blocks meshed independently at different orders (hex20 /
   hex8), saved, composed, mortar-tied across the non-matching
   interface, `nu = 0`: (i) non-zero record count asserted; (ii) K
   within tolerance of `EA/L/2` — target 0.1 % (fork mortar patch
   tolerance is 1e-5·target; quadrature makes mortar measurably looser
   than collocation, by design); (iii) **order-swap symmetry**: the
   same K with master/slave exchanged — the property being bought;
   (iv) the existing collocation tests unchanged.
7. **S7 — docs** (how-to tie section + compose-modules note) and the
   Cerro Lindo real check: hex20 ribs + hex8 plates, `a` should move
   from 0.195 toward 0.230 — run by the user on the fuse model, not
   in CI.

## Consequences

* Order-mismatched mortar ties become expressible, on composed
  assemblies, with quadratic faces — none of which the fork command
  route could deliver.
* The mortar math lives in apeGmsh numpy: it must be validated against
  the fork's oracles (S1 does), and a future fork kernel improvement
  does not flow in automatically. Accepted: the alternative is a
  feature that cannot handle quad8 at all.
* `dofs` stays translations-only for mortar v1 (the fork's rotational
  and Hermite transfers are collocation-only there too); shell-edge
  mortar and mortar-Hermite remain fork-deferred and apeGmsh-deferred
  alike.
* The `contact` chain-phase silent drop becomes a loud error (D4) —
  strictly a bug fix, but noted as a behaviour change.

## Cross-check against the fork kernel (2026-08-04) — D2 closed

D2 promised the two kernels would cross-check. The fork side wrote the
contract down as ADR-78's companion
(`78_apegmsh_mortar_crosscheck_requirements.md`, R1–R8); this is what
came back when we ran it.

**R7, the deliverable, passes.** On the agreed patch — a 2×2 quad8
slave (21 nodes) on a 3×3 quad4 master (16 nodes) over [0,1]² — the
numpy kernel reproduces the fork oracle's `‖P_dual‖_F = 3.927978688773`
to 5e-13, and every weight in the reference corner row to the last of
its 9 quoted digits. It is now a regression test
(`test_q4_crosscheck_matches_fork_oracle`), so the numbers are pinned in
both repos as R7 asked. One correction for the fork side: the reference
row's master numbering is the oracle's `nid` **creation** order, not
row-major — creation order is what makes the row symmetric under x↔y.

**Two contract items were genuinely missing, and both were silent.**

* R2's midside-at-midpoint guard. All geometry runs on the corner
  polygon, which is exact only because the serendipity map collapses to
  the corner map for straight edges. A quad8 midside slid 10 % along its
  own edge was accepted with a **0.30** linear-patch error and every
  other guard clean. Now a hard error naming the facet and the edge.
* R5.3, the fork's flagged MAJOR finding: overlapping master facets.
  Coverage sums pair-clip areas, so it counts multiplicity — two
  coincident masters over the left half of a slave facet and nothing
  over the right half read as 100 % covered. The uncovered half was then
  silently *extrapolated*, not left free, which is why nothing
  downstream could see it: `Σw = 1` on every row (dual rowsums are
  identically 1) and the linear patch passed at 1.6e-14, because a
  linear field extrapolates exactly. Now refused before assembly.

**R8's test-honesty items are in.** There is a linear-patch assertion
through each master type — quad4, quad8 and now tri6 — plus a live
mutation test that corrupts the tri6 basis and asserts the patch
assertion catches it. Worth noting against R8's prediction: a *permuted*
tri6 midside ordering, the case it says partition-of-unity provably
cannot catch, is now caught structurally by the R2 guard, since permuted
midsides are no longer at their edge midpoints.

**Three requirements are met differently, on purpose** (recorded in the
kernel docstring, which is the operative version): R3's quadrature —
this kernel's Duffy 5×5 rule is exact wherever the fork's 12-point
Dunavant rule is, which is why R7 matches at all, but the two will
diverge on skewed facets where neither is exact; R5.2's conforming-gap
L1 measure, subsumed by the stricter v1 flat-and-coincident scope and
which must return if that scope is ever lifted; and R6's export-side
`LadrunoBrick20(lumped=True)` requirement for mortar-tied hex20 under
`LadrunoProjection` — the knob exists and is documented on the element,
but nothing verifies the combination yet.

## Questions for sign-off

1. **API (D1):** `tie(method="mortar")` over `enforce="mortar"` —
   accept the reframing? (Your instinct said enforce=; the
   investigation says mortar composes WITH enforce="equation" rather
   than replacing it.)
2. **Kernel home (D2):** apeGmsh-side numpy port (quad8-capable,
   emission unchanged) over emitting the fork's `LadrunoTie` (blocked
   at tri3/quad4 for your ribs)? If you'd rather extend the fork
   kernel to quad8 first and keep apeGmsh thin, that is a fork ADR +
   C++ effort with its own timeline — state the preference.
3. **v1 scope (D3):** flat + coincident + convex facets, hard error
   otherwise — sufficient for the fuse (all weld ties are coincident
   planes)?
