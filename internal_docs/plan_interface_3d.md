# `interface()` in 3D — scope only (TIMs A10)

**Status (2026-09-08):** S1, S2 and S3 built; only S4 (verification) is
open. A 3D surface master resolves into records AND emits a complete
deck — the three gates in the table below are gone, and so is the
emission refusal S2 left standing. **Blocker 1
(F1) is cleared** — fork #808 / ADR 96, minimum build `a240b9183`
(`TIMS_FORK_BATCH_MIN_BUILD`): `zeroLength` accepts any 3-D pair with both
ndf ≥ 3, acts on DOFs 1–3 only, only `-dir 1 2 3` exist on a mixed pair, the
`force` response is element-sized (`ndf1 + ndf2`), and the differing-dof
case is a warning plus an inert element, not a crash. Contract recorded in
`contact_3d_passenger_dof_adoption.md`. **Blocker 2 is still open** and is
entirely ours; the three refusal gates below still fire. Nothing here
changes code.

## What the model needs

Rung 1 of PM-01's footing interface slot: per coincident pair between the
ndf-3 footing skin and the ndf-4 u–p soil, one `zeroLength` with an `ENT`
normal law and a Coulomb shear law — `g.constraints.interface()` (ADR 0093)
as it exists in 2D, on a 3D surface master.

## Where it was refused before S2 (verified 2026-09-07; all three lifted 2026-09-08)

| gate | where | what it says |
|---|---|---|
| declaration | `core/ConstraintsComposite.py:1169-1191` `_refuse_3d_interface` | `gmsh.model.getDimension() == 3` → `NotImplementedError`, cites ADR 0093 D2 |
| resolve | `core/ConstraintsComposite.py:1217-1221` `resolve_interfaces` | `model_dim != 2` → `NotImplementedError` |
| resolve | `:1236-1240` | any master entity of dim ≠ 1 → `NotImplementedError` (surface masters deferred, D2) |

## The two blockers, in dependency order

1. **Fork — `ZeroLength::setDomain` (F1).** `SRC/element/zeroLength/ZeroLength.cpp:611-673`
   requires `dofNd1 == dofNd2` (`:615`) and, in 3D, accepts only ndf 3
   (`:659`) or ndf 6 (`:665`); an ndf-4 u–p node hits the error at `:673`.
   In 2D the bridge sidesteps this with the D4 phantom bridge (a phantom at
   the LOWER ndf + nested `equalDOF`), which works because both sides carry
   DOFs 1–2 as translations. In 3D the soil side is ndf 4 with DOF 4 = pore
   pressure: a phantom at ndf 3 against a 4-DOF node is still a mismatch for
   `:615`, and a phantom at ndf 4 would give the spring a pressure "DOF" it
   must not touch. The relaxation asked of the fork is: accept `dofNd1 ≠
   dofNd2` in 3D as long as both are ≥ 3, and act on DOFs 1–3 only (a
   count-blind element would repeat exactly the A1 failure: tying pressure
   as if it were a rotation). Until that lands, no apeGmsh-side emission
   can be correct — approximating it (e.g. an ndf-4 phantom + a 4-DOF
   `zeroLength` with a dummy material on slot 4) would silently couple the
   soil's pore pressure to the skin.
2. **apeGmsh — ADR 0093 D2/D3 in 3D.** Per-pair outward normals from the
   adjacent surface facets (the 3D analogue of the edge-normal average with
   the centroid sign-fix; corner/edge nodes averaged and renormalised, fail
   loud on a reentrant fold), and a surface tributary model (`A_trib` from
   the facet-area accumulation — the sibling of `_tributary_areas` that
   `distributing_coupling(weighting="area")` already uses — with no
   `thickness` kwarg in 3D). The Coulomb shear law needs TWO in-plane
   tangent directions per pair (2D has one), so the material bundle per
   pair becomes normal + t1 + t2 with the `zeroLength -orient` frame
   emitted per pair.

## Slices once F1 lands (each PR-able, in the ADR 0093 register)

- S1 — kernel: 3D per-facet frames + surface tributary. **Done
  2026-09-07** — `surface_frames()` in
  `src/apeGmsh/_kernel/geometry/_surface_frames.py` (the geometry
  package, not the resolver: that is where the 2D `edge_frames` was
  lifted to for the same reuse reason), tested in
  `tests/_kernel/geometry/test_surface_frames.py` on the flat patch, the
  box corner, the reentrant fold, mixed tri3/quad4 and a flipped
  winding. The three gates below still fire — lifting them is S2.
- S2 — composite: lift the three gates above for dim-2 masters; the D4
  phantom bridge parameterised by the fork's accepted ndf pairs — `(3,4)`,
  `(4,3)`, `(4,4)`, `(3,6)`, `(6,4)` on builds ≥ `a240b9183`; refuse below it.
  **Done 2026-09-08** — all three gates lifted; a 3D surface master
  resolves, `thickness` is refused by name, `slave_ndf` takes
  `(None, 3, 4, 6)`, and no phantom is minted (the accepted pairs need
  no bridge — the parameterisation IS the retirement of D4 in 3D). The
  record's `orient` widens to nine floats, `(n, t1, t2)`, on an appended
  h5 column (neutral 2.32.0) that leaves a 2D row untouched. Emission
  still refuses, loudly and by S3's name. Details in the ADR 0093
  register, entry 13.
- S3 — emit: per-pair `-orient` with two tangents; the Coulomb law as the
  existing 2D material bundle plus the second tangent; MPCO/Ladruno
  recorder channels per pair as in 2D. **Done 2026-09-08** —
  `-mat mN mT mT -dir 1 2 3 -orient n t1` per pair. `-orient` carries
  only the record's first six floats because the engine derives local-3
  as `1 x 2`; that the record's `t2` equals `n x t1` is asserted (1e-9)
  before a line is written, not trusted. The tangential tag is repeated
  on dirs 2 and 3 rather than minted twice — `ZeroLength` deep-copies
  every `-mat` slot, so the sliders are independent from one material.
  Those two sliders are **uncoupled**: each carries the full
  `tau_b * A_trib`, so the slip locus is a square in the tangent plane,
  up to sqrt(2) too strong on the diagonal. That is this plan's own
  choice; **S4 owes the measurement of what it costs**. The emit-time
  ndf gate imports the resolver's `_ACCEPTED_3D_NDF_PAIRS` (one table,
  no drift), and the generic zeroLength-family endpoint guard now
  accepts unequal ndf iff `ndm == 3` and both ends are >= 3. Recorder
  side needed nothing: `n_springs` comes from `META/NUM_COMPONENTS`, so
  three springs read back as `spring_force_0..2`. Details in the ADR
  0093 register, entry 14.
- S4 — verification: the 2D ADR 0093 convergence case rotated into 3D
  (a strip footing on a plane-strain slab meshed as one element deep must
  reproduce the 2D result), then a u–p soil with the pressure datum (A2)
  showing the pore pressure is untouched by the interface. S3 adds one
  item: measure the square-vs-circular slip locus of the two uncoupled
  tangential sliders on a case that slides off-axis.

## Runway (named, not fixed here)

- `viewers/diagrams/_spring_force.py` deduces a spring's arrow direction
  from the canonical suffix as a GLOBAL axis (`0 -> x`, `1 -> y`,
  `2 -> z`). The interface springs act along the pair's own `(n, t1, t2)`
  frame, so a 3D interface's `spring_force_1` arrow points along global
  y rather than along `t1`. `SpringForceStyle.direction` overrides it per
  diagram; reading the frame off the record is not S3's business.

## Not asked / explicitly out

- A mortar or `contact()`-based 3D interface — already available and is rung 2.
- Any approximation that lets the deck emit before F1.

## Adversarial review (2026-09-08)

Probe-driven review of S1-S3 by a reviewer who wrote none of it: every
row below was **executed**, not read off the diff. Fork binary
`C:\Users\nmb\Documents\Github\OpenSees\dist\bin\OpenSees.exe`,
build `1652f945c` (ADR 96). One defect (F1) confirmed and fixed; the
rest are non-findings, kept because a hypothesis discarded on evidence
is part of the result.

| # | hypothesis | probe | measurement | verdict |
|---|---|---|---|---|
| 1 | tilted interface: rotation invariance of the frame, `A_trib` and the answer | both boxes rotated 37 deg about `(0.3,-0.7,0.5)` via `g.model.transforms.rotate` before meshing; records compared to the flat model, then both decks run and the cap displacements compared | `max abs(n - R z) = 2.8e-16`; triad orthonormal to `2.2e-16`; `n x t1 - t2` exactly 0; `sum(A_trib)` `1.0` vs `1.0` (delta `1.1e-16`), per-node share delta `2.8e-17`; both decks 0 fork refusals; `max abs(u_rot - R u_flat) = 7.0e-11` on `u ~ 9.8e-5` (rel `7e-7`, at the recorder's own 6-figure output width) | DISCARDED - the frame is rotation-invariant |
| 2 | non-uniform master mesh mis-weights `A_trib` | transfinite `n=(5,3,3)`: a 4x2 division of the unit face | 15 pairs, closure error exactly `0.0`; shares `{0.03125 (corner), 0.0625 (edge), 0.125 (interior)}` = the hand-computed quarters of the `0.25 x 0.5` cells | DISCARDED |
| 3 | tri3 master facets (tet soil + tet footing, `recombine=False`) | same fixture, `set_transfinite(..., recombine=False)`, deck run | 9 pairs, closure error exactly `0.0`, shares `1/24` / `1/12` / `1/8` / `1/4`, deck runs with 0 refusals | DISCARDED |
| 4 | corner-wrapping / reentrant masters | not reached through the verb: a master spanning two box faces needs a slave that is ONE mesh across the corner, which the un-fragmented two-body fixture cannot express (the two slave bodies put two nodes at the corner and the resolver's ambiguity refusal fires first). The kernel cases (box corner, reentrant fold) are already pinned in `tests/_kernel/geometry/test_surface_frames.py` | - | NOT PROBED end-to-end; the kernel half is covered, the composite half belongs to S4 |
| 5 | non-coincident sides pair partially and silently | soil `n=3` against footing `n=4` | `ValueError` naming all 12 unmatched SLAVE nodes and the count (`12 of 16`), pointing at `master_entities=` / `tie` / `contact` | DISCARDED - refused by name, no partial pairing |
| 6 | quadratic master facets get the linear tributary rule | `g.mesh.generation.set_order(2)` | `NotImplementedError` naming the facet as a `quad9` and ADR 0093 D3 | DISCARDED |
| 7 | h5 round-trip loses the 9-float frame; an in-window 2.31.x file breaks | `to_h5` -> `from_h5` on a 3-D model; then a 2.31.x-shaped payload row (the `orient_t2` / `has_orient_t2` columns removed from the dtype) fed to `_decode_interface` | round-trip float delta exactly `0.0` on `orient` and `a_trib`, widths stay 9, laws and names identical; the trimmed row decodes to a 6-float orient via the reader's presence probe | DISCARDED |
| 8 | `g.compose` mangles the 3-D triad (only 2 of 3 vectors rotated) | 3-D interface module composed into a plain host with `translate=(0,7,0)`, `rotate=(1,0,0,pi/2)` | 9 records, widths 9, `n` = `(0,-1,0)` = the expected rotated `+z`, `abs(n x t1 - t2) = 0`, orthonormality error `0` | DISCARDED |
| 9 | stage claim / partitioned emit drop or duplicate a 3-D pair | `s.interface(name=)` inside an `install` stage; then `partition(2)` flat and `per_rank=True` | staged: all 9 units inside the `install` block, once each, before its `domainChange`; partitioned: 9 unique `zeroLength` lines, **sorted set identical to the flat deck's**; per-rank: all 9 on rank 1 (`ranks/rank1_0.tcl`), none on rank 0, no duplicates | DISCARDED - but neither route had a 3-D test; see below |
| 10 | the emit gate refuses an ADR 96 pair the fork accepts | 13 hand Tcl decks, one per `(ndf1, ndf2)`, each with `-mat/-dir 1 2 3 -orient` and a trailing `puts MARKER_RAN`; then the same matrix through the verb | the fork runs `(3,3) (3,4) (4,3) (4,4) (3,6) (6,4) (4,6) (6,6) (6,3) (3,5) (5,3) (4,5) (3,7)` - **every** pair with both ends >= 3 - each giving the exact `u = -1e-4`, zero refusals. apeGmsh refused `(4,6)` | **CONFIRMED (F1)** - fixed |
| 11 | the passenger DOF is read or written | ndf-4 / ndf-3 pair, `sp` imposing `p = 1e6` on the master's DOF 4 under `constraints Transformation` | node 2's displacement identical to the `p = 0` twin (`-1.0e-4`), `nodeReaction` DOF 4 exactly `0.0`, `eleResponse force` is 7-wide `[0,0,100, 0, 0,0,-100]` with the passenger slot exactly `0.0`, `basicForce` unchanged | DISCARDED - the fork's own G2 assertion reproduces |
| 12 | the two tangential sliders are coupled (or the square locus is not what ships) | footing pushed along `t1` and along `(t1+t2)/sqrt(2)`, `LoadControl` at 1e4/step until the mechanism forms; last converged step read off the recorder | axis capacity `2.5e5 N` = `tau_b * A` exactly; diagonal capacity in `(3.5e5, 3.6e5]`, ratio `1.40` against `sqrt(2) = 1.414` (step granularity 1e4) and against `1.0` for a circular locus | MEASURED - matches the documented uncoupled-slider choice; S4's owed number is `sqrt(2)` |
| 13 | a non-orthonormal frame reaches the deck | `_validate_interface_orient_triad` fed a `t1` skewed 10 deg OUT of the tangent plane (with `t2 = n x t1`), and a degenerate `t1 == n, t2 == 0`; both then run on the exe | both PASS the triad check - it compares `t2` to `n x t1` and never checks that `n`/`t1` are unit or mutually orthogonal. Consequence measured: the skewed frame is silently re-orthogonalised by `ZeroLength::setUp` and gives the identical answer; the degenerate frame fails to converge (`analyze` returns `-3`), loudly | DISCARDED as a defect - no reachable route produces a wrong deck, and both off-contract frames are either corrected or loud. A one-line orthonormality assert would close it as hardening |
| 14 | tag / deck determinism (ADR 0027) | the same 3-D model built twice, records compared, then two decks compared byte-for-byte | records identical; decks byte-identical (3779 bytes both) | DISCARDED |
| 15 | a hostile interface name breaks the Tcl | names `soil "footing" [x] $a {b}` and `brace{open` through to a run deck | the name only ever appears as a top-level `# comment`; both decks emit and run | DISCARDED (shared `_emit_name` lane, unchanged by S1-S3) |
| 16 | the results reader mis-shapes three springs per pair | not probed - would need an MPCO/`.ladruno` run | - | NOT PROBED; S3 claims `n_springs` comes from `META/NUM_COMPONENTS`, S4 owes the read-back |

### F1 - the accepted-ndf table was the note's examples, not its rule

`_ACCEPTED_3D_NDF_PAIRS` transcribed the seven pairs
`contact_3d_passenger_dof_adoption.md` spells out by name as if that list
were exhaustive. The note's actual rule, one sentence earlier, is "**any**
pair with both ndf >= 3" - which is also the rule
`validate_adaptive_element_endpoints` in the same file already applies to
every other zeroLength-family element in 3-D. So the build carried two
contradictory rules, and the interface one refused `(4, 6)`: an ndf-4 u-p
soil master under an ndf-6 shell raft. `_SLAVE_NDF_VALUES_3D` accepts
`slave_ndf=6` and its docstring asserts "every such pair is one the fork's
zeroLength takes directly" - false against an ndf-4 master - so the model
resolved, saved, reloaded and only then died at emit. The pre-existing
unit test even pinned the gap as intended (`(4, 6)  # the table is not
symmetric`).

Falsifier run: with `(4, 6)` added to the table the *same* apeGmsh deck
runs on the fork with zero refusals. Fixed here by replacing the set
membership with `accepts_3d_ndf_pair(a, b)` (both >= 3), declared beside
the resolver's `slave_ndf` contract and imported by the emit gate; the
named pairs survive as examples in the refusal text. Regressions:
`test_3d_interface_up_master_shell_slave_emits` (e2e),
`test_3d_interface_up_soil_shell_raft_deck_runs` (live, 0 fork refusals)
and `test_3d_gate_mirrors_the_resolvers_rule` - all three fail with the
old table restored and pass with the rule.

### Named, not fixed

- **S4** - the corner-wrapping master (row 4) end-to-end, and the
  results-side read-back of three springs per pair (row 16).
- **Hardening (any slice)** - `_validate_interface_orient_triad` could
  also assert the frame is orthonormal (row 13). Not done here: no
  reachable route produces such a record, and the two synthetic ones are
  either silently corrected by the engine or fail to converge.
- **Test coverage gap** - the staged and partitioned interface tests
  (`test_interface_partitioned_emit.py`,
  `test_interface_partitioned_staged_emit.py`) are 2-D only; the 3-D
  routes are correct (row 9) but unpinned. Cheap S4 addition.
- **Out of scope, observed in passing** - `ops.element.ShellMITC4(pg=...)`
  against a physical group carrying no elements in the snapshot (e.g.
  after `get_fem_data(dim=3)` dropped the dim-2 rows) emits the section
  and **zero** element lines, silently; and `apeSees.tcl(bin=<a
  directory>)` passes the directory straight to `CreateProcess` (WinError
  5) instead of resolving the binary inside it, while
  `APEGMSH_OPENSEES_BIN` IS a directory.
