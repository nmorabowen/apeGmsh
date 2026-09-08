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
