# ADR 0063 — Split-node kinematic slip fault (meshed fault discontinuity)

**Status:** Proposed (2026-06-17; design draft, for later implementation). The
heavier sibling that [0062](0062-moment-tensor-equivalent-body-force-source.md)
explicitly deferred ("Split-node is a separate future facility, not this
ADR"). Where 0062 radiates a **point** moment-tensor source as mesh-light
equivalent body forces, this ADR puts an actual **fault surface** in the mesh
and prescribes **slip** across it. Reuses the meshing-side split machinery
adjacent to higher-order line splitting ([0037](0037-higher-order-line-broker-split.md))
and the constraint/pattern emission of [0034](0034-stage-bound-bcs-and-recorders.md)
/ [0005](0005-patterns-explicit.md). Source-verification against the upstream
OpenSees tree is **owed at implementation** (marked ⚠owed).

## Context

A fault can enter a continuum model two ways:

1. **Equivalent body force (ADR 0062).** The fault is a *point* (or a cloud of
   points); its radiated field is reproduced by RHS nodal forces
   $F^a=M\cdot\nabla N_a$. No discontinuity in the mesh, no constraint,
   integrator-agnostic. Right for **radiating a known kinematic source** into
   the medium (teleseismic / regional input, ShakerMaker ruptures).

2. **Split-node fault (this ADR).** The fault is a *surface* meshed as a
   **discontinuity**: the nodes on the fault plane are **duplicated** into a
   "+" sheet and a "−" sheet, and the relative displacement across the pair is
   the **slip** $\Delta u(\xi,t)$. Right for the **near field on the fault**:
   prescribed-kinematic finite-fault rupture where the on-fault slip-time
   function matters, and the precursor to **spontaneous dynamic rupture**
   (a friction law replaces the prescribed slip).

We have (1). We do **not** have (2), and it is the natural request once a user
wants the wavefield *near* an extended fault rather than the far-field
radiation of a point source. The split-node / traction-at-split-node (TSN)
method (Day 1982; Andrews 1999) is the standard discretization.

Two regimes share the same meshed discontinuity but differ at the fault:

- **Kinematic (this ADR's v1).** Slip is **prescribed**: $\Delta u(\xi,t) =
  s(\xi)\,\hat d(\xi)\,S(t-t_r(\xi))$ — the per-point slip magnitude, rake
  direction, onset, and moment function are inputs (the same FFSP / SRF data
  0062's `from_ffsp` consumes, but applied *on* the fault instead of as a body
  force). No friction, no nonlinear fault constitutive — a boundary condition.
- **Spontaneous (deferred).** Slip is an **output** of a friction law
  (slip-weakening / rate-and-state) resolving the on-fault traction each step.
  Far heavier (a fault constitutive model + contact bookkeeping); out of scope
  here, but the meshed-discontinuity substrate this ADR builds is its
  prerequisite.

## Decision (proposed)

Add a **meshed fault discontinuity** as a first-class geometry/mesh operation
plus a bridge-side **kinematic slip** driver. Unlike 0062 it is **not** a load
on an intact mesh — it changes the **topology** (node duplication) and adds
**constraints**, so it lives on the `g.*` mesh side with a thin `ops.*`
consumption, mirroring how embedded ties (ADR 0036) and stage-bound
constraints (ADR 0034) split responsibilities.

1. **Mesh-side split: `g.mesh.editing.split_fault(surface_pg)`.** Duplicate
   the nodes of an internal surface PG into a `+`/`−` pair, rewiring the
   elements on each side to their own sheet so the two sides are
   mechanically disconnected across the fault. Produces:
   - the `+`/`−` node-pair map (parallel arrays, fault-local ordering),
   - the per-node fault **normal** $\hat n$ and an in-plane **strike/dip**
     basis (for resolving slip rake), and
   - a fault PG carrying the pairs for the bridge to consume.
   This is the load-bearing new capability and the hard part — splitting a
   conformal mesh along an internal surface, keeping the boundary of the fault
   (the tip line) **welded** (no slip at the rupture front edge), and not
   orphaning elements. Builds on the in-place topology rewrite proven for
   higher-order line splitting (ADR 0037). ⚠owed: confirm gmsh's node/element
   rewiring primitives handle an internal 2-D surface cleanly.

2. **Welded tip / partial rupture.** Only the *ruptured* portion of the
   surface splits; the fault tip line and any unruptured patch stay tied
   (slip $\to 0$ at the edge). The user names the ruptured region (PG / label
   / a slip distribution whose support defines it); the complement is welded
   via the `+`/`−` pairs being constrained to **zero** relative displacement.

3. **Kinematic slip driver — `s.fault_slip(...)` / `ops.fault.kinematic(...)`.**
   Per fault-node pair, prescribe the relative displacement
   $\Delta u = u^+ - u^- = s\,\hat d\,S(t-t_r)$. The realization in OpenSees
   (⚠owed source-verification against `OpenSees_Compile`):
   - **Decompose the pair into 3 relative DOFs** in the fault-local
     $(\hat d_{\text{strike}}, \hat d_{\text{dip}}, \hat n)$ frame.
   - **Fault-normal + off-rake components:** tie $u^+ = u^-$ (no opening, no
     off-rake slip) via `equalDOF` on those directions — equivalently a
     `zeroLength`-with-rigid-channels or a transformed MP constraint.
   - **Slip (rake) component:** prescribe the *relative* displacement
     $S(t-t_r)$. OpenSees has no native "relative SP", so v1 candidates:
     (a) the **two-sided forcing** TSN form — equal-and-opposite nodal forces
     on $+$/$-$ whose work imposes the slip (the dynamically-correct split-node
     force, $f^\pm = \pm \tfrac12 \dots$); or (b) a **prescribed `sp` on a
     retained master + `equalDOF`** to the slave; or (c) a **`zeroLength` with a
     prescribed-deformation channel**. Pick at implementation after a live
     source check; (a) is the textbook Day/Andrews choice and is RHS-friendly
     (explicit), echoing 0062's no-stiffness-change virtue.
   - Per-pair onset $t_r$ rides a per-pair time-shifted `Path`/`Yoffe` (reuse
     ADR 0062's MT-3 `S(t)` helpers + the one-pattern-per-onset-group idea).

4. **Slip distribution from the same sources as 0062.** A finite-fault rupture
   (FFSP / SRF) supplies per-subfault slip / rake / onset / rise — 0062 turns
   these into body forces; here they map onto the fault-surface node pairs by
   nearest-subfault or interpolation. Share the converter (`from_ffsp` units
   work, once corrected) and the `Yoffe` STF; only the *application* differs
   (on-fault relative displacement vs in-volume body force).

5. **No new element for kinematic v1 (preferred).** Prefer constraints +
   prescribed relative motion (or TSN forces) over a fork fault element, to
   keep the "no solver change" property. A dedicated fault/cohesive element is
   the spontaneous-rupture path (deferred) and would be a fork dependency.

6. **H5 / provenance.** The split (pair map, normals, ruptured region) and the
   slip distribution round-trip in `model.h5` so the viewer can draw the fault
   surface + slip contours and `Results` can label on-fault quantities. Likely
   a schema bump (new topology + a `/opensees/faults` group) — unlike 0062
   which needed none (it emitted ordinary `load` lines).

## Why not the alternatives

- **Just use the moment-tensor source (ADR 0062).** Correct for the *radiated*
  field of a known source, wrong for the *near-fault* field: a point/cloud of
  body forces cannot represent the displacement discontinuity, the static
  offset across the fault, or the on-fault traction. Different physics, not a
  substitute.
- **A fork "fault element" up front.** Couples to a specific OpenSees build and
  breaks the no-solver-change property for the common kinematic case. Reserve
  it for spontaneous rupture (friction), deferred.
- **Cohesive-zone / contact interface only.** That is the spontaneous-rupture
  tool; overkill (and a constitutive law) for prescribed kinematic slip.
- **Author the split by hand (duplicate nodes in raw gmsh + hand constraints).**
  The escape hatch, but the pair bookkeeping, welded tip, rake frame, and
  per-pair onset are exactly the footguns this facility centralizes.

## Gotchas the implementation must honor (⚠owed verification)

- **Welded tip line.** Slip must taper to zero at the rupture boundary, or the
  tip radiates a spurious stress singularity. The split must not free the edge
  pairs.
- **No interpenetration / opening (kinematic v1).** Constrain fault-normal
  relative motion to zero (a kinematic fault slips, it does not open); only the
  rake direction carries prescribed slip.
- **Apply slip $S(t)$, not slip-rate.** Same lesson as 0062 — the prescribed
  *displacement* is the moment function (integral of slip-rate).
- **Mass/partitioning of duplicated nodes.** Split nodes double the local DOFs;
  mass lumping, `ndf`, and partition ownership (cross-partition fault pairs)
  must be handled — interacts with ADR 0027 cross-partition MP constraints.
- **Explicit stability at the split.** TSN forces keep the LHS = mass (explicit
  friendly); a stiff penalty tie across the fault would shrink the stable dt.

## Slice plan (proposed, for later)

- **SF-1 — mesh split (`split_fault`).** Internal-surface node duplication +
  element rewiring + `+`/`−` pair map + fault normal/strike/dip basis + welded
  tip; fail-loud on non-manifold / boundary-touching faults. The crux.
- **SF-2 — kinematic slip driver.** Per-pair relative-slip prescription
  (TSN forces vs constraint, decided after a live source check) + welded
  normal/off-rake tie; reuse MT-3 `S(t)` + per-onset grouping.
- **SF-3 — finite-fault mapping.** Map FFSP/SRF slip/rake/onset/rise onto the
  fault pairs (share 0062's converter + `Yoffe`).
- **SF-4 — H5 + viewer.** Persist the split + slip; fault-surface + slip-contour
  rendering; `Results` on-fault labels.
- **SF-5 — validation.** Static slip → analytic Okada surface displacement;
  dynamic kinematic → compare to a point-MT (ADR 0062) run in the far field
  (the two must agree where the point approximation holds).
- **Deferred beyond this ADR:** spontaneous dynamic rupture (slip-weakening /
  rate-and-state friction on the split) — needs a fault constitutive model.

## Open questions

1. Slip realization: TSN equal-and-opposite forces (a) vs prescribed-relative
   constraint (b/c)? (Lean (a) — explicit-friendly, textbook — but confirm
   against `OpenSees_Compile` what cleanly imposes a *relative* prescribed
   displacement.) ⚠owed.
2. Does the meshing layer (gmsh) expose a clean internal-surface split, or must
   apeGmsh duplicate + rewire connectivity itself (à la ADR 0037)? ⚠owed.
3. Fault geometry authoring: a meshed surface PG that must be planar-ish, or a
   general curved surface? (v1: an existing internal surface PG; curved/branched
   faults later.)
4. Cross-partition fault pairs (a fault cut by the METIS partition) — how do the
   `+`/`−` pairs and their tie/forcing land per rank (ADR 0027 interaction)?
5. Static pre-stress + kinematic slip in one staged run (gravity → rupture):
   reuse ADR 0052 staged reference-position contract?
