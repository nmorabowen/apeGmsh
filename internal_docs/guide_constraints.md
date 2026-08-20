> **Absorbed** into `docs/concepts/constraints.md` (ADR 0079 P2-B). This guide is retained as working memory; the concepts page is canonical.

# apeGmsh constraints

A guide to defining multi-point constraints in an apeGmsh session —
from simple DOF ties to surface coupling and mortar methods. This
document covers the apeGmsh abstraction; see `guide_fem_broker.md` for
how constraints land in the broker and get consumed by a solver.

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
# ... geometry, parts, mesh ...
```


## Tasks on this page

- [Tie co-located nodes (equal_dof)](#level-1-node-to-node) · [Couple a master to a node group](#level-2-node-to-group) · [Bridge beam-to-solid DOFs (node_to_surface)](#level-2b-mixed-dof-coupling) · [Tie non-matching surfaces](#level-3-surface-coupling) · [Surface-to-surface tie / mortar](#level-4-surface-to-surface) · [Let an interface open and slide (contact)](#level-5-contact-fork) · [Declare contact in a 2D model](#contact-in-a-2d-model) · [Consume constraints in a solver](#5-consuming-constraints-in-a-solver) · [Inspect what you declared](#6-introspection) · [Stage-bind a constraint (SSI)](#65-stage-binding-constraints-in-apesees-ssi-workflows)


## 1. The two-stage pipeline: define, then resolve

Constraints follow the same pattern as loads and masses. You *define*
constraints before meshing against high-level targets (part labels,
physical groups), and the library *resolves* them to node-level records
after the mesh exists.

```python
# Stage 1 — define (pre-mesh)
g.constraints.equal_dof("slab", "column_top", dofs=[1, 2, 3])

# Stage 2 — resolve (happens inside get_fem_data)
fem = g.mesh.queries.get_fem_data(dim=3)

# Resolved records are now on the broker
fem.nodes.constraints       # NodeConstraintSet  (node-pair types)
fem.elements.constraints    # SurfaceConstraintSet (surface types)
```

The definitions reference *names*, not tags — so they survive remeshing.


## 2. DOF numbering

All constraint definitions use **1-based DOF numbering**:

| DOF | Meaning |
|-----|---------|
| 1   | ux — translation X |
| 2   | uy — translation Y |
| 3   | uz — translation Z |
| 4   | rx — rotation X |
| 5   | ry — rotation Y |
| 6   | rz — rotation Z |

For a 3-DOF model (ndf=3, solids), only DOFs 1-3 exist.
For a 6-DOF model (ndf=6, frames/shells), all 6 are available.


## 3. Constraint types

### Level 1 — Node-to-node

**`equal_dof`** — tie co-located nodes so they share selected DOFs.
The most common constraint. Finds node pairs within a tolerance and
couples them.

```python
g.constraints.equal_dof(
    "slab", "column_top",
    dofs=[1, 2, 3],        # couple translations only
    tolerance=1e-6,
)
```

Use this when two parts share a boundary and you want continuity of
displacement across it. If you omit `dofs`, all DOFs are tied.

**`rigid_link`** — rigid bar coupling between a master node (or point)
and slave nodes. Two types:

```python
# "beam" — full 6-DOF rigid body motion (default)
g.constraints.rigid_link("center", "perimeter", link_type="beam")

# "rod" — translations only, rotations free
g.constraints.rigid_link("center", "perimeter", link_type="rod")
```

Use `rigid_link` for connecting a single master (e.g., column centroid)
to a ring of slave nodes. The difference from `equal_dof` is that
rigid links enforce rigid body kinematics — rotations at the master
produce translations at the slaves proportional to their offset.

**`penalty`** — soft spring approximation of `equal_dof`. Useful when
the solver has trouble with algebraic constraints (Lagrange multipliers)
and a spring coupling is more stable:

```python
g.constraints.penalty("part_A", "part_B", stiffness=1e10, dofs=[1, 2, 3])
```

The stiffness should be large enough to be effectively rigid but not so
large that it causes ill-conditioning. A rule of thumb: 10x to 1000x the
stiffest element in your model.


### Level 2 — Node-to-group

**`rigid_diaphragm`** — enforces in-plane rigidity at a floor level.
All slave nodes at the plane follow the master node's in-plane DOFs:

```python
g.constraints.rigid_diaphragm(
    "center_col",         # master label (the reference node)
    "floor_nodes",        # slave label (all floor nodes)
    plane_normal=(0, 0, 1),
    plane_tolerance=0.5,  # tolerance for "at the plane"
)
```

This is the classic floor-diaphragm constraint for building models.
The `plane_tolerance` filters slave nodes to only those within the
specified distance of the diaphragm plane.

**`rigid_body`** — full 6-DOF rigid body constraint. Every slave
follows the master as if welded:

```python
g.constraints.rigid_body("master_point", "slave_region")
```

Use this for modeling rigid blocks, pile caps, or foundation mats
that are much stiffer than the surrounding structure.

**`kinematic_coupling`** — RBE2: a reference node rigidly drives a node
set, with per-slave DOF selectivity:

```python
g.constraints.kinematic_coupling(
    "center", "ring",
    dofs=[1, 2, 3],  # only translations; None ⇒ all the slave has
)
```

This emits the **Ladruno-fork `element LadrunoKinematicCoupling`**
(class tag 33012) — a penalty rigid-body driver carrying the correct
moment-arm transport `u_i = u_R + θ_R × d_i`, so an *offset* reference
node is coupled rigidly. It replaced the previous `equalDOF`-per-slave
expansion, which ignored the lever arm (correct only for coincident
nodes). **Fork-only:** the deck emits on any build, but running it needs
the Ladruno fork — stock OpenSees fails loud at the element line.

The penalty/enforcement knobs are exposed directly on the factory:
`k` (numeric or `"auto"` + `k_alpha`/`host`), `kr`, `enforce="penalty"|"al"`,
`bipenalty_dtcr` / `bipenalty_wcap` (explicit-dynamics penalty mass), and
`absolute` (skip the `g0` stress-free birth). `host` is given as a **FEM
element id**; the bridge translates it to the emitted OpenSees tag.

Under partitioned (OpenSeesMP) emit the element lands on a **single
canonical rank** (the rank where every slave is present; the reference
is ghost-declared there when foreign) — a slave set split across
partitions fails loud.

Reach for this when the region must move as a **rigid body** (loading
platen, rigid connection block). To *introduce a load at a point while
the region stays flexible*, use `distributing_coupling` (RBE3) instead.


### Level 2b — Mixed-DOF coupling

**`node_to_surface`** — couples a 6-DOF node (e.g., a frame node) to a
3-DOF surface mesh (e.g., a solid). This is a compound constraint that
creates phantom nodes:

1. Duplicate slave positions as phantom nodes (6-DOF)
2. Rigid link from master to each phantom
3. EqualDOF from each phantom to the original slave (translations only)

```python
g.constraints.node_to_surface("frame_end", "solid_face")
```

When `master` is a string it is resolved via `resolve_to_tags(..., dim=0)` —
i.e. the name must refer to a geometric **point** entity (a vertex), not a
curve/surface/volume label. `slave`, in contrast, is resolved at `dim=2`
(surface entities). See `ConstraintsComposite.py:837`.

Note: `node_to_surface` (and `embedded`, below) bypass the strict
``part-label`` validation that other constraint factories run in
`_add_def` — both definition types are allowed to carry bare entity
tags / mixed labels because they are looked up later through their
own resolvers (`ConstraintsComposite.py:236`).

For **pure BC application** (force/moment or prescribed displacement on a
face without a structural element at the reference), prefer
`g.loads.surface.force_resultant_center_mass()` / `g.displacements.surface()` instead — they distribute
directly to face nodes and avoid the phantom node conditioning issue.
See `guide_loads.md` §10–11.

This is the constraint you need when connecting a beam/frame model
to a solid model. The phantom nodes bridge the DOF mismatch.

**`node_to_surface_spring`** — same topology as `node_to_surface`, same
call signature, but the master → phantom links are emitted as stiff
`elasticBeamColumn` elements instead of kinematic `rigidLink('beam', ...)`
constraints:

```python
g.constraints.node_to_surface_spring("frame_end", "solid_face")
```

Use this variant when the master carries **free rotational DOFs** (e.g. a
fork support on a solid end face) that receive direct moment loading. The
constraint-based `node_to_surface` can produce an ill-conditioned reduced
stiffness matrix in that case because the master rotation DOFs only get
stiffness through kinematic constraint back-propagation, with nothing
attaching directly to them. Stiff beams give those rotations a real
elastic stiffness path and the matrix conditioning recovers.

Pick `node_to_surface` when the master rotations are themselves
constrained or carry no moment; pick `node_to_surface_spring` when the
master is a free rotation node receiving moments.

**Emission contrast.** The plain `node_to_surface` variant emits its
master→phantom links through `fem.nodes.constraints.phantom_nodes()`
plus `pairs()` (kinematic `rigidLink('beam', ...)`). The spring
variant routes the same topology through
`fem.nodes.constraints.stiff_beam_groups()` instead — each group
becomes one `elasticBeamColumn` element per master/phantom pair. The
factory entry point lives at `ConstraintsComposite.py:906`.


### Level 3 — Surface coupling

**`tie`** — surface tie via shape function interpolation. Slave nodes
project onto the closest master face and their DOFs are interpolated
from the master face nodes:

```python
g.constraints.tie("flange_surface", "web_surface", tolerance=1.0)
```

Use `tie` when two non-matching meshes share a boundary and you want
displacement continuity without requiring conformal meshing. The
tolerance controls how far a slave node can be from the master surface.

> For a step-by-step recipe, see [How-to: Tie non-matching meshes](../how-to/tie-meshes.md).

**`method="mortar"` — the integral upgrade (ADR 0086).** `tie` defaults to
`method="collocation"`, the projection above, and that is the right answer
until the two sides differ in element *order*. Hex20 faces tied onto hex8
faces pin quadratic slave nodes to a bilinear master field, which
over-constrains the finer side (measured on the Cerro Lindo fuse: rib-root
fixity `a = 0.195` against a 0.230 conformal reference, K +7.7 %).

```python
g.constraints.tie("plate_face", "rib_face",
                  method="mortar", enforce="equation", dofs=[1, 2, 3])
```

`method="mortar"` integrates the bond over the slave/master facet overlaps
with a dual (biorthogonal) slave basis, so neither side's interpolation
order is imposed on the other, and the tie is master/slave symmetric in a
way collocation is not. The weights are computed here, in pure numpy
(`_kernel/resolvers/_mortar.py`), and ride the **existing**
`enforce="equation"` emission — same `InterpolationRecord`, same handlers,
no new emitter verbs. `method=` chooses how the weights are computed;
`enforce=` still chooses how the record is enforced.

The contract is narrower than collocation's, and every violation is a hard
`MortarTieError` — a mortar tie never silently resolves to nothing:

- `enforce="equation"` in v1, so it needs the Lagrange handler and an
  unsymmetric system like any equation tie; `dofs` are translations only.
- A **flat, coincident** interface (`tolerance` becomes the out-of-plane gap
  allowance) of **convex** facets with **straight edges** — every midside
  node at its edge midpoint, because the kernel integrates on the corner
  polygon, which equals the real facet only when the edges are straight.
- **No master facet may overlap another.** The coverage check sums pair-clip
  areas, so it counts multiplicity: two layers of master surface can mask an
  untied slave strip, which is then *extrapolated* rather than left free —
  with `Σw = 1` and a linear patch test both staying clean. Nothing
  downstream can catch that, so it is refused before assembly.
- **tri6 SLAVE facets are refused.** Their corner shape functions integrate
  to zero over the parent triangle, so the dual row-sum scaling divides by
  zero. tri6 is fine as a master and quad8 works on both sides — swap the
  sides, or use collocation.

Both sides resolve from dim-2 element groups, so the parts must have been
extracted with `dim=None`. It works on composed assemblies (`from_h5` +
`g.compose`, or `Assembly` with `kind="tie"`), which is where
order-mismatched models live in the first place: `set_order` is global per
gmsh session, so mixing element order at all requires compose.

**Tie-force recovery (`enforce="equation"`).** A tie emitted on the exact
kinematic route (`g.constraints.tie(..., enforce="equation")`, the
`LadrunoProjection`-handled constraint variant — see
`internal_docs/handoff_equation_tie_adr0068.md`) can report the interface
force it carries, the apeGmsh analogue of LS-DYNA `*DATABASE_NCFORC`:

```python
# live query — single value from the last projection step of a live analyze
ops = apeSees(fem)
...                                          # build + LadrunoProjection handler
ops.analyze(steps=1, dt=1e-3)
f = ops.ladruno_projection_tie_force(slave_node, dof=1)   # f = M(a_raw - a_proj)

# recorded time history — vec3/node, explicit analyses only
ops.recorder.Ladruno(file="ties.ladruno",
                     nodal_responses=("constraintTieForce",))
# ... after the run:
results.nodes.get(component="constraint_tie_force_x", ids=[slave_node])
```

Both routes are **fork-only** (they need the `LadrunoProjection` handler /
the `ladruno` recorder; a stock build fails loud). The live query works under
implicit and explicit analyses; the recorder channel is explicit-only (it is
scattered each commit by `CentralDifferenceLadruno`).

**`distributing_coupling`** — RBE3: distribute a load at a reference
point over a node set while the set stays **flexible**:

```python
g.constraints.distributing_coupling(
    "load_point", "bearing_surface",
    weighting="area",  # or "uniform"
)
```

This emits the **Ladruno-fork `element LadrunoDistributingCoupling`**
(class tag 33011), replacing the prior `NotImplementedError` stub: the
reference (dependent) node R is the weighted-average rigid-body fit of
the independent set, and a force/moment at R distributes as a
statically-equivalent pattern (`Σ Fᵢ = F`, `Σ rᵢ × Fᵢ = M`) **adding no
stiffness** to the independents. It is the flexible counterpart of
`kinematic_coupling` (RBE2, which holds the set rigid). **Fork-only:**
stock OpenSees fails loud at the element line.

`weighting="uniform"` ⇒ equal weights (the element default).
`weighting="area"` ⇒ apeGmsh computes each independent node's
**tributary area** over the slave surface (each face's area split
equally among its nodes — the same lumping model as `g.loads` surface
tributary resolution) and emits `-w w1..wN`, so a force at R distributes
like a uniform traction. An independent node on no slave face fails loud.

The same penalty/enforcement knobs as `kinematic_coupling` are exposed
(`k`/`k_alpha`/`host`, `kr`, `enforce`, `bipenalty_dtcr`/`bipenalty_wcap`,
`absolute`). Note R is **massless by construction**, so an explicit run
needs `bipenalty_dtcr` (or `bipenalty_wcap` with a `host`) or the stable
step collapses to zero.

**`embedded`** — embedded element constraint (reinforcement in concrete):

```python
g.constraints.embedded("concrete_volume", "rebar_lines")
```

Slave (embedded) nodes follow the displacement field of the host elements.

Implemented end-to-end: at resolution time `_resolve_embedded`
(`ConstraintsComposite.py:1218`) collects the host tet4/tri3 elements
and the embedded-curve nodes, drops embedded nodes that coincide with
a host corner (already rigidly attached via shared connectivity), and
hands the remainder to `resolver.resolve_embedded` which lands real
`InterpolationRecord`s on `fem.elements.constraints`.


### Level 4 — Surface-to-surface

**`tied_contact`** — full surface-to-surface tie. Bidirectional check:

```python
g.constraints.tied_contact(
    "master_surface", "slave_surface", tolerance=1.0
)
```

More robust than `tie` for large non-matching meshes because it checks
projections in both directions.

**`mortar`** — **deprecated alias, and not the integral mortar.** Since ADR
0073 `g.constraints.mortar(...)` is a thin alias for
`contact(formulation="mortar", tie=True)`: the fork's ALM-penalty
segment-to-segment contact-tie. It warns, returns a `ContactDef`, and
resolves onto `fem.elements.contacts` — the serial-only contact subsystem
emitted by `emit_contacts` — not onto the MP-constraint channels. Call
`contact()` directly instead.

It is also the wrong tool for a permanent bond. A contact-tie is enforced by
penalty/augmentation, so the gap is driven toward zero but never reaches it;
it mandates the `LadrunoContact` handler, which cannot coexist with any
`enforce="equation"` tie in the same analysis; and it accepts linear facets
only. Reach for it when the interface must also *behave* as contact later in
the run.

For a true integral mortar — surface integration of the coupling operator
over the intersected facets, which is what the older text in this guide said
did not exist — use **`tie(method="mortar")`** in Level 3 above. That one
ships (ADR 0086), handles quadratic facets, and emits equation constraints.


### Level 5 — Contact (fork)

Everything above is a **bond**: the slave follows the master in tension as in
compression, from the first step to the last. `contact` is the one verb in the
composite that can *let go* and *slide* — a face-to-face interaction with a
gap, a normal penalty and Coulomb friction, emitted through the Ladruno fork's
contact subsystem (`contactSurface` + `contact`, enforced by the
`LadrunoContact` handler):

```python
g.constraints.contact(
    "rock_face", "liner_face",
    formulation="nts", kn="auto", kt=5.0e5, mu=0.3,
)
```

Two formulations. `"nts"` is node-to-segment penalty: the slave is a bare node
set, each node projected onto the master facet that owns it. `"mortar"` is
segment-to-segment augmented Lagrange: both sides are faceted and the interface
is integrated over the facet overlaps — the accuracy lane for non-matching
meshes. `tie=True` (mortar only) freezes the pair into a permanent
*contact-tie*; for a plain permanent bond prefer `tie(method="mortar",
enforce="equation")` from Level 3, for the reasons under **`mortar`** above.

Contact does not land on the MP-constraint channels. It resolves *additively*
onto `fem.elements.contacts` (and `fem.elements.contact_planes`) — a separate
**serial-only** subsystem emitted by `emit_contacts`. It needs a live gmsh
session, so declaring one on a `from_h5` / composed session raises; and the
deck emits on any build but **runs only on the fork**.

**`contact_plane`** — a rigid analytical plane, with no master mesh at all.
The slave contacts a fixed infinite plane given by a `normal` and a `point`,
frictionless, with a **required** `kn` (there is no `"auto"` on this lane):

```python
g.constraints.contact_plane(
    "footing", normal=(0.0, 1.0), point=(0.0, 0.0), kn=1.0e7,
)
```

Reach for it for a rigid floor, wall or foundation where the counter-body
needn't be meshed. It is also the only contact lane that keeps the fork's
`ndf >= ndm` rather than demanding `ndf == ndm`, so a 6-DOF shell can sit on
one; the NTS/mortar/tie lanes cannot.


#### Contact in a 2D model

The fork routes contact on the **node coordinates** of the referenced surface,
not on interpreter `ndm` state, and apeGmsh emits exactly `ndm` coordinates per
node — so a plane model reaches the fork's 2D lanes automatically. What changes
is the shape of a surface, and the way it is oriented.

**A 2D contact surface is a meshed dim-1 physical group** — the boundary
*curve* of the body, not the body. Naming the dim-2 plane PG is the natural
mistake and is refused by name: it would collect the continuum's own tri3 /
quad4 elements and build a structurally valid record out of solid elements.

```python
import gmsh

def _edge_at_x(surface, x, tol=1e-6):
    """The boundary curve of `surface` lying wholly at this x."""
    for _, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        lo_x, _, _, hi_x, _, _ = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(lo_x - x) < tol and abs(hi_x - x) < tol:
            return abs(tag)
    raise LookupError(f"surface {surface} has no boundary curve at x={x}")

left  = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
g.model.sync()
g.mesh.structured.set_transfinite([(2, left), (2, right)], n=4)
g.mesh.generation.generate(2)

g.physical.add(2, [left],  name="rock")                        # the bodies
g.physical.add(2, [right], name="liner")
g.physical.add(1, [_edge_at_x(left,  1.0)], name="rock_face")  # the interface
g.physical.add(1, [_edge_at_x(right, 1.0)], name="liner_face")

g.constraints.contact(
    "rock_face", "liner_face",
    formulation="nts", kn="auto", kt=5.0e5, mu=0.3,
    outward=(1.0, 0.0), name="joint",
)
fem = g.mesh.queries.get_fem_data(dim=2)
```

The two rectangles are **not** fragmented, so the two curves at `x = 1` are
geometrically coincident but distinct entities carrying distinct node sets —
which is exactly the topology a contact needs, and the opposite of what you
want for a `tie`.

which emits

```tcl
contactSurface 1 -master 2 3 12 12 11 11 2
contactSurface 2 -slave 23 24 8 5
contact 1 1 2 auto 500000.0 0.3 -outward 1.0 0.0
```

**Six tags for three segments.** `-master 2` (and, on the mortar lane,
`-slave-segments 2`) takes a *flat stride-2 pair list chained head-to-tail* —
`101 102  102 103  103 104`, not `101 102 103 104`. The four-tag shorthand is
**silently legal** fork-side and declares two disjoint segments with a hole
where the middle one should be: the deck converges, balances its reactions, and
transmits the load through the wrong distribution. apeGmsh walks the chain out
of the meshed PG's own edge connectivity, on both surfaces, so the holed form
is unreachable by construction — which is the single strongest reason to route
2D contact through the library rather than hand-writing the deck.

**Orienting the interface — and why it matters more here than in 3D.** The 3D
kernel computes a correct per-facet normal, so apeGmsh deliberately never
auto-derives a global `outward` there. The 2D lanes instead take one
interface-level sign from a centroid vote computed once at `handle()`, and that
vote is genuinely ambiguous when the two surface centroids coincide. **A flush
interface is the normal 2D case, not an edge case** — the masonry joint, the
footing seated on soil, every zero-gap interface — so apeGmsh refuses a flush
declaration by name and tells you what to add:

- `outward=(ox, oy)` — a 2-component direction toward the slave's allowed
  half-space. The **only** vector form the 2D lane accepts (a 3-vector with a
  non-zero `oz` is refused), and it works on **both** formulations.
- `outward="winding"` — declare the side through the master chain's own
  traversal instead of a direction: every segment's normal is `perp(t)` of its
  own tangent, so **the slave lies to the LEFT of chain travel**. Because the
  sign is per-segment, this is the only thing that can orient a *curved* or
  *closed* master, which no single direction vector can express. It needs one
  connected chain, and a fork build carrying `-outward winding`.

`outward="winding"` is **NTS-only**, and that asymmetry is permanent until the
fork moves. The fork shipped declared winding on the NTS lane alone because its
2D mortar lane runs no chain-integrity scan, so the one-connected-chain
invariant winding leans on is not established there. Two consequences worth
carrying rather than papering over: a flush **mortar** interface — the
workhorse case — always needs an explicit `outward=(ox, oy)`, and a curved or
closed **mortar** master stays undeclarable. apeGmsh refuses the combination at
declaration, and the flush refusal on the mortar lane offers the vector alone,
saying why.

**Nothing fork-side catches a master pointed the wrong way**, on either lane:
the centroid vote only picks a *sign*, so a master boundary on the far side of
the body resolves happily and orients a contact against a boundary the slave
never reaches. That guard is apeGmsh's, and it runs on both formulations.

**`thickness=` — keep three conventions apart.** A plane model carries an
out-of-plane thickness in three places, and only one of them is this parameter:

| Where | What it does |
|---|---|
| `ops.element.FourNodeQuad(thickness=…)` | Baked into the **element's** own stiffness. Contact never re-reads or re-derives it. |
| `contact(formulation="mortar", thickness=h)` → `-thickness h` | Scales the **explicit** `eps_n` / `eps_t` / `visc` / `cohesion` / `tau_max` and the tie stiffness — **once**, at the fork's 2D injection site. Fork default 1.0, so omitting it is bit-identical to `thickness=1.0`. |
| `eps_n="auto"` | **Not** h-scaled: it already absorbs the element thickness through `getInitialStiff()`, so scaling it again would be an h² error. An `eps_t="auto"` — or an `eps_t` defaulted from `eps_n` under friction — inherits `eps_n`'s provenance, and h-scales only when `eps_n` is explicit. |

`-thickness` is **mortar-only**: the NTS lane has no such parameter at all
(`kn` there is a force per unit length of gap, never a pressure, and
`kn="auto"` already folds the element's thickness), and a 3D model is refused
by name. apeGmsh never invents an `h` from the declared element thickness.

```python
g.constraints.contact("rock_face", "liner_face",
                      formulation="mortar", eps_n="auto",
                      outward=(1.0, 0.0), thickness=0.30, name="bed")
```

```tcl
contactSurface 1 -master 2 3 12 12 11 11 2
contactSurface 2 -slave-segments 2 5 24 24 23 23 8
contact 1 1 2 -mortar -epsN auto -thickness 0.3 -outward 1.0 0.0
```

**The rigid plane, in 2D.** `normal` and `point` become 2-vectors and the slave
is a meshed boundary curve (or a point set); apeGmsh z-pads and emits the
fork's permanently valid zero-padded 9-argument form, so there is only ever one
grammar to read:

```python
g.constraints.contact_plane("footing", normal=(0.0, 1.0), point=(0.0, 0.0),
                            kn=1.0e7, name="ground")
```

```tcl
contactSurface 1 -slave 5 6 1 2
contactPlane 1 1 0.0 1.0 0.0 0.0 0.0 0.0 10000000.0
```

**Meshing the interface.** Four properties of the mesh decide whether a 2D
contact deck transmits anything at all, and none of them is visible in the
declaration:

1. **Seed a small initial overlap on the NTS lane.** A zero initial gap does
   **not** arm it. Measured on a two-square deck of the shape above, pushed
   into the interface under load control (fork build `e7555f2c9`): with the
   bodies exactly coincident the very first Newton step diverges to `inf` and
   the interface transmits `0.0`; offsetting the slave body `1e-4` into the
   master, the identical deck converges and the summed normal contact force
   closes on the applied `1.0e5` to five significant figures. The
   **rigid-plane lane does not need this** — it arms from a zero gap.
2. **Restrain the free body transversally.** 2D contact is normal-only, so a
   body whose only other support is the interface keeps an unrestrained
   transverse / rotational mode. When that mode is excited Newton drifts (a
   measured increment norm of 5.3e+11) and the only thing that surfaces is a
   misleading near-**PERPENDICULAR** warning about the interface geometry —
   which sends you to inspect the mesh instead of the boundary conditions.
3. **Size a curved master's facets from the expected penetration, not from the
   surrounding mesh.** On a faceted arc the NTS narrow phase stops arming a
   pair once its penetration exceeds roughly *twice* the local facet length, so
   a deck seeded with a large initial penetration against a fine facet mesh
   silently disarms the interior of the contact patch and collapses the load
   onto a rim of surviving nodes. It looks exactly like ordinary discretization
   error, and it is not. Worse: if the arc's facet length is driven by the same
   `g.mesh.sizing` parameter as the surrounding elastic mesh — the natural
   choice, and the one the sizing API makes easiest — **refining the mesh makes
   it worse**. Drive the interface curve's size separately.
4. **A closed loop works only if it is adequately refined.** A ring whose
   facets are coarse relative to its own diameter converges, balances its
   reactions, and transmits **exactly zero** — fork-measured on a unit regular
   n-gon at the default cell size: `n <= 8` inert, `n >= 9` exact. The same
   rule as (3), for a second reason: size a loop's facets from the loop's own
   extent, never from the elastic mesh around it.

**Gates.** Every node of both surfaces must carry the same coordinate size, and
on the NTS / mortar / tie lanes the fork requires **`ndf == ndm` exactly** (the
rigid plane alone keeps `ndf >= ndm`). Every violation is a named abort, never
a silent DOF shift. **Parallel / DDM 2D contact is out of scope fork-side** and
is refused by name on all three lanes — `contact` in either formulation, and
`contact_plane` — as soon as the model is partitioned.

`Ladruno_implementation/LadrunoContact2D_guide.md` in the fork is the authority
on kernel behaviour: the vertex / corner policy, the D4 radial end-cap, the
broad-phase `cell` sizing and the full units table live there, not here.


## 4. How constraints land in the broker

After `get_fem_data()`, constraints are split across two composites
based on what solver commands they produce:

**Node-level** (`fem.nodes.constraints`):
- `equal_dof` → `NodePairRecord`
- `rigid_link` → `NodePairRecord`
- `penalty` → `NodePairRecord`
- `rigid_diaphragm` → `NodeGroupRecord` (expands to pairs)
- `rigid_body` → `NodeGroupRecord`
- `kinematic_coupling` → `NodeGroupRecord` (emits the fork
  `element LadrunoKinematicCoupling`, not `equalDOF` pairs; carries an
  optional `CouplingControl` with the penalty knobs)
- `node_to_surface` → `NodeToSurfaceRecord` (phantom nodes + pairs)
- `node_to_surface_spring` → `NodeToSurfaceRecord` (phantom nodes + stiff-beam pairs)

**Surface-level** (`fem.elements.constraints`):
- `tie` → `InterpolationRecord` (emits `ASDEmbeddedNodeElement`)
- `distributing_coupling` → `InterpolationRecord` (emits the fork
  `element LadrunoDistributingCoupling`; `weights` carries the
  tributary areas under `weighting="area"`, `None` = uniform; optional
  `CouplingControl`)
- `embedded` → `InterpolationRecord` (emits `ASDEmbeddedNodeElement`)
- `tied_contact` → `SurfaceCouplingRecord`

`tie(method="mortar")` lands here too, as an ordinary
`InterpolationRecord` — the weights come from the overlap integrals
instead of collocated shape functions, but a record is a record, so the
three `enforce=` routes and the H5 schema are untouched.

**Contact is not in this table at all.** `contact` resolves onto
`fem.elements.contacts` as a `ContactRecord` and `contact_plane` onto
`fem.elements.contact_planes` as a `ContactPlaneRecord` — a separate
serial-only subsystem emitted by `emit_contacts`, not the MP-constraint
channels. `ContactRecord.master_nps` is the dimension discriminator: 3 or 4
for a 3D tri/quad facet set, **2** for a 2D chained line segment
(`ContactRecord.ndm` is derived from it, never stored). `mortar` lands there
too, as a deprecated alias for the fork contact-tie.


## 5. Consuming constraints in a solver

Use the `Kind` constants for linter-friendly comparisons — no magic
strings:

```python
K = fem.nodes.constraints.Kind

# 1. Create phantom nodes first
for nid, xyz in fem.nodes.constraints.phantom_nodes():
    ops.node(nid, *xyz)

# 2. Emit node-pair constraints (compound records expanded automatically)
for c in fem.nodes.constraints.pairs():
    if c.kind == K.RIGID_BEAM:
        ops.rigidLink("beam", c.master_node, c.slave_node)
    elif c.kind == K.RIGID_ROD:
        ops.rigidLink("rod", c.master_node, c.slave_node)
    elif c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
    elif c.kind == K.PENALTY:
        # custom penalty spring element
        ...

# 3. Surface constraints
for interp in fem.elements.constraints.interpolations():
    # build multi-point constraint from interpolation weights
    # interp.slave_node, interp.master_nodes, interp.weights
    ...
```


## 6. Introspection

```python
# What constraints do I have?
print(fem.inspect.constraint_summary())
# Node constraints (12 records):
#   equal_dof                  8  (source: 'slab_column_tie')
#   rigid_beam                 4  (source: 'node_to_surface coupling')
#   phantom nodes              4  (created by node_to_surface)
# Surface constraints (2 records):
#   tie                        2  (source: 'flange_web_tie')

# DataFrame summaries
fem.nodes.constraints.summary()       # kind, count, n_node_pairs
fem.elements.constraints.summary()    # kind, count, n_interpolations
```


## 6.5 Stage-binding constraints in apeSees (SSI workflows)

When using `apeSees` to emit a multi-stage analysis deck, any
constraint can be **claimed by name** for a specific stage so it
emits inside that stage's block instead of the global pre-stage
MP-constraint pass. This is what you want when the constraint
references nodes / elements that only come online via
`s.activate(pgs=[...])` in a later stage — emitting globally would
reference nodes that don't exist at parse time and crash OpenSees.

Name the constraint when you declare it:

```python
# At apeGmsh time — name= is the contract for stage routing later:
g.constraints.embedded(
    host_label="Rock", embedded_label="Lining",
    name="lining_embed", stiffness=1e8,
)
```

Then inside the stage block, claim by that name:

```python
# At apeSees bridge time:
with ops.stage(name="install_lining") as s:
    s.activate(pgs=["Lining"])
    s.embedded(name="lining_embed")    # claim — emits here, not globally
    s.analysis(...)
    s.run(n_increments=10, dt=0.1)
```

Available CLAIM verbs on `_StageBuilder`: `s.embedded`, `s.tie`,
`s.distributing`, `s.equal_dof`, `s.rigid_link`,
`s.rigid_diaphragm`, `s.kinematic_coupling`, `s.node_to_surface`,
`s.node_to_surface_spring`. `s.tied_contact` and `s.mortar` are
deferred — see [_DEFERRED.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/_DEFERRED.md).
Forgetting to claim is caught at build time by the V1 ownership-
tier validator with an actionable offender list.

Full guide: `guide_opensees.md` §4.4 "Multi-point constraints" and
ADR 0034 §5a "Stage-bound constraints via CLAIM-by-name".


## 7. Guidelines

- **Start with `equal_dof`** — it covers 80% of constraint needs.
- **Use `tie` for non-matching meshes** — it handles mesh incompatibility
  without conformal remeshing.
- **Use `node_to_surface` for beam-to-solid** — it bridges the DOF
  mismatch automatically.
- **Reach for `tie(method="mortar")` when the two sides differ in element
  order** — collocation over-constrains the quadratic side. Don't reach for
  `g.constraints.mortar()`: it is a deprecated alias for the contact-tie,
  not the integral mortar.
- **Check `phantom_nodes()`** — if you have `node_to_surface` constraints,
  phantom nodes must be created in the solver before emitting constraints.
- **Set `tolerance` carefully** — too tight and no pairs are found; too
  loose and you couple nodes that shouldn't be coupled.
- **Reach for `contact` only when the interface must open or slide** — it
  costs the `LadrunoContact` handler (exclusive with any `enforce="equation"`
  tie), a fork build at run time, and serial emit. Every Level 1–4 verb is a
  cheaper answer when the bond is permanent.
- **In 2D, decide the orientation before you mesh** — a flush interface is the
  normal case and is refused without an `outward`; on the mortar lane the
  vector is the only option there is.


## See also

- `guide_fem_broker.md` — how the broker organizes constraints
- `guide_loads.md` — the analogous two-stage pipeline for loads
- `guide_basics.md` — session lifecycle
- `guide_opensees.md` §4.4 — MP constraint emit + stage-binding in
  `apeSees` (SSI-2.D extension)
- [How-to: Tie non-matching meshes](../how-to/tie-meshes.md) — recipe for
  `tie` / `tied_contact` between non-conformal parts
- `OpenSees/Ladruno_implementation/LadrunoContact2D_guide.md` — the fork's own
  2D contact user guide: kernel behaviour, the vertex/corner policy, the D4
  end-cap, the units table. **The authority** for anything Level 5 states
  about what the solver does.
- `contact_2d_adoption.md` — the adoption record for the 2D lane (fork ADR-85),
  including what was measured on which fork build. Read its STATUS header
  first; the body below it is pre-adoption history.


??? note "For maintainers — source map"

    - `src/apeGmsh/core/ConstraintsComposite.py` — the user-facing composite
    - `src/apeGmsh/_kernel/defs/constraints.py` — `ConstraintDef` subclasses
    - `src/apeGmsh/_kernel/records/_constraints.py` — `ConstraintRecord` subclasses
    - `src/apeGmsh/_kernel/resolvers/_constraint_resolver/_resolver.py` —
      `ConstraintResolver`
    - `src/apeGmsh/_kernel/resolvers/_constraint_resolver/_geom.py` — geometry
      helpers shared by the resolver (shape functions, projection)
    - `src/apeGmsh/_kernel/resolvers/_mortar.py` — the dual-basis mortar kernel
      behind `tie(method="mortar")`
    - `src/apeGmsh/_kernel/records/_kinds.py` — `ConstraintKind` (and `LoadKind`)
      constants
    - `src/apeGmsh/opensees/_internal/build.py` — solver-side emission
      (`emit_mp_constraints` and the partitioned/stage variants)
    - `src/apeGmsh/_kernel/record_sets.py` — `NodeConstraintSet`, `SurfaceConstraintSet`
    - `src/apeGmsh/_kernel/geometry/_boundary_chain.py` — the 2D master/slave
      chain walk and the winding sign (Level 5)
    - `src/apeGmsh/opensees/element/contact.py` — the `contactSurface` /
      `contact` / `contactPlane` emit grammar
