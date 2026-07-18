# Constraints

This page explains how apeGmsh couples regions of a model that don't share
nodes — which constraint fits which coupling situation, when declarations turn
into node-level records, and why you never hand-write a single `equalDOF` for
the solver.

## One equation, four situations

Every multi-point constraint in the library is ultimately the same statement:
some slave DOFs are a linear function of some master DOFs,
`u_slave = C · u_master`. What distinguishes the dozen factory methods on
`g.constraints` is *how the coefficient matrix gets built*, and that follows
directly from the geometry of the interface:

- the two sides **share node positions** → pair co-located nodes directly, or
  drive a node group rigidly from one master;
- the two sides **meet at a surface but the meshes don't match** → project
  slave nodes onto master faces and interpolate through shape functions;
- one part is **buried inside another** (rebar in concrete) → locate each
  embedded node inside a host element and couple it to the host's corners;
- the two sides may **touch, slide, or separate** → a genuine contact
  formulation, not a kinematic constraint at all.

Pick the situation and the method follows. Everything below walks the four in
that order, but first the timing — because it's the same for all of them, and
it's the part that makes constraints feel different from other FEM tools.

## Declare, then resolve, then emit

Constraints follow the declare-then-resolve rhythm from
[the mental model](mental-model.md). You *declare* a constraint before the mesh
exists, against names — part labels or physical groups, never node tags:

```python
# Stage 1 — declare (pre-mesh)
g.constraints.equal_dof("slab", "column_top", dofs=[1, 2, 3])

# Stage 2 — resolve (happens inside get_fem_data)
fem = g.mesh.queries.get_fem_data(dim=3)

# Resolved records are now on the broker
fem.nodes.constraints       # node-pair and node-group records
fem.elements.constraints    # surface interpolation records
```

The declaration is pure intent — no node exists yet when you make it, which is
exactly why it survives remeshing. At `get_fem_data(...)` the resolver walks
your declarations against the real mesh: it finds the co-located pairs, runs
the projections, computes the interpolation weights, and lands concrete records
on the snapshot — node-level constraints on `fem.nodes.constraints`,
surface-coupling records on `fem.elements.constraints`.

Then the third stage happens without you: the typed `apeSees(fem)` bridge reads
those records and **emits the solver commands automatically** — `equalDOF`,
`rigidLink`, `rigidDiaphragm`, the `ASDEmbeddedNodeElement` penalty elements,
phantom nodes and all. You declare the tie; the bridge writes the deck. The one
refinement worth knowing at concept level: in a staged analysis, a constraint
whose nodes only come alive in a later stage can be given a `name=` at
declaration and *claimed* inside that stage's block
(`s.tie(name=...)`, `s.embedded(name=...)`) so it emits there instead of
globally.

One convention runs through all of it: DOFs are **1-based indices**
(`1=ux, 2=uy, 3=uz, 4=rx, 5=ry, 6=rz`), and every geometric `tolerance` is in
model units — `1e-6` is right for a metre model and uselessly tight for a
millimetre one. The single exception to the index convention is
`g.constraints.bc(...)`, the fix-to-ground boundary condition, which takes an
OpenSees-style restraint *mask* (`dofs=[1, 1, 0]` means "fix x and y") because
it becomes `ops.fix` downstream. It lives on `g.constraints` because fixity is
permanent, not load-pattern-scoped; the recipe is in
[Supports & BCs](../how-to/supports-bcs.md).

## Same mesh: co-located pairs and rigid clusters

When two parts genuinely share node positions at their boundary — a conformal
interface, typically the product of `g.parts.fragment_all()` or careful
construction — the constraint is just bookkeeping. `equal_dof` finds every
master/slave node pair within `tolerance` and ties the selected DOFs:

```python
g.constraints.equal_dof(
    "slab", "column_top",
    dofs=[1, 2, 3],        # couple translations only
    tolerance=1e-6,
)
```

Omit `dofs` and all available DOFs are tied. Two siblings cover the less-common
shapes of the same situation: `equal_dof_mixed` ties *differently numbered*
DOFs across a pair (master `uz` driving slave `rz`, say, at a solid-to-shell
joint), and `penalty` replaces the algebraic constraint with a stiff spring
when the solver's constraint handler struggles.

`rigid_link` adds kinematics: each slave follows the master through a rigid
offset arm, so a rotation at the master produces translations at the slaves.
`link_type="beam"` carries all six DOFs; `"rod"` couples translations only.
Three group-level constraints build on the same idea. `rigid_diaphragm` is the
classic floor constraint — every slab node within `plane_tolerance` of the
diaphragm plane shares in-plane motion with a master node:

```python
g.constraints.rigid_diaphragm(
    "slab", "slab_master",
    master_point=(2.5, 2.5, 3.0),
    plane_normal=(0, 0, 1),
    plane_tolerance=0.05,
)
```

`rigid_body` welds a whole region to a master (all six DOFs), and
`kinematic_coupling` is the RBE2 of the family: a reference node rigidly drives
a node set with a per-set DOF selection, carrying the correct moment-arm
transport so an *offset* reference couples rigidly. Its flexible counterpart is
`distributing_coupling` — RBE3 — which distributes a force at a reference point
over a node set as a statically equivalent pattern while the set stays
deformable, with `weighting="area"` for a traction-like spread. Reach for RBE2
when the region must move rigidly (a loading platen); RBE3 when you want to
*introduce a load* without stiffening anything. Both emit dedicated Ladruno
fork elements: the deck is written on any build, but running it needs the fork.

## Non-matching meshes: ties

Fragmenting everything conformal is not always possible or desirable — a shell
floor on a solid column, two independently meshed blocks, a refined region
against a coarse one. `tie` handles displacement continuity across a
non-matching interface: each slave node is projected onto the closest master
face and its DOFs are interpolated from that face's corner DOFs through the
face's shape functions, `u_slave = Σ Nᵢ(ξ, η) · u_masterᵢ` — the same idea as
Abaqus `*TIE`.

```python
g.constraints.tie("flange_surface", "web_surface", tolerance=1.0)
```

The `tolerance` is the maximum projection distance — generous enough to bridge
any geometric gap, tight enough not to grab the wrong face. As a rule, make the
**finer** mesh the master: more shape functions to project onto. When a part
has many faces, scope the search with `master_entities=` / `slave_entities=`.

How the tie is *enforced* is a separate choice, exposed as `enforce=`. The
default `"penalty"` emits `ASDEmbeddedNodeElement` penalty elements — robust
and handler-independent. `"equation"` emits exact multi-point equations
(translations only) enforced by a constraint handler, and `"penalty_al"` uses
the fork's augmented-Lagrange penalty element. Start with the default;
switch routes when penalty stiffness becomes a conditioning problem or you
need the interface force exactly.

`tied_contact` is the surface-to-surface version of the same projection —
every slave-surface node tied to the master surface. It is one-directional
(slave conforms to master; pick the finer mesh as master), and takes the same
`enforce=` routes as `tie`.

The step-by-step recipe is in
[Tie non-matching meshes](../how-to/tie-meshes.md), and a complete worked model
in [the tie example](../examples/tie-non-matching-meshes.md).

## Buried parts: embedded and beam-to-solid

Two situations look like ties but aren't surface-to-surface. The first is
**embedding**: a lower-dimensional part living inside a host — rebar curves in
a concrete volume, stiffeners in a shell.

```python
g.constraints.embedded("concrete_volume", "rebar_lines")
```

Each embedded node is located inside a host element and constrained to follow
the host's displacement field through its shape functions. Non-simplex and
higher-order hosts are decomposed to linear sub-tets/sub-tris under the hood,
and embedded nodes that already coincide with a host corner are dropped —
they're attached through shared connectivity and constraining them again would
be redundant. Emission is `ASDEmbeddedNodeElement`, automatic like everything
else.

The second is the **DOF-mismatch bridge**: connecting a 6-DOF frame node to a
3-DOF solid face. `node_to_surface` builds a compound constraint — phantom
6-DOF nodes duplicated at the slave positions, rigid links from the master to
each phantom, `equalDOF` from each phantom down to the real solid node:

```python
g.constraints.node_to_surface("frame_end", "solid_face")
```

Here the master is a geometric *point* entity, not a part label — the one
place in the family where you target an entity rather than a name. The
`node_to_surface_spring` variant replaces the rigid links with stiff beam
elements; use it when the master carries free rotational DOFs under direct
moment loading, where the purely kinematic version leaves those rotations
without a stiffness path and the matrix conditioning suffers.

## Contact

Everything above is kinematic — a permanent bond, active from the first step.
`g.constraints.contact(...)` is the genuinely different animal: a face-to-face
contact interaction that can open, close, slide, and carry friction, emitted
through the Ladruno fork's contact subsystem.

```python
g.constraints.contact("wall_face", "soil_face",
                      formulation="nts", kn="auto", mu=0.4)
```

Two formulations: `"nts"` (node-to-segment penalty, with Coulomb friction) and
`"mortar"` (segment-to-segment augmented-Lagrange — the accuracy lane for
non-matching interfaces). Passing `tie=True` to the mortar formulation turns
it into a permanent *mesh-tie bond* — a mortar alternative to `tie` when you
want surface-integrated coupling rather than node collocation. Contact
declarations resolve additively onto `fem.elements.contacts` rather than the
MP-constraint channels, and like the fork coupling elements, the deck emits on
any build but runs only on the fork.

One historical note, since older material mentions it: `g.constraints.mortar()`
still exists but is a deprecated alias that delegates to
`contact(formulation="mortar", tie=True)` and warns. Call `contact()` directly.

## Reading back what you declared

After resolution the snapshot can tell you what actually happened —
`fem.inspect.constraint_summary()` prints the record counts by kind, and
`fem.nodes.constraints.summary()` / `fem.elements.constraints.summary()` give
DataFrame views. If a tie found zero pairs, the answer is almost always the
`tolerance`: too tight finds nothing, too loose couples nodes that shouldn't
be coupled. Check the summary before blaming the solver.

---

*Next: [Loads & masses](loads-and-masses.md).*
