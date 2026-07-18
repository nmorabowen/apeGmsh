# Loads & masses

This page explains how the loads, prescribed displacements, and masses you
declare against *names* become the per-node numbers a solver actually
consumes — the declare-then-resolve pipeline, why everything resolves to
nodal records, the split between geometry-side *cases* and bridge-side
*patterns*, and where mass differs from weight.

## Declare, then resolve

Loads follow the declare-then-resolve timing from
[the mental model](mental-model.md), and it is worth spelling out what the
two stages actually do. When you call a load verb, nothing is computed. The
call records a small definition object — *"apply −3 kN/m² to the surfaces
named `slabs`"* — holding the target name, the magnitudes, and the active
load case. It can't compute anything: the nodes it will eventually act on
don't exist yet, because the mesh doesn't. Resolution happens inside
`get_fem_data(...)`, where every definition is matched against the real
mesh and turned into per-node force records on the snapshot,
`fem.nodes.loads`.

This ordering is what makes a load script durable. You can refine the mesh,
swap a part, or rebuild the geometry, and the same declarations re-resolve
against whatever mesh exists now — which is also why load targets are
names (labels, physical groups, part labels, mesh selections), never raw
Gmsh tags. A typo in a target name doesn't wait until resolution to bite:
targets are validated when `g.mesh.generation.generate(...)` runs, before
any expensive mesh work.

The verbs are indexed by the dimension of what they act on, so the API
reads like the load itself:

```python
g.loads.point.force / .moment / .force_closest / .moment_closest
g.loads.line(...)          # distributed force per unit length
g.loads.surface.pressure / .traction / .shear / .force_resultant_center_mass
g.loads.volume(...)        # generic body force per unit volume
g.loads.gravity(...)       # self-weight: rho * g on a volume
```

A point load gives every node of its target the *same* force — so aim it
at a single-node target, or use `point.force_closest(xyz, ...)` to snap a
coordinate to the nearest mesh node
([how-to](../how-to/point-load.md)). A surface `pressure` is a scalar
acting perpendicular to each face, so it follows a sloped or curved
surface without you resolving components; `traction` is a vector applied
identically on every face regardless of orientation
([how-to](../how-to/face-pressure.md)). And when you need a resultant
force or moment on a solid face with no structural node to hang it on,
`surface.force_resultant_center_mass` splits the force equally across the
face nodes and converts the moment into a statically exact,
minimum-norm set of nodal forces about the face centroid.

## Everything becomes nodal

Resolution has one job: turn a distributed intent into equivalent nodal
forces. The default strategy is **tributary** — each node gets its
geometric share of the total. A uniform line load `q` on an edge of length
`L` puts `q·L/2` on each end node; a pressure `p` on a triangular face of
area `A` puts `p·A/3` on each corner. It is exact for uniform loads on
linear elements and, crucially, it is *checkable*: sum the resolved forces
and you should recover `p·A` or `ρ·g·V` by hand. The alternative,
`reduction="consistent"`, integrates the load against the element shape
functions instead. For linear elements the two coincide; consistent earns
its keep on quadratic elements, where mid-side nodes need the correct
weighting. Rule of thumb: tributary unless your elements are higher-order
and the answer depends on it.

There are, deliberately, no element loads. You will not find an `eleLoad`
anywhere in this pipeline — every `g.loads` verb resolves to nodal force
records, and the OpenSees bridge consumes nothing else (ADR 0051). Nodal
records are the portable currency: every solver understands "apply this
force to this node", they survive the `model.h5` round-trip, and they
compose across modules. The honest cost is that nodal lumping of a beam's
span load drops the fixed-end moments the native `beamUniform` would
carry — a known, logged trade, not an oversight. If that moment
distribution matters to your answer, refine the beam into more elements.

## Cases on the geometry, patterns on the bridge

Load declarations group into named **cases**:

```python
with g.loads.case("dead"):
    g.loads.gravity("concrete", g=(0, 0, -9.81), density=2400)
    g.loads.line("beams", magnitude=-2e3, direction="z")

with g.loads.case("live"):
    g.loads.surface.pressure("slabs", -3.0e3)
```

A case is a grouping label and nothing more. It has no time series, no
analysis stage, no temporal meaning — those are properties of an OpenSees
*pattern*, and a pattern is an analysis-time decision the geometry has no
business making. The same `"dead"` case can ramp linearly in one deck and
be held constant in another; the declaration doesn't change.

The consequence is that loads are **opt-in** at the bridge. Nothing you
declare on `g.loads` reaches the OpenSees deck by itself; a bridge pattern
must *import* the case:

```python
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("dead")
    p.from_model("live")
```

`from_model(case)` replays every resolved record tagged with that case as
a `load` (or `sp`) line inside the pattern, scaled by the pattern's time
series, and mixes freely with ad-hoc `p.load(...)` calls. The deck is
authoritative: the bridge applies exactly the cases you import, no audit
of what the geometry declared. A case you don't import is simply not
applied — which also means there is no double-count trap, because nothing
is ever applied twice behind your back. One geometry, many decks — a
gravity-only deck, a gravity-plus-lateral deck — is the intended workflow,
not an error.

## Gravity

Self-weight is common enough to get its own verb. Instead of computing
`ρ·g` into a body-force vector yourself, you write the physics:

```python
g.loads.gravity("concrete_columns", density=2400)   # g defaults to (0, 0, -9.81)
```

Two things about it are easy to trip on. First, gravity targets must be
**volumes** — aimed at a surface or curve it resolves to zero records,
because self-weight of a shell belongs on the shell section's thickness
and density, not on a load. Second, both `g` and `density` are
unit-sensitive: a kg-mm-s model wants `g=(0, 0, -9810)`, and a density in
g/cm³ instead of kg/m³ is a silent factor-of-a-thousand error. The cheap
insurance against both is the same sanity check tributary reduction makes
possible — after `get_fem_data`, sum `force_xyz` over the case and compare
against `ρ·g·V` by hand. The full recipe, including the checks, is in
[Apply gravity](../how-to/gravity.md). Any body force that *isn't*
gravity — centrifugal, magnetic, a thermal driving force recast as a body
load — goes through `g.loads.volume(...)`, which takes the full
force-per-volume vector with no density term.

## Prescribed displacements

Prescribed motion is force-free loading, and it lives on its own
composite, `g.displacements` — deliberately separate from both `g.loads`
(which owns forces) and `g.constraints.bc` (which owns permanent
homogeneous fixes). The ownership rule: if a support is fixed at zero
forever, declare it through `g.constraints.bc`; if a value is imposed —
a settlement, a pushover displacement, anything nonzero or time-varying —
it belongs here. A zero authored through `g.displacements` is allowed,
but it is a pattern-bound *hold* you chose explicitly, not an alias for a
fix.

```python
# Prescribed translation at a face centroid, mapped rigidly to its nodes
g.displacements.surface(pg="base_face", dofs=[1, 1, 1], disp_xyz=(0.01, 0, 0))

# The same value applied verbatim at every node of the target
g.displacements.point("col_base", dofs=[0, 0, 1], values=[0, 0, -0.02])
```

The `surface` verb treats the face as rigid: each node receives
`u_i = disp_xyz + rot_xyz × r_i`, where `r_i` runs from the face centroid
to the node, so you can prescribe a rotation of a whole face in one call.
The `point` verb skips the mapping and writes the values directly.
Prescribed displacements group into cases exactly like loads and ride the
same opt-in import — `from_model(case)` emits them as `sp` lines inside
the pattern, scaled by its series. Homogeneous fixes are never imported
that way; they are model-level and emit through `ops.fix(...)`.

## Masses

Masses are declared like loads — same dimension-indexed verbs, same
name-based targets, same resolve-at-`get_fem_data` timing — but they are
not loads, and the differences are the concept. There are no cases: mass
is intrinsic to the model, not something you switch between scenarios.
Definitions that overlap simply accumulate per node, so each part can
declare its own mass without double-counting at shared boundaries.

```python
g.masses.volume("Slab", density=2400)          # solid: rho per unit volume
g.masses.surface("Slab", areal_density=200)    # non-structural finishes, per m^2
g.masses.line("Cladding", linear_density=120.0)
g.masses.point("tip_node", mass=500.0, rotational=(10.0, 10.0, 50.0))
```

Resolution mirrors the loads pipeline with mass in place of force. The
default reduction is `"lumped"` — each element's mass splits equally
among its nodes, giving the diagonal mass matrix explicit dynamics
requires and a total you can verify by hand — with `"consistent"`
available for the same higher-order reasons as loads. Every resolved
record carries a full six-component vector `(mx, my, mz, Ixx, Iyy, Izz)`;
the bridge slices it to the model's DOF count at emit time, so the same
snapshot serves a 3-DOF solid and a 6-DOF frame. The one number to always
check is `fem.nodes.masses.total_mass()` against a hand calculation — a
density unit mistake shows up there instantly.

On the bridge, masses follow the model-definition side of the split, not
the load-case side: they emit through an explicit `ops.mass(...)`, or in
one stroke with `ops.mass_from_model()`, which streams every resolved
nodal mass from the snapshot into the deck. And keep mass and weight
distinct in your head, because the library does: `g.masses` populates the
mass matrix (dynamics, eigenvalues), `g.loads.gravity` applies a force
(statics). A dynamic model under gravity legitimately declares both from
the same density — that is the physics, not a double count.

---

*Next: [The FEM broker](fem-broker.md).*
