# Meshing & partitioning

This page explains how a mesh happens in apeGmsh — who decides how big the
elements are, what `generate()` actually does, when a mesh is structured, what
element order changes, and why you'd split the finished mesh into partitions —
so you can predict the mesh a script will produce before you run it.

Meshing is the middle of the workflow spine: geometry goes in, a solver-ready
mesh comes out, and everything downstream — physical groups, constraint
resolution, the `FEMData` snapshot — reads from that mesh. The work lives under
`g.mesh`, split by concern like every other composite: `recipe` for one-call
meshing, `generation` for the mesh itself, `sizing` and `field` for element
size, `structured` for mapped grids, `editing` for post-generation surgery,
`queries` for reading the result out, and `partitioning` for parallel
decomposition. You don't memorize that list — when you wonder where a knob
lives, the concern is the answer.

## The topology comes first

A mesher can only be as good as the topology it's given, and the mistake worth
naming up front is *touching parts that don't share topology*. If two solids
merely occupy adjacent space, Gmsh meshes them independently: the interface
gets two clouds of nodes, coincident in space but unconnected, and any load
path or constraint crossing it is silently broken. The fix is to **fragment**
the assembly before meshing — fragmentation splits every shape at every
intersection, so the interface becomes one shared surface in the topology and
the mesher produces genuinely shared nodes on both sides:

```python
g.parts.fragment_all()
```

Fragment when parts must share nodes but remain distinct bodies (two materials
meeting at an interface); fuse when they're really one body that deserves one
label. The full assembly story — Parts, placement, fragmentation — is walked
end to end in [Multipart assembly](../examples/multipart-assembly.md). And when
two meshes *can't* conform — different element sizes on either side of an
interface — you skip fragmentation and tie them instead:
[Tie non-matching meshes](../how-to/tie-meshes.md).

## Recipes: the easy path

Most models don't need the granular controls below. `g.mesh.recipe` collapses
sizing, fields, transfinite setup, and generation into one call per region:

```python
# Whole model, one line each (generates immediately):
g.mesh.recipe.unstructured(min_size=0.2, max_size=1.0)   # tets/tris in the band
g.mesh.recipe.structured(size=0.5)                        # transfinite hex/quad

# Mixed model — compose region recipes, then generate once:
g.mesh.recipe.structured("soil_block", size=2.0, recombine=False)
g.mesh.recipe.unstructured("tunnel_liner", max_size=0.4)
g.mesh.generation.generate(dim=3)
```

Recipes are pure orchestration over the granular composites — nothing is
hidden, they just bake in the judgment calls you'd otherwise learn the hard
way. Whole-model `unstructured` disables size inheritance from imported CAD
points (see the sizing gotcha below); targeted `unstructured` sizes its region
through a background field rather than per-point lengths that bleed across
shared corners; `structured` falls back to an unstructured region when a shape
can't be hex-decomposed, so you always get a mesh. Recipes also run a
mixed-interface guard when generating in 3-D: a recombined-structured volume
sharing a face with an unstructured neighbor fails loud, because quads cannot
conform to tets and Gmsh has no pyramid transition. When the recipe's defaults
stop fitting — per-edge bias, boundary layers, algorithm choices — drop down to
the composites below. Everything still works before or after a recipe.

## Sizing: who decides how big an element is

Element size is negotiated between several inputs, and knowing the pecking
order saves real debugging time. The coarsest layer is a global floor and
ceiling:

```python
g.mesh.sizing.set_global_size(0.15)
```

But global bounds are not alone. Gmsh also consults *size sources*: per-point
characteristic lengths (on by default) and surface curvature (off by default).
The classic surprise is an imported STEP file whose points carry their own
sizes — your `set_global_size` call appears ignored because
`Mesh.MeshSizeFromPoints` is quietly overriding it. Make the global bound
authoritative again by switching that source off:

```python
g.mesh.sizing.set_size_sources(from_points=False)
```

For finer control there are per-point sizes (`set_size`, or a Python callback
that computes size from position), and at the top of the ladder sit **mesh
fields** — the right tool whenever you want "small elements *here*, large
elements *there*" driven by geometry rather than by hand-picked points. The
canonical refine-near-a-feature pattern is a distance field feeding a
threshold ramp:

```python
d = g.mesh.field.distance(curves=crack_tip_curves)
t = g.mesh.field.threshold(d, size_min=0.1, size_max=5.0,
                               dist_min=0.5, dist_max=10.0)
g.mesh.field.set_background(t)
g.mesh.generation.generate(3)
```

`g.mesh.field` has helpers for the common field types (distance, threshold,
box, boundary layer, min-combiners); prefer fields over uniform refinement —
`refine()` subdivides everything and is rarely what a production mesh wants.

## Generation: a thin, honest wrapper

```python
g.mesh.generation.generate(dim=3)
```

`generate(dim)` is deliberately a thin wrapper over Gmsh's own generator, and
that thinness is a contract: apeGmsh touches **no** mesh option behind your
back at session start. Every knob is either at Gmsh's factory default or was
set by a visible apeGmsh call — and once set, options stick on the session for
every subsequent `generate()` until you change them. You can always reach
through to `gmsh.option.setNumber(...)` for anything without a first-class
name; apeGmsh never hides Gmsh, it only promotes the knobs that matter often.

The `dim` argument is explicit because one session frequently carries several
dimensions at once — a solid with shells glued to its surface, beam lines
sharing nodes with a volume. `dim=3` runs the full cascade (1-D and 2-D meshes
are generated on the way), `dim=2` stops at surfaces, `dim=1` at curves. Note
that `get_fem_data(dim=...)` later selects the *element* dimension the solver
sees, which need not match the dimension you generated at.

Algorithms are chosen by name — strings are preferred, aliases and case
variations tolerated, typos raise a `ValueError` listing the real names:

```python
g.mesh.generation.set_algorithm(surf_tag, "frontal_delaunay_quads")
g.mesh.generation.set_algorithm(0, "hxt", dim=3)
```

Two asymmetries are worth internalizing. First, 2-D algorithms are set *per
surface* while the 3-D algorithm is a single global option — Gmsh has no
per-volume selection, so the `tag` is ignored at `dim=3`. Second, Gmsh's raw
3-D default is plain Delaunay, but apeGmsh's `"auto"` alias points at HXT — the
modern, parallel, robust choice for large models. If you never call
`set_algorithm(..., dim=3)`, you get Delaunay, not HXT; say `"hxt"` when you
want it.

Because mesh-control calls take raw entity tags but you build models in terms
of names, every per-entity command has a `*_by_physical` twin that fans out
over a named group — `set_algorithm_by_physical("Flanges", ...)`,
`set_recombine_by_physical(...)`, and friends. That keeps mesh setup in the
same name-driven register as the rest of the script; the naming machinery
itself is the next page's subject.

## Structured where it counts

An unstructured mesh (triangles, tets) goes anywhere; a **structured**
(mapped, transfinite) mesh trades generality for quality — regular quads or
hexes with node counts you control exactly. Structured meshing is opt-in and
regional: you mark curves with node counts and grading, then surfaces and
volumes whose boundaries carry compatible constraints, via
`set_transfinite_curve` / `_surface` / `_volume` — or let
`set_transfinite_automatic` find the mappable regions itself. Recombination
(`set_recombine`) is the separate step that merges triangles into quads, which
is why `recipe.structured` exposes a `recombine=` switch.

The by-physical wrappers make grading read like intent:

```python
g.mesh.structured.set_transfinite_by_physical(
    "WebEdges", dim=1, n_nodes=40, mesh_type="Progression", coef=1.1,
)
```

Structured and unstructured regions compose in one model — Gmsh uses the
transfinite path where you marked it and the general algorithm everywhere
else. The one hard incompatibility is the interface rule from the recipes
section: a recombined hex region cannot share a face with a tet region. Keep
the interface triangular (`recombine=False`) or structure the neighbor too.

## Element order

Elements are linear by default. `set_order(2)` promotes the whole mesh to
quadratic — 9-node quads, 10-node tets — and it's applied *after* generation
(it elevates the existing mesh in place; calling it before `generate()` does
nothing). It is also **global**, which creates one specific trap in mixed
models: 1-D curves you meant as OpenSees frame elements become 3-node lines,
and OpenSees beam-columns are strictly 2-node. The bridge fails loud rather
than guess; the resolution is to split those lines back to first order after
generating and before taking the snapshot:

```python
# Quadratic shells + 1st-order frame lines in one model
g.mesh.generation.generate(3).set_order(2)
g.mesh.editing.split_higher_order_lines("BeamLines", policy="split")
fem = g.mesh.queries.get_fem_data(dim=3)
```

`policy="split"` replaces each 3-node line with two 2-node lines — exact for
prismatic elastic frames (mind that distributed-plasticity elements get twice
the integration points). `policy="forbid"` is the build-time lock: fail if the
group picked up any higher-order line at all.

## Partitioning: one mesh, many ranks

Everything above produces one mesh for one solver process. **Partitioning**
splits it into `N` non-overlapping subdomains so that each OpenSeesMP rank
owns one — the prerequisite for `mpiexec -np N OpenSeesMP model.tcl`, and
worthwhile even on one machine, because MPI ranks over a genuinely parallel
distributed solver often beat a single process that serializes the factor.

```python
g.mesh.partitioning.renumber(dim=1, method="simple", base=1)
info = g.mesh.partitioning.partition(n_parts=4)
```

Renumber first, then partition: partitioning is a labelling pass over an
already-numbered mesh. The default backend is Gmsh's built-in METIS, which
balances by *element count* — fine when elements cost roughly the same to
solve. When they don't (fiber-section columns next to elastic beams can differ
by 10–100× per element), pass one weight per element and apeGmsh routes
through an external METIS binding to balance by *sum of weights* instead:

```python
weights = [...]            # one float per element
info = g.mesh.partitioning.partition(n_parts=4, weights=weights)
```

From there the decomposition simply flows: the snapshot carries it as
`fem.partitions`, and the OpenSees bridge brackets each rank's nodes and
elements in `if {[getPID] == K}` blocks, replicating cross-partition
constraints and declaring foreign nodes so a rigid diaphragm whose corners
scatter across ranks still gives the same answer as a single-process run. The
emitted deck stays runnable under plain OpenSees too — a small shim makes
`getPID` return 0 so only rank 0's block executes. Partitioning is opt-in and
zero-cost: an unpartitioned model (or one after
`g.mesh.partitioning.unpartition()`) emits byte-identical output to a model
that never heard of ranks.

---

*Next: [Selection & queries](selection.md).*
