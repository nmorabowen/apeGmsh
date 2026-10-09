# Assemble saved models

This page shows how to build one OpenSees model out of saved `model.h5`
files with `Assembly`: place each file as a named instance, join the
instances with ties and couplings, declare the analysis on one bridge,
write the deck and the archive, and give each instance its own MPI rank.
It assumes you can already save a model with its materials and elements
([Save & reload a model](save-reload.md)).

## The recipe

The example saves one steel block, places it twice, and stacks the two
copies. The upper copy is turned half a turn about the vertical axis and
lifted onto the lower one; a tie joins the two faces that meet, and a
reference node above the stack carries the load through a rigid cap. With
Poisson's ratio zero the stack is two springs of `EA/H` in series, so the
cap must settle by `P·2H/(E·A)`.

```python
from math import pi

import openseespy.opensees as osp

from apeGmsh import apeGmsh
from apeGmsh.assembly import Assembly
from apeGmsh.opensees import apeSees

E, SIDE, H = 200_000.0, 200.0, 400.0          # N, mm, MPa

# 1. One model, saved once: mesh, groups, material and elements.
with apeGmsh(model_name="block", save_to="block_mesh.h5") as g:
    g.model.geometry.add_box(0.0, 0.0, 0.0, SIDE, SIDE, H, label="body")
    g.physical.add_volume("body", name="Vol")
    faces = g.model.select(None, dim=2)
    faces.in_box((-1, -1, -1), (SIDE + 1, SIDE + 1, 1)).to_physical("bot")
    faces.in_box((-1, -1, H - 1), (SIDE + 1, SIDE + 1, H + 1)).to_physical("top")
    g.mesh.recipe.structured(size=100.0, fallback="strict")
    fem = g.mesh.queries.get_fem_data(dim=None)   # dim=None: ties need faces

block = apeSees(fem)
block.model(ndm=3, ndf=3)
steel = block.nDMaterial.ElasticIsotropic(E=E, nu=0.0, name="steel")
block.element.stdBrick(pg="Vol", material=steel)
block.h5("block.h5")

# 2. Two instances of the file, tied, with a reference node on a rigid cap.
asm = (
    Assembly("stack")
    .instance("lower", "block.h5")
    .instance("upper", "block.h5",
              rotate=((0.0, 0.0, 1.0), pi), translate=(SIDE, SIDE, H))
    .tie("lower.top", "upper.bot", enforce="equation", dofs=[1, 2, 3])
    .node("cap", (SIDE / 2, SIDE / 2, 2 * H))
    .rigid_link("cap", "upper.top", link_type="rod")
)

# 3. One bridge. Supports, loads and the analysis are declared here.
ops = asm.bridge(ndm=3, ndf=3)
(cap,) = ops.fem.nodes.select(label="cap").ids
P = 10_000.0
ops.fix(pg="lower.bot", dofs=(1, 1, 1))
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
    pat.load(node=int(cap), forces=(0.0, 0.0, -P))
ops.constraints.Lagrange()
ops.numberer.Plain()
ops.system.FullGeneral()
ops.test.NormUnbalance(tol=1e-6, max_iter=10)
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1.0)
ops.analysis.Static()

ops.tcl("stack.tcl")                     # serial: no instance has a rank
ops.tcl("stack_flat.tcl", flat=True)     # the same deck
asm.h5("stack.h5")                       # the bridge's model.h5 + /assembly

# 4. The archive lists what was declared.
again = Assembly.from_h5("stack.h5")
assert [i.label for i in again.instances] == ["lower", "upper"]
assert [n.name for n in again.nodes] == ["cap"]
assert [t.master for t in again.ties] == ["lower.top", "cap"]
print("reloaded:", [i.label for i in again.instances],
      [t.master for t in again.ties])

# 5. Solve, and check against the closed form.
ops.run()
ops.analyze(steps=1)
u = osp.nodeDisp(int(cap), 3)
exact = -P * 2 * H / (E * SIDE * SIDE)
print(f"cap settles {u:.6e} mm, closed form {exact:.6e} mm")
assert abs(u - exact) <= 1e-9 * abs(exact)

# 6. One rank per instance, for OpenSeesMP.
ranked = (
    Assembly("stack_mp")
    .instance("lower", "block.h5", partition_rank=0)
    .instance("upper", "block.h5", partition_rank=1,
              rotate=((0.0, 0.0, 1.0), pi), translate=(SIDE, SIDE, H))
    .tie("lower.top", "upper.bot", enforce="equation", dofs=[1, 2, 3])
)
ranked.bridge(ndm=3, ndf=3).tcl("stack_mp.tcl")
deck = open("stack_mp.tcl", encoding="utf-8").read()
print("getPID blocks:", deck.count("if {[getPID] =="))
serial = open("stack.tcl", encoding="utf-8").read()
assert "getPID" not in serial
assert serial == open("stack_flat.tcl", encoding="utf-8").read()
print("serial deck:", serial.count("element stdBrick"), "bricks, no getPID")
```

Run against stock openseespy, it prints:

```text
reloaded: ['lower', 'upper'] ['lower.top', 'cap']
cap settles -1.000000e-03 mm, closed form -1.000000e-03 mm
getPID blocks: 2
serial deck: 32 bricks, no getPID
```

The settlement matches `P·2H/(E·A) = 1.0e-3 mm` to round-off, which is
what an exact tie on matching faces and a rigid cap must give.

## Instances and names

`instance(label, source)` places a file. Every name the file owns comes in
as `{label}.{name}`, so the block's `top` group is `lower.top` in one
instance and `upper.top` in the other, and its `steel` material becomes two
materials, `lower.steel` and `upper.steel`. There is no host: every
instance is namespaced the same way, and the file is read once however
many times you place it. `rotate=((ax, ay, az), theta)` turns the instance
about the origin by `theta` radians, and `translate=` moves it after the
rotation. A label must be non-empty and free of `.`, `/` and whitespace,
and may not start or end with `_`; instance labels, reference nodes and tie
names share one namespace, so a repeated name raises `AssemblyError`.

## Ties and couplings

Every port is `"{instance}.{group or label}"`. `tie(master, slave)`
resolves exactly as `g.constraints.tie` does on a non-matching interface;
`enforce="equation"` is exact and needs the Lagrange handler. The other
verbs follow the session's constraint vocabulary: `equal_dof` for
co-located nodes, `rigid_link` and `rigid_diaphragm` for rigid bodies,
`embedded` for nodes of one instance inside the elements of another, and
`couple(target, kind="kinematic" | "distributing", reference=)` for RBE2
and RBE3, which emit Ladruno-fork elements that stock OpenSees refuses.
`node(name, coords)` declares a reference node owned by the assembly; it
is a port of `equal_dof`, `rigid_link` and `rigid_diaphragm` and the
`reference` of `couple`, and you read it back by label, as step 3 does.
`rigid_diaphragm` needs 6 DOFs on its nodes in 3-D, which is why a brick
model like this one uses a `rigid_link` rod. A tie or coupling whose ports
exist but couple nothing raises at `bridge()`.

## What an instance carries

Model content travels with the instance; analysis content belongs to the
assembly. The mesh, groups, labels, intra-instance constraints (ties,
embeds, interfaces and contacts the source declared inside itself) and the
rebar stream travel, and from the source's solver zone so do its
materials, sections, transforms, beam integrations, dampings attached
through an element's `damp=` and its element specs. The source's fixes,
masses, load patterns, recorders, stages and analysis stay behind: declare
them on the bridge, as step 3 does.

## Decks and ranks

An assembly whose instances carry no `partition_rank` is serial: `tcl()`
writes one unpartitioned deck, identical to `tcl(flat=True)`. Give every
instance `partition_rank=k`, one instance per rank and ranks running
`0 .. n-1`, and `tcl()` writes one `getPID` block per rank. Reference
nodes live on rank 0 and are declared on every other rank a coupling
needs them, and each tie across instances is written on every rank that
owns one of its nodes, with the same tags. Without your own numberer and
system, the partitioned deck picks `ParallelPlain` and `Mumps` at run time,
with serial fallbacks, and says so in a warning.

## The archive

`asm.h5(path)` writes exactly the `model.h5` the bridge's own `ops.h5`
writes, plus an `/assembly` zone that lists the instances and ties.
Every reader opens it unchanged, and `OpenSeesModel.from_h5` rebuilds the
model from it. `Assembly.from_h5(path)` reads only the listing: it never
re-reads the instance files, and the assembly it returns cannot
`bridge()` or `h5()` again. Declare supports, loads and the analysis
before `h5()`, and call `bridge()` again if you declare another instance
or tie after it, or `h5()` raises.

## What refuses

Each of these raises `AssemblyError`, at the declaring call or at
`bridge()`, so no partial model comes back:

- `contact` and `interface` across instances. They are not assembly
  couplings; declare them inside the source model, where they travel.
- Element rows whose arguments vary inside one physical group. Each group
  rehydrates as one element spec; the per-row selector is not built yet
  ([#1542](https://github.com/nmorabowen/apeGmsh/issues/1542)).
- A damping attached by region, globally or in a stage. Attach it with an
  element's `damp=` in the source, or declare it on the bridge.
- A material, section, transform, integration or element type the
  assembly does not rehydrate yet. The error lists the supported ones.
- A source whose own modules carry a partition rank, such as a ranked
  assembly archive, checked at `instance()` and again at `bridge()`, and a
  `partition_rank` on an instance whose source is itself an assembly
  archive.

## The older compose path

`g.compose`, `FEMData.compose`, `apeGmsh.compose` and the `Assembly`
verbs `add`, `couple(kind=, ports=)` and `materialize` are the first
version of this feature. They are to be removed without a deprecation
period once `Assembly` covers their uses ([ADR 0117
D7](https://github.com/nmorabowen/apeGmsh/blob/main/architecture/decisions/0117-assembly-compose-v2.md)).
Start new models with `instance`, `tie` and `bridge`; one assembly cannot
mix the two APIs.

---

*Next: [Compose modules](compose-modules.md), which documents the older
path for models that still use it; or go on to
[Apply gravity / self-weight](gravity.md).*
