# The OpenSees bridge

This page teaches you how a finished snapshot becomes a running OpenSees
model — why the bridge speaks typed objects instead of command strings, what
it carries across on its own, and how one set of declarations feeds a live
solve, a Tcl deck, a Python deck, and an HDF5 archive alike.

The [mental model](mental-model.md) ended with the hand-off: you build and
name a model in the session, freeze it into a `FEMData` snapshot, and the
solver side begins *after* that. The bridge is one class, `apeSees`,
constructed from the snapshot — not a session composite, and not dependent on
a live Gmsh kernel:

```python
from apeGmsh.opensees import apeSees

fem = g.mesh.queries.get_fem_data(dim=3)
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
```

`ops.model(ndm=, ndf=)` comes first — everything after it depends on the
declared dimensions. From there you declare materials, elements, supports,
loads, and the analysis itself on `ops`, and the bridge resolves every
declaration against the snapshot when it builds. Because the input is a
snapshot rather than a live session, the same `fem` can feed several bridges —
a gravity-only deck and a seismic deck from one geometry — and a snapshot
reloaded from `model.h5` works exactly like a fresh one.

## Typed handles, not strings

Raw OpenSees scripting is string-templated: you invent integer tags, spell
material names as strings, and a typo becomes a runtime error deep inside the
solver — or worse, a command OpenSees silently ignores. The bridge replaces
that with typed constructors. Every material, section, transform, and
integration rule is a Python call with an explicit signature, and it *returns
a handle* you pass to whatever consumes it:

```python
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(
    pg="Block", material=conc,
    body_force=(0.0, 0.0, -9.81 * 2400),
)
ops.fix(pg="Base", dofs=(1, 1, 1))
```

There are no tag numbers and no name registry — the variable *is* the
reference, and handles register themselves on the bridge. Misspell a keyword
and Python tells you at the call site; forget the `geomTransf` a beam needs,
pass a `dofs` mask of the wrong length, or assign an element type incompatible
with the declared `ndm`/`ndf`, and `build()` raises a pointed error before a
single solver line is written. That is the whole argument for typing: the
class of mistakes that OpenSees reports late, cryptically, or not at all is
moved to the moment you make them.

The typing goes one level deeper than signatures. Shells carry six DOFs per
node, solids three — and you never declare which: the bridge infers each
node's `ndf` from the elements you assigned to it, emits per-node DOF counts
where they differ from the model envelope, and fails loud if two elements
disagree about a shared node (a shell meeting a solid), because OpenSees would
mis-assemble that node rather than complain.

## Everything targets a name

Notice what the snippet above never does: loop over nodes. `pg="Block"` on the
element call writes every mesh element in that physical group;
`pg="Base"` on the fix clamps every node on that face, however many the mesher
produced. The same `pg=` keyword appears on masses, loads, prescribed
displacements, and recorders, and it resolves both physical groups and
apeGmsh labels against the snapshot. This is the naming discipline from the
[selection page](selection.md) paying off on the solver side: the mesh can be
refined, the geometry edited, the node numbering reshuffled — the bridge
script does not change, because it never mentioned a node.

## What crosses the boundary by itself

The session let you declare physics — loads, masses, constraints — before the
mesh existed. How much of that reaches the deck automatically? The answer is
deliberately narrow: **multi-point constraints, and nothing else.**

Ties, embedded regions, rigid links and diaphragms, and their relatives
declared via `g.constraints.*` resolve into the snapshot and emit into the
deck without any bridge-side step:

```python
g.constraints.tie("ColumnTop", "SlabBottom", dofs=[1, 2, 3])
# ... mesh, snapshot, apeSees(fem), materials, elements ...
ops.tcl("model.tcl")   # the tie lines appear automatically
```

You declare the tie once, where the geometry lives; the bridge writes the
`equalDOF` / `rigidLink` / `rigidDiaphragm` / embedded-element lines and
switches on the constraint handler they need. Constraints auto-emit because
they are part of what the model *is* — a tied interface is meaningless to
omit.

Loads are different: which load cases belong in *this* analysis is a per-deck
decision, so session loads are **opt-in**. The session groups loads by *case*
— a label with no temporal meaning. Time enters on the bridge, where a
*pattern* owns a time series, and the pattern imports the cases it wants:

```python
ts = ops.timeSeries.Linear()
with ops.pattern.Plain(series=ts) as p:
    p.from_model("dead")                       # import the resolved "dead" case
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))  # + ad-hoc bridge-authored load
```

`p.from_model(case)` replays the resolved nodal records of a session case
inside the pattern; `p.load` / `p.sp` author loads directly on the bridge.
A case you don't import is simply not applied — there is exactly one channel
into the deck, so the old double-counting trap (declared on the session *and*
the bridge) cannot happen. Masses and support fixities are re-declared
explicitly with `ops.mass` and `ops.fix` for the same reason: the deck is
authoritative, and reading it tells you exactly what the analysis contains.
Everything the session declared — imported or not — still persists into
`model.h5` for the viewer and `Results`.

## Stages

Real analyses rarely happen in one shot: gravity first, then pushover; excavate,
then install the lining, then shake. The bridge models this directly with
`ops.stage`. A stage is a scoped block that owns its own patterns, its own
analysis chain, and its own run:

```python
with ops.stage(name="push") as s:
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("live")
    s.analysis(...)
    s.run(n_increments=10, dt=0.1)
```

Loads from a finished stage are frozen (`loadConst`) before the next begins,
stages can bring element groups online with `s.activate(pgs=[...])`, and an
MP constraint whose nodes only exist from a later stage onward is claimed by
name inside that stage (`s.tie(name=...)`) instead of emitting globally. A
model is either staged or not — a global pattern alongside stage blocks
raises at build, because it would double-apply across stage boundaries. The
[staged-construction example](../examples/staged-gravity-ssi.md) shows the
idiom at full length.

## One declaration surface, four outputs

Everything above — model, materials, elements, fixes, patterns, stages, and
the analysis chain itself (`ops.constraints`, `ops.numberer`, `ops.system`,
`ops.test`, `ops.algorithm`, `ops.integrator`, `ops.analysis`) — is pure
declaration. Nothing touches OpenSees until you pick an output:

```python
ops.tcl("model.tcl")            # complete OpenSees Tcl deck
ops.py("model.py")              # equivalent openseespy script
ops.h5("model.h5")              # native HDF5: solver zone + neutral zone
ops.run()                       # in-process openseespy, right now
ops.analyze(steps=10, dt=0.01)
```

Each call builds the same resolved model internally and hands it to a
different emitter; they are separate statements, not a chain, and you can call
several on one bridge. This is the payoff of declare-then-emit: the deck you
archive, the script a colleague runs, and the in-process solve are guaranteed
to be the *same model*, because they came from the same declarations. The
`model.h5` written here carries both the runnable solver zone and the neutral
zone the viewer reads — the session-side `g.save()` writes only the latter
(see [Save & reload a model](../how-to/save-reload.md)).

## Recorders, and reading back

Output declarations live on the same surface. `ops.recorder.<Type>(...)`
declares classic OpenSees recorders that emit with the deck; for in-process
runs, a capture spec names what to record — by physical group, of course —
and writes results next to the model:

```python
spec = DomainCaptureSpec(opensees=ops)
spec.nodes(pg="Tip", components=["displacement"])
with ops.domain_capture(spec, path="run.h5") as cap:
    cap.begin_stage("tip_load", kind="static")
    ops.analyze(steps=1)
    cap.step(t=1.0)
    cap.end_stage()
```

You saw this run end-to-end in [the first tutorial](../tutorials/first-model.md):
the same `"Tip"` name that placed the load fetches the deflection back out of
`Results`. That symmetry — names in, names out — is where the bridge hands
over to the results layer, which the next page covers.

The bridge's surface is larger than this page: eigen and modal analysis,
damping declarations, fiber sections, explicit dynamics, and more all live on
the same typed namespaces. The [OpenSees API reference](../api/opensees.md)
is the complete signature-level map, and the [examples](../examples/index.md)
— [modal analysis](../examples/modal-analysis.md),
[pushover](../examples/pushover-steel-frame.md),
[non-matching-mesh ties](../examples/tie-non-matching-meshes.md) — carry the
full workflows. What stays constant everywhere is the concept you now have:
typed declarations, resolved against a named snapshot, emitted wherever you
need them.

---

*Next: [Results](results.md).*
