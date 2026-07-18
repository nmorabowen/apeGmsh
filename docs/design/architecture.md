# Architecture

This page explains how apeGmsh is built — the layers, the one data
hand-off at the center, and the boundaries between modules — so you can
predict where any piece of behavior lives before you open the source.
The [mental model](../concepts/mental-model.md) teaches how to *use* the
library; this page is about how it is *made*.

## The shape of the library

apeGmsh is a pipeline with one hard boundary in the middle:

```
live Gmsh session (g)  →  FEMData broker (frozen)  →  apeSees bridge  →  Results
      mutable                  the hand-off             post-session     post-run
```

Everything on the left of the broker is a thin, stateful wrapper around
one live Gmsh kernel. Everything on the right is pure Python and numpy
that never imports gmsh. The broker — `FEMData`, produced by
`g.mesh.queries.get_fem_data(...)` — is a *fork*, not a view: it copies
the mesh, the names, and every resolved physics record into frozen
arrays and dataclasses, and from that moment the solver bridge, the
results system, and persistence all work from the snapshot. You can
close the Gmsh session, pickle the snapshot, or reload it years later
from `model.h5`; nothing downstream will notice.

Five invariants, in dependency order, hold the design together:

1. **`(dim, tag)` is ground truth.** Gmsh identifies every entity by a
   dimension–tag pair, and tags are not stable across boolean
   operations. apeGmsh never wraps them in a persistent handle — the
   user holds a *name*, and the library keeps the name's tags current.
2. **Names survive OCC booleans.** Labels and physical groups are
   remapped through every fragment, fuse, and cut. This is the single
   biggest ergonomic delta from raw Gmsh.
3. **Pre-mesh intent is decoupled from mesh realization.** Loads,
   masses, constraints, and displacements are declared against names
   before any node exists, and resolved against the real mesh at
   snapshot time.
4. **The broker is a fork, not a view.** After `get_fem_data()`, no
   downstream consumer touches gmsh.
5. **Resolvers are pure; composites are impure.** All mesh math lives
   in functions on numpy arrays with zero gmsh imports; all gmsh state
   lives in the session's composites.

Everything else — parts, sections, contact, the viewers — is built on
these five.

## One session, many composites

The session object `g` is a container, not a base class. When
`begin()` runs, it walks a plain class-level list of
`(attribute, module, class)` entries and attaches one composite per
concern: `g.model` (with `geometry`, `boolean`, `transforms`, `io`,
`queries` beneath it), `g.mesh` (with `sizing`, `generation`, `field`,
`structured`, `editing`, `queries`, `partitioning`), and the physics
composites — `g.loads`, `g.masses`, `g.constraints`,
`g.displacements`, plus the reinforcement family (`g.reinforce`,
`g.rebar`, `g.embed`). No metaclass, no registration decorators: a
reader tracing where `g.mesh` comes from finds one tuple in
`_core.py`. A few surfaces are session-level facades rather than
composites — `g.compose(...)`, which grafts a saved `model.h5` into
the current session as a subassembly, is the notable one.

Three class flavors recur everywhere, and telling them apart tells you
what a class is allowed to do:

- **Composites** (`Model`, `Mesh`, `LoadsComposite`, `PartsRegistry`)
  are stateful, hold a reference to the session, and call gmsh. They
  are integration-tested against live kernels.
- **Defs** (`PointLoadDef`, `EqualDofDef`, `ContactDef`) are mutable
  dataclasses recording pre-mesh intent — a name, a magnitude, some
  options. No mesh math.
- **Records** (`NodePairRecord`, `NodalLoadRecord`, `MassRecord`,
  `ContactRecord`) are frozen dataclasses emitted by resolvers and
  stored on `FEMData`. They are the vocabulary the solver bridge reads.

The def → resolver → record pipeline is the same for every physics
concern. The composite collects defs; at snapshot time it hands the
defs plus the relevant node and element arrays to a resolver; the
resolver returns records that are frozen into the broker. Resolvers
import numpy and nothing else, which is why a 700-line constraint
resolver stays testable with hand-built arrays and no mesh.

## Names, not tags

Gmsh already has a naming mechanism — the physical group — and its
machinery participates in OCC booleans. Rather than build a parallel
registry, apeGmsh splits the physical-group namespace into two tiers:
**labels**, stored as physical groups with a reserved `_label:` prefix
and invisible to `g.physical`, and **user physical groups**, the
solver-facing names. One remapping routine then serves both tiers.

That routine is the load-bearing wall. Gmsh does not migrate
physical-group memberships when a boolean renumbers tags — it leaves
them pointing at dead tags, and the model breaks silently on the next
query. So every boolean in apeGmsh runs inside a snapshot/remap
envelope: capture every group's membership before the operation, run
the OCC call, then rebuild the groups from the operation's result map.
Fragment (one input splits into many), fuse (many inputs absorbed into
one survivor), and cut (object follows the fragment rule, the tool is
consumed with a warning) each get their own remap semantics, and all
three are exercised across dimensions in the test suite. Every
label-aware API downstream — parts fragmentation, plane cuts,
constraints that target labels, every `get(pg=...)` query on the
broker — relies on this one promise holding.

## Parts and the sidecar trick

A `Part` is a standalone, isolated Gmsh session: it initializes its own
kernel, builds geometry as a mini-model, and on exit persists itself to
a STEP tempfile. The main session imports that STEP — the two kernels
never overlap, which is how apeGmsh works around Gmsh's process-global
state.

STEP carries geometry but no Gmsh metadata, so labels would die at the
boundary. The fix is a sidecar file written next to the STEP,
recording `(dim, label, center-of-mass)` for every labeled entity. On
import, apeGmsh applies the requested transform, computes the center of
mass of each fresh entity, and greedily matches against the sidecar to
rebind labels — deterministically, with ties broken by tag order. Each
instance then prefixes its label onto every part-side name, so two
columns placed from the same part get independent namespaces
(`col_A.shaft`, `col_B.shaft`).

There is deliberately no `Assembly` class. The assembly is emergent —
whatever set of instances the parts registry currently tracks — and
`g.parts.fragment_all()` makes it conformal by fragmenting everything
against everything, with the boolean-survival machinery carrying the
names through. Making the assembly a type would add a layer without
adding a verb.

## The broker

`FEMData` holds two query composites — `fem.nodes` and `fem.elements`,
each with ids, coordinates or connectivity, and a composable
`get(pg=..., label=..., partition=...)` filter — plus the resolved
record sets for constraints, loads, single-point conditions, masses,
and contact. It is frozen because reproducibility demands it: a live
view would mean "whatever gmsh holds right now," while a fork means
"the state at the moment of the snapshot," which is the only thing you
can hash, cache, and compare across runs.

The broker is also the persistence unit. `fem.to_h5()` /
`FEMData.from_h5()` round-trip the entire snapshot — mesh, names,
records, and the model's provenance — through a versioned neutral HDF5
schema (`model.h5`), and `apeGmsh.from_h5()` can rehydrate a session
around a saved snapshot with no gmsh state at all. `g.compose()`
builds on the same file format to assemble multi-model systems from
independently authored and saved models. The schema is versioned and
append-only precisely because saved models outlive library releases.

## The bridge is thin on purpose

`apeSees(fem)` is the OpenSees adapter — a post-session class
constructed from the snapshot, never from the session. (An in-session
`g.opensees` composite existed once and was deliberately torn down;
ADR 0009 records why.) It never imports gmsh; every node, element, and
name it needs comes through the broker.

The bridge stays thin because the heavy lifting happened upstream: the
records are already close to solver-command shape, so emission is
mostly one line per record. Its own machinery is a registry mapping
mesh element types to OpenSees element commands (with node reordering
and material-family metadata), typed-primitive namespaces
(`ops.nDMaterial.*`, `ops.section.*`, `ops.element.*`,
`ops.pattern.*`, ...) whose constructors return handles rather than
string tags, and a staged-analysis orchestrator (`ops.stage`) for
sequential load histories. Emission targets are a Tcl deck, an
openseespy script, an in-process `ops.run()`, or remote HPC
submission.

What crosses the boundary automatically is a deliberate, narrow list.
Multi-point constraints — `equalDOF`, rigid links, diaphragms, embedded
ties — auto-emit from the broker's records, because they are model
definition (ADR 0022). Loads and imposed displacements are *opt-in*:
you pull a resolved load case into a bridge pattern with
`p.from_model("dead")`, or author loads directly on the bridge —
nothing is silently injected into your load history (ADR 0051). Fixes
and masses are declared explicitly on `ops`. The rule of thumb: the
model's topology flows through automatically; the analysis narrative
is always authored.

Nothing in the broker or the resolvers mentions OpenSees. A second
adapter would consume the same records; if a record cannot express
what a new solver needs, the record grows a field — the adapter never
compensates with its own math.

## Beyond the core pipeline

Three younger subsystems follow the same patterns rather than invent
new ones. The **sections** package (ADR 0078) is both a parametric
geometry builder (`g.sections`) and a standalone cross-section
analyzer — a small internal FE solver for torsion, warping, shear, and
plastic properties that feeds section constants to the bridge. The
**contact** family extends the constraints ladder with fork-native
contact records (ADR 0073), resolved and persisted like every other
record type. And **results** is a three-reader post-processing system:
`Results.from_native(...)` for the library's own recorder stream,
`from_mpco(...)` for STKO/MPCO HDF5 output, and `from_ladruno(...)`
for the Ladruno fork's format — all normalizing into one query and
plotting surface with an interactive viewer and a shareable web view.

The interactive viewers (Qt + PyVista, an optional install) are the
one sanctioned exception to the broker boundary: scene builders read
gmsh directly, because geometry views exist before any broker does and
mesh views would otherwise copy large arrays. The rule they observe is
read-only — viewers may read gmsh for scene construction but never
mutate model state. All heavy dependencies (Qt, PyVista, openseespy,
matplotlib) are imported lazily at call time, so the core installs and
runs headless with only gmsh, numpy, and pandas.

## Where the decisions live

Every non-obvious choice above — and roughly eighty more — is recorded
as an append-only ADR next to the code it governs. When this page says
"deliberately," the ADR says why, with the alternatives that lost. The
index is at
[the decisions log](https://github.com/nmorabowen/apeGmsh/tree/main/src/apeGmsh/opensees/architecture/decisions).

---

*Next: [Principles](principles.md).*
