# ADR 0125 — Discrete mesh import: a ready-made mesh into a live session, then the normal `get_fem_data` route

**Status:** Proposed (2026-10-26). No implementation yet. A public session
verb, `g.mesh.editing.import_discrete(...)`. It changes no Emitter Protocol,
no H5 schema and no `FEMData` constructor. It promotes the private mechanism
of the STKO translator ([ADR 0111](0111-stko-translator.md) D1,
`src/apeGmsh/interop/stko/translate_mesh.py`) to a general entry point.

**Sources read for this ADR:** `interop/stko/translate_mesh.py` (`_Builder`,
`build_mesh`, `_PGs`); `core/_model_io.py` (`load_msh`, the one mesh-in path
that exists today); `mesh/Mesh.py` and `mesh/_mesh_editing.py` (the
sub-composites); `mesh/_mesh_queries.py::get_fem_data`;
`mesh/_fem_extract.py` (element walk over dims 0–3, PG nodes through
`getNodesForPhysicalGroup`); `core/_resolution.py::resolve_target` (how loads
and masses find a name); `core/_compose_errors.py` and
`_session.py::_check_chain_phase` (the build-phase guard, ADR 0038);
`_kernel/payloads.py::ElementGroup`; `mesh/_element_types.py` (the alias
table); `opensees/apesees.py` (`element_tags=`, ADR 0111 D2).

## Context

The YOO tower model (ETABS RevK, 1,288 rectangular walls) meshes its walls
outside apeGmsh so that its nodes match ETABS node for node: the comparison
against ETABS is only a proof if the nodes, and therefore the reactions and
drifts, are the same points. It then has to get that mesh into apeGmsh. There
is no public way to do it, so the project builds a `FEMData` by importing
apeGmsh's private underscore modules. That works until one of those modules
moves, and it leaves the model outside the session: no `g.loads`, no
`g.constraints`, no `g.masses`, no `g.save()`, nothing for the viewers to open
as a live model.

The two routes that exist today do not fit:

- **Mesh the geometry in Gmsh.** Rebuilding 1,288 walls as OCC surfaces and
  fragmenting them so they share edges means a fragment over about 13k
  entities, and Gmsh's mesh of those surfaces will not reproduce the ETABS
  node layout.
- **`g.model.io.load_msh`.** It merges a `.msh` file, so the project would
  have to write Gmsh's file format by hand. It does not keep a caller's
  group names unless they are already PGs in the file, and it checks nothing.

apeGmsh already has the mechanism, privately. The STKO translator puts
STKO's own mesh into an empty session through Gmsh's *discrete entities*:
`gmsh.model.addDiscreteEntity(dim)` makes an entity with no geometry,
`gmsh.model.mesh.addNodes` gives it nodes with caller-chosen tags, and
`gmsh.model.mesh.addElementsByType` gives it elements with caller-chosen tags
and connectivity. Physical groups are then unions of those entities
(`g.physical.add`). The caller goes straight to
`g.mesh.queries.get_fem_data(dim=None)`, and everything downstream (the
broker, loads, constraints, H5, viewers, apeSees) sees an ordinary session.
ADR 0111 has run this at production scale on the San Ramon models.

Facts from the sources that shaped the decisions:

- A Gmsh node belongs to exactly one entity, but an element may reference
  nodes owned by any entity. The STKO builder tracks ownership (`owned`) and
  gives each new entity only the nodes not yet owned.
- A physical group has one dimension and is a set of entities. So a group
  of elements can be a PG only if its elements fill whole entities.
- `getNodesForPhysicalGroup` returns the nodes **of the elements** on the
  group's entities. A group of nodes alone therefore needs elements to carry
  it: the STKO builder puts one `point1` element (Gmsh type 15) per node on
  a dim-0 entity ("carrier"), with element ids above the largest real id,
  and records them as `carrier_ids` (never emitted).
- `g.physical.add` **merges** into an existing PG of the same name. The STKO
  builder wraps it (`_PGs`) and refuses a second use of a name instead.
- `get_fem_data(dim=None)` walks elements of dims 0–3, so the `point1`
  carriers appear in `fem.elements` as a `point1` group. apeSees emits only
  the PGs a spec is assigned to, so an unassigned carrier group is never an
  OpenSees element.
- apeSees allocates OpenSees element tags sequentially by default.
  `apeSees(fem, element_tags="fem")` (ADR 0111 D2) makes each PG-fanned
  element's tag its FEM element id and allocates synthesised elements above
  the largest FEM id.
- `_check_chain_phase` already guards every build-phase chokepoint (geometry,
  booleans, fragment, generate, refine, optimize, set_order, renumber,
  partition, PG and label writes). It refuses them once the session has
  produced a `FEMData`.

## Decision

### D1 — The verb lives at `g.mesh.editing.import_discrete`, not `g.model.io`

```python
info = g.mesh.editing.import_discrete(
    nodes=(node_ids, coords),                 # (N,) int, (N, 3) float
    blocks=[
        ("quad4", wall_eids, wall_conn),       # (M,) int, (M, 4) int
        ("line2", beam_eids, beam_conn),
    ],
    groups={
        "W1":        {"elements": w1_eids},
        "base":      {"nodes": base_nids},
        "story_12":  {"nodes": s12_nids},
    },
)
fem = g.mesh.queries.get_fem_data()
```

`g.model.io` is the file boundary: it reads and writes STEP, IGES, DXF,
`.geo` and `.msh`, mostly geometry. This verb reads no file and makes no
geometry; it makes a mesh, the state `g.mesh.generation.generate` would
otherwise produce. `g.mesh.editing` already holds Gmsh's discrete-mesh path
(`import_stl`, `classify_surfaces`, `create_geometry`), and the guard of D7
is a mesh-side state. A file-reading convenience (`g.model.io.load_mesh(path)`
for a neutral format) can be added later as a thin caller of this verb; it
is out of scope here.

The verb returns a small frozen record, `DiscreteImport`: the node and
element counts, `entities` (block index and group signature → entity tag,
D2), `groups` (name → PG dimension), and `carrier_ids` (the synthetic
element-id range of D5, empty when no node group was given).

### D2 — One discrete entity per (block, group signature)

An element's **signature** is the set of group names that list it. Each
block is split by signature, and each distinct (block, signature) pair
becomes one discrete entity of the block's dimension holding exactly those
elements. Each group's PG is then the union of the entities whose signature
contains it, so a group may take any subset of any block, and the elements
of two groups may overlap.

Entities are created in block order, then signature order (sorted), so the
same input gives the same entity tags. Elements that belong to no group
still get their own entity (the empty signature) and appear in
`fem.elements`; they are reachable by element type, by id, or by a
`g.mesh_selection` box, but not by name.

Groups are named PGs. Labels (`_label:` PGs) are not created: a label is a
geometry-time name and there is no geometry.

The rejected alternative, one entity per block, cannot represent a group
that takes part of a block. One entity per element would make every PG a
union of up to a million entities.

### D3 — Element types are named with apeGmsh's alias vocabulary; Gmsh codes are also accepted

A block's type is the short name `fem.elements` reports as
`ElementGroup.type_name` (`"line2"`, `"tri3"`, `"quad4"`, `"tet4"`,
`"hex8"`, `"tri6"`, `"quad8"`, `"quad9"`, `"tet10"`, `"hex20"`, … from
`mesh/_element_types.py`), so what goes in reads back under the same name.
An integer is taken as a Gmsh type code, as `fem.elements.select(element_type=)`
already accepts both. An unknown name or code is refused before anything is
added to the session. `point1` is refused as a block type: node groups are
D5's job, and a user `point1` block would be confused with the carriers.

Connectivity is taken verbatim in **Gmsh's node order** for that type (the
order `ElementGroup.connectivity` reads back). The verb does not reorder,
reorient or check element quality; shell orientation and local axes are the
caller's. It checks only what would corrupt the session silently: the
connectivity has the type's node count per row, every node id exists in
`nodes`, and no element repeats a node.

### D4 — Ids are kept, and the session must be empty

Node ids and element ids are kept exactly as given: they become Gmsh's node
and element tags and therefore `fem.nodes.ids` and `ElementGroup.ids`. Ids
must be positive integers. Node ids must be unique, and element ids must be
unique across all blocks. Coordinates are float64 `(N, 3)`; a 2-D caller
passes `z = 0`.

The verb requires a session with no model entities and no mesh, and refuses
otherwise. This is the STKO builder's rule, and for the same reason: Gmsh's
own entity, node and element tags would collide with the caller's, and a
session holding both a geometric model and an imported mesh would need D7's
refusals to apply to only part of it. One call imports the whole mesh; a
second call is refused (merge the inputs instead).

Keeping the ids into OpenSees is the bridge's job, not this verb's:
`apeSees(fem, element_tags="fem")` (ADR 0111 D2) makes the element tags the
imported ids. Node tags already follow the FEM node ids. The verb's
docstring and the user guide say so, because with the default
`element_tags="sequential"` the OpenSees element tags will not be the
caller's ids.

Every node passed in is kept, including a node no element and no group uses.
A caller that wants such nodes dropped passes
`get_fem_data(remove_orphans=True)`.

### D5 — Node ownership: the first block that uses a node owns it; node groups are point carriers

Ownership in Gmsh only decides which entity a node is classified on; no
downstream consumer reads it. So the rule is the cheapest deterministic one:
entities are created in D2's order, and each new entity owns the nodes of
its elements that no earlier entity owns (the STKO builder's `entity()`).
Nodes shared between blocks, or between the entities of one block, are
shared by reference and never duplicated, so two blocks that meet at a node
are connected. Two coincident nodes with different ids stay two nodes: the
verb never merges by coordinate, because the caller's node layout is the
point of the verb.

Nodes that no element uses (used only by a node group, or by nothing) are
owned by one trailing dim-0 entity with no elements.

A group given as `{"nodes": ids}` becomes a dim-0 PG carried by a dim-0
discrete entity holding one `point1` element per node, exactly as the STKO
diaphragm carriers do. The carrier elements get ids
`max(element id) + 1, …` and the range is returned as
`DiscreteImport.carrier_ids`. They appear in `fem.elements` as `point1`
groups and are never emitted, because apeSees emits only the PGs a spec is
assigned to and nobody assigns one to a node group.

A group gives `elements` or `nodes`, not both. A PG has one dimension, and a
name spanning two dims would make `resolve_target`'s `expected_dim` scoping
answer differently for loads and for constraints. A caller who wants the
nodes of an element group already has them (`fem.nodes.get(target="W1")`
returns the nodes of its elements); one who wants a different node set
names it separately. An element group must also have a single dimension:
a group that lists elements of a `line2` block and of a `quad4` block is
refused, with the two dims named.

### D6 — Declarations resolve against the imported groups by name, through the existing resolvers

Groups are ordinary named PGs, so nothing in the declaration layer changes:

- **Loads and masses** (`g.loads.*`, `g.masses.*`) go through
  `resolve_target`: the group name is found at step 4 (physical group). A
  surface pressure on an element group of `quad4` walls resolves at
  `expected_dim=2`; a nodal load or a nodal mass on a node group resolves to
  its carrier's nodes. A name used at the wrong dimension fails loud, as it
  does today.
- **Constraints** (`g.constraints.rigid_diaphragm`, `equal_dof`,
  `rigid_link`, fixities) take node groups the same way: a story's node
  group is a diaphragm slave set, a base node group is a support set.
- **`g.mesh_selection`** boxes and planes work on the imported nodes and
  elements unchanged, for sets nobody named in `groups`.

Name clashes are refused, not merged: the verb registers every group name
before writing any PG (the STKO builder's `_PGs`), and refuses a group whose
name starts with `_label:` or another reserved prefix. Since the session is
empty (D4), no earlier PG can clash.

Declarations that need geometry (anything taking an OCC entity, `g.embed`
onto a host volume, `g.rebar.place` along a curve, `g.parts`) have nothing
to act on. They fail at resolution with their own not-found errors; this
ADR adds no special message for them.

### D7 — After an import, meshing and geometry changes are refused

An imported mesh is the model. Anything that would regenerate it, renumber
it, or add geometry that `get_fem_data` would then mix in, breaks the ids and
the groups this verb exists to keep. So `import_discrete` puts the session
in an **imported** state, and from then on every chokepoint that
`_check_chain_phase` guards refuses, with a message that names the import:

- geometry: `g.model.geometry.*`, `g.model.boolean.*` (fragment included),
  `g.model.transforms.*`, `g.model.io.load_*`, `g.parts.*`;
- meshing: `g.mesh.generation.generate`, `refine`, `optimize`, `set_order`,
  `g.mesh.structured.*`, `g.mesh.sizing.*`, `g.mesh.field.*`;
- renumbering: `g.mesh.partitioning.renumber` (the ids are the contract);
- PG and label writes: `g.physical.*` writes and `g.labels.*` writes (the
  groups come from the `groups` argument, D6).

The implementation is one extra state in `_check_chain_phase`, so every
existing guard site gains the refusal at once and a new build-phase API that
calls the guard gets it for free. The chain-phase surface ADR 0038 leaves
open (`g.loads`, `g.masses`, `g.constraints`, `g.displacements`,
`g.decouple_node`) stays open, as do queries, `g.mesh_selection`,
`get_fem_data`, the viewers and `g.save()`.

`g.mesh.partitioning.partition` is also refused in this version. Gmsh
partitioning replaces entities with partition entities, and whether the
imported PGs and carriers survive that has not been measured. Lifting the
refusal is a follow-up with its own test, not part of this ADR.

### D8 — The read-back is the public `fem.elements` / `ElementGroup`, and needs no new API

The YOO project reads its connectivity back today through private modules.
The public path already exists and is what this ADR points users at:

```python
for grp in fem.elements:                  # one ElementGroup per element type
    grp.type_name, grp.type_code          # "quad4", 3
    grp.ids                               # (M,) int64, the imported ids
    grp.connectivity                      # (M, npe) int64, Gmsh node order
    for eid, conn in grp: ...             # plain ints, safe for OpenSees

sel = fem.elements.select(pg="W1")        # one group's elements
ids, conn = sel.ids, sel.connectivity     # same order (ADR 0111 D2)
fem.nodes.get(target="base")              # a node group's ids and coords
```

`ElementGroup` and `ElementTypeInfo` are exported from `apeGmsh.mesh`; the
user guide states that as their public import path. The round trip is the
acceptance test of this ADR: for every block, `fem.elements` returns the
same ids, the same connectivity rows and the same type name that went in,
and every group's `select` returns exactly its ids. `point1` carrier groups
are told apart by `DiscreteImport.carrier_ids`.

## Alternative considered: `FEMData.from_arrays(nodes, element_blocks, groups)`

A classmethod that builds the broker directly from arrays, with no Gmsh
session. It is simpler to write and faster: no Gmsh entities, no
`getElements` walk. It is **rejected as the primary route and deferred** as
a possible later addition, because it bypasses the session:

- `g.loads`, `g.constraints`, `g.masses` and `g.displacements` are session
  declarations. They resolve their targets against the live Gmsh model
  (`resolve_target`, PG lookups, `getNodesForPhysicalGroup`) when
  `get_fem_data` runs. A `FEMData` built from arrays has no session, so every
  declaration listed in D6 has nothing to resolve against. The caller would
  have to compute nodal loads, masses and constraint pairs by hand and pass
  them through the `FEMData.with_*` transforms, which is exactly the work
  apeGmsh exists to do.
- `g.save()`, the chain-phase cache, the model viewer on a live session and
  the composite inspectors all hang off the session, not the broker.
- Two constructors that produce a `FEMData` would have to agree on node and
  group semantics forever. The session route has one: `get_fem_data`.

`from_arrays` remains a reasonable addition for a caller that already has
loads and constraints as arrays (a format translator, a test fixture). If it
is added, it should be implemented on top of this verb (an internal throwaway
session plus `get_fem_data`), so the two cannot drift.

## Consequences

- A project with its own mesher gets a public, supported entry point and
  keeps apeGmsh's whole downstream: declarations by group name, H5, viewers,
  apeSees with the caller's ids.
- The STKO translator's `_Builder` becomes a caller of the public verb where
  its rules allow (its local-axis split and diaphragm carriers are
  STKO-specific and stay in `interop/stko`). This is a follow-up, not a
  requirement of this ADR.
- An imported session can declare and extract but not mesh or change
  geometry. A user who needs both (a meshed soil box under an imported
  superstructure) composes two saved models with `Assembly` (ADR 0117)
  rather than mixing them in one session.
- Cost: an element group touching many blocks makes many entities (one per
  block and signature). For YOO (one `quad4` block, groups per wall and per
  story) that is about one entity per wall, the same order as the STKO
  models; it is measured in the implementation PR, not assumed.

## Open questions (for the implementation PR)

1. Does `getNodesForPhysicalGroup` on a dim-0 entity with no elements return
   its owned nodes? If yes, node groups could skip the `point1` carriers. The
   carriers are proven (ADR 0111); this is an optimisation, measured first.
2. Partitioning after import (D7): measure whether PGs and carriers survive
   `gmsh.model.mesh.partition` on discrete entities, then lift the refusal.
3. Whether `g.save()` / `apeGmsh.from_h5` round-trips an imported session
   with the imported state set (so a reload also refuses meshing).
