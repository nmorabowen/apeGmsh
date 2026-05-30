# How-to recipes

**Task-oriented. You have a goal; this is the shortest path to it.**

Each recipe answers one "how do I…?" question. They assume you already
know your way around apeGmsh — if you don't yet, work through the
[Tutorials](../tutorials/index.md) first, and lean on the
[Concepts](../concepts/index.md) pages when you want to understand *why*
something works the way it does.

Dedicated recipe pages are being written. Until each one lands, every task
below links to the section of the topic guides where the answer lives
**today** — so this index is useful right now, not just a promise. Tasks
marked *dedicated page coming* don't yet have a tidy home; the link points
at the closest existing material.

## Geometry & CAD

- **[Import and heal a STEP file](../internal_docs/guide_cad_import.md)** — load CAD, heal dirty geometry, and get a meshable model.
- **[Tag a face as a physical group](../internal_docs/guide_queries.md)** — name an imported or constructed face by query so you can target it later.

## Meshing

- **[Set a local mesh size](../internal_docs/guide_meshing.md)** — refine the mesh on a specific entity instead of globally.

## Build

- **[Build a multi-part assembly](../internal_docs/guide_parts_assembly.md)** — template a Part, place copies, and fragment them into one conformal mesh.
- **[Save a model and reload it](../internal_docs/guide_fem_broker.md)** *(dedicated page coming)* — persist a model to `model.h5` with `save_to` / `to_h5` and bring it back with `from_h5`. The round-trip contract lives in the FEM broker guide today.
- **[Compose models from saved modules](../internal_docs/guide_parts_assembly.md)** *(dedicated page coming)* — combine independently-saved `.h5` parts with `g.compose` / `apeGmsh.from_h5` (and the sub-path `Assembly` builder). No standalone guide yet; see the parts & assembly guide for the multi-part mental model.

## Physics

- **[Apply gravity](../internal_docs/guide_loads.md)** — add a gravity body force to a part or the whole model.
- **[Apply a face pressure](../internal_docs/guide_loads.md)** — put a normal pressure on a named face.
- **[Add a point load](../internal_docs/guide_loads.md)** — apply a concentrated force at a node or query target.
- **[Fix supports](../internal_docs/guide_loads.md)** *(dedicated page coming)* — pin, roller, or fully-fix boundary nodes. The face-level homogeneous `fix` lives in **guide_loads §11 (`face_sp`)**; the node-level `apeSees.fix` deck verb is in the [OpenSees bridge guide](../internal_docs/guide_opensees.md). This recipe will pull both into one place — for now, start at `face_sp`.
- **[Prescribe a support displacement (SP)](../internal_docs/guide_loads.md)** — impose a non-zero displacement at a face or node (guide_loads §11).
- **[Tie non-matching meshes](../internal_docs/guide_constraints.md)** — couple two members across a non-conformal interface; the constraint auto-emits through the bridge.
- **[Add a rigid diaphragm or rigid link](../internal_docs/guide_constraints.md)** — constrain a set of nodes to move as a rigid body.

## Solve

- **[Run a static analysis](../internal_docs/guide_opensees.md)** — drive a gravity/lateral static solve through `apeSees(fem)`.
- **[Run a modal (eigenvalue) analysis](../internal_docs/guide_opensees.md)** — set up mass, call `ops.eigen`, and pull periods.
- **[Run a pushover](../internal_docs/guide_opensees.md)** — displacement-controlled nonlinear static analysis to a target drift.
- **[Export to OpenSees Tcl or openseespy](../internal_docs/guide_opensees.md)** — emit a standalone `.tcl` / `.py` deck from the bridge.

## Results

- **[Read a node's displacement and reactions](../internal_docs/guide_results.md)** — pull nodal results back by `pg=` / `label=` / `selection=`.
- **[Plot a deformed shape or contour](../internal_docs/guide_results.md)** — render results with the notebook-safe `show_web` viewer.
- **[Choose a results strategy](../internal_docs/guide_obtaining_results.md)** — pick between `from_native`, `from_recorders`, and `from_mpco`, and understand the tradeoffs.
