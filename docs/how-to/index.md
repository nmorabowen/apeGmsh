# How-to recipes

**Task-oriented. You have a goal; this is the shortest path to it.**

Each recipe answers one "how do I…?" question. They assume you already
know your way around apeGmsh — if you don't yet, work through the
[Tutorials](../tutorials/index.md) first, and lean on the
[Concepts](../concepts/index.md) pages when you want to understand *why*
something works the way it does.

Most tasks below have a dedicated recipe page. A few still link into the
topic guide where the answer lives — so this index is useful for every
task, not just the ones with a tidy page yet.

## Geometry & CAD

- **[Import and heal a STEP file](import-step.md)** — load CAD, diagnose and heal dirty geometry, name faces, and get a meshable model.
- **[Tag a face as a physical group](../internal_docs/guide_queries.md)** — name an imported or constructed face by query so you can target it later.
- **[Set a local mesh size](../internal_docs/guide_meshing.md)** — refine the mesh on a specific entity instead of globally.

## Build & assemble

- **[Build a multi-part assembly](../internal_docs/guide_parts_assembly.md)** — template a Part, place copies, and fragment them into one conformal mesh.
- **[Save a model and reload it](save-reload.md)** — persist to `model.h5` with `save_to` / `g.save`, and bring it back with `FEMData.from_h5` / `apeGmsh.from_h5`.
- **[Compose models from saved modules](compose-modules.md)** — combine independently-saved `.h5` parts with `g.compose` / `apeGmsh.from_h5` (and the sub-path `Assembly` builder).

## Physics

- **[Apply gravity / self-weight](gravity.md)** — add a gravity body force to a part or the whole model.
- **[Apply a face pressure or traction](face-pressure.md)** — put a distributed load on a named face or edge.
- **[Add a point load](point-load.md)** — apply a concentrated force or moment at a node set.
- **[Fix supports & boundary conditions](supports-bcs.md)** — pin, roller, or fully-fix nodes, and prescribe a non-zero support displacement (SP).
- **[Tie two non-matching meshes](tie-meshes.md)** — couple two members across a non-conformal interface; the constraint auto-emits through the bridge.
- **[Add a rigid diaphragm or rigid link](../internal_docs/guide_constraints.md)** — constrain a set of nodes to move as a rigid body.

## Solve (the OpenSees bridge)

- **[Run a static analysis](../internal_docs/guide_opensees.md)** — drive a gravity/lateral static solve through `apeSees(fem)`.
- **[Run a modal (eigenvalue) analysis](../examples/modal-analysis.md)** — set up mass, call `ops.eigen`, and pull periods and mode shapes.
- **[Run a pushover](../internal_docs/guide_opensees.md)** — displacement-controlled nonlinear static analysis to a target drift.
- **[Export to a Tcl or openseespy script](export-script.md)** — emit a standalone runnable deck with `ops.tcl` / `ops.py` instead of solving in-process.
- **[Checkpoint & resume an analysis](checkpoint-resume.md)** — save committed solver state with `database`/`save`, recover it with `restore`, and benchmark that a restart reproduces the uninterrupted run.

## Results

- **[Read a node's displacement and reactions](read-results.md)** — pull nodal results back by physical-group name.
- **[Plot a deformed shape or contour](../internal_docs/guide_results.md)** — render results with the notebook-safe `show_web` viewer.
- **[Get results via MPCO (STKO)](results-mpco.md)** — record to `.mpco` and read with `Results.from_mpco(model_h5=...)`.
- **[Choose how to get results](choose-results-strategy.md)** — the run × read decision: in-process vs export, and `from_native` / `from_recorders` / `from_mpco`.
