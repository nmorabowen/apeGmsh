# Results system internals

This page explains how the results system is engineered beneath the
`Results` API — the three-broker chain that pairs numbers with the model
they came from, the reader protocol behind the constructors, the
recorder-spec seam and transcoder on the write side, and the seams that
keep the viewer's renderer and pickers replaceable.

The [concepts page](../concepts/results.md) covers what a `Results`
object does; this one covers why it is shaped that way. It is written
for contributors: every mechanism named here has an ADR, and the ADR
numbers (plain text throughout) are where the full trade-off records
live, under `src/apeGmsh/opensees/architecture/decisions/`.

## The three-broker chain

The system's spine is an asymmetry: the bridge writes, a separate broker
reads, and `Results` carries that broker. `apeSees(fem)` is a pure
producer — it emits `model.h5` and never reads its own output back.
The read side is `OpenSeesModel` (ADR 0019), a frozen broker that
rehydrates the `/opensees/` zone of an archived file and carries the
embedded `FEMData` for the neutral zone. It has no mutable surface; to
author a model you return to the bridge. This split is what lets the
archive format evolve as an emit target without ever becoming the
bridge's input contract.

`Results` completes the chain (ADR 0020): every constructor requires
the model broker and stores it, so `results.model` → `model.fem` →
`FEMData` is named in the type system. The `TypeError` you get for
omitting `model=` is the pairing contract — the design deliberately
replaced procedural binding (the old `BindError` hash check, long
retired) with structural pairing. `bind()` still accepts any `FEMData`
without enforcing hash equality; the `snapshot_id` is computed and
stored as metadata, and the lineage chain *warns* on drift rather than
raising. Pairing a session `fem` with a results file remains the user's
responsibility; the model broker travels with the object so the viewer
and downstream tooling never have to re-derive it.

Two consequences follow. First, the **Composed-file pattern**: when
apeGmsh writes a native results file, `NativeWriter` embeds the full
`/opensees/` zone alongside the run data at file-open time (open, not
close — HDF5 fragments badly when the bulk lands during the closing
fsync). One `run.h5` answers both "what did you measure?" and "on what
model?", so a file emailed to a colleague is self-sufficient. MPCO is
the deliberate exception: it is the third-party-file case, the broker
is rehydrated in memory from the sibling `model.h5` you point at, and
nothing is copied into a derived file.

Second, the subprocess viewer is **file-mediated**. When
`results.viewer(blocking=False)` spawns a child process, only paths
cross the boundary — the child re-opens the file and re-rehydrates the
broker itself. The broker object is never pickled across.

## One spec, every write strategy

On the write side the pattern is declare → resolve → execute. Recorder
declarations are pure data — category, components, a `pg=`/`label=`/
`selection=`/`ids=` selector, a cadence. Resolution flattens them
against the bound `FEMData` into a frozen `ResolvedRecorderSpec`
(`results/spec/`): concrete ID arrays, shorthand components expanded to
canonical names, section metadata attached, the whole thing tagged with
the fem's `snapshot_id`. The spec is gmsh- and OpenSees-independent at
the type level — numpy and dataclasses only.

That frozenness is the seam: **anything that can walk a spec is a valid
execution strategy**, and the strategies are just its consumers.

- **Script emission** — `to_tcl_commands()` / `to_python_commands()`
  translate each record into `recorder Node …` / `recorder Element …`
  source inlined into an exported deck; the user runs OpenSees
  elsewhere and the spec doubles as the manifest for parsing the output.
- **Live recorders** — `spec.emit_recorders(out_dir)` pushes the same
  commands into the running openseespy domain via `ops.recorder(*args)`,
  with a per-stage filename prefix (`<stage>__…`) so multi-stage runs
  don't collide.
- **Live MPCO** — `spec.emit_mpco(path)` issues one in-process
  `recorder mpco` call, with a build-gate at `__enter__` that raises
  with a remediation pointer when the active build lacks the recorder.
- **Domain capture** — the native path. This one grew its own
  declarative class, `DomainCaptureSpec`, resolved on the bridge:
  `ops.domain_capture(spec, path=...)` returns a `DomainCapture`
  context manager that queries the live domain (`ops.nodeDisp`,
  `ops.eleResponse`, …) on each `cap.step()`, buffers in RAM, and
  flushes chunked writes through `NativeWriter` at stage end.
  `capture_modes()` runs the eigen analysis and writes one stage per
  mode. The bridge is forwarded through so the capture file gets its
  `/opensees/` zone composed in — the Composed-file pattern again.

Text and live emission cannot drift apart because they share one core.
`emit_logical()` turns a resolved record into `LogicalRecorder`
values — one classic recorder command in structured form — and
`format_tcl` / `format_python` / `to_ops_args` are three renderings of
the same object (with `mpco_ops_args` playing the same role for MPCO).
Cross-check tests assert the source-code form and the live-args form
agree, so a change to recorder argument shape lands in one place.

## Readers and the slab pipeline

Every constructor lands on the same contract: `ResultsReader`, a
`typing.Protocol` in `results/readers/_protocol.py`. It covers stage
discovery (`StageInfo` with a `kind` of static / transient / mode and
the mode-only eigenvalue attributes), per-level component discovery,
and a `read_*` method per topology level — nodes, elements,
line stations, gauss points, fibers, layers, springs. The composite
layer above — `results.nodes`, `results.elements.gauss`, and friends —
never branches on which backend it is talking to.

The protocol has earned its keep: there are now four reader families
behind it, one more than the design started with.

- **`NativeReader`** — apeGmsh's own HDF5, written by capture or by the
  transcoder.
- **`MPCOReader`** plus a multi-partition merger — parallel STKO runs
  write one `.mpco` per rank; the merger deduplicates boundary nodes by
  ID, concatenates elements, and presents one virtual reader. For
  composed models, where the OpenSees element tag diverges from the FEM
  element ID, an ops-tag ↔ fem-eid translator is attached at read time
  so the query API keeps speaking FEM IDs.
- **`LadrunoReader`** plus its own multi-partition merger — the Ladruno
  fork's canonical recorder writes a self-sufficient `.ladruno` file
  carrying its own geometry, so `model_h5=` is optional there: the
  broker can be synthesized from the file's own `MODEL` group. Adding
  this reader touched no composite code — the protocol working under a
  fourth implementer is the design's proof point.
- **`from_recorders`** is not a fourth on-disk format: it transcodes
  classic recorder output into a native file and opens *that* through
  `NativeReader`.

Readers hide two kinds of stitching. Element-level data is stored in
rectangular per-group tensors keyed by `(class_tag, int_rule)` — the
reader resolves which groups the requested elements live in, reads each,
and concatenates, so callers never see groups. Partition stitching works
the same way one level up. What comes out is a **slab**: a frozen
dataclass carrying the `values` array (time-first) plus the location
index that says what each column is. Slabs are read lazily through open
h5py handles; RAM is for queries, not storage, which is what keeps a
million-DOF multi-thousand-step file affordable.

The native schema itself is short to summarize: an embedded `FEMData`
snapshot under `/model/`, per-stage groups under `/stages/` each with
their own time vector and partitions, underscore-prefixed datasets for
index/metadata versus bare names for result components, natural
(not global) gauss coordinates, CSR-style flat arrays for fibers and
layers where per-element counts legitimately vary, and modes stored as
ordinary one-step stages with `kind="mode"` and the eigenvalue /
frequency / period in stage attributes — no separate modes machinery
anywhere in the pipeline.

## Transcoding and the cache

`RecorderTranscoder` is the bridge from classic recorder output to the
native schema. The spec serves as the manifest: it knows which files a
run produced and what each column means, so the parser can decode
headerless text output. Raw OpenSees tokens are renamed to the canonical
vocabulary on the way in, and the model's `/opensees/` zone is composed
into the transcoded file so downstream `from_native` behaves as if the
run had been captured natively. The current cut parses text (`.out`)
nodal records; element-level transcoding shares its unflattening logic
with the capture path and is tracked as a follow-up.

Transcoding is paid once. The cache root resolves to `<cwd>/results/`
(overridable via `APEGMSH_RESULTS_DIR` or an explicit kwarg), and the
cache key hashes the source files' mtime and size together with the
parser version and the fem `snapshot_id`. Unchanged inputs open the
cached HDF5 directly; touching a source file or bumping the parser
re-transcodes; a different fem simply keys a different cache entry.

## The viewer behind its seams

The viewer package is deliberately boxed in by three structural
contracts — one per boundary — each enforced by an AST guard test
rather than by convention.

**Read seam.** `apeGmsh.viewers` is a pure `model.h5` consumer
(ADR 0014): it imports `apeGmsh.results` and the emitter's `h5_reader`,
and nothing else from the broker layers. All structural data flows
through one adapter, `ViewerData`, with interchangeable builders
(`from_fem`, `from_h5`, `from_reader`) — downstream viewer code never
learns which source a snapshot came from. The shape of "a model file
the viewer can read" is codified as the `H5ModelReader` contract
(ADR 0026): identity and capability probes (`has_opensees_orientation`,
`has_neutral_zone`), neutral-zone and bridge-zone accessors, a
lifecycle. The emitter's `H5Model` implements it structurally, and a
future foreign-format adapter (d3plot, Exodus, xDMF) drops in by
satisfying the same contract, touching no viewer code.

**Render seam.** Everything the viewer draws goes through a two-part
seam (ADR 0042). `viewers/scene_ir/` defines a declarative **SceneLayer
IR** — frozen value types (`MeshLayer`, `GlyphLayer`, `LabelLayer`,
typed array bundles like `PointSet` / `CellBlocks` / `ScalarField`,
plus `ColorSpec` and `VisibilityMask`) that carry plain numpy and
import neither vtk nor pyvista. The **`RenderBackend` Protocol**
consumes them: `add_layer` / `update_layer` / `remove_layer` /
`set_visibility` / `render`, with `supports_picking()` as a capability
probe. Diagrams and overlays emit IR and call the Protocol; all VTK
construction lives inside a backend. `PyVistaQtBackend` is the desktop
reference; `TrameBackend` is the web/Jupyter implementation behind
`show_web()` / `serve_web()`. Two payoffs drove this: visibility
*semantics* live in the IR rather than in backend tricks, and every
diagram became headlessly testable by asserting on the layers it emits
— no GPU, no pixels, which matters in CI environments that cannot get
an OpenGL context.

**Pick seam.** Picking is optional by design — a view-only backend is
legal. The geometric core is the `PickBackend` Protocol (ADR 0047): a
stateless `resolve_pick(PickRequest) → PickHit` shared by the desktop's
event-driven face and the web's request/response face, plus projection
primitives (`project_points`, `frustum_planes`) for box selection. The
hit carries only geometry (`prop_id`, `cell_id`, world point); what a
pick *means* is domain logic layered above it (ADR 0045): a VTK-free
`SelectionTarget` keyed per substrate (BREP / mesh topology / results
topology), one canonical `BBox` type, a `FilterController` owning the
dimensional 0/1/2/3/4 filter, and a serialized `SelectionLog` of
per-gesture operations giving undo, redo, and replay across all three
viewers.

Above the seams, two later contracts govern the parts that churned
hardest and are worth knowing at that altitude only. ADR 0056 fixes
*when the picture changes*: every piece of view state has exactly one
owner, derived state is recomputed by a reconciler rather than stored,
and every gesture funnels through one dispatcher whose four primitives
(STEP / DEFORM / GATE / RENDER) coalesce into a single render per
event. ADR 0058 made a Geometry a *scene instance* — each owns its own
substrate copy with per-geometry deformation, so side-by-side deformed
vs. reference or stage-vs-stage views are ordinary concurrent
geometries, with deformation fan-out running through one universal
per-diagram hook. Anything more specific than that (dock layouts,
panel widgets, dialog flows) is implementation, not architecture, and
is deliberately not documented here.

The through-line of the whole system is the same move made four times:
freeze a contract at the boundary — the recorder spec, the reader
protocol, the model-reader contract, the scene IR — and let both sides
vary independently. Adding a write strategy means consuming the spec;
adding a results format means implementing the reader protocol; adding
a renderer means implementing `RenderBackend`. None of them touches the
layers on the other side of the seam.

---

*Next: [back to the Design overview](index.md).*
