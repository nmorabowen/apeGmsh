# ADR 0112 — The files are the model; apeGmshViewer, a read-only TypeScript app, replaces the Qt viewers

**Status:** Accepted (2026-10-03; proposed and accepted the same day by the owner)

**Owner:** nmora

**Evidence:** an owner workshop on 2026-10-03, run as a Socratic dialogue
on one question: *how does a human get a feel for a model that agents
wrote?* This ADR records the answers and the decisions that follow from
them.

**Canonical for viewers.** For everything that renders a model to a
human, this ADR is the canonical decision. It overrides the viewer
parts of the remediation program
([PROGRAM.md](../../internal_docs/program/PROGRAM.md), the panel report
and chain X), as set out in [Relation to the remediation
program](#relation-to-the-remediation-program), and the earlier viewer
ADRs listed below.

**Relation to earlier ADRs:**

- [ADR 0014](0014-viewer-is-pure-h5-consumer.md) (viewer is a pure
  `model.h5` consumer): **carried to its end**. 0014 walled the viewers
  off from `FEMData`; this ADR walls the app off from apeGmsh entirely.
- [ADR 0095](0095-apegmsh-studio.md) (Studio habitat): **supersedes
  Part 5** ("Electron is a skin … must not rewrite the visualizers in
  Three.js") and the Qt `ViewerWindow` as the v1 host in Part 2.
  **INV-2 stands**: the app never writes the script, OCC, or `model.h5`.
  The daemon, MCP verbs and selection envelope are out of scope here and
  keep their own status.
- [ADR 0102](0102-warrant-the-snapshot.md) (warrant the snapshot):
  **consistent with D1**. The app is a consumer of the snapshot and
  raises its warrant (D2 forces the files to be complete). It moves the
  Qt viewers from 0102's *inventory* bucket to *replace*, and it is the
  owner's explicit exception to 0102 D2's freeze on a new viewer
  ontology.
- ADRs 0087–0090, 0098 (viewer design canon, results session): frozen
  with the Qt viewers (D8); not ported.

**Name.** The application is **apeGmshViewer**, in the top-level
directory `apeGmshViewer/`.

## Context

apeGmsh began as a mesh bridge, and the declarative script became its
strength: agents describe a model in code, and the code compiles to
`FEMData`, an OpenSees deck, and `model.h5`. Agents now write almost all
of that code. The owner reports that the models have become *ether*:
impossible to see, and impossible to feel the way an ETABS or STKO
model is felt.

The workshop established, in the owner's words where it matters:

1. **Feel comes from looking**, not from authoring. Some feel arrives
   through the agent conversation; none arrives from the code, which is
   "buried in functions, not human parsable".
2. **A human must know the decisions**: BCs, loads, assignments,
   recorders, analysis types, stages. Today they are visible only in
   conversation.
3. **Opening a model needs an agent.** "If I want to look at an ETABS
   model I just open it; an apeGmsh model is a bunch of code that only
   the agent gets." The Qt viewers exist but are rarely used because an
   agent has to open them.
4. **The viewers are the weakest part of the library**: sluggish;
   tangled ("change one thing, two break"), duplicated; not beautiful and
   not simple. The model / mesh / results split served the code, not the
   user.
5. **Visual first**: click a beam and a box shows its definitions. The
   model-level decisions that have no geometry (stages, analysis,
   recorders, patterns) get their own windows, such as a stage diagram.
6. **Reference**: Plasticity (visually striking, simple UI). Not ETABS.
   STKO is fast but cumbersome.
7. **Editing is not needed for feel.** If it comes, it is a way to point
   from the visual app back to the source code.

Probes of `main` at `77c9c801` confirmed two mechanical causes behind
items 3 and 5:

- `viewers/model_viewer.py` does `import gmsh` and reads the live kernel
  (`gmsh.model.getEntities`, `occ.synchronize`). The geometry view
  **cannot open a file**; it exists only inside a running session.
  ADR 0014 freed the mesh and results views, never the geometry.
- `model.h5` is written only when `save_to=` or `.h5(...)` is called
  (`_session.py` autosave runs only when `_save_to` is set). A model run
  without it leaves nothing to open.

Stages are already archived (ADR 0055, `/opensees/stages/stage_NNN`),
but as store-and-echo emit tokens built for replay. They are readable,
but not yet shaped for a human diagram. No declaration records where in
the user's code it came from.

## Decision

### D1 — The files are the model

A model is the set of files a run leaves behind, not the script that
produced them. The script is the **generator**. Every session run
writes its artifacts **unconditionally**, at a conventional path next
to the script, and overwrites them on the next run; there is no opt-in.
The artifacts are:

| Artifact | Phase | Exists today |
|---|---|---|
| geometry (tessellated, named) | after geometry | **no** (D2a) |
| `model.h5` (neutral + `/opensees/`) | after mesh / bridge | yes, opt-in |
| results (`.h5` / `.mpco`) | after analysis | yes |

`save_to=` remains as a path override, not as the switch that decides
whether files exist.

### D2 — The files carry their meaning without apeGmsh code

The app **never imports apeGmsh**, and never runs Python, to understand
a model. Anything the app needs that today only Python can derive goes
into the file. The test: a developer with only the schema documents and
the three files can build the app. Known gaps follow. Each new kind of
data is **its own optional zone** with its own version key (ADR 0023),
so adding it does not bump the neutral or `/opensees/` zones, and a
reader that ignores it is unaffected. The zone designs are an
irreversible schema decision and go through the program's architect
pair (V0):

- **(a) Geometry.** A `/geometry` zone: per-entity tessellation
  (vertices, triangles, polylines) keyed by dimtag, plus the
  dimtag → label / physical-group map. Tessellation, not BRep: a
  browser cannot read OCC.
- **(b) Readable stages.** Each stage carries, besides its replay
  tokens, the fields a diagram needs in plain form: name, order, analysis
  type, integrator / algorithm / test, increments, the activated groups,
  and the patterns, BCs and recorders it adds or removes. These fields
  are **derived at write time from the same records chain K archives**
  (the `VERBS` table, K1), never kept as a second registry beside them.
- **(c) Names everywhere.** Every declaration the app shows is
  reachable by its name (`/labels`, `/physical_groups`,
  `/opensees/names`). Tags are evidence, not handles (ADR 0095
  names-first).
- **(d) Provenance** (D3).

### D3 — Editing is provenance

Each declaration records the location in the **user's** code that made
it: file, line and function, plus the outermost script line when the
call came through a helper. The capture walks to the first frame
outside the `apeGmsh` package; the same walk already exists for warning
stack levels (`opensees/_internal/build.py`, `_stacklevel_outside_package`). The
records go in a provenance table keyed by declaration path.

In the app, "edit" means **go to source**: open that line in the
editor, or hand the object's name and its source location to the agent.
The app does not write files (0095 INV-2). If direct editing is ever
wanted, it is an override keyed by name that lives in the app's state
store (D6) and that the code can absorb. That is a later ADR, and this
design leaves room for it.

### D4 — One repo, a hard wall

The app lives in this repository, in `apeGmshViewer/` beside `src/`,
so agents keep the context of both sides and nothing has to be
synced between repos. The wall is enforced, not advised:

- The app reads the artifacts and the schema documents only. A lint in
  `static-gates` fails on any import from, or subprocess into,
  `apeGmsh`.
- The only change that crosses the wall is a schema change, and it
  bumps the zone version the app checks.

### D5 — TypeScript and three.js, in an Electron shell

The app is TypeScript with three.js for the viewport, packaged with
Electron as a desktop application that opens artifacts by file
association or drag-and-drop. HDF5 is read in-process (h5wasm or an
equivalent). A thin local data server is allowed **only** if a measured
results file is too large to read lazily in-process, and that server
may import only a reader, never a session.

**What gets measured gets managed.** From P0, every phase records the
same numbers on the owner's models: file size, nodes and elements, time
to first frame, frame rate while orbiting, memory, and time to fill the
inspector. They are kept in a committed ledger
(`apeGmshViewer/MEASUREMENTS.md`), one row per measured model and
commit. The data-server decision and the kill criteria are read from
that ledger, not from estimates.

### D6 — One state store, one-way data flow

Artifacts load into a single model state. Views render from that
state. User actions are events that update the state. **No panel talks
to another panel**, and no view keeps a private copy of model data.
This is the structural answer to "change one thing, two break": the Qt
viewers decayed through per-panel state that each slice added to.

### D7 — One window, laid out like Plasticity

- **One window.** The viewport fills it. Model, mesh and results are
  positions on one phase axis (*geometry → mesh → stage 1…n →
  results(t)*), not three applications.
- **Contextual inspector.** Selecting an object (element, node, BC,
  load, constraint, group) shows **that object's own parameters** and
  its definition chain (element → transform → integration → section →
  materials). The inspector does not aggregate neighbours.
- **Model-definition windows** for what has no geometry: a stage
  diagram, analysis, patterns and time series, recorders, materials and
  sections. Each is a view of the same state (D6).
- **The owner approves every screen**, from a screenshot compared with
  the reference. Agents implement; nothing visual lands unseen.

### D8 — The new app replaces the Qt viewers

From acceptance, the Qt viewers (`viewers/`, the Studio host window) take
**bug fixes only**. No new features, panels or design ADRs. They are
deleted when the app reaches parity on the owner's own models (P4). The
two do not grow side by side. The headless stills that agents use
(`fem.render` / `results.render`, ADR 0094) stay until the app can
produce the same stills headlessly.

The Studio daemon and its MCP verbs (ADR 0095) change with the host:
`highlight`, `get_selection` and the stills move to apeGmshViewer, and
the go-to-source path (D3) replaces the selection envelope's role as
the human-to-agent handle. That amendment to 0095 is its own ADR, due
before P4.

## Phasing

| Phase | Delivers | Done when |
|---|---|---|
| **P0 spike** | the app opens one real `model.h5` with no apeGmsh import, draws the mesh coloured by physical group, and a click on a beam shows its definition chain | the owner clicks a beam in one of the owner's models and reads its section and material; load time and frame rate are recorded |
| P1 | D1 unconditional write; D2a geometry zone; D3 provenance | an agent-built model opens by double-click, with go-to-source |
| P2 | D2b readable stages; the model-definition windows | the owner reads a staged model's sequence without the agent |
| P3 | results: contours, deformed shape, time, section cuts | parity with the results features the owner uses |
| P4 | parity sign-off; Qt viewers deleted | `viewers/` is gone |

**Kill criteria** (checked at P0, from the measurement ledger): the app
cannot open a model of the size the owner uses at interactive frame
rates, or the inspector cannot be filled without Python interpreting the
file. Either one stops the chain for a re-think before P1.

## Relation to the remediation program

The remediation program (charter [PROGRAM.md](../../internal_docs/program/PROGRAM.md),
board #1203) is running while this ADR is decided. Where the two
disagree about viewers, **this ADR wins**. Point by point:

| Program item | Disposition |
|---|---|
| Panel K21, the **two-render-technology rule (VTK + matplotlib)** | **Superseded.** The target technologies are three.js (apeGmshViewer) and matplotlib. VTK is in sunset: allowed in the frozen Qt viewers until P4, banned in new code. The X1 ADR that was to record K21 records this rule instead. |
| X1, cut the Three.js `geom_transf_viewer` and the plotly `NotebookPreview` | **Stands.** Both are cut. A transform-orientation view returns, if wanted, as an apeGmshViewer window. |
| X2, port `export_animation`, `render`, `render_pack`, `assess` onto `ResultsSession` | **Stands, narrowed.** It moves existing features only and adds none to the session path (D8). `ResultsSession` becomes the last Qt path, deleted at P4. |
| X3, delete the legacy `ResultsViewer`, director chain and trame | **Stands.** It is the first half of D8. |
| X4, the freeze list and the studio LOC ratchet | **Extended.** `src/apeGmsh/viewers/` joins the freeze list with a shrink-only LOC ratchet. |
| Chain K, "the archive is the program" | **Complementary.** K makes the archive complete for replay; D2 makes it readable for people. D2b derives from K1's records, so V3 is gated on K1. |
| Panel S13, no picture IR | **Respected.** The D2 zones are archive data, not a render IR. The state store (D6) lives in the TypeScript app, not in Python. |
| 0102 D2 freeze on new viewer ontologies | **Explicit owner exception**, recorded here. |

### Chain V — apeGmshViewer

The ADR runs as a program chain, with the program's roster, merge
rights and cross-family review. The model and effort of every worker
are pinned by slice type (PROGRAM.md §2), not chosen per call; the rule
is to save tokens where quality allows it, and to spend them where it
does not.

| Link | Work | Workers | Gate |
|---|---|---|---|
| V0 | Design of the D2 zones (`/geometry`, readable stages, provenance) and of the D6 state store | `prog-architect-opus` ∥ `prog-architect-fable`; the orchestrator reconciles; the maintainer ratifies | coordinate with K0 (tag law) |
| V1 | P0 spike in `apeGmshViewer/`, the D4 import-wall lint, the measurement ledger | `prog-builder-opus` + `prog-reviewer-fable`; `prog-auditor` for the numbers; the maintainer approves the screenshots | none (no `src/` change) |
| V2 | P1: unconditional write, `/geometry`, provenance | `prog-builder-fable` + `prog-reviewer-opus` | V0 ratified; V1 kill criteria passed |
| V3 | P2: readable stages, model-definition windows | `prog-builder-opus` + `prog-reviewer-fable` | V2; K1 |
| V4 | P3: results | builder pair by family alternation | V3; measured results sizes |
| V5 | P4: parity sign-off, delete the Qt viewers, the 0095 amendment | `prog-mechanic` + the builder for the 0095 ADR; **human gate** | V4; the owner's parity sign-off |

The orchestrator runs Opus @ high for V0–V1, where the design is set,
and Opus @ medium from V2.

## Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Repair the Qt viewers | Sluggish, tangled, duplicated and not simple: all three causes at once. The owner's verdict is that this is the weakest part of the library. |
| A database (SQLite) as the source of truth, with the script as an importer | Gives up the declarative code, which is the strength that lets agents work. The files are a compiled product, not a second source. |
| Bidirectional script ↔ GUI editing now | Arbitrary Python does not round-trip. Feel comes from looking (Context 1), and editing reduces to provenance (D3). |
| A separate repo for the app | Its cost, keeping two repos in step, is the owner's stated pain. The wall (D4) gives the separation without the second repo. |
| Python app (Qt / VTK rewritten clean) | Keeps the sluggishness of VTK through Python and does not enforce D2: a Python app can always import apeGmsh. |

## Open questions

- Results size strategy (D5): read lazily in-process, or a data server.
  Decided at V4 from the measurement ledger, not before.
- Agent stills after P4. The owner expects headless capture from the
  Electron app to work; V1 tries it on the spike, and ADR 0094's
  renderer is not deleted until it does.
