# ADR 0095 — `apeGmsh.studio`: agent + script + viewer habitat

**Status:** Proposed (2026-08-12; owner workshop the same day;
host-projection / INV-9 workshop 2026-08-13)

**Amends** [ADR 0094](0094-agent-assess-and-viewer-render.md) **only**
in the disposition of “Later: optional MCP wrapping S5” and the
rejected “CLI / MCP / JSON schema in v1” row’s *later* half. ADR 0094
S0–S5 (assess, stills, no Qt for diagnosis) remain in force. Append-only:
this ADR does not rewrite 0094.

**Evidence:** an architecture workshop on consolidating Cursor (prompt
+ editor + skills) with the three Qt viewers and the declarative
script. Constrained by `docs/design/principles.md` (off-chain features
are sidecars), ADR 0014 (viewer is an H5 consumer), ADR 0045
(SelectionTarget / SelectionState), ADR 0056 (dispatcher; no
puppeteering docks), ADR 0060 (sidecar shape: `apeGmsh.hpc`), ADR 0088
(ViewerWindow), ADR 0094 (assess + `viewers.render`), and ADR 0047
R-D (web picking not landed).

## Context

Coding agents already write apeGmsh Python through the skills. After a
script they can `fem.assess()` / `results.assess()` and
`fem.render()` / `results.render()` (ADR 0094 S1–S2). What they still
cannot do is share a live viewport with the human: the three viewers
are isolated phase surfaces, selection stays inside `SelectionState`,
and there is no apeGmsh MCP server — ADR 0094 deferred CLI/MCP because
assess had no consumer yet. The habitat is still the library + skill.

The product intent is a triple surface — prompt, declarative code,
spatial view — without a new modeller and without replacing Cursor:

- The `.py` file is the source of truth (Geometry → Names → Mesh →
  Broker → Solver). The viewport consumes a snapshot; it does not
  write CAD.
- The human picks in 3D; the agent must see a **name** (label /
  physical group), not a drifting tag.
- Preview should show geometry as the script becomes runnable, without
  character-by-character exec or a shadow copy of the file.
- Electron is attractive as an installable window with hot-reload
  chrome. The viewers that actually pick and match the aesthetic are
  Qt (PySide + PyVista). Web picking is still ADR 0047 R-D.

MCP does not exist in this repo. The live Cursor MCP catalog in the
workshop session was Cursor app-control plus a Revit bridge
(`status` / `snapshot` / `inspect` / `element`) — the proven
agent-sees-the-host pattern, not an apeGmsh server.

## Decision

### Part 1 — `apeGmsh.studio` is a sidecar, not a composite

A standalone `apeGmsh.studio` package, same administrative shape as
`apeGmsh.hpc` (ADR 0060) and `apeGmsh.assess` (ADR 0094):

- **Not** a `_COMPOSITES` entry. No `g.studio()`. A `from_h5`
  chain-phase session cannot honestly promise a live CAD preview.
- **Not** re-exported from `apeGmsh/__init__.py` (the root import
  chain eagerly pulls `gmsh`).
- Public door: `python -m apeGmsh.studio`. Types + runners inside
  the package. Notebook sugar, if any, is a later thin wrapper on
  objects the caller already holds.

ADR 0094 is the **headless half** of the same product (verdict +
Qt-look stills, no event loop). Studio is the **interactive half**
(live host + selection + refresh). They share stills and findings;
they do not share a Qt event loop with the agent.

### Part 2 — Three processes, one contract

```
Cursor (IDE + agent + skills + MCP client)
        │  MCP (later) / files (v0)
apeGmsh.studio daemon (Python)
        │  same payload
Studio host window (Qt ViewerWindow, promoted)
```

1. **Cursor** stays the IDE. The human writes and diffs the
   declarative script and tells the agent how code should be written
   (skills, rules, chat). Studio does not grow a second editor or a
   second prompt.
2. **Daemon** owns replay/refresh, last-good tessellation, assess,
   render, and the selection envelope. It owns the Gmsh singleton in a
   **worker process** (the agent does not share that kernel). Frozen
   means frozen: refresh produces a new snapshot, never a patched
   `FEMData`.
3. **Host** is the existing `ViewerWindow` promoted to a studio
   window: one window, three phase modes (model / mesh / results),
   gated by what the script has actually declared. Picks go through
   ADR 0045 `SelectionState`; a mutator publishes the envelope and
   accepts `highlight(names)` — dispatcher-legal (ADR 0056), not
   Outline puppeteering (ADR 0094 rejected that).

### Part 3 — Payload before MCP

MCP is a Cursor adapter, not the product. v0 ships the payload on
disk so a prototype does not wait on a server:

- On pick change the host writes a `SelectionEnvelope` JSON under
  the workspace (e.g. `.apegmsh/selection.json`).
- The skill: before a spatial edit, read that file; after
  `generate()` / `get_fem_data` / `Results.from_*`, `assess()` and
  `render()` a still and `read_file` the PNG.
- When the envelope is stable, the same functions become MCP tools
  (`get_selection`, `highlight`, `run_until`, `assess`, `render`,
  `promote_selection`). That is the consumer ADR 0094 said v1 assess
  lacked. The Revit MCP already in this office (`snapshot` /
  `inspect` / `element`) is the tool-shape to copy, not a new
  protocol family.

Envelope identity is **names first** (labels, physical groups, phase,
substrate, bbox, `unnamed`). Dimtags / node ids are evidence. An
unnamed pick’s first script edit is `.to_label()` / `.to_physical()`.
Tags are not the agent’s handle (names-before-nodes).

### Part 4 — Preview is refresh; statement-step is later

Default: save, end of agent turn, or explicit refresh. The daemon
replays the current file up to the requested phase and tessellates
via the existing `build_brep_scene` throwaway coarse 2-D mesh (then
restores mesh options). `ModelViewer._rebuild_scene` already rebuilds
VTK from the live kernel and keeps the camera — reuse it.

On parse/exec failure: **keep the last good frame**; the traceback
stays in Cursor. Do not clear the viewport. Do not exec
keystroke-by-keystroke (incomplete Python will not run; tessellation
is too expensive).

The temporary artifact is the tessellation the host loads, not a
shadow `.py`. Geometry hash unchanged → skip remesh.

Mesh and results are **swaps**, not streams: BRep stays visible
during `generate()`; mesh replaces it when `FEMData` exists;
undeformed mesh then contour when `Results` binds. The time scrubber
is playback, not authoring.

Statement-step (unwrap `with apeGmsh() as g:` into `begin()`/`end()`,
exec complete AST statements, rebuild after each geometry statement)
is a later slice, geometry-phase only. It is not required for the
product to exist.

### Part 5 — Electron is a skin, not the architecture

The architecture is the daemon + contract + host. The toolkit of the
host is replaceable.

- **v1 host = Qt `ViewerWindow`.** Picking, three aesthetics, docks,
  dispatcher, and session restore already live there. The viewers are
  Python (PySide + PyVista); changing chrome does not require a C++
  compile. `python -m apeGmsh.studio` opens this window.
- **Electron** (or a browser tab around `serve_web`) is a later
  frontend swap **if** non-3D studio UI outgrows Qt docks **and** web
  picking (ADR 0047 R-D) has landed. It does not replace the daemon
  and must not rewrite the visualizers in Three.js.
- Electron as the whole IDE (prompt + editor + viewport) is rejected
  while Cursor is the navigator: that duplicates skills, git, and the
  agent already driving the code.
- Electron becomes the leading **shell** only if the habitat changes:
  apeStudio absorbs the prompt and Cursor is optional for office
  users. That is a product-habitat decision, recorded here as a
  named fork, not a 3D one. The daemon does not change.

### Part 6 — Host projections: see the script, do not own it

The human needs to *observe* the declaration next to the solid
without leaving the viewport. That is not a second IDE.

The host MAY project the file that `run_until` just replayed as a
**read-only text view** (plaintext with syntax color; optional name
tints that follow the viewport PG / label palette so `Footing` is
the same color in 3D, in the binding graph, and in the source).
Click a graph node or a picked name → scroll / highlight that name
in the text. Stale or last-good (INV-4) is visually distinct from
the current file on disk.

This view is a **projection of the last good replay**, same refresh
cycle as the tessellation. It does not:

- write the `.py`, OCC, or `model.h5` (INV-2);
- grow an editor, undo stack, or prompt;
- become Grasshopper: a later **binding graph** (geometry / PGs /
  loads / constraints → `FEMData` → one or more `apeSees(fem)`
  lowerings, `pg=` edges) is the same class of projection — click
  highlights, it does not wire decks.

Cursor remains the author. The graph and the text view are how the
host *shows* the self-declarative script; they are how
implementations stay maneuverable (highlight, color, last-good)
without studio owning the source.

## Invariants

- **INV-1 (sidecar).** `apeGmsh.studio` is not on `_COMPOSITES` and
  is not re-exported from `apeGmsh/__init__.py`. AST-guard siblings
  ADR 0014 / 0094 INV-1 as needed.
- **INV-2 (script owns geometry).** The host never write-backs OCC or
  `model.h5`. Promote-to-label is a suggested script edit, applied by
  the agent or the human in Cursor.
- **INV-3 (names first).** The envelope’s identity fields are labels /
  PGs / phase / substrate. Tags are evidence.
- **INV-4 (last good frame).** A failed replay does not blank the
  viewport.
- **INV-5 (no shared kernel).** The daemon’s Gmsh singleton lives in
  a worker. The agent’s own script runs do not attach to it.
- **INV-6 (0094 stands).** Agents still do not call `viewer()` /
  `show_web` / `gui()` for diagnosis. Interactive `viewer()` is for
  the human; the agent uses assess, render, and the envelope.
- **INV-7 (dispatcher).** Host selection/highlight goes through owner
  mutators. No Outline-click automation, no second reconciler.
- **INV-8 (phase gate).** Results mode is disabled until `Results`
  exists; mesh mode until `FEMData` exists. No fake contours.
- **INV-9 (project, do not own).** The host may show the replayed
  `.py` and a PG→apeSees binding graph as read-only projections of
  the last good `run_until`. It does not edit or write those
  artifacts. Name colors MAY match the viewport palette. Cursor
  remains the author.

## Alternatives rejected

| Rejected | Why |
|---|---|
| **`g.studio()` on `_COMPOSITES`** | Looks like a sixth authoring stage. `from_h5` cannot honor live CAD preview. Same veto as `g.assess`. |
| **Electron as the IDE** | Duplicates Cursor. Loses skills, rules, git, and the navigator already driving the code. |
| **Electron as v1 3D host** | Wraps the weaker surface (`serve_web`; picking is R-D) and adds Chromium on top of VTK. Qt already is the pick authority. |
| **Rewrite Qt visualizers in Three.js / Electron** | 55k LOC, cascade-fragile (ADR 0084). The aesthetic and SelectionState would be re-litigated. |
| **MCP server before the payload** | Empty ceremony. ADR 0094 deferred MCP for this reason. JSON + skills is the prototype; MCP wraps a stable envelope. |
| **Long-lived kernel the agent appends to without updating the file** | Breaks self-declaration. The next human cannot replay the model. |
| **Character-by-character preview** | Incomplete Python will not run; `build_brep_scene` tessellation would thrash. |
| **Shadow temp `.py` as source of truth** | Forks from Cursor. The temp artifact is the tessellation. |
| **Viewer as parametric CAD** | apeGmsh refuses to be CAD. Geometry stays declared in Python. |
| **Merge the three viewer aesthetics** | Model / mesh / results answer different questions. Unify the host and the envelope, not shading or id space (ADR 0045 INV-3). |
| **Studio code editor / write-back from the text view** | Second author, second undo, forks from Cursor. Observe in the host; edit in Cursor. INV-9. |
| **Binding graph as Grasshopper (wires write `pg=` / elements)** | Same veto as a studio editor. The graph projects `apeSees` records after replay; it does not author decks. |

## Slices

| Slice | Ships | Depends |
|---|---|---|
| **S0** | This ADR + skill paragraph: after solve, assess + render (0094); before a spatial edit, there is not yet a pick file — say so. No library. | ADR 0094 S0–S2 |
| **S1** | `SelectionEnvelope` projector from `SelectionState` (pure, unit-tested). Host writes `.apegmsh/selection.json` on pick. Skill: read it. | ADR 0045 |
| **S2** | `apeGmsh.studio` daemon: `run_until` = subprocess replay of the current file, last-good tessellation via `build_brep_scene` / `viewers.render`, geometry-hash skip remesh. `python -m apeGmsh.studio` opens the Qt host (phase tabs on `ViewerWindow`). | S1, ADR 0088, 0094 S1 |
| **S3** | Qt protocol: `highlight(names)` mutator; phase resource. Same payload as the JSON. | S2, ADR 0056 |
| **S4** | MCP wrapping S1–S3 (`get_selection`, `highlight`, `run_until`, `assess`, `render`, `promote_selection`). Cursor MCP config. | S3; this is 0094’s “MCP wrapping S5” consumer |
| **Later** | Statement-step BRep (geometry only). Host projections (read-only script view + PG→apeSees binding graph, INV-9). Live refresh loop (file-watch / Refresh, worker replay, camera-preserving scene swap). Electron/trame chrome after ADR 0047 R-D, only if non-3D studio UI outgrows Qt. In-app agent habitat (the Electron-as-shell fork) — requires an explicit habitat amendment, not a drive-by. | S4 |

S0 can merge alone (this PR). Claiming the feature “done” requires
S0 + S1 + S2 (a human can pick, an agent can read the envelope and
refresh). S4 is the Cursor-native door.

## Acceptance

- A `SelectionEnvelope` round-trip test: `SelectionState` with a
  labeled face projects to JSON whose identity fields are the label /
  PG / phase, not only dimtags.
- Failed replay leaves the previous tessellation on the host (INV-4);
  a unit test on the runner, not a GUI click test.
- `apeGmsh.studio` AST-guard: not imported from `apeGmsh/__init__.py`;
  not referenced from `_COMPOSITES`.
- Skill happy path after solve still does not mention `viewer()` as
  the check (0094 INV). Skill spatial-edit path mentions the envelope
  file once S1 lands.
- No Electron, no MCP server, and no statement-step runner in S0–S2.

## Open questions

Resolved in this draft (veto here):

1. **Sidecar, not composite** — yes. Same as hpc / assess.
2. **v1 host is Qt** — yes. Electron is a named later skin.
3. **v0 transport is JSON** — yes. MCP is S4.
4. **Preview default is refresh / last-good-frame** — yes.
   Statement-step is Later.
5. **Cursor stays the IDE** — yes, until an explicit habitat
   amendment.
6. **Host may project the script read-only** — yes (INV-9). Not an
   editor. Binding graph is the same class of projection.

Still open (do not block S0–S2):

1. Envelope file path: workspace `.apegmsh/selection.json` vs next to
   the script vs a temp dir the skill is taught. Decide at S1 with
   the code in hand.
2. Does `python -m apeGmsh.studio` attach to an already-open
   `ViewerWindow` or always spawn the host? Lean spawn; attach is
   S3.
3. Geometry-hash grain for skip-remesh: AST of the `with` body vs
   serialized OCC vs `model.h5` hash. Decide at S2.
