# ADR 0095 — `apeGmsh.studio`: agent + script + viewer habitat

**Status:** Accepted (2026-08-16 — S0–S4e, S5a–S5j, and F-status-1
shipped; Amendments 1–3 closed; contract 1.4.0. Proposed 2026-08-12;
owner workshop the same day; host-projection / INV-9 workshop
2026-08-13)

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
| **S2b** | Truthful `run_until(phase)`: `model` raises `StopAtPhase` at `generate()`, `mesh` at `apeSees`/`Results`; dump `.apegmsh/names.json` (labels / PGs / counts / bboxes) at the stop. Skip-remesh keys on `(hash, phase)`. | S2 |
| **S2c** | Append-only `.apegmsh/runs.jsonl`: timestamp, hash, phase, stopped_at, labels / PGs / counts. Skip does not append. Failed exec appends `ok: false`. | S2b |
| **S2d** | `--status` / `collect_status`: read names + last ledger line + pick without replay. PoC fixture `tests/studio/fixtures/box.py`. | S2c |
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
- A `run_until(phase="model")` test: a script that would mesh and
  construct `apeSees` does not call `generate()`; `.apegmsh/names.json`
  identity fields are the label / PG, and element counts at dim 3 are 0.
- A second session can reconstruct phase / counts / names from the last
  line of `.apegmsh/runs.jsonl` plus `names.json`, without the chat.
  Skip-remesh does not append a duplicate line.
- `--status` on a workspace that has those files prints phase / names /
  counts without calling `run_until`; an empty `.apegmsh/` exits 2.
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

## Amendment 1 (2026-08-14) — MCP growth law, ReportBundle, emit, animate

Append-only. Parts 1–6, INV-1–INV-9, S0–S3, and S2b–S2d stay. This
amendment restates **S4** so the Cursor adapter can land without
freezing a moving library, and without waiting on `highlight`
(S3). It does not reopen sidecar / script-owns-geometry / names-first
/ last-good-frame / no shared kernel / 0094 stands / Cursor-as-IDE.

Evidence: owner workshop on ImageMage Studio (self-documentation,
reporting, results pin, stills, animations) plus the shipped S1–S2d
payload (`SelectionEnvelope`, truthful `run_until(phase)`,
`names.json`, `runs.jsonl`, `--status`). Constrained by ADR 0094
INV-10 (closed `view=` set) and ADR 0076 (derived scalars).

### Growth law (INV-10)

Skills write Python against apeGmsh. MCP wraps **habitat verbs**
only. New geometry, materials, constraints, and solvers never become
MCP tools.

MCP may wrap:

| Family | Tools | Payload already in tree |
|---|---|---|
| **See** | `status` / `inspect`, `get_selection` | `--status --json`, `names.json`, `selection.json` |
| **Judge** | `assess` | `fem.assess()` / `results.assess()` (ADR 0094) |
| **Show** | `render`, `animate(kind=)` | `viewers.render`, `results.export_animation` |
| **Pin** | `results_get`, `results_pin` | `Results.from_*` + `model.h5`; ledger keys reserved |
| **Emit** | `emit_report(format=)` | ReportBundle → markdown / html / canvas |
| **Point** | `highlight`, `promote_selection` | S3; not required for read-only MCP |

MCP must not wrap: `g.model.*` / `g.mesh.*` / `g.constraints.*` /
`g.rebar.*`; `apeSees` primitives; tags or node ids as identity;
`viewer()` / `show_web` / Outline clicks; `setup=` on `animate`;
character-by-character exec; a second editor or prompt.

v1 transport is **CLI / JSON wrappers**. Tools shell to
`python -m apeGmsh.studio` or read `.apegmsh/*.json`. The MCP process
does not import a live Gmsh session (INV-5). A long-lived studio
daemon is not required for S4a. A new ADR is owed only if that
process model later changes.

### ReportBundle and emit skins

Self-documentation is not MCP inventing prose. Studio assembles one
frozen **ReportBundle** from pieces the library already produces:

| Field | Source |
|---|---|
| names / counts / bboxes | `names.json` |
| pick | `selection.json` |
| findings + markdown text | `AssessmentReport` |
| stills (PNG paths) | `render` / `render_pack` |
| clips (mp4/gif paths) | `export_animation` / `animate` |
| lineage / results handle | `Results` + `model.h5` |
| run id + hashes | `runs.jsonl` |

**Markdown** is the canonical record (git-tracked chapter in the
model workspace `docs/`). **HTML** is a shareable/print skin of the
same bundle. **Canvas** (Cursor `.canvas.tsx`) is a live review
projection beside chat — not git-portable, not the archive. The
agent may emit a canvas; it must still write the Markdown chapter.
Chat is not the archive.

`emit_report(format=)` is one tool. New narrative is not a new MCP
verb.

### Authored vs generated (INV-12)

`.apegmsh/` stays generated (already gitignored): selection, names,
ledger, working visors. Authored docs, captioned figures, and pinned
runs that must survive review do **not** live there. A visor is a
working still or clip; a **figure** is a visor that earned a caption
and a home in the chapter.

### Animate kinds (INV-11)

One tool, `animate(kind=)`. New motion is a new kind token — same
discipline as ADR 0094 INV-10 for stills. `setup=` stays a
human/script escape hatch on `export_animation`; it is not on the
MCP.

| kind | What it shows | Engine | When |
|---|---|---|---|
| `history` | Deformation / contour over recorded steps | `results.export_animation` (mp4/gif) | S4b — proves the tool |
| `yield` | Von Mises contour on an auto-scaled deforming mesh (stills 0.12 of diagonal). Not an iso-clip of \(f(\sigma)=f_y\); π-plane later | Same encoder; derived scalars (ADR 0076); named kind | S4d — proves the catalog |
| `formation` | How the mesh / elements come into being | Not in the library yet | Later |

Two implementations (`history`, then `yield`) are enough to prove
the tool. `formation` does not block S4.

### S4 restated

The original S4 row (“MCP wrapping S1–S3”, depends on S3) is
replaced for implementation order only. S3 v1 is the INV-5 file
adapter (`highlight.json` + host poll + `select_batch`). A Qt
`viewer.highlight(names)` mutator is later. Read-only MCP does
**not** wait on it.

| Slice | Ships | Depends |
|---|---|---|
| **S4a** | Read-only MCP: `status`, `get_selection`, `run_until`. Cursor MCP config. Shells to the existing CLI / JSON. | S2d |
| **S4b** | Studio verbs + MCP for `assess`, `render`, `animate(kind=history)` | S4a, ADR 0094 S1–S2 |
| **S4c** | `results_pin` + `emit_report(format=markdown)`. HTML / canvas are later skins of the same bundle | S4b |
| **S4d** | `animate(kind=yield)`; `highlight` / `promote_selection` (S3) | S4b, S3 |
| **S4e** | `emit_report(format=html\|canvas)` skins of the same bundle. Markdown is still written as the archive. | S4c |
| **Later** | `kind=formation`; statement-step; daemon-as-middle-tier; Electron | S4c / S4d / S4e |

S4a can merge alone. Claiming “the MCP exists” requires S4a (an
agent reads a pick and names without opening Qt). Claiming
self-documentation requires S4c.

### Alternatives rejected (this amendment)

| Rejected | Why |
|---|---|
| **MCP tools for `g.model.*` / `apeSees` primitives** | Freezes a moving library. Skills already author Python. |
| **Wait for S3 before any MCP** | Read-only verbs are already files. Empty ceremony, same class as “MCP before the payload.” |
| **New ADR 0096 for the adapter** | S4 was always the Cursor door. A new ADR is owed only if v1 CLI/JSON is reversed. |
| **`write_geometry_chapter` / `explain_aci` tools** | Invented prose. Emit from the bundle. |
| **Canvas or HTML as the archive** | Not git-portable (canvas) / not the source of truth (HTML). Markdown chapter is. |
| **Authored docs in `.apegmsh/`** | Gitignored live state. They would vanish from review. |
| **`setup=` on the MCP `animate` tool** | Open hatch. New pictures are new `kind=` tokens, reviewed here. |
| **Scaffold `memory/` + project ADRs + four INDEX files in this PR** | Premature. The bundle and the ledger are the day-3 archive. Revisit after S4a–S4c have been used. |

### Acceptance (this amendment)

- This file names INV-10 / INV-11 / INV-12 and the S4a–S4d table.
- S4a ships here: `python -m apeGmsh.studio.mcp` with tools `status`,
  `get_selection`, `run_until` (CLI/JSON wrappers; no live Gmsh in
  the MCP process).
- S4b ships here: `assess`, `render`, `animate(kind=history)`.
  `render` shells to `python -m apeGmsh.viewers render`.
- S4c ships here: `results_pin` + `emit_report(format=markdown)`.
  ReportBundle schema is 1. HTML / canvas skins are later.
- S4d ships here: `animate(kind=yield)` is a von Mises contour of a
  derived fluency scalar (`von_mises_stress`, ADR 0076) on
  stills-style auto-scaled deform (0.12 of the model diagonal) — not
  an iso-clip of \(f(\sigma)=f_y\), not a π-plane.
  `highlight(names)` writes `.apegmsh/highlight.json` only (does not
  write `selection.json`); the Qt host applies names via live
  resolve + `select_batch` (INV-7) after a mtime change, ignoring a
  leftover file at host open. Dimtags in the JSON are evidence
  (INV-3). `promote_selection` returns
  `g.model.select(None, dim=).in_box(lo, hi).to_label()` /
  `.to_physical()` and does not write the `.py` (INV-2 / INV-9).
  File poll is the S3 adapter, not a Qt `viewer.highlight(names)`
  mutator.
- Skill Studio paragraph: S4a–S4d tools; do not wrap library
  composites; visors stay under `.apegmsh/visors/`; authored docs
  live in `docs/`.
- Original S0–S2d acceptance still holds.

### Open questions (this amendment)

Resolved here:

1. **v1 MCP = CLI / JSON wrappers** — yes.
2. **Read-only S4a before S3** — yes.
3. **Markdown canonical; HTML and canvas are skins** — yes.
4. **One `animate(kind=)` tool** — yes. `history` then `yield`.

Still open (do not block S4a–S4d):

1. Canvas output path (IDE `canvases/` vs project `docs/`) — decide
   at the canvas-skin slice. **Resolved in Amendment 2** (IDE
   `canvases/`).

Resolved at S4c:

1. **ReportBundle schema integer** — 1.

Resolved at S4d:

1. **`yield` first picture** — von Mises contour of a derived scalar
   (`von_mises_stress` / fluency) on auto-scaled deform (0.12 of
   diagonal). Not an iso-clip. π-plane is later.

## Amendment 2 (2026-08-14) — HTML and canvas emit skins

Append-only. Parts 1–6, INV-1–INV-12, S0–S4d, and Amendment 1 stay.
This amendment ships the skins S4c deferred. It does not add MCP verbs,
does not make canvas or HTML the archive, and does not reopen
formation / statement-step / daemon / Electron.

### Path law

| format | Default path | Role |
|---|---|---|
| `markdown` | `docs/studio-report.md` | Archive (git-portable) |
| `html` | `docs/studio-report.html` | Shareable / print skin of the same bundle |
| `canvas` | IDE `canvases/studio-report.canvas.tsx` | Live Cursor projection. Not git-portable. |

Canvas resolution, in order: `output=` (must be a `.canvas.tsx` path);
`CURSOR_CANVAS_DIR`; `~/.cursor/projects/<slug>/canvases/` when that
folder exists. If none of those exist, raise — do not write a
non-live `.canvas.tsx` under `docs/`. `format=html` and
`format=canvas` still write the markdown chapter (ADR: the agent may
emit a canvas; it must still write the Markdown chapter).

`<slug>` is Cursor's project folder name: absolute path, strip `:`,
`/`, `\`, and `.`, lowercase the drive letter.

### Slice

| Slice | Ships | Depends |
|---|---|---|
| **S4e** | `emit_report(format=html\|canvas)`. Same ReportBundle schema 1. No new MCP tool. | S4c |

### Alternatives rejected (this amendment)

| Rejected | Why |
|---|---|
| **Canvas under `docs/` as the default** | Looks archivable. Cursor will not live-load it. The open question was this fork. |
| **HTML as the archive** | Print skin. Markdown stays the source of truth. |
| **New MCP verbs `emit_html` / `emit_canvas`** | INV-10: new narrative is not a new verb. `format=` on the existing tool. |
| **Embed visor PNGs as data URIs in the canvas** | Bloats the projection; pictures belong in the markdown/HTML chapter. |

### Acceptance (this amendment)

- `emit_report(format="html")` writes `docs/studio-report.html` and
  `docs/studio-report.md`. Labels are HTML-escaped.
- `emit_report(format="canvas", output=…)` writes a `.canvas.tsx` that
  imports only from `cursor/canvas` and a markdown archive. Without
  `output=` / `CURSOR_CANVAS_DIR` / an existing IDE `canvases/` dir,
  it raises.
- No authored file lands in `.apegmsh/` (INV-12).
- Skill Studio paragraph names `format=markdown|html|canvas`.
- `kind=formation` stays later.

### Open questions (this amendment)

Resolved here:

1. **Canvas output path** — IDE `canvases/`, not project `docs/`.

## Amendment 3 (2026-08-16) — explicit root and the out-of-process contract freeze

Append-only. Parts 1–6, INV-1–INV-12, S0–S4e, and Amendments 1–2 stay.
This amendment hardens the habitat for **any** out-of-process consumer
(Cursor MCP, a test harness, a future product shell). It does not reverse
CLI/JSON transport, does not require a long-lived daemon, and does not
add MCP verbs for host lifecycle (INV-6).

**Motivation (non-normative):** a separate product shell that consumes
Studio’s contract motivated this freeze. The contract is deliberately
consumer-agnostic; Studio does not depend on that product.

### New invariants

- **INV-15 (explicit root).** The habitat root is resolved by one rule:
  explicit `root=` / `--root` → `APEGMSH_ROOT` → nearest ancestor of the
  start path that contains a `.apegmsh/` directory → process cwd.
  A `.apegmsh/` under the user home is ignored unless the start path
  *is* home (stale `~/` habitat must not capture unrelated projects).
  Cwd is the last fallback, never the definition. One MCP process MAY
  serve multiple projects by passing different `root=` values. Habitat
  writers (`names.json`, `runs.jsonl`, `mcp_calls.jsonl`, …) use that
  root even when script exec temporarily `chdir`s to `script.parent`.
  Ledger records MAY carry additive `root` and `cwd` fields.
- **INV-16 (atomic single-writer).** Snapshot habitat JSON
  (`selection.json`, `names.json`, `highlight.json`) is written
  atomically by a single owner; readers tolerate torn / unparseable
  payloads without raising out of `status`. JSONL files
  (`runs.jsonl`, `mcp_calls.jsonl`) are append-best-effort and MAY have
  more than one appender; `status` skips bad ledger lines and reports
  `ledger_error`. *(S5b + S5f.)*
- **INV-17 (versioned published contract).** The out-of-process contract
  is versioned and published as schemas + golden fixtures; unknown major
  is refused. *(Shipped in S5d.)*

### Slices

| Slice | Ships | Depends |
|---|---|---|
| **S5a** | `resolve_root`; `root=` on every MCP tool + CLI `--root`; replay / mcp_calls write under that root; `scripts/studio-mcp.ps1` no longer `Set-Location`s the library repo | S4e |
| **S5b** | Atomic writes (`os.replace`) for envelope / names / highlight; `status` degrades on torn `names.json` | S5a |
| **S5c** | Uniform `{ok, error:{code,message}}` MCP envelope; subprocess timeout; `--json` for render/animate workers | S5a |
| **S5d** | JSON Schema package data + golden fixtures + `contract_version` on `status` | S5a |
| **S5e** | One docs page: spawn, file ownership, poll contract, cold-start, local trust boundary | S5a |
| **S5f** | Red/blue harden: ledger torn-line degrade (`ledger_error`); docs honesty on JSONL vs atomic JSON; empty `root=` / `APEGMSH_ROOT` treated as unset | S5b, S5e |
| **S5g** | `status.host` via `host.json` (claim/clear + PID liveness); `.apegmsh/project.json` entry script; `run_until` may omit `script=` | S5f |
| **S5h** | Habitat `busy.json` lock; concurrent `run_until` → `BUSY`; stale PID steal; `status.busy` | S5g |
| **S5i** | Root-relative `script` / `cwd` / MCP path fields under the habitat root; `root` stays absolute | S5h |
| **S5j** | Opus harden: torn `busy.json` grace; Windows `pid_alive` ACCESS_DENIED; CLI `project.json`; ownership on clear/release; `OUTSIDE_ROOT`; MCP `display_path` widen | S5i |
| **F-status-1** | MCP `status(mode=brief\|full)` — brief default for agent token budget; full keeps `collect_status` | S5j |

### Alternatives rejected (this amendment)

| Rejected | Why |
|---|---|
| **New ADR for the freeze** | Amendment 1’s bar (“new ADR only if CLI/JSON is reversed”) is not met |
| **Product-shell invariants in this ADR** | Pre-commits a product that is not this repo; keep normative text consumer-agnostic |
| **MCP verbs for host open/close** | INV-6; surface host presence later via `status` fields + spawn docs (S5e), not puppeteering |
| **Freeze schemas before root** | Path semantics change with INV-15; freeze after S5a |
| **Keep `Set-Location` to the library repo in `studio-mcp.ps1`** | Silently pins one project and fights explicit `root=` |

### Acceptance (S5a)

- `resolve_root` honors explicit → `APEGMSH_ROOT` → nearest `.apegmsh/` → cwd.
- Every MCP tool accepts optional `root=`; `run_until` passes `--root` to the CLI subprocess.
- Two `run_until` calls with different `root=` from one process write two `.apegmsh/` trees; `status(root=A)` never sees B’s ledger.
- Replay dumps `names.json` / `runs.jsonl` under the habitat root even when exec cwd is `script.parent`.
- `mcp_calls.jsonl` lands under the same resolved root as other habitat files.
- `scripts/studio-mcp.ps1` does not `Set-Location` the apeGmsh checkout.

### Acceptance (S5b)

- `write_envelope` / `write_names` / `write_highlight` use temp + `os.replace`
  (no truncate-in-place).
- `collect_status` never raises on torn / unparseable `names.json`; it
  returns `names=null` and `names_error` (parity with `selection_error`).
- A concurrent rewrite + `status` poll test completes with zero exceptions
  and never returns a partial names object.

### Acceptance (S5c)

- Validation failures (`phase`, `view`, `kind`, empty highlight, …) return
  `ok: false` with `error: {code, message}` — no raise into the MCP adapter.
- `_shell` honors `APEGMSH_STUDIO_TIMEOUT` (default 600s); timeout →
  `error.code == "TIMEOUT"`.
- `viewers render --json` and `studio --animate --json` emit a
  `{"ok", "written"}` object; MCP prefers that over path-line scrape.

### Acceptance (S5d)

- `apeGmsh/studio/schemas/*.schema.json` + `fixtures/*.json` ship as
  package data (INV-17).
- `status` includes `contract_version` (semver, currently `1.0.0`).
- Golden fixtures validate; unknown contract major / wrong file-schema
  int is refused; additive unknown keys are ignored.

### Acceptance (S5e)

- Published how-to `docs/how-to/studio-habitat.md` covers spawn (MCP +
  host), root rule, file ownership, poll mtimes, cold start, and the
  local trust boundary for `run_until`.
- Linked from the How-to index and `mkdocs.yml` nav.

### Acceptance (S5f)

- `read_runs` / `collect_status` never raise on torn `runs.jsonl` lines;
  good objects are kept and `ledger_error` names the skip (including a
  UTF-8 tear mid-line — decode with replace, skip the bad line).
- Habitat how-to states atomic replace for snapshot JSON only; JSONL is
  append-best-effort with multi-appender `runs.jsonl`.
- Empty / whitespace `root=` and `APEGMSH_ROOT` are treated as unset
  (INV-15 chain continues); CLI `--root` is a string so `""` does not
  become `Path(".")`.
- `promote_selection` returns `{ok:false, error.code=UNREADABLE}` on torn
  `selection.json` (parity with `get_selection`).
- Subprocess MCP verbs return `BAD_ROOT` when the resolved root is not a
  directory (no raise into the adapter).

### Acceptance (S5g)

- Qt `open_host` claims `.apegmsh/host.json` for the duration of `show()`
  and clears it afterward; `status.host` reports `{running, pid, phase,
  stale}` and treats a dead PID as `running=false` / `stale=true`.
- Successful replay writes `.apegmsh/project.json` with the entry
  `script` (root-relative when possible).
- MCP / CLI `run_until` may omit `script=` and use `project.json`;
  missing both → `NO_SCRIPT`.
- Published schemas/fixtures cover `host` and `project`;
  `contract_version` is `1.1.0`.

### Acceptance (S5h)

- `ReplayRunner.run_until` acquires `.apegmsh/busy.json` via
  `O_CREAT|O_EXCL` for the duration of an exec (skip-hash path does not
  lock); a second concurrent claim fails.
- Stale busy claims (dead PID) are stolen; `status.busy` reports
  `{busy, pid, op, phase, stale}`.
- MCP `run_until` returns `error.code == "BUSY"` (CLI exit 3) when the
  habitat is locked; `contract_version` is `1.2.0`.

### Acceptance (S5i)

- Ledger `script` / `cwd` and MCP tool path fields that fall under the
  habitat root are stored as root-relative posix paths; the `root`
  field remains absolute.
- Paths outside the habitat root stay absolute. `contract_version` is
  `1.3.0`.

### Acceptance (S5j)

- Empty / unparseable mid-write `busy.json` is not stolen until a short
  grace window elapses; live busy claims are never stolen by peers.
- Windows `pid_alive` treats `OpenProcess` ACCESS_DENIED as alive.
- CLI `run_until` may omit the script path and use `project.json`
  (parity with MCP); relative script args resolve under `--root`.
- `write_project` runs only on successful replay with a live session;
  `clear_host` / `release_busy` only clear this process's claim.
- `project.json` default outside the habitat root → `OUTSIDE_ROOT`
  (MCP) / CLI exit 2. `contract_version` is `1.3.1`.

### Acceptance (F-status-1)

- MCP `status` defaults to `mode="brief"`: root, last phase/ok/script,
  label + physical-group lists, run/pin counts, pick summary, and
  `text` matching CLI `format_status` — no `names.entities`.
- Brief payload on a names dump with hundreds of entities stays
  ≤ ~2 KB; `mode="full"` still returns `collect_status`.
- Habitat how-to / skill: root check → brief; entity hunting →
  `lookup` / `names.json`. `contract_version` is `1.4.0`.

### Open questions (this amendment)

Resolved here:

1. **Root resolution order** — INV-15 as above.
2. **Amendment vs new ADR** — amend 0095.
3. **`status.host`** — S5g (`host.json` + PID check).
4. **`.apegmsh/project.json`** — S5g (entry script).
5. **Habitat concurrency lock / `BUSY`** — S5h.
6. **Root-relative paths in status payloads** — S5i.
7. **S5g–S5i harden** — S5j.
8. **MCP status brief mode** — F-status-1.

Still open (do not block S5a–S5j / F-status-1):

1. Event stream (mtime poll remains the contract until proven insufficient).

## Amendment 4 (2026-08-16) — live refresh (S6) and Later-shelf pruning

Append-only. Parts 1–6, INV-1–INV-17, S0–S5j, F-status-1, and
Amendments 1–3 stay. This amendment ships the Later row's "live
refresh loop" as S6 and prunes two Later items that no longer earn
their place. It does not reopen transport, the busy/host contract,
statement-step-as-shipped, or Electron gating.

Context: the shipped host opens once — `python -m apeGmsh.studio
script.py --phase X` replays in-process, opens the Qt window
(MeshViewer if meshed, else ModelViewer), and never updates again.
The product promise (Part 4: "preview should show geometry as the
script becomes runnable") is not real until the window follows the
file. The highlight watcher already proves the pattern in-tree: a
QTimer mtime poll applying changes through owner mutators (INV-7).

### New invariants

- **INV-18 (refresh never steals).** A host refresh honors
  `busy.json` exactly as S5h left it: when a live claim exists (an
  agent's `run_until` is executing), refresh reports busy in the
  status bar and does nothing. Stale-claim stealing stays the
  runner's; the refresh path adds no second steal heuristic.
- **INV-19 (disk is the source).** Refresh replays the file as
  saved on disk. Unsaved editor buffers are Cursor's; studio never
  sees them. A corollary of INV-2, stated because file-watch makes
  the temptation concrete.

INV-4 (last good frame) now binds the host explicitly: a failed
refresh keeps the previous scene and camera; the failure lands in
`runs.jsonl` (`ok: false`) and the status bar — not a dialog.

### Slices

| Slice | Ships | Depends |
|---|---|---|
| **S6a** | Manual Refresh (toolbar action / F5): re-run the entry script (`project.json`, else the script the host opened) at the current phase through `ReplayRunner.run_until` in the host process — busy claim per S5h (INV-18), skip-hash no-op ("Up to date"), success rebuilds the scene preserving camera (the `_rebuild_scene` pattern) and rewrites `names.json` / appends `runs.jsonl` through the existing writers. Ledger records gain an additive `trigger` field (`open` / `refresh` / `watch`); ledger schema + golden fixtures updated; `contract_version` 1.5.0. | S5j |
| **S6b** | Auto-refresh: QTimer mtime poll on the entry script (~1 s, debounced until the mtime is stable for one tick), on by default, `--no-watch` opt-out plus a host toggle. Poll, not an event stream — the standing open question stays resolved as-is. The unchanged `(hash, phase)` skip makes idle saves free. | S6a |
| **S6c** | Phase selector on the host (the phase tabs the original S2 named), gated by INV-8: a refresh may advance model → mesh → results as `FEMData` / `Results` appear and must retreat when they do not. `host.json.phase` updates on swap (value change, no schema change). | S6a |

S6a can merge alone. Claiming "the studio is live" requires S6a +
S6b (save in Cursor, watch the window follow).

### Later-shelf pruning

- **Qt `viewer.highlight(names)` direct mutator — retired.** The S3
  file adapter (`highlight.json` + QTimer poll + `select_batch`) is
  the contract and serves every cross-process consumer; a direct
  mutator would serve only in-process callers that do not exist.
  Struck from Later; reviving it requires an amendment with a named
  consumer.
- **Statement-step BRep — demoted.** S6b's save-triggered refresh
  covers the intent at file grain. Statement-step is no longer a
  standing commitment: it is re-evaluated only if dogfood shows
  save-grain refresh is insufficient for geometry authoring, and
  closed otherwise.

Still parked, unchanged: Electron/trame (ADR 0047 R-D gate),
`kind=formation`, daemon-as-middle-tier, event stream (mtime poll
until proven insufficient), in-app agent habitat (explicit habitat
amendment).

### Alternatives rejected (this amendment)

| Rejected | Why |
|---|---|
| **Worker-process replay for refresh** | The host already owns the live session and kernel; a worker per refresh doubles kernel cost and re-opens the attach question S2 settled (spawn). Blocking the UI thread for the replay duration is v1-acceptable; the status bar says "Replaying…". A worker is owed an amendment only if that proves unacceptable. |
| **File-system watcher (`QFileSystemWatcher` / watchdog)** | Event streams are the open question mtime poll already answers; the highlight watcher set the poll precedent. Revisit only with evidence that poll is insufficient. |
| **Failure dialogs** | INV-4 is a quiet contract: last good frame + status bar + ledger. A dialog interrupts the human mid-edit. |
| **Auto-refresh default off** | The default that makes the product real is on; `--no-watch` and the toggle cover recording / benchmark sessions. |
| **Keeping the highlight mutator and statement-step on Later** | An unbounded Later shelf reads as commitments. Pruned explicitly; revival needs an amendment. |

### Acceptance (this amendment)

- **S6a:** runner-level tests, no GUI — refresh while a peer holds
  `busy.json` returns busy and leaves the scene object untouched
  (INV-18); an unchanged file + phase is a no-op that appends
  nothing; a failed replay keeps the previous tessellation object
  (INV-4) and appends `ok: false` with `trigger: "refresh"`. A
  Qt-marked smoke covers the action. `contract_version` is 1.5.0;
  the ledger schema and golden fixtures carry `trigger`; readers
  ignore its absence (additive, INV-17).
- **S6b:** debounce is a pure function with unit tests (mtime
  flapping during a save produces one replay); `--no-watch` disables
  the timer; a watch tick honors INV-18 the same as manual refresh;
  watch-initiated runs append `trigger: "watch"`.
- **S6c:** a refresh whose replay binds `Results` enables results
  mode; a refresh whose replay stops at `model` disables mesh /
  results modes again (INV-8); `host.json.phase` reflects the swap.
- Skill Studio paragraph: refresh + watch exist; the agent still
  uses `run_until` / `status` / the envelope, never the host window
  (INV-6).

### Open questions (this amendment)

Resolved here:

1. **Refresh runs in the host process** — yes (v1).
2. **Auto-refresh default on** — yes, with opt-out.
3. **Highlight mutator retired; statement-step demoted** — yes.

Still open (does not block S6):

1. Watch-poll period and debounce constants — fix at S6b with the
   code in hand; acceptance pins behavior, not milliseconds.

