# apeGmshViewer

A read-only desktop app that opens a `model.h5` and shows the model: the mesh
coloured by physical group, and, for a clicked element, its definition chain
(element → geomTransf → beamIntegration → section → materials) with the HDF5
path behind every value. This is the P0 spike of
[ADR 0112](../architecture/decisions/0112-files-are-the-model-and-a-read-only-app.md).

The app reads files only. It never imports apeGmsh, never runs Python and
starts no process other than its own Electron binary (ADR 0112 D4) and, for
go-to-source, the user's editor (D3). What it knows about a model comes from
the file and from [`architecture/h5-schema.md`](../architecture/h5-schema.md).

## Install (Windows)

apeGmsh writes `model.h5` (and `<stem>.geometry.h5`) on every run, but it
does not install the viewer: until you install it once, "Open with" does not
list apeGmshViewer and double-clicking a `.h5` does nothing for it. You need
Node 22.18 or newer and this repository checked out. Build the installer from
this directory, then run the `.exe` it writes:

```powershell
cd apeGmshViewer
npm ci
npm run dist        # writes out/installer/apeGmshViewer Setup <version>.exe
& ".\out\installer\apeGmshViewer Setup 0.0.1.exe"
```

To try it without installing, `npm start -- path\to\model.h5` runs it from the
checkout (see [Commands](#commands)); "Open with" lists it only once the
installer has run.

The installer is per-user: it installs under `%LOCALAPPDATA%\Programs` and
adds apeGmshViewer to the **"Open with"** list of `.h5` files under `HKCU`, so
it needs no admin rights. It does not make itself the default `.h5` program,
because other tools open `.h5` too. To open every `.h5` with it on
double-click, choose it once in "Open with", then "Always". It is not
code-signed, so SmartScreen may ask once ("More info", then "Run anyway").
Uninstall it from Windows Settings, Apps; the uninstaller removes the `.h5`
registry entries the installer wrote (`packaging/installer.nsh`).

Then:

- **Open** a `.h5` file: "Open with", apeGmshViewer, or a double-click once it
  is your `.h5` program. A second open goes to the window already running.
- **Drag and drop** a file onto the window.
- **The set.** Opening `<stem>.h5` also opens `<stem>.geometry.h5` and
  `<stem>.results.h5` when they exist beside it. Opening the geometry or
  results file finds its model the same way.
- **Watching.** Each run rewrites the files (ADR 0112 D1). The app reloads a
  file once its writer has stopped for 300 ms, and picks up a sibling that
  appears or vanishes.
- **Go to source** opens the user's editor at a line. It uses
  `$APEGMSH_EDITOR` when set, a command template with `{file}` and `{line}`
  (for example `code -g {file}:{line}`, or
  `"C:\Program Files\Notepad++\notepad++.exe" -n{line} {file}`). Otherwise
  it uses VS Code (`code -g file:line`) when `code` is on PATH, and otherwise
  the OS text editor, without the line (Notepad on Windows). It never uses
  the OS default opener, because on Windows that runs a `.py` file.
- **Results.** A `<stem>.results.h5` opens like the model: "Open with", a
  drop, `npm start -- x.results.h5`, or beside an opened model. It is drawn
  on the model only when its `/meta/lineage/model_hash` equals the model's
  (else its `fem_hash`, which matches the mesh only): never by name or path.
  A results file of another model, or one with no model open, is not drawn,
  and the banner says why (the file's own embedded `/model` is not drawn
  yet). A paired file adds **results** to the phase bar, with the time
  scrubber: the results stage, the step slider, the time, play/pause and
  step -1/+1. Steps are read one at a time in the main process, when shown;
  the file is closed after half a second without a read (so a writer can
  replace it), and a file that changed is re-read before any step is shown;
  the newest request wins. The viewport draws the undeformed mesh unless
  the deformed shape is on.
- **Deformed shape.** On a results stage, **deformed** in the legend moves
  every drawn node to `x + scale · u`, with `u` the file's nodal
  `displacement_x`, `_y` and `_z` at the step shown (rotations are not
  used). The scale defaults to **1** (true scale); any other scale is
  stated in the legend. **reference ghost** draws the undeformed mesh behind
  it at 0.3 opacity. Beams and trusses stay straight chords between their
  displaced end nodes. It composes with contours (the deformed mesh is
  coloured), stage activation and the time scrubber, and a click picks what
  is drawn where you click. A drawn node with no displacement (not in
  `_ids`, or NaN) is drawn undeformed and marked with a grey square, and
  the banner says how many there are: it is never moved by a displacement
  of 0. The deformed shape is refused, with the reason, when an axis was
  refused as a partial write, when the components name no axis (`[k]`),
  or when `displacement_z` is missing from a model that is not declared
  2-D (`/meta/ndm` 2, neutral 2.34.0 on): it is never drawn from the
  surviving axes. The outline edges follow their own nodes, like the
  faces; the ghost keeps the undeformed ones.
- **Nodal contours.** On a results stage, **contour** in the legend colours
  the mesh by a nodal field at the step shown: pick the field and its
  component (`x`, `y`, `z` as the file names them, the **magnitude** for a
  translational vector such as displacement, velocity, acceleration or
  reaction force, `[k]` for an unnamed component). The colour range is the
  field's **whole history** on that stage by default: main reads every step
  in the background, and until it is done the range is the step's own, with
  "range: this step (history scan n%)" in the legend. **this step** and
  **fixed** (your min and max) are the other modes, and the legend always
  says which is in use. The colormap is **jet**, or viridis or coolwarm. A
  node the field does not record (not in its `_ids`, or NaN) is **no data**,
  in the grey the legend names: every face or line that touches it is drawn
  in it, never as zero and never blended from its neighbours. The file
  stores no units, so the legend shows none. A component whose value columns
  disagree with `_ids` (a partial write) is not read, and the banner says so;
  a re-read that no longer has the picked component clears the pick, with a
  notice.
- **Stage activation.** On a results stage of a staged model, the elements
  that stage does not have are hidden: those a later stage activates
  (`s.activate`) and those it or an earlier stage removed
  (`s.remove_element`), as the model's `/sequence` element rows say. The
  results stage pairs with a `/sequence` stage by name. No match (a `mode_1`
  stage), two matches, an unnamed results stage, or a model with no
  `/sequence` draws every element under a banner that names the state; a
  model with no stages draws every element without one. Results paired by
  `fem_hash` only are not tied to the model's program, so they are drawn whole
  under an `unbound` banner; `/sequence` element rows the app cannot apply
  without guessing are `refused`, also with a banner.
- **Model-definition windows.** The **Windows** menu at the bottom of the
  window opens Stages, Analysis, Patterns & series, Recorders, and Materials &
  sections. Each lists what the file declares, with the HDF5 path of every
  row; a material or section row selects it in the inspector.

## Commands

Run from this directory with Node 22.18 or newer (TypeScript runs natively
through Node's type stripping; nothing is compiled for the tests).

| Command | Does |
|---|---|
| `npm ci` | install the exact pinned dependencies from `package-lock.json` |
| `npm run typecheck` | `tsc` over `src/` and `test/` |
| `npm test` | reader, chain resolution, mesh and navigation tests, on the fixture and on synthetic files |
| `npm start -- <model.h5>` | open the app on a file; without a file, drop one on the window |
| `npm run measure -- <model.h5> [--uncapped]` | print one [MEASUREMENTS.md](MEASUREMENTS.md) row; `--uncapped` lifts the vsync cap and prints the throughput row; with a `<stem>.results.h5` beside the model, also the contour scrub row (V4c1) and the same scrub with the deformed shape on (V4d) |
| `npm run capture -- <model.h5> <out.png> [--pick=<OpenSees type>]` | write a still from a hidden window, with a beam selected (or an element of the `--pick` type); a second still `<out>.chain-end.png` when the chain is taller than the window; `<out>.framed.png`, `<out>.groups.png` and `<out>.role.png`; one `<out>.window-<kind>.png` per model-definition window; and, when the model has a `<stem>.results.h5` beside it, one `<out>.results-<i>-<stage>.png` per results stage at its last step, then the contour stills `<out>.contour-<n>-<what>.png` (the scan running, its first and last step, one component, and 1280 px), then the deformed-shape stills `<out>.deform-<n>-<what>.png` (scale 1 and 10 with the ghost, with a contour, two staged views, and 1280 px) |
| `npm run dist` | build the Windows installer into `out/installer/` (see [Install](#install-windows)) |

## Layout

| Path | Holds |
|---|---|
| `src/reader/` | `read.ts` turns one file into a `ModelFile` through h5wasm; `node.ts` opens it from disk; `geometry.ts` reads the `/geometry` sibling and pairs it by `session_id`; `provenance.ts` reads `/provenance` and finds a declaration's source line |
| `src/chain/` | `resolve.ts` builds the definition chain; `signatures.ts` is the only table of OpenSees syntax the app uses |
| `src/mesh/build.ts` | flat render buffers coloured by physical group (pure, tested in Node) |
| `src/state/store.ts` | the single state store and its reducer (ADR 0112 D6); `results.ts` pairs a results file with the model by hash, and `activation.ts` derives a results stage's active elements from `/sequence` |
| `src/main/` | the Electron main process (file reads, measurement, capture) and the preload bridge; `pairing.ts` (the set of a stem), `session.ts` (the open set), `watch.ts` (watch the set), `source.ts` (go to source), `zones.ts` (reads `/geometry`, `/provenance`, the model's `/meta/lineage` and its `/sequence` element tags for the renderer) and `results.ts` (the open results file: its index, then one step at a time) |
| `src/panels/windows/` | the model-definition windows: `registry.ts` (one view per `WindowKind`), `frame.ts` (the Windows menu and the open windows, driven by `openWindow` / `closeWindow`) and one pure `(state) → DOM` view per kind, `<kind>.ts` |
| `src/renderer/` | the three.js viewport, the panels and the page; `navigation.ts` is the Z-up turntable camera, and `bindings.ts` is the only table of mouse and key bindings (left-click selects, right-drag orbits about the point under the cursor, middle-drag or shift + right-drag pans, the wheel zooms to the cursor, `F` fits) |
| `scripts/` | `build.mjs` (esbuild bundles) and `launch.mjs` (starts Electron in a mode) |
| `fixtures/` | a small committed `model.h5` with beams; its README names the command that made it |
| `screenshots/` | stills for the maintainer to approve |

## How the app reads a model

- Every inspector value names its source as an HDF5 path, with `@attr` for an
  attribute and `[i]` for a row or slot.
- Values marked **interpreted** needed OpenSees command syntax to find: the
  file stores element and beamIntegration arguments positionally, with
  cross-references as bare tags. `src/chain/signatures.ts` holds that syntax
  for the types it knows. Any other type is reported as unresolved by name.
- Section → material links are read directly: a Fiber section's patches,
  fibers and layers name their material by HDF5 path (`material_ref`).
- Schema versions follow ADR 0113, with one floor per zone (`ZONE_FLOOR` in
  `src/reader/read.ts`: neutral 2.10, opensees 2.12, geometry 1.0, provenance
  1.0, sequence 1.0), equal to Python's `reader_floor(zone)`
  (`tests/test_app_floor_drift.py`). A file from the floor to the app's version
  opens with no banner. A newer minor of the same major opens with one banner.
  A file below the floor, or of another major, is refused with a sentence
  naming the floor. The root `/sequence` zone (1.0, V3a) is read when a file
  carries it and is otherwise "not in this file", never an error.
- The app's version of each zone (`NEUTRAL_TARGET`, `OPENSEES_TARGET`,
  `GEOMETRY_TARGET`, `PROVENANCE_TARGET`, `SEQUENCE_TARGET` in
  `src/reader/read.ts`; `RESULTS_TARGET` in `src/reader/results.ts`) equals
  main's Python writer version for that zone, major and minor
  (`tests/test_app_target_drift.py`, which fails when a target is behind or
  ahead). So a PR that bumps a writer's schema version also bumps the app's
  target, in the same PR: read the zone's history entry for the new version in
  `architecture/h5-schema.md` first, then either read the new data or say,
  above the target constant, why the app does not need it. Never bump blind.
  The `/assembly` zone is the one Python zone the app does not read; the test
  lists it by name with the reason.
- The model-definition data: `/opensees/decls` and `/opensees/decl_params`
  (each declaration's parameters by field name, references as
  `{"$decl": key}`), the analysis chain, time series, patterns, recorders, fix
  records and `/opensees/stages/stage_NNN`, all into the one store. A
  malformed `decl_params` row (an unknown `$` tag, a reference to an undeclared
  key, a row out of range) refuses the file with its path, as Python's reader
  does.
- Colour: each element takes the colour of the smallest element-side physical
  group that contains it. Elements in no group are grey; OpenSees elements with
  no mesh cell (embedded rebar trusses) are drawn from their inline
  connectivity in near-white.
