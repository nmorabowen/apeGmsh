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

Build the installer from this directory, then run it:

```powershell
npm ci
npm run dist        # writes out/installer/apeGmshViewer Setup <version>.exe
```

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

## Commands

Run from this directory with Node 22.18 or newer (TypeScript runs natively
through Node's type stripping; nothing is compiled for the tests).

| Command | Does |
|---|---|
| `npm ci` | install the exact pinned dependencies from `package-lock.json` |
| `npm run typecheck` | `tsc` over `src/` and `test/` |
| `npm test` | reader, chain resolution, mesh and navigation tests, on the fixture and on synthetic files |
| `npm start -- <model.h5>` | open the app on a file; without a file, drop one on the window |
| `npm run measure -- <model.h5> [--uncapped]` | print one [MEASUREMENTS.md](MEASUREMENTS.md) row; `--uncapped` lifts the vsync cap and prints the throughput row |
| `npm run capture -- <model.h5> <out.png> [--pick=<OpenSees type>]` | write a still from a hidden window, with a beam selected (or an element of the `--pick` type); a second still `<out>.chain-end.png` when the chain is taller than the window |
| `npm run dist` | build the Windows installer into `out/installer/` (see [Install](#install-windows)) |

## Layout

| Path | Holds |
|---|---|
| `src/reader/` | `read.ts` turns one file into a `ModelFile` through h5wasm; `node.ts` opens it from disk; `geometry.ts` reads the `/geometry` sibling and pairs it by `session_id`; `provenance.ts` reads `/provenance` and finds a declaration's source line |
| `src/chain/` | `resolve.ts` builds the definition chain; `signatures.ts` is the only table of OpenSees syntax the app uses |
| `src/mesh/build.ts` | flat render buffers coloured by physical group (pure, tested in Node) |
| `src/state/store.ts` | the single state store and its reducer (ADR 0112 D6) |
| `src/main/` | the Electron main process (file reads, measurement, capture) and the preload bridge; `pairing.ts` (the set of a stem), `session.ts` (the open set), `watch.ts` (watch the set) and `source.ts` (go to source) |
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
- A schema version outside the reader window (ADR 0023) is shown as a warning
  banner; a different major version, or a neutral layout before 2.10, is refused.
- Colour: each element takes the colour of the smallest element-side physical
  group that contains it. Elements in no group are grey; OpenSees elements with
  no mesh cell (embedded rebar trusses) are drawn from their inline
  connectivity in near-white.
