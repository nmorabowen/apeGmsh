# ADR 0128 — Agent views in apeGmshViewer: a view spec, headless stills with a manifest, and offers the human applies

**Status:** Accepted (2026-11-01; proposed and accepted the same day by the owner)

**Owner:** nmora

**Plan:** [plan_viewer_agent_stills.md](../../internal_docs/plan_viewer_agent_stills.md)

**Relation to earlier ADRs:**

- [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) asks for this
  ADR. D8 keeps the ADR 0094 stills "until the app can produce the same
  stills headlessly" and moves ADR 0095's `highlight`, `get_selection` and
  stills to apeGmshViewer "in its own ADR, due before P4". Its open
  questions list "agent stills after P4". This ADR answers both. D4 (read
  only, no process but its own Electron binary) and D6 (one store, one-way
  flow) bind every decision below.
- [ADR 0094](0094-agent-assess-and-viewer-render.md) Part 2 is **amended**:
  the stills renderer it defines (`fem.render`, `results.render`,
  `render_pack`, VTK offscreen) is retired by slice 6 here, once parity is
  signed off. Its closed view set (INV-10) is the model for D1.
- [ADR 0095](0095-apegmsh-studio.md) is **amended**: its `render` verb
  becomes D4 (batch stills), its `highlight` becomes an offer (D5), and its
  `get_selection` envelope becomes a snapshot (D6).
- [ADR 0116](0116-render-technologies.md) is unchanged: VTK leaves the
  stills when slice 6 lands, as it already says.

## Context

The viewer is the human's review tool: a person opens a `model.h5` set and
reads the model and its results. Agents need images too, for two jobs:

1. **Assess.** After a run, an agent looks at the deformed shape, a
   contour or a cut, and judges whether the result is plausible.
2. **Present.** An agent shows a human a finding ("the peak drift is at
   this column line"), or puts a figure in a report.

Today agents get stills only from the ADR 0094 renderer: a separate VTK
pipeline in `src/apeGmsh/viewers/render.py`, with its own scene code and
its own look. Its pictures do not match what the human sees in
apeGmshViewer, and ADR 0112 D8 has already scheduled its deletion.

The app can already take stills without showing a window. `--mode=capture`
opens a hidden window and writes PNGs through `webContents.capturePage()`
(`capture:still`, `capture:resize` in `apeGmshViewer/src/main/main.ts`).
But the shots are a fixed slice-acceptance sequence, hardcoded in
`apeGmshViewer/src/renderer/app.ts`. An agent cannot ask for a view.

Four facts about the app shape the design:

- **The store describes almost the whole view.** The `Event` union
  (`apeGmshViewer/src/state/types.ts`) covers the phase and step, contour
  field, range and colormap, the deformed shape and its scale, the
  reference ghost, section planes, isolate and hide, colour-by, selection
  and line diagrams.
- **Two parts of the view are outside the store.** The camera lives in
  `apeGmshViewer/src/renderer/navigation.ts`. The theme (`ThemeMode`,
  dark or light) is set on the renderer (`viewport.setTheme`). Neither can
  be reproduced from state today.
- **A second open reaches the running window.** In `view` mode the app
  holds the single-instance lock, and the `second-instance` handler hands
  the new argv to the window already open.
- **The office style is already in the app.** `apeGmshViewer/src/theme/tokens.ts`
  is the one module for colours and fonts. It copies the office palette
  from apeGraphStyle (protanopia-safe, Archivo Narrow), and its `LIGHT`
  theme is the white-paper theme "for engineering report screenshots".

Pixels alone are a weak basis for an agent. A contour without its range,
step and no-data count can mislead a reader, and an image cannot say that
the banner refused the deformed shape. The house rule is to fail closed,
not to show something plausible.

## Decision

### D1 — A view spec, versioned and closed

The contract between an agent and the app is a **view spec**: a JSON
document with a version tag (`"spec": "apegmsh-view/1"`). The app compiles
it into reducer events plus the camera and theme. The raw `Event` union is
never the external contract, so an internal rename cannot break agents.

The v1 keys are a closed set. Names are illustrative; slice 2 fixes them:

```json
{
  "spec": "apegmsh-view/1",
  "files": { "model": "frame.h5" },
  "size": [1600, 1000],
  "theme": "light",
  "shots": [
    {
      "name": "uz-last",
      "caption": "Vertical displacement, gravity stage, last step.",
      "stage": "gravity",
      "step": "last",
      "camera": "iso",
      "colour_by": "group",
      "contour": { "field": "displacement", "component": "z", "range": "history", "cmap": "jet" },
      "deform": { "scale": 10, "ghost": true },
      "clip": [{ "normal": "z", "offset": 3.0, "flip": false }],
      "isolate": ["/physical/Columns"]
    }
  ]
}
```

- `files` names the set the way an open does. The results sibling is found
  by ADR 0112's pairing rules, never named into a pairing it would not
  make.
- `camera` is a named view (`iso`, `+x`, `-x`, `+y`, `-y`, `top`,
  `bottom`, `fit`, `fit-selection`) or explicit `eye`, `target` and `up`.
- Fields and components use the names the file uses, exactly as the
  legend shows them (`DISPLACEMENT/Ux` for a `.ladruno`).
- Every key has one state it maps to, and every state the spec can set can
  be read back into a spec. That round trip is what D6 needs.

### D2 — The spec fails closed

- **The whole spec is validated before any shot is taken.** An unknown
  key, a wrong type or an unknown spec version refuses the spec, names the
  key and writes no PNG.
- **A shot refuses when the app would refuse the same view to a human.**
  This covers a stage or step out of range, a field or component the file
  does not have, a deformed shape the banner refuses, or a contour that
  never reaches the screen (the `until(contourShown)` checks of the
  capture code). The refusal is recorded in the manifest with the app's
  own reason. The other shots still run, and the exit code is non-zero.
- **There is no fallback.** A shot never draws the nearest stage, the step
  before or the surviving axes. A refused shot writes no PNG.

### D3 — Every still ships with a manifest

Each run writes `manifest.json` beside its PNGs, with one entry per shot:

- the shot as applied, and the camera as resolved (eye, target, up, fov)
- the files read: path, size, and model and results hashes
  (`/meta/lineage/model_hash`, `fem_hash`), plus the app version
- the stage, step and time, and the pairing state (paired, unbound, refused)
- for a contour: the field, the component, the range and its mode (history,
  step or fixed, with whether the history scan finished), the colormap, and
  the count of no-data nodes or cells
- the extrema of the field shown: value, node or element id and position
- for a deformed shape: the scale, and the count of nodes drawn undeformed
- the text of every banner and notice on screen
- `ok` or `refused`, with the reason

The manifest is the agent's evidence, and the PNG is the picture of it. An
agent reports a value from the manifest, never by reading pixels.

### D4 — Batch stills: `--render`

`apeGmshViewer --render spec.json --out <dir>` opens a hidden window, takes
every shot in one launch, writes the PNGs and the manifest into `<dir>`, and
exits. It writes nothing outside `<dir>` (ADR 0112 D4). A shot is taken
only after its step has been read and two frames have been painted, the
rule `captureResults` already follows. `--describe <file> --out <dir>`
writes what a set offers without drawing: its stages, steps and times,
fields and components, groups and bounding box. With it, an agent can
write a spec that will not refuse.

Batch mode does not take the single-instance lock, so it runs while the
human has the app open, and it never touches their window.

### D5 — Presenting is an offer; the human applies it

`apeGmshViewer --present spec.json` sends the spec to the running window
through the second-instance path. The window **never changes on its own**.
The spec arrives as an **offer**, listed with its name, its caption and
the time it arrived. The view changes only when the human clicks it.

- Applying an offer is the same compile as D4, so a human sees exactly the
  view the agent's still showed.
- An offer that names files outside the open set says it will open them.
  Clicking opens the set first, as a drop would.
- An offer that would refuse (D2) is listed as refused, with the reason,
  and cannot be applied.
- **Back** returns to the view from before the last applied offer.
- Offers are a list, newest first, capped at eight; a ninth drops the
  oldest. They last for the window session, like section planes, because
  the app has no session store yet.
- With no window running, `--present` exits non-zero and opens nothing.
  An agent that wants a window open asks the human to open one.

The owner ruled this on 2026-11-01: the viewer is the human's review tool,
so an agent may propose a view but never take one over.

### D6 — Snapshot: what the human is looking at

`apeGmshViewer --snapshot --out <dir>` asks the running window for its
current view. It writes the view as a spec (`view.json`, through the D1
round trip), a PNG at the window's size, and a manifest entry in the D3
form. The selection is included as the decl paths the inspector shows. A
snapshot never changes the window. It replaces ADR 0095's selection
envelope as the handle from the human to the agent: the human sets up a
view, and the agent reads it back exactly.

### D7 — The app never runs Python; Python may run the app

The CLI is the contract, so any agent can use it without apeGmsh. A thin
Python shim in apeGmsh (its name is fixed in slice 4) only finds the
installed executable (`$APEGMSH_VIEWER`, else the per-user install),
writes the spec, spawns the app, and returns the manifest as a dict. It
imports nothing from `apeGmshViewer/`, and the viewer imports nothing from
`src/` (the ADR 0112 D4 wall, crossed only by a process boundary and files).

### D8 — Style: English, the office voice, the office palette

- **Text.** All text written around a still is in English, in the
  ape-tono voice: impersonal, declarative, and complete, with a value
  given beside its reference. This covers an offer's name and caption, a
  shot's caption, and any summary an agent writes from a manifest. The
  app does not rewrite captions; the voice is the agent's duty, set in
  the skill (slice 4).
- **Colours and fonts.** These come only from `apeGmshViewer/src/theme/tokens.ts`
  (office palette, Archivo Narrow). A spec cannot pass a colour literal.
- **Theme.** `theme` is `light` or `dark`. Batch stills (D4) default to
  `light`, the white-paper theme made for report screenshots. A snapshot
  (D6) keeps the human's theme, and an offer (D5) leaves the human's theme
  alone unless the spec names one.
- **Colormap.** Contours default to jet, as the app does.

## Invariants

- **INV-1 (read only).** No mode writes outside its `--out` directory or
  edits an open file. An offer changes only the view.
- **INV-2 (one compile).** D4, D5 and D6 use one spec compiler. The still
  an agent took and the view a human applies come from the same events.
- **INV-3 (round trip).** For every valid spec, applying it and then
  taking a snapshot gives the same spec (camera within float tolerance).
- **INV-4 (fail closed).** No shot draws a view other than the one asked
  for. A refusal writes no PNG and records the app's reason.
- **INV-5 (no takeover).** No CLI call changes the view of a running
  window without a click from the human.
- **INV-6 (evidence).** Every PNG has a manifest entry, and every number
  an agent quotes from a still can be found in that entry.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| Raw reducer events as the external format | couples agents to internal names; every rename breaks them, and events such as `fileLoaded` are not the agent's to send |
| Keep the VTK renderer for agents, apeGmshViewer for humans | two looks and two scene pipelines; ADR 0112 D8 already rejects growing them side by side |
| The agent takes over the human's window (with undo) | owner ruling 2026-11-01 (D5): the viewer is the human's review tool |
| Images only, no manifest | an agent reading pixels misquotes ranges and misses refusals; INV-6 |
| A long-lived daemon with an MCP or HTTP API | a listening process outside the second-instance path the app already has; CLI plus files is enough for v1 and keeps ADR 0112 D4 simple. Revisit if launch cost blocks a real workflow |
| Agent-chosen colours or fonts in the spec | breaks the office style; D8 |

## Slices

| # | Slice | Needs | Done when |
|---|---|---|---|
| 0 | This ADR, Accepted | none | the owner accepts it |
| 1 | Camera and theme into state: named views and explicit eye, target and up, through a `setCamera` event; `setTheme` through the store | 0 | the same events give the same frame twice (pixel diff); the camera survives a re-read |
| 2 | The spec compiler and validator, `--render` and `--describe` | 1 | specs for the four ADR 0094 views render on the fixtures; each D2 refusal case refuses with its reason and writes no PNG |
| 3 | The D3 manifest | 2 | manifest values match the Python oracle (`Results`) on a fixture, extrema ids included |
| 4 | The Python shim and the skill paragraph (`skills/apegmsh/`, then `scripts/sync_skill.py`), including the D8 voice rule for captions | 3 | an agent goes from a model to PNGs plus a manifest in one call |
| 5 | `--present` (the offer list, Apply, Back) and `--snapshot` | 2 | INV-3 and INV-5 tests pass; the owner approves the offer list from a screenshot (ADR 0112 D7) |
| 6 | Parity with `render_pack`, then delete the ADR 0094 stills renderer | 3, 4 | parity checklist signed off by the owner |

## Acceptance

- A spec that asks for a contour at a step the file lacks refuses, names
  the step, and writes no PNG.
- A spec run twice on the same files gives byte-identical manifests,
  except for timestamps, and pixel-identical PNGs on one machine.
- With the human's window open, `--render` runs to completion, and the
  window's view and selection are unchanged afterwards.
- `--present` adds an offer and the view does not change. Clicking it
  gives the view in the agent's still, and `--snapshot` then returns the
  presented spec.
- The ADR 0094 views (`mesh`, `contour`, `deformed`, `reactions`) and the
  `render_pack` set are reproduced before slice 6 deletes anything.

## Open questions

- **Linux and CI.** Does `capturePage` on a hidden window work under xvfb,
  or with no display? Slice 2 measures this first. Until it works, batch
  stills are Windows-only, and the ADR 0094 renderer stays for Linux
  agents.
- **Annotations.** Markers and labels at node ids, a pin on the extremum,
  a dimension between two nodes. These are kept out of v1 so the spec
  ships small. They are a candidate for `apegmsh-view/2`.
- **Launch cost.** Electron start-up is paid once per batch. If agents
  need many single shots, a measurement decides whether the daemon
  alternative comes back.
- **Line diagrams and model-definition windows in the spec.** The store
  covers them, but v1 leaves them out until an agent workflow asks.

## Amendment 1 (2026-11-01): capture layers (D9)

**Status:** Accepted (2026-11-01) by the owner's merge of this amendment.
The owner asked for it the same day, on approving the VA1 stills.

The owner asked that an agent can capture only the model and its results:
the app's panels are turned off as layers, or left out of the picture.
D1 to D8 assumed a still is the window as a human sees it. That is right
for "show the human what I see", and wrong for a report figure, where the
panels are noise and cover the model (in the VA1 stills, the loads and
physical-group panels sit on the `loads3d` slab and column).

### D9 — A shot chooses its layers

A shot (or the spec, as the default for its shots) takes two new keys:

```json
{ "chrome": "model", "layers": { "scale_bar": true, "axes": true, "banner": false } }
```

- **`chrome: "app"`** (the default) is the window as D1 to D8 define it. In
  this mode, `layers` can switch off single panels by name, from a closed
  set: `header`, `phase`, `viewcube`, `left_rail` (loads, line diagram,
  legend), `rail` (section planes, sources, inspector), `windows`,
  `banner`. A panel that is off is **omitted from the layout**, not drawn
  over: the viewport does not move or resize because of it.
- **`chrome: "model"`** draws the viewport alone, at `size`, with every
  panel omitted. Only these still-only layers can be added, each drawn
  into the image by the render mode, never by the interactive panels:
  - `scale_bar`: the colour ramp of the contour or line diagram shown, with
    its field, component, min, max and range mode, in the D8 fonts and
    colours. **It is on by default whenever a contour or line diagram is
    drawn**, because a contour without its scale misleads. An agent turns
    it off explicitly, and the manifest records that it did.
  - `axes`: a small X/Y/Z triad in a corner, in place of the ViewCube.
    Default on.
  - `banner`: the text of every banner on screen, as one line under the
    model. Default off; the manifest always has the text (D3).
  - `caption`: the shot's `caption` as a line of text. Default off.
- **`background`** is `"theme"` (the default, the theme's viewport
  background) or `"transparent"` (PNG alpha, for figures placed on a
  page). No other value; colours stay in `tokens.ts` (D8).

**Rules.**
- A layer never changes what is drawn or refused. A shot that would refuse
  with the panels on refuses with them off, with the same reason (D2), and
  a banner that is off still goes into the manifest.
- The manifest entry records `chrome`, every layer's state and
  `background`, so INV-6 holds: the numbers a hidden panel would have
  shown are in the manifest.
- An offer (D5) never changes the human's panels: `chrome`, `layers` and
  `background` apply to stills only and are ignored when an offer is
  applied (the offer says so). A snapshot (D6) writes the window as it is.
  To get a clean picture of the human's view, an agent passes the
  snapshot's spec to `--render` with `chrome: "model"` (INV-2: one
  compiler).
- `--describe` lists the layer names the app accepts.

**Slice.** D9 belongs to VA2 (#2651), which owns the spec compiler and the
render mode. Its done-when adds:
- a `chrome: "model"` still is exactly `size`, and no panel element is in
  the layout (DOM check);
- a contour shot in model mode carries the scale bar unless the spec turns
  it off, and the manifest records the layer state;
- a refused shot refuses identically in both modes;
- the owner approves model-mode stills (ADR 0112 D7).

**Invariant.** **INV-7 (layers are presentation).** Two shots that differ
only in `chrome`, `layers` or `background` draw the same model from the same
camera pose, and have the same manifest entry apart from those keys.

## Amendment 2 (2026-11-01): what Apply promises (D5)

**Status:** Accepted (2026-11-01) by the owner's
[ruling on #2654](https://github.com/nmorabowen/apeGmsh/issues/2654#issuecomment-6569224049).

D5 says the human sees "exactly the view the agent's still showed". This
amendment states what "exactly" holds to. The camera and the view are the
contract: Apply keeps the spec's camera exactly and draws the same plan.
Pixel identity with a model-chrome still is not part of it.

The human's panels stay where they are, so the open right rail can shift where
the model sits on screen (about 245 px on the fixture). With the rail collapsed,
only the panels differ. Apply neither reframes around the rail nor collapses it.

## Amendment 3 (2026-11-02): the agent_view module path (D7)

**Status:** Accepted on the maintainer's merge of this amendment, by the
ruling of 2026-11-02 on [#2713](https://github.com/nmorabowen/apeGmsh/pull/2713).

D7 leaves the shim's name to slice 4 and reads as if it lived with the viewer
code. It landed at `src/apeGmsh/agent_view.py` (#2713), so its public name is
`apeGmsh.agent_view`, not a path under `apeGmsh.viewers`.

The reason is the `viewers` line-count ratchet (`tests/test_loc_ratchet.py`).
The maintainer ruled to move the module out of `viewers` rather than raise the
ratchet. The move changes nothing D7 asks of the shim: it still only finds the
installed executable, writes the spec, spawns the app and returns the manifest,
and it imports nothing from `apeGmshViewer/`.

D7's original wording stays as written and is read through this amendment:
where it implies a `viewers` location, the module path is `apeGmsh.agent_view`.

## Pointer (2026-11-10): ADR 0132

[ADR 0132](0132-workspace-file-and-variants-manifest.md) (Accepted 2026-11-03)
supersedes INV-1 in part, by narrowing it to the agent modes (`--render`,
`--describe`, `--snapshot`, `--present`). In those modes INV-1 holds as
written: no mode writes outside its `--out` directory or edits an open file.
The human's window is not an agent mode, and it writes `<stem>.workspace.h5`
when the human saves. ADR 0132 now governs `<stem>.workspace.h5` and the
variants manifest `<study>.variants.json`. The decision text above is
unchanged.

