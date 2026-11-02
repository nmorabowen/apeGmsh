# Parity of the VTK stills with `apegmsh-view/1` (ADR 0128 slice 6)

This page maps every still the ADR 0094 VTK renderer can write today to an
`apegmsh-view/1` spec that `apeGmshViewer --render` draws, or marks it a gap,
so the owner can rule on the deletion in VA6
([#2655](https://github.com/nmorabowen/apeGmsh/issues/2655)). Slice card:
VA6a ([#2741](https://github.com/nmorabowen/apeGmsh/issues/2741)). Status:
**awaiting the owner's sign-off**. VA6 deletes nothing until the owner signs.

## Gaps

Nine gaps stand between the VTK stills and `apegmsh-view/1`. Each row below
is backed by a probe from the "Evidence" section. G1 and G2 are the two gaps
the card asked to confirm, and both are confirmed.

| # | Gap | What v1 does | What v2 lacks | Rows |
|---|---|---|---|---|
| G1 | **Reaction arrows** | `view="reactions"` draws a reaction glyph at each support, every axis (session `Reactions()` slot); the pack writes `reactions.png` for a static stage that recorded reactions | No spec key draws reactions (`reactions` and `view` are refused as keys). The nearest view is a nodal contour of `reaction_force`, a different picture (the e2e test `test/spec/render.e2e.ts` says so) | R11, R26 |
| G2 | **Headless stills on Linux** (and macOS) | VTK offscreen renders on Linux/Mesa. CI's image goldens (`tests/viewers/test_x2_golden_stills.py`) are Linux frames | No PNG under xvfb on ubuntu: WebGL2 is blocklisted under software GL (VA0p, #2649, probe PR #2656). macOS was not measured. There is no installer outside Windows, and `agent_view.find_viewer` finds no install there | P2, P3 |
| G3 | **Deform auto-scale** | `deform=None` on `view="deformed"` scales the largest \|u\| at the drawn step to 0.12 of the model diagonal (`_results_stills` → session `Deform(scale=None)`) | `deform.scale` is a number. Left out, it is 1, not a fit, and `"auto"` is refused | R10, R18, R25 |
| G4 | **Deform by velocity or acceleration** | `deform=(field, scale)` with `field` in displacement, velocity, acceleration | `deform` has only `scale` and `ghost` (`deform.field` is refused). The app warps by displacement only (`apeGmshViewer/src/state/deform.ts`) | R17 |
| G5 | **The canned pack** | `render_pack` chooses the stage, the default component (uz, ux, uy, then the first nodal), whether to add reactions, and the file set | No pack. Each choice has to be made by a caller that reads `--describe` and writes the shots | R23–R28 |
| G6 | **History plot** | `history.png`, a labelled matplotlib history at the extrema node | v2 draws no 2-D chart (`history` is refused as a key). This still was never VTK: it can stay as `results.plot.history` | R27 |
| G7 | **Live-session stills** | `g.model.render` draws the BRep, and `g.mesh.render` draws the live mesh, both from the open Gmsh kernel | The app reads files only (ADR 0112 D4). A spec cannot name a geometry file (`files.geometry` is refused). A live mesh is drawable only after it is saved to `model.h5` | R29, R30 |
| G8 | **Inputs the app cannot open** | Any `FEMData` or `Results` in memory, including `Results.from_mpco` / `from_recorders`, and a native results file under any name | `files` takes only paths on disk: `<stem>.h5` with a `<stem>.results.h5`, `<stem>.ladruno` or part-set sibling. An `.mpco` is refused before the run starts, with exit 2 | R1, R31 |
| G9 | **The skip contract** | `None` or `()` with a `[skip viewer]` line, under `APEGMSH_SKIP_VIEWER=1` or without GL | No skip. `agent_view.render_stills` raises `FileNotFoundError` when no app is installed, and `AgentViewError` when the app refuses. This is the fail-closed design of ADR 0128 D2, but the callers below rely on `None` | R21 |

Two facts outside the table bear on the deletion ruling:

- **`results.render` and `render_pack` no longer run through `viewers/render.py`.**
  `Results.render` calls `viewers/session/_results_stills.py`, which draws
  through `ResultsSession.render` (`viewers/session/_stills.py`). In
  `render.py`, `render_results` and `render_pack` are reached only from
  `tests/viewers/test_render.py`. The live callers of `render.py` are
  `fem.render`, `g.model.render`, `g.mesh.render` and the CLI.
- **The session path imports both files.** `session/_stills.py`,
  `session/_animation.py` and `session/_results_stills.py` import
  `_resolve_camera`, `_require_view`, `_require_deform` and `_CAMERAS` from
  `render.py`, and the plotter, theme, camera and screenshot helpers from
  `_stills.py`. Deleting either file breaks `ResultsSession.render` and
  `export_animation`. Neither of those is an ADR 0094 still. The deletion
  therefore has to move those helpers, or narrow to the still verbs.

The other v1 callers that VA6 must rewrite or retire are
`assess(figures=True)` (`assess/_results.py` → `render_pack`,
`assess/_fem.py` → `fem.render`), `python -m apeGmsh.viewers render`
(`viewers/__main__.py`) and the studio `render` verb (`studio/_mcp.py`).

## The table

Each v1 row was read from the code: `python scripts/nav.py map` on
`viewers/render.py`, `viewers/_stills.py` and
`viewers/session/_results_stills.py`, plus the public signatures in
`FEMData.render`, `Results.render`, `Results.render_pack`, `Model.render`,
`Mesh.render` and the CLI `_main_render`. "Covered" means that a spec gives
the same view, though not the same pixels. ADR 0128 Amendment 2 makes the
camera and the view the contract, and the look comes from the app's own
theme (D8).

The evidence names a probe from the next section. `A:<shot>` and `B:<shot>`
are shots of a real `--render` run (`ok` wrote a PNG; `refused` gives the
app's reason). `V:<case>` is an offline run of the validator
(`src/spec/spec.ts` `validateSpec`). A source path is code that refuses or
decides the case.

### `fem.render(path, *, camera=None, window_size=(1280, 720))`

| # | v1 call and option | v2 spec | Evidence |
|---|---|---|---|
| R1 | `fem.render(path)`: the undeformed mesh of a `FEMData` | `files.model` names a saved `model.h5`; one shot with no `stage`. Covered, **once the FEMData is on disk** (`fem.to_h5`): G8 for one held in memory only | B:`fem-mesh` ok; B:`fem-refuse-stage` refused, "the set has no results file, so it has no stage 0" |
| R2 | the mesh look: a grey substrate with edges (`_stills._add_substrate`) | `colour_by` is `group` (default) or `role`. Nothing draws a single uniform colour. This is a difference in look, not a gap | `spec.ts` `shot()`, closed set `group, role, contour` |
| R3 | `camera=None`: `xy` for a planar model, else `iso` (`render._resolve_camera`, ADR 0094 Amendment 3) | No planar rule: `camera` defaults to `iso`. A spec for a planar model sets `"camera": "top"` | V:`camera omitted` → `"iso"` |
| R4 | `camera="iso"` (VTK: looks along (−1, −1, −1), up +z) | v2 `iso` looks along (−1, 1.3, −0.9) (`navigation.ts` `ISO_FORWARD`), a different side of the model. For the VTK direction, use an explicit `camera: {eye: c + d(1,1,1), target: c, up: [0,0,1]}`, with `c` the centre of `--describe`'s `bbox` | A:`mesh-iso-vtk` ok (the app keeps the eye and target and makes up roll-free); A:`mesh-iso-app` ok |
| R5 | `camera="xy"` (looks along −z, up +y) | `"camera": "top"`: the same heading (`navigation.ts`: "top looks down -Z with +Y up") | A:`mesh-top` ok, up (0, 1, 0); V:`camera xy` refused |
| R6 | `camera="xz"` (looks along +y, up +z) | `"camera": "-y"` (the front face) | A:`mesh-front` ok, eye on −y, up +z |
| R7 | `camera="yz"` (looks along −x, up +z) | `"camera": "+x"` (the right face) | A:`mesh-right` ok, eye on +x, up +z |
| R8 | `window_size=(w, h)`, any size; default 1280 × 720 | `"size": [w, h]`, whole numbers from 64 to 8192; default 1600 × 1000. Covered for any real still | V:`size 1280x720` accepted; V:`size 32x32` refused |

### `results.render(path, *, view="contour", component=None, step=-1, deform=None, camera=None, window_size=(1280, 720))`

`camera` and `window_size` map as R3–R8 do.

| # | v1 call and option | v2 spec | Evidence |
|---|---|---|---|
| R9 | `view="mesh"`: the undeformed mesh of the bound model (`deform=` is ignored) | A shot with no `stage` | A:`mesh-top` ok |
| R10 | `view="deformed"`: a contour of `component` on the shape warped by `deform` | `stage` + `contour` + `deform: {scale: s}`. Covered with an explicit scale. **G3** without one | A:`contour-deform`, A:`deformed-ghost` ok |
| R11 | `view="reactions"` (`component` None or `"reactions"`): reaction arrows at the supports | **G1**. The nearest view is `contour: {field: "reaction_force", component: "magnitude"}`, a contour, not arrows | V:`reactions key` and V:`view key` refused; `test/spec/compile.test.ts` "the four ADR 0094 views" |
| R12 | `view="reactions"`, `component="reaction_x"` etc. | Not applicable: v1 already refuses a one-axis reaction still (`_results_stills._refuse_unsupported`) | code |
| R13 | `view="contour"`, `component="displacement_z"` | `contour: {field: "displacement", component: "displacement_z"}`. The component may also be the legend's label (`"z"`) or `"magnitude"` | A:`contour-last`, A:`contour-step1` ok |
| R14 | `component=None`: the starter heuristic (uz, ux, uy, then the first nodal component) | No default: `contour.field` and `contour.component` are required. The caller picks one from `--describe` (G5) | `spec.ts` `contour()`, `text()` |
| R15 | a component the stage lacks: `ContourDataError` | The shot is refused, the reason listed, and no PNG written | A:`refuse-component`, A:`refuse-field` refused |
| R16 | the contour's range: auto-fit to the step drawn; the colormap from the palette | `contour.range: "step"` (the default is `"history"`); `contour.cmap` from a closed set, default `jet` (D8) | `spec.ts` `range()`; manifest `contour.range` |
| R17 | `deform=("velocity" \| "acceleration", s)` | **G4** | V:`deform field key` refused |
| R18 | `deform=None` on `view="deformed"`: auto-fit to 0.12 of the diagonal | **G3**: a left-out scale is 1. A workaround takes two runs: read `contour.extrema` from a first manifest and `bbox` from `--describe`, then set `scale = 0.12 · diag / max\|u\|`. That uses the history maximum where v1 uses the step's | V:`deform scale auto` refused; V:`deform {}` → `scale: 1` |
| R19 | `deform=s` or `deform=("displacement", s)` on `contour` or `reactions` | `deform: {scale: s}` with the contour. The `reactions` case falls under G1 | A:`contour-deform` ok |
| R20 | `step=k ≥ 0`, `step=-1`, `step=-k` | `step: k`, `"last"` (default), or `n − k`, with `n` from `--describe` `stages[i].steps`. Out of range: v1 raises `ValueError`, and v2 refuses the shot | A:`refuse-step` refused ("stage "unprop" has 4 steps (0 to 3)"); V:`step -2` refused |
| R21 | the return value: the `Path`, or `None` under `APEGMSH_SKIP_VIEWER` or without GL | `agent_view.render_stills(spec, out)` returns the manifest. PNGs are `<name>.png` in an empty `out` folder. A refusal raises (**G9**) | `agent_view.py`; `main.ts` exit codes 0, 1, 2 |
| R22 | the stage: the `results.stage(s)` scope, else the last stage, with a warning (`_resolve_stage`) | `stage`, by name or index, is required for any results shot. A scoped results object has no counterpart; the spec names the stage | A:`refuse-stage` refused ("the results have 3 stages") |

### `results.render_pack(out_dir, *, camera=None, window_size=(1280, 720))`

The pack is five calls of `results.render` (`_results_stills.render_results_pack`),
so each still maps through the rows above. What v2 lacks is the pack's own
logic (G5).

| # | Pack still | v2 spec | Evidence |
|---|---|---|---|
| R23 | `mesh.png` | a shot with no `stage` (R9) | A:`mesh-top` ok |
| R24 | `contour.png`: the default component at the last step | a `contour` shot (R13). The component is picked by the caller (R14, G5) | A:`contour-last` ok |
| R25 | `deformed.png`: the same, auto-scaled | R10 with an explicit scale (**G3**) | A:`deformed-ghost` ok |
| R26 | `reactions.png`: written only for a `static` stage that recorded `reaction_*` | **G1** | as R11 |
| R27 | `history.png`: matplotlib at the extrema node | **G6**. This still never came from VTK | V:`history key` refused |
| R28 | the pack's choices (stage, component, whether to add reactions) | **G5**: none in v2 | none |

### Live-session stills (`Model.render`, `Mesh.render`)

| # | v1 call | v2 spec | Evidence |
|---|---|---|---|
| R29 | `g.model.render(path, camera="iso", window_size)`: the BRep of the open kernel | **G7**. The spec has no geometry key, and the app opens no kernel | V:`files geometry key` refused |
| R30 | `g.mesh.render(path, camera="iso", window_size)`: the live mesh | Covered only after `g.save()` or `fem.to_h5`, through R1. There is no live path (G7) | as R1 |

### The command line

| # | v1 call | v2 spec | Evidence |
|---|---|---|---|
| R31 | `python -m apeGmsh.viewers render PATH [OUT] --view --component --step --deform --camera --window --pack --model-h5 --json` | `apeGmshViewer --render spec.json --out DIR`. The flags become R9–R22, `--pack` becomes G5, and `--json` becomes `manifest.json`. `--model-h5` with an `.mpco` falls under G8. The CLI's `--camera` defaults to `iso`, with no planar rule, so R3 does not apply there | the `.mpco` probe: exit 2, "apeGmshViewer opens <stem>.h5, <stem>.geometry.h5, <stem>.ladruno, <stem>.results.h5 or <stem>.user.h5" |

### Platforms

| # | Platform | v1 (VTK offscreen) | v2 (`apeGmshViewer --render`) |
|---|---|---|---|
| P1 | Windows desktop | yes | yes: probes A and B on this machine; the installer is per-user NSIS |
| P2 | Linux, headless (CI, SSH) | yes: Mesa offscreen; CI's goldens are Linux frames | **no** (G2): VA0p, #2649 / PR #2656. The sandbox can be fixed; WebGL2 under software GL needs Chromium switches set in the app |
| P3 | macOS | where PyVista offscreen runs | **unmeasured** (G2): no installer target in `electron-builder.yml`, and `find_viewer` refuses outside Windows unless `$APEGMSH_VIEWER` is set |

VA0p already settles the deletion: while P2 is "no", VA6 can only narrow the
renderer, not delete it. That is ADR 0128's open question on Linux and CI,
answered.

## Evidence

The probes ran on 2026-11-02 on Windows 11, against this branch's base
(`a9609e869`), with the viewer built in the worktree (`node scripts/build.mjs`)
and run through `node scripts/launch.mjs render|describe`.

`--describe fixtures/contour_staged.results.h5` reported the set as
`paired`, with three static stages (`base` 3 steps, `erect` 2, `unprop` 4).
Each stage has one nodal field, `displacement`, with components
`magnitude, displacement_x, displacement_y, displacement_z`. The groups are
`Col`, `Prop` and `Top`, and the bbox runs from (0, 0, 0) to (2, 1, 3.5).

**Probe A.** `files.model` is `fixtures/contour_staged.h5`, with
`"size": [1280, 720]` and `"chrome": "model"`. The run exits 1: nine stills
are written and four shots refused.

```json
[
  { "name": "mesh-top", "camera": "top" },
  { "name": "mesh-front", "camera": "-y" },
  { "name": "mesh-right", "camera": "+x" },
  { "name": "mesh-iso-app", "camera": "iso" },
  { "name": "mesh-iso-vtk", "camera": { "eye": [11, 10.5, 11.75], "target": [1, 0.5, 1.75], "up": [0, 0, 1] } },
  { "name": "contour-last", "stage": 2, "step": "last", "camera": "iso", "contour": { "field": "displacement", "component": "displacement_z" } },
  { "name": "contour-step1", "stage": "unprop", "step": 1, "camera": "iso", "contour": { "field": "displacement", "component": "z" } },
  { "name": "contour-deform", "stage": 2, "camera": "iso", "contour": { "field": "displacement", "component": "displacement_z" }, "deform": { "scale": 10 } },
  { "name": "deformed-ghost", "stage": 2, "camera": "iso", "contour": { "field": "displacement", "component": "magnitude" }, "deform": { "scale": 10, "ghost": true } },
  { "name": "refuse-stage", "stage": 5, "camera": "iso" },
  { "name": "refuse-step", "stage": 2, "step": 9, "camera": "iso" },
  { "name": "refuse-component", "stage": 2, "contour": { "field": "displacement", "component": "displacement_w" } },
  { "name": "refuse-field", "stage": 2, "contour": { "field": "velocity", "component": "velocity_z" } }
]
```

The manifest gives each camera as resolved. `mesh-top` has its eye on +z
with up (0, 1, 0); `mesh-front` has its eye on −y with up +z; `mesh-right`
has its eye on +x with up +z. `mesh-iso-vtk` keeps eye (11, 10.5, 11.75) and
target (1, 0.5, 1.75), with up made roll-free. `deformed-ghost` records scale
10 with 216 of 216 nodes drawn, the range mode `history`, and its extrema:
0 at node 2 and 0.11408 at node 21. The VTK side's camera directions come
from PyVista 0.46.4: `view_isometric` looks along (−0.577, −0.577, −0.577),
`view_xy` along (0, 0, −1) with up y, `view_xz` along (0, 1, 0) with up z,
and `view_yz` along (−1, 0, 0) with up z.

**Probe B.** `files.model` is `fixtures/shoebuckle.h5`, a model with no
results. `fem-mesh` is written, and `fem-refuse-stage` (`"stage": 0`) is
refused because the set has no results file.

**The `.mpco` probe.** `files: {model: "m.h5", results: "m.mpco"}`, where
`m.mpco` is a copy of `contour_staged.results.h5`. The run exits 2 with
"render failed to start", and writes no manifest. This also shows that an
unopenable results file is refused before a manifest exists. Other
spec-level refusals write one, so the owner may want this run to as well
(INV-6).

**V: the validator, offline.** A Node script imports
`apeGmshViewer/src/spec/spec.ts` and runs `validateSpec` on single-shot
specs. Each case gives the key the validator refused, or what it accepted:

| Case | Result |
|---|---|
| `camera xy` | refused, `shots[0].camera`: must be one of iso, +x, -x, +y, -y, top, bottom, fit |
| `step -2` | refused, `shots[0].step`: "first", "last" or an integer from 0 |
| `step without stage` | refused, `shots[0].step`: needs a stage |
| `deform field key` | refused, `shots[0].deform.field`: not a key; the keys are scale, ghost |
| `deform scale auto` | refused, `shots[0].deform.scale`: must be a finite number |
| `deform {}` | accepted as `{scale: 1, ghost: false}` |
| `reactions key`, `view key`, `history key`, `colour literal` | refused as keys that are not part of the spec |
| `files geometry key` | refused, `files.geometry`: the keys are model, results |
| `size 1280x720` / `size 32x32` | accepted / refused, `size[0]`: from 64 to 8192 |
| `camera omitted`, `theme omitted` | accepted as `iso` and `light` |

## Next

The owner signs off this table or amends it on #2655. That ruling decides
what VA6 deletes, given G1–G9 and the two facts under the gap list: VA6 can
retire the VTK renderer only on Windows, and only where the gaps are closed
or ruled acceptable.
