# Outcomes for the VTK stills under `apegmsh-view/1` (ADR 0128 slice 6)

This page decides, for every still the ADR 0094 VTK renderer can write
today, whether VA6 ([#2655](https://github.com/nmorabowen/apeGmsh/issues/2655))
**repoints** it to an `apegmsh-view/1` spec, **adds** what v2 lacks,
**drops** it, or **keeps** the VTK path. The old renderer is not a target:
it was buggy, so where v1 and v2 disagree, v2's behaviour is not a defect.
Slice card: VA6a ([#2741](https://github.com/nmorabowen/apeGmsh/issues/2741)).

**Ruling.** On 2026-11-02 the owner ruled "Accept all" on the recommended
outcomes ([#2655 comment](https://github.com/nmorabowen/apeGmsh/issues/2655#issuecomment-6591785652)).
A gap row marked **ruled 2026-11-02** carries one of those outcomes. A
row marked **repoint (covered)** needs no ruling, because v2 already draws
that view. Two rows
are marked **needs ruling**: in each, the code check changed the
recommended outcome, so the ruling does not cover it (see "Needs ruling").

**Platform rule.** G2 keeps the VTK path on Linux. Every "repoint" below is
a repoint on Windows. On Linux the same call keeps drawing through VTK
until v2 can capture headless there.

## Gaps and outcomes

| # | Gap | What v1 does | What v2 lacks | Outcome | Rows |
|---|---|---|---|---|---|
| G1 | Reaction arrows | `view="reactions"` draws a reaction glyph at each support, every axis; the pack writes `reactions.png` for a static stage that recorded reactions | No spec key draws reactions. The nearest view is a nodal contour of `reaction_force` | **drop**, ruled 2026-11-02 | R11, R26 |
| G2 | Headless stills on Linux (and macOS) | VTK offscreen renders on Linux/Mesa. CI's image goldens (`tests/viewers/test_x2_golden_stills.py`) are Linux frames | No PNG under xvfb on ubuntu: WebGL2 is blocklisted under software GL (VA0p, #2649, probe PR #2656). macOS was not measured: no installer target, and `agent_view.find_viewer` finds none outside Windows | **keep** on Linux, ruled 2026-11-02 | P2, P3 |
| G3 | Deform auto-scale | `deform=None` on `view="deformed"`: the largest \|u\| component at the drawn step becomes 0.12 of the model diagonal (`session/_realize.py`, `_DEFORM_FRACTION`) | `deform.scale` is a number. Left out, it is 1, and `"auto"` is refused | **add** an auto scale to v2, ruled 2026-11-02 | R10, R18, R25 |
| G4 | Deform by velocity or acceleration | `deform=(field, scale)` with `field` in displacement, velocity, acceleration | `deform` has only `scale` and `ghost`. The app warps by displacement only (`apeGmshViewer/src/state/deform.ts`) | **drop**, ruled 2026-11-02 | R17 |
| G5 | The canned pack | `render_pack` chooses the stage, the default component (uz, ux, uy, then the first nodal) and the file set | No pack. Each choice has to be made by a caller that reads `--describe` | **repoint** as one v2 pack spec built for `assess`, ruled 2026-11-02 | R23–R28 |
| G6 | History plot | `history.png`, a labelled matplotlib history at the extrema node | No 2-D chart in v2. This still was never VTK | **keep** (matplotlib), ruled 2026-11-02 | R27 |
| G7 | Stills of what is not on disk | `g.model.render` draws the BRep; `g.mesh.render` draws the live mesh, both from the open Gmsh kernel; `fem.render` and `results.render` draw a `FEMData` or `Results` held in memory | The app reads files only. A job does open `<stem>.geometry.h5` (`job.ts:95`), but `compileShot` never sets the geometry phase: it sets `mesh` or `results` only (`compile.ts:113-130`). A saved file therefore yields the mesh, never the BRep | **repoint** through a temporary `model.h5` (and its results file), ruled 2026-11-02, for `g.mesh.render` (R30) and in-memory objects (R1). For `g.model.render`, **needs ruling** (R29) | R1, R29, R30 |
| G8 | Files the app cannot open | `Results.from_mpco` reads an `.mpco`, and a native results file opens under any name | `files` takes `<stem>.h5` with a `<stem>.results.h5`, `<stem>.ladruno` or part-set sibling, and nothing else. An `.mpco` is refused before the run starts (exit 2), and a results file the stem rule does not find is not paired. A conversion exists on `origin/main`: `apeGmsh.results.convert_mpco(src, dst, model_h5=...)` writes `.ladruno` (ADR 0122 D5). In-memory objects are G7, not G8 | **drop**, ruled 2026-11-02. The converter is the user's route for `.mpco` | R31 |
| G9 | The skip contract | `None` or `()` with a `[skip viewer]` line, under `APEGMSH_SKIP_VIEWER=1` or without GL | No skip. `agent_view.render_stills` raises `FileNotFoundError` with no app, and `AgentViewError` on a refusal | **take v2's raise**, ruled 2026-11-02. Callers that test for `None` change with VA6 | R21 |
| G10 | Derived-scalar contours | `component=` accepts a scalar computed from the recorded components, such as `von_mises_stress` (`session/_specs.resolve_contour_topology`, `available_derived`) | The app offers only recorded datasets. A derived name is refused as an unknown component or field (`compile.ts` `fieldOf`) | **add** derived scalars to v2, ruled 2026-11-02 | R16c |

Stored Gauss components are **not** a gap. The review expected one, but
v2 draws them through `contour.source: "element"` with a `gauss` mode
(R16b). That row's outcome is repoint, and it is listed under "Needs
ruling".

### Needs ruling

- **R16b, stored Gauss components → repoint.** The coordinator's pick was
  "add" for the Gauss and derived-scalar gap. The probe shows v2 already
  draws stored components: `gauss-cell` and `gauss-node` were drawn on
  `elem_solid`. Only derived scalars (G10) need adding.
- **R29, `g.model.render` (the BRep).** "Repoint through a temporary
  `model.h5`" draws the mesh, not the BRep, because no spec reaches the
  geometry phase (G7). The two choices are to add a geometry phase to the
  spec and compiler, or to drop BRep stills.

### Open

- **INV-6 for unopenable inputs.** An `.mpco` results file is refused at
  start-up, and no `manifest.json` is written. Other spec refusals write
  one. No outcome is proposed.

## Deletion scope

Because of the "keep" outcomes, VA6 cannot delete either file whole.

| File | Stays because |
|---|---|
| `src/apeGmsh/viewers/_stills.py` | The Qt session imports it (`session/_stills.py`, `session/_animation.py`, `session/_results_stills.py`): plotter, theme, camera, screenshot and skip. Linux also keeps the VTK path (G2) |
| `src/apeGmsh/viewers/render.py`: `render_fem`, `render_model`, `render_mesh`, `_resolve_camera`, `_require_camera`, `_require_view`, `_require_deform`, `_require_live_kernel`, `_require_live_mesh`, `_VIEWS`, `_CAMERAS`, `DeformArg` | The session modules import `_resolve_camera`, `_require_view`, `_require_deform`, `_CAMERAS` and `DeformArg`. The CLI imports `_VIEWS`, `_CAMERAS` and `_DEFAULT_WINDOW`. The three verbs keep the Linux path (G2) |
| `src/apeGmsh/viewers/session/_results_stills.py` | `results.render` and `render_pack` run through it (not through `render.py`), on Linux (G2). It also holds `_try_history` (G6, keep) |
| `src/apeGmsh/viewers/session/_stills.py`, `_animation.py` | `ResultsSession.render` and `export_animation`, the Qt session (kept: R32, R33) |

What VA6 may delete is the director path in `render.py`: `render_results`,
`render_pack`, `_ensure_stage`, `_resolve_step`, `_require_deform_field`,
`_apply_deform`, `_read_deform_field`, `_pack_primary_component`,
`_has_static_reactions`, `_extrema_node`, `_try_history`,
`_DEFORM_FRACTION` and `_HISTORY_LABEL`. No `src/` module outside
`render.py` calls them. The tests that reference them are
`tests/viewers/test_render.py`, and one string in
`tests/assess/test_import_guard.py`. The live versions of these helpers
are in `session/_results_stills.py`.

The director path also holds `render.py`'s only imports of the retired
`viewers.diagrams` ontology: lines 182-187 in `render_results`, and lines
528-529 in `_pack_primary_component`. `tests/viewers/test_diagrams_import_guard.py`
allowlists `apeGmsh.viewers.render` in `_ALLOWED` (line 47) and in
`test_allowlisted_modules_are_admitted` (line 212). That guard is ratcheted
both ways, so the deletion must remove both entries in the same PR.

VA6 also repoints the Windows callers:
`assess(figures=True)`, the CLI and the studio `render(path=)` door.

## The rows

Each v1 row was read from the code: `python scripts/nav.py map` on
`viewers/render.py`, `viewers/_stills.py` and
`viewers/session/_results_stills.py`, plus the public signatures in
`FEMData.render`, `Results.render`, `Results.render_pack`, `Model.render`,
`Mesh.render`, `ResultsSession.render`, the CLI `_main_render` and
`studio/_mcp.render`. In the Evidence column, `A:`, `B:` and `C:` name shots
of a real `--render` run, `V:` an offline validator case, and `K:` the v1
colour-range probe. All of them are under "Evidence".

### `fem.render(path, *, camera=None, window_size=(1280, 720))`

| # | v1 call and option | v2 spec | Outcome | Evidence |
|---|---|---|---|---|
| R1 | `fem.render(path)`: the undeformed mesh | `files.model` names a `model.h5` on disk; one shot with no `stage`. A `FEMData` held only in memory is first written to a temporary `model.h5` (G7) | repoint (covered); in memory: repoint through a temporary `model.h5` (G7), ruled 2026-11-02 | B:`fem-mesh` ok; B:`fem-refuse-stage` refused, "the set has no results file, so it has no stage 0" |
| R2 | the mesh look: a grey substrate with edges | `colour_by` is `group` (default) or `role` | repoint (covered; the look changes) | `spec.ts` `shot()` |
| R3 | `camera=None`: `xy` for a planar model, else `iso` | `camera` defaults to `iso`; a planar spec sets `"top"` | repoint (covered) | V:`camera omitted` → `"iso"` |
| R4 | `camera="iso"`: VTK looks along (−1, −1, −1), up +z | v2's `iso` looks along (−1, 1.3, −0.9) (`navigation.ts` `ISO_FORWARD`). The VTK direction is reached through an explicit `{eye: c + d(1,1,1), target: c, up: [0,0,1]}`, with `c` the centre of `--describe`'s bbox | repoint (covered) to v2's `iso` | A:`mesh-iso-app`, A:`mesh-iso-vtk` ok |
| R5 | `camera="xy"`: looks along −z, up +y | `"top"` | repoint (covered) | A:`mesh-top` ok; V:`camera xy` refused |
| R6 | `camera="xz"`: looks along +y, up +z | `"-y"` | repoint (covered) | A:`mesh-front` ok |
| R7 | `camera="yz"`: looks along −x, up +z | `"+x"` | repoint (covered) | A:`mesh-right` ok |
| R8 | `window_size=(w, h)`; default 1280 × 720 | `"size": [w, h]`, from 64 to 8192; default 1600 × 1000 | repoint (covered) | V:`size 1280x720` accepted; V:`size 32x32` refused |

### `results.render(path, *, view="contour", component=None, step=-1, deform=None, camera=None, window_size=(1280, 720))`

`camera` and `window_size` follow R3–R8.

| # | v1 call and option | v2 spec | Outcome | Evidence |
|---|---|---|---|---|
| R9 | `view="mesh"` (`deform=` is ignored) | a shot with no `stage` | repoint (covered) | A:`mesh-top` ok |
| R10 | `view="deformed"`: a contour on the warped shape | `stage` + `contour` + `deform: {scale}`; the auto scale is G3 | repoint, plus add (G3), ruled 2026-11-02 | A:`contour-deform`, A:`deformed-ghost` ok |
| R11 | `view="reactions"`: arrows at the supports | none (G1) | drop, ruled 2026-11-02 | V:`reactions key`, V:`view key` refused |
| R12 | `view="reactions"`, `component="reaction_x"` etc. | v1 already refuses it (`_refuse_unsupported`) | none needed: v1 never draws it | code |
| R13 | `component="displacement_z"` (a recorded nodal component) | `contour: {field: "displacement", component: "displacement_z"}`. The component may also be its label (`"z"`) or `"magnitude"` | repoint (covered) | A:`contour-last`, A:`contour-step1` ok |
| R14 | `component=None`: the starter heuristic | no default; the caller picks one from `--describe`. For the pack, G5 | repoint (covered) | `spec.ts` `contour()` |
| R15 | a component the stage lacks: `ContourDataError` | the shot is refused, with the reason; no PNG | repoint (covered) | A:`refuse-component`, A:`refuse-field` refused |
| R16 | the colour range: the field's **whole history** (`_contour._compute_initial_clim` takes the visual store's history range; it falls back to the step's range only with no store, or under a plane override). The colormap comes from the palette | `contour.range: "history"`, which is v2's default. `contour.cmap` defaults to `jet` (D8) | repoint (covered) | K: on `unprop`, `displacement_z`, steps 0, 1 and 3 are mapped over (−0.033, 0) each time; the step data is (−0.00825, 0), (−0.0165, 0), (−0.033, 0). A:`deformed-ghost` manifest `range.mode: "history"` |
| R16b | `component` a stored Gauss (element) component, e.g. `stress_xx` | `contour: {field: "stress", component: "stress_xx", source: "element", gauss: "cell" \| "cell_averaged" \| "node" \| "node_discrete"}`. The range is history, as in R16 | repoint, **needs ruling** | C:`gauss-cell` ok (`topology: element_mean`), C:`gauss-node` ok (`topology: gauss_node`) |
| R16c | `component` a derived scalar, e.g. `von_mises_stress` | none (G10) | add, ruled 2026-11-02 | C:`refuse-von-mises` and C:`refuse-von-mises-field` refused, "element field "stress" of stage "load" has no component "von_mises_stress"" |
| R17 | `deform=("velocity" \| "acceleration", s)` | none (G4) | drop, ruled 2026-11-02 | V:`deform field key` refused |
| R18 | `deform=None` on `view="deformed"`: auto-fit | none (G3); a scale left out is 1 | add, ruled 2026-11-02 | V:`deform scale auto` refused; V:`deform {}` → `scale: 1` |
| R19 | `deform=s` or `("displacement", s)` on `contour` | `deform: {scale: s}` with the contour | repoint (covered) | A:`contour-deform` ok |
| R20 | `step=k ≥ 0`, `-1`, `-k` | `k`, `"last"`, or `n − k` with `n` from `--describe` | repoint (covered) | A:`refuse-step` refused; V:`step -2` refused |
| R21 | returns the `Path`, or `None` on a skip | `agent_view.render_stills(spec, out)` returns the manifest and raises on a refusal (G9) | repoint, take v2's raise, ruled 2026-11-02 | `agent_view.py`; `main.ts` exit codes 0, 1, 2 |
| R22 | the stage: the `results.stage(s)` scope, or the `results.modes[i]` scope (a mode stage), else the last stage, with a warning (`_results_stills._resolve_stage`) | `stage`, by name or index, is required for every results shot. A mode is its mode stage (`--describe` `kind: "mode"`), named by index | repoint (covered) | A:`refuse-stage` refused |

### `results.render_pack(out_dir, *, camera=None, window_size=(1280, 720))`

| # | Pack still | v2 spec | Outcome | Evidence |
|---|---|---|---|---|
| R23 | `mesh.png` | a shot with no `stage` | repoint (covered), into the pack spec (G5, ruled 2026-11-02) | A:`mesh-top` ok |
| R24 | `contour.png`: the default component, last step | a `contour` shot | repoint (covered), into the pack spec (G5, ruled 2026-11-02) | A:`contour-last` ok |
| R25 | `deformed.png`, auto-scaled | R10, with G3's auto scale | repoint into the pack spec, plus add (G3), ruled 2026-11-02 | A:`deformed-ghost` ok |
| R26 | `reactions.png` | none (G1) | drop, ruled 2026-11-02 | as R11 |
| R27 | `history.png` (matplotlib) | none (G6) | keep, ruled 2026-11-02 | V:`history key` refused |
| R28 | the pack's choices | one pack spec for `assess`, built from `--describe` (G5) | repoint, ruled 2026-11-02 | none |

### Live-session stills

| # | v1 call | v2 spec | Outcome | Evidence |
|---|---|---|---|---|
| R29 | `g.model.render(path, camera="iso", window_size)`: the BRep | none. A temporary `model.h5` would draw the mesh, because `compileShot` never sets the geometry phase (G7) | **needs ruling**: add a geometry phase, or drop | V:`files geometry key` refused; `compile.ts:113-130` |
| R30 | `g.mesh.render(path, camera="iso", window_size)`: the live mesh | write a temporary `model.h5`, then R1 | repoint through a temporary `model.h5` (G7), ruled 2026-11-02 | as R1 |

### The Qt session

| # | v1 call | v2 spec | Outcome | Evidence |
|---|---|---|---|---|
| R32 | `ResultsSession.render(path, pane=None, *, camera=None, window_size)`: one pane of a session, drawn from the ADR 0098 slot catalog | none: the slot catalog is the Qt session's, not a view spec | keep, ruled 2026-11-02 (the Qt session callers) | `results/session/_session.py:262`; `session/_stills.render_still` |
| R33 | studio `render(session=...)`: one pane of a saved snapshot, out of process (`python -m apeGmsh.results.session render`) | none | keep, ruled 2026-11-02 (the Qt session callers) | `studio/_mcp.py:600`, `_render_session` |

### The command line and the studio door

| # | v1 call | v2 spec | Outcome | Evidence |
|---|---|---|---|---|
| R31 | `python -m apeGmsh.viewers render PATH [OUT] --view --component --step --deform --camera --window --pack --model-h5 --json` | `apeGmshViewer --render spec.json --out DIR`. The flags map as R9–R22, `--pack` as G5, `--json` as `manifest.json`; `.mpco` falls under G8. `--camera` defaults to `iso` here, with no planar rule | repoint (covered) | `.mpco` probe: exit 2, "apeGmshViewer opens <stem>.h5, <stem>.geometry.h5, <stem>.ladruno, <stem>.results.h5 or <stem>.user.h5" |
| R34 | studio `render(path=...)`: runs R31 out of process (`studio/_mcp.py:689`) | as R31 | repoint (covered) | as R31 |

### Platforms

| # | Platform | v1 (VTK) | v2 (`--render`) | Outcome |
|---|---|---|---|---|
| P1 | Windows desktop | yes | yes: probes A, B and C | repoint (covered) |
| P2 | Linux, headless | yes (Mesa) | **no** (G2) | keep, ruled 2026-11-02 |
| P3 | macOS | where PyVista offscreen runs | unmeasured (G2) | keep, ruled 2026-11-02 |

## Evidence

The probes ran on 2026-11-02 on Windows 11, against this branch's base
(`a9609e869`). The viewer was built in the worktree and run through
`node scripts/launch.mjs render|describe`. Python probes ran with
`PYTHONPATH=<worktree>/src`.

`--describe fixtures/contour_staged.results.h5` reported the set as
paired, with three static stages: `base` (3 steps), `erect` (2) and
`unprop` (4). Each stage has the nodal field `displacement`
(`magnitude, displacement_x, displacement_y, displacement_z`). The groups
are `Col`, `Prop` and `Top`, and the bbox runs from (0, 0, 0) to (2, 1, 3.5).
`--describe fixtures/elem_solid.results.h5` reported one stage, `load`
(3 steps), with the nodal field `displacement` and the element field
`stress` (`stress_xx, stress_yy`).

**Probe A** (`contour_staged.h5`, `"size": [1280, 720]`, `"chrome": "model"`)
exits 1: nine stills are written and four shots refused.

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

The resolved cameras in the manifest:

- `mesh-top` has its eye on +z, with up (0, 1, 0).
- `mesh-front` has its eye on −y; `mesh-right` has its eye on +x. Both
  have up +z.
- `mesh-iso-vtk` keeps its eye and target, and its up is made roll-free.

The VTK directions come from PyVista 0.46.4:

| PyVista call | Looks along | Up |
|---|---|---|
| `view_isometric` | (−0.577, −0.577, −0.577) | +z |
| `view_xy` | (0, 0, −1) | +y |
| `view_xz` | (0, 1, 0) | +z |
| `view_yz` | (−1, 0, 0) | +z |

**Probe B** (`shoebuckle.h5`, a model with no results). `fem-mesh` is
written, and `fem-refuse-stage` is refused.

**Probe C** (`elem_solid.h5`) exits 1: two stills are written and two shots
refused.

```json
[
  { "name": "gauss-cell", "stage": 0, "contour": { "field": "stress", "component": "stress_xx", "source": "element", "gauss": "cell" } },
  { "name": "gauss-node", "stage": 0, "contour": { "field": "stress", "component": "stress_xx", "source": "element", "gauss": "node" } },
  { "name": "refuse-von-mises", "stage": 0, "contour": { "field": "stress", "component": "von_mises_stress", "source": "element" } },
  { "name": "refuse-von-mises-field", "stage": 0, "contour": { "field": "von_mises_stress", "component": "von_mises_stress" } }
]
```

**K: v1's colour range.** The probe ran `results.stage("unprop").render(...,
view="contour", component="displacement_z", step=k)` with
`BasePlotter.screenshot` hooked, as `tests/viewers/test_x2_numeric_oracle.py`
does, and read the colour-mapped mapper's range at each step:

| Step | Mapped range | Step data |
|---|---|---|
| 0 | (−0.033, 0) | (−0.00825, 0) |
| 1 | (−0.033, 0) | (−0.0165, 0) |
| 3 | (−0.033, 0) | (−0.033, 0) |

**The `.mpco` probe.** `files: {model: "m.h5", results: "m.mpco"}` exits 2
with "render failed to start", and writes no manifest.

**V: the validator, offline.** A Node script ran
`apeGmshViewer/src/spec/spec.ts` `validateSpec` on single-shot specs:

| Case | Result |
|---|---|
| `camera xy` | refused, `shots[0].camera`: must be one of iso, +x, -x, +y, -y, top, bottom, fit |
| `step -2` | refused, `shots[0].step`: "first", "last" or an integer from 0 |
| `step without stage` | refused, `shots[0].step`: needs a stage |
| `deform field key` | refused, `shots[0].deform.field`: the keys are scale, ghost |
| `deform scale auto` | refused, `shots[0].deform.scale`: must be a finite number |
| `deform {}` | accepted as `{scale: 1, ghost: false}` |
| `reactions key`, `view key`, `history key`, `colour literal` | refused as keys outside the spec |
| `files geometry key` | refused, `files.geometry`: the keys are model, results |
| `size 1280x720` / `size 32x32` | accepted / refused, `size[0]`: from 64 to 8192 |
| `camera omitted`, `theme omitted` | accepted as `iso` and `light` |

## Next

The owner rules on the two "needs ruling" rows on #2655. VA6 then works
within the deletion scope above: it repoints the Windows callers, deletes
the director path in `render.py`, and leaves the VTK verbs and the Qt
session path in place.
