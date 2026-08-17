# ADR 0094 — Agent assess/report + offscreen viewer render

**Status:** Accepted (2026-08-13 — S0–S5 shipped; Q1–Q3 closed below)

**Evidence:** a scoping pass on the three Qt viewers plus a five-seat
workshop (architect / agent-UX / results-domain / simplicity skeptic /
engineering QA). The scoping documents
(`internal_docs/results_viewer_assessment.md`,
`results_viewer_adversarial_review.md`,
`results_viewer_gui_perf_assessment.md`) ship with this ADR — they
are also ADR 0084's cited evidence and were untracked. Constrained by ADR 0014 (viewer is an H5 consumer),
ADR 0042 (render-backend seam), ADR 0021 (lineage never raises),
ADR 0056 / 0084 (do not grow a second reconciler), ADR 0060 (sidecar
pattern), and `docs/design/principles.md` (chain ends at `FEMData`;
features that do not fit Geometry → Names → Mesh → Broker → Solver
are sidecars).

## Context

Coding agents now build apeGmsh models and run OpenSees. After
`g.end()` they typically hold `model.h5` plus a results file. What
they do *not* have is a pair of eyes on a Qt window.

The three Qt viewers (`g.model.viewer()`, `g.mesh.viewer()`,
`results.viewer()`) are the human review surface: picking, docks,
overlays, concurrent geometries, session restore. They are the wrong
agent surface:

- An **explicit** `results.viewer(blocking=True)` in a Jupyter
  kernel still native-crashes it. The **default is now auto**
  (`blocking=None` → `True` in scripts/CLI, `False` / `show_web`
  in a notebook). S0 exists because the skill and the published
  docs still teach the old “default `blocking=True`” line.
- `blocking=False` / `serve_web` opens a window nobody watches, or
  blocks until Ctrl-C.
- `fem.viewer()` is `NotImplementedError` (`FEMData.viewer`).
- `g.inspect` / SICN / unassigned-entity counts need a live Gmsh
  kernel and die on a chain-phase `from_h5` session.

Inventory already exists and is fragmented:
`g.inspect.print_summary`, `fem.inspect.*` (including
`find_coincident_node_pairs`), `results.inspect.summary` /
`components` / `diagnose`, `results.lineage`,
`g.model.io.diagnose() → ImportHealth`, `results.plot.*` (matplotlib),
`results.export_animation` (full `ResultsViewer.show(run_loop=False)`).
None of these emit a **verdict**. The skill still teaches `viewer` /
`show_web` as the visual check, so agents open Qt and then narrate.

Two still pipelines already produce Qt-family pixels:

| Path | Where | What it is |
|---|---|---|
| **A** | `Results.export_animation` | Full `ResultsViewer`, `run_loop=False`, focus-mode, screenshot. Pixel-identical chrome-free viewport. Window flashes. On Windows, `QT_QPA_PLATFORM=offscreen` **hard-crashes** (`ViewerWindow`). |
| **B** | `scripts/render_showcase/stills.py` | `pv.Plotter(off_screen=True)` + `build_fem_scene` + `ContourDiagram` + `ResultsDirector`. Same substrate / LUT / contour stack. No `QMainWindow`. Docs PNGs are made this way. Deform is still hand-rolled. |

ModelViewer / MeshViewer have no headless path: `show()` always
blocks; screenshot is a menu action.

The job is therefore two coupled products, not a fourth viewer:

1. A **finding compiler** an agent can act on.
2. A **render** API that writes stills from the *viewer’s scene and
   diagram pipeline*, so a report looks like the Qt viewer without
   opening one.

## Decision

### Part 1 — `apeGmsh.assess` is a sidecar; inspect stays inventory

A standalone `apeGmsh.assess` package (the sidecar rule is
`docs/design/principles.md`; the shape is ADR 0060's `hpc`: types +
pure runners, **not** a session composite, **not** re-exported from
`apeGmsh/__init__.py` — the root import chain eagerly pulls `gmsh`
(Qt itself is function-deferred), and assess must import without it).

Public doors are one-shot methods on the brokers an agent already
holds after a script:

```python
report = fem.assess()                 # FEMData, H5-safe
report = results.assess()             # Results, H5-safe
report = results.assess(figures=True) # + render_pack (Part 2)
```

No `g.assess` on `_COMPOSITES`. Live CAD health stays
`g.model.io.diagnose() → ImportHealth`. A chain-phase session must
not promise geometry checks it cannot run.

`inspect` / `diagnose` / `assess` stay three verbs:

| Verb | Job | Existing home |
|---|---|---|
| `inspect` | Inventory (tables, strings) | `fem.inspect`, `results.inspect`, `g.inspect` |
| `diagnose` | Routing / CAD health of *one* thing | `results.inspect.diagnose(component)`, `g.model.io.diagnose()` |
| `assess` | Verdict: findings + short markdown | **this ADR** |

Assess *calls* inspect, lineage, and
`find_coincident_node_pairs`. It does not grow those composites into
a junk drawer. Live CAD health stays `g.model.io.diagnose() →
ImportHealth` (S4) — v1 assess must not import `gmsh` (INV-1).

#### Return type

Frozen records, same flavour as `ImportHealth` and
`cuts._preflight.PreflightIssue` / `PreflightReport`:

```python
@dataclass(frozen=True)
class Finding:
    code: str                          # closed catalog, e.g. "RES.LINEAGE"
    severity: Literal["error", "warning", "info"]
    message: str                       # one line, no overclaim
    detail: Mapping[str, object] | None = None
    # detail carries where/evidence (pg, ids[:K], xyz, step, counts).
    # It does NOT carry Python snippets to exec.

@dataclass(frozen=True)
class AssessmentReport:
    findings: tuple[Finding, ...]
    text: str                          # 40–80 lines markdown
    figures: tuple[Path, ...] = ()     # PNG paths; empty if figures=False
    lineage: Lineage | None = None     # Results path; None on fem.assess()
```

`text` is what the agent prints. `findings` is what it branches on.
`tables` stay on `inspect` — the report may quote a *top-N* extrema
appendix (node id + xyz + step), never `node_table()` /
`get_geometry_info()` entity frames / all-step slabs.

`APEGMSH_SKIP_VIEWER=1` skips figures only; findings still run.

#### Closed finding catalog (v1)

Emit a finding only when the data exists. **Skip ≠ pass** — a
missing energy channel is not `OK`. Default extrema use `time=-1` or
`node_envelope`; never `nodes.get(..., time=None)` (loads every
step).

| Code | Severity | Source | Notes |
|---|---|---|---|
| `MODEL.EMPTY` | error | `fem.info` | 0 nodes or 0 elements |
| `MESH.INVERTED` | error | connectivity + coords | Zero / negative corner-node volume, **linear cell types only**. Quadratic / Bézier cells (ADR 0091 control values are not nodal coords) are listed as skipped, not judged. **Not** SICN. |
| `MESH.COINCIDENT_UNBRIDGED` | warning | `find_coincident_node_pairs` — pairs whose bridge list is **empty** (the return is `dict[(i, j), list[str]]`; there is no typed `refs` field) | Call once, cap evidence. Bridged pairs (element / constraint strings present, incl. zeroLength / equalDOF / contact) are not findings. v1 branches on emptiness only — never parse the bridge strings. |
| `MESH.EMPTY_PG` | warning | `PhysicalGroupSet` directly | Named PG with 0 nodes. Do not parse `summary()` text. |
| `RES.UNBOUND_FEM` | error | `results.fem is None` | |
| `RES.NO_STAGE` | error | `results.stages` | |
| `RES.NAN` / `RES.INF` | error | last-step recorded component | Only on names from `results.inspect.components()` (`available_components()` lives on the composites / readers, not on `Results`). NaN *sentinels* (unvisited / fill) must not be failed — if the slot is a known fill, skip. |
| `RES.LINEAGE` | warning | `results.lineage.warnings` | Integrity, not physics. Assess never raises (ADR 0021 INV-2). |
| `RES.U_VS_DIAG` | **info** (always) | ‖u‖_max / model diagonal | Measurement. **Never warning or error in v1** — a ratio threshold would false-flag SSI / pushover. Skip `kind="mode"`. Phrase the ratio + node + xyz + step. Cap evidence at K=8. S2 promotes `results/plot/_arrows.model_diagonal` to a shared H5-safe helper — no public bbox helper exists on `Results` / `FEMData` today, and `ImportHealth.bbox_diag` is live-only. Target home: a numpy-only `results/_geometry.py`, re-exported from `plot._arrows` for back-compat. |
| `RES.ENERGY_ERR` | warning | `Results.energy()` `ERR` | Ladruno `-G energy` only. `TypeError` on native/MPCO → skip. |

**Out of v1 (theater, live-only, or not computable):**
`CAD.SLIVER_EDGE` / `CAD.SLIVER_FACE` (live `ImportHealth` — S4;
do **not** emit `CAD.NO_SOLIDS`, shells are legal);
`MESH.SICN_LOW` from an H5 snapshot (SICN lives on live
`gmsh.model.mesh.getElementQualities`);
`MESH.BANDWIDTH`; `MODEL.NO_SP` keyed off names / `fem.nodes.sp`
(supports are `ops.fix` / `model.fixes()`); `MODEL.NO_LOAD` on eigen;
`RES.COMP_MISSING` (that is `diagnose()`); global “in equilibrium”;
“mechanism” as a second FAIL on huge *u*; AISC/ACI/fy; invented
units (the library has none — report bbox diagonal, not mm).

`OpenSeesModel.assess()` (fixes / patterns / “case never imported” /
`RES.ZERO_U` — deciding a run “looks static-with-loads” needs pattern
knowledge `FEMData` and `Results` do not have) is a named follow-up,
not v1.

FAIL is reserved for `MODEL.EMPTY`, `RES.UNBOUND_FEM`, `RES.NO_STAGE`,
`RES.NAN`/`RES.INF` on a recorded field, and `MESH.INVERTED`.
Everything else is a measured finding. The skill maps codes to next
actions (`bind`, re-record, `make_conformal`, …); the library does
not emit `fix=` Python.

### Part 2 — Render is the viewer’s camera, not its window

Promote `scripts/render_showcase/stills.py` into
`apeGmsh.viewers.render`. (Naming: this module *writes stills*; the
ADR 0042 backends keep `RenderBackend.render()` = pump the scene —
same word, different verb; the module docstring says so.) Stills come
from the **same scene builders
and diagrams** the Qt viewers use, on an offscreen PyVista plotter
— not from puppeteering docks, and not from matplotlib as the
primary look.

```python
fem.render("mesh.png")                 # snapshot; replaces dead fem.viewer()
results.render(
    "uz.png",
    view="contour",                    # mesh | contour | deformed | reactions
    component="displacement_z",
    step=-1,
    deform=None,                       # None | float | (field, scale)
    camera="iso",                      # iso | xy | xz | yz
    window_size=(1280, 720),
)
results.render_pack("out_dir/")        # canned 3–5 stills
g.model.render("geom.png")             # live session; later slice
g.mesh.render("mesh.png")              # live session; later slice
```

`render` returns the written `Path`; `render_pack` returns the tuple
of paths. Under `APEGMSH_SKIP_VIEWER=1` or with no GL they return
`None` / `()` and print the existing `[skip viewer]` notice, so an
agent never `read_file`s a PNG that was not written.

Implementation (results / FEMData):

1. `pv.Plotter(off_screen=True)` — VTK offscreen, **not** Qt’s
   `offscreen` platform.
2. `build_fem_scene` + `ResultsDirector` + the registered diagram
   (`ContourDiagram`, …) attached through `PyVistaQtBackend`.
3. Deform goes through `director.geometries` (ADR 0058). The
   hand-rolled point warp in `stills.py` is retired on this path.
4. Theme + scalar bar from the existing view annotation stack
   (ADR 0081 / 0090). No docks, no outline, no inspector.
5. `plotter.screenshot(path)`; `plotter.close()`.

Canned `render_pack` (report set, not a poster):

1. Undeformed mesh.
2. Last-step contour of the primary field (`mag(displacement)` or a
   named component).
3. Same, deformed (auto scale = a visible fraction of the model
   diagonal — same idea as `stills.py` `deform_fraction`, but via
   the director).
4. Reactions, static only, if recorded.
5. Optional: history at the extrema node. That one plot may be
   matplotlib; it is 2-D and must be labeled as such.

#### Fallback ladder

1. **VTK offscreen plotter** (preferred).
2. **Hidden `ResultsViewer.show(run_loop=False)`** — only if (1)
   cannot get a GL context. Reuses the existing
   `export_animation` realize path. Must **not** set
   `QT_QPA_PLATFORM=offscreen` on Windows. Pending open question 2
   — S1 ships steps 1 + 3 only until that call lands.
3. **Skip figures.** `APEGMSH_SKIP_VIEWER`, or no GL (CI / SSH).
   The report says “no stills; numbers only.” Matplotlib is a last
   resort and must be labeled so nobody thinks it is the Qt viewer.

A later slice adds `python -m apeGmsh.viewers render <path> …` next
to the existing interactive `__main__`, so a GL crash stays out of
the kernel. In-process is the default.

Agents never call `g.model.viewer()` / `g.mesh.viewer()` /
`results.viewer()` / `show_web` / `gui()` for diagnosis.
`viewer(blocking=False)` is allowed only when the **human** asked
to look, and only after assess.

### Part 3 — Skill change is part of the decision

The existing apeGmsh skill (not a new zip) grows an `assess.md`
reference and the happy-path / gotchas stop listing viewers as the
visual check. After `generate()` / `get_fem_data` / `Results.from_*`:

1. `assess()` (or the inspect checklist, until the method lands).
2. Print `report.text`; `read_file` each PNG in `report.figures`.
3. Act on `severity="error"`; do not proceed to a human memo.
4. Open Qt only if the human asked.

A library API without this skill rewrite is dead code (workshop
agent-UX). The stale “`blocking=True` is the default” claim must be
corrected to the live auto-detect behaviour everywhere it appears:
`results.md` (×4), `workflows.md` (×2), `SKILL.md`, and
`section-properties.md` — in the last one only the `results.viewer`
half is stale; `sec.viewer` genuinely defaults to `blocking=True`.

## Invariants

- **INV-1 (import).** `apeGmsh.assess` must not import
  `apeGmsh.viewers` or `gmsh`. Figure requests call
  `results.render_pack` / `fem.render` through the already-imported
  broker object (the method lives on `Results` / `FEMData` and
  lazy-imports `viewers.render`). An AST guard siblings the
  ADR 0014 / 0056 guards.
- **INV-2 (no event loop).** `render` / `render_pack` never call
  `app.exec_()` / `win.exec()`. They do not present docks.
- **INV-3 (skip ≠ pass).** A check that cannot run (no energy
  channel, no live kernel, no recorded reactions) is omitted, not
  scored OK. The markdown lists skipped checks.
- **INV-4 (FAIL reserved).** Only the error codes in the v1 table.
  `RES.U_VS_DIAG` is never `error`.
- **INV-5 (H5-first).** `fem.assess` / `results.assess` /
  `fem.render` / `results.render` work with no live Gmsh session.
  Live-only evidence (`ImportHealth`, SICN, BRep stills) degrades
  or waits for the live-session slice.
- **INV-6 (closed catalog).** New finding codes are an ADR
  amendment (appendix bump), not a drive-by in a diagram PR.
- **INV-7 (Qt look).** Primary stills come from the viewer scene /
  diagram pipeline. Matplotlib stills in a pack are labeled.
- **INV-8 (lineage).** Assess never raises on hash drift. Warnings
  become `RES.LINEAGE` findings and `report.lineage` carries the
  object.
- **INV-9 (time).** P0 extrema and contours use `time=-1` or
  `node_envelope`. `node_envelope` (like `energy()`) is Ladruno-only
  and raises `TypeError` on native / MPCO and on partitioned-envelope
  merges — fall back to `time=-1`, never to all-steps. Walking every
  step is opt-in and not the default.
- **INV-10 (no director scripting).** The agent-facing `view=` set
  is closed (`mesh` / `contour` / `deformed` / `reactions`).
  `setup(plotter, director)` stays a human/script escape hatch on
  `export_animation`, not on `render`.

## Alternatives rejected

| Rejected | Why |
|---|---|
| **Wrap the Qt viewers** (agent clicks Outline, screenshots the window) | 55k LOC, cascade-fragile (ADR 0084). Agents cannot see the window. Explicit `blocking=True` still kills a Jupyter kernel; the default is no longer that. |
| **Grow `fem.inspect` / `results.inspect` with `notes() -> str` only** | Smallest possible ship, but agents narrate prose and dump tables. Inventory and verdict collapse. `find_coincident_node_pairs` is the cautionary leak, not the pattern. |
| **`g.assess` as a `_COMPOSITES` entry** | Looks like a sixth authoring stage. On `from_h5` it promises CAD checks that cannot run. |
| **Package-only, no methods** (`from apeGmsh.assess import …` as the only door) | Breaks notebook tab-complete (principles: notebooks are the habitat). |
| **Matplotlib `results.plot` as the report look** | Fine as labeled fallback. Wrong as the default: reports should match the Qt viewer the human will open. |
| **Path A (`ResultsViewer.show(run_loop=False)`) as the default still** | Works, but flashes a window, pulls the full shell, and is illegal under Windows `QT_QPA_PLATFORM=offscreen`. Keep as fallback (2). |
| **P4 offscreen VTK *inside* `apeGmsh.assess`** | That is how assess becomes a second viewer. Render lives in `viewers.render`. |
| **CLI / MCP / JSON schema in v1** | No consumer. Habitat is the library + skill. A `python -m apeGmsh.viewers render` subcommand is the later MCP body. |
| **AISC / ACI / fy / “in equilibrium” gates** | Broker is units-agnostic and solver-neutral. Reaction vs applied load is not generally computable (no λ(t), element-load resultants, often partial reaction recorders). Design checks belong in apeSteel / apeConcrete. |
| **`MESH.SICN_LOW` from H5** | Quality arrays are not in the snapshot. Inventing SICN would embarrass the library. |
| **Library-level Qt ban** | Viewers stay the human surface. The skill forbids them for *agents*; the library does not delete them. |
| **`fix=` snippets on `Finding`** | Will rot the first time an API moves. Skill maps codes → next action. |

## Consequences

- After a script, the agent’s happy path is `results.assess(figures=True)`
  then `read_file` on a handful of PNGs. No Qt window, no
  `node_table` in context.
- `fem.render` fills the hole `fem.viewer()` left, for both agents
  and “give me a snapshot PNG” humans.
- Docs stills (`scripts/render_showcase/stills.py`) should call
  `viewers.render` once it exists — one still path.
- CI without GL keeps going: findings run, figures skip.
- Assess cannot become a second catalog of diagram kinds (INV-10,
  ADR 0084 D1 spirit): new pictures are new `view=` tokens, reviewed
  here.
- Skill freshness: `assess.md` is stamped like the other
  references; a stale “open the viewer” line is a bug.

## Slices

| Slice | Ships | Depends |
|---|---|---|
| **S0** | Skill checklist + gotchas: inspect surfaces, never Qt for diagnosis; fix the stale `blocking` default claim in `results.md` / `workflows.md` / `SKILL.md` / `section-properties.md`. No library. | — |
| **S1** | `apeGmsh.viewers.render` + `results.render` / `fem.render`. One still, last step, iso. Productize `stills.py`; deform via director. | ADR 0042 / 0058 |
| **S2** | `apeGmsh.assess` + `fem.assess()` / `results.assess()`. v1 catalog. AST import guard. | inspect / lineage / coincident pairs |
| **S3** | `render_pack` + `assess(figures=True)` (default **False**). | S1 + S2 |
| **S4** | `g.model.render` / `g.mesh.render` (live session). | S1, live kernel |
| **S5** | `python -m apeGmsh.viewers render …` subprocess. | S1 |
| **Later** | `OpenSeesModel.assess()`; inverted-cell tests on mixed types; envelope-first extrema when `-envelope`; optional MCP wrapping S5. | S2 |

S0 can merge alone. Claiming the feature “done” requires S0 + S1 + S2
at minimum (an agent can judge and attach one Qt-look still). S3 is
the report pack.

## Acceptance

The list below is the implementer contract, independent of who — or
what — writes the code. The AST guard, the invariant tests, and the
GL-less CI lane land **in the same PR** as S2, not as follow-ups.
Ambiguities discovered while implementing (NaN sentinel semantics per
reader, evidence caps, message phrasing) are resolved *in the PR
description*, not silently in code — they are the review's first
stop.

- `fem.assess()` / `results.assess()` on the demo / a small H5
  fixture return a frozen report; dirty lineage, empty refs on a
  planted coincident pair, and a non-finite recorded component
  appear as the matching codes.
- `results.render("x.png", view="contour", …)` writes a PNG and
  returns its `Path` without entering a Qt event loop;
  `APEGMSH_SKIP_VIEWER=1` returns `None` with the `[skip viewer]`
  notice and no file.
- `apeGmsh.assess` AST-guard: no `viewers` / `gmsh` import.
- Skill / `workflows.md` happy path after solve does not mention
  `viewer()` / `show_web` as the check.
- A GL-less CI job runs S2 tests green (figures skipped).

## Open questions

Resolved in this draft (veto here, do not re-litigate in the first
PR):

1. **`OpenSeesModel.assess()`** — deferred. Solver-zone gates
   (`fixes()`, patterns, case never imported) belong there, not on
   `FEMData`.
2. **`figures=` default** — `False`. Headless stays cheap; the skill
   passes `True` when the agent needs eyes.
3. **`MESH.INVERTED` in v1** — yes. It is the honest off-session
   quality check. SICN stays live-session-only and is not faked.
4. **`RES.U_VS_DIAG` severity** — always `info` in v1. A warning
   threshold is an ADR amendment after we have numbers.
5. **`CAD.SLIVER_*`** — not v1. Live kernel + `ImportHealth`; S4.

Closed (2026-08-13 workshop — architecture / viewer / agent-UX /
skeptic / API):

1. **S4 BRep tessellation** — keep the existing coarse on-the-fly
   tessellation (`build_brep_scene`, same as ModelViewer, including a
   throwaway 2-D mesh when none exists). Need fine triangles:
   `generate()` then `g.mesh.render`. No public size knob. Requiring a
   mesh would make `g.model.render` useless in the geometry-phase
   notebook.
2. **Hidden-window fallback (ladder step 2)** — **never.** VTK
   offscreen or skip. `ResultsViewer.show(run_loop=False)` flashes,
   pulls the full shell (ADR 0084), and Windows
   `QT_QPA_PLATFORM=offscreen` still crashes `ViewerWindow`.
   `export_animation` already owns that cost for movies. S1/S5 stay
   ladder 1+3.
3. **`stills.py`** — stays a **poster** script (titles, tubes, zoom,
   hand-rolled warp). `viewers.render` is the product still. No
   pixel-parity harness. Do not grow `title=` / `line_tubes=` /
   `setup=` onto the closed `view=` set (INV-10).

## Amendment 1 (2026-08-14) — catalog honesty after adversarial review

Append-only. The Accepted body (S0–S5, Q1–Q3, closed workshop
items) stays. This amendment resolves residual false-FAIL and
false-confidence defects found after ship: a red/blue pass plus an
independent review against `origin/main`. It does not reopen the
product shape (sidecar, three verbs, FAIL reserved, skip ≠ pass,
render is the camera, skill is the agent contract).

INV-6: `RES.ENERGY_NONFINITE` is the only new code. There is no
`MESH.QUALITY_UNJUDGED` — `AssessmentReport.skipped` already carries
that information.

### Return type

`AssessmentReport` gains a branchable skip list (empty default, so
existing frozen construction stays valid):

```python
skipped: tuple[tuple[str, str], ...] = ()
# (code, reason) — the same pairs the markdown already listed.
```

`findings` remains what the agent branches on for emitted verdicts.
`skipped` is what it branches on for “this FAIL-reserved check never
ran.” Text-only disclosure is not enough.

### Verdict contract

`report.text`’s Verdict block must state:

1. Evaluation scope: last recorded step only (INV-9). A mid-history
   blow-up that finishes finite is invisible; that is accepted.
2. **Not evaluated (FAIL-reserved):** every catalog error code that
   appears in `skipped` (today: `MESH.INVERTED` when no judged cell
   type ran; `RES.NAN` when there are no stages). Agents must not
   read `0 error(s)` as “the mesh was judged.”

### `MESH.INVERTED`

**Judged** (linear corner measure, not SICN, not Jacobian):

| Type | Rule |
|---|---|
| `tet4`, `hex8` | Signed corner volume. Negative or zero is inverted. |
| `tri3`, `quad4` | **Degeneracy only** (`abs(measure) == 0`): zero area or bowtie. `_signed_quad4` already returns `0.0` for both. |

**Not judged** (listed in `skipped`, not scored OK):

- `prism6`, `pyramid5` (no rehearsed signed-volume formula in-tree)
- all quadratic / Bézier cells (ADR 0091 control values are not nodal coords)
- all 1-D cells (`line2`, `line3`, …)
- `tri3` / `quad4` when the mesh is not planar

**Planarity is mesh-wide**, not per-group: one out-of-plane node
disables 2-D judging for the whole model. Conservative (no false
FAIL). 2-D inversion is therefore effectively never judged on a 3-D
building model. Documented, not changed.

**Winding is a convention.** A uniformly clockwise planar slab is
legal (STEP face normals). Judging `measure <= 0` on planar `tri3` /
`quad4` false-FAILs every cell. That is the same class of defect as
the hex8 signed-volume blocker. Tradeoff: a genuinely reversed
plane-stress mesh is a disclosed miss. A disclosed miss beats an
undisclosed 100 %-of-elements false FAIL.

v1 still does not invent prism6 / pyramid5 formulas. That stays in
the Later row and needs its own sweep harness.

### `RES.NAN` — operational fill rule

ADR body § catalog: “NaN *sentinels* (unvisited / fill) must not be
failed.” S2 shipped “judge `slab.values` as returned,” which
false-FAILs apeGmsh’s own native writer. `capture/_domain.py:end_stage`
union-merges node records (displacements on all nodes + reactions on
supports) and fills unvisited slots with NaN — the documented
`spec.nodes(components=…, pg=…)` API. MPCO partition merge and
Ladruno quaternion / station-ξ pads are **not** this bug (the former
keys the union of recorded ids; the latter are metadata, not
`values`).

**Rule (nodes level only):** at the last step of a stage, collect
each node-level component’s finite mask. A NaN slot at a node where
**any other** node-level component is finite is union-merge fill —
not a finding; a `skipped` entry with component + count. A node
where **every** recorded node-level component is non-finite is
unexplained and stays `RES.NAN`. Other levels (gauss / fiber /
layer / line-station / elements) are unchanged.

This is decidable from data already loaded (no `time=None`, no
`node_envelope`, INV-9 intact). A native-schema per-component id set
is the real fix and belongs to the results-schema line, not here.

Gauss / fiber / layer / line-station NaN evidence uses
`element_index` when `node_ids` / `element_ids` are absent.

### `RES.ENERGY_ERR` / `RES.ENERGY_NONFINITE`

`Results.energy()` `ERR` is the **normalized energy-balance error %**
(unit-free). `np.isclose(err, 0.0)` is `|err| ≤ ~1e-8` — one part in
1e10 — so the warning fired on essentially every real Ladruno run.
The “unit-scale sensitive (N·mm·tonne)” comment was wrong.

| Code | Severity | When |
|---|---|---|
| `RES.ENERGY_ERR` | **info** (always, when a finite ERR exists) | Measurement, same discipline as `RES.U_VS_DIAG`. Phrase as `%` with `abs()`. No threshold in this amendment. |
| `RES.ENERGY_NONFINITE` | warning | Last-step `ERR` is NaN or Inf. The one real energy signal. |

`TypeError` on native/MPCO still skips (INV-3). FAIL reserved is
unchanged — neither code is an error.

### Render (implementation note, not a catalog change)

`results.render` / `render_pack` validate bound FEM (and the closed
`view=` / `camera` / `deform` sets) **before** `APEGMSH_SKIP_VIEWER`.
An unbound Results must raise, not print `[skip viewer]`. Broad
`except Exception` around pack helpers is illegal; the legitimate
empty case is `RuntimeError` / `ValueError`.

### Skill

`assess.md` (canonical `skills/apegmsh/` **and** the synced
`.claude/skills/apegmsh-helper/` copy) must name the judged /
unjudged inversion set, last-step scope, the NaN fill rule,
`report.skipped`, energy codes, coincident `tol=1e-6` as an absolute
unit trap (misses, does not over-report), zero-length lines as
bridges not findings, and that assess does not check supports or
loads. A stale “open the viewer” line remains a bug.

### Explicitly out of this amendment

prism6 / pyramid5 inversion formulas; SICN from H5; a `U_VS_DIAG` or
energy **threshold**; parsing coincident bridge strings; hidden-Qt
fallback; `g.assess`; `OpenSeesModel.assess()`; per-group planarity;
step-walking; native-schema per-component coverage; a new ADR.
`MESH.QUALITY_UNJUDGED` was considered and rejected (duplicates
`skipped`).

## Amendment 2 (2026-08-16) — `OpenSeesModel.assess()`: the solver-zone verdict

Append-only. The Accepted body, the resolved and closed questions,
and Amendment 1 stay. This amendment ships the Later row's
`OpenSeesModel.assess()` — the solver-zone gates v1 deliberately
refused because `FEMData` and `Results` lack pattern knowledge. It
does not reopen the sidecar shape, the three verbs, FAIL-reserved,
skip ≠ pass, or INV-1–INV-10.

Evidence: a read of the shipped read-side broker
(`opensees/opensees_model.py`, `_internal/typed_records.py`,
`_internal/build.py::validate_from_model_cases`, `emitter/h5.py`).
Three facts constrain the catalog:

1. **`eigen` is not archived.** The H5 emitter treats `eigen` /
   `eigen_feast` as runtime one-shot retrievals (deliberate no-op, no
   schema). An eigen-only deck is indistinguishable from a deck that
   never declared a run. No code below may claim to know the
   analysis is modal.
2. **`from_model(case)` provenance is not archived.** The read-side
   `PatternRecord` carries the already-expanded `loads` / `sps` /
   `ele_loads` lines, not case names. Per-case "declared but never
   imported" is undecidable from an archive. (The inverse — a
   pattern importing a case with zero records — already fails loud
   at emit: `validate_from_model_cases`, ADR 0051.)
3. The broker-side case inventory is `LoadRecord.pattern` (default
   `"default"`) on `fem.nodes.loads` / `fem.elements.loads` plus the
   non-homogeneous rows of `fem.nodes.sp`.

### Entry point

```python
report = osm.assess()                  # OpenSeesModel, H5-safe
report = osm.assess(results=results)   # + RES.ZERO_U cross-check
```

Same frozen `AssessmentReport` / `Finding` types. The runner lives
in `apeGmsh.assess._opensees` and receives the broker object — the
package still imports neither `gmsh`, `apeGmsh.viewers`, nor
`apeGmsh.opensees` (INV-1 extended; the AST guard grows the third
name). The method on `OpenSeesModel` lazy-imports the runner, same
as `FEMData.assess` / `Results.assess`. No `figures=` — there is no
`osm.render`; stills stay on `fem` / `results`.

`osm.assess()` judges the `/opensees` zone only. It does not re-run
`fem.assess(osm.fem)` — the skill teaches the trio (`fem` → `osm` →
`results`) and `report.text` says so. One door per zone; no
duplicated catalogs.

Small API addition: a public `OpenSeesModel.analyze_call()` accessor
(`tuple[steps, dt | None] | None`) — the field exists today, private
only.

**Staged decks are out of v1.** When `stages()` is non-empty the
whole solver catalog lands in `report.skipped` with the reason
"staged deck — stage-level judging is a follow-up". Stage support
patterns hold DOFs and carry their own load imports; judging a
staged deck with vanilla rules would false-flag every SSI model.

### Catalog (solver zone, v1)

| Code | Severity | Source | Notes |
|---|---|---|---|
| `OSM.NO_ANALYZE` | info | `analyze_call() is None` and `stages()` empty | "No `ops.analyze` archived." Eigen-only decks trip this by design — the message says eigen is not archived and a modal deck is fine. |
| `OSM.NO_SUPPORT` | warning | `fixes()` empty **and** no homogeneous `SPRecord` in `fem.nodes.sp` **and** no pattern `sps` | Free-free eigen / DRM-style models are legal — hence warning, never error. |
| `OSM.NO_PATTERN` | warning | `analyze_call()` present, `patterns()` empty | Free-vibration transients (initial conditions only) are the disclosed legal case. Skipped, not emitted, when `OSM.NO_ANALYZE` fired. |
| `OSM.LOADS_UNIMPORTED` | warning | broker carries load records (`nodes.loads` / `elements.loads` / non-homogeneous `sp`) but every pattern block has empty `loads` + `sps` + `ele_loads` and no single-line pattern exists | The classic agent bug: `g.loads` declared, `from_model` forgotten. Fires also on a **zero-pattern** deck when an analyze is archived (vacuous-truth reading — `OSM.NO_PATTERN` alone reads as "free vibration is legal" and hides the stranded loads); with no archived analyze, declared loads may target a later deck (multi-deck workflow) — skipped, not judged. Zero-lines signal only — per-case attribution is undecidable (fact 2) and goes to `skipped` with that reason. |
| `RES.ZERO_U` | info | `results=` passed; last-step displacement exactly all-zero while `analyze_call()` and ≥ 1 pattern exist | Measurement, `U_VS_DIAG` discipline: info until dogfood numbers justify promotion (amendment required). Listed in `skipped` when `results=None`. |

FAIL-reserved is unchanged: none of these is `error`. Emit-time
validation already hard-fails the certain cases (phantom case →
`BridgeError`); assess reports what a run archive *looks like*, and
every look-alike here has a legal reading.

Skip ≠ pass (INV-3) applies: `RES.ZERO_U` without `results=`,
per-case attribution, staged decks, and any check on a pre-schema
archive (empty `stages()` on a pre-2.18.0 file is absence of
evidence) land in `report.skipped`.

### Alternatives rejected (this amendment)

| Rejected | Why |
|---|---|
| `osm.assess()` delegates to `fem.assess()` | Duplicated findings, two catalogs behind one door. The trio stays composable; the skill sequences it. |
| Any `error` severity in v1 | Every trigger has a legal reading (free-free eigen, free vibration, multi-deck workflows). Promotion needs numbers, per the Q4 discipline. |
| Per-case `OSM.CASE_UNUSED` by matching expanded lines back to broker records | Fragile count-matching heuristic — one consistent-load expansion mismatch away from a false claim. |
| Archiving `from_model_cases` provenance here | Real fix, wrong line — that is a `/opensees` schema change and belongs to the emitter/schema program. Named follow-up; it is what makes `OSM.CASE_UNUSED` honest. |
| Inferring eigen from `analysis()` attrs | The chain attrs describe the last configured chain, not what ran. Guessing embarrasses the library (same class as faking SICN). |
| Stage-aware judging in v1 | Stage capture buckets need their own sweep of hold/import semantics; a blanket skip is honest today. |

### Acceptance (this amendment)

- `osm.assess()` on a fixture deck with fixes + patterns + analyze
  returns zero findings from this catalog; deleting the `fix` lines
  yields `OSM.NO_SUPPORT`; dropping `from_model` while keeping
  `g.loads` yields `OSM.LOADS_UNIMPORTED` — including on a
  zero-pattern deck with an archived analyze (fires alongside
  `OSM.NO_PATTERN`), while the same deck with no archived analyze
  puts it in `skipped`; an eigen-style deck (no analyze) yields
  `OSM.NO_ANALYZE` info and **skips** `OSM.NO_PATTERN`; a staged
  fixture puts the whole catalog in `skipped`.
- `results=` with an all-zero last step yields `RES.ZERO_U` info;
  omitted `results=` lists it in `skipped`.
- AST guard: `apeGmsh.assess` imports none of `gmsh` /
  `apeGmsh.viewers` / `apeGmsh.opensees`.
- `OpenSeesModel.analyze_call()` is public and covered.
- Skill `assess.md` (canonical `skills/apegmsh/` and the synced
  mirror) grows the solver-zone section: the trio order, the five
  codes, and the three undecidables (eigen, per-case attribution,
  staged decks) as honest limits.

### Open questions (this amendment)

Resolved here:

1. **One door per zone** — yes; no delegation.
2. **All non-error in v1** — yes; promotion is a numbered amendment
   after dogfood.
3. **Staged decks** — blanket skip in v1.

Named follow-ups (not this amendment): archive `from_model_cases` on
the pattern zone (schema bump) → per-case `OSM.CASE_UNUSED`;
stage-aware solver judging.

## Amendment 3 (2026-08-17) — run evidence from results; planar camera

Append-only. The Accepted body and Amendments 1–2 stay. This
amendment closes the **custom-driver gap** found on the first dogfood
run and adds one render default. It does not add codes (INV-6: the
catalog names are unchanged), does not touch severities, and does not
reopen FAIL-reserved.

Evidence: dogfood pass #1 (`examples/shoebuckle_arch.py`, PR #990).
The deck had 124 broker load records, **zero patterns**, and a
hand-written `analyze(1)` driver — so nothing was archived by
`ops.analyze`, and the run "converged" 200 steps to exactly zero
displacement everywhere. `OSM.LOADS_UNIMPORTED` and `RES.ZERO_U` —
the two codes built for precisely this failure — were both **skipped**
by the Amendment 2 gates (`analyze_call()` as the only run evidence).
Custom analyze loops are how limit-point and staged-restart runs are
driven; the most safety-critical decks were the least judged.

### Run evidence

A deck is judged as *having run* when either:

1. `analyze_call()` is archived (Amendment 2's gate), **or**
2. `results=` was passed and carries at least one **non-mode** stage
   whose last step is readable (`time=-1`, INV-9 — no step walking).

Modal-only results are deliberately **not** run evidence: an
eigen-only session stays exactly as legal as Amendment 2 left it, and
a deck whose declared loads target a later static deck (multi-deck
workflow) is still skipped, not judged, when the only results at hand
are mode shapes.

### Gate changes (same codes, same severities)

| Code | Amendment 2 gate | This amendment |
|---|---|---|
| `OSM.LOADS_UNIMPORTED` | zero-pattern decks judged only when `analyze_call()` archived | judged when **run evidence** exists; the no-evidence skip reason still says declared loads may target a later deck |
| `RES.ZERO_U` | `results=` and `analyze_call()` and ≥ 1 pattern | `results=` with ≥ 1 non-mode stage (the results are themselves the evidence). The ≥ 1-pattern requirement is dropped; when the deck has zero patterns the message cross-references `OSM.LOADS_UNIMPORTED` so the two findings read as one story. Severity stays **info** — one dogfood datapoint does not promote a threshold (Q4 discipline). |
| `OSM.NO_ANALYZE` | message: eigen is not archived | when `results=` is present, the message adds that the results show a run happened — the deck was driven by a custom loop. Still info. |

`OSM.NO_SUPPORT` / `OSM.NO_PATTERN` and the staged blanket skip are
unchanged.

### Render default (implementation note, not a catalog change)

`render` / `render_pack` / assess figures: when the model is planar
(2-D `ndm`, or a bbox whose z-extent is negligible against the
diagonal), the default camera becomes `xy` instead of `iso` — a
planar frame rendered isometrically is skewed for no reason (dogfood
friction #4). An explicitly passed `camera=` from the closed set is
honored unchanged; INV-10's set does not grow.

### Acceptance (this amendment)

- A broken-shoebuckle-shaped fixture (broker loads, zero patterns, no
  archived analyze, `results=` all-zero non-mode stage) yields **both**
  `OSM.LOADS_UNIMPORTED` (warning) and `RES.ZERO_U` (info); the
  `RES.ZERO_U` message names the zero-pattern cross-reference.
- The same deck with modal-only `results=` keeps both codes in
  `skipped` (multi-deck reading preserved).
- `results=` with a non-mode stage and nonzero last step: no
  `RES.ZERO_U`; `OSM.LOADS_UNIMPORTED` still fires when broker loads
  exist and nothing imports them.
- No `results=` and no archived analyze: identical behavior to
  Amendment 2 (skips, with reasons).
- A planar `fem.render()` / `results.render()` with no `camera=`
  writes an `xy` view; `camera="iso"` still forces iso; a 3-D model
  defaults to iso as before.
- Skill `assess.md` (canonical + mirror) states the run-evidence rule
  and that custom-driver decks are judged when results are passed.

### Open questions (this amendment)

Resolved here:

1. **Results as run evidence** — yes, non-mode stages only.
2. **Drop ZERO_U's pattern requirement** — yes; cross-reference
   instead.
3. **Planar camera auto-default** — yes; explicit `camera=` wins.

Named follow-ups (unchanged): `from_model_cases` provenance (schema
bump) → per-case `OSM.CASE_UNUSED`; stage-aware judging; threshold
promotions after more dogfood datapoints.

## Amendment 4 (2026-08-17) — thresholds from dogfood numbers

Append-only. The Accepted body and Amendments 1–3 stay. This is the
amendment the body's Q4 and Amendment 2 promised: severity and
threshold decisions made **after** real numbers, from two dogfood
passes (shoebuckle arch; steel-connection habitat: elastic static,
step-load transient, mortar and NTS contact cases). One promotion is
made, one measurement is refined, and one promotion is **declined** —
the numbers argue against it.

### Evidence — the datapoint table

| Run | Kind | `U_VS_DIAG` | Energy `ERR` (last step) |
|---|---|---|---|
| Shoebuckle arch, λ=1 pushover | MPCO, LoadControl | 5.3e-4 | no channel |
| Shoebuckle arch, **stranded loads** (zero patterns) | broken by construction | **exactly 0** | no channel |
| Steel connection, elastic static 100 kN | Ladruno static | 0.0185 | **all-zero frame** (channel recorded, unpopulated) |
| Steel connection, step-load transient (Newmark, 0.04 s) | Ladruno transient | 0.0358 | **0.141 %** (startup spike 31 % at t=dt; last-step scope ignores it) |
| Steel connection, mortar contact, 90 mm drive | Ladruno contact | 0.0516 | no channel |
| Steel connection, NTS contact, 150 mm drive | Ladruno contact | 0.1245 | no channel |

Cross-checks that make these trustworthy: the transient's energy
closure is physical (KE + IE tracks ULW; DW = 0 on the undamped
elastic model) and its peak displacement is 1.94× the static case —
the textbook ≈ 2.0 step-load amplification.

### Decision 1 — `RES.U_VS_DIAG`: promotion **declined**, permanently info

Legitimate, converged, intended runs span **5.3e-4 to 0.1245** —
nearly three orders of magnitude — because displacement-driven tests
legitimately push past 10 % of the model diagonal. No fixed ratio
separates a heavily-driven contact test from a diverging model; any
threshold either false-flags every displacement-controlled run or is
too high to catch anything a human would miss. The ratio's value is
the **number itself** in the report, next to the extrema — not a
gate.

Revisit only if a context signal (load- vs displacement-driven)
becomes honestly archivable — that is the `from_model` /
analyze-provenance schema line, not this catalog.

### Decision 2 — `RES.ENERGY_ERR`: unpopulated-channel honesty + a 5 % warning

Two changes, both from the data:

1. **All-zero frame → skipped, not info.** Ladruno **static** runs
   record the `-G energy` channel with every component zero — the
   accumulator populates on transients only. Today that reads
   "|ERR| = 0 %", which an agent parses as a verified balance. When
   the last-step frame is entirely zero (KE, IE, DW, ULW, RES, ERR
   all 0), the check lands in `report.skipped` with reason "energy
   channel recorded but unpopulated (static run) — not evidence of
   balance". Skip ≠ pass, INV-3.
2. **Nonzero frame: warning above 5 %.** Last-step `|ERR| > 5.0` (the
   value is a percent) promotes the finding to **warning**; at or
   below, it stays info with the measured value. Basis: the one
   healthy implicit transient closes at 0.141 % (a 35× margin), and
   ~5 % is the conventional energy-balance alarm in explicit-dynamics
   practice. The 31 % first-step spike is a small-denominator
   normalization artifact that the last-step scope (INV-9) already
   ignores — no startup special-casing is added. Single-datapoint
   basis is disclosed: the constant is revisited when explicit-run
   datapoints exist (fork program), by amendment.

`RES.ENERGY_NONFINITE` is unchanged (warning). FAIL-reserved is
unchanged — neither energy code is ever `error`.

### Decision 3 — `RES.ZERO_U`: info → **warning**

Across six assessed runs: one true positive (the stranded-loads arch
— exactly the bug the code exists for) and zero false positives. The
one legal reading — a free-vibration transient started from rest and
never perturbed — is a run whose author needs no protection from a
warning, and the message already discloses it. Severity table entry
flips to `warning`. Still not FAIL-reserved; still requires run
evidence (Amendment 3) and `results=`.

### Acceptance (this amendment)

- Catalog table: `RES.ZERO_U` severity `warning`; existing tests
  asserting `info` updated to the amendment, named in the PR body.
- An all-zero last-step energy frame yields a `RES.ENERGY_ERR`
  skip entry (reason names the unpopulated channel), no finding; the
  transient fixture at 0.141 %-style values stays info; a fixture
  frame with last-step `|ERR| = 6 %` yields severity `warning`.
- `RES.U_VS_DIAG` emits info at 0.1245-style ratios — a regression
  test pins that no threshold exists (the declined promotion is a
  contract, not an omission).
- The threshold constant lives once in `_catalog.py` (or sibling),
  named, with the revisit note.
- Skill `assess.md` (canonical + mirror): the three decisions in the
  solver/results sections — U_VS_DIAG is a number to read, not a
  gate; zero-energy statics are silent, not verified; ZERO_U is now
  a warning.

### Open questions (this amendment)

Resolved here:

1. **U_VS_DIAG threshold** — declined; permanent info (Q4 closed).
2. **ENERGY_ERR warning at 5 %** — yes, single-datapoint basis
   disclosed; revisit with explicit-run data by amendment.
3. **ZERO_U severity** — warning.

Named follow-ups (unchanged plus one): `from_model_cases` /
drive-provenance archiving; stage-aware judging; explicit-run
energy datapoints to confirm or move the 5 % constant.
