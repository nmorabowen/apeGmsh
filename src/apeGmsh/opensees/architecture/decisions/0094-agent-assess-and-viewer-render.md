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
