# ADR 0094 — Agent assess/report + offscreen viewer render (summary)

**Status:** Accepted (2026-08-13 — S0–S5 shipped; Q1–Q3 closed below)

## Decision

**Part 1: `apeGmsh.assess` is a sidecar; inspect stays inventory.** A standalone package (the
ADR 0060 `hpc` shape: types + pure runners, not a session composite, not re-exported from
`apeGmsh/__init__.py`). Doors are one-shot methods on brokers: `fem.assess()`,
`results.assess()`, `results.assess(figures=True)`; no `g.assess`. Three verbs stay distinct:
`inspect` (inventory), `diagnose` (routing / CAD health of one thing), `assess` (verdict).
It returns frozen `Finding(code, severity, message, detail)` and `AssessmentReport(findings,
text, figures, lineage)`; `text` is 40–80 lines of markdown, `detail` never carries code to
exec. `APEGMSH_SKIP_VIEWER=1` skips figures only. A **closed catalog** (`MODEL.EMPTY`,
`MESH.INVERTED`, `MESH.COINCIDENT_UNBRIDGED`, `MESH.EMPTY_PG`, `RES.UNBOUND_FEM`,
`RES.NO_STAGE`, `RES.NAN`/`RES.INF`, `RES.LINEAGE`, `RES.U_VS_DIAG`, `RES.ENERGY_ERR`) emits a
finding only when the data exists. FAIL is reserved for `MODEL.EMPTY`, `RES.UNBOUND_FEM`,
`RES.NO_STAGE`, `RES.NAN`/`RES.INF` and `MESH.INVERTED`; the library emits no `fix=` Python.

**Part 2: render is the viewer's camera, not its window.** `scripts/render_showcase/stills.py`
is promoted into `apeGmsh.viewers.render`: `fem.render(path)`, `results.render(path, view=,
component=, step=, deform=, camera=, window_size=)`, `results.render_pack(dir)`, and later
`g.model.render` / `g.mesh.render`. A VTK offscreen `pv.Plotter` drives the same scene
builders, `ResultsDirector` and diagrams as the Qt viewers; deform goes through
`director.geometries`. Under `APEGMSH_SKIP_VIEWER=1` or no GL it returns `None` / `()`.
Fallback ladder: VTK offscreen, else skip figures (the hidden-window step was closed "never").
Agents never open a viewer for diagnosis.

**Part 3: the skill change is part of the decision.** The apeGmsh skill gains `assess.md`;
the agent loop is assess, print `report.text`, read each PNG, act on errors, open Qt only if
the human asked.

Invariants: **INV-1** `apeGmsh.assess` imports neither `apeGmsh.viewers` nor `gmsh` (AST
guard). **INV-2** render never runs an event loop. **INV-3** skip ≠ pass. **INV-4** FAIL
reserved; `RES.U_VS_DIAG` is never `error`. **INV-5** H5-first: works with no live Gmsh.
**INV-6** closed catalog: a new code is an ADR amendment. **INV-7** primary stills come from
the viewer pipeline; matplotlib stills are labeled. **INV-8** assess never raises on lineage
drift (`RES.LINEAGE`). **INV-9** time: `time=-1` or `node_envelope`, never all steps.
**INV-10** the `view=` set is closed (`mesh` / `contour` / `deformed` / `reactions`).

## Amendments

- 2026-08-14, A1, catalog honesty: `AssessmentReport.skipped` (branchable skip list); a
  Verdict block naming unevaluated FAIL-reserved checks; `MESH.INVERTED` judged set (signed
  volume on `tet4`/`hex8`, degeneracy only on planar `tri3`/`quad4`, others skipped); the
  `RES.NAN` union-merge fill rule; `RES.ENERGY_ERR` info plus new `RES.ENERGY_NONFINITE`
  (warning); render validates before `APEGMSH_SKIP_VIEWER`.
- 2026-08-16, A2: `OpenSeesModel.assess()` ships the solver zone: `OSM.NO_ANALYZE` (info),
  `OSM.NO_SUPPORT`, `OSM.NO_PATTERN`, `OSM.LOADS_UNIMPORTED` (warnings), `RES.ZERO_U`.
- 2026-08-17, A3: run evidence is an archived `analyze_call()` or `results=` with a readable
  non-mode stage; planar models default to camera `xy`.
- 2026-08-17, A4, thresholds from dogfood numbers: `RES.U_VS_DIAG` stays permanently info;
  `RES.ENERGY_ERR` skips an all-zero frame and warns above 5 %; `RES.ZERO_U` becomes warning.

Full text: [../0094-agent-assess-and-viewer-render.md](../0094-agent-assess-and-viewer-render.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
