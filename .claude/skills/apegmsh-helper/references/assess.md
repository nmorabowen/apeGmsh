# Assess + render — agent check after mesh / solve
<!-- skill-freshness: verified against apeGmsh feat/adr94-a3-run-evidence (2026-08-17) · signatures: python -m apeGmsh.studio.lookup SYMBOL (ADR 0096); src/ is not the authoring lookup -->

After `get_fem_data` / `Results.from_*` the check is the trio
`fem.assess()` → `osm.assess()` → `results.assess()`, then `read_file`
each PNG in `report.figures`. Not `viewer()` / `show_web()`. Qt is for
humans who asked to look, and only after assess.

Read `report.skipped` **and** the Verdict line before treating
`0 error(s)` as a pass. A FAIL-reserved check that did not run is
listed as **Not evaluated**, not OK. Scope is the last recorded step
only (INV-9) — a mid-history blow-up that finishes finite is
invisible.

Three verbs stay distinct:

| Verb | Job | Home |
|---|---|---|
| `inspect` | Inventory (tables, strings) | `fem.inspect`, `results.inspect` |
| `diagnose` | Routing of *one* empty query | `results.inspect.diagnose(component)` |
| `assess` | Verdict: findings + short markdown | `fem.assess()`, `osm.assess()`, `results.assess()` |

There is no `g.assess`. Live CAD health stays
`g.model.io.diagnose() → ImportHealth`.

## Doors

```python
report = fem.assess()                       # findings only
report = osm.assess()                       # OpenSeesModel, solver zone
report = osm.assess(results=results)        # + RES.ZERO_U cross-check
report = results.assess()                   # + lineage as RES.LINEAGE
report = results.assess(figures=True)       # + results.render_pack
report = fem.assess(figures=True)           # + fem.render → mesh.png

print(report.text)                          # 40–80 lines markdown
for path in report.figures:                 # () if skipped / figures=False
    # read_file each PNG — do not invent a file that was not written
    pass
for code, reason in report.skipped:         # branchable (Amendment 1)
    ...
for f in report.findings:
    if f.severity == "error":
        # do not proceed to a human memo
        ...
```

`# src/apeGmsh/mesh/FEMData.py` (`assess` / `render`)
`# src/apeGmsh/opensees/opensees_model.py` (`assess` / `analyze_call`)
`# src/apeGmsh/results/Results.py` (`assess` / `render` / `render_pack`)
`# verified: tests/assess/test_assess.py`, `tests/assess/test_assess_opensees.py`

`figures=True` default is `False`. Headless stays cheap. Pass `True`
when you need eyes. `APEGMSH_SKIP_VIEWER=1` or no GL: findings still
run, `figures` is `()`, prints
`[skip viewer] APEGMSH_SKIP_VIEWER set` or
`[skip viewer] no GL context` (exact). Skip ≠ pass.
Unbound `results.render` / `render_pack` raise even under that env
(need `model=` / `model_h5=` / `bind`).

Stills come from `viewers.render` (VTK offscreen). Closed `view=`:
`mesh` / `contour` / `deformed` / `reactions`. No `setup=`. In-process
is the default; `python -m apeGmsh.viewers render …` is the
out-of-kernel door so a GL crash does not kill the notebook.

Live session (needs an open Gmsh kernel; `from_h5` / closed →
`RuntimeError`): `g.model.render("geom.png")` (BRep, coarse tessellation
if no mesh), `g.mesh.render("mesh.png")` (live Gmsh, not `fem.render`).
Snapshot mesh stills stay on `fem.render`.

Do not call `g.model.viewer()` / `g.mesh.viewer()` / `results.viewer()`
/ `show_web` / `gui()` for diagnosis.

## Finding catalog (v1 + Amendment 1) → next action

New codes are an ADR 0094 amendment. The library does not emit `fix=`
Python. Branch on `finding.code` and `report.skipped`.

| Code | Sev | Next action |
|---|---|---|
| `MODEL.EMPTY` | error | No nodes or elements. Do not solve. Rebuild geometry / mesh. |
| `MESH.INVERTED` | error | `tet4`/`hex8`: negative/zero corner volume. Planar `tri3`/`quad4`: **degeneracy only** (zero area / bowtie). Winding is a convention — a clockwise slab is not inverted. **Not judged** (listed in `skipped`, not OK): `prism6`, `pyramid5`, quadratic/Bézier, all 1-D, 2-D cells in a non-planar 3-D mesh. Planarity is mesh-wide. |
| `MESH.COINCIDENT_UNBRIDGED` | warning | Pair shares XYZ and the bridge list is empty. `g.model.queries.make_conformal(...)` or `g.constraints.equal_dof` / `rigid_link` / contact. Bridged pairs (including a zero-length `line2`) are not findings. `tol=1e-6` is **absolute** (unit trap in mm models — misses, does not over-report). |
| `MESH.EMPTY_PG` | warning | Named PG with 0 nodes. Fix the group or drop it. |
| `RES.UNBOUND_FEM` | error | `results.fem is None`. Construct with `model=` / `model_h5=` or `results.bind(fem)`. |
| `RES.NO_STAGE` | error | No stages recorded. Re-run with a recorder / stage. |
| `RES.NAN` / `RES.INF` | error | Unexplained non-finite on a recorded component at last step. Node-level **union-merge fill** (displacement everywhere + reactions on supports) is skip, not FAIL — see `skipped`. Re-record unexplained NaNs. |
| `RES.LINEAGE` | warning | `results.lineage.warnings` — integrity, not physics. Assess never raises (ADR 0021). |
| `RES.U_VS_DIAG` | info | ‖u‖_max / model diagonal. Never warning/error in v1. Skip `kind="mode"`. |
| `RES.ENERGY_ERR` | info | Ladruno `ERR` in **%**. Measurement, not a gate. Native/MPCO → skip (`TypeError`), not OK. |
| `RES.ENERGY_NONFINITE` | warning | Last-step `ERR` is NaN/Inf. The one real energy signal. |

A check that cannot run is omitted, not scored OK. `report.skipped`
and the Verdict **Not evaluated (FAIL-reserved)** line name those
codes.

## Solver zone — `osm.assess()` (ADR 0094 Amendment 2 + 3)

`osm` is the `OpenSeesModel` broker (`OpenSeesModel.from_h5(...)`, or
the object a chain-phase build already hands you). It judges the
`/opensees` zone only — it does **not** re-run `fem.assess(osm.fem)`.
Run the trio in order: `fem.assess()` → `osm.assess()` →
`results.assess()`. No `figures=` — there is no `osm.render`.

**Run evidence (Amendment 3).** A deck *has run* when either an
`ops.analyze` call is archived, **or** `results=` was passed and
carries at least one **non-mode** stage with readable data. This
closes the custom-driver gap: a hand-written `analyze()` loop leaves
no `analyze_call()` trace, so without `results=` the deck reads as
"never ran" and `OSM.LOADS_UNIMPORTED` / `RES.ZERO_U` both go quiet —
exactly the decks most worth judging. Pass `results=` to get them
judged. Modal-only `results=` (`kind="mode"` stages only) is
deliberately **not** evidence — an eigen-only session, or a deck whose
loads target a later static deck, stays legal and skipped.

| Code | Sev | Next action |
|---|---|---|
| `OSM.NO_ANALYZE` | info | No `ops.analyze` archived. `eigen` is never archived, so an eigen-only / modal deck trips this by design — not a defect. If `results=` supplies run evidence, the message says so (the deck was driven by a custom analyze loop). |
| `OSM.NO_SUPPORT` | warning | No `fix`, no homogeneous `sp`, no pattern `sp` line. Free-free eigen / DRM models are legal — confirm intent, don't assume a missing `ops.fix`. |
| `OSM.NO_PATTERN` | warning | Analyze archived but no pattern declared. Free-vibration transients (initial conditions only) are legal. Skipped, not emitted, when `OSM.NO_ANALYZE` already fired. |
| `OSM.LOADS_UNIMPORTED` | warning | Broker carries `g.loads` / prescribed-displacement records but no pattern imports them — the classic `from_model` forgotten bug. Add `p.from_model("case")` (or the matching pattern body). Judged whenever run evidence exists (archived analyze **or** `results=`); with no run evidence at all, declared loads may target a later deck (multi-deck workflow) — skipped, not judged. |
| `RES.ZERO_U` | info | `results=` passed with >=1 non-mode stage — the results are themselves the run evidence, no `analyze_call()` / pattern required. Last-step displacement exactly all-zero. Measurement, not a gate — check the pattern actually drives the model. On a zero-pattern deck the message cross-references `OSM.LOADS_UNIMPORTED` so the two findings read as one story. Listed in `skipped` when `results=` is omitted. |

Three honest limits, not bugs to file:

- **Eigen is not archived.** `ops.eigen` / `ops.eigen_feast` are
  runtime one-shot retrievals with no `/opensees` schema — an
  eigen-only deck is indistinguishable from a deck that never
  declared a run. No code claims to know the analysis was modal.
- **Per-case attribution is undecidable.** `from_model(case)`
  provenance is not archived — `OSM.LOADS_UNIMPORTED` is a
  zero-lines-imported signal only, never "case X was forgotten."
- **Staged decks are a blanket skip.** When `osm.stages()` is
  non-empty the whole solver catalog lands in `report.skipped`
  ("staged deck — stage-level judging is a follow-up") — stage
  support patterns hold DOFs and carry their own load imports, so
  vanilla rules would false-flag every SSI model. `0 error(s)` on a
  staged deck means "not judged," not "clean."

## Out of v1

`CAD.SLIVER_*`, `MESH.SICN_LOW` from H5, `g.assess`, AISC/ACI/fy, “in
equilibrium”, invented units. Design checks belong in apeSteel /
apeConcrete. Solver zone: per-case `OSM.CASE_UNUSED` (needs
`from_model_cases` archival — a schema bump), stage-aware judging.

`inspect` / `diagnose` remain for tables and one-component routing.
Do not dump `node_table()` into context as the verdict.
