# Assess + render — agent check after mesh / solve
<!-- skill-freshness: verified against apeGmsh feat/adr94-a4-thresholds (2026-08-16) · signatures: python -m apeGmsh.studio.lookup SYMBOL (ADR 0096); src/ is not the authoring lookup -->

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
report = results.assess(figures=True)       # + results.render_pack (apegmsh-view/1 on Windows, VTK on Linux)
report = fem.assess(figures=True)           # + fem.render → mesh.png (same split)

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
when you need eyes. `APEGMSH_SKIP_VIEWER=1` (any platform, any source), or
no GL on the Linux VTK path: findings still run, `figures` is `()`, prints
`[skip viewer] APEGMSH_SKIP_VIEWER set` or
`[skip viewer] no GL context` (exact). Skip ≠ pass.
Unbound `results.render` / `render_pack` raise even under that env
(need `model=` / `model_h5=` / `bind`).

On **Linux and macOS**, `results.render` / `render_pack` stills are drawn by
`ResultsSession.render` (VTK offscreen; one session mesh view per
still). Closed `view=`: `mesh` / `contour` / `deformed` / `reactions`.
Refused with a `ValueError` that names the alternative:
`view="reactions"` with `component="reaction_x|_y|_z"` (use
`"reactions"`), and `deform=(field, s)` with a field other than
displacement / velocity / acceleration. No `setup=`. In-process
is the default; `python -m apeGmsh.viewers render …` is the
out-of-kernel door so a GL crash does not kill the notebook.

**On Windows** (ADR 0128 slice 6) every still verb draws through
apeGmshViewer (`apegmsh-view/1`, via `agent_view.render_stills`, below),
not VTK: `fem.render`, `results.render` / `render_pack`,
`results.assess(figures=True)`, the `python -m apeGmsh.viewers render …`
door and studio `render(path=)`. Each PNG is the render canvas only
(`"chrome": "model"`: scale bar and axes, no app panels). `fem.render`
writes the snapshot to a temporary `model.h5` first.

- **Sources.** A Results is drawn from the file `results.path` names, which
  must be a **`.ladruno` beside its `<stem>.h5`** (the door also reads a
  model `.h5`). Every other source is refused with a `ValueError` naming
  `.ladruno`: `.mpco`, a native `.h5`, recorders, a Results with no file,
  and a `to_units(...)` view. A missing `<stem>.h5` is a
  `FileNotFoundError`, and a door `--model-h5` that is not that
  `<stem>.h5` is refused.
- **Pairing.** The bound FEMData must be the `<stem>.h5` mesh. Node ids and
  coordinates must match exactly, and each element must have the same node
  set; element ids, type names and node order are ignored. A `fem=` or
  `bind()` override of another mesh is refused as an override. A `.ladruno`
  whose stamped `fem_hash` is not the `<stem>.h5`'s is refused with
  "do not pair".
- **The pack** is one spec: `mesh.png`, `contour.png`, `deformed.png`
  (auto-scaled: 0.12 of the model diagonal over the largest |u|), the
  app's `manifest.json`, and the matplotlib `history.png`. There is no
  `reactions.png`. Its contour is the starter component (uz, ux, uy, then
  the first nodal), even when that component is all zero.
- **Stored Gauss components** (e.g. `stress_xx`) draw as the app's element
  field (VA6e, #2794): the `.ladruno` token and its own column label
  (`stress` / `stress/sigma11`), averaged as v1 drew them (one Gauss point:
  `cell_averaged`; more: `node`, extrapolated to the corners). The app draws
  element results only from a file stamped `element_index_space = "fem"`.
- **Refused shots:** `view="reactions"`, a velocity or acceleration
  `deform`, a derived scalar such as `von_mises_stress`, and a component
  the file holds in more than one column.
- **Errors and cameras.** No installed app is an error (exit 2 /
  `FileNotFoundError`), never a skip; only `APEGMSH_SKIP_VIEWER=1` skips,
  for any source, in process and from `python -m apeGmsh.viewers render`.
  `camera xy|xz|yz` become the app's `top|-y|+x`.

Live session (needs an open Gmsh kernel; `from_h5` / closed →
`RuntimeError`): `g.model.render("geom.png")` (BRep, coarse tessellation
if no mesh), `g.mesh.render("mesh.png")` (live Gmsh, not `fem.render`).
Snapshot mesh stills stay on `fem.render`.

Do not call `g.model.viewer()` / `g.mesh.viewer()` / `results.viewer()`
/ `show_web` / `gui()` for diagnosis.

## Viewer stills — apeGmshViewer (ADR 0128)

The installed apeGmshViewer takes stills headlessly, drawn exactly as the
human sees the set, each with a manifest of the numbers behind it. One
call goes from the saved files to PNGs plus the manifest:

```python
from apeGmsh.agent_view import render_stills, AgentViewError

spec = {
    "spec": "apegmsh-view/1",
    "files": {"model": "frame.h5"},          # results found by pairing; relative = cwd
    "shots": [
        {"name": "mesh", "camera": "iso", "chrome": "model"},
        {"name": "uz-last", "stage": "gravity", "step": "last",
         "contour": {"field": "displacement", "component": "z"},
         "deform": {"scale": 10, "ghost": True},
         "caption": "Vertical displacement, gravity stage, last step."},
    ],
}
try:
    manifest = render_stills(spec, "out/stills")   # out: absent or empty
except AgentViewError as e:
    print(e)              # names every refusal: spec key, shot reason, stderr
    e.manifest            # this run's manifest, or None; drawn shots are in it
```

- The app is the only validator: an unknown key or value refuses the whole
  spec and names it (`shots[0].camera`); a stage, step or field the file
  lacks refuses that shot, and the other shots still run. Any refusal
  raises; the shim never returns a manifest with a refused shot, and
  never reads a `manifest.json` that was in `out` before the run.
- The executable is `$APEGMSH_VIEWER`, else the per-user install
  (`%LOCALAPPDATA%\Programs\apegmsh-viewer\apeGmshViewer.exe`). A
  `$APEGMSH_VIEWER` that names no file raises; it is never passed over.
- `chrome: "model"` draws the model alone, with the scale bar on any
  contour, for a report figure; `chrome: "app"` (the default) is the window
  with its panels. Batch stills default to the `light` theme, contours to
  jet. Colours and fonts are the app's; a spec cannot name one.
- Field and component names are the file's, as the legend shows them
  (`DISPLACEMENT/Ux` for a `.ladruno`). To write a spec that will not
  refuse, ask the set what it offers first:
  `apeGmshViewer --describe frame.h5 --out <empty dir>` writes
  `describe.json` (stages, steps, fields, groups); `find_viewer()` gives
  the executable.
- **Quote values from the manifest, never from the pixels** (ADR 0128 D3):
  each `shots[i]` entry has the resolved camera, the file hashes, stage,
  step and time, the contour range and its mode, `extrema` (value, node or
  element id, position), the no-data count, the deformed scale and every
  banner line. `png` paths are relative to `out`.
- **Voice (ADR 0128 D8).** Every caption, offer name and summary written
  from a manifest is in English, in the office voice (ape-tono):
  impersonal, declarative and complete, a value given beside its
  reference. "Peak vertical displacement −12.4 mm at node 57, gravity
  stage, last step; limit L/360 = 16.7 mm." Not "Looks like the max is
  around 12!". The app does not rewrite captions; the voice is the
  agent's duty.

## Finding catalog (v1 + Amendments 1, 4) → next action

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
| `RES.U_VS_DIAG` | info | ‖u‖_max / model diagonal. **Permanently** info — Amendment 4 declines a threshold by contract: legitimate dogfood runs span 5.3e-4 to 0.1245, so no fixed ratio separates a heavily-driven displacement/contact case from a diverging model. It is a number to read next to the extrema, not a gate. Skip `kind="mode"`. |
| `RES.ENERGY_ERR` | info / **warning** above 5% | Ladruno `ERR` in **%** at the last step. An **all-zero last-step frame** (KE, IE, DW, ULW, RES, ERR all exactly 0) is a Ladruno **static** run — the accumulator only populates on transients, so a zero reading is unpopulated-channel silence, not a verified balance; it lands in `skipped`, never a finding. A nonzero frame promotes to `warning` when last-step `|ERR| > 5.0` (`ENERGY_ERR_WARN_PCT` in `_catalog.py`), else stays `info`. Basis: the one healthy dogfood transient closes at 0.141% (35x margin); ~5% is conventional explicit-dynamics practice — single-datapoint, revisit by amendment. Native/MPCO → skip (`TypeError`), not OK. |
| `RES.ENERGY_NONFINITE` | warning | Last-step `ERR` is NaN/Inf. The one real energy signal. |

A check that cannot run is omitted, not scored OK. `report.skipped`
and the Verdict **Not evaluated (FAIL-reserved)** line name those
codes.

## Solver zone — `osm.assess()` (ADR 0094 Amendments 2, 3, 4)

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
| `RES.ZERO_U` | **warning** | `results=` passed with >=1 non-mode stage — the results are themselves the run evidence, no `analyze_call()` / pattern required. Last-step displacement exactly all-zero. Amendment 4 promotes info → warning: across six dogfood-assessed runs there was one true positive (a stranded-loads deck — exactly the bug this exists to catch) and zero false positives. Still not FAIL-reserved, still requires run evidence. Check the pattern actually drives the model — a free-vibration transient started from rest is a legal reading and the message already discloses it. On a zero-pattern deck the message cross-references `OSM.LOADS_UNIMPORTED` so the two findings read as one story. Listed in `skipped` when `results=` is omitted. |

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
