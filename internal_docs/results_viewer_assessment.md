# Qt Results Viewer — Strengths & Improvement Assessment

> **Date:** 2026-07-31  
> **Scope:** Post-solve desktop viewer (`results.viewer()` → `ResultsViewer`)
> and its shared stack under `src/apeGmsh/viewers/`. Web/trame and pre-solve
> model/mesh viewers are discussed only where they share fate with results.  
> **Tone:** Honest critique. Incremental polish through full rebuild are all
> on the table.  
> **Status:** Working assessment for product/architecture decisions — not a
> committed plan and not user-facing docs.
>
> **Sequencing supersession:** An adversarial multi-agent review
> (`results_viewer_adversarial_review.md`) found that the **dominant pain is
> fix–break cascade coupling**, not missing catalog items. **Do not follow
> §5 Tier A order for isosurface/scope/slice until the cascade-freeze program
> in the adversarial doc (Phase 0) is done.** Inventory and strengths below
> remain useful; priorities for *what to code next* live in the adversarial
> review.
>
> **GUI & performance:** See `results_viewer_gui_perf_assessment.md` (layout /
> IA / scrub & orbit cost model). Summary: shell layout is strong; Geometry–
> Composition–Layer jargon + visibility triality is the UX debt; data path
> (visual store) is solid; frame path still pays dual STEP/RENDER.

---

## 0. Executive summary

The results viewer is **not** a thin plotter wrapper. It is a domain-native
post-processor with real structural-FEM depth (fibers, line-force diagrams,
Gauss shape-function extrapolation, concurrent geometries, stage pins,
section-cut specs, isochrones). It has **strong written contracts** (read /
render / pick / event ADRs) and **uneven enforcement** — the matrix is real;
adoption is incomplete (Director still dual-pumps STEP/STAGE; `results_viewer.py`
is outside the G-RENDER AST guard; pumps swallow exceptions).

It is also **heavy, uneven, and hard to evolve**:

| Metric (approx., 2026-07-31) | Value |
|---|---|
| `viewers/` Python | ~55k LOC / ~163 files |
| Diagrams package | ~14k LOC |
| UI package | ~18k LOC |
| `results_viewer.py` alone | ~3.9k LOC / ~69 methods |
| `tests/viewers/` | ~31k LOC / ~133 files |
| `except Exception` hits (tree) | ~650 |
| `try:` blocks | ~760 |
| `@pytest.mark.qt` in default CI | **0** (15 marked, deselected) |

**Verdict in one line:** keep the *domain brain* and the *seams*; **freeze the
cascade** (dual STEP/RENDER, silent pumps, incomplete batch exit, unguarded god
shell) **before** catalog growth; then carve the shell; only then add region
tools. A full product rewrite only pays if you change the delivery vehicle —
not if you re-skin Qt+VTK to escape fix–break without finishing ADR 0056.

### 0.1 Fragility model (why fixes break other things)

```
Intent state  (often dual-homed: registry / AO / outline / actors)
      │  channels: Dispatcher  |  Director direct pump  |  Qt/AO
      ▼
Pumps (closures in results_viewer.show)  ── except:pass ──► empty/wrong scene
      │  restack detach/attach · pin-blind vs pin-aware STEP
      ▼
RENDER  ◄── also ad-hoc plotter.render in results_viewer (not G-RENDER-guarded)
      │
      ▼
Pick / outline / settings  ── re-enter ──► events
```

**Amplifiers:** concurrent geometries × N scenes; session restore bulk write;
batch exit skips RENDER-lane; multi-agent PRs; interaction graph mostly outside CI.

**Disease treatments (P0 engineering):** single STEP path when dispatcher bound;
complete batch-exit protocol; loud pumps + escalate-in-tests; extend G-RENDER to
`results_viewer.py`; pad session layer indices on skipped specs.

**Symptom treatments (P0 product, after or beside freeze):** notebook-safe
default; time envelope. **Not first:** isosurface / scope / slice (new graph nodes).

See `results_viewer_adversarial_review.md` for the full scoreboard and Phase 0 plan.

---

## 1. What it is today

### 1.1 Role

```
Results (broker chain: Results → OpenSeesModel → FEMData)
    └── results.viewer()  →  ResultsViewer  (Qt + PyVista desktop)
    └── results.show_web() / serve_web()  →  WebViewer  (trame; thinner)
```

Post-solve only. Pre-solve lives in `MeshViewer` / `ModelViewer` on the same
stack but different jobs (authoring vs exploration). That split is deliberate
and still correct.

### 1.2 Stack

```
┌─────────────────────────────────────────────────────────┐
│  ui/          Qt docks, outline, scrubber, settings     │  ~18k LOC
├─────────────────────────────────────────────────────────┤
│  results_viewer.py  +  diagrams/_director.py            │  orchestration
├─────────────────────────────────────────────────────────┤
│  diagrams/    Contour, line force, fiber, isochrone…    │  ~14k LOC
│  core/        pick, selection, LUT, visibility…         │  ~6k LOC
│  scene/ + scene_ir/   FEM substrate + pure IR           │  ~3k LOC
├─────────────────────────────────────────────────────────┤
│  backends/    PyVistaQtBackend  |  TrameBackend         │
├─────────────────────────────────────────────────────────┤
│  PyVista / pyvistaqt / VTK 9 / PySide6                  │
└─────────────────────────────────────────────────────────┘
```

Same **engine** as ParaView (VTK). Different **organization** layer (hand-rolled
Python vs ParaView's proxy/executive/C++ stack). See
`internal_docs/plans_viewer/engine-comparison.md`.

### 1.3 Shipped diagram kinds (registry)

Domain-facing kinds registered under `viewers/diagrams/`:

| Kind | Notes |
|---|---|
| `contour` | Nodal + Gauss (cell / extrapolated node paths) |
| `vector_glyph` | Resultant / per-axis arrows |
| `line_force` | Classic beam N/V/M diagrams |
| `fiber_section` | 3-D fiber cloud + section panel |
| `layer_stack` | Shell layers |
| `gauss_marker` | GP spheres |
| `spring_force` | Zero-length spring arrows |
| `loads` / `reactions` | Applied loads & reactions |
| `section_cut` | Spec visualization (engineering cuts live in `apeGmsh.cuts`) |
| `principal_glyph` | Principal directions (tensor gap partially closed) |
| `sand` | Sand / discrete-marker style |
| `isochrone_map` / `_profile` / `_strobe` | Arrival-time / wave-propagation family |

Deformation is **not** a diagram: it is per-geometry state (ADR 0058). That
was the right redesign.

### 1.4 Architectural contracts worth keeping

These are the assets. Do not throw them away in any rebuild.

| Contract | ADR | What it buys |
|---|---|---|
| Viewer is pure H5 / Results consumer | 0014, 0026 | No Gmsh kernel in the viewer; foreign formats plug in at the reader |
| Scene IR + `RenderBackend` | 0042 | Headless diagram tests; Qt and trame share diagrams |
| Pick / selection targets | 0045, 0047 | Pick meaning is domain, not VTK prop soup |
| Single owners + dispatcher (STEP/DEFORM/GATE/RENDER) | 0056 | One coalesced render per gesture; cascade batching |
| Concurrent geometries | 0058 | Side-by-side deform/reference/stage-pin scenes |
| User scalar expressions | 0076 | Fields like `von_mises_stress/250` appear in pickers |
| Legend as view annotation | 0081 | Scales are view state, session-persisted |

Machine-checked guards (`test_viewers_pure_h5_consumer.py`,
`test_diagrams_pure_no_pyvista.py`, `test_scene_ir_pure.py`, event allowlists)
are why multi-agent development has not completely shredded the design.

---

## 2. Strengths (be honest about what is already good)

### 2.1 Domain depth beats generic post-processors

For structural OpenSees work, the catalog is ahead of “dump VTU → ParaView”
on several axes:

- **Fibers** as a first-class diagram (ParaView needs Programmable Filter
  boilerplate).
- **Line-force** diagrams in textbook engineering style.
- **Gauss multipoint** shape-function extrapolation that respects element
  type (not `vtkCellCenters` one-point-per-cell).
- **Section cuts** as engineering integration specs (`apeGmsh.cuts` → force /
  moment histories), not only a cosmetic plane.
- **Isochrones** for wave / arrival-time workflows.
- **Reactions / loads / springs** as native kinds, not filter graphs.

The “barebones” feeling people report is usually about **missing generic VTK
filters** (isosurface, threshold, scope box) and **UX chrome**, not about
having too few diagram kinds. That distinction matters when prioritising.

### 2.2 Seams that actually work

- Diagrams emit IR; backends build VTK. Desktop and web share the domain
  stack in principle.
- Results slabs stay lazy (h5py); visual store float32-caches full-time
  slabs for scrub (code comments in `results_viewer.py` still say
  "float16" in places — stale; store is float32).
- Session JSON (schema v8) restores diagrams, geometries, pins, offsets,
  legend placement — with retired-kind migration, not silent crash.
- Action log (`viewers/_log.py`) + dispatcher categories make “attach the
  last session log” a real debug workflow.
- `Results.demo()` + `show_web()` give zero-setup tryout paths.

### 2.3 Event matrix exists; adoption is incomplete (post–ADR 0056)

Pre-0056 chronic bugs (eye icon bypass, composition gate no-op after backend
migration, write-only runtime flags) were the predictable failure mode of
“prose contracts.” The matrix + owner-fired events + `gesture_batch` are real
strengths.

**Incomplete adoption (adversarial finding):** Director still runs
`update_to_step` + `_render` *before* observers fire the dispatcher (dual
STEP/RENDER); `results_viewer.py` is **outside** G-RENDER; batch exit skips
RENDER-lane subscribers (session patches one of them). Treat “mature” as
**matrix mature, shell mid-migration** — that gap is the fix–break engine.

### 2.4 Concurrent geometries is a product differentiator

Stage pins, spatial offsets, reference ghosts, multi-geometry visibility,
and composition gates are unusually strong for an in-house Python viewer.
Side-by-side “gravity vs seismic stage” or “undeformed ghost + deformed
contour” is ordinary API surface, not a hero demo hack.

### 2.5 Test investment is real

~31k LOC under `tests/viewers/` plus headless IR assertions is unusually
serious. Full-window `@pytest.mark.qt` tests exist (and historically surfaced
real bugs when finally run with `pytest-qt`). That test base is a rebuild
cost and a keep-it asset at the same time.

### 2.6 Modernization wave partially landed

From `plans_viewer/00-overview.md` + code today:

| Plan | Intent | Status (code-level) |
|---|---|---|
| 08 Dock layout persistence | QSettings + View menu | Landed (results) |
| 01 Output dock | Tracebacks / logs | Landed (results) |
| 05 Apply / Auto-apply | Settings commit model | Landed |
| 04 ActiveObjects | Shared selection hub | Wired on all three viewers; still parallel |
| 06 Color map editor | Shared LUT dock | **Present** (`_color_map_editor.py`, toolbar toggle) |
| 03 Outline eye icons | Visibility column | Partially superseded by later outline work |
| 02 View-frame chrome | Extensible viewport toolbar | Still thin |
| 07 Property descriptors | Unified settings generation | Not justified yet |

So the “barebones chrome” critique is less true than it was in May 2026 —
but the **shell is still a wiring labyrinth**, not a finished product shell.

---

## 3. Weaknesses and structural debt

### 3.1 `ResultsViewer` is a god object

`results_viewer.py` (~3.6k LOC, ~69 methods) owns:

- QApplication / platform guards / live-viewer refcounting  
- Dock construction and ActiveObjects bridges  
- Dispatcher pump closures (DEFORM composition, GATE, substrate rebuild)  
- Probe / local-axes / HUD / scrubber / export animation  
- Session save/load  
- Side-panel sync  
- Color-map editor open paths  
- Geometry scene materialisation for concurrent geometries  

None of that is *wrong* individually. Together it means **every feature PR
touches the same file**, review cost is high, and the mental model for “where
does this gesture go?” is “read half of `show()`.”

**Smell class:** orchestration-by-closure. Pumps are nested functions closed
over dozens of locals. That was a reasonable way to land ADR 0058 quickly;
it is a poor long-term home.

### 3.2 UI mass vs domain mass

```
ui/        ~18k   ← largest package
diagrams/  ~14k   ← domain value lives here
core/       ~6k
```

`ui/_diagram_settings_tab.py` alone is ~2k LOC. Settings cards are
hand-built per kind. That is fine at 8 kinds; painful at 20. Plan 07
(property descriptors) was correctly deferred until a second concrete
consumer exists — but the **settings-tab tax** is already visible.

### 3.3 Silent degradation culture

~650 broad `except Exception` sites in `viewers/` is not “robustness.” It is
**bug camouflage**. The codebase itself has documented this class of failure
(ADR 0056: locked fallbacks indistinguishable from bugs; Gauss coords parked
at origin; etc.). Every new `try: … except Exception: pass` re-opens that
class.

Prefer:

- typed errors (`NoDataError`, already a good pattern),
- one aggregated warning per call,
- escalate-in-tests mode (already used successfully for Gauss coords).

### 3.4 Three viewers, one stack, three wirings

Model / mesh / results share `ui/`, `core/`, `scene/`, `Dispatcher`,
`ActiveObjects` — but each viewer’s `show()` re-implements:

- ActiveObjects construction and bridges  
- Dispatcher bind of *different* pump sets  
- Outline trees with parallel implementations  
- Dock registration patterns  

Results is the most advanced. Mesh and model lag by design and by
history. Shared *modules* without a shared *shell coordinator* means every
cross-viewer feature costs ×3.

### 3.5 Web path is a second-class product

`WebViewer` docstring is explicit: **Slice 1 is view-only**. No full
trame UI for layer management, limited picking (deferred), no parity with
dock/session/export. Notebook users get:

| Path | Reality |
|---|---|
| `results.viewer()` | Default `blocking=True` **kills Jupyter kernels** |
| `viewer(blocking=False)` | Subprocess; file-mediated rehydrate (good) |
| `show_web()` | Kernel-safe but thin |

This is documented, but documentation is not product design. The default is
still hostile to the most common modern workflow (notebooks).

### 3.6 Catalog holes vs industry baseline

From comparison docs (still largely valid for *missing* tools):

| Capability | Status |
|---|---|
| Surface scalar contour | Strong |
| Isolines / banded contours | Missing |
| Isosurfaces (3-D solids) | Missing |
| Threshold hide | Missing |
| Field-driven slice plane (drag) | Missing / partial (`section_cut` is tag/spec) |
| Scope box | Missing |
| Interactive clip (results mesh) | Clip plane is BRep/model-side |
| Range / envelope time modes | **Enum exists; non-SINGLE raises `NotImplementedError`** |
| Derived fields in-viewer | Partial — expressions on Results (0076), not full calculator |
| Multi-view tile layout | Missing |
| Motion LOD / still-vs-interactive quality | **Partial** — `MotionLOD` installed (results/mesh/model); results LODs node cloud only. Not ParaView still/interactive policy; depth peel exists via `OpacityController` / window |

You do **not** need ParaView parity. You need the five tools structural
engineers reach for weekly: **slice, threshold, isosurface, scope, envelope**.

### 3.7 Platform and dependency fragility

- Qt + VTK + pyvistaqt on Windows / Wayland / Jupyter is a known rough surface
  (platform guards, sphere-as-points GL bugs, subprocess escape hatches).
- Optional `[viewer]` extra is correct packaging; install discovery still
  trips people.
- Fiber dots: `render_points_as_spheres=True` drew zero pixels on some GL
  stacks (fixed by going flat) — class of “looks fine on my machine.”

### 3.8 Comparison / plan docs drift

Useful internal docs under `plans_viewer/` and `architecture/` are valuable
but **time-stamped snapshots**. Some claim multi-render per diagram or
“color map editor pending” after those issues moved. Treat them as
**hypothesis sources**, re-verify against code, or they become decision noise.

### 3.9 What the architecture deliberately refused

These are not bugs; they are costs of the chosen envelope:

- No ParaView-style proxy/property auto-UI  
- No client/server MPI viz  
- No plugin .so system for third-party diagram DLLs  
- Diagram still combines **data selection + representation** (no
  Source/Representation split)

That refusal keeps the codebase ownable. It also means **every new filter
is a hand-written diagram + settings card**, not a property XML drop-in.

---

## 4. Critique by severity

### P0 — Product risk (users hit this)

1. **Jupyter default is a footgun** (`blocking=True`).  
2. **Web viewer parity gap** forces either subprocess desktop or a thin 3-D
   view for notebook workflows.  
3. **Silent exceptions** hide real data/render failures as “empty scene.”  
4. **Missing envelope / max-over-time** for design checks (engineers live on
   envelopes for seismic).

### P1 — Velocity risk (team hits this)

1. **`results_viewer.py` size** — every feature is a merge conflict waiting.  
2. **Triple wiring** of ActiveObjects/Dispatcher across three viewers.  
3. **Settings-tab growth** without declarative properties.  
4. **Stale planning docs** → agents re-implement already-shipped work or
   re-open settled debates.

### P2 — Competitiveness gap

1. Scope box, interactive slice, isosurface, threshold.  
2. Still/interactive render quality (depth peeling alone is high leverage).  
3. Multi-view layouts for report figures.  
4. Calculator UX beyond ADR 0076 expressions (if users don’t discover
   `results.nodes.define`).

### P3 — Aesthetic / polish

1. Theme / scalar-bar / chrome consistency across three viewers.  
2. Preference surface sprawl.  
3. “Details panel is a near-empty placeholder” (own comment in
   `results_viewer.py`).

---

## 5. Improvement options (tiered)

### Tier A — Surgical (days–2 weeks each)

> **Superseded for sequencing.** Adversarial review: A5–A8 are **not** low-risk
> while cascade coupling remains. Do **Phase 0 freeze** in
> `results_viewer_adversarial_review.md` first (single STEP path, batch-exit
> RENDER-lane, loud pumps, G-RENDER on `results_viewer.py`, session index pad,
> interaction tests). Then A1/A2/A3. Then A5–A8.

| Item | Why | Risk until freeze | Effort |
|---|---|---|---|
| **A0′. Cascade freeze (F0–F5)** | Dual STEP/RENDER, silent pumps, incomplete batch, unguarded god shell | *This is the treatment* | 1–2 w |
| **A1. Notebook-safe defaults** | Detect Jupyter / existing QApp; default `blocking=False` or force `show_web` | Low | 0.5–1 d |
| **A2. Time envelope + range** | Finish Phase-6 `TimeMode`; max/min abs over steps | **Medium** (time path dual-pump) — after F1 | 3–5 d |
| **A3. Exception diet** | Ban silent pumps first; escalate-in-tests; tree-wide ratchet | *Disease* | ongoing, start 2 d |
| **A4. MSAA / quality knobs** | Depth peel **already exists**; still/interactive policy incomplete | Low | 0.5–2 d |
| **A5. Threshold filter diagram** | Weekly structural use | **High** until freeze | 2–3 d |
| **A6. Isosurface kind** | VTK contour on solids | **High** until freeze | 2–3 d |
| **A7. Scope box** | Box widget → cell visibility | **High** until freeze | ~3 d |
| **A8. Interactive field slice** | Drag plane cut (≠ engineering section_cut) | **High** until freeze | ~2–4 d |
| **A9. Expression discoverability** | Surface ADR 0076 in Add Diagram; presets | Low | 1–2 d |
| **A10. Plan-doc freshness stamps** | “Verified / superseded” on comparison docs | Low | 0.5 d |

**Suggested order:** A0′ → A3 → A1 → A2 → A9; catalog A5–A8 only after A0′ green.

### Tier B — Structural refactors (weeks–2 months)

Keep stack; make the codebase ownable again.

| Item | Shape | Effort |
|---|---|---|
| **B1. Carve `ResultsViewer`** | Extract `ResultsViewerShell` (docks), `PumpSet` (DEFORM/GATE closures), `SessionController`, `OverlayHost` — `results_viewer.py` becomes a thin assembler | 1–2 w |
| **B2. Shared `ViewerRuntime`** | One place constructs ActiveObjects + Dispatcher + OutputDock + layout persistence; model/mesh/results only supply pump bindings | 1–2 w |
| **B3. Representation split (optional)** | `DiagramSource` (selector + slab) vs `Representation` (style + IR). Enables wireframe+solid of same field without dual diagrams | 1–2 w |
| **B4. Declarative settings** | Property descriptors only if B3 or a second consumer appears; otherwise extract shared card widgets only | 3–5 d or skip |
| **B5. Web Slice 2** | trame time scrubber, layer toggles, stage select, pick readout — same director, real notebook product | 1–2 w |
| **B6. Headless screenshot API** | Stabilize non-interactive figure export for reports (partially exists via animation/export) | 2–4 d |

**Success criteria for B:** a new diagram kind requires **one class file +
registry decorator + tests**, and a new dock does **not** require a 200-line
edit in `results_viewer.show()`.

### Tier C — Rebuild / replace options

Only if Tier A/B cannot meet the product goal. Ranked by *fit to apeGmsh*,
not by fashion.

#### C1. Keep architecture, rewrite the shell (recommended if “rebuild”)

- **Keep:** Results readers, director, diagram kinds, scene IR, ADRs, tests
  that assert IR.  
- **Rewrite:** Qt windowing, dock layout, settings cards, maybe switch
  pyvistaqt → raw `QVTKRenderWindowInteractor` or a thinner host.  
- **When:** shell bugs dominate and B1 is not enough.  
- **Cost:** months; preserves domain moat.

#### C2. Web-first product (trame / VTK.js)

- Make `show_web` / `serve_web` the primary UI; desktop becomes optional.  
- Reuse director + diagrams + TrameBackend; invest in UI in browser.  
- **Pros:** notebooks, HPC remote viz, no kernel death, shareable links.  
- **Cons:** pick/interaction parity hard; engineering diagrams (line force,
  fibers) need careful WebGL work; offline desktop users regress.  
- **When:** primary users live in Jupyter / remote clusters.

#### C3. ParaView plugin or PVPython app

- Emit VTU/XDMF + custom filters, or a thin ParaView-based app.  
- **Pros:** instant generic catalog (clip, threshold, isosurface, client/server).  
- **Cons:** lose fibers/line-force/isochrone as first-class unless rebuilt as
  plugins; two ecosystems; branding/control; STKO-adjacent users may already
  have this path.  
- **When:** you want generic CFD-style post more than OpenSees-native
  diagrams.

#### C4. Blender / FreeCAD / CAD host

- FEM viz as an add-on.  
- **Pros:** beautiful rendering, animation, presentation.  
- **Cons:** terrible scientific scalar workflow; not an engineering
  post-processor.  
- **When:** marketing visuals, not design checks.

#### C5. Rust / native / WebGPU sibling

- Old charter reserved a frozen `apeGmshViewer/` idea for scale-out.  
- **Pros:** performance ceiling, ship binary.  
- **Cons:** rewrite everything that is currently the moat (diagrams +
  Results integration); multi-year.  
- **When:** commercial productization with dedicated team — not an internal
  engineering library roadmap.

#### C6. “Just use STKO / ParaView / GiD”

- Valid product answer for some users.  
- apeGmsh already has export paths; Results is the differentiator for
  *scripted* post and *native* OpenSees topology levels.  
- Killing the viewer entirely is only rational if maintenance cost exceeds
  its value to *your* daily workflows — measure that before ideology.

---

## 6. Rebuild decision matrix

| Goal | Prefer |
|---|---|
| Faster scrub / fewer ghost bugs | Tier A + B1 (not rebuild) |
| Notebook-first workflows | A1 + B5 or C2 |
| Match ParaView filter catalog | A5–A8 first; C3 only if still short |
| Hire-friendly codebase | B1 + B2 + A3 |
| Publish a polished commercial app | C1 or C5 with dedicated UI owners |
| Cut maintenance to near zero | C6 + export, accept feature loss |

**Default recommendation:** **no full product rebuild.**  
Execute **cascade freeze (adversarial Phase 0) first** (~1–2 weeks + tests),
then product P0 (notebook default, envelope), then B1/B2 under those tests,
then catalog. If shell carve fails twice under freeze tests, consider **C1
shell-only rewrite** — not ParaView/Rust. Domain diagrams and ADR seams are
the hard part; they already exist.

---

## 7. Proposed success metrics (if you invest)

Define “better” before coding:

| Metric | Baseline idea | Target |
|---|---|---|
| Time-to-first-contour (cold) | Measure on a fixed `run.h5` | No regression >10% |
| Scrub FPS (3 diagrams, mid model) | Profile once | Coalesced; no multi-render |
| `results_viewer.py` LOC | ~3600 | <1500 after B1 |
| Silent `except Exception` in viewers | ~650 | Trend down; zero in pumps |
| Notebook open path | Kernel death default | Safe default or forced safe API |
| Envelope mode | Missing | Works for nodal + gauss contour |
| New diagram kind PR size | Touches 4–6 files + god show() | 1 diagram module + tests |
| Web scrub + layer toggle | Missing | Present in `show_web` |

---

## 8. What I would *not* do

1. **Do not re-open the engine debate** (VTK is correct; engine-comparison
   already settled this).  
2. **Do not implement a ParaView proxy layer** in Python “for cleanliness.”  
3. **Do not unify pre- and post-solve into one megaviewer** — different
   jobs; shared runtime is enough.  
4. **Do not add diagram kinds without a real workflow** (catalog bloat is
   worse than holes).  
5. **Do not rewrite diagrams to chase web** until B5 proves the IR path
   carries them.  
6. **Do not delete session persistence** to “simplify” — schema v8 is
   working product memory.

---

## 9. Suggested narrative for the team

**Strengths to protect**

> Domain-native diagrams, Results pairing, Scene IR, dispatcher ownership,
> concurrent geometries, session restore, test/guard culture.

**Debts to pay**

> God-shell, silent exceptions, notebook defaults, unfinished time modes,
> missing region tools, thin web product, triple viewer wiring.

**Strategy**

> Polish the product surface (A), carve the shell (B), only change delivery
> vehicle (C) if the product audience moves (notebooks/HPC vs desktop).

---

## 10. Appendix — map of important files

| Path | Role |
|---|---|
| `viewers/results_viewer.py` | Desktop orchestrator (god object) |
| `viewers/diagrams/_director.py` | Stage/step/registry source of truth |
| `viewers/diagrams/_dispatch.py` | Event matrix + batching |
| `viewers/diagrams/_base.py` | Diagram protocol |
| `viewers/diagrams/_kinds.py` | Kind registry (anti-drift) |
| `viewers/diagrams/_session.py` | `.viewer-session.json` |
| `viewers/scene_ir/` | Pure render IR |
| `viewers/backends/pyvista_qt.py` | Desktop backend |
| `viewers/backends/trame.py` | Web backend |
| `viewers/web_viewer.py` | Notebook/web shell (thin) |
| `viewers/ui/_outline_tree.py` | Layer/composition/geometry tree |
| `viewers/ui/_diagram_settings_tab.py` | Per-layer settings mass |
| `viewers/ui/_color_map_editor.py` | Shared LUT editor |
| `viewers/ui/_time_scrubber.py` | Time UI + export button |
| `docs/design/results.md` | Canonical design write-up |
| `opensees/.../decisions/0056-*.md` | Event contract ADR |
| `opensees/.../decisions/0058-*.md` | Concurrent geometries ADR |
| `internal_docs/plans_viewer/` | Modernization + comparison analysis |

---

## 11. Open questions for you (product, not engineering)

These decide Tier A order and whether C2 is real:

1. **Primary habitat:** desktop scripts, Jupyter, or remote HPC browsers?  
2. **Envelope vs isosurface:** which missing tool costs more design time
   this month?  
3. **Is web parity a product goal or a notebook escape hatch?**  
4. **Accept maintenance cost** of three viewers, or force B2 this quarter?  
5. **Who is the viewer for?** daily engineer self-use vs client demos vs
   teaching notebooks — those three want different polish.

---

*End of assessment. Next concrete step if accepted: turn §5 Tier A into a
sequenced PR plan with done criteria, without expanding scope into C.*
