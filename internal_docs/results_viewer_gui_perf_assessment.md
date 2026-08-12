# Results Viewer — GUI Design & Performance Assessment

> **Date:** 2026-07-31  
> **Scope:** Desktop `ResultsViewer` chrome, layout, interaction design, and
> interactive performance (scrub, orbit, multi-diagram, concurrent geometries).  
> **Companions:**  
> - Inventory / rebuild options → `results_viewer_assessment.md`  
> - Fix–break / cascade disease → `results_viewer_adversarial_review.md`  
> **Method:** Code + existing plans/aesthetic docs + perf tests (no new profiling run).

---

## 0. Verdicts in one page

### GUI design

| Grade | Summary |
|---|---|
| **B+ architecture** | Correct desktop post-processor shell: outline · viewport · property docks · time scrubber; dock persistence; theme system; legend-as-view-annotation. |
| **B− product clarity** | Power-user dense. Geometry / composition / layer / visibility triality is coherent for authors of the system, hostile for occasional users. Dual creation paths, empty Details, probe/plot placeholders, and three “active” concepts leak. |
| **C consistency** | Results is ahead of mesh/model chrome; web is a different product; emoji tool buttons and B++-era comments drift from the live tree model. |

**GUI is not the main reliability problem** (cascade coupling is). GUI *is* a
discoverability and cognitive-load problem. Fixing chrome without freeze will
not stop “fix visibility breaks contour,” but freeze without GUI cleanup will
leave a capable engine that still feels hard to drive.

### Performance

| Grade | Summary |
|---|---|
| **B data path** | Visual store (float32, lazy, optional LRU budget), in-place scalar updates, scrubber throttle (33 ms), UI-lane coalesce, gesture_batch on eye cascades — real engineering. |
| **C− frame path** | Dual STEP/RENDER on scrub; auto-apply multi-fire without batch; MotionLOD only hides node cloud; no still/interactive quality for diagrams; restack = full detach/attach; glyph rebuilds whole PolyData. |
| **C memory policy** | Default unbounded cache; multi-component multi-stage megamodels can RAM-spike; first touch of a new component still pays full-time HDF5. |
| **B− measurement** | Solid micro-benches (`@pytest.mark.bench`); almost no end-to-end scrub FPS budget for “N diagrams × M nodes × concurrent geom” in default CI. |

**Performance and fix–break share a root:** extra pumps and dual render paths
make every gesture more expensive *and* more non-local. Cascade freeze (adversarial
Phase 0) is also a perf win, not only a correctness win.

---

## 1. GUI design — what the product looks like

### 1.1 Spatial layout (strength)

Documented shell (`ui/_results_window.py`):

```text
┌────────────┬─────────────────────────────┬───────────────────┐
│ Outline    │                             │ Plots / Diagram   │
│ (left)     │   3D viewport (central)     │ Geometry/Details  │
│            │                             │ Session (+ ext.)  │
├────────────┴─────────────────────────────┴───────────────────┤
│ Time scrubber (bottom)                                       │
└──────────────────────────────────────────────────────────────┘
```

This matches industry muscle memory (ParaView / STKO-ish: tree · view · properties
· time). Good decisions:

| Decision | Why it works |
|---|---|
| Viewport is **central immovable** | Avoids the “lost my 3D view in a tab” failure mode |
| Docks movable / floatable / tabifiable | Multi-monitor and report workflows |
| Layout persistence + schema version (`_LAYOUT_SCHEMA_VERSION = 5`) | Broken layouts discarded after dock set changes |
| `LAYOUT` metrics (`_layout_metrics.py`) | One place for min widths / scrubber height — was a real “crushed dock” bug class |
| Extension docks (`DockSpec`) | Output, color-map editor ride same View menu + sanitize path |
| OS title bar only | Avoided custom chrome that fought Windows/Qt |

**Grade: strong skeleton.** Do not rebuild layout from scratch.

### 1.2 Information architecture

Live hierarchy (outline + director):

```text
Stages (activate analysis stage)
Geometry 1..N
  └─ Composition 1..K
       └─ Layer (diagram) 1..M
Time (scrubber, not in tree)
```

That model is **correct for concurrent geometries** (ADR 0058) and matches how
power users stack contour + line force + reactions. It is also **three levels
deeper** than most users’ first need (“show me displacement”).

| Concept | User-facing name | Reality |
|---|---|---|
| Geometry | “Geometry N” | Scene instance + deform + offset + pin + substrate visibility |
| Composition | “Composition” | Stack gate (which layers paint) |
| Layer | Diagram kind | Contour / glyph / fiber / … |
| Layer eye | Intent visibility | Multiplied by composition gate × geom.visible × occlusion |
| Geometry eye | Geometry visible | Does **not** cascade layer intent (S2b) |
| Show mesh | Substrate fill/wire | Third visibility knob |

**Cognitive load:** “I turned the contour off and the mesh vanished” is a
**product design failure** as much as a batch-exit bug — occlusion + triality
with no status explanation.

**Outline tree module docstring still describes B++ groups** (Stages / Diagrams /
Probes / Plots). Live tree is geometry-centric. That drift is a maintainer and
agent trap.

### 1.3 Core interaction flows

#### Add a field (happy path)

1. `+ Add layer` (settings) or Add Diagram dialog  
2. Kind-first catalog (good: “I want contour,” not “I want nodes topology”)  
3. Data component filtered by file  
4. Attach → outline + card appear  

**Strengths:** kind registry drives dialog (no kind-list drift). Disabled kinds
with empty data is correct.  
**Friction:** modal dialog *and* inline create mode in settings tab; two
mental paths. Selector (pg/label/ids) is powerful and under-guided. Custom
expressions (ADR 0076) exist but are not first-class in the form (presets /
“define field” affordance missing).

#### Scrub time

Bottom scrubber: first/prev/play/next/last · slider · step · t · FPS · loop · export.

**Strengths:** transport is complete; drag coalesced (~33 ms); play respects
loop/bounce; stage change stops animation; export on scrubber is discoverable.  
**Friction:** stage selector lives in the outline, not next to time (split
attention). Combined “All stages” mode is advanced and under-explained. No
on-viewport time annotation (ParaView has one; we only have scrubber chrome).

#### Style a contour

Settings card stack + optional Color Map Editor dock + legend right-click.

**Strengths:** Apply / Auto-Apply (150 ms debounce) is the right ParaView-derived
pattern; default Auto-Apply **off** avoids surprise. Legend placement/session
(ADR 0081) is a real product differentiator.  
**Friction:** `_diagram_settings_tab.py` ~2k LOC — stack / create / single-select
modes; cards rebuild often; clim/cmap/opacity live in multiple places (card,
color editor, legend menu). New users do not know which surface is canonical.

#### Concurrent geometries

Geometry panel: deform field/scale, offset, stage pin, display.

**Strengths:** deep capability (side-by-side stages, ghost reference).  
**Friction:** no guided “Add comparison geometry” wizard; default second
geometry at same origin **z-fights**; offset is expert; pin vs active stage is
easy to misread; cosmetics (node cloud, line width) track **active** geom only.

### 1.4 Visual language & chrome

From `theme.py` + `apeGmsh_aesthetic.md`:

| Layer | Status |
|---|---|
| Qt chrome palettes (mocha / neutral_studio / paper) | Shipped, live switch |
| Viewport + chrome coupled by default | Correct per aesthetic charter |
| Results substrate colors on palette | Present (substrate_color, etc.) |
| Results aesthetic vs mesh soft shading | Intended inheritance; field dominates |
| Theme editor dialog | Exists for power users |

**Strengths:** Paper theme for figure capture; dark themes for long sessions;
stylesheet factory; theme subscribe for VTK refresh.  
**Weaknesses:**

- Toolbar actions use emoji glyphs (`⏪`, `▩`, …) — inconsistent across OS
  fonts; look unpolished next to ParaView/STKO tool icons.  
- Mesh/model viewers share theme module but **not** the same dock density /
  modernization level — product family feels uneven.  
- Preferences sprawl vs `LAYOUT` vs session JSON — three persistence
  namespaces for “how the window looks.”

### 1.5 Affordance & discoverability scorecard

| Affordance | Grade | Notes |
|---|---|---|
| Open results / multi-file | B | File menu + format sniff (strong) |
| Show displacement contour | B | Kind catalog; still multi-click |
| Deform shape | B− | Geometry panel, not a layer — correct but surprising |
| Play animation / export | A− | Scrubber-centric; good |
| Hide one part of model | C | Dim filter / element visibility / pick hide — scattered |
| Compare two stages | C+ | Concurrent geom + pin; powerful, unguided |
| Probe values | B− | HUD + overlay; B++ placeholders still in outline docs |
| Time-history plot | C | Plot pane exists; not first-class in outline |
| Keyboard pick N/E/G | B | Power-user good; no onboarding |
| Notebook path | D | Default blocking kills kernel; web thin |

### 1.6 Comparison to reference UIs (design only)

| Pattern | ParaView | apeGmsh results | Verdict |
|---|---|---|---|
| Pipeline browser | Source/filter tree | Outline geom/comp/layer | FEM-native better; denser |
| Properties panel | Auto from proxies | Hand-built cards | Ours more intentional, less scalable |
| Time controls | Animation View | Scrubber dock | Ours is adequate for FEM |
| Color map editor | Dedicated | Dock + legend + card | Feature-complete, **too many doors** |
| View layout multi-viewport | Strong | Single viewport | Gap for report figures |
| Empty states | Mixed | “No diagram selected” thin | Need guided empty state |

### 1.7 GUI strengths (keep)

1. Spatial shell + dock persistence + layout schema versioning.  
2. Kind-first layer model + registry-driven Add Diagram.  
3. Time scrubber transport + export entry point.  
4. Apply / Auto-Apply + debounce.  
5. Legend as view annotation (drag, format, session).  
6. Theme system coupling chrome and viewport.  
7. Layout metrics centralization (anti-collapse).  
8. Output dock for real errors (when not swallowed by pumps).  
9. Color map editor dock following active layer.  
10. Concurrent-geometry *capability* (even if UX is expert-only).

### 1.8 GUI weaknesses (fix)

| Priority | Issue | Why it hurts |
|---|---|---|
| **P0** | Visibility triality without feedback | Users “break” the picture and blame the app |
| **P0** | Empty / first-run experience | Open file → grey mesh → “now what?” |
| **P1** | Geometry / composition / layer jargon | Occasional users bounce |
| **P1** | Settings surface sprawl (card + editor + legend) | Same clim fixed three ways |
| **P1** | Outline docstring / Probes / Plots placeholders | Lies about product completeness |
| **P1** | Dual add-layer paths | Inconsistent muscle memory |
| **P2** | Emoji toolbar | Unprofessional; accessibility |
| **P2** | Details panel hollow | Wasted real estate |
| **P2** | No multi-viewport | Report workflows leave the app |
| **P2** | Mesh/model/results chrome parity | Family inconsistency |
| **P2** | Web GUI not a peer | Notebook users get a stub product |

### 1.9 GUI improvement options (do not confuse with cascade freeze)

**Tier G0 — clarity without new features (days)**

| Item | Change |
|---|---|
| G0.1 | Status-bar / toast when occlusion hides substrate, gate hides layer, pin holds stage |
| G0.2 | First-run empty state: “Add contour of displacement” one-click template |
| G0.3 | Rename user strings: Geometry→“View”, Composition→“Layer stack” (or keep terms but tooltip glossary) |
| G0.4 | Fix outline module docs; remove or implement Probes/Plots rows |
| G0.5 | Single primary “Add layer” path; demote secondary |
| G0.6 | Replace emoji toolbar with themed icons (or text) |

**Tier G1 — density & IA (1–2 weeks)**

| Item | Change |
|---|---|
| G1.1 | Collapse Diagram + Geometry settings into one contextual inspector driven by ActiveObjects |
| G1.2 | Stage control adjacent to scrubber |
| G1.3 | “Compare stages” preset: clone geom + pin + offset + reference ghost |
| G1.4 | Unified color controls: card owns numbers; editor is advanced; legend is view only |
| G1.5 | Visibility legend in outline (icons for gate/occlusion/geom) |

**Tier G2 — product chrome (later)**

Multi-viewport, richer plot pane, web parity shell, preference density modes
(compact / comfortable).

**Do not** start G2 before cascade freeze — multi-viewport multiplies pump bugs.

---

## 2. Performance — hot paths

### 2.1 Cost model of one scrub tick (intended vs actual)

**Intended (dispatcher matrix `step_changed`):**

```text
STEP  → each visible diagram: update_to_step (mutate scalars in place)
DEFORM → each visible geometry: compose points; sync diagram substrates
RENDER → one plotter.render()
```

**Actual (adversarial dual path):**

```text
Director.set_step
  → registry.update_to_step (pin-blind)     # STEP #1
  → plotter.render()                       # RENDER #1
  → observer → Dispatcher.fire(STEP)
       → _pump_step pin-aware              # STEP #2
       → _pump_deform                      # DEFORM
       → plotter.render()                  # RENDER #2
```

So scrub is **structurally ~2× paint + 2× step application** before any
diagram work. That is the single highest-leverage perf + correctness issue.

### 2.2 What is already good

| Mechanism | Role | Location |
|---|---|---|
| **VisualDataStore** | float32 full-time slabs; row slice on scrub; vmin/vmax once; lazy default; optional `APEGMSH_VIEWER_CACHE_BYTES` LRU | `diagrams/_visual_store.py` |
| **In-place `update_to_step`** | Contract: no re-add actors per step | `diagrams/_base.py` |
| **Scrubber drag coalesce** | 33 ms (~30 Hz) pending step | `_time_scrubber.py` |
| **UI-lane coalesce** | Outline full rebuild storms → ≤7 rebuilds/tick | `_outline_tree.py` (~6.86 ms noted on 10×20 matrix) |
| **gesture_batch** | N eye toggles → one GATE + one render | `_dispatch.py` + outline |
| **Auto-apply debounce** | 150 ms | `_diagram_settings_tab.py` |
| **MotionLOD** | Hide node cloud during camera; restore after 120 ms | `core/motion_lod.py` |
| **Depth peeling** | Transparency ordering | window + `OpacityController` |
| **DEFORM visual cols cache** | Column map memoized per (stage, component) | `results_viewer` deform reader |
| **Perf benches** | dispatcher, outline, settings, diagrams audit, visual store | `tests/viewers/test_*_perf*.py` |

This is **not** an unoptimized hobby viewer. The data path was thought through.

### 2.3 Hot path breakdown

#### A. Time scrub (primary interactive cost)

| Stage | Work | Scale |
|---|---|---|
| Slice deform field | visual store row or HDF5 | O(nodes) copy into float64 compose buffer |
| Compose `ref + offset + scale·u` | numpy | O(nodes) per **visible** geometry |
| Write `scene.grid.points` | VTK points | O(nodes) |
| Diagram STEP | contour: write scalar array; glyphs: often rebuild | O(active DOFs) / O(glyphs) |
| `sync_substrate_points` | copy deformed points into extracted submesh | O(extracted points) per contour |
| Render | full OpenGL frame | O(triangles + glyphs) × **~2 today** |

**Amplifiers**

- Second concurrent geometry doubles DEFORM.  
- Contour + line_force + reactions → N STEP bodies.  
- Auto-apply while scrubbing (if user leaves Auto-Apply on and edits) → interleaved DIAGRAM_MODIFIED storms.  
- First scrub of uncached component → full-time HDF5 load into store (stall).

#### B. Camera orbit

| What | Behavior |
|---|---|
| Node cloud | MotionLOD hides (good) |
| Diagram actors | **Full detail always** |
| AA / still-interactive | No ParaView-style interactive drop |
| Wireframe | Always on if show_mesh |

For large solids with multiple transparent contours, orbit is GPU-bound and
**unmitigated**. Mesh viewer profiling already flagged `extract_all_edges` as
cold-start cost; results reuses wireframe substrate similarly.

#### C. Style / clim edit

Auto-Apply debounced 150 ms → `DIAGRAM_MODIFIED` → STEP+DEFORM per layer.
Multi-card flush does **not** gesture_batch → **N renders** (adversarial).
Settings rebuilds are heavy (~2k LOC panel) but UI-lane coalesced.

#### D. Session restore / stage change

`reattach_all` + pin reattach + full STEP/DEFORM: cold path, expected to be
slow. Silent failures make “slow empty view” look like hang.

#### E. Restack (layer reorder)

Detach+attach **all** attached diagrams — O(all layers), destroys actor
identity (pick/legend side effects). Correctness hammer with poor perf profile.

### 2.4 Per-diagram cost classes (from code + perf audit intent)

| Kind | Per-step cost class | Notes |
|---|---|---|
| Contour (nodes) | Low–medium | Scalar write; substrate sync copies points |
| Contour (gauss extrap) | Medium–high | Extrapolation matrix work if not fully cached |
| Vector glyph | **High** | Audit: rebuilds glyph PolyData each step |
| Line force | Medium | Geometry rebuild of hatches; has dedicated perf test |
| Fiber section | Medium | Cloud + optional panel |
| Gauss markers | High if many GPs | Sphere glyphs |
| Reactions / loads | Medium | Glyph counts |
| Section cut | Low per step | Static quad (`update_to_step` noop) |
| Isochrone family | Medium–high | Map/strobe extra geometry |

### 2.5 Memory

| Policy | Behavior |
|---|---|
| Default visual store | Lazy per component; **unbounded** once loaded |
| `APEGMSH_VIEWER_EAGER=1` | Preload whole stage — dangerous on megamodels |
| `APEGMSH_VIEWER_CACHE_BYTES` | LRU eviction under cap; never drops last/hot entry |
| Concurrent geom scenes | Clone grids: points unique, topology shared (`clone_scene`) — good |
| Float32 store | Half of float64 public slabs — good for viz |

**Risk:** engineer opens multi-stage, multi-component earthquake run, touches
many fields → RAM climbs until OS thrash; no UI memory meter.

### 2.6 Measurement reality

| What we have | Gap |
|---|---|
| Unit benches (`-m bench`) | Not in default CI |
| Outline rebuild timing comments | One matrix size |
| box_wave profiling script | **Model/mesh**, not results scrub |
| Dispatcher storm bench | Mock pumps, not VTK |
| Dual-render cost | **Unmeasured in CI** but structurally certain |
| Target FPS for “portal frame + 3 diagrams” | Not codified |

**Recommendation:** one golden scrub benchmark (fixed H5, 3 diagrams, report
ms/step and render count) in CI with a soft budget — even if flaky on shared
runners, gate dual-render regressions via render-call spy.

### 2.7 Performance strengths (keep)

1. Visual store design (float32, lazy, clim sidecar-in-RAM).  
2. In-place step mutation contract.  
3. Scrubber + UI-lane + gesture_batch coalescing.  
4. MotionLOD pattern (extend coverage, don’t delete).  
5. Scene clone sharing connectivity.  
6. Perf test files as regression hooks.  
7. Optional cache budget env for HPC users.

### 2.8 Performance weaknesses (fix)

| Priority | Issue | Est. impact |
|---|---|---|
| **P0** | Dual STEP/RENDER on scrub | ~2× paint + wasted STEP; pin flicker |
| **P0** | Auto-apply multi DIAGRAM_MODIFIED unbatched | N renders during style drag |
| **P1** | MotionLOD only node cloud | Orbit jank with heavy diagrams |
| **P1** | Glyph / marker rebuilds per step | Scrub FPS cliff |
| **P1** | Unbounded visual cache default | RAM surprise |
| **P1** | First-component full-time load stall | Perceived “freeze” on field change |
| **P1** | Restack full detach/attach | Reorder / z-order expensive and identity-churny |
| **P2** | No still/interactive AA / decimation | Large model orbit |
| **P2** | DEFORM walks all visible geoms every step even if field unchanged | Multi-geom tax |
| **P2** | No in-app FPS / step-time HUD | Blind tuning |
| **P2** | Results scrub unprofiled vs model/mesh scripts | Priorities guesswork |

### 2.9 Performance improvement options

**Tier P0 — cascade freeze side effects (do with adversarial F1/F2)**

| Item | Effect |
|---|---|
| Single STEP path when dispatcher bound | −~1× STEP + −1× render per scrub |
| gesture_batch auto-apply flush | −(N−1) renders on multi-card commit |
| Complete batch exit (no double-fix patches) | Correctness + fewer ad-hoc extra syncs |

**Tier P1 — interactive quality (days–week)**

| Item | Effect |
|---|---|
| Extend MotionLOD to optional heavy diagram actors during camera | Orbit FPS |
| Still/interactive: disable MSAA / reduce glyph res while cam moves | Orbit FPS |
| Vector glyph: mutate points/scalars in place (no PolyData rebuild) | Scrub FPS |
| Prefetch visual store for active deform field + active contour component only | Kill first-touch stall |
| Default soft `APEGMSH_VIEWER_CACHE_BYTES` or status-bar RAM hint | Memory safety |
| Restack: reorder backend handles without full detach when possible | Layer reorder |

**Tier P2 — scale-out (later)**

| Item | Effect |
|---|---|
| Decimated LOD mesh for orbit (true ParaView-style) | Huge models |
| Worker-thread slab decode (careful with VTK affinity) | Scrub smoothness |
| Multi-viewport cost model | Only after single-view solid |
| Profile script for results scrub (sibling to box_wave) | Evidence-driven |

**Do not** pursue OSPRay/ANARI or client/server MPI viz until single-view scrub
is honest (one render, measured).

---

## 3. GUI × performance coupling (where they meet)

| UX choice | Perf consequence |
|---|---|
| Concurrent geometries default-visible | 2× DEFORM + z-fight → user adds offset → more complexity |
| Auto-Apply on dense cards | Easy to leave on → render storms while scrubbing |
| Node cloud “show nodes” for large FE | MotionLOD helps camera; scrub still pays cloud rebuild paths |
| Multiple scalar bars | Extra annotation actors every frame |
| Session restore full diagram set | Cold reattach wall; silent fail → empty slow open |
| Kind catalog with many layers | User stacks 5 diagrams → scrub cost linear in kinds (expected) |

**Product rule:** every new default-on visual must declare its scrub cost class
(low/med/high) in the kind registry — grey out or warn when stacking high-cost
kinds beyond a threshold.

---

## 4. Recommended sequencing (GUI + perf + cascade)

Align with adversarial Phase 0; do not run a pure GUI cosmetic sprint first.

```text
Phase 0  Cascade freeze (F0–F5)          ← correctness + free scrub ×2 win
Phase 0b Scrub render-count spy test     ← lock the win
Phase 1  P1 glyph in-place + MotionLOD expand + cache policy UI/default
Phase 1b G0 clarity (status, empty state, visibility feedback)
Phase 2  G1 inspector collapse + compare-stages preset
Phase 3  Catalog tools (threshold/slice/…) on stable graph
Phase 4  Multi-viewport / web shell parity
```

---

## 5. Scorecard

| Dimension | Score (A–F) | One-line |
|---|---|---|
| Layout architecture | **A−** | Right shell, persisted docks |
| Information architecture | **C+** | Powerful, jargon-heavy, triality |
| First-run / empty states | **D+** | Grey mesh, no guide |
| Time UX | **B+** | Scrubber strong; stage split |
| Color / legend UX | **B** | Feature-rich, too many doors |
| Concurrent geom UX | **C** | Expert-only |
| Visual polish | **B−** | Themes good; emoji toolbar weak |
| Family consistency (3 viewers + web) | **C** | Results ahead; web thin |
| Scrub data path | **B** | Visual store solid |
| Scrub frame path | **C−** | Dual render / dual STEP |
| Camera path | **C** | Node LOD only |
| Memory policy | **C+** | Tools exist, defaults unsafe |
| Perf measurement | **C+** | Benches yes; scrub golden no |
| **Overall product feel** | **B−** | Capable internal tool, not yet a calm product |

---

## 6. Bottom line

**GUI:** The window *layout* is good. The *language* of Geometry / Composition /
Layer / three visibilities is the real design debt. Invest in feedback,
empty states, and one inspector — not a Qt rewrite.

**Performance:** The *data* path is thoughtfully engineered. The *frame* path
is still paying for incomplete event adoption (double STEP/RENDER) and incomplete
interactive quality (LOD only on nodes, glyphs rebuild, unbounded cache). Fix
the dual path first; it is both the fix–break disease and free FPS.

**Together:** Cascade freeze → scrub budget test → glyph/LOD/cache → GUI clarity
→ then features. That order maximizes reliability *and* perceived speed *and*
learnability.

---

*End of GUI & performance assessment.*
