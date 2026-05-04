# ResultsViewer — Design Brief

> **Status:** this brief froze pre-v1.4 — see `architecture/apeGmsh_results_viewer.md` for current shell.

> Audience: a UI/UX designer with no finite-element-analysis background.
> Purpose: give them enough domain context, current-state grounding, and reference points to propose a layout for the post-solve results viewer.

---

## 1. What we're building

ResultsViewer is the **post-analysis exploration tool** inside apeGmsh, a Python library that engineers use to build and run structural simulations of buildings, bridges, and similar structures. After a simulation finishes, the engineer opens this viewer to *see what happened* — how the structure deformed, where it was stressed, where it failed.

Think of it as the "results dashboard" that comes after a long-running computation. Equivalent products in adjacent industries: ParaView, Abaqus/Viewer, ANSYS Mechanical post-processor. Our users typically come from those tools and have strong expectations about what should be where.

## 2. Who uses it

- **Structural / earthquake engineers** running nonlinear FEM simulations.
- Technical users — comfortable with Python, scripts, and CLIs — but they want the *visual exploration* part to feel direct and tactile, not scripted.
- Single-user, desktop, local files. No collaboration, no cloud.

## 3. How results get here (the pipeline)

The user's full workflow in apeGmsh:

1. **Build the model** — geometry, materials, supports, loads (Python API).
2. **Mesh it** — divide into finite elements (we wrap Gmsh).
3. **Solve it** — run OpenSees (a C++ FEM solver). This is the long step — minutes to hours.
4. **Read results** — the solver writes one or more files (HDF5 binaries, mostly). We wrap those into a `Results` object.
5. **Open the viewer** — `results.viewer()` launches what you're designing.

Three input file formats land in step 4 (the designer doesn't need to distinguish them — they all become the same `Results` object).

## 4. What "results" actually contain

For every **time step** of the simulation (think: animation frame), the solver records numerical values at:

- **Nodes** — points in 3-D space (e.g. displacement vectors, reaction forces).
- **Elements** — the volumes/lines/surfaces between nodes (e.g. internal forces, stresses).
- **Sub-element points** — fibers inside a beam cross-section, layers inside a shell, integration points inside a solid.

A typical run has thousands of nodes/elements and hundreds of time steps. The user wants to **slice this 4-D space** (3 spatial dims + time) along whatever axis matters for their question.

## 5. What the viewer shows

A 3-D scene with the meshed structure (gray) plus one or more **diagrams** painted on top. A diagram is a way of visualizing one quantity — e.g. "color every node by displacement magnitude" or "draw shear-force curves along every beam."

We currently support **8 diagram kinds**:

| Kind | What it shows |
|---|---|
| Contour | Heatmap on nodes/elements (e.g. stress field) |
| Deformed Shape | The structure warped to show how it actually moved |
| Line Force | Classic engineering hatched diagrams along beams (shear, moment) |
| Fiber Section | Dot cloud at sub-points inside a beam cross-section |
| Layer Stack | Heatmap on shells, with through-thickness detail |
| Vector Glyph | Arrows showing direction + magnitude at points |
| Gauss Point | Sphere markers at interior integration points |
| Spring Force | Arrows on connector elements |

Multiple diagrams can be active at once.

## 6. What the user does

1. **Open** a results file.
2. **Pick a stage** (a phase of the analysis — gravity load, then earthquake, etc.).
3. **Add one or more diagrams** — choose kind, choose what quantity, choose which subset of the model.
4. **Scrub time** — drag a slider, watch the structure animate.
5. **Probe** — click anywhere on the model to read off the exact value at that point. Three probe types: single point, line of N samples, plane slice.
6. **Inspect** — type a specific node ID, see its values at the current time, optionally pop open a **time-history plot** (one quantity vs. time) for that node.
7. **Style** — tweak color maps, scales, etc., per diagram.

## 7. Current layout (today)

Built in Qt + PyVista (embedded VTK 3-D viewport). The shell is:

- **Center**: 3-D viewport (the structure, central widget — immovable).
- **Left dock**: **Outline** — three-level tree (Geometries → Compositions/"Diagrams" → Layers).
- **Right docks**: **Plots**, **Details**, **Session** — diagram side plots, context-sensitive editor, and theme/session controls.
- **Bottom dock**: **Time Scrubber** — step slider, play/pause, stage dropdown, mode browser.
- **Top**: OS title bar + left vertical toolbar (camera presets, screenshot).
- All five side panels are `QDockWidget` instances — movable, floatable, tabifiable; layout persists across launches via `QSettings`.

Key file paths (for reference, not for the designer):

- `src/apeGmsh/viewers/results_viewer.py` — orchestration
- `src/apeGmsh/viewers/ui/viewer_window.py` — Qt window shell
- `src/apeGmsh/viewers/diagrams/` — the 8 diagram kinds
- `src/apeGmsh/viewers/overlays/probe_overlay.py` — probe interaction

---

## 8. How established tools solve this (reference)

The designer should look at three reference products. They share the same fundamental layout — **3-D viewport center, model/results tree on one side, time controls along the bottom** — and then differ in how they organize the *result objects* and the *probe/inspect* workflow.

### ANSYS Mechanical (Workbench)

- **Structure**: hierarchical **outline tree** on the left. The "Solution" branch holds every result object the user has added (Total Deformation, Equivalent Stress, Force Reaction, etc.). Each is a child node with its own settings.
- **Add a result**: right-click "Solution" → Insert → pick a result type. The new node appears in the tree; clicking it renders it in the viewport.
- **Per-result settings**: a "Details" panel sits **below** the tree. Selecting a tree node populates this panel with that result's options (scope, scale factor, color bar, etc.). One panel, context-driven by tree selection.
- **Time**: a "Graph" panel at the bottom shows a time/step axis; you click a point on it to set the active step. There's also a play/scrub toolbar.
- **Probe**: right-click on geometry → "Probe" creates a labeled, persistent annotation in the viewport. The label stays attached to the geometry as you orbit.
- **Style**: ribbon-style toolbar at the top with view, display, and result groups.
- **Mental model**: *every result the user cares about becomes a persistent, named tree node*. The tree is the project state.

### Abaqus/CAE Visualization module

- **Structure**: top-level **module switcher** (Part, Assembly, Step, Mesh, … **Visualization**). Visualization is its own module with its own toolbox.
- **Pick a quantity**: a "Field Output" toolbar at the top — variable dropdown (S, U, Mises, S11…), then component picker. Changing the dropdown re-paints the viewport. There's no "result tree" — only one field is rendered at a time.
- **Plot state**: separate buttons for Contour, Deformed, Vector, Symbol — toggled like view modes, not stacked as objects.
- **Time**: step + frame controls along the bottom, plus a separate Animate toolbar (time-history animation vs. scale-factor animation are distinct concepts).
- **Subsetting**: "Display Groups" — a side dialog where the user assembles a subset of the model (by element set, material, etc.) and plots only that.
- **Probe**: modal "Probe Values" dialog. User enters a picker mode, clicks geometry, values stream into the dialog as a list. Closing the dialog ends the mode.
- **Line probe equivalent**: a "Path" tool — define a polyline through the model, then "XY Data → from Path" extracts values along it into a separate 2-D plot window.
- **Time-history**: "XY Data Manager" — a separate plot window for line graphs, decoupled from the 3-D viewport.
- **Mental model**: *the viewport always shows one current state* (one field, one frame). Persistence lives in saved XY plots and Display Groups, not in stacked result objects.

### ParaView (open-source, closest to our stack)

- **Structure**: "Pipeline Browser" left — a stack of filters/sources, each with its own visibility toggle.
- **Per-item settings**: "Properties" panel below the pipeline; selecting a pipeline item swaps the panel's content.
- **Color/quantity**: variable + component dropdowns in a dedicated toolbar.
- **Time**: animation panel at the bottom.
- **Probe / line plot / time history**: all implemented as **filters** that produce new pipeline items, each with its own 2-D plot view. Probing a point creates a "Probe Location" filter; line probe creates a "Plot Over Line" filter, etc.
- **Mental model**: *everything is a node in a pipeline*. Probes, slices, and plots are all first-class persistent objects.

### What this means for our design

apeGmsh's current layout (right-dock tabs: Stages / Diagrams / Settings / Inspector / Probes) is **closest to Abaqus's modular approach** — each concern in its own panel — but our **Diagrams** concept is closer to ANSYS's tree-of-result-objects (multiple stacked, persistent, individually styled).

Tensions worth flagging to the designer:

1. **One quantity vs. many** — Abaqus shows one at a time and keeps the UI shallow; ANSYS stacks many and pays for it with a deeper tree. We allow many. Our right-dock tabs need to handle that without becoming a tree-within-a-tab.
2. **Probes as ephemeral readouts (Abaqus) vs. first-class objects (ParaView/ANSYS annotations)** — we're currently in the middle. Designer should pick a side.
3. **Per-diagram side panels** — neither ANSYS nor Abaqus has a direct equivalent. ParaView's "each filter gets its own view" is the closest. We need a story for these orphan 2-D plots.
4. **Diagram-add is currently modal** — both ANSYS (right-click insert) and ParaView (filter menu) handle this without a modal. Worth reconsidering.

---

## 9. Constraints to know about

- **Framework is locked**: Qt (PySide/PyQt) + PyVista 3-D viewport. We're not switching.
- **Single window** preferred; floating sub-windows OK if they help.
- **Desktop, mouse-first**, no touch, no mobile.
- **Theming** exists but is unfinished — the designer can propose a palette; a future "theme tweaker" feature is planned.
- The 3-D viewport must stay center stage — it's the product.

## 10. Open questions for the designer

- Should the right dock collapse into a single tree-driven panel (ANSYS-style) or stay as separate concern-tabs (current)?
- Are probes ephemeral readouts or persistent objects?
- How should per-diagram side panels relate to their parent diagram visually and spatially?
- Should "Add diagram" become a non-modal flow?
- Time scrubber: dedicated bottom dock, or fold into the toolbar with a popover?
