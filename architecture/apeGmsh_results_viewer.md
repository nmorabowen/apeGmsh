---
title: apeGmsh Results Viewer
aliases:
  - results-viewer
  - ResultsViewer
  - results.viewer
  - diagrams
  - viewer-rebuild
tags:
  - apeGmsh
  - architecture
  - visualization
  - results
  - pyvista
  - qt
---

# apeGmsh Results Viewer

> [!note] Companion document
> This file is the design charter for the **post-solve** viewer — what
> `results.viewer()` opens, how it consumes the [[Results_architecture|Results]]
> module, and how it renders the seven topology levels (nodes, elements,
> line stations, gauss points, fibers, layers, springs).
>
> Pre-solve review (geometry, mesh, constraints, loads, masses) is
> covered by `MeshViewer` / `ModelViewer` and is documented in
> [[apeGmsh_visualization]]. **Pre- and post-solve are deliberately
> two viewers** — same Qt+PyVista stack, same shared infrastructure
> under `viewers/{scene,core,ui,overlays}`, but different jobs and
> separately evolved.
>
> This document is the **directives** layer. The implementation
> phasing lives in `internal_docs/plan_results_viewer.md` (to follow).

---

## 1. Scope

### 1.1 What the results viewer is

A Qt + PyVista desktop viewer that opens against a [[Results_architecture|Results]]
file (native `.h5` or STKO `.mpco`) and renders post-solve quantities
on top of the bound `FEMData` mesh. Opened by:

```python
results = Results.from_native("run.h5")
results.viewer()                               # blocks until closed
results.viewer(blocking=False)                 # spawns subprocess
```

`results.viewer()` is the **only** results viewer entry point. The
matching `fem.viewer()` (pre-solve) opens `MeshViewer`; the two are
parallel, not unified.

### 1.2 What it is not

- **Not a notebook-inline tool.** Use `g.model.preview()` /
  `g.mesh.preview()` (Plotly) or matplotlib via `g.plot` for inline
  figures. The results viewer needs Qt; for Colab / headless runs,
  produce static figures or VTU exports.
- **Not a solver-agnostic file viewer.** It consumes `Results` objects.
  Reading raw `.vtu` / `.pvd` is dropped from the viewer surface (those
  remain export targets — `results.export.vtu(...)`).
- **Not the future scale-out path.** The frozen sibling package
  `apeGmshViewer/` is reserved for a future WebGL / Rust rewrite. See
  §10.

### 1.3 Phase boundary

| Concern                    | Pre-solve (`MeshViewer`)            | Post-solve (`ResultsViewer`)                     |
| -------------------------- | ----------------------------------- | ------------------------------------------------ |
| Substrate                  | live `FEMData`                      | bound `FEMData` from `Results.fem`               |
| Visual subjects            | mesh, constraints, loads, masses    | result slabs (7 topology levels) + diagrams      |
| Time                       | static                              | scrubber, play/pause, animation                  |
| Stages / modes             | n/a                                 | first-class — stage selector, modal browser      |
| State mutation             | edits PGs, picks selections         | read-only on data; mutates view state            |
| Entry                      | `g.mesh.viewer(fem=...)`            | `results.viewer()`                               |

Mesh-resolved overlays from the pre-solve viewer (constraint markers,
load arrows, mass spheres) **do not appear** in the results viewer.
A user who needs both reviews their model first with `g.mesh.viewer`,
runs the analysis, opens `results.viewer`. They are sequential tools.

---

## 2. The data shape

The viewer renders seven topology levels from
[[Results_architecture|Results]] §Layer 5. Each level has a slab type,
a natural rendering, and a selector grammar identical to `FEMData`'s
(`pg=` / `label=` / `selection=` / `ids=`). The seventh, springs, was
added in Phase 11d / 11e.

| Topology         | Slab type           | `values` shape       | Natural rendering(s)                                              |
| ---------------- | ------------------- | -------------------- | ----------------------------------------------------------------- |
| Nodes            | `NodeSlab`          | `(T, N)`             | volume / surface contour, deformed shape, vector glyphs           |
| Elements         | `ElementSlab`       | `(T, E, npe)`        | per-element-node force arrows, color-per-cell                     |
| Line stations    | `LineStationSlab`   | `(T, sum_S)`         | classic beam moment / shear / axial diagrams (hatched fill)       |
| Gauss points     | `GaussSlab`         | `(T, sum_GP)`        | sphere markers at GPs **and** interpolated continuum contour      |
| Fibers           | `FiberSlab`         | `(T, sum_F)`         | 2-D side-panel section plot **and** 3-D dot cloud                 |
| Layers           | `LayerSlab`         | `(T, sum_L)`         | shell mid-surface contour **and** through-thickness sub-panel     |
| Springs          | `SpringSlab`        | `(T, E)`             | force arrow along spring direction                                |

The viewer's renderer never sees raw `Results` reads — every slab is
fetched through the composite API (`results.nodes.get(...)`,
`results.elements.line_stations.get(...)`, etc.) so that lazy h5py
reads, partition stitching, and `(class_tag, int_rule)` grouping all
stay in the Results layer.

---

## 3. The Diagram abstraction

### 3.1 Why a Diagram is the right unit

Each result topology has a different rendering — beam diagrams are not
"color the mesh," fiber sections need a side panel, springs are
zero-length. A monolithic `set_scalar_field()` does not generalize.
What does: a **Diagram** — a stateful object that takes a slab spec and
the bound `FEMData`, produces VTK actors, and updates them on time
change.

Diagrams are first-class. They compose: showing **stress contour on
solids + bending moment on beams + fiber section on a picked column**
is three diagrams, not three special cases. Each has its own
selector, its own colormap, its own scale, its own visibility toggle.

### 3.2 Diagram protocol

```python
class Diagram:                        # composite-like (stateful)
    """A renderable layer driven by one or more Results slabs.

    Subclasses implement attach/update/detach. Constructed against a
    Results scoped to one stage; reads slabs through the composite API
    and keeps lazy h5py handles open via the bound reader.
    """

    id: str                            # unique within DiagramRegistry
    kind: str                          # see catalogue below
    style: DiagramStyle                # frozen dataclass — render params
    selector: SlabSelector             # frozen dataclass — pg/label/ids/...

    def attach(self, plotter, fem) -> None:
        """Build initial actors at step 0, register with plotter."""

    def update_to_step(self, step_index: int) -> None:
        """Refresh actors for a new step. Bounded mutation; no re-attach."""

    def detach(self) -> None:
        """Remove actors and release any cached arrays."""

    def settings_widget(self):
        """Return a Qt widget for the per-diagram settings tab."""
```

`DiagramStyle` is **per-diagram-type** (frozen dataclass record) — a
`ContourStyle` is not the same as a `LineForceStyle`. `SlabSelector` is
shared:

```python
@dataclass(frozen=True)
class SlabSelector:
    pg:        str | tuple[str, ...] | None = None
    label:     str | tuple[str, ...] | None = None
    selection: str | tuple[str, ...] | None = None
    ids:       tuple[int, ...] | None = None
    component: str = ""                # canonical name, never shorthand
```

The viewer never holds a slab in RAM longer than one step's render. On
time change, each diagram reads its slab for the new step (lazy h5py)
and mutates its actors in place. This keeps memory bounded for million-
DOF, multi-thousand-step files.

### 3.3 Diagram catalogue

Eight diagrams, one per natural rendering of the data shapes in §2.
The implementation phasing is in the plan; this is the target catalogue.

| Diagram                   | Slab(s) consumed                | Renders                                                                                                  |
| ------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `ContourDiagram`          | `NodeSlab` or interpolated `GaussSlab` | Per-cell or per-node colored mesh, scalar bar, optional deformed warp                                |
| `DeformedShapeDiagram`    | `NodeSlab` (translational + rotational components) | Warped mesh + undeformed reference + scale slider; optional contour overlay     |
| `VectorGlyphDiagram`      | `NodeSlab` (3 components)       | Arrows at nodes, colored / scaled by magnitude                                                           |
| `LineForceDiagram`        | `LineStationSlab`               | Hatched fill perpendicular to beam axis (axial / shear / moment / torsion); textbook engineering style   |
| `FiberSectionDiagram`     | `FiberSlab`                     | 3-D dot cloud at section locations + side panel: 2-D section plot, dots colored by fiber stress / strain |
| `LayerStackDiagram`       | `LayerSlab`                     | Shell mid-surface contour + side panel: through-thickness profile at picked GP                           |
| `SpringForceDiagram`      | `SpringSlab`                    | Force arrow along the configured spring direction at zero-length elements                                |
| `GaussPointDiagram`       | `GaussSlab`                     | Sphere markers at natural→global GP locations, colored by component value                                |

Two intentional non-diagrams:

- **Per-element nodal force arrows** (`ElementSlab` with
  `nodal_resisting_force_*` components) collapse into
  `VectorGlyphDiagram` driven off the per-element-node slab. Same
  visual; different data wiring.
- **Mode shapes** are not a separate diagram — they are
  `DeformedShapeDiagram` opened on a stage with `kind="mode"`. The
  Stage panel detects mode stages and offers an animation toggle.

### 3.4 What a Diagram is *not*

- Not a renderer. It does not own a `Plotter`. It receives one in
  `attach()`.
- Not session-aware. It takes a scoped `Results` (a stage) plus a
  selector. It never imports `gmsh.*`.
- Not pickle-serializable. State is the actor handles + cached arrays
  for the current step. Persistence is via `DiagramSpec` (record —
  frozen dataclass capturing `(kind, style, selector)`), saved /
  loaded by the Director.

### 3.5 Stack-neutrality

Diagrams take **slabs and a selector**, not a pre-rendered VTK mesh.
The transformation `slab + fem -> actors` lives entirely inside the
diagram. A future `WebGLRenderer` swap rewrites the renderer adapter,
not the diagram catalogue. This is the cheapest forward-compatibility
guarantee we can buy without paying for a full IR layer today.

---

## 4. ResultsDirector — single source of truth

### 4.1 What the Director owns

```
ResultsDirector  (composite)
├── results        — bound Results (stage-scoped)
├── stage_id       — active stage id
├── step_index     — active time index
├── time_mode      — single | range | envelope | animation
├── registry       — DiagramRegistry (ordered list of Diagrams)
├── on_step_changed       — observer
├── on_stage_changed      — observer
└── on_diagrams_changed   — observer
```

The Director is the **only** thing that knows what step / stage is
active. Diagrams are passive — they receive `update_to_step(i)` calls
when the Director's step changes. UI tabs subscribe to Director
observers and fire user actions back through Director methods.

### 4.2 Time modes

| Mode        | Step semantics                                   | Use case                                  |
| ----------- | ------------------------------------------------ | ----------------------------------------- |
| `single`    | One step rendered                                | Scrub the timeline, pick a moment         |
| `range`     | A step range — diagrams render envelope          | "Max stress over the dynamic stage"       |
| `envelope`  | Full stage — same as range over `[0, n_steps)`   | Capacity / demand checks                  |
| `animation` | Auto-advance step at fixed FPS                   | Visual review of a transient or mode      |

`range` / `envelope` modes change the slab read: each diagram's slab
becomes `(T, ...)` instead of `(1, ...)`, the diagram reduces over
time (max / abs-max / signed-extreme), and the actor renders the
reduction. Implementation detail of each diagram.

### 4.3 Stage navigation

```python
director.stage("gravity")               # switch to a stage by name/id
director.step(40)                       # jump to step 40
director.set_time_mode("animation")
director.next_mode()                    # for kind="mode" stages
```

Stage change calls `detach()` on every diagram, re-binds them to the
new stage's scoped Results, and `attach()` again. This is more
expensive than step change; it happens rarely.

For modal browsing (`kind="mode"` stages), the Director exposes
`results.modes` as the source set. The Mode panel (UI) iterates over
them; each mode is its own scoped Results so step semantics collapse to
`step_index=0`, `T=1`.

---

## 5. Selection — what gets data

The viewer reuses the **exact** selection vocabulary from
[[Results_architecture|Results]] §Layer 5. Every diagram is constructed
*with* a `SlabSelector`. Two independent dials:

- **What carries data** — the diagram's selector. PGs, labels,
  selection sets, raw IDs. Resolved through the bound `FEMData`.
- **What is shown** — the viewer's `VisibilityManager` (reused from
  `viewers/core/visibility.py`). Per-cell extraction; hidden geometry
  leaves no silhouette.

Concretely: a `ContourDiagram` selecting `pg="Body"` paints data on
Body's elements only, regardless of what's hidden. Hiding the slab via
the visibility manager makes those cells invisible; the data is still
there if revealed.

This separation matters for "show stress on the structure, hide the
ground but keep the foundation contoured" — two diagrams (structure +
foundation), one visibility filter (hide ground).

### 5.1 Pick-driven selection

The reused `PickEngine` + `SelectionState` (from `viewers/core/`) drive
two pick targets:

- **Element / node pick** — feeds the **Inspector** panel. Shows the
  element / node id, coordinates, all components currently bound by
  any diagram, and a one-click "create probe at this node" action.
- **Pick-to-Diagram** — pick an element, right-click → "Show line
  diagram for this beam" / "Open fiber section here." Constructs a
  Diagram pre-selecting the picked entity.

Box selection works the same way it does in `MeshViewer` — produces a
list of dim-tags or element IDs, stages them as a `SlabSelector` for a
new diagram.

---

## 6. UI surface

### 6.1 Window layout

Reuses `viewers/ui/viewer_window.ViewerWindow` — same QMainWindow
shell as `MeshViewer` / `ModelViewer`. Tabs differ.

```
┌───────────────────────────────────────────────────────────┐
│ Menu Bar                                                  │
├──────┬──────────────────────────────────────┬────────────-┤
│ Tool │                                       │  Tabs dock │
│ bar  │   VTK Viewport                        │  (right)   │
│      │                                       │            │
├──────┴──────────────────────────────────────┴────────────-┤
│ Time scrubber  ── play ── stage selector ── mode picker   │
├───────────────────────────────────────────────────────────┤
│ Status Bar                                                │
└───────────────────────────────────────────────────────────┘
```

The **time scrubber bar** is a dedicated dock at the bottom of the
viewport — not a tab. It carries the step slider, time-value readout,
play / pause / step / loop controls, stage dropdown, and (for modal
files) the mode browser. Always visible because time is the primary
axis.

### 6.2 Tabs

| Tab                  | Owns                                                         |
| -------------------- | ------------------------------------------------------------ |
| **Diagrams**         | Diagram list — add / remove / reorder / toggle visibility    |
| **Diagram settings** | Per-diagram styling — colormap, clim, scale, opacity, …      |
| **Selection**        | Active picked entity / box-selection results, "make diagram" |
| **Inspector**        | Picked element / node — id, coords, current values, history  |
| **Probes**           | Point / line / plane probes (mined from `apeGmshViewer/`)    |
| **Stages**           | Stage list with kind / n_steps / time range; mode browser    |
| **Visibility**       | `VisibilityManager` controls — hide / isolate / reveal       |
| **Session**          | Theme + per-session preferences (reused from MeshViewer)     |

The **Diagrams** tab is the spine of the workflow — adding a diagram
opens a small dialog (kind, slab selector, component, stage, initial
style) and the new diagram appears in the list. Toggling its visibility
mutes its actors without detaching. Removing it detaches and disposes.

### 6.3 Diagram-specific side panels

Two diagram kinds need extra real estate:

- **`FiberSectionDiagram`** — a docked side panel showing the 2-D
  section plot for the currently picked beam station. Re-renders on
  pick / time change. Backed by matplotlib-in-Qt (FigureCanvasQTAgg).
- **`LayerStackDiagram`** — a docked side panel showing the through-
  thickness profile (component vs `_thickness`) at the currently
  picked GP. Same matplotlib-in-Qt pattern.

Both panels live as dockable widgets so the user can detach them or
arrange them alongside the 3-D viewport. Closing the diagram closes
its panel.

---

## 7. Renderer integration

### 7.1 What we reuse from `viewers/`

The integrated viewer infrastructure already covers the lion's share
of what a results viewer needs. Reuse, don't fork:

| Module                                  | Used for                                                           |
| --------------------------------------- | ------------------------------------------------------------------ |
| `viewers/scene/mesh_scene.py`           | Substrate mesh — same Gmsh→VTK linearization, same dim batching    |
| `viewers/scene/glyph_points.py`         | Node cloud, gauss-point sphere glyphs                              |
| `viewers/core/entity_registry.py`       | DimTag ↔ cell-index maps; pick resolution                          |
| `viewers/core/pick_engine.py`           | Cell + box picking with modifier keys                              |
| `viewers/core/selection.py`             | `SelectionState` for picked entities                               |
| `viewers/core/color_manager.py`         | Hidden / picked / hovered / idle priority on the substrate mesh    |
| `viewers/core/visibility.py`            | Hide / isolate / reveal via `extract_cells`                        |
| `viewers/core/navigation.py`            | Quaternion orbit, pan, zoom (camera bindings)                      |
| `viewers/ui/viewer_window.py`           | QMainWindow shell, tab dock, toolbar                               |
| `viewers/ui/theme.py`                   | 10 palettes + theme editor + custom-theme loader                   |
| `viewers/ui/preferences_manager.py`     | 26-field persistent preferences                                    |
| `viewers/ui/preferences.py`             | Session tab (theme, point size, line width, edges, AA)             |
| `viewers/overlays/origin_markers_overlay.py` | World-origin marker overlay                                   |

What `ResultsViewer` adds on top:

- `viewers/results_viewer.py` — the top-level class
- `viewers/diagrams/` — the Diagram catalogue
- `viewers/ui/results_tabs.py` — Diagrams / Stages / Inspector tabs
- `viewers/ui/_time_scrubber.py` — bottom dock
- `viewers/overlays/probe_overlay.py` — mined from `apeGmshViewer/`

### 7.2 Color manager and diagrams — boundary

The reused `ColorManager` owns per-cell RGB on the **substrate** mesh
(idle / hidden / picked / hovered). Diagrams render to **separate
actors**; they do not fight the substrate ColorManager.

A `ContourDiagram` builds its own colored actor on a slice of the
substrate mesh (extracted by selector). Idle substrate stays flat. This
is the same pattern overlays use today — additive, not overlapping.

### 7.3 Origin shift, scale

`viewers/scene/mesh_scene.py` applies a numerical-stability origin
shift to large-coordinate models. Diagrams must respect
`registry.origin_shift` when placing glyphs at coordinates derived from
slabs (e.g., GP world coords from natural coords + shape functions).
Same convention as the existing origin markers overlay.

---

## 8. Probes

The probe system from `apeGmshViewer/visualization/probes.py` is mined
into `viewers/overlays/probe_overlay.py` with one structural change:
probes consume `Results` slabs through the Director, not VTU
`point_data` dicts.

| Probe         | Action                                                                   | Result                                                |
| ------------- | ------------------------------------------------------------------------ | ----------------------------------------------------- |
| **Point**     | Click on the mesh; samples all bound diagrams at the closest mesh node    | `PointProbeResult` — node id, distance, value-per-diagram |
| **Line**      | Pick A and B endpoints; samples N points along the line                  | `LineProbeResult` — chainage, value-per-diagram       |
| **Plane**     | Drag an interactive plane widget to slice the mesh                       | `PlaneProbeResult` — slice mesh + scalars             |
| **Time series** | Right-click → "plot history" on a picked node or element / GP            | `TimeHistoryResult` — time-vs-value matplotlib chart  |

The fourth (time series) is a new probe — `apeGmshViewer/` had no
notion of time. It piggybacks on the lazy slab read: pulling a single
node's full `(T,)` displacement is one h5py call.

Probe results display in the **Probes** tab and stay alive across
step / stage changes. A line probe seen during the gravity stage
remains visible after switching to dynamic, with values updated.

---

## 9. Subprocess opt-in

`results.viewer()` defaults to **blocking** — same as `MeshViewer` and
`ModelViewer`. The notebook user adds `%gui qt` once, or accepts that
the viewer blocks until closed.

`results.viewer(blocking=False)` is the documented escape hatch:

1. The Results file path is already known (the `Results` was opened
   from disk).
2. The viewer launches as `python -m apeGmsh.viewers.results <path>` in
   a subprocess.
3. The subprocess opens a fresh `Results.from_native(path)` (or
   `from_mpco`), constructs a `ResultsViewer`, and runs its own Qt
   event loop.
4. The notebook returns immediately. The viewer survives notebook /
   kernel crashes.

Implementation footprint: ~30 LOC. The subprocess machinery is the
*only* concession to the notebook workflow — everything else is
in-process.

If the Results was constructed in-memory (no file path — e.g., from a
domain-capture session that hasn't flushed), `blocking=False` raises
with a clear message: *"In-memory Results cannot launch in a subprocess.
Either pass blocking=True or call results.flush(path) first."*

---

## 10. The frozen `apeGmshViewer/` sibling

The top-level `apeGmshViewer/` package stays in the repository. It is:

- **Frozen.** No active development. Bug fixes only if it actively
  breaks under a dependency upgrade.
- **Detached from the dispatch path.** `results.viewer()` no longer
  spawns it. Nothing in `src/apeGmsh/` imports it.
- **A placeholder.** When the time comes for a genuine WebGL / Rust
  rewrite (better large-dataset handling, browser-deployable, no Qt
  dependency), the work happens here. The Diagram catalogue from
  §3.3 is the spec; the renderer adapter is what changes.

The 4,444 LOC currently in `apeGmshViewer/` is largely duplicated by
`src/apeGmsh/viewers/`. The mining ledger in §11 details what gets
salvaged for the integrated viewer; the rest stays where it is, for
the future rewrite to either keep or discard.

---

## 11. Mining ledger — what comes from `apeGmshViewer/`

Concrete file-by-file disposition. Quantified against the integrated
viewer to make the duplication explicit.

### 11.1 Lift (refactored against Results)

| Source                                         | LOC   | Destination                                       |
| ---------------------------------------------- | -----:| ------------------------------------------------- |
| `apeGmshViewer/visualization/probes.py`        | 605   | `viewers/overlays/probe_overlay.py`              |
| `apeGmshViewer/panels/probe_panel.py`          | 260   | `viewers/ui/_probes_tab.py`                       |

Both files import `apeGmsh.viewers.ui.theme` already; the refactor
swaps `MeshData.point_data["disp"]` reads for `Director.read_at_pick`
calls into the bound `Results`.

### 11.2 Pattern lift (rebuilt against slabs)

| Source / pattern                                  | What's kept                                                  | Rebuilt as                                                |
| ------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| `renderer.show_deformed` / `create_deformed_mesh` | Idea: warp mesh by displacement, optional undeformed ref    | `diagrams/_deformed_shape.py`                            |
| `renderer.show_vectors`                           | Idea: glyph orient/scale from a vector field                | `diagrams/_vector_glyph.py`                              |
| `renderer.set_active_time_step`                   | Idea: replace mesh data on step change without re-build      | `Diagram.update_to_step` contract                         |
| `controls.py` (time slider region)                | Idea: scrubber + play/pause + step / loop                   | `ui/_time_scrubber.py`                                    |

### 11.3 Discard

| Source                                       | LOC   | Reason                                                  |
| -------------------------------------------- | -----:| ------------------------------------------------------- |
| `apeGmshViewer/loaders/vtu_loader.py`        | 290   | Obsolete — Results is the canonical input               |
| `apeGmshViewer/ui/theme.py`                  | 895   | Older duplicate of `viewers/ui/theme.py` (1054 LOC)     |
| `apeGmshViewer/ui/preferences.py`            | 212   | Older duplicate of `viewers/ui/preferences_manager.py`  |
| `apeGmshViewer/main_window.py`               | 725   | Older duplicate Qt shell — `viewers/ui/viewer_window.py` is newer |
| `apeGmshViewer/panels/model_tree.py`         | 141   | Integrated `_browser_tab.py` + `_parts_tree.py` are richer |
| `apeGmshViewer/panels/controls.py` (most)    | ~150  | Colormap / opacity scaffolding — re-emit on integrated chrome |
| `apeGmshViewer/panels/properties.py`         | 131   | Replaced by Diagram-aware Inspector tab                 |
| `apeGmshViewer/visualization/renderer.py`    | ~400  | Most duplicates `mesh_scene` + `color_manager` + `visibility` |
| `apeGmshViewer/visualization/navigation.py`  | 9     | Trivially wraps the integrated install_navigation       |

**Total mined: ~865 LOC lifted, ~330 LOC of patterns rebuilt.**
**Total discarded: ~3,000 LOC of duplicate infrastructure.**

The discarded code stays inside `apeGmshViewer/` (frozen package, §10).
It is not deleted; it is simply no longer on the integrated viewer's
import path.

---

## 12. File structure

Target layout under `src/apeGmsh/viewers/` after the rebuild:

```
src/apeGmsh/viewers/
├── __init__.py
├── model_viewer.py                  (existing — pre-solve geometry)
├── mesh_viewer.py                   (existing — pre-solve mesh + overlays)
├── geom_transf_viewer.py            (existing — Three.js beam-axis tool)
├── results_viewer.py                NEW — top-level ResultsViewer
│
├── diagrams/                        NEW package
│   ├── __init__.py
│   ├── _director.py                 ResultsDirector (composite)
│   ├── _registry.py                 DiagramRegistry (composite)
│   ├── _base.py                     Diagram base class + DiagramSpec record
│   ├── _selectors.py                SlabSelector record + helpers
│   ├── _styles.py                   per-diagram DiagramStyle records (frozen dataclasses)
│   ├── _contour.py                  ContourDiagram
│   ├── _deformed_shape.py           DeformedShapeDiagram
│   ├── _vector_glyph.py             VectorGlyphDiagram
│   ├── _line_force.py               LineForceDiagram
│   ├── _fiber_section.py            FiberSectionDiagram
│   ├── _layer_stack.py              LayerStackDiagram
│   ├── _spring_force.py             SpringForceDiagram
│   └── _gauss_marker.py             GaussPointDiagram
│
├── ui/
│   ├── results_tabs.py              NEW — Diagrams / Stages / Inspector tabs
│   ├── _time_scrubber.py            NEW — bottom dock
│   ├── _diagrams_tab.py             NEW — list + add / remove / reorder
│   ├── _diagram_settings_tab.py     NEW — per-diagram style editor
│   ├── _stages_tab.py               NEW — stage list, mode browser
│   ├── _inspector_tab.py            NEW — picked entity + values
│   ├── _probes_tab.py               NEW — mined from apeGmshViewer
│   └── ... (existing — reused unchanged)
│
├── overlays/
│   ├── probe_overlay.py             NEW — mined from apeGmshViewer
│   └── ... (existing — reused unchanged)
│
├── scene/  ── existing, reused unchanged
└── core/   ── existing, reused unchanged
```

Outside the viewers package:

```
src/apeGmsh/results/Results.py
└── viewer()                         REPLACED — was subprocess to apeGmshViewer;
                                                now constructs ResultsViewer in-process

apeGmshViewer/                       FROZEN — placeholder for future stack swap
```

---

## 13. Class-flavor inventory

Following [[apeGmsh_principles]] §5 tenet (ix):

| Class / module                | File                                                  | Flavor      |
| ----------------------------- | ----------------------------------------------------- | ----------- |
| `ResultsViewer`               | `viewers/results_viewer.py`                           | composite   |
| `ResultsDirector`             | `viewers/diagrams/_director.py`                       | composite   |
| `DiagramRegistry`             | `viewers/diagrams/_registry.py`                       | composite   |
| `Diagram` (base)              | `viewers/diagrams/_base.py`                           | composite   |
| `ContourDiagram`, …           | `viewers/diagrams/_*.py`                              | composite   |
| `DiagramSpec`                 | `viewers/diagrams/_base.py`                           | record      |
| `SlabSelector`                | `viewers/diagrams/_selectors.py`                      | record      |
| `ContourStyle`, …             | `viewers/diagrams/_styles.py`                         | record (one per kind) |
| `PointProbeResult`, …         | `viewers/overlays/probe_overlay.py`                   | record      |
| Tab widgets                   | `viewers/ui/_*_tab.py`                                | def         |
| `_time_scrubber`              | `viewers/ui/_time_scrubber.py`                        | def         |

This matches the existing visualization surface — composites for
state-bearing classes, frozen records for the typed data they pass
around, defs for stateless UI factories.

---

## 14. Contributor rules

Six rules for adding to the results viewer surface, parallel to
[[apeGmsh_visualization]] §6:

1. **Diagrams take slabs and selectors, not pre-rendered VTK.** The
   transformation `slab + fem -> actors` lives inside the Diagram. A
   future renderer swap rewrites the actor side; the catalogue stays.

2. **Mutate state through the Director.** `step_index`, `stage_id`,
   `time_mode`, `registry` are the Director's. UI widgets call
   Director methods; they do not poke diagram internals.

3. **One slab per diagram per step.** Lazy h5py reads are the budget
   constraint. A diagram that needs 5 components at once reads 5
   slabs, releases them after `update_to_step` completes. No global
   slab cache.

4. **Diagrams subscribe to step changes; Director subscribes to UI.**
   Observer chain is `UI → Director → Diagrams`. Reverse calls are
   forbidden — a diagram does not call `director.set_step(...)`.

5. **Reuse from `viewers/scene,core,ui,overlays` always.** The results
   viewer adds Diagrams and stage/time UI; it does not reimplement
   theme, preferences, navigation, picking, visibility, or color
   management. New shared infrastructure goes under those existing
   directories.

6. **Lazy-import Qt.** Same rule as the rest of the viewer surface —
   `apeGmsh.results.Results` must remain importable on a headless
   kernel. `Results.viewer()` imports `viewers.results_viewer` inside
   the function body, never at module load.

---

## 15. Open items deferred to the plan

These are decisions that affect the implementation phases but do not
need to be settled at the directives level. They land in
`internal_docs/plan_results_viewer.md`:

- **Diagram auto-suggestion.** When a results file is opened, should
  the Director propose a default diagram set (e.g., contour on
  `displacement_z`, line forces on beams)? Or always start blank?
- **Style presets.** A `DiagramStyle` library (saved JSON, like custom
  themes) — let users save and re-apply favorite colormaps / clims.
  Useful but not v1.
- **Multi-pane synchronized views.** Two viewports of the same model
  with different selectors / different stages. Defer.
- **Animation export.** Render a stage to a `.mp4` / `.gif`. Useful
  but not v1.
- **Comparison mode.** Open two `Results` files side by side (e.g.,
  before / after a parameter change). Touches Director internals;
  defer.

---

## See also

- [[apeGmsh_visualization]] — the full visualization surface map
  (pre-solve viewers + viz/ inline tools)
- [[Results_architecture]] — Phase 9 explicitly defers this work to a
  separate plan; this document is the directives layer of that plan
- [[apeGmsh_principles]] — tenets (viii) "the viewer is core and
  environment-aware" and (ix) "three object flavors, three class styles"
- `src/apeGmsh/results/Results.py` — the API the viewer consumes
- `src/apeGmsh/viewers/mesh_viewer.py` — the parallel pre-solve viewer
  to crib structure from
