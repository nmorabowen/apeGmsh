# Engine Comparison — What's Actually Under Each

Not a plan — analysis. The fourth comparison axis. The earlier three docs covered
UI patterns (plotting, catalog, region tools) — this one is one layer down: the
**engine** that puts pixels on screen for both apps.

## TL;DR

We run on **the same engine** ParaView runs on: **VTK**. The differences live in the
*wrapper layer*, not the engine. ParaView wraps VTK with its own proxy / executive /
client-server / Qt stack. We wrap VTK with PyVista + pyvistaqt + a thin custom layer.

This means:

- **We can use any VTK feature ParaView can** — by reaching through PyVista.
- **We cannot match ParaView's organization** without writing our own proxy/executive
  layer (already decided not to — see [`00-overview.md`](00-overview.md)).
- **The "engine debate" is really a wrapper-and-organization debate.**

From `pyproject.toml`:
```
viewer = ["PySide6>=6.5", "pyvista>=0.43", "pyvistaqt>=0.11", "vtk>=9.2", "qtpy"]
```
That `vtk>=9.2` is the same dependency [`ParaView v5.13+ ships
with`](https://www.kitware.com/paraview-5-13-release/).

In the viewers, **30 files use PyVista, 7 reach directly into VTK**. We're already
operating in mixed-mode — the precedent for going around PyVista exists.

---

## Stack diagrams

### ParaView's stack

```
┌────────────────────────────────────────────┐
│  pq* (Qt UI, ParaView's own)               │  ← .ui files, behaviors, reactions
├────────────────────────────────────────────┤
│  Remoting (proxies, properties, sessions)  │  ← organization, persistence, IPC
├────────────────────────────────────────────┤
│  ServerManager + Views (state, layout)     │  ← server-manager wrappers
├────────────────────────────────────────────┤
│  vtkPV* (ParaView's VTK subclasses)        │  ← extends VTK for distributed cases
├────────────────────────────────────────────┤
│  VTK 9.x (C++ engine)                      │  ← the actual engine
├────────────────────────────────────────────┤
│  OpenGL / OSPRay / ANARI                   │  ← rendering backends
└────────────────────────────────────────────┘
```

ParaView is *six* layers from OpenGL to UI. Every layer is C++. Python wrappers are
auto-generated for the SM and Views layers.

### apeGmsh's viewer stack

```
┌────────────────────────────────────────────┐
│  apeGmsh.viewers.ui (Qt UI, our own)       │  ← Python, ~3000 LOC
├────────────────────────────────────────────┤
│  apeGmsh.viewers.diagrams / core / scene   │  ← Python, the "organization" layer
├────────────────────────────────────────────┤
│  pyvistaqt.QtInteractor (Qt embedding)     │  ← embeds a PyVista plotter in Qt
├────────────────────────────────────────────┤
│  PyVista (Pythonic VTK wrapper)            │  ← method-style API, numpy interop
├────────────────────────────────────────────┤
│  VTK 9.x (C++ engine, Python bindings)     │  ← the actual engine
├────────────────────────────────────────────┤
│  OpenGL                                    │  ← rendering backend
└────────────────────────────────────────────┘
```

Six layers — but ours has fewer C++ layers and more Python. The engine is the same;
the org layer is hand-rolled in Python.

---

## What each layer gives us

### VTK (the shared engine)

Both apps get this for free:

- Data structures: `vtkPolyData`, `vtkUnstructuredGrid`, `vtkImageData`,
  `vtkRectilinearGrid`, `vtkStructuredGrid`, `vtkMultiBlockDataSet`, `vtkTable`,
  `vtkHyperTreeGrid` and more.
- ~500 filter classes: `vtkContourFilter`, `vtkClipDataSet`, `vtkCutter`, `vtkGlyph3D`,
  `vtkTensorGlyph`, `vtkStreamTracer`, `vtkWarpVector`, `vtkProgrammableFilter`, …
- Rendering primitives: `vtkRenderer`, `vtkRenderWindow`, `vtkMapper`, `vtkActor`,
  `vtkLightKit`, `vtkCamera`.
- Implicit functions: `vtkPlane`, `vtkBox`, `vtkSphere`, `vtkCylinder`, …
- 3D widgets: `vtkBoxWidget`, `vtkPlaneWidget`, `vtkSphereWidget`, …
- Render passes: depth peeling, FXAA, SSAO, OSPRay/ANARI integration.
- I/O readers/writers for ~60 file formats.

**We have access to all of this.** PyVista exposes ~70% as one-liners; the rest is
reachable via `import vtk` and direct construction. The 7 VTK-direct files in our
viewers already demonstrate this pattern works.

### ParaView's organization layer (above VTK)

What ParaView adds:

- **Proxy/property system**: every VTK object gets a proxy wrapper with typed,
  validated, observable properties. Used for UI auto-generation, state serialization,
  client-server marshaling.
- **Executive-driven pipeline**: DAG of producer/consumer relationships; MTime-based
  invalidation; lazy update propagation.
- **Client/server framework**: data on a server (possibly N-process MPI), proxies on
  client, marshaled via `vtkClientServerStream`.
- **State serialization**: `.pvsm` XML format; reflects the proxy graph.
- **View layout** (`vtkSMViewLayoutProxy`): N-view tiling within one window.
- **Plugin system**: dynamic load of compiled .so/.dll with XML proxy registration.
- **Animation** (`vtkSMAnimationSceneProxy`): timeline + keyframes for camera and
  property changes.
- **Auto-generated UI from XML**: properties become widgets without per-property code.

That's a lot. **Most of it we explicitly decided not to port** — the client/server
split, the plugin system, the proxy/property infrastructure. See
[`00-overview.md`](00-overview.md) "Out of scope."

### PyVista's wrapper layer (above VTK)

What PyVista adds:

- **Method-style API**: `mesh.contour()`, `mesh.clip()`, `mesh.slice()`, `mesh.warp_by_vector()`
  instead of constructing filter objects.
- **NumPy interop**: `mesh.points`, `mesh.cell_data["array"]` return numpy views; no
  copying.
- **Plotter convenience**: `plotter.add_mesh(...)`, `plotter.show_grid()`,
  `plotter.add_scalar_bar(...)`, `plotter.add_axes()` — one-liners.
- **Common idioms baked in**: anti-aliasing toggle, parallel projection toggle, axis
  presets, screenshot.
- **Less Python ↔ VTK conversion ceremony**: PyVista handles the numpy ↔ VTK array
  marshaling.

PyVista does *not* add:

- Executive-driven pipeline (each method call executes inline; no caching).
- Auto-generated UI.
- State serialization.
- Multi-process.
- Render pass orchestration (you reach through to VTK).
- vtkProgrammableFilter integration (no Pythonic wrapper for user-script filters).

It's a *productivity* wrapper, not an *organization* wrapper. Different from ParaView's
Remoting layer in intent.

### Our own layer on top

What we add (the ~3000 LOC of `apeGmsh.viewers`):

- FEM-aware diagram kinds (Contour, vector glyph, line force, fiber section, layer
  stack, gauss marker, spring force, loads, reactions, section cut, time history).
- FEM-aware data extraction (selectors, slabs, snapshot binding).
- The active-objects / selection / pick-engine machinery (plans 04, 03).
- Theme + density + persistence wiring.
- OpenSees-tag ↔ FEM-eid mapping (`apeGmsh.cuts.FemToOpsTagMap`).

This is where our "Diagram = data + render" abstraction lives, the equivalent of
ParaView's Source + Representation pair but combined.

---

## Where we already reach through PyVista to VTK

The 7 files that `import vtk` already demonstrate the pattern:

```
src/apeGmsh/viewers/...
  diagrams/_section_cut.py     ← vtkPlane for two-tone face property
  diagrams/_gauss_marker.py    ← vtk sphere source for glyphs
  overlays/clip_plane_overlay.py ← vtkPlane attached to mappers
  scene/...                    ← vtk cell-type ID constants, mesh assembly
  core/pick_engine.py          ← vtkCellPicker tuning
  ui/_view_frame...            ← (TBD)
```

The pattern works. PyVista handles the common case; VTK direct handles the edge case
that PyVista hides or simplifies away.

---

## What VTK features we're under-using

Each of these is in our box (we already depend on VTK 9.2+) but we don't currently
exploit:

| VTK feature | What it does | Where it'd help us |
|---|---|---|
| `vtkTensorGlyph` | Principal-direction arrows or ellipsoids at points/cells | Filling the tensor-visualization gap (catalog Tier 1) |
| `vtkBoxWidget` + `vtkClipDataSet` | Interactive 3D box widget that clips to inside | Scope box feature (region-tools Tier 1) |
| `vtkPlaneWidget` + `vtkCutter` | Drag-the-plane cutting with field on the cut | Field-driven slice plane (region-tools Tier 1) |
| `vtkContourFilter` (3D mode) | Isosurfaces through a 3D scalar field | Catalog Tier 1 |
| `vtkBandedPolyDataContourFilter` | Stepped/banded contour lines | Catalog Tier 2 |
| `vtkStreamTracer` | Streamlines through vector field | Catalog Tier 3 |
| `vtkProgrammableFilter` | User-supplied Python computes a derived field | Derived-fields framework (catalog Tier 1) |
| `vtkRenderer.SetUseDepthPeeling` | Order-independent transparency | Plotting-comparison Win #1 |
| `vtkFXAAPass` | Cheap post-render AA | Plotting-comparison item 5 (skip per our analysis) |
| `vtkHardwareSelector` | Color-coded ID picking through transparency | Selection overlay in future/ |

**Nothing on this list requires changing engines.** All of it is in VTK 9.2, all of it
is one `import vtk` away.

---

## Alternative engines (and why we shouldn't switch)

For completeness — what else is out there for Python-side scientific 3D viz:

| Engine | Position | Why not for us |
|---|---|---|
| **Mayavi** | Pythonic VTK wrapper, predecessor to PyVista | Less actively maintained; PyVista has surpassed it |
| **K3D-Jupyter** | WebGL, browser-rendered | No Qt integration; we're Qt-native |
| **Plotly** | WebGL/SVG, browser-rendered | Different paradigm; weaker for mesh-heavy scenes |
| **Three.js / Babylon.js** | JavaScript, browser | Wrong language for our stack |
| **Polyscope** | Lightweight C++/Python, geometry-research focused | Limited filter library; no FEM idioms |
| **Trame** | ParaView's web-rendering framework, VTK-based | Optional for us — could add browser delivery later; doesn't replace the Qt viewer |
| **Vedo** | Another VTK Python wrapper | Smaller community than PyVista |
| **Open3D** | Geometry processing, not FEM-focused | Different intent |

The realistic competitor is **direct VTK** (skip PyVista, write everything yourself).
That's the path ParaView takes. Pros: full control, no wrapper overhead. Cons: 3–4×
the code, manual numpy↔VTK marshaling, lose PyVista's idioms.

**Verdict:** PyVista + reach-through is the right choice for our team size. We get
80% of the productivity at 20% of the cost.

---

## When to go around PyVista

Concrete decision criterion: **reach through to VTK when PyVista's abstraction is in
your way, not when it merely doesn't help.**

In-your-way examples (do reach through):
- Configuring depth peeling on the renderer (PyVista exposes anti-aliasing but not
  peeling).
- Building a `vtkTensorGlyph` (no PyVista equivalent).
- Attaching a `vtkBoxWidget` (PyVista has `add_box_widget` but limited).
- Setting up a `vtkProgrammableFilter` for derived fields.

Doesn't-help examples (don't reach through):
- Looping over actors to set visibility — PyVista's `mesh.actor` is fine.
- Reading point/cell data — `mesh.point_data[...]` is fine.
- Adding scalar bars — `plotter.add_scalar_bar(...)` is fine.

The rule of thumb: if you find yourself wishing PyVista exposed *one specific thing*,
just write the VTK call. Don't write a wrapper.

---

## What this means for the existing plan docs

No new plan items. Three existing ones already imply VTK-direct reaches:

- **Plan 06** (color-map editor): needs `vtkRenderer.SetUseDepthPeeling`.
- **Plan 01** (output dock): needs a custom `vtkOutputWindow` subclass (PyVista doesn't
  expose VTK's message routing).
- **Plan 08** (dock & layout): pure Qt, no VTK reach.

And every `future/` doc surfaced by the previous comparisons (`scope-box.md`,
`slice-plane.md`, `tensor-glyphs.md`, `derived-fields.md`, `isosurfaces.md`,
`clip-plane-everywhere.md`) is a VTK-direct project, leveraging filters and widgets
PyVista doesn't bridge.

So the engine choice is **already correct**. The remaining work is exploiting what's
already in the box.

---

## Honest summary

There's no engine to swap. We're running the same VTK 9.2 as ParaView, and we have
mechanisms in place to use any of it. The productivity wrapper (PyVista) is well-chosen
for our scale.

The "we're barebones" complaint is not "our engine is weak" — it's "we haven't yet
exploited what our engine can do." That's a much smaller problem.

The biggest single takeaway: **stop framing additions as "things ParaView has that we
don't" and start framing them as "VTK filters/widgets that exist and we haven't wired
up yet."** Most of our future/ doc list collapses to that.
