# Results Viewer — Implementation Plan

> [!note] Status
> Phases 0-6 complete (shipped through v1.4-v1.5). Post-phase
> additions: Composition/Geometry hierarchy, event-loop dispatcher,
> per-card Apply, per-Geometry display, applied-loads + reactions
> diagrams. See `architecture/apeGmsh_results_viewer.md` for current
> state.
>
> Phasing for the post-solve viewer rebuild. Companion to
> [[apeGmsh_results_viewer|the directives document]] — read that first.
> The directives are the architectural charter (what we build); this
> file is the order of operations (when each piece lands).
>
> Phase 9 of [[Results_architecture]] explicitly defers viewer planning
> here. Critical path for *that* plan ends at Phase 3 (MPCO reading
> through the composite API), which is now done — viewer rebuild can
> proceed.

---

## Build order rationale

Two ordering constraints drive the phasing:

1. **Lock the Diagram protocol early.** Every subsequent phase adds
   subclasses; if the protocol is wrong, every later phase pays.
   Phase 0 ships only the scaffolding — no data rendering — to keep
   the protocol-change cost cheap.
2. **First diagrams must validate the in-place mutation contract.**
   The single biggest performance win over `apeGmshViewer/` is mutating
   actor scalars in place instead of re-adding actors per step.
   Phase 1 ships `ContourDiagram` + `DeformedShapeDiagram` (both on
   `NodeSlab`) because they exercise this contract on the simplest
   data shape, with the most-used quantities (displacement / stress).

Subsequent phases climb the data-shape ladder: line stations →
fiber/layer side-panel diagrams → simpler glyphs → probes → polish.

---

## Top-level layout (target, from directives §12)

```
src/apeGmsh/viewers/
├── results_viewer.py                NEW
├── diagrams/                         NEW package
│   ├── _director.py
│   ├── _registry.py
│   ├── _base.py
│   ├── _selectors.py
│   ├── _styles.py
│   ├── _contour.py
│   ├── _deformed_shape.py
│   ├── _line_force.py
│   ├── _fiber_section.py
│   ├── _layer_stack.py
│   ├── _vector_glyph.py
│   ├── _gauss_marker.py
│   └── _spring_force.py
├── ui/
│   ├── results_tabs.py              NEW
│   ├── _time_scrubber.py            NEW
│   ├── _diagrams_tab.py             NEW
│   ├── _diagram_settings_tab.py     NEW
│   ├── _stages_tab.py               NEW
│   ├── _inspector_tab.py            NEW
│   ├── _probes_tab.py               NEW (mined)
│   └── _add_diagram_dialog.py       NEW
└── overlays/
    └── probe_overlay.py             NEW (mined)

src/apeGmsh/results/Results.py
└── viewer()                         REPLACED
```

---

## Phase 0 — Scaffolding + Director + scrubber

**Goal:** `results.viewer()` opens, renders the substrate mesh, time
scrubber wired, no diagrams. End-to-end skeleton with the protocol
locked. This is the biggest single-phase footprint (everything later
phases bolt onto).

### New files

| File | Purpose |
|---|---|
| `viewers/results_viewer.py` | `ResultsViewer` class — composes window + director + plotter; reuses `viewers/scene/mesh_scene.build_mesh_scene` and `viewers/ui/viewer_window.ViewerWindow` |
| `viewers/diagrams/__init__.py` | Re-exports `Diagram`, `DiagramSpec`, `SlabSelector`, the catalogue (initially empty) |
| `viewers/diagrams/_director.py` | `ResultsDirector` — owns stage_id, step_index, time_mode, registry; observers for step / stage / diagrams changes |
| `viewers/diagrams/_registry.py` | `DiagramRegistry` — ordered list, add / remove / reorder / toggle visibility |
| `viewers/diagrams/_base.py` | `Diagram` base class with `attach`/`update_to_step`/`detach`/`settings_widget`; `DiagramSpec` frozen record |
| `viewers/diagrams/_selectors.py` | `SlabSelector` frozen record + helpers (resolve through fem) |
| `viewers/diagrams/_styles.py` | `DiagramStyle` base + per-kind subclasses (placeholder; populated as diagrams land) |
| `viewers/ui/results_tabs.py` | Top-level tab assembly for the results window |
| `viewers/ui/_time_scrubber.py` | Bottom dock — step slider, time-value readout, play / pause / step / loop controls; `QTimer`-throttled to 33ms during drag |
| `viewers/ui/_stages_tab.py` | Stage list with kind / n_steps / time range; "Set active stage" action |
| `viewers/ui/_diagrams_tab.py` | Diagram list panel — empty button bar (Add / Remove / Up / Down) and tree; remains disabled until Phase 1 ships a diagram class |

### Modified files

| File | Change |
|---|---|
| `results/Results.py` | Replace `viewer()` body. Lazy-import `viewers.results_viewer.ResultsViewer`; construct in-process; `blocking=True` default. The `blocking=False` path is stubbed (raises `NotImplementedError("subprocess opt-in lands in Phase 6")`) so the signature is final from day one. |
| `viewers/__init__.py` | Re-export `ResultsViewer` |

### Tests

| File | Asserts |
|---|---|
| `tests/viewers/test_results_viewer_smoke.py` | `ResultsViewer(results=...)` constructs without raising; `.show(blocking=False)` exits cleanly when the test harness closes the window via `QTimer.singleShot` |
| `tests/viewers/test_results_director.py` | step_index transitions, stage switch detaches and re-attaches diagrams (against a stubbed Diagram), observers fire in correct order |
| `tests/viewers/test_results_registry.py` | add / remove / reorder / toggle on a registry of stub diagrams; `DiagramSpec` round-trips |
| `tests/viewers/test_results_selector.py` | `SlabSelector.resolve(fem)` returns expected node / element ID arrays for `pg=`, `label=`, `selection=`, `ids=` cases |

### Verification

- Open `tests/fixtures/results/cantilever_small.h5` (the fixture from
  Results Phase 1) via `Results.from_native(...).viewer()`. Mesh
  appears, scrubber is functional but does nothing visible (no
  diagrams). Window closes cleanly.
- Director observers proven via the registry test — adding /
  removing / step-changing fires the right callbacks.
- Performance baseline: viewer open → first frame visible in <500 ms
  on the cantilever fixture. Scrubber drag at 60 fps with no
  diagrams active.

### Performance contracts locked in this phase

These are the rules every later phase has to honor:

1. `Diagram.update_to_step(i)` must mutate existing actor scalars
   in place. Re-creating actors per step is forbidden — `Diagram`
   base class has no `add_actor` helper, only `mutate_array`.
2. `Diagram.attach(plotter, fem)` resolves selectors **once** to
   concrete IDs; selectors are not re-resolved on step change.
3. The Director coalesces N diagram updates into one
   `plotter.render()` call per step.
4. Scrubber drag uses `QTimer` 33ms coalescing; final render on
   release.

---

## Phase 1 — `ContourDiagram` + `DeformedShapeDiagram`

**Goal:** the two highest-value diagrams ship together. Both ride on
`NodeSlab`; share most of the slab-read + actor-mutation code path.
This phase validates that the protocol from Phase 0 actually performs.

### New files

| File | Purpose |
|---|---|
| `viewers/diagrams/_contour.py` | `ContourDiagram` — paints per-node or per-cell scalars onto the substrate mesh slice selected by `SlabSelector`. Supports nodal kinematics and (deferred to Phase 4) gauss-interpolated stress. |
| `viewers/diagrams/_deformed_shape.py` | `DeformedShapeDiagram` — warps a mesh copy by `displacement_xyz`; optional undeformed reference; scale slider. |
| `viewers/ui/_diagram_settings_tab.py` | Per-diagram styling — colormap, clim, opacity, show-edges, scale-slider for deformed |
| `viewers/ui/_add_diagram_dialog.py` | Modal dialog: pick kind, pick component, pick selector, pick stage. Opens on Diagrams tab "Add" button. |

### Modified files

| File | Change |
|---|---|
| `viewers/diagrams/__init__.py` | Re-export `ContourDiagram`, `DeformedShapeDiagram`, their style records |
| `viewers/diagrams/_styles.py` | Add `ContourStyle`, `DeformedShapeStyle` frozen records |
| `viewers/ui/_diagrams_tab.py` | Wire "Add" button to `_add_diagram_dialog`; enable now that two kinds exist |

### Tests

| File | Asserts |
|---|---|
| `tests/viewers/test_contour_diagram.py` | Attach to fixture, set step, scalar array on the mesh matches `slab.values[i]` for the selected node IDs; clim auto-fits when not explicit |
| `tests/viewers/test_deformed_diagram.py` | Warped mesh point coords equal `undeformed_coords + scale * displacement` for current step; scale-slider observer updates without re-attach |
| `tests/viewers/test_inplace_mutation.py` | After 100 step transitions, the actor's mapper is the same VTK object as after step 0 — no re-creation |
| `tests/viewers/bench_step_change.py` | (skipped in normal CI) Step-change time on 100k-node displacement fixture under 5 ms; on 1M-node under 50 ms |

### Verification

- Open cantilever fixture, add `ContourDiagram` on `displacement_z`,
  scrub the timeline — contour updates smoothly.
- Add `DeformedShapeDiagram`, drag the scale slider — undeformed
  reference stays in place, deformed mesh warps live.
- Both diagrams visible simultaneously, each with its own selector
  (e.g., contour on `pg="Top"`, deformed on `pg="Body"`).
- The benchmark proves the in-place mutation contract isn't violated.

---

## Phase 2 — `LineForceDiagram`

**Goal:** classic textbook beam diagrams (axial / shear / moment /
torsion) — no equivalent in `apeGmshViewer/`. This is the most novel
rendering and the most visible value-add for structural users.

### New files

| File | Purpose |
|---|---|
| `viewers/diagrams/_line_force.py` | `LineForceDiagram` — consumes `LineStationSlab`, generates per-beam perpendicular fill geometry in the section local frame, hatched style |
| `viewers/diagrams/_beam_geometry.py` | Per-beam local-frame construction (mining the OpenSees `vecxz` convention from `geom_transf_viewer.py`); pure functions returning local axes per element |
| `viewers/overlays/glyph_helpers.py` (modify) | Add `make_hatched_fill_strip(start, end, axis, magnitudes, hatch_spacing)` factory — reusable beyond beam diagrams |

### Modified files

| File | Change |
|---|---|
| `viewers/diagrams/_styles.py` | Add `LineForceStyle` (component, fill_color, hatch, scale, sign_convention) |
| `viewers/diagrams/__init__.py` | Re-export `LineForceDiagram` |

### Tests

| File | Asserts |
|---|---|
| `tests/viewers/test_line_force_geometry.py` | Per-beam local frame matches the OpenSees convention for canonical orientations (X-axis beam, Z-axis beam, skew); identical to `geom_transf_viewer` math |
| `tests/viewers/test_line_force_diagram.py` | For a known cantilever under tip load, the moment diagram is linear from peak-at-base to zero-at-tip; sign convention matches structural convention (sagging positive) |
| `tests/viewers/test_line_force_perf.py` | 1000-beam frame, step change <10 ms (hatch geometry built once at attach; only magnitudes mutate per step) |

### Verification

- Open cantilever fixture, add `LineForceDiagram` on `bending_moment_y`
  for the beam selection — hatched fill perpendicular to beam axis,
  amplitude proportional to moment, sign convention readable.
- Sign-flip toggle in the settings tab works.
- Multi-beam frame fixture: scrubbing the timeline animates all beam
  diagrams simultaneously without lag.

### Performance note

Hatch geometry is **built once at attach**; per-step updates only
mutate the magnitudes (which scale a precomputed unit-fill mesh). The
attach cost is `O(n_beams * hatch_density)` once; per step is
`O(n_beams * n_stations_per_beam)` numpy work plus one render.

---

## Phase 3 — `FiberSectionDiagram` + `LayerStackDiagram`

**Goal:** the two diagrams with side panels. Group them because they
share the matplotlib-in-Qt pattern and the "picked GP" state model.

### New files

| File | Purpose |
|---|---|
| `viewers/diagrams/_fiber_section.py` | `FiberSectionDiagram` — 3-D dot cloud at section locations; side panel renders 2-D section plot at the picked beam station |
| `viewers/diagrams/_layer_stack.py` | `LayerStackDiagram` — shell mid-surface contour; side panel shows through-thickness profile at picked GP |
| `viewers/ui/_section_panel.py` | Dockable Qt widget hosting `FigureCanvasQTAgg` for fiber section plot |
| `viewers/ui/_thickness_panel.py` | Dockable Qt widget hosting `FigureCanvasQTAgg` for through-thickness profile |

### Modified files

| File | Change |
|---|---|
| `viewers/diagrams/_styles.py` | Add `FiberSectionStyle`, `LayerStackStyle` |
| `viewers/diagrams/_director.py` | Add `picked_gp: tuple[int, int] | None` state field (element_id, gp_index); observers for fiber/layer panels to subscribe |
| `viewers/diagrams/__init__.py` | Re-export both |
| `viewers/results_viewer.py` | Wire dockable panels — they appear when the corresponding diagram is added |

### Tests

| File | Asserts |
|---|---|
| `tests/viewers/test_fiber_diagram.py` | Dot cloud has one point per `(element, gp, fiber_index)` tuple; section plot scatter coords match `slab.y, slab.z`; coloring by `fiber_stress` clipped to clim |
| `tests/viewers/test_layer_diagram.py` | Through-thickness profile `(stress, thickness)` for picked GP matches direct slab read; layer ordering matches MPCO sub_gp_index |
| `tests/viewers/test_section_panel_perf.py` | Picked-GP change → section panel re-renders in <50 ms on a 200-fiber section |

### Verification

- Open MPCO fixture with fiber-section beam-columns, add
  `FiberSectionDiagram`, click on a beam — side panel opens showing
  the 2-D section, fibers colored by current stress, scrubbing the
  timeline animates both 3-D dots and 2-D section.
- Open layered-shell fixture, add `LayerStackDiagram`, click on a
  shell GP — side panel shows the through-thickness profile.
- Closing a diagram closes its side panel; reopening restores it.

---

## Phase 4 — `VectorGlyphDiagram` + `GaussPointDiagram` + `SpringForceDiagram`

**Goal:** three simpler diagrams that share glyph-generation patterns.
Round out the catalogue to all eight kinds.

### New files

| File | Purpose |
|---|---|
| `viewers/diagrams/_vector_glyph.py` | `VectorGlyphDiagram` — arrows at nodes from a 3-component slab; doubles as the per-element-node force renderer |
| `viewers/diagrams/_gauss_marker.py` | `GaussPointDiagram` — sphere markers at GP world coords (computed via shape functions from natural coords); colored by component |
| `viewers/diagrams/_spring_force.py` | `SpringForceDiagram` — force arrow along the configured spring direction at zero-length elements |

### Modified files

| File | Change |
|---|---|
| `viewers/diagrams/_styles.py` | Add `VectorGlyphStyle`, `GaussMarkerStyle`, `SpringForceStyle` |
| `viewers/diagrams/__init__.py` | Re-export all three |
| `viewers/diagrams/_contour.py` | Extend to consume `GaussSlab` via shape-function interpolation (deferred from Phase 1) |

### Tests

| File | Asserts |
|---|---|
| `tests/viewers/test_vector_glyph.py` | Arrow length / orientation match `(values, scale)` for known synthetic vector field |
| `tests/viewers/test_gauss_marker.py` | GP world coords match `fem.element_shape_fn(natural_coords)` for tet / hex / quad on a unit-cube fixture; color matches slab values |
| `tests/viewers/test_spring_force.py` | Arrow on a zero-length spring element points in the configured direction; magnitude scales with `spring_force_<n>` |
| `tests/viewers/test_gauss_interp_contour.py` | Continuum stress contour interpolated from GPs to nodes matches MPCO STKO output to 1e-6 |

### Verification

- A fixture with both solid elements and zero-length springs renders
  GP markers, vector glyphs, and spring arrows simultaneously.
- Gauss-interpolated contour produces visually equivalent output to
  STKO for the same MPCO file.

---

## Phase 5 — Probes + Inspector

**Goal:** mine probes from `apeGmshViewer/`; add Inspector tab; add
the new time-history probe (no equivalent today).

### New files

| File | Purpose |
|---|---|
| `viewers/overlays/probe_overlay.py` | Mined from `apeGmshViewer/visualization/probes.py` (605 LOC). Refactored: probes consume Director slab reads instead of `MeshData.point_data` dicts. Result records (`PointProbeResult`, `LineProbeResult`, `PlaneProbeResult`) carry per-diagram values, not raw VTU fields. |
| `viewers/ui/_probes_tab.py` | Mined from `apeGmshViewer/panels/probe_panel.py` (260 LOC). Retheme against integrated theme system. |
| `viewers/ui/_inspector_tab.py` | NEW — picked element/node id, coords, current values from all bound diagrams, "create probe here" button |
| `viewers/diagrams/_time_history.py` | NEW — `TimeHistoryProbe` (not a Diagram; a probe overlay). Pulls a `(T,)` slab for the picked entity, displays in a matplotlib panel. Cached for the probe's lifetime. |

### Modified files

| File | Change |
|---|---|
| `viewers/results_viewer.py` | Wire Probes tab and Inspector tab into the right-side tab dock |
| `viewers/diagrams/_director.py` | Add `read_at_pick(node_id, component)` and `read_history(node_id, component)` helper methods — single-row slab reads |

### Tests

| File | Asserts |
|---|---|
| `tests/viewers/test_probe_point.py` | Click→sample returns expected nodal value at closest mesh node; per-diagram values keyed correctly |
| `tests/viewers/test_probe_line.py` | N samples along line return interpolated values matching VTK probe baseline |
| `tests/viewers/test_probe_plane.py` | Slice mesh has expected n_points; scalars match interpolation |
| `tests/viewers/test_time_history.py` | Time-history for picked node matches direct `results.nodes.get(ids=[node_id], component=...).values` |
| `tests/viewers/test_inspector_tab.py` | Picked entity update fires Inspector refresh; values reflect current step |

### Verification

- Pick a node, open time-history probe — matplotlib chart appears,
  shows component vs time, persistent across step / stage changes.
- Line probe across a fixture beam shows axial-force decay matching
  the analytical solution.

---

## Phase 6 — Polish — subprocess opt-in, animation, presets

**Goal:** the v1 finishing touches.

### New files

| File | Purpose |
|---|---|
| `viewers/__main__.py` | `python -m apeGmsh.viewers <path>` — argparse, opens `Results.from_native` or `from_mpco` based on extension, launches `ResultsViewer` blocking |

### Modified files

| File | Change |
|---|---|
| `results/Results.py` | Implement `blocking=False`: spawn subprocess. Raises clean error if Results was constructed in-memory (no `path`). |
| `viewers/ui/_time_scrubber.py` | Wire animation button — `QTimer` advancing step at user-set FPS (default 30); loop / bounce modes |
| `viewers/diagrams/_styles.py` | Add `StylePreset` save/load — JSON files in `<config>/apeGmsh/style_presets/`, loaded by Add Diagram dialog |

### Tests

| File | Asserts |
|---|---|
| `tests/viewers/test_subprocess_launch.py` | `results.viewer(blocking=False)` returns immediately; subprocess exit code 0 when window closed via test fixture |
| `tests/viewers/test_animation.py` | At 30 fps, 60 step animation completes in ≈ 2 s ± 200 ms |
| `tests/viewers/test_style_presets.py` | Save / load round-trip; named preset applies on diagram creation |

### Verification

- Notebook user opens `results.viewer(blocking=False)`, the kernel
  stays alive, the window appears, killing the kernel doesn't kill
  the viewer.
- Animation runs smoothly on the dynamic fixture (1000 steps).
- Save a "Stress contour, viridis, [-100, 100] MPa" preset; apply it
  in a fresh session.

---

## Cross-phase concerns

### Test fixtures

Build one per phase, accumulating into `tests/fixtures/viewers/`:

| Fixture | Phase | Contents |
|---|---|---|
| `cantilever_small.h5` (reuse from Results Phase 1) | 0 | 4-element beam, gravity stage, 10 steps |
| `frame_with_beams.h5` | 2 | 3-D portal frame, beams with line stations + fiber sections |
| `layered_shell.h5` | 3 | Plate with `LayeredShellSection`, transient with 50 steps |
| `mixed_solid_spring.h5` | 4 | Solid block on zero-length springs |
| `modal_frame.h5` | 6 | Modal stage fixture (10 modes) for animation testing |

A single `tests/fixtures/viewers/_generate.py` script generates all of
them via openseespy + apeGmsh. Committed to repo; rerun only when
schema or model definitions change.

### Performance gates per phase

Each phase ships with one benchmark test (skipped in normal CI, runs
under `pytest -m bench`). Numbers are gates, not goals — phases
violating them block:

| Phase | Gate |
|---|---|
| 0 | Viewer open → first frame <500 ms on cantilever; scrubber 60 fps with no diagrams |
| 1 | Step-change <5 ms on 100k-node displacement; <50 ms on 1M-node |
| 2 | Step-change <10 ms on 1000-beam line-force diagram |
| 3 | Side-panel re-render <50 ms on 200-fiber section |
| 4 | Gauss-interp contour <30 ms on 100k-GP fixture |
| 5 | Time-history probe load <100 ms for 1000-step / 1-node read |
| 6 | Subprocess launch <2 s wall time end-to-end |

### Theming and preferences

Phases 0+ reuse the existing theme + preferences systems from
`viewers/ui/`. No new persistent state is added until Phase 6 (style
presets — separate JSON dir, not in the `Preferences` schema).

### Lazy imports

Every viewer file imports Qt / pyvistaqt **inside function bodies**,
matching the pattern in `viewers/ui/viewer_window.py:_lazy_qt`. The
public API (`Results.viewer`) lazy-imports `viewers.results_viewer`
inside its function body. `import apeGmsh.results` must remain
Qt-free.

---

## Phase ordering check

Dependencies:

- 1 needs 0
- 2 needs 0 (independent of 1)
- 3 needs 0 (independent of 1, 2)
- 4 needs 1 (extends `_contour.py` with gauss interpolation)
- 5 needs 0 + at least 1 (Inspector / probes useful only with diagrams)
- 6 needs everything (subprocess wraps the whole stack)

**Critical path: 0 → 1 → 4 → 5 → 6.** That delivers the v1 with the
most-used diagrams (contour / deformed / vector / gauss / spring) +
probes + subprocess. **Phases 2 and 3 are parallelizable with 1 once
0 lands.** They can be picked up by a different contributor or
deferred without blocking v1 release.

If we slip, the ordering of risk vs. value says drop Phase 3 (fiber +
layer side panels — most novel UI, biggest implementation surface)
before Phase 2 (line force — smaller, higher-visibility feature for
structural users).

---

## Estimated phase footprint

Rough estimates to give a sense of scale, not commitments:

| Phase | LOC (src) | LOC (tests) | New files |
|---|---:|---:|---:|
| 0 — scaffolding + Director + scrubber | ~1500 | ~500 | 11 |
| 1 — Contour + Deformed | ~800 | ~600 | 4 |
| 2 — Line force | ~600 | ~400 | 2 |
| 3 — Fiber + Layer (side panels) | ~1200 | ~600 | 4 |
| 4 — Vector + Gauss + Spring | ~700 | ~500 | 3 |
| 5 — Probes + Inspector | ~1100 | ~400 | 4 |
| 6 — Polish (subprocess, animation, presets) | ~300 | ~200 | 1 |
| **Total** | **~6200** | **~3200** | **29** |

For comparison: existing `apeGmshViewer/` is 4,444 LOC (frozen);
existing `src/apeGmsh/viewers/` is 14,205 LOC (reused as-is). Net
new code in this plan: ~6,200 LOC of viewer + ~3,200 LOC of tests.

---

## Decision log (re-affirmed)

- **Greenfield rebuild.** No bridge from `apeGmshViewer/`. Mining
  ledger in directives §11 specifies what gets lifted (probes +
  patterns) vs. discarded.
- **Diagram protocol locked in Phase 0.** Every later phase is a
  subclass; protocol changes after Phase 0 are breaking.
- **In-place mutation contract enforced from Phase 1.** Re-creating
  actors per step is a bug, not an alternative implementation.
- **`blocking=True` default** matches `MeshViewer`/`ModelViewer`.
  Subprocess opt-in lands in Phase 6.
- **Side-panel diagrams (fiber / layer) are dockable Qt widgets**,
  not separate windows. Closing the diagram closes the panel.
- **`apeGmshViewer/` stays frozen.** No fixes, no features. The
  future WebGL/Rust rewrite owns it; nothing in this plan touches it.
- **Reuse over rewrite** for `viewers/scene`, `core`, `ui`,
  `overlays`. New code goes in `viewers/diagrams/` and
  `viewers/results_viewer.py`; everything else is shared.

---

## What lands first

The first PR off this plan delivers Phase 0:

1. `viewers/results_viewer.py` — class shell, opens window, builds
   substrate mesh via `build_mesh_scene`, attaches Director
2. `viewers/diagrams/_director.py`, `_registry.py`, `_base.py`,
   `_selectors.py`, `_styles.py` (skeleton)
3. `viewers/ui/results_tabs.py`, `_time_scrubber.py`, `_stages_tab.py`,
   `_diagrams_tab.py` (empty)
4. `results/Results.py` — `viewer()` replacement; `blocking=False`
   stub raises
5. The four Phase 0 test files

End state of the first PR: `Results.from_native(...).viewer()` opens
a window showing the mesh, with a working time scrubber that does
nothing visible (no diagrams), and a stage selector that switches
the active stage but still has no diagrams to update. The Diagrams
tab has an "Add" button that's disabled.

This is the smallest end-to-end slice that proves the architecture.
Phase 1 makes it useful.

---

## See also

- [[apeGmsh_results_viewer]] — directives (the *what*; this file is
  the *when*)
- [[Results_architecture]] — Phase 9 explicitly defers viewer planning
  here
- `src/apeGmsh/viewers/mesh_viewer.py` — closest existing analogue;
  crib structure for `ResultsViewer`
- `src/apeGmsh/viewers/scene/mesh_scene.py` — substrate mesh builder
  (reused unchanged)
- `apeGmshViewer/visualization/probes.py` — Phase 5 mining target
