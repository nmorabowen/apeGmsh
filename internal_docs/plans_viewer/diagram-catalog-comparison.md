# Diagram Catalog — What We Plot vs What ParaView Plots

Not a plan — analysis. [`plotting-comparison.md`](plotting-comparison.md) covered HOW
pixels land on screen. This one covers WHAT gets plotted, by FEM convention, organized
around the five categories you raised: fibers, gauss points, nodes, contours, averaging.

## Surprising finding up front

The catalog isn't actually barebones — it's broader than I'd assumed:

| Shipped today | Where |
|---|---|
| Contour (nodal & gauss) with explicit **averaged vs discrete** toggle | [`_contour.py`](../../src/apeGmsh/viewers/diagrams/_contour.py), [`_kind_catalog.py:292-300`](../../src/apeGmsh/viewers/diagrams/_kind_catalog.py) |
| Vector glyph (resultant + per-axis) | [`_vector_glyph.py`](../../src/apeGmsh/viewers/diagrams/_vector_glyph.py) |
| Line force diagrams (beams) | [`_line_force.py`](../../src/apeGmsh/viewers/diagrams/_line_force.py) |
| Fiber section (through-thickness panel + mid-surface contour) | [`_fiber_section.py`](../../src/apeGmsh/viewers/diagrams/_fiber_section.py) |
| Layer stack (shell layer aggregation: mid / mean / max-abs) | [`_layer_stack.py`](../../src/apeGmsh/viewers/diagrams/_layer_stack.py) |
| Gauss point markers (sphere glyphs, world-space sizing) | [`_gauss_marker.py`](../../src/apeGmsh/viewers/diagrams/_gauss_marker.py) |
| Spring force | [`_spring_force.py`](../../src/apeGmsh/viewers/diagrams/_spring_force.py) |
| Applied loads (force arrows) | [`_loads.py`](../../src/apeGmsh/viewers/diagrams/_loads.py) |
| Reactions (resultant + per-axis) | [`_reactions.py`](../../src/apeGmsh/viewers/diagrams/_reactions.py) |
| Section cut (OpenSees-tag-driven slice) | [`_section_cut.py`](../../src/apeGmsh/viewers/diagrams/_section_cut.py) |
| Deformed shape (as global view modifier on Geometry, not a layer) | [`_geometries.py`](../../src/apeGmsh/viewers/diagrams/_geometries.py) |
| Time history XY plots (separate 2D view pane) | [`ui/_time_history.py`](../../src/apeGmsh/viewers/ui/_time_history.py), [`ui/_plot_pane.py`](../../src/apeGmsh/viewers/ui/_plot_pane.py) |

So the "barebones" feeling isn't about *count of diagram kinds* — it's about depth and
derived-field computation. More on that at the end.

---

## Side-by-side: your five categories

### 1. Fibers

| | apeGmsh | ParaView |
|---|---|---|
| Native concept? | **Yes** — `fiber_section` kind with through-thickness scatter panel + mid-surface contour | No — fibers are not a ParaView primitive |
| What user does | Pick `fibers` topology + component → diagram renders | Build a polydata at fiber points via Programmable Filter; glyph it; color it |
| Strength | FEM-aware: knows about section types, fiber count, material assignment | Generic — works for any per-point dataset |
| Weakness | Hardcoded behaviors (no expression-based "color by f(σ, ε)") | Multiple manual steps for the common case |

**Verdict:** We are **ahead** here. A FEM-specific built-in beats Programmable Filter
boilerplate for the common workflow.

**Gap:** No way to plot derived fiber quantities — e.g. "yield ratio" `|σ| / σ_y` per
fiber, or principal stress at a fiber in a shell layer. Would need either hardcoded
options on `FiberSectionStyle` or a small expression evaluator.

---

### 2. Gauss points

| | apeGmsh | ParaView |
|---|---|---|
| Marker rendering | `gauss_marker` kind — sphere glyphs sized off model diagonal | `vtkCellCenters` + `vtkGlyph3D` with sphere source |
| Per-element GP count handling | Per-element shape-function aware extrapolation ([`_contour.py:499-520`](../../src/apeGmsh/viewers/diagrams/_contour.py)) | Generic — assumes cell-center; multi-GP needs a custom filter |
| Coloring by GP scalar | Direct from `gauss` composite; tensor suffixes silently skipped ([`_kind_catalog.py:113`](../../src/apeGmsh/viewers/diagrams/_kind_catalog.py)) | Same array picker as any other field |

**Verdict:** We are **ahead** for FEM-correct shape-function extrapolation. ParaView's
`vtkCellCenters` only gives one point per cell; ours handles e.g. 4-point Gauss
quadrature properly via the element's shape function.

**Gap:** **Tensor visualization at Gauss points is the elephant in the room.** We
explicitly skip tensor suffixes in the catalog. So when a recording has
`stress_xx/yy/zz/xy/yz/xz`, the user can color by *each component* but cannot see
*principal directions*. ParaView's `vtkTensorGlyph` (ellipsoid / cross / arrow) would
fill this gap — and it's the single most-requested feature in any FEM postprocessor for
solid mechanics.

---

### 3. Nodes

| | apeGmsh | ParaView |
|---|---|---|
| Per-node scalar coloring | `contour` kind with `topology="nodes"` — directly paints `scene.grid` point_data | Source has point_data; rep's `ColorArrayName` picks the array |
| Per-node vector glyphs | `vector_glyph` kind — resultant or per-axis arrows | `Glyph` filter with arrow source, `Scale by` = vector array |
| Per-node visibility / hide | Per-Diagram visibility; node markers via `_gauss_marker`-style spheres if needed | Plain — every node visible in wireframe; vertex glyph filter for explicit dots |
| Tensor at a node | Not exposed | `vtkTensorGlyph` |

**Verdict:** Roughly **equivalent** for scalars and vectors. Same tensor gap as gauss
points.

---

### 4. Contours

| | apeGmsh | ParaView |
|---|---|---|
| Surface coloring by scalar | `contour` kind, color the existing surface | Same — `ColorArrayName` on representation |
| **Isolines** (banded constant-value curves on a surface) | **Missing** | `vtkBandedPolyDataContourFilter` |
| **Isosurfaces** (equi-value surfaces through a 3D scalar field) | **Missing** | `vtkContourFilter` (confusingly named — produces isosurfaces in 3D) |
| Cutting planes through volume | Partial — `section_cut` is OpenSees-tag-driven, not field-driven | `vtkCutter` + plane widget — drag a plane through the model, contour on the cut |
| Threshold (hide cells outside a range) | Not exposed | `vtkThreshold` |
| Banded contours (stepped colormap) | Indirect via discrete LUT | Native via `vtkBandedPolyDataContourFilter` |

**Verdict:** Surface coloring is equivalent. **Three real gaps:** isolines/isosurfaces,
field-driven cutting planes, and threshold filtering. All three are standard FEM
postprocessor features — they belong on the catalog roadmap.

The terminology confusion to flag: in ParaView, "Contour" means *isosurface* (a 3D
filter producing a 2D surface where the field equals a value). What we call "Contour"
ParaView calls "Surface coloring." If we add isosurface support, we should not
reuse the name.

---

### 5. Averaging — the most interesting one

This is where ParaView and apeGmsh diverge most clearly in *philosophy*.

| | apeGmsh | ParaView |
|---|---|---|
| Where averaging lives | On the `ContourStyle.averaging` field; `_contour.py:_resolve_averaging()` ([`_contour.py:292-300`](../../src/apeGmsh/viewers/diagrams/_contour.py)) | As a separate *filter*: `vtkCellDataToPointData` |
| User-facing modes | `"averaged"` / `"discrete"` — explicit toggle per Contour diagram | Pick "POINTS" or "CELLS" in the color-by array picker; or insert a CellDataToPointData filter |
| What "averaged" means | Per-element GP extrapolation to corner nodes, then average across elements sharing each node | Generic: sum cell values into each touched point, divide by touch count (no FEM knowledge) |
| Topology awareness | Four effective backends: `nodal`, `gauss_cell`, `gauss_cell_averaged`, `gauss_node`, `gauss_cell_no_avg` | One generic averaging step; user composes with extraction |
| Where the choice surfaces | `ContourStyle` field, written into the diagram spec | Implicit in the pipeline (which array you color by) |

**Verdict:** Our averaging is **FEM-correct and explicit**. ParaView's is **generic and
composable**.

The downside of our approach: averaging only exists for `contour`. To add averaging to,
say, the `vector_glyph` kind, we'd have to wire `_AVG_AVERAGED` / `_AVG_DISCRETE` into
its style class too, and reimplement the extrapolation logic. ParaView's
cell-to-point filter is reusable for *anything*.

The downside of ParaView's approach: averaging is naive (no shape-function-aware
extrapolation). For our use cases (true Gauss-point extrapolation) we'd have to write a
custom filter anyway.

**Gap, if any:** averaging is currently exposed only on Contour. As the catalog grows
(yield indicators, derived fields), each kind will redeclare it. A small averaging
utility module — `_averaging.py` — that the diagrams call could deduplicate.

---

## The real gaps

Reading the catalog with fresh eyes against canonical FEM postprocessor expectations
(Abaqus/CAE, LS-PrePost, STKO, Femap), here are the missing pieces, ranked by impact:

### Tier 1 — power-user FEM expectations we lack

1. **Tensor visualization (principal directions)** — biggest gap. For 3D stress
   recordings (`stress_xx/yy/zz/xy/yz/xz`), no way to compute or display principal
   stresses or directions. Anyone analyzing solid stresses will notice. ParaView's
   `vtkTensorGlyph` is the model.
2. **Isosurfaces / isolines** — equi-value surfaces and contour lines. Standard in any
   3D solids viewer. For our shell/beam dominance this is less critical, but it's
   absent.
3. **Field-driven cutting planes** — drag-a-plane to slice the model and contour on
   the cut. We have `section_cut` but it's OpenSees-tag-driven, not interactive.

### Tier 2 — derived-field features

4. **Yield / damage / failure indicators** — color cells where `|σ| > σ_y` or where a
   damage scalar crosses a threshold. Today the user has to know which component name
   to color by; there's no semantic "is this element in trouble?" view.
5. **User-defined derived fields** — "color by Von Mises stress" requires either a
   built-in kind for each invariant or a Python expression. ParaView's
   `pythonCalculator` filter takes a one-line expression; ours has nothing.
6. **Envelope across stages** — already designed (`TimeMode.ENVELOPE` exists in
   [`_director.py:53`](../../src/apeGmsh/viewers/diagrams/_director.py)) but deferred per the Phase 6 plan in your auto-memory.

### Tier 3 — niche or already planned

7. **Streamlines / pathlines** — useful for displacement fields and some flow problems;
   minor for structural FEM.
8. **Animation mode** — deferred per memory.
9. **Mode shape side-by-side comparison** — needs multi-view.
10. **Hodograph / orbital** — 2D trace of a node's displacement; specialty.

---

## What "barebones" really means

Looking at this honestly, the *catalog* isn't where Results.viewer is barebones. The
gaps are:

- **No derived-field computation** (Tier 1 #1, Tier 2 #4–5). Every FEM postprocessor
  worth its salt computes Von Mises, principal stresses, yield ratios, damage indicators
  on the fly. We compute none of these — the user has to record them explicitly in the
  analysis, *if* OpenSees even outputs them. This is the single biggest catalog gap.
- **No interactive cutting planes** (Tier 1 #3). Section cuts exist but require defining
  them in OpenSees first.
- **Animation / envelope / range are designed but not landed** (Tier 3 #8, Tier 2 #6).

The first one is structural — it implies a new framework, not a new diagram kind. The
others are scoped extensions.

---

## What this means for the plan

This analysis doesn't replace any of the existing plan docs but suggests three new
candidates for `future/`:

- **`future/derived-fields.md`** — a small expression engine + a registry of built-in
  invariants (Von Mises, max shear, principals, yield ratio, damage). New diagram kind:
  `derived_contour`. Or, better: extend `ContourStyle.component` to accept an
  expression rather than just a recorded component name. This is the most valuable
  next-tier feature.
- **`future/tensor-glyphs.md`** — a `tensor_glyph` diagram kind producing principal-
  direction arrows or ellipsoids at element or Gauss-point locations. Self-contained;
  shippable independently. Maps to `vtkTensorGlyph`.
- **`future/isosurfaces.md`** — `isosurface` kind via `vtkContourFilter`; only useful
  for 3D solid models, but a clean addition. Lower priority unless you actively model
  solids.

A fourth candidate worth flagging — extending the **interactive section cut** so it can
be field-driven (drag a plane through the model) — would slot in alongside the existing
`section_cut.py` rather than a new kind. Probably a follow-up to Phase 5, not a new
plan.

---

## Honest summary

The catalog is one of the better parts of `results.viewer`. The "barebones" feel comes
from three things, in order:

1. **No derived-field computation** — Von Mises, principals, yield. This is the killer.
2. **No tensor visualization** — explicit skip in `_kind_catalog.py:113`.
3. **Already-planned features that haven't landed** — animation, envelope, range mode.

Items 1 and 2 are net-new framework / kind work. Item 3 is execution against an existing
plan. If we add a `derived-fields` future doc and a `tensor-glyphs` future doc, the
catalog story is fully scoped.
