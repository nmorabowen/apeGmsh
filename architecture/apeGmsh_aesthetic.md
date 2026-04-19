---
title: apeGmsh Aesthetic
aliases:
  - aesthetic
  - apeGmsh-aesthetic
  - theme
  - themes
  - viewer-theme
tags:
  - apeGmsh
  - architecture
  - aesthetic
  - theme
  - pyvista
  - qt
---

# apeGmsh Aesthetic

> [!note] Companion document
> This file defines the *visual language* of apeGmsh — how
> geometry, mesh, and results are rendered across the three
> user-facing viewers. It builds directly on
> [[apeGmsh_principles]] tenet **(viii)** "the viewer is core and
> environment-aware" and on [[apeGmsh_visualization]] §§2–5 for
> the module layout that will host the implementation. The
> `viewers/ui/theme.py` Qt chrome theme already in the codebase
> is treated as one of the three starting themes (Catppuccin
> Mocha), not as a replacement for this system.

A 3D FEM model is unreviewable without good visualization, and
"good" is not a single style. A CAD-like outline view answers
the question *"is the geometry right"* in seconds; a soft-shaded
mesh view answers *"is the mesh connected and well-formed"*; a
scalar-colored results view answers *"where does the field live."*
apeGmsh commits to all three answers, and each has its own
visual treatment. This document specifies what those treatments
are and how they vary across themes and entity dimensions.

---

## 1. Two layers: aesthetic and theme

apeGmsh separates the *aesthetic* (rendering philosophy) from
the *theme* (palette instantiation of that philosophy).

The **aesthetic** is the fixed set of rules that describe *how*
each viewer draws the model. Model viewer uses flat matte
shading with black BRep outlines. Mesh viewer uses soft
photographic shading with a subordinate wireframe and a radial
background vignette. Results viewer inherits the mesh viewer's
lighting and composition and replaces the per-body color with a
scalar colormap. These choices do not change when the user
switches themes.

The **theme** is the palette instantiation: background gradient,
default body palette, outline color, mesh-line color treatment,
node accent color, colormap family, and the Qt UI chrome.
Themes vary; the aesthetic does not.

Keeping these layered has one load-bearing consequence: *a theme
cannot break the "Model view reads like CAD, Mesh view reads
like a studio render" promise*. A theme that wanted to would be
rejected — it has crossed the boundary between palette and
philosophy.

### 1.1 Chrome and viewport

Two theme layers exist today and they are distinct:

1. **Qt UI chrome** — panels, docks, groupboxes, buttons,
   toolbars. Styled via `viewers/ui/theme.py`.
2. **Viewport rendering** — background, body palette, outline
   color, mesh-line color, node glyph color, colormap family.
   To be added alongside `theme.py`.

**Default coupling: the two switch together.** Selecting
"Paper" applies the Paper chrome and the Paper viewport; the
user does not have to make two consistent choices. A power-user
override (`viewport=...` kwarg on the viewer factories) keeps
them independently selectable but is not the default UX.

---

## 2. Three viewers, one visual language

The aesthetic is defined per viewer. Each viewer answers a
different question, so each has a different rendering approach.
The three are reached through the session tree as documented in
[[apeGmsh_visualization]] §3.

### 2.1 Model viewer — flat matte + BRep outlines

Opened via `g.model.viewer(...)`. Purpose: geometry review —
"is the CAD right, and are the parts where I think they are."

Canonical treatment:

- **Shading**: one diffuse key light from upper-left, weak fill,
  no specular, no reflection, no anisotropy. Appearance matches
  pedagogical CAD (SolidWorks, Fusion 360 "shaded with edges").
- **Body color**: one solid color per body / `PhysicalGroup`.
  No per-face gradients, no textures.
- **Outlines**: continuous black polyline along every BRep edge
  — silhouette + feature (dihedral angle above threshold,
  default 25°) + explicit BRep edges (bolt-hole circles,
  engraved grooves). Implementation is a BRep-edge pass, not a
  screen-space Sobel. The outline weight is theme-dependent
  (see §5).
- **Background**: theme-dependent; see §5.
- **Cast shadows / ground plane**: none.
- **Ambient occlusion**: mild, in concavities only.

### 2.2 Mesh viewer — soft shading + subordinate wireframe

Opened via `g.mesh.viewer(...)`. Purpose: mesh review — "are my
elements connected, sized, and shaped."

Canonical treatment:

- **Shading**: one key + one fill + ambient. Softer than the
  model viewer — real value gradient from lit to unlit face,
  not toon-flat. Feels photographic.
- **Body color**: still one color per body / `PhysicalGroup`,
  but the chroma is lower than the model viewer (the mesh will
  do the visual work; the fill supports it).
- **Mesh lines**: drawn on top of the shaded surface, with
  color computed *relative to the underlying fill* — a
  darker-than-fill line on light bodies, a lighter-than-fill
  line on dark bodies. Never a fixed black; black mesh lines
  drown the body color.
- **Mesh line opacity**: 60–80% depending on element density.
  At fine resolution the lines must subordinate to the fill,
  not overwhelm it. Tuned per theme.
- **Outlines**: no bold silhouette outline. The shading
  gradient and the background contrast provide separation.
- **Background**: radial vignette (all themes, including
  Paper — see §5 for how Paper implements this without a
  gradient). The vignette is what makes the part "float"
  rather than sit on a slab.
- **Contact AO**: explicit ambient-occlusion darkening at
  contact interfaces between bodies. This is what makes two
  parts *read as in contact* rather than merely coincident.
- **Ground plane**: none.

### 2.3 Results viewer — colormap + mesh shading

Reached via `fem.viewer(...)` → `Results.viewer(...)`, or the
external `apeGmshViewer` for result timelines. Purpose: field
review — "where and how does the scalar / vector / tensor
live on my mesh."

Canonical treatment:

- **Lighting and composition**: inherited from the mesh viewer.
  Same shading model, same vignette, same contact AO.
- **Body color**: *replaced* by the scalar colormap. The
  mesh-viewer per-body palette is never shown at the same time
  as a field — they would fight.
- **Colormap family**: perceptually uniform — **viridis** on
  dark backgrounds, **cividis** on light. Diverging fields
  (signed stress, positive/negative displacement) use
  **coolwarm** or **BrBG**. *Rainbow / jet / turbo is
  prohibited* — it is perceptually non-uniform, misleads the
  eye about magnitude, and is unreadable under common
  colorblind profiles. Ansys image 1 in the design reference
  is the look apeGmsh deliberately does not produce.
- **Mesh lines**: dim by default (20% opacity), toggleable
  bright. Bright mesh on a colored field fights the field;
  default off-ish.
- **Legend**: positioned bottom-right, theme-consistent font
  and tick style. Scalar range, units, and title drawn with
  the chrome font (see §5).
- **Deformed shape**: rendered with a scalar warp on the mesh.
  Undeformed outline shown at 30% opacity for reference (in
  dark themes) or dashed (in the Paper theme).

---

## 3. Four dimensions, per-viewer treatment

Geometry and mesh live at four dimensions: points/nodes (0D),
curves/line elements (1D), surfaces/shells (2D), volumes/solids
(3D). Each dimension needs its own rendering rules, *because
a mixed-dimensional model is the common case* — an RC frame
with shell slabs, beams embedded in solids, lumped masses at
nodes. The aesthetic works only if all four can appear on one
screen without clashing.

### 3.1 Nodes (0D)

Nodes are always available and always legible. They are the
user's primary working surface for attaching loads, constraints,
masses, and selection operations; a view where nodes are hard
to see or hard to pick is broken.

- **Geometry**: sphere glyph (existing `build_node_cloud` in
  `viewers/scene/glyph_points.py`). Radius proportional to the
  model diagonal (default factor ≈ 0.003 — already the apeGmsh
  convention). Spheres, not billboards — they give shading
  cues so nodes read as objects.
- **Color**: theme-driven *neutral accent* — never competing
  with body color. See §5 per theme.
- **Depth behavior**: **depth-tested by default** (nodes behind
  a solid are occluded — clean), with a *"nodes on top"* toggle
  for connectivity debugging (renders node glyphs with depth
  test off, so every node is visible).
- **Filter categories**: user-facing toggle groups, OR-combined:
  *all / boundary / on-selected-entities / free (unconnected) /
  constrained / loaded*. These map to broker queries: free
  means no element contains the node; constrained means the
  node is referenced by a constraint record; loaded means a
  load record references it. Constrained/loaded are FEM-level,
  so available only in the mesh viewer with a `FEMData`
  snapshot.
- **Pick affordance**: hovered node grows to 1.5× radius;
  picked node grows to 2.0× and changes to the pick red
  (#E74C3C — aligned with `ColorManager` per
  [[apeGmsh_visualization]] §2.2). Priority *hidden > picked >
  hovered > idle*, same as all other cells.
- **Off means off**: when the node display is disabled there
  is no dimmed ghost. The user sees pure geometry.

### 3.2 Line elements (1D)

Raw 1D lines render as aliased thin strokes with no shading
and no depth cue — unacceptable for structural review. Three
treatments are specified, with the default varying by theme.

1. **Tube rendering (default in dark themes)**. Each line
   element is extruded to a cylinder (radius ≈ 0.002 × model
   diagonal; exact factor is a tuning parameter per theme).
   Cylinders receive shading, AO, silhouette — a frame model
   reads as a physical truss, not a wireframe.
2. **Section-extruded rendering (toggle — all themes)**. If
   the line has an assigned beam section and geometric
   transformation (typical for OpenSees frame models — see the
   `GeomTransfViewer` in [[apeGmsh_visualization]] §2.1),
   extrude the actual cross-section (I-beam, box, channel,
   circular, rectangular, fiber) along the line axis in the
   `geomTransf` local frame. This is the truth view — section
   orientation (strong vs weak axis) becomes visible; asymmetric
   sections read as oriented objects. Not default because for
   pre-design sketches the tube is faster and less cluttered;
   becomes essential the moment the user reviews a structural
   model for design.
3. **Stroke (Paper theme default)**. Flat 2D line in screen
   space, fat pen weight (~2–3 px along silhouette, ~1 px
   elsewhere), no tube geometry. Matches the
   technical-drawing language of the Paper theme. Tube
   rendering is *disabled* in Paper — would look inconsistent
   with the CAD-drawing vibe; the section-extruded toggle
   remains available for structural figures.

Tubes additionally support a **local-axis color mode** (toggle,
all dark themes): red = local *x*, green = local *y*, blue =
local *z*. Conventional RGB-axis colormap; identical to the
`GeomTransfViewer` scheme so the two viewers agree.

### 3.3 Surfaces (2D)

Shells and plates carry thickness as a section property.

- **Default**: rendered as zero-thickness surfaces — matte fill
  in the model viewer, softly shaded with subordinate mesh
  lines in the mesh viewer.
- **Thickness-extrusion toggle**: render the shell as a solid
  of the assigned thickness, centered on its mid-surface along
  the shell normal. For thickness visualization, debug, and
  figure production where real physical scale matters.
- **Local axis colors** (toggle): red = shell local *1*, green
  = local *2*, blue = normal. Same RGB convention as beams.

### 3.4 Solids (3D)

Already covered in §2. Flat matte + BRep outlines in the
model viewer; soft shading + subordinate mesh + contact AO in
the mesh viewer.

### 3.5 Axis and world references

Orientation and spatial reference are first-class visual
elements, independent of entity dimensions. This section
covers the world reference frame and every on-screen
indicator of origin, orientation, and model extent.

**Convention** — apeGmsh commits to **Z-up, right-handed**
world coordinates and the universal **R=X, G=Y, B=Z** axis
color mapping. This matches OpenSees, every major structural
CAE tool (SolidWorks, Ansys, Abaqus, ETABS, SAP2000, STKO),
Gmsh's own rendering when axes are shown, and the element
local-frame convention in §§3.2–3.3. The RGB mapping is
**theme-independent** — do not vary it per theme, even in
Paper. One convention, one mental model.

**Five indicators** are defined, each with a toggle and a
default. apeGmsh does not track units at the session level,
so a unit indicator is explicitly out of scope. View-preset
keybindings are not assigned either — those key chords are
already reserved for selection (rubber-band, modifier-
assisted picking) per [[apeGmsh_visualization]] §2.2.

| Indicator            | Dark themes default                 | Paper default                       | Per-theme palette? |
| -------------------- | ----------------------------------- | ----------------------------------- | ------------------ |
| Corner triad         | On                                  | Off                                 | Hub color only     |
| Origin triad + hub   | On (model, mesh) · off (results)    | On (model, mesh) · off (results)    | Hub color only     |
| Reference grid       | Off                                 | Off                                 | Yes                |
| Bounding box         | Off                                 | Off                                 | Yes                |
| Element local axes   | Off — toggles live in §§3.2, 3.3    | Off                                 | No (always RGB)    |

#### Corner triad

Classic bottom-left gizmo — non-clickable, mirrors camera
orientation. Navigation-cube behavior (Fusion, Onshape) is
rejected for v1. Sized to ≈80 px in screen space; rendered
with depth test **off** (this is UI, not geometry). Arrow
colors are fixed R/G/B for X/Y/Z; the hub sphere inherits the
theme's node-accent color (§3.1) for visual continuity with
node glyphs. Line weight 1.5 px in dark themes, 2.0 px in
Paper. Default **on** in dark themes, **off** in Paper —
captions in published figures state orientation; a gizmo eats
pixels on a printed page.

#### Origin triad + hub

A world-space triad drawn at (0, 0, 0). Cone-tipped arrows
along +X, +Y, +Z emanate from a small hub sphere. The triad
sits in *model space* (not screen space) and scales with the
model bounding-box diagonal.

Starting sizes — tunable per theme:

| Component             | Factor × model diagonal |
| --------------------- | ----------------------- |
| Hub sphere radius     | 0.005                   |
| Arrow shaft length    | 0.050                   |
| Arrow shaft radius    | 0.004                   |
| Cone tip length       | 0.015                   |
| Cone tip radius       | 0.008                   |

Arrow colors are theme-independent R/G/B. The hub sphere
uses the theme's node-accent color. In Paper, arrows
additionally receive a thin black silhouette stroke (1.5 px)
so the RGB reads sharply on the near-white background.

Depth-tested by default — model geometry at the origin
occludes the triad. An *"origin on top"* toggle renders the
triad with depth test off, mirroring the node depth-behavior
toggle (§3.1). Consistency with the node pattern is the
point.

Default visibility by viewer:

- Model viewer — on.
- Mesh viewer — on.
- Results viewer — off. The field is the subject; a spatial
  mark is noise.

The origin triad and the corner triad coexist. They answer
different questions: the corner shows *camera orientation*,
the origin shows *where zero is in the model*. Both remain
independently toggleable — apeGmsh does not auto-hide one
when the other is visible.

#### Reference grid

Grid on the XY plane at Z = 0. **Off by default** — honors
the §6 rule "model floats in space, no ground plane." A
grid is subtly different from a ground plane (no opacity, no
shadow), but visually adjacent enough that it shouldn't be
the landing state. Toggle is available for users who need
Z = 0 grounded (structural reviewers, slab-on-grade models,
site plans).

Adaptive spacing — major gridlines at the nearest power of
10 spanning the model XY footprint; minor gridlines at 1/10
of major. Grid extent ≈ 1.5 × model XY bbox.

**Axis rulers** — the grid row through Y = 0 and the grid
column through X = 0 render at the major line weight in
**red** and **green** respectively. This is where the grid
visually connects to the origin triad's +X and +Y arrows.

#### Bounding box

Axis-aligned wireframe around the model with numeric min/max
tick labels on each axis. Off by default. Tick label font
matches the chrome font. A tick label that would overlap the
model is suppressed rather than drawn on top.

#### Element local axes

Toggles live in §§3.2 (line elements) and §3.3 (surfaces).
The *color convention* is identical to the world triads
(R = local 1, G = local 2, B = local 3 / normal) — one RGB
convention across every axis indicator in apeGmsh.

---

## 4. Themes

> [!note] v2 update — what actually shipped
> The original §4 description below is preserved as design
> context, but the implementation has evolved. Three changes
> worth surfacing up front:
>
> 1. **CAD-neutral fills were adopted across all themes.** The
>    "body-relative mesh-line" rule (30% shift toward opposite
>    luminance) in §4.1–4.2 was dropped in favor of gray fills
>    (`dim_srf`/`dim_vol` ≈ `(210,210,210)` on dark themes,
>    `(192,192,192)` on Paper) with black wire, black mesh edges,
>    and black nodes. Theme identity lives in chrome, background
>    mode, outlines, and interaction/accent colors — not in body
>    color. The `Palette` gained a single `mesh_edge_color` field;
>    the old `mesh_line_mode` / `mesh_line_shift_pct` /
>    `mesh_line_opacity` / `mesh_line_fixed_color` fields were
>    removed.
> 2. **The roster grew from 3 to 10 built-ins.** Catppuccin Latte,
>    Solarized Dark, Solarized Light, Nord, Tokyo Night, Gruvbox
>    Dark, and High Contrast (accessibility) joined the original
>    three. Reserved slot 4.4 "Cappuccino" was not used.
> 3. **User-authored custom themes are supported.** JSON files at
>    `<config>/apeGmsh/themes/*.json` auto-load at startup; a
>    modal Theme Editor dialog (see [[apeGmsh_visualization]] §2.4)
>    authors them with live preview. Built-in names are
>    protected from overwrite via `theme._BUILTIN_THEME_IDS`.
>
> The palette rules, per-viewer aesthetics, and per-dimension
> treatment (§§2–3 and §6) did *not* change.

Each theme fully specifies the viewport for all three viewers
and the Qt chrome it ships with. Themes switch together by
default (chrome + viewport), with an override for power users.

### 4.1 Neutral Studio

The "product photography" theme. A neutral gray radial vignette
with muted industrial body colors. Reads as a premium CAE tool;
closest in spirit to the Ansys image 2 reference.

- **Viewport background**: radial gradient, `#2a2a2a` at center
  → `#0f0f0f` at edges. No hue — deliberately neutral.
- **Body palette (v1)**: steel blue `#5B8DB8`, olive `#A9A878`,
  graphite `#4A4A4A`, mint `#A8C8B5`, warm off-white `#EAE6DE`.
  Desaturated; strong value contrast between adjacent bodies.
- **Model viewer outline**: pure black `#000000`, 1.5 px on
  silhouette, 1.0 px on feature edges.
- **Mesh viewer line color**: computed per-body as body color
  shifted 30% toward black (light bodies) or 30% toward white
  (dark bodies). Opacity 70%.
- **Node accent**: warm off-white `#EAE6DE`.
- **Contact AO intensity**: moderate (default).
- **Results colormap**: viridis (sequential), coolwarm
  (diverging).
- **Qt chrome**: dark neutral — `#1f1f1f` panels on `#141414`
  base, text `#d0d0d0`. Accent `#7aa2d7` (matches steel blue
  body color for visual continuity between viewport and
  chrome).

### 4.2 Catppuccin Mocha

The existing `viewers/ui/theme.py` palette, preserved. Viewport
matched to the chrome for continuity.

- **Viewport background**: linear vertical gradient — Mantle
  `#181825` top → Crust `#11111b` bottom. No radial vignette
  (optically similar effect with less complexity).
- **Body palette (v1)**: Sapphire `#74c7ec`, Peach `#fab387`,
  Green `#a6e3a1`, Mauve `#cba6f7`, Rosewater `#f5e0dc`. All
  drawn from Catppuccin Mocha accents — stays coherent with
  the chrome.
- **Model viewer outline**: Crust `#11111b` (near-black with a
  slight warm tint so it doesn't fight the cool background).
  1.5 px silhouette, 1.0 px feature.
- **Mesh viewer line color**: same per-body shift rule as
  Neutral Studio; opacity 70%.
- **Node accent**: Rosewater `#f5e0dc`.
- **Contact AO intensity**: moderate.
- **Results colormap**: viridis, coolwarm.
- **Qt chrome**: existing Catppuccin Mocha stylesheet
  (unchanged).

### 4.3 Paper

The figure-production theme. Optimized for journal papers,
reports, EOS lecture slides, and anywhere the model must read
on a printed page. Matches the SolidWorks image 1 language.

- **Viewport background**: flat `#FAFAFA` (near-white — pure
  white loses the outline relationship). No radial vignette in
  the geometric sense; instead a *very soft* luminance
  darkening at the extreme corners (~5%) to prevent the part
  from blending into the UI chrome edge. The mesh viewer
  retains the vignette effect subtly to preserve the "floating
  part" read.
- **Body palette (v1)**: steel blue `#8BA8C4`, olive-tan
  `#B9B681`, spring green `#A0C893`, rubber black `#2F2F30`,
  cream `#E8E0C8`. Slightly more saturated than Neutral Studio
  — on white, desaturated colors turn to mud.
- **Model viewer outline**: pure black `#000000`, 2.0 px on
  silhouette, 1.2 px on feature — heavier than dark themes
  because white backgrounds need more weight to read.
- **Mesh viewer line color**: `#303030` at 40% opacity — a
  soft gray that reads over any body color without fighting.
- **Node accent**: pure black `#000000`.
- **Contact AO intensity**: light (white backgrounds amplify
  AO; too much feels dirty).
- **Results colormap**: cividis (sequential — designed for
  colorblind readability on light backgrounds), BrBG
  (diverging).
- **Qt chrome**: light — `#F5F5F5` panels on `#FAFAFA` base,
  text `#202020`. Accent `#2E5C8A`.

### 4.4 Reserved: Cappuccino (v2)

A literal warm-cream theme — cream vignette, warm earth body
palette, dark brown outlines. Not implemented in v1; included
here as a named slot so the theme enum reserves the ID and
early users can request it without the name being claimed by
something else.

---

## 5. Palette and treatment matrices

Two matrices. The first is *what color* per dimension per
theme; the second is *what rendering technique* per dimension
per viewer. Together they fully determine the output.

### 5.1 Theme × dimension — palette matrix

| Dimension    | Neutral Studio              | Catppuccin Mocha            | Paper                          |
| ------------ | --------------------------- | --------------------------- | ------------------------------ |
| 0D (nodes)   | Warm off-white `#EAE6DE`    | Rosewater `#f5e0dc`         | Pure black `#000000`           |
| 1D (lines)   | Body palette; tube radius ≈ 0.002×dia | Body palette; tube radius ≈ 0.002×dia | Stroke; 2 px silhouette, 1 px interior |
| 2D (surfaces)| Body palette (muted)        | Body palette (Mocha accents)| Body palette (more saturated)  |
| 3D (solids)  | Body palette (muted)        | Body palette (Mocha accents)| Body palette (more saturated)  |
| Background   | Radial `#2a2a2a`→`#0f0f0f`  | Linear Mantle→Crust         | Flat `#FAFAFA` + corner falloff |
| Outline (model view) | `#000000` 1.5/1.0 px | Crust `#11111b` 1.5/1.0 px | `#000000` 2.0/1.2 px         |
| Mesh line (mesh view)| Body-relative, 70% op | Body-relative, 70% op     | `#303030`, 40% op              |
| Colormap (seq / div) | viridis / coolwarm  | viridis / coolwarm          | cividis / BrBG                 |
| Chrome accent        | `#7aa2d7`           | Mocha accent (unchanged)    | `#2E5C8A`                      |
| Axis arrows (all triads) | R=X / G=Y / B=Z (theme-independent) | R=X / G=Y / B=Z (theme-independent) | R=X / G=Y / B=Z + 1.5 px black stroke |
| Grid (major / minor) | `#3a3a3a` / `#2a2a2a` | Surface1 `#45475a` / Surface0 `#313244` | `#d0d0d0` / `#e8e8e8`    |
| Bounding box         | `#9a9a9a` 1 px      | Overlay1 `#7f849c` 1 px     | `#000000` 1 px                 |

### 5.2 Viewer × dimension — treatment matrix

| Dimension    | Model viewer                          | Mesh viewer                               | Results viewer                            |
| ------------ | ------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| 0D (nodes)   | Sphere glyph, BRep-point accent color | Sphere glyph, FEM node accent color, filters apply | Same as mesh viewer; hidden by default when field is on |
| 1D (lines)   | Tube (dark) / stroke (Paper), BRep outline | Tube (dark) / stroke (Paper); section-extruded toggle | Colormap-fill along tube/stroke; section-extruded toggle |
| 2D (surfaces)| Matte fill + BRep outline             | Soft fill + subordinate wireframe; thickness-extrusion toggle | Colormap fill; mesh lines dim by default |
| 3D (solids)  | Matte fill + BRep outline             | Soft fill + subordinate wireframe + contact AO | Colormap fill on surface; mesh lines dim by default |

---

## 6. Shared design rules

Rules that apply in every viewer, every theme, every dimension.
These are the *aesthetic* (the fixed philosophy) — themes cannot
override them.

1. **No reflections, no specular highlights.** The rendering is
   matte-to-soft, never glossy. Glossy surfaces imply physical
   materials, which apeGmsh does not know.
2. **No ground plane, no cast shadow.** The model floats in
   space. Ground planes imply a real scene; cast shadows add
   visual noise without adding engineering information.
3. **One color per body / `PhysicalGroup`.** Per-face gradients
   and textures are reserved for encoding *data* (scalar fields,
   colormap results). A theme that uses per-face color
   decoration is wrong.
4. **Desaturated, value-dominant palette.** Strong *value*
   contrast between adjacent bodies; modest *hue* contrast.
   Adjacent bodies never share a hue even if the value differs
   — the eye separates hue before value.
5. **Ambient occlusion is always on in concavities and at
   contact interfaces.** Intensity varies by theme (Paper is
   light, dark themes moderate). AO is the single cue that
   most distinguishes a "real" render from a raw OpenGL
   dump.
6. **Camera: orthographic or mild perspective.** Wide-angle
   perspective distorts engineering scale. Ortho is the
   default for the model viewer; mild perspective (fov 30°)
   is the default for mesh and results viewers for depth
   cues.
7. **Mesh lines never fixed black on colored fill.** Body-
   relative color (darker-than-fill on light, lighter-than-fill
   on dark) or a neutral gray with reduced opacity. Fixed
   black mesh lines on body color always look cheap.
8. **State priority for per-cell color**: *hidden > picked >
   hovered > idle* — already in `ColorManager`, applies to
   nodes, lines, surfaces, solids uniformly.

---

## 7. Colormap policy

Results rendering depends on good colormap choices. apeGmsh
takes a stance.

**Required**:

- **Sequential fields** (unsigned magnitudes, damage, strain
  energy, scalar results): *viridis* on dark backgrounds,
  *cividis* on light. Both are perceptually uniform (equal
  steps in data → equal steps in perceived lightness) and both
  are colorblind-safe. Cividis is specifically designed to be
  legible to all forms of color vision deficiency.
- **Diverging fields** (signed stress components, signed
  displacement, any field with a meaningful zero):
  *coolwarm* on dark backgrounds, *BrBG* (brown-blue-green) on
  light. Both are perceptually symmetric around the midpoint.
- **Categorical fields** (element type, partition ID,
  `PhysicalGroup` ID): the theme's body palette, sampled by
  hash. Never a colormap.

**Prohibited**:

- **Rainbow / jet / turbo**. Perceptually non-uniform — a small
  step in data can produce a large perceived color change
  (yellow band) or none (green plateau). Misleads the eye
  about magnitude. Unreadable under common colorblind
  profiles. The Ansys engine-block reference image (jet on
  stress) is exactly the look apeGmsh does *not* produce.

**Practice**:

- Colormap is part of the theme, not the result. Switching
  theme switches the colormap default; switching result does
  not.
- Users can override per-call (`results.plot(cmap='plasma')`),
  but the default must be correct.
- The colormap legend always renders units and range — no
  anonymous color bars.

---

## 8. Environment and mode interactions

The aesthetic must work across the three environments
[[apeGmsh_principles]] tenet (viii) promises — Desktop, Jupyter,
Colab/remote — without code change.

- **Desktop (Qt+PyVista)**: full aesthetic, all themes. The
  reference implementation.
- **Jupyter (local, Qt available)**: full aesthetic via
  `pyvistaqt.QtInteractor`. Identical.
- **Jupyter (local, trame/HTML fallback)**: full aesthetic
  minus the Qt chrome (there is no chrome in the HTML
  viewport). Viewport palette, outlines, node glyphs,
  colormaps unchanged.
- **Colab / remote**: `GeomTransfViewer` (Three.js) and the
  external `apeGmshViewer` subprocess are the two surfaces.
  The external viewer is out of scope for this document and
  follows its own theme; `GeomTransfViewer` inherits the
  *Neutral Studio* palette unconditionally (it has no theme
  selector).
- **VTK / ParaView export** (`.vtu` / `.pvd`): apeGmsh writes
  raw geometry and data. Paraview applies its own rendering.
  The aesthetic defined here does not travel with exports.

---

## 9. Contributor notes

Rules for anyone adding to the visual system.

1. **Aesthetic changes are architectural, not decorative.**
   Adding a new dimension treatment, changing the
   always-on rules in §6, or redefining a viewer's purpose
   (§2) requires updating this document first. A PR that
   changes rendering behavior without updating the aesthetic
   doc is incomplete.
2. **Themes are additive.** Adding a new theme requires
   filling *every* cell of the §5 matrices. A partial theme
   that inherits some values from another theme is rejected —
   it means the new theme is actually a modification, not a
   new identity.
3. **The `ColorManager` / `VisibilityManager` / `SelectionState`
   priorities** from [[apeGmsh_visualization]] §2.2 are the
   source of truth. A theme that wants a different priority
   (e.g., hovered-over-picked) is proposing an *aesthetic*
   change, not a theme change — see rule 1.
4. **Palette hex codes in §4 are v1 starting points.** They
   will be tuned once the implementation lands and we can look
   at screenshots. The *rules* (body-relative mesh lines, no
   fixed black mesh, no jet colormap) are non-negotiable; the
   specific hex values are.
5. **Overlays inherit the theme.** Constraint overlays, moment
   glyphs, mass markers (see `viewers/overlays/` in
   [[apeGmsh_visualization]] §2.5) use the theme's accent
   palette. No overlay picks its own color.

---

## 10. Open items

Decisions deliberately deferred from v1.

- **Annotation typography**: font, size, and anti-aliasing
  for entity tags, coordinates, node IDs, legend ticks.
  Defers to implementation; the contract is that every
  theme picks one monospace + one sans family and uses them
  consistently.
- **Hover tooltip style**: floating panel with entity info.
  Should match the theme's chrome. Defers to implementation.
- **Legend layout for results**: bottom-right is the
  convention, but vertical/horizontal orientation and tick
  density are tuning parameters.
- **Section-extruded rendering performance**: unknown cell
  count at which the toggle becomes laggy. Benchmark before
  committing.
- **Cappuccino theme (v2)**: warm-cream palette, deferred.
- **Print export aesthetic**: saving a model viewer frame to
  `.svg` or `.pdf` (as a true vector drawing, not a raster)
  would pair beautifully with the Paper theme. Not in v1.

---

## Reading order

1. [[apeGmsh_principles]] tenet **(viii)** — the promise that
   three environments work without code change, and that the
   viewer is core.
2. [[apeGmsh_visualization]] §§1–5 — the module layout
   (`viz/` vs `viewers/`, scene/core/ui/overlays) that will
   host the aesthetic.
3. This document §§1–2 — the two-layer model and the
   per-viewer philosophy.
4. §§3–4 — dimension treatments and the three themes.
5. §5 — the two matrices that, together, fully specify any
   concrete rendering call.
6. §§6–7 — the non-negotiable rules (design rules, colormap
   policy). Read these before proposing any aesthetic change.
