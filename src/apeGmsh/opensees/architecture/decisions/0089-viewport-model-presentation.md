# ADR 0089 — Viewport model presentation: edge hierarchy, defaults, colors

**Status:** Proposed (2026-08-04) — rendering changes await user sign-off
on the A/B sheets cited below. Viewport-side companion to ADR 0087 (which
governs the Qt *chrome* and explicitly excludes the viewport). This ADR
AMENDS `internal_docs/architecture/apeGmsh_aesthetic.md` where noted (§
"Amendments" below) and contradicts it nowhere silently. ADR 0088 (window
layout) is being authored concurrently; nothing here touches its
territory.

## Context

The FE model rendering in the ResultsViewer has no *designed*
presentation — every element of it is an engineer default that shipped
with the first working commit:

- **No line-width hierarchy.** The substrate draws exactly one edge
  layer: a uniform wireframe at `prefs.mesh_line_width` (default
  **3.0 px**) in `palette.substrate_edge_color` (**#444444**), built in
  `add_substrate_actors` (`viewers/results_viewer.py`). There is no
  silhouette or feature-edge pass at all in the results viewer —
  `add_silhouette` is used only by the model viewer. Every edge has the
  same weight; the outer contour of the body has no more presence than
  an interior element boundary.
- **The node cloud is ON at boot, black, and oversized.** The results
  viewer builds it at `prefs.point_size` (**10.0 px** — the *BRep point
  glyph* preference, not the mesh-viewer `node_marker_size` 6.0) in
  `palette.node_accent`, which is **#000000** in all three canonical
  palettes. `_geometries.py` defaults `show_nodes: bool = True`. Result:
  black dots sprinkled over everything, including over contour fields.
- **The edge color is metamerically invisible.** Measured from the
  rendered pixels (mocha, demo column): the lit top face renders at RGB
  ≈ (172,169,159) but the side faces at ≈ (72,72,74)–(88,88,88),
  because the fill actor has **ambient = 0** (pure diffuse under the
  key light). The #444444 edge (= 68,68,68) sits *inside* that range —
  edges vanish on side faces regardless of the 3 px width. The same
  math kills every "tuned edge color" attempt until the shading floor
  is lifted.
- **The substrate ignores the theme.** `substrate_color` (#bfbfbf) and
  `substrate_edge_color` (#444444) are dataclass *defaults* on
  `Palette`; no built-in theme overrides them. Paper renders the same
  near-charcoal shaded faces on its white page as Mocha does on its
  dark vignette.
- **Contours inherit all of it.** A contour diagram draws the field,
  and the substrate wireframe (3 px, polygon-offset toward the camera)
  plus the black node cloud render *on top of* it. The field reads
  second.
- **Flag (out of this ADR's render scope, in its policy scope):**
  `ContourStyle.cmap` defaults to `"jet"` (`diagrams/_styles.py`) — a
  direct violation of the aesthetic doc §7 colormap policy. The viewer
  UI paths pass the palette's `cmap_seq` (viridis/cividis), so the
  violation is latent, but the dataclass default is wrong and the
  implementing phase must route it from the palette.

### Evidence — rendered A/B study, 2026-08-04

GL renders of the 3D column demo (`column_demo.h5`, 15-step
displacement history), captured headless by mutating live actor
properties in-process (no `src/` changes). Files live in the review
worktree's untracked `scratchpad_shots/` directory:

- **`SHEET_1_edge_hierarchy_and_nodes.png`** — current (uniform 3 px +
  black 10 px nodes) → nodes off → proposed hierarchy (2.5 px black
  feature/silhouette outline + 1 px near-black interior mesh at full
  opacity) → proposed + ambient 0.35 lift. Panels: `01_mocha_current`,
  `02_mocha_nodes_off`, `30_mocha_hierB1`, `61_mocha_ambient35`.
- **`SHEET_2_substrate_color_paper.png`** — mocha neutral vs
  cool-tinted fill; Paper current (charcoal faces + black dots on the
  white page) vs Paper proposed (#d9d9d9 fill + ambient + hierarchy).
  Panels: `61_mocha_ambient35`, `33_mocha_cool_full`,
  `10_paper_current`, `63_paper_ambient35`.
- **`SHEET_3_contour_over_field.png`** — contour + deform at last
  step: current (3 px web + black dots fight the field) vs proposed
  (nodes off, 1 px edges @ 40 % opacity, outline recomputed on the
  deformed surface) vs 25 % and edges-off brackets. Panels:
  `20_contour_current`, `50_contour_edges40`, `21_contour_proposed`,
  `22_contour_edges_off`.

Failed variants are part of the record: opacity-blended dark edges
(60 % opacity) disappear into the shaded faces (`04_mocha_hier_outline`)
— edge demotion on *plain* substrate must come from width and hue, not
opacity; lighter-than-fill edges (`32_mocha_light_edges`) are elegant on
dark faces but vanish on the lit top face, so a single light edge color
cannot serve all faces; a subdued node cloud (4 px, overlay gray —
`03_mocha_nodes_subdued`) is indistinguishable from noise and answers no
question — nodes are better honestly off.

## Decision

### D1 — Line-width roles: three weights, existing homes

One hierarchy, three roles. No new token home is created; each value
lives where its kind of value already lives (ADR 0087 D1 ownership
logic):

| Role | Width | Color | Home |
|---|---|---|---|
| Silhouette / feature outline | `palette.outline_silhouette_px` (2.5 built-ins; 2.0 when a field diagram is active) | `palette.outline_color` | `Palette` (fields already exist — reused, not added) |
| Interior mesh edge | **1.0 px** (new default; was 3.0) | `palette.substrate_edge_color` (per-theme, D3) | `Preferences.mesh_line_width` — stays the Display-panel user knob |
| Overlay / diagram lines | ≥ 2.0 px | per diagram style | diagram `*Style` dataclasses (unchanged) |

- The outline is a **new actor** in the results viewer: static feature
  edges of the render surface (boundary + dihedral > `feature_angle`,
  the existing 25° preference), polygon-offset above the wireframe.
  Static extraction (not `vtkPolyDataSilhouette`) is deliberate — see
  D4.
- Aesthetic constants that do **not** vary per theme are module-level
  frozen values in `results_viewer.py`, following the `GHOST_OPACITY`
  precedent (promotion to `Palette` fields is the fallback if a theme
  ever needs to vary them):
  - `SUBSTRATE_AMBIENT = 0.35` — ambient coefficient on the substrate
    fill actor (diffuse compensated to ≈ 0.85). This is the single
    change that makes every edge/fill color decision *legible*: it
    lifts side faces from RGB ≈ 72–88 to a readable mid-gray
    (`60_mocha_ambient25` vs `61_mocha_ambient35`; 0.35 chosen).
  - `CONTOUR_EDGE_OPACITY = 0.40` — wireframe opacity while any field
    diagram is active (D3).
- Node glyphs: the results viewer switches from `prefs.point_size`
  (10.0, a BRep preference — wiring mistake) to
  `prefs.node_marker_size` (6.0, the mesh/FEM preference the Display
  panel already edits).

### D2 — Default visibility at boot, per viewer

Each viewer answers a different question (aesthetic doc §2); the boot
state must show exactly what answers it:

| Layer | Model viewer | Mesh viewer | Results viewer |
|---|---|---|---|
| Fill | on | on | on |
| Interior mesh edges | n/a (BRep only) | on | **on** (1 px) |
| Silhouette / feature outline | on (existing) | on (existing treatment) | **on** (new actor) |
| Node cloud | on (BRep points) | **on** (nodes are the working surface for loads/constraints/selection) | **off** (was on) |
| Labels | off | off | off |

- `GeometryRecord.show_nodes` (`diagrams/_geometries.py`) defaults to
  `False` for results-viewer geometries. The Geometry panel checkbox,
  session persistence, and the per-geometry eye all keep working — this
  changes only the *boot* value. Reference ghosts already force
  `show_nodes=False`; unchanged.
- The mesh viewer's node default is **not** touched: pre-solve mesh
  review wants nodes (aesthetic doc §3.1 "always available and always
  legible" is a *mesh/model* commitment; §5.2 already says results
  nodes are "hidden by default when field is on" — D2 extends that to
  the results boot state, where the substrate is a stand-in for the
  field, not a mesh under review).

### D3 — Color roles, per canonical palette

`substrate_color` / `substrate_edge_color` stop being one-size
dataclass defaults and get per-theme values (no literals at call sites;
all through `Palette`, per ADR 0087 G-HEX discipline):

| Role | catppuccin_mocha | neutral_studio | paper |
|---|---|---|---|
| Substrate fill (`substrate_color`) | `#bfbfbf` | `#bfbfbf` | **`#d9d9d9`** |
| Substrate edge (`substrate_edge_color`) | **`#16161c`** (near-black, warm-cool matched to Crust) | **`#1a1a1a`** | **`#3c3c3c`** |
| Outline (`outline_color`, existing) | `#11111b` (unchanged) | `#000000` (unchanged) | `#000000` (unchanged) |
| Node cloud (when toggled on) | `node_accent` (unchanged) | `node_accent` (unchanged) | `node_accent` (unchanged) |

- The remaining seven built-ins inherit the dark-theme values
  (`#bfbfbf` / near-black edge tinted toward each theme's base hue at
  the implementer's discretion under the same rule: **edge luminance
  must sit below the ambient-lifted shading floor**, i.e. darker than
  ≈ RGB 100). User JSON themes keep loading — both fields carry
  defaults (the `substrate_color` precedent, ADR 0087 D1.1a).
- The cool-tinted alternative fill `#b4bac8` (`33_mocha_cool_full`) is
  recorded as viable but **not** the default: theme identity lives in
  chrome/background per the aesthetic doc v2 note, and neutral gray
  keeps contour colormaps honest on every theme.
- **Edges over fields:** while any field diagram (contour, isochrone
  map) is active on a geometry, that geometry's wireframe drops to
  `outline_color` at `CONTOUR_EDGE_OPACITY` (0.40) and the outline to
  2.0 px, so the field reads first and the discretization second
  (`50_contour_edges40`; 0.25 too faint on dense meshes, edges-off
  loses the FE-ness — both bracketed on SHEET_3). Node cloud stays
  off. Restored when the last field diagram detaches.
- **Colormap default routing:** `ContourStyle.cmap` default changes
  from `"jet"` to deferred-to-palette (`cmap_seq` / `cmap_div` by field
  signedness), closing the §7 policy violation at the dataclass level.

### D4 — Interaction rules

- **MotionLOD:** the results viewer's LOD set stays *node cloud only*.
  The new outline actor is a **static** feature-edge extraction —
  camera-independent, zero per-frame cost — precisely so it does NOT
  need LOD hiding (the model viewer's `vtkPolyDataSilhouette` actors
  recompute per camera tick and are LOD-hidden for that reason;
  results substrates are single closed surfaces where static feature
  edges and the true silhouette coincide well enough). If a future
  measurement shows otherwise, the outline joins the LOD set before
  any degradation of the wireframe.
- **Deformation:** the outline must follow the deformed surface. The
  DEFORM pump already scatters deformed points onto the render
  surface; the outline either shares those point buffers or is
  re-extracted on step change (implementation's choice; SHEET_3 panel
  B shows the re-extracted result). Ghost references keep
  `GHOST_OPACITY = 0.3` (unchanged).
- **User overrides vs defaults:** `Preferences.mesh_line_width` and
  `node_marker_size` remain the Display-panel knobs and always win
  over the defaults above; only their *default values* change (3.0 →
  1.0; wiring 10.0 → 6.0). A user-set width applies to the interior
  wireframe only — the outline width is a theme value, not a
  preference (one knob, one meaning). Theme switching retints
  fill/edge/outline live (existing `_refresh_substrate_colors`
  subscription) but never resizes user-sized things.
- **Session persistence:** `show_nodes` continues to round-trip; a
  restored session's value beats the D2 boot default.

### D5 — Acceptance checklist (implementing phase)

1. Boot the results viewer on `column_demo.h5` under each canonical
   palette: outline 2.5 px, interior edges 1 px, **no node dots**;
   side-face pixel luminance ≥ 100 (ambient lift applied).
2. Toggle "Show nodes" on: glyphs at 6 px in `node_accent`; MotionLOD
   still hides them during orbit; off leaves no ghost.
3. Attach a contour: wireframe drops to 40 % `outline_color`, outline
   2.0 px, field visibly dominant; detach restores D1 widths/colors.
4. Enable deform, scrub to last step: outline hugs the deformed
   surface (no reference-configuration outline left behind).
5. Theme switch mocha → paper live: fill/edge/outline retint per D3
   with no restart; Display-panel width overrides survive the switch.
6. Mesh viewer boot unchanged (nodes still on); model viewer
   unchanged; `pytest tests/viewers -q` green.
7. `ContourStyle.cmap` no longer defaults to `"jet"`.
8. `internal_docs/architecture/apeGmsh_aesthetic.md` updated per the
   Amendments below (aesthetic doc rule: rendering PRs update the doc).

### Amendments to `apeGmsh_aesthetic.md`

1. §2.3 (results viewer): add the substrate presentation — silhouette/
   feature outline present by default; interior mesh edges 1 px; node
   cloud off by default at boot (extends the existing "hidden by
   default when field is on").
2. §2.2/v2 note ("black wire" CAD-neutral fills): record the ambient
   floor — black-family edges are legible only with the substrate
   ambient lift (0.35); pure-diffuse shading buries them (measured
   72–88 RGB side faces).
3. §5.2 matrix, results column: "mesh lines dim by default" is made
   precise — full-strength 1 px when no field is shown; 40 % over
   fields. The 20 % figure in §2.3 is superseded by the measured 40 %
   (20/25 % vanishes on dense dark-theme meshes; SHEET_3 panel C).

## Alternatives considered

- **Silhouette-only, no interior edges** (`05_mocha_hier_silhouette`,
  `22_contour_edges_off`): clean product-render look, but the results
  viewer's subject includes the discretization; hiding it by default
  hides FE information. Rejected as default; reachable via the
  Display-panel width/opacity knobs.
- **Demote edges by opacity instead of width** (pass-1 attempt, 60 %
  opacity): opacity blends toward the face color, which is exactly the
  metamerism that broke #444444 — the edges vanished. Width + darker
  hue demotes without erasing. Opacity is used only *over fields*,
  where the background is the (bright) colormap, not the shaded fill.
- **Lighter-than-fill edges** (aesthetic doc §4.1 original rule;
  `32_mocha_light_edges`): works on shaded faces, fails on lit faces;
  a per-face adaptive edge color would fix it at the cost of a custom
  shader or per-cell edge coloring — machinery this problem does not
  justify. The v2 CAD-neutral direction (dark edges) is kept and made
  viable by the ambient lift instead.
- **`vtkPolyDataSilhouette` for the outline**: true view-dependent
  silhouette, but ~107 ms/frame on large meshes (measured previously in
  `core/navigation.py`) and would force LOD hiding — the outline would
  disappear exactly while the user orbits. Static feature edges are
  free at frame time and stable during motion.
- **A new `ViewportTokens` dataclass** for the two aesthetic constants:
  speculative — two constants with a documented precedent
  (`GHOST_OPACITY`) do not justify a fourth token home (ADR 0087
  rejected the same unification for the chrome).
- **Keeping nodes on but subdued** (`03_mocha_nodes_subdued`): at any
  size/color quiet enough not to pollute the render, the dots are pure
  noise — they no longer *answer* the "where are the nodes" question.
  Off-with-a-toggle is honest; subdued is decoration.

## Consequences

**Positive.** The substrate finally has figure-quality hierarchy: body
silhouette > mesh > (optional) nodes, on every theme, with the field
always winning when one is shown. All knobs trace to existing homes
(`Palette`, `Preferences`, module constants), so no new machinery and
no new guard classes; ADR 0087's G-HEX already polices the call sites.
The A/B sheets give the implementing PR pixel-level acceptance targets
rather than taste debates.

**Negative / accepted.** New-boot behavior changes (nodes gone, thinner
edges) will surprise existing users for one session; the Display panel
and Geometry panel toggles are the escape hatch, and saved sessions are
honored. The static outline is an approximation of a true silhouette on
open/concave geometries (acceptable for FE substrates; revisit if
section cuts expose it). Two aesthetic constants live as module
constants rather than theme fields — a deliberate simplicity bet that
loses only if a theme needs to vary them, at which point promotion to
`Palette` (with defaults) is mechanical. Per-theme
`substrate_edge_color` values add three cells to every future theme
review (ADR 0087 INV-6 canonical-palette bar applies).
