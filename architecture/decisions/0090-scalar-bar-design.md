# ADR 0090 — Scalar bar design: typography, container chip, per-theme color routing

**Status:** Proposed (2026-08-04) — awaits user veto. Design only: this
ADR changes no `src/`; it is the pixel-level contract for the
implementing phase. Third leg of the viewer design series — ADR 0087
governs the Qt chrome, ADR 0089 the model presentation in the viewport;
this one governs the **colormap legend**, the one annotation that lives
in the viewport but reads like chrome. It sits entirely on top of
ADR 0081 (`LegendController`, content-derived boxes, priority-12
interactor, session persistence) and changes none of its invariants —
0081 solved *where the legend is and how it behaves*; this ADR decides
*what it looks like*.

## Context

ADR 0081 made the scale owned, placeable, durable and correctly sized —
and left its face exactly as pyvista shipped it. The result (evidence:
`B_contour_deformed.png`, `SHEET_3_contour_over_field.png` from the
0089 review, reproduced as panel `T0_current` in this ADR's sheets):
a tiny raw-key title ("displacement_z"), dim cramped tick labels, no
frame or backdrop, floating over whatever the scene happens to be.

### Measured defects (2026-08-04, live actor inspection + GL renders)

- **E1 — The legend text is unthemed, and *which* wrong color you get
  depends on window creation order.** Measured on the booted results
  viewer under `catppuccin_mocha`:
  `bar.GetLabelTextProperty().GetColor() == (0.0, 0.0, 0.0)` — black
  labels on a dark vignette. Root cause is an ordering bug in
  `ui/viewer_window.py`: the `QtInteractor` is constructed (line 263)
  **before** `pv.set_plot_theme("dark")` / `global_theme.font.color =
  "white"` run (lines 270–271), and pyvista snapshots the global theme
  at plotter construction — so the first window of a process keeps
  pyvista's stock document theme (black text), while every *later*
  window picks up the mutated global ("white" — which is then wrong on
  `paper`). No path retints the bar text on `THEME.set_theme`; the
  backend's `add_scalar_bar` (`backends/pyvista_qt.py:704`) passes no
  colors at all.
- **E2 — The title is the storage key.** `LegendController._title_of`
  returns the raw component (`displacement_z`) — implementation naming
  as user-facing text, the exact class of defect ADR 0087 INV-5 exists
  to outlaw.
- **E3 — No container.** The ramp and labels float directly on the
  scene. Over the substrate this merely looks unfinished; over a bright
  contour field the horizontal legend's labels and title sit on the
  field itself (panel `H0_horizontal_current`: the title is flatly
  unreadable on a yellow region).
- **E4 — discovered while rendering: the VTK-native chip cannot cover a
  horizontal legend.** `vtkScalarBarActor.SetDrawBackground` paints the
  actor's own box only, and ADR 0081 L0 moved the horizontal title into
  a reserved band *outside* that box (drawn by the backend as a
  separate `vtkTextActor`). Panel `H1_horizontal_proposed`: chip under
  the ramp and labels, title orphaned on the field. Any chip must be
  drawn by the projection as its own backdrop, not delegated to the bar
  actor.
- **E5 — discovered while rendering: VTK's NaN / out-of-range swatches
  explode a content-derived box.** `DrawNanAnnotation` and the
  above/below range swatches re-layout *inside* the box the controller
  sized for ramp + labels only: the ramp collapses, the swatch takes a
  third of the height, and the annotation text draws left of the box
  and clips (panels `R2b_nan_only`, `R2_banded_nan`). Swatches are not
  a poke; they are a `box_px` accounting feature.

### Evidence — rendered A/B study, 2026-08-04

GL renders of the 3D column demo (`column_demo.h5`, contour + deform at
last step — the ADR 0089 SHEET_3 state), captured by mutating the live
`LegendController` / `vtkScalarBarActor` in-process; zero `src/`
changes. Harness committed as
`scratchpad_shots/render_scalar_bar_sheets.py` +
`compose_scalar_bar_sheets.py`; sheets untracked in
`scratchpad_shots/`:

- **`SHEET_10_typography_and_ticks.png`** — current vs text-role color
  vs mono labels + designed title; `%#.3g` vs `%.2e` vs 7 ticks vs
  `font_scale` 1.25.
- **`SHEET_11_container_chip.png`** — naked vs VTK `DrawBackground`
  chip (0.85 / 0.92) vs the proposed backdrop-actor chip with 8 px
  outer pad; paper current vs paper chip (`surface0` hairline refuted,
  `surface2` chosen).
- **`SHEET_12_ramp_and_swatches.png`** — continuous vs 10-band LUT
  (field and bar banded *together*); the swatch failures (E5) on
  record.
- **`SHEET_13_horizontal_and_multi.png`** — horizontal current /
  VTK-chip defect (E4) / backdrop chip at 0.85 and 0.92; two legends
  current vs proposed.
- **`SHEET_14_proposed_context.png`** — the full proposal in context
  under all three canonical palettes.

Failed variants kept on the record: `%.2e` labels (`T4`) are uniformly
noisy — printf `g` already switches to scientific exactly when
magnitude demands it; 7 ticks (`T3`) produce ragged values
(−0.0133, −0.0267 …) because `vtkScalarBarActor` spaces ticks evenly
and cannot take controller-chosen "nice" values; a `surface0` hairline
on `paper` (`C3_paper_chip`) is invisible at 1 px against `#FAFAFA`;
VTK-native chips and swatches per E4/E5.

## Decision

### D1 — Text: palette-routed color, mono values, designed title

| Element | Rule | Value |
|---|---|---|
| Label + title color | `palette.text`, retinted live on theme switch | mocha `#cdd6f4` · neutral_studio `#d0d0d0` · paper `#202020` |
| Label font | monospace (ADR 0087 D1.3 value role) | VTK Courier (see Accepted below) |
| Title font | sans (VTK Arial), the bar's existing weight | unchanged |
| Title size | `BASE_TITLE_PT` 14 × `font_scale` | unchanged (ADR 0081) |
| Label size | `LABEL_PT_RATIO` 0.85 × title | unchanged (ADR 0081) |

- The legend is the **one deliberate crossing** of a chrome text role
  into the viewport (the viewport HUDs already are Qt chrome; the
  legend must render in-scene per ADR 0081/0042, so the role crosses
  the seam as data). Ownership per the existing pattern: the viewer's
  `THEME.subscribe` callback pushes palette-derived values into the
  controller (a `set_theme_colors(...)` mutator, same shape as
  `_refresh_substrate_colors` for the substrate), the controller
  carries them on `ScalarBarSpec`, the backend projects and consults no
  theme (ADR 0081 Part 3 discipline). G-HEX applies at the viewer call
  site: palette fields only.
- **Fixes E1 as a side effect** — the spec carries the color, so the
  plotter-construction-order bug stops mattering for legends. The
  ordering defect in `viewer_window.py` itself is still worth the
  two-line fix (move the theme lines above the `QtInteractor`
  construction) for every *other* pyvista-themed default; noted for
  the implementing phase, not load-bearing here.
- **Title** (fixes E2): a display-name map from component key to
  designed title — split on `_`, sentence-case the words, uppercase a
  trailing axis token: `displacement_z` → "Displacement Z",
  `stress_von_mises` → "Stress von Mises" (particles stay lower;
  the map is a function, not a lookup table, with an override hook for
  irregulars). Display only — `LegendKey` and the ADR 0081 L3 snapshot
  key stay the raw component, so persistence and the
  concurrent-geometry prefix collapse (`_title_of`) are untouched; the
  geometry prefix, when shown, prefixes the designed title.
- **No units line.** The viewer is unit-agnostic (aesthetic doc /
  ADR 0087 INV-5: show none rather than a wrong one). When the results
  schema someday carries units metadata, the title gains a second muted
  line; designed then, not speculatively now.

### D2 — Number format: `%#.3g` default

`fmt` defaults change `"%.3g"` → `"%#.3g"` (the nine style dataclasses
in `diagrams/_styles.py:87…711`, plus `LegendEntry.fmt` and the
`ScalarBarSupport` fallbacks).

- The `#` flag keeps trailing zeros: −0.02 / −0.04 / −0.0800 ragged
  columns become 0.00 / −0.0200 / −0.0400 / −0.0800 — aligned decimal
  columns under the mono label font (panel `T6` vs `T2`).
- The `g` conversion **is** the engineering-notation policy: fixed
  notation while the exponent is in [−4, precision), scientific
  outside — 3 significant figures, `1.23e-05` when the field is small,
  `1.23e+06` when it is large. No custom per-range switcher is built
  (`T4`'s always-scientific `%.2e` is the refuted alternative).
- Tick count stays **5** (`n_labels`), the `T3` evidence: VTK spaces
  ticks evenly across the range, so more ticks means uglier values,
  and controller-side nice-number ticks are unrepresentable in
  `vtkScalarBarActor`. Not pursued.
- Per-legend `fmt` and the "Number format…" context-menu path are
  unchanged; a session-restored fmt beats the new default (existing
  ADR 0081 L3 behavior).

### D3 — Container: a backdrop chip, drawn by the projection

The legend gets the PickReadoutHUD glass-card language — and because of
E4 it is **its own 2D backdrop pair** (fill quad + hairline outline) at
VTK background display location under the bar, sized to the **full
legend box including the horizontal title band**, never
`vtkScalarBarActor.SetDrawBackground`:

| Token | Value | Note |
|---|---|---|
| Chip fill | `palette.mantle` at **0.92** alpha | the QSS `_rgba(p.mantle, 0.92)` glass-card value; 0.85 lets a bright field glow through the title band (`H2` vs `H3`) |
| Chip border | 1 px hairline, per-theme (D5 table) | `surface0` on dark themes, `surface2` on light — `surface0` measured invisible on `paper` (`C3`) |
| Outer pad | **8 px** beyond the content box on all sides | inner `PAD_PX` 6 unchanged; the 8 px is what makes the card read (`V2` vs `C1`) |
| Corners | square in v1 | VTK 2D has no rounded rect; tessellating one is an optional refinement, not a blocker |
| Default | **on**, viewer-wide | the chip is what makes the legend read on any scene (`SHEET_14`); no per-legend toggle in v1 — no new state |

**Footprint rule (the one layout change):** the slot allocator,
viewport containment, hit-testing and L-NOOVERLAP all operate on the
**chip rect** (content box + 8 px pad), not the content rect.
Without it, two docked chips overlap: `GUTTER_PX` 14 < 2 × 8 pad.
Concretely: `layout()` and `_contain` grow the box by the pad;
`LegendInteractor` hit-tests the chip rect (a slightly larger grab
target — strictly an improvement); MARGIN_PX 16 already clears the pad.

### D4 — Geometry: blessed as-is, recorded

ADR 0081's content-derived box **is** the geometry policy; this ADR
records the resulting tokens as the reviewed defaults and changes none
of them:

| Token | Value | Home |
|---|---|---|
| `BASE_TITLE_PT` / label ratio | 14 / 0.85 | `core/_legend.py` |
| `n_labels` | 5 | `LegendEntry` |
| `MIN_RAMP_PX` | 14 | `core/_legend.py` |
| `PAD_PX` / `GUTTER_PX` / `MARGIN_PX` | 6 / 14 / 16 | `core/_legend.py` |
| `MIN_FONT_SCALE` / `MAX_FONT_SCALE` | 0.5 / 3.0 | `core/_legend.py` |
| Edge snap / re-dock | `SNAP_PX` 14 / `REDOCK_PX` 48 | `core/_legend_interactor.py` |

Resulting default footprint, measured at 1356 × 842: a vertical legend
at `font_scale` 1.0 is ≈ 130 × 155 px (≈ 10 % × 18 % of the viewport)
— about a sixth of the viewport height, docked mid-right. Judged right
on `SHEET_14`; `font_scale` (wheel-resize, Display-panel preference,
per-legend override) remains the user's knob, `T5` shows 1.25.
Horizontal proportions (label run × `title band + ramp + label line`)
unchanged; the D3 chip is what fixes the horizontal variant's real
problem (E3/E4), not new proportions.

### D5 — Per-theme table

No new `Palette` fields. All values route through existing roles;
remaining built-ins and user JSON themes follow the same role mapping
(light themes take the light column by their base luminance):

| Role | catppuccin_mocha | neutral_studio | paper |
|---|---|---|---|
| Legend text (`text`) | `#cdd6f4` | `#d0d0d0` | `#202020` |
| Chip fill (`mantle` @ 0.92) | `#181825` | `#1f1f1f` | `#F5F5F5` |
| Chip border | `surface0` `#313244` | `surface0` `#2a2a2a` | **`surface2` `#C0C0C0`** |
| Colormap (unchanged, 0089) | viridis | viridis | cividis |

### D6 — Ramp presentation: continuous default, banding is a LUT fact

- **Continuous stays the default.** Banding is an analysis choice, not
  a style default.
- **Banded display = `LutSpec.n_colors`** (field already exists,
  default 256). Setting it to N ≤ 32 bands the *field and the bar
  together* (panel `R1_banded10` — the honest version of what
  engineers ask "discrete bands" for; a banded bar over a continuous
  field is a lie and is not offered). The bar projection mirrors it via
  `SetMaximumNumberOfColors(n_colors)` when `n_colors` ≤ 32. Surface:
  the colour-map editor gains a "Bands" control writing `n_colors`
  through the existing LUT mirror; the legend itself needs no new
  state and no new event.
- **NaN / out-of-range swatches: deferred, with the accounting named.**
  E5 shows the VTK annotations destroy a content-derived box. Proper
  support = `box_px` grows by one swatch row (+ gap) per enabled
  swatch and the annotation column joins the width budget, plus
  below/above colors routed from the LUT editor — a self-contained
  future slice that becomes worth shipping when 0089's threshold work
  makes out-of-range values common. Until then NaN/threshold
  communication stays with 0089's mechanisms (blackout toast, gated
  eye).
- **No hover/pick band highlight.** Requires a custom bar renderer for
  a decoration; out.

### D7 — Multi-legend: 0081 slots + per-legend chips

The ADR 0081 stacking rules (docked slots along the right edge,
leftward, wrap; horizontal along the bottom, upward) are blessed
unchanged — `M0`/`M1` show them working. Each legend carries its own
chip (one card per scale, `M1`); the D3 footprint rule keeps adjacent
chips separated by `GUTTER_PX − 2·pad_chip` ≥ visible gap — with the
current tokens, docked chip edges sit 14 px apart *by chip rect*,
which reads as two cards. No merged mega-chip for stacks (rejected: a
shared card would re-couple legends the controller deliberately keeps
independent, and re-layout on any member change would move the
others).

### D8 — What changes where (implementing phase)

| Home | Change |
|---|---|
| `viewers/core/_legend.py` | `set_theme_colors(...)` mutator (fires `LEGEND_CHANGED` once, idempotent-skip like every 0081 mutator); title display-name function in `_title_of`; layout/containment on chip rects (D3); spec gains the resolved chip rect + colors |
| `viewers/scene_ir/_layers.py` | `ScalarBarSpec` widens: `text_rgb`, `label_mono: bool`, `chip_rect`, `chip_rgb`, `chip_alpha`, `chip_border_rgb` (all with defaults — trame and tests keep constructing bare specs) |
| `viewers/backends/pyvista_qt.py` | project text color + Courier label family onto the bar and the horizontal title actor; draw/remove the backdrop pair per spec; `move_scalar_bar` fast path moves the backdrop in the same frame (a drag must not shear the card off its content) |
| `viewers/diagrams/_styles.py` | nine `fmt` defaults `"%.3g"` → `"%#.3g"` |
| `viewers/results_viewer.py` (+ mesh/model shells when they grow legends) | `THEME.subscribe` push → `legends.set_theme_colors(palette.text, palette.mantle, border_role(palette))`; unsubscribed in the existing teardown loop (the 0081 L1 leak lesson) |
| `viewers/ui/viewer_window.py` | the E1 two-line ordering fix (theme before `QtInteractor`) |
| `viewers/ui/_color_map_editor.py` | "Bands" control → `LutSpec.n_colors` (D6) |
| QSS / `theme.py` | **no change** — no new palette fields, no new selectors |
| Session schema | **no change** — no new per-legend state (chip is viewer-wide, colors are theme-derived, fmt already persists) |

ADR 0081 invariants explicitly preserved: content-derived box (chip is
derived *from* it, never sized independently); owner-fires-events (the
theme push is one mutator, one fire); geometry-only drag fast path
(extended to the backdrop, not bypassed); abort-only-on-hit
(hit-testing a slightly larger rect changes no miss behavior);
L-STICKY/L-GEOM/L-FIT semantics (L-NOOVERLAP and containment move to
chip rects).

### D9 — Acceptance checklist (implementing phase)

1. Boot the results viewer on `column_demo.h5` + contour under each
   canonical palette: label/title color == `palette.text` (measure the
   text property, not the screenshot), mono labels, designed title
   ("Displacement Z"), chip visible with the D5 border; **first**
   window of the process — the E1 ordering trap.
2. Live theme switch mocha → paper: legend text and chip retint
   without a rebuild losing placement (anchor unchanged — L-STICKY
   extended to the theme push).
3. `%#.3g` renders aligned trailing-zero columns; "Number format…"
   still overrides per legend and survives the session round-trip.
4. Horizontal legend: chip covers title band + labels + ramp as one
   card (E4 regression); drag moves card and content together with no
   one-frame shear.
5. Two contours on two components: two cards, no chip overlap at any
   window size (L-NOOVERLAP on chip rects), gutter visibly separates
   them.
6. Colour-map editor "Bands" at 10: field and bar band together
   (`R1` parity); back to 256 restores continuous.
7. Legend drag/resize/re-dock/context menu all behave bit-for-bit as
   before on the chip rect; pick and navigation suites green.
8. Trame backend renders text colors + chip from the same specs
   (extend `test_the_trame_backend_renders_legends_too`).
9. `pytest tests/viewers -q` green; no new G-HEX hits (colors reach
   the seam only as palette-derived spec fields).

### Amendments to `apeGmsh_aesthetic.md`

1. §5 (annotations): the colormap legend adopts the glass-card
   language — `mantle` @ 0.92 fill, hairline border, 8 px pad — and
   chrome `text` for its type; it is the sanctioned chrome-role
   crossing into the viewport, alongside the HUDs.
2. §7 (colormaps): banded display is a LUT property (`n_colors`),
   never a legend-only presentation; a banded bar over a continuous
   field is prohibited.

## Alternatives considered

- **`vtkScalarBarActor.SetDrawBackground/SetDrawFrame` for the chip.**
  One call, no new actor — refuted by E4: it cannot cover the
  horizontal title band, and its padding is not controllable. Panels
  `C1`/`H1` are the record.
- **Qt overlay widget for the chip** (draw the card in chrome, leave
  the bar naked). Breaks ADR 0042/0081 alternative-3 reasoning all
  over again: no card in trame, screenshots or exports. The card must
  live in the scene projection.
- **White text everywhere** (finish the `viewer_window.py` intent).
  Wrong on `paper` and any future light theme; the role exists.
- **JetBrains Mono via `SetFontFile`** for exact chrome parity.
  Accepted as a refinement hook, rejected as the v1 mechanism: it
  introduces a font-file path dependency into the render seam for a
  10 pt label whose family (fixed-pitch) matters far more than its
  face. VTK's built-in Courier delivers the alignment (`T2`/`T6`).
- **7 ticks / nice-number ticks.** `T3`: VTK's even spacing makes 7
  ticks uglier than 5; nice-number ticks require tick *values* the
  actor cannot accept. A custom tick-drawing bar is not worth the
  maintenance for a cosmetic gain over 5 evenly spaced ticks with
  aligned formatting.
- **Always-scientific `%.2e`.** `T4`: uniform noise for mid-range
  fields; `%#.3g` gives scientific exactly when warranted.
- **Per-legend chip toggle + persisted chip state.** New state, new
  snapshot field, new menu item — for a card whose whole point is
  consistency. The viewer-wide default can grow a preference later
  without schema impact.
- **Banded bar via `SetMaximumNumberOfColors` alone** (bar banded,
  field continuous). Actively misleading; rejected outright. Banding
  goes through the LUT both ways or not at all.

## Consequences

**Positive.** The legend stops being the last unthemed object in the
viewport: text from the palette, card from the HUD language, title in
the user's vocabulary, labels in aligned mono columns — under every
theme, over any scene, both orientations, N legends (SHEET_14 is the
target picture). Every change is a projection-side or display-side
concern: the 0081 state model, session schema, event matrix and
interactor survive byte-identical, so the implementing PR is contained
(seven files, one new spec shape, zero schema bumps). E1's
order-dependent text color — a real heisen-defect — dies as a side
effect of routing colors through specs.

**Negative / accepted.** The chip rect footprint slightly enlarges
docked stacks (2 × 8 px per legend) — at the current tokens two
vertical legends still fit any viewport ≥ ~360 px wide, and the wrap
path covers the rest. VTK Courier is not the chrome's JetBrains Mono —
close at 10–12 pt, not identical; parity is a named refinement. Square
chip corners deviate from the chrome's 6 px radius until someone cares
enough to tessellate. The swatch deferral (D6) leaves NaN/out-of-range
values without a legend affordance for now — 0089's viewport feedback
carries that weight, and the box-accounting design is on file for the
slice that adds it.
