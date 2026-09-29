# ADR 0087 — Viewer visual design system: tokens, component rules, enforcement

**Status:** Accepted (2026-08-04) — the standing rulebook for the
two-phase Qt-viewer polish now being implemented, and for every future
agent or contributor touching `viewers/ui/**`. Chrome-side companion to
`internal_docs/architecture/apeGmsh_aesthetic.md` (which governs the
*viewport*: shading, palettes, colormaps — untouched here). Builds on
ADR 0056 / ADR 0084 for the enforcement pattern
(`tests/viewers/test_viewer_state_contract.py` AST guards with two-way
ratcheted budgets).

## Context

The viewer chrome grew feature-by-feature, each panel styled by whoever
shipped it. The result is competent theming infrastructure
(`viewers/ui/theme.py`: a frozen `Palette` dataclass, ten built-in
palettes, an observable `THEME` singleton, and `build_stylesheet()`
generating app-wide QSS keyed by `objectName`) undermined by
"engineer-designed" surfaces: duplicated headers, bare-letter toolbar
buttons, unthemed widgets falling through to Windows white, forms with
no visual structure, empty states that look live, and labels that name
the implementation rather than the user's concept.

Three non-color token homes already exist and are good — they are
under-used, not wrong:

- `viewers/ui/_layout_metrics.py` — frozen `LayoutMetrics` (`LAYOUT`
  singleton): dock min/initial sizes, header heights, corner radii
  (4/2/6), separator widths. Density-*independent* metrics.
- `viewers/ui/density.py` — `DensityTokens` (`DENSITY` singleton):
  row height, `pad_x`/`pad_y`/`gap`, `fs_body`/`fs_head` in compact
  (22/8/4/4/11.5/11) and comfortable (28/12/6/6/12.5/12) sets.
  Density-*dependent* metrics.
- `viewers/ui/_eye_icon_delegate.py` — the house icon technique:
  DPI-aware `QPainter` glyph pixmaps, cached per (color, dpr), colored
  from the palette. No icon files, no font-fallback risk.

What is missing is not machinery but *law*: which home owns which kind
of value, what the type scale is, what a compliant panel looks like,
and a guard that stops the next contributor from hardcoding `white`
because it looked fine on their monitor.

### Evidence — GUI audit, 2026-08-04 (screenshots in the polish
### worktree's `scratchpad/gui_audit/`)

- **`10_dock_Outline.png`** — the dock's Qt title bar says "Outline";
  6 px below it an inner header repeats "OUTLINE" in letter-spaced
  all-caps with its own `+` button. Two titles, one panel.
- **`10_dock_Geometry.png`** — dock title "Geometry", inner label
  "Geometry" again; a "Tied to" combo (implementation jargon for the
  deformation field); Min/Max fields showing a live-looking
  `0.000000` while their governing "Threshold" checkbox is off — the
  form reads as *configured to zero*, not *inactive*.
- **`10_dock_Session.png`** — a dock named "Session" that contains
  label toggles, point size, line width, and the theme picker: display
  preferences, nothing session-scoped.
- **`10_dock_Output.png`** — a white unthemed text console on the dark
  theme (`_output_dock.py` styles it locally instead of joining the
  QSS).
- **`20_toolbar_0.png`** — camera-preset buttons rendered as bare
  letters: T / Bo / F / Bk / L / R. Unreadable without prior
  knowledge; "Bo"/"Bk" are abbreviations invented to disambiguate a
  bad idea.
- **`00_full_window.png`** — menu bar ordered File / Help / View; the
  Definitions panel showing the API signature
  `ops.<family>.<Type>(..., name=...)` as permanent panel furniture in
  its empty state.
- Code sweep: 32 `setStyleSheet(` call sites across 18 files in
  `viewers/ui/` outside `theme.py`; ~50 literal color tokens (hex or
  named) outside `theme.py`, e.g. `constraints_tab.py` (14),
  `loads_tab.py` (7), `_output_dock.py` (4).

None of these are bugs a test currently catches, and each was
introduced by a competent contributor solving a local problem. That is
the definition of a problem needing a standing decision plus guards,
not another one-off cleanup.

## Decision

### D1 — Tokens: one home per kind of value

**D1.1 Color.** The `Palette` dataclass fields in `theme.py` are the
single source of color, blessed as-is. The chrome roles and their
meanings are frozen:

| Role | Use |
|---|---|
| `base` / `mantle` | window + panel background / bars, headers |
| `surface0/1/2` | borders + input bg / hover / pressed |
| `text` / `subtext` / `overlay` | primary / labels / muted + hints |
| `accent` | focus, selection spine, slider handles — never decoration |
| `icon` | toolbar + glyph icons |
| `success` / `warning` / `error` / `info` | semantic only, never "I needed a green" |

Rules: (a) new UI color needs must map to an existing role first;
adding a `Palette` field is a last resort and MUST carry a default (so
the ten built-ins and user JSON themes at `<config>/apeGmsh/themes/`
keep loading — the `substrate_color` precedent). (b) No color literal
(hex, `QColor("white")`, named CSS color) anywhere in `viewers/ui/`
outside `theme.py` — enforced by G-HEX below. (c) Alpha variants go
through `_rgba(palette_field, a)` in the QSS factory, never baked hex.

**D1.2 Spacing.** A 4-px-based scale with named roles, all existing
values already on it:

| Step | px | Role |
|---|---|---|
| xs | 2 | hairline insets, icon padding |
| s | 4 | sibling gap (compact), dock separator, control padding-y |
| m | 8 | control padding-x (compact), gap between related controls |
| l | 12 | section padding, gap between sections |
| xl | 16 | dialog margins |

Ownership rule: a spacing value that changes with density comes from
`DENSITY` (`pad_x`, `pad_y`, `gap`, `row_h`); one that does not comes
from `LAYOUT` (which gains named fields as needed, per its own
documented promotion path). Ad-hoc integer margins in panel code are
review findings; the number must trace to a token or the scale above.

**D1.3 Type scale.** Five text roles, pixel sizes consistent with what
`build_stylesheet` already renders (px, not pt — the QSS is px-based
and stays so):

| Role | Size | Weight | Color | Notes |
|---|---|---|---|---|
| Dock title | 11 px | 600 | `text` | `QDockWidget::title` — the ONLY title a docked panel has |
| Section header | `fs_head` (11/12) | 600 | `text` | sentence case, inside panels |
| Label | `fs_body` (11.5/12.5) | 400 | `subtext` | form labels, tree rows |
| Value / readout | 10 px mono | 400 | `text` | numeric readouts, IDs, coordinates |
| Hint / empty | `fs_body` italic | 400 | `overlay` | empty states, shortcuts; 9 px `overlay` is the absolute floor (HUD micro-hints) |

The monospace stack is unified to one canonical string —
`"JetBrains Mono", "Cascadia Code", "Consolas", monospace` — replacing
the two variants currently in `theme.py`. Bold is a header weight,
never inline emphasis; letter-spaced all-caps text is retired with the
inner headers (INV-1).

**D1.4 Radius + hairlines.** `LAYOUT.corner_radius` 4 (default) /
`corner_radius_small` 2 (tight circles, close buttons) /
`corner_radius_large` 6 (containers, HUDs, dialogs) — blessed, no new
radii. Borders are 1 px: `surface0` for structure (panel edges,
separators), `surface1` for interactive affordances (inputs, buttons),
`accent` reserved for focus/active. The single legitimate 2 px border
is the active-selection spine (the `PlotPaneTabRow[active="true"]`
left-edge pattern); reuse that pattern for "this row is the active
one", nothing else.

### D2 — Component rules (each an invariant, checkable in review)

**INV-1 (One name).** A docked panel's `QDockWidget` title is its only
title. Inner header labels repeating or shouting the panel name
("OUTLINE", "Geometry" under "Geometry") are banned; the inner header
*strip* may survive as a toolbar row (buttons, filters) but carries no
name text. Exception: panels that render outside a dock (viewport
HUDs, floating cards, dialogs) own their single title themselves.
Corollary: a dock's name states the user's concept — the "Session"
dock is renamed to what it is (display preferences).

**INV-2 (Honest empty state).** A panel with nothing to show shows
exactly one muted hint line (hint role of D1.3, the
`DiagramSettingsEmptyHint` pattern), and its controls are disabled or
hidden. Never a form of live-looking `0.000000` fields, never enabled
combos over no data, never API signatures or implementation notes as
panel furniture. The hint says what to *do* ("Add a section plane to
cut the model"), not what is absent.

**INV-3 (Sections, not piles).** Multi-concern forms are built from
visually grouped sections. A section that is toggleable carries its
enable-toggle *in its header row* (checkbox + section title), and its
body controls enable/disable with it — no orphan checkboxes floating
between rows. Labels sit in a shared aligned column per panel
(`QFormLayout` with a consistent label width per dock, wide enough for
the longest label at comfortable density). One section pattern,
reused, so every settings panel reads the same way.

**INV-4 (Icons are drawn, once, one way).** No bare-letter or
invented-abbreviation buttons. All icons come from ONE technique: the
`_eye_icon_delegate.py` approach generalized into a small icon factory
(`viewers/ui/_icon_factory.py`) — vector `QPainter` drawing into a
pixmap at the widget's device-pixel-ratio, cached per
(glyph, color, dpr), stroked in `palette.icon`/`text`. Glyph metrics:
16 px design size in a 22 px hit area, stroke width
`max(1.0, size * 0.09)` (the shipped eye-glyph value), round caps, no
fills except state dots. Camera presets, clear/close, play/pause —
every toolbar glyph goes through the factory. No SVG asset folder, no
icon-font dependency, no emoji.

**INV-5 (Microcopy).** Labels name the user's concept, not the
implementation: "Field", not "Tied to"; "Display", not "Session".
Sentence case everywhere (labels, buttons, menu items, section
headers); no Title Case, no ALL CAPS. Numeric fields show units where
the model has them (the viewer is unit-agnostic per the aesthetic doc
— then show none rather than a wrong one, but scale factors say "×").
Menu bar order is File / View / Help.

**INV-6 (Themed by default).** Every new widget must render correctly
under the three canonical review palettes — `catppuccin_mocha`,
`neutral_studio`, `paper` — before merge; the other seven built-ins
and user themes follow from role discipline. The mechanism: style via
`objectName` + a block in `build_stylesheet()`, or via palette-derived
values pushed on a `THEME.subscribe` callback. Local
`setStyleSheet()` with interpolated palette fields is tolerated legacy
(budgeted by G-INLINE); local `setStyleSheet()` with literal colors is
banned outright (G-HEX).

**INV-7 (objectName ↔ QSS coupling).** Every `objectName` referenced
by a selector in `build_stylesheet()` must be set by exactly this
codebase (no dangling selectors), and any widget whose look deviates
from the bare widget-class defaults gets its styling through a named
QSS block rather than inline code. Naming convention:
`PascalCase`, suffixed by role (`…Header`, `…HeaderLabel`,
`…EmptyHint`, `…HUD`) so the QSS reads as a component inventory.

### D3 — Enforcement: four guards, one ratchet

A new guard test `tests/viewers/test_viewer_style_contract.py`,
sibling of `test_viewer_state_contract.py` and reusing its shape:
AST/text walk over `viewers/ui/**`, per-file allowlisted budgets,
**two-way ratchet** (over budget fails; under budget fails with
"ratchet the allowlist down"; unlisted files fail on first violation;
raising a budget requires citing this ADR in the allowlist comment).
Budgets are measured at guard-landing time, not guessed here.

- **G-HEX — no literal colors outside `theme.py`.** Walk string
  constants and `QColor(...)` calls in `viewers/ui/**` (excluding
  `theme.py`); flag hex colors (`#rgb`/`#rrggbb`/`#aarrggbb`) and
  named-color strings from a small denylist (`white`, `black`, `red`,
  `gray`, `grey`, `transparent` as a *color value* — the
  implementation may scope to strings inside `setStyleSheet` calls
  plus `QColor` constructor args to avoid false positives on prose).
  `theme_editor_dialog.py` (which legitimately manipulates hex text)
  enters with a measured budget; the target for everything else is a
  burn-down to 0.
- **G-SHOUT — no all-caps header labels.** Flag any string constant
  matching `^[A-Z][A-Z0-9 /]{3,}$` passed to `QLabel(...)` or
  `.setText(...)` in `viewers/ui/**` construction paths (catches
  "OUTLINE", "PLOTS", "DETAILS", "NEW PLOT"). Acronym allowlist
  (`FEM`, `HUD`, `ID`, `IDS`, axis triples like `X Y Z`) kept in the
  test, not scattered as inline pragmas. Target budget 0 at the end
  of polish phase 1 — this is the cheapest mechanical proxy for INV-1.
- **G-QSS-SYNC — no dangling QSS selectors.** Render
  `build_stylesheet(PALETTE_DARK)`, extract every `#ObjectName`
  selector token, collect all `setObjectName("…")` string literals
  under `viewers/ui/**` (+ `viewers/*.py` shells), assert selector ⊆
  set. One direction only — the reverse (every objectName styled) is
  unenforceable because objectNames also serve layout persistence and
  tests. Dead QSS is how the sheet rots; this keeps it a live
  inventory.
- **G-INLINE — `setStyleSheet` call-site ratchet.** Count
  `<expr>.setStyleSheet(...)` call sites per file (AST, same collector
  pattern as G-RENDER). Baseline ≈ 32 sites / 18 files enters the
  allowlist with per-file counts; every polish PR that migrates a
  widget to an `objectName` block ratchets its file down. `theme.py`'s
  own single site (the window host applying the sheet) is the
  designated writer.

What is deliberately *not* automated: INV-2/INV-3/INV-5 (empty-state
honesty, section structure, microcopy) are review-checklist items —
a guard that greps for `0.000000` would be theater. The screenshots
convention (a `gui_audit`-style capture attached to any PR that
touches panel layout) is the enforcement surface for those, and the
review checklist cites the INV numbers.

### D4 — Non-goals

- **No rebrand, no new palettes.** Ten built-ins is already three too
  many to review; the roster is frozen (user JSON themes remain open).
- **No light/dark redesign.** Palette values may be tuned per the
  aesthetic doc's rule 4 ("hex codes are v1 starting points"); the
  role structure and the three-canonical-palettes review bar do not
  change.
- **No icon library dependency.** Self-rendered glyph pixmaps only
  (INV-4). An SVG/icon-font dependency is a new ADR, not a PR.
- **Nothing in `viz/`, the web/trame viewer, or the viewport
  aesthetic.** Those are `apeGmsh_aesthetic.md` and ADR 0042 territory.
- **No Qt-widget framework swap, no QML.** The stack is locked per the
  design brief.

## Alternatives considered

- **Keep taste in review, skip the guards.** Rejected: the audit shows
  every violation class was introduced *past* review by competent
  contributors. The state contract proved AST budgets are the only
  ratchet that survives multi-agent development; style gets the same
  treatment or regresses the same way.
- **Adopt an icon set (Material, Lucide, qtawesome).** Rejected: a
  runtime dependency and a licensing surface for ~15 glyphs, visual
  language imported rather than owned, and the house already has a
  proven crisper technique (DPI-aware hand-rendered pixmaps, themed
  from the palette live).
- **Move all metrics into `Palette`.** Rejected: color varies per
  theme; spacing and type vary per *density*; conflating them forces
  every theme to restate layout. The three-home split
  (Palette / LayoutMetrics / DensityTokens) matches how the values
  actually vary.
- **A `tokens.py` rewrite unifying the three homes behind one facade.**
  Rejected as speculative abstraction: three small, typed, frozen
  dataclasses with clear ownership beat one indirection layer; the
  cost today is knowing which home to look in, which D1's ownership
  table settles.
- **Enforce "every objectName has a QSS block" (the reverse of
  G-QSS-SYNC).** Rejected: objectNames are also functional (layout
  persistence, tests, accessibility); the guard would force noise
  blocks or a pragma system. The forward direction plus the INV-7
  naming convention gets the value without the false positives.

## Consequences

**Positive.** A panel built by a future agent from this ADR alone
lands looking native: right title discipline, right empty state, right
section shape, themed under all palettes. The four guards make the
worst regressions (hardcoded white, shouted headers, dead QSS,
stylesheet sprawl) fail CI instead of shipping. The token tables turn
"looks off" review comments into citable rule numbers. The two-phase
polish now has acceptance criteria: phase PRs ratchet G-INLINE and
G-HEX down and take G-SHOUT to zero.

**Negative / accepted.** The guard allowlists are more maintenance
surface, and budgets must be honestly measured at landing (an inflated
baseline neuters the ratchet). G-HEX's named-color denylist will need
occasional false-positive grooming. INV-4 makes adding an icon a
~20-line factory function instead of dropping in an SVG — deliberate
friction, but friction. Renaming docks ("Session" → display
preferences) invalidates saved `QSettings` dock layouts keyed by
objectName unless the objectName is kept while the *title* changes —
implementation must keep persistence keys stable and change only
user-visible text. The unenforceable invariants (INV-2/3/5) still
depend on reviewers actually opening the screenshots.

---

## Appendix A — Icon vocabulary (2026-08-04)

The complete glyph roster of `viewers/ui/_icon_factory.py`. This table
is the INV-4 contract: **every icon-bearing control in `viewers/ui/**`
maps to a row here**, and a control that needs a glyph not in this
table adds it to the factory (one `_draw_*` function + registry entry
+ a row below) before wiring it. Phase 1 wired the toolbar families;
the "where used" column marks the remaining unicode/emoji call sites
Phase 2 rewires (file:line as of this appendix).

Evidence for the vocabulary review: contact sheets rendered per
canonical palette by `scratchpad_shots/render_contact_sheet.py`
(untracked; regenerate at will — it never touches the persisted
theme).

### Naming scheme

`snake_case`; family prefix where a family exists (`view_*`,
`probe_*`, `step_*` / `skip_*`, `move_*`); verbs for actions
(`close`, `add`, `flip`, `clear`), nouns for objects (`gear`,
`camera`, `dot`). A glyph is named for its *meaning*, never its shape
(`eye_off`, not `slashed_eye`) so a redesign never forces a rename.

### Metaphor registry (one metaphor, one meaning)

| Metaphor | Meaning | Never used for |
|---|---|---|
| Iso cube, highlighted face | camera preset | anything else 3D-ish |
| Eye (open / slashed) | visibility on / off | selection, focus |
| Corner brackets | view framing (`fit`, `isolate`) | cropping |
| Square cut by a diagonal | section / clip plane | slice *probe* (grid) |
| Gear | settings / preferences | actions with effects |
| Triangle | play | step (chevron), export (framed triangle) |
| Chevron | step ±1 | collapse/expand (trees use Qt's own) |
| Chevron + terminal bar | jump to first/last | — |
| Filled dot | state / series swatch | decoration |
| Cell + interior grid (`mesh`) | the FE mesh itself | sampling grids (`probe_slice`) |
| Cell + dots | where discrete quantities LIVE — on the corners (`nodes`) or inside (`gauss`) | any other point cloud |

Transport deliberately never distinguishes controls by *size alone*
(the audit's ▶ vs ▶︎ failure): play is a filled triangle, steps are
chevrons, jumps are chevron+bar — distinct silhouettes at 16 px.

### Glyph roster

| Glyph | Metaphor | Where used |
|---|---|---|
| `view_top` `view_bottom` `view_front` `view_back` `view_left` `view_right` | iso cube, face highlight placed by mirroring | left toolbar camera presets (wired) |
| `view_iso` | iso cube, no highlight | toolbar isometric preset (wired) |
| `ortho` | front square + offset rear square | toolbar ortho/perspective toggle (wired) |
| `fit` | corner brackets + center dot | toolbar fit view (wired) |
| `camera` | camera body + lens | toolbar copy screenshot (wired) |
| `save` | down arrow into tray | toolbar save screenshot to file (wired) |
| `image` | frame + mountain + sun dot | mesh/model save image (wired) |
| `eye` | open eye + pupil | Phase 2: clip-plane gizmo toggle `_clip_planes_panel.py:303` "👁". Tree-row visibility keeps `_eye_icon_delegate.py`'s renderer (tri-state); factory `eye`/`eye_off` are the *button* forms |
| `eye_off` | eye + slash | hide-selected toolbar (wired) |
| `isolate` | inward brackets + dot | isolate-selected toolbar (wired) |
| `reveal` | four outward arrows | reveal-all toolbar (wired) |
| `palette` | circle + diagonal dot triad | color-by-group toolbar (wired) |
| `colormap` | stepped alpha-ramp bar | color-mapping toolbar (wired) |
| `section` | square cut by diagonal, half filled | section-planes toolbar (wired) |
| `axes` | origin triad | local-axes toolbar (wired) |
| `stages` | two overlapping squares | stage-activation toolbar (wired) |
| `mesh` | square + interior grid + two diagonals (an FE triangulation) | ADR 0098 mesh-view "Mesh" style button, and the mesh pane's kind glyph (`viewers/session/_frame.py`) |
| `outlines` | closed boundary polygon, NO interior lines | ADR 0098 mesh-view "Outlines" style button. Distinct from `mesh` at 16 px by exactly what distinguishes the two pictures: interior grid vs border only |
| `nodes` | cell outline + three filled corner dots | ADR 0098 mesh-view "Nodes" style button. The dots carry the meaning; the faint cell places them as *corners of a cell* rather than `palette`'s free dot triad |
| `gauss` | cell with a 2×2 INTERIOR dot pattern | ADR 0098 mesh-view "Gauss" style button (integration-point locations). Distinct from `probe_slice` (a 3×3 line grid, no dots) and from `nodes` (dots ON the corners) — Gauss points sit inside the cell |
| `add` | plus | Phase 2: outline "+" buttons `_outline_tree.py:103`, `_model_outline_tree.py:155` |
| `close` | diagonal cross (12 px extent — close targets read lighter) | Phase 2: plane delete `_clip_planes_panel.py:315` "✕", HUD dismiss `_empty_state_hud.py:67` "×", plot-tab close `_plot_pane.py:310` "×", preferences chip `preferences.py:240/250` "×" |
| `gear` | ring + eight teeth + filled hub | Phase 2: background toggle `_bg_toggle_gear.py:14` "⚙" |
| `flip` | two opposing horizontal arrows | Phase 2: clip-plane reverse `_clip_planes_panel.py:293` "⇄" |
| `move_up` / `move_down` | full-shaft arrow | Phase 2: layer reorder `_diagram_settings_tab.py:760/765` "↑"/"↓" |
| `dot` | filled circle | Phase 2: plot-tab series dot `_plot_pane.py:296` "●" |
| `play` / `pause` | filled triangle / twin bars | Phase 2: `_time_scrubber.py:108/114` play toggle (checked state swaps to `pause`) |
| `step_back` / `step_forward` | chevron | Phase 2: `_time_scrubber.py:103/108` "◀"/"▶︎" |
| `skip_first` / `skip_last` | chevron + terminal bar | Phase 2: `_time_scrubber.py:98/119` "⏪"/"⏩" |
| `movie` | rounded frame + play triangle | Phase 2: export animation `_time_scrubber.py:198` "🎬" |
| `probe_point` | circled dot | Phase 2: `_viewport_hud.py:81` "◉" |
| `probe_line` | arc between endpoint dots | Phase 2: `_viewport_hud.py:82` "⌒" |
| `probe_slice` | 3×3 sampling grid | Phase 2: `_viewport_hud.py:83` "▦" |
| `stop` | rounded square, face fill | Phase 2: stop-active-probe `_viewport_hud.py` "✕" (NOT `close` — stopping a mode is not dismissing a thing) |
| `clear` | backspace pentagon + inner cross | Phase 2: clear-all-probes `_viewport_hud.py` "⌫" |

Not icons (left alone): "—" placeholder readouts, "Browse…"/"New…"
ellipsis buttons, prose hints ("⇧ click → add to plot" becomes text +
`add` glyph only if Phase 2 finds it worth it), and data fallback "?"
strings in count labels.

### Phase-1 refinements recorded here

- Face-highlight / state fill alpha raised 0.40 → 0.50 — at 16 px the
  0.40 fills (camera cubes, section, colormap ramp) sat too close to
  the background on `paper`.
- `gear` gained a filled hub dot: ring+teeth alone read as a sun.
- `palette` dots rearranged to a diagonal triad: two-eyes-plus-mouth
  read as a smiley at 16 px.
- Camera cubes judged legible at 16 px on all three canonical
  palettes after the alpha bump: the flipped variants (`view_bottom`,
  `view_back`) are the weakest of the family but the filled face
  still reads; revisit only if user reports confirm.

---

## Appendix B — Menu architecture (2026-08-04, design only)

Target menu structure for the three viewers. **Exists** = shipped
behavior at this appendix's commit; **proposed** = Phase 2 (or later)
work. Rules: menu order File / View / Help (INV-5); sentence case
(INV-5) — existing Title-Case items are case-fixed in Phase 2 without
changing behavior; every item that has a keyboard shortcut declares
it via `QAction.setShortcut` so the text renders right-aligned in the
menu (today only Help → Shortcuts does this).

Ownership rules:

- **File** owns things that produce or consume files: open, import,
  export (images, animations, STEP), close. Global preferences also
  land here (bottom, after a separator) — pragmatic: no one-item Edit
  menu just to host it.
- **View** owns everything that changes what you see without changing
  data: dock toggles, layout (reset, focus mode), camera, orbit axis,
  theme. One-item menus are banned — the model viewer's "Info" menu
  retires and its item joins View.
- **Help** owns shortcuts and nothing else until there is more help.

### Results viewer

| Menu | Item | Action | Shortcut | Status |
|---|---|---|---|---|
| File | Open results… | open-results dialog → new window | Ctrl+O | exists (case fix + shortcut proposed) |
| File | Save screenshot… | toolbar `save` action, mirrored | — | proposed |
| File | Export animation… | scrubber `movie` export flow | — | proposed |
| File | Close window | close this viewer window | Q | proposed (shortcut exists, unlisted) |
| View | *(dock toggles, one per dock)* | show/hide dock | — | exists |
| View | Reset layout | restore default dock arrangement | — | exists (case fix) |
| View | Focus mode | hide chrome, viewport (+scrubber) only | Ctrl+H | proposed (shortcut exists, menu item missing) |
| View | Camera ▸ Top/Bottom/Front/Back/Left/Right/Isometric | snap preset (mirrors toolbar) | — | proposed |
| View | Camera ▸ Fit view | fit | — | proposed |
| View | Camera ▸ Orthographic | projection toggle (checkable) | — | proposed |
| View | Orbit axis ▸ Z/X/Y | turntable spin axis (radio) | — | exists |
| View | Theme ▸ *(ten built-ins + user themes)* | switch theme (radio) | — | proposed |
| View | Theme ▸ Theme editor… | open theme editor dialog | — | proposed |
| Help | Shortcuts | shortcut list dialog | F1 | exists |

### Mesh viewer

| Menu | Item | Action | Shortcut | Status |
|---|---|---|---|---|
| File | Save image… | screenshot to file (mirrors toolbar) | — | proposed |
| File | Preferences… | global preferences dialog (today a button inside the Display tab) | — | proposed |
| View | Outline toggle, tab-dock toggle | show/hide | — | exists |
| View | Reset layout | restore defaults | — | exists (case fix) |
| View | Camera ▸ *(as results)* | snap presets | — | proposed |
| View | Theme ▸ *(as results)* | switch theme | — | proposed |
| Help | Shortcuts | shortcut list dialog | F1 | exists |

### Model viewer

| Menu | Item | Action | Shortcut | Status |
|---|---|---|---|---|
| File | Import STEP… | additive CAD import | — | exists |
| File | Import DXF… | additive CAD import | — | exists |
| File | Export STEP… | write current model | — | exists |
| File | Preferences… | global preferences dialog | — | proposed |
| View | *(dock toggles: Outline, Selection, View, Filter, Markers, Section, Measure, Display, Loads, Masses, Boolean, Transform)* | show/hide | — | exists ("Session" dock retitles to "Display" per INV-1, objectName `dock_model_session` unchanged) |
| View | Reset layout | restore defaults | — | exists (case fix) |
| View | Model info… | open model-info panel window | — | proposed move (exists today under a one-item "Info" menu, which retires) |
| View | Camera ▸ / Theme ▸ *(as results)* | — | — | proposed |
| Help | Shortcuts | shortcut list dialog | F1 | exists |

Non-goals: no menu-bar reordering machinery (the existing
insert-File-leftmost pattern is fine); no Edit menu; no per-dock
"window" menu; camera presets stay ALSO on the toolbar (menus are the
discoverable path, the toolbar the fast one).
