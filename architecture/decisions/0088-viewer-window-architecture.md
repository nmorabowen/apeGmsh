# ADR 0088 — Viewer window architecture: dock partition, inspector, on-demand panels

**Status:** Accepted (2026-08-04) — the Phase-2 blueprint for the Qt
viewer window layouts. Companion to ADR 0087 (visual design system:
tokens, invariants, guards) — 0087 says how a panel *looks*, this ADR
says which panels *exist*, where they sit, and when they appear.
Builds on the dock machinery of `viewers/ui/_dock_registry.py`
(`DockSpec`, sanitize, View-menu toggles), `_layout_metrics.py`
(`LAYOUT`), and the persistence pattern of `_results_window.py` /
`viewer_window.py`.

## Context

The results viewer today mounts **eleven** docks: seven built-ins
(`dock_results_outline`, `dock_results_right` [Plots],
`dock_results_diagram`, `dock_results_geometry`,
`dock_results_details`, `dock_results_session` [Display],
`dock_results_scrubber`) plus four extensions (`dock_output`,
`dock_color_map_editor`, `dock_results_definitions`,
`dock_results_clip_planes`). Six of them boot visible, stacked into
one right column.

### Evidence — before/after screenshots, 2026-08-04

(`gui_audit/` and `gui_after/` capture sets; the layout findings
survived Phase 1 because Phase 1 was deliberately de-noise-only.)

- **`gui_after/40_full_window_final.png`** — at boot with one contour
  diagram: the right column stacks Plots ("No plots yet"), Diagram,
  Color Mapping ("No diagram selected." above a *live-looking* preset
  combo + Min/Max `0.000000` fields + "Fit to data" button —
  an INV-2 violation as panel furniture), and Definitions ("No named
  definitions."). Four panels, three of them empty, consuming ~35% of
  the window's width.
- Same capture — a **rotated tab spine** (Diagram / Geometry /
  Details / Display / Section planes) between viewport and right
  column: five sideways tabs whose panels are contexts of "what am I
  configuring", not five independent tools. Navigating them is a
  second, parallel navigation system beside the Outline.
- Same capture — the Diagram panel carries an inner bold "Diagram"
  header (INV-1 violation) **plus a hint line clipped mid-glyph**
  above the New-layer card: a real layout defect, not a style nit.
- Same capture — the summoned Output dock floats as a small box
  *beside* the time scrubber, stealing the scrubber's right end
  instead of behaving like a console.
- **`gui_after/00_full_window.png`** — Section planes summoned: the
  whole 40%-wide right column serves three checkboxes and an Add
  plane row.
- The empty-state HUD (R1.4) and blackout toasts (R1.3) live in the
  viewport and already behave; the docks around them do not.

The audit's standing question (Grok G1.1): should the tab spine
consolidate into one contextual inspector driven by outline
selection?

## Decision

### D1 — Results viewer: the four-region window

```
┌──────────────────────────────────────────────────────────────┐
│ File / View / Help                                           │
├───┬──────────┬──────────────────────────────┬────────────────┤
│ T │ Outline  │                              │ Plots (hidden  │
│ o │          │                              │  until 1st     │
│ o │          │        3D viewport           │  plot)         │
│ l │          │   (HUDs: probes, pick        ├────────────────┤
│ b │          │    readout, empty-state,     │ Inspector      │
│ a │          │    toasts — viewport-owned)  │ (∘ Section     │
│ r │          │                              │  planes tab)   │
├───┴──────────┴──────────────────────────────┴────────────────┤
│ Time scrubber                                                │
│ (Output console — hidden; full-width UNDER the scrubber)     │
└──────────────────────────────────────────────────────────────┘
```

Visible at boot (fresh profile): **Outline, Inspector, Time
scrubber** — three docks, not six. Everything else is on-demand (D3).

Proportions and floors (all from `LAYOUT`; new fields follow its
documented promotion path):

| Region | Initial | Min | Source |
|---|---|---|---|
| Outline (left) | 260 px | 180 px | `outline_initial_width` / `outline_min_width` |
| Inspector (right) | 380 px | 240 px | `right_initial_width` / `right_min_width` |
| Plots (right-top, when shown) | 380 × ~40% of column | 240 px | same |
| Time scrubber (bottom) | 84 px | 60 px | `scrubber_initial_height` / `scrubber_min_height` |
| Output (bottom, when shown) | 140 px | 100 px | new `LAYOUT.output_initial_height` / `output_min_height` |
| Viewport | remainder | — | — |

Invariant: at a 1280 px-wide window the viewport keeps **≥ 50%** of
the width with Outline + Inspector at initial sizes; the two side
columns together never exceed 45%. (1280 − 260 − 380 = 640 = 50% —
the initial widths already satisfy this; the acceptance test pins
it.)

### D2 — The inspector: consolidate the tab spine (G1.1: yes)

The five-tab spine (Diagram / Geometry / Details / Display / Section
planes) is replaced by **one Inspector dock**
(`dock_results_inspector`) whose content is a stacked context driven
by Outline selection, plus two survivors that are *not* selection
contexts:

| Today | Becomes |
|---|---|
| Diagram tab | Inspector context: *diagram row selected* — layer stack + New-layer card + a **Color** section (see below) |
| Geometry tab | Inspector context: *geometry row selected* — deformation / threshold / scope / display sections (INV-3 sections) |
| Details tab | Inspector context: *pick or probe result* — the readout table |
| Color Mapping dock | **Dissolved** into the diagram context's Color section (preset, min/max, log scale, scalar bar). The standalone `dock_color_map_editor` dock is removed; its toolbar `colormap` action selects the active diagram and raises the Inspector |
| Display tab | **Stays a separate dock** (`dock_results_session`, title "Display") — app-level preferences are not properties of a selection. Hidden by default, View menu summons |
| Section planes | **Stays a separate dock** — it is a *list* of view annotations with its own toolbar affordance, not a property sheet. Default hidden, tabified behind Inspector, summoned by the `section` toolbar glyph (existing wiring) |

Contexts, exhaustively: *nothing selected* → one hint line
("Select an item in the outline to edit it") + the same starter
actions the empty-state HUD offers; *stage/step row* → stage readout
(activation table); *geometry row*; *diagram row*; *plot row* →
plot info + "Show in Plots" ; *pick/probe* → details readout. Every
context obeys INV-1 (no inner title), INV-2 (controls only when they
act on something), INV-3 (sectioned forms).

**Rationale.** The Outline is already the navigation spine — Stages /
Geometries / Diagrams / Plots live there. A second navigation system
(five rotated tabs) duplicates it and guarantees empty panels at
boot. Consolidation makes selection the single "what am I editing"
signal, matches the tree-plus-property-editor model every CAE
operator already knows, and kills three of the four
mostly-empty-dock findings in one move. Display and Section planes
stay out because forcing *non-selection* content through a selection
switch is how inspectors rot into junk drawers.

**Cost accepted.** Two contexts cannot be visible at once (e.g.
geometry threshold + diagram color simultaneously). Mitigation: the
Inspector is floatable like any dock, and the workflows below show
the sequential pattern dominates. A pin/split-inspector affordance is
explicitly deferred — not in Phase 2.

### D3 — On-demand panels and their summon affordances

| Dock | Default | Summon path |
|---|---|---|
| Plots (`dock_results_right`) | hidden | **auto-shows** on first plot creation (probe→plot, shift-click time history); View menu toggle; closing it never discards plot data |
| Output (`dock_output`) | hidden | status-bar badge (exists — error/warn counts), View menu. Opens **full-width in the bottom area, stacked below the Time scrubber** — never split beside it (the 40_full_window defect). Auto-raise on first error stays as shipped |
| Definitions (`dock_results_definitions`) | hidden | View menu only (inspection tool for named primitives) |
| Display (`dock_results_session`) | hidden | View menu. Theme also reachable via View → Theme (ADR 0087 App. B), so hiding this dock costs no capability |
| Section planes (`dock_results_clip_planes`) | hidden | toolbar `section` glyph (exists, incl. the show-AND-raise guard), View menu |
| Inspector, Outline, Time scrubber | visible | — (scrubber remains non-closable) |

The empty-state HUD remains the diagram-less boot's call to action;
its buttons create a diagram, which selects the new row, which gives
the Inspector its first real context — the HUD and the Inspector
empty state advertise the *same* actions, one in the viewport, one in
the dock.

### D4 — Named defects this blueprint resolves

1. Diagram panel's inner bold "Diagram" header — removed (INV-1); the
   dock title (now "Inspector") is the only title.
2. The clipped hint line above the New-layer card — replaced by one
   properly-laid-out hint row (INV-2 hint role, `fs_body` italic,
   eliding disabled; must render unclipped at both densities).
3. Color Mapping's empty state (preset combo + `0.000000` Min/Max on
   no data) — gone with the dock; the Color section renders disabled
   with a single hint until the diagram has a scalar field.
4. Output-beside-scrubber — bottom-area stacking order fixed per D3.
5. Six-docks-at-boot — reduced to three (D1).

### D5 — Mesh and model viewers

Phase 2 touches results only; mesh/model get the ADR 0087 Appendix B
menu work and INV fixes. The *target* partition, for the phase that
follows:

- **Mesh viewer** — left: Outline (visible, sanitized — as shipped).
  Right: the legacy `QTabWidget` dock (Info / Display / Filter /
  Markers / Clipping / probe tabs) migrates to `DockSpec` docks with
  the same grouping, default-hidden except Filter. Until then it
  stays as-is; no schema bump for a rename-only change.
- **Model viewer** — left: Outline + Selection stacked (as shipped).
  Right: today's ten tabs regroup into two clusters: **view**
  (View / Filter / Markers / Section / Measure) and **tools**
  (Loads / Masses / Boolean / Transform), tools default-hidden;
  "Session" retitles to "Display" keeping `dock_model_session`.
  One-item "Info" menu retires (App. B).

### D6 — State rules

- **objectNames.** Survivors keep their names exactly:
  `dock_results_outline`, `dock_results_right`,
  `dock_results_session`, `dock_results_scrubber`, `dock_output`,
  `dock_results_definitions`, `dock_results_clip_planes`. New:
  `dock_results_inspector`. Retired (never reused for different
  content): `dock_results_diagram`, `dock_results_geometry`,
  `dock_results_details`, `dock_color_map_editor`.
- **Schema versions.** `ResultsWindow._LAYOUT_SCHEMA_VERSION` bumps
  6 → 7 **once**, in the single PR that lands the new partition —
  the version history (1→…→5 "stuck outline dock", 6 current) shows
  one bump per structural change, never per sub-step. Saved v6
  layouts are discarded on first launch (existing mismatch path).
  `ViewerWindow._LAYOUT_SCHEMA_VERSION` stays 5 — the mesh/model
  dock sets do not change in Phase 2.
- **Reset layout** restores exactly the D1 boot state: visible set
  {Outline, Inspector, Time scrubber}, initial sizes, hidden set
  hidden — not whatever the user's first-launch restore produced.
- **Focus mode (Ctrl+H)** shows viewport **plus the Time scrubber**
  (reviewing an animation is the focus-mode use case; today's
  everything-hidden variant loses the playhead). Toolbar and all
  other docks hide; a second Ctrl+H restores the exact prior
  visible set (the existing snapshot mechanism). Viewport HUDs
  (probe strip, pick readout, empty-state card, blackout toast) are
  viewport-owned and unaffected — in focus mode they are the only
  chrome left, which is correct: they explain the pixels.
- **HUD / toast interaction with the layout.** HUDs anchor to the
  viewport widget, never to docks, so dock changes can never occlude
  them (the R1.3 occlusion fix stays load-bearing). The blackout
  toast and empty-state HUD keep their R1.3/R1.4 lifecycles;
  the empty-state HUD additionally hides while focus mode is active
  only if a diagram exists (an empty focus-mode viewport should
  still say why it is empty).

### D7 — Workflow validation

- **Open → contour → scrub.** Boot: three docks, big viewport,
  empty-state HUD offers "Add contour". One click creates the
  diagram, selects its row, Inspector shows layer + Color. Scrubber
  is already there. Zero dock summons; before: user faced six docks,
  four empty.
- **Probe → plot.** Probe HUD (viewport) → shift-click → Plots
  auto-shows top-right above the Inspector; the Inspector meanwhile
  shows the probe readout context. Both visible at once — the
  probe-vs-plot pairing is the reason Plots is a separate dock and
  not an Inspector context.
- **Compare stages.** Outline stage rows + `stages` toolbar toggle +
  scrubber. Inspector shows the stage readout for whichever stage is
  selected; the viewport comparison itself is diagram state, not
  dock state.
- **Threshold / scope explore.** Select geometry row once; the
  threshold and scope sections (INV-3, enable-toggle in the section
  header) live in that one context; scrubbing while the context is
  open works because selection does not change on scrub.

### D8 — Acceptance criteria (Phase 2 verifies each)

1. Fresh profile boot (no QSettings): visible docks are exactly
   {Outline, Inspector, Time scrubber}; at 1280×800 the viewport is
   ≥ 50% of window width.
2. The Inspector with nothing selected shows one hint line + starter
   actions; zero enabled data controls (INV-2).
3. Selecting geometry / diagram / stage / plot rows switches the
   Inspector context on the same Qt tick; no context shows another
   kind's controls.
4. No standalone Color Mapping dock exists; color controls appear
   only in the diagram context and are disabled (single hint) until
   the diagram carries scalar data; the `colormap` toolbar action
   raises the Inspector on the active diagram.
5. The diagram context has no inner "Diagram" title and its hint
   line renders unclipped at compact and comfortable density
   (screenshot evidence in the PR).
6. Creating the first plot auto-shows the Plots dock; closing the
   dock preserves plots; the View menu re-summons it showing them.
7. Summoned Output is full-window-width in the bottom area, below
   the Time scrubber; the badge and auto-raise-on-error behaviors
   are unchanged.
8. Ctrl+H leaves viewport + Time scrubber + HUDs; a second Ctrl+H
   restores the exact prior dock-visibility set.
9. View → Reset layout reproduces criterion 1's state from any
   arrangement, including after dock drags and summons.
10. `ResultsWindow._LAYOUT_SCHEMA_VERSION == 7`; saved v6 state is
    discarded once; surviving objectNames byte-identical (layout
    persistence tests updated to the new roster, green).
11. Retired objectNames appear nowhere in `viewers/**` except the
    schema-history comment.
12. Full `pytest tests/viewers -q` green; ADR 0087 guard budgets
    unchanged or ratcheted down (no new G-HEX/G-INLINE debt).
13. The PR attaches re-captured `00/10/20/40` screenshot series in
    the three canonical palettes (ADR 0087 INV-6 review bar).

## Alternatives considered

- **Keep the tab spine, just hide empty tabs.** Rejected: hiding
  tabs makes the spine's contents unpredictable (tabs appear and
  vanish as data changes), and the spine remains a second navigation
  system. The failure mode of the current design is structural, not
  cosmetic.
- **Full consolidation including Display and Section planes.**
  Rejected: neither is a property of an outline selection. An
  inspector that shows "whatever we had lying around" stops being
  predictable — the D2 boundary (selection contexts in, tools and
  preferences out) is the durable rule.
- **Color Mapping as its own context (not folded into diagram).**
  Rejected: a color map without its diagram is meaningless — every
  interaction ("which field? what range?") is diagram state. Folding
  it kills the audit's worst INV-2 offender for free.
- **Auto-show Output on any write (not just errors).** Rejected:
  the badge + auto-raise-on-error already covers the signal; consoles
  that self-open on chatter train users to close them permanently.
- **Reuse `dock_results_diagram` as the Inspector's objectName to
  preserve saved geometry.** Rejected: the dock's meaning changes;
  inheriting a stale six-dock-era geometry into the three-dock
  layout is exactly the half-broken restore the schema bump exists
  to prevent.
- **Per-dock schema versions instead of one window version.**
  Rejected: Qt's `saveState` blob is monolithic; partial restores
  are not a thing without custom serialization, which is not worth
  owning.

## Consequences

**Positive.** Boot shows a viewport, not a dashboard of empty
panels; the Outline becomes the single navigation spine with the
Inspector as its property editor; three INV-2 violations and the
double-header defect die by construction; on-demand panels each have
a discoverable summon path; Phase 2 has a numbered checklist instead
of taste.

**Negative / accepted.** One-context-at-a-time inspector (pin/split
deferred); every user's saved v6 layout resets once; the Inspector
becomes the results viewer's most complex widget (a stacked context
host with per-kind panels) and needs its own state tests; Plots
auto-show is one more implicit behavior to document (View menu +
this ADR are the record); mesh/model target partitions (D5) are
declared but not scheduled, and will drift if the follow-up phase
never lands — the ADR's D5 is the tripwire for that review.
