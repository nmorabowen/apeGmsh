# ADR 0080 — Interactive section builder (`SectionDocument` + Qt builder GUI)

**Status:** Proposed (2026-07-19) — **design RATIFIED by the user
2026-07-19** after the drafting-aids addition (#838): the runway
B1–B7 is authorized; this ADR flips to Accepted at close-out with the
per-slice PR numbers, per house convention. Ratified scope decisions:
authoring = parametric palette **plus** a straight-segment
freehand polygon tool in v1, with AutoCAD-style drafting aids (grid +
object snap, ortho, dynamic length/angle input — see the canvas
section); RC =
**both lanes** (classic fiber
patch/layer templates AND a `bars=` overlay on analyzer sections);
persistence = **declarative JSON section document + one-click Python
script export**; v1 extras = apeSteel catalog picker, live properties
panel, moment–curvature preview, bridge-handoff snippet.

## Context

ADR 0078 closed the *analysis* gap: any meshed 2-D face gets
geometric/warping/plastic/stress results and a declarative OpenSees
handoff (`ComputedSection`, elastic + Gauss-fiber lowerings), with a
read-only Qt inspector (`sec.viewer()`). What remains is the
*authoring* gap — building the section in the first place:

1. **Continuum sections** are authored in code: eight `*_face`
   builders, raw OCC loops, booleans, `fragment_pair`, PG-per-material
   — powerful, but a composite SRC section is ~15 lines of API calls
   whose partition/conformity laws (cut-then-fragment, shared-lines)
   are exactly the traps the skill's gotchas list documents. A GUI
   that *owns* those laws removes the whole trap class.
2. **RC sections have no path at all.** The classic OpenSees idiom for
   reinforced concrete — concrete patches + steel bar layers — exists
   as typed primitives (`Fiber`, `RectPatch`, `StraightLayer`,
   `FiberPoint`) and exactly one builder (`W_fiber`). There are no RC
   builders (no cover, no bars-per-face, no confined-core split), and
   hand-laying `RectPatch` coordinates is the most error-prone
   authoring surface in the bridge.
3. **Sections are not durable artifacts.** A section built today lives
   only in the script that built it. There is no document a user can
   save, reopen, tweak a bar diameter in, and re-analyze — the
   round-trip that section design actually is.

Precedents that bound the design:

- **The S6 inspector contract** (ADR 0078): a section GUI is a
  standalone Qt + matplotlib panel, deliberately **outside** the ADR
  0014/0042/0056 viewer family (no SceneLayer IR, no H5 consumption
  contract, no dispatcher obligations). Offscreen-guard + `blocking=`
  notebook contract + every capability reachable headless.
- **Declaration/resolution split** (ADR 0078): the artifact the GUI
  produces must be a *declaration* the bridge resolves late — never
  hand-copied numbers.
- **"Authored, never a knob"**: the GUI makes authoring easier; it
  does not invent modeling knobs the headless API lacks.
- The A1 provenance sidecar (`/opensees/computed_sections`) gives
  built sections a place to record where their numbers came from.

## Decision

Add a **declarative section document** — `SectionDocument`, a
versioned JSON-backed model that fully describes a section in either
lane — and a **standalone Qt builder GUI** that is *an editor for that
document*. The document is the source of truth and the public
headless API; the GUI is one client of it. One-click export generates
the equivalent plain apeGmsh Python script.

```python
from apeGmsh.sections import SectionDocument, launch_builder

# GUI (notebooks must pass blocking=False, S6 contract):
launch_builder()                       # blank document
launch_builder("src600.section.json")  # edit an existing one

# Headless — the same surface the GUI drives:
doc = SectionDocument.open("src600.section.json")
sec = doc.build()          # continuum lane -> SectionProperties
fib = doc.build()          # fiber lane    -> resolved Fiber recipe
doc.export_script("src600_build.py")   # readable apeGmsh code
```

### The document model (`sections/_document.py`)

A `SectionDocument` is JSON on disk (suggested extension
`.section.json`), versioned with its own semver constant
(`SECTION_DOC_VERSION = "1.0.0"`) under the additive-minor law of ADR
0023 — stated with the **corrected** window semantics (#836): the
current loader opens the previous minor's documents; a loader older
than the document refuses it loudly.

Top-level shape:

- `kind: "continuum" | "fiber"` — the lane. One document, one lane
  (a mixed SRC-with-rebar section is a *continuum* document with a
  `bars` overlay, see below).
- `name`, free-text `notes`, and a display-only `units` string
  (apeGmsh stays unit-agnostic; the field is a label, never a
  converter).
- `materials`: a named table. Each entry may carry **both roles**:
  the continuum params (`E, nu, G, fy, density` →
  `SectionMaterial`) and an optional uniaxial spec
  (`{"type": "Concrete01", "params": [...]}` → resolved to a bridge
  `UniaxialMaterial` at handoff). One name, usable from either lane.
- **Continuum lane**: ordered `shapes` list — parametric shapes
  (`{"shape": "W_face", "bf": ..., "translate": ..., "rotate": ...}`
  mirroring the eight builders 1:1), freehand
  `{"shape": "polygon", "points": [[x, y], ...]}` (straight segments
  only in v1; arcs/splines stay the DXF route), and boolean steps
  (`cut` with `remove_tool`, `fragment_pair`) referencing shapes by
  their document ids. Region → material assignments, mesh prefs
  (`lc`, `order`, default 2), `disconnected` policy. `build()` runs a
  private apeGmsh session headlessly (builders → booleans → mesh →
  `get_fem_data`) and returns a `SectionProperties` — **the document
  owns the cut-then-fragment and shared-lines laws** so the user
  cannot author the double-cover or duplicate-edge traps from the
  GUI.
- **Fiber lane**: `patches` (rect/quad/circ), `layers`
  (straight; circular reserved), `points`, `GJ`, each referencing a
  material name. Plus **RC templates** — parametric generators stored
  *as parameters* and expanded deterministically at build:
  - `rc_rect_column(b, h, cover, bars_per_face | corner+side layout,
    bar_area, core_split=False)`
  - `rc_circ_column(d, cover, n_bars, bar_area, core_split=False)`
  - `rc_beam(b, h, cover, top_bars, bottom_bars, bar_area, ...)`
  `core_split=True` splits concrete into confined-core + cover
  patches (two concrete material slots — the Mander-style split the
  user assigns materials to; the document never computes confinement
  parameters, per "authored, never a knob"). Templates re-expand on
  every build, so editing `cover` in the GUI just works.
  `build()` returns a resolved fiber recipe; the bridge handoff
  constructs the `Fiber` primitive with bridge-registered materials.
- **`bars` overlay (continuum documents)**: discrete bar points /
  bar lines (n bars between two anchor points) with a material name
  and area, riding on top of the meshed face. Lowered through
  `ComputedSection(kind="fiber")` by **appending bar `FiberPoint`s to
  the Gauss fibers** — this extends Amendment A2 with a `bars=` axis
  (see gate G-E below). Concrete area is **not deducted** at bar
  locations (standard fiber-section practice; the ~ρ·(1−Ec/Es) error
  is documented, not knobbed).

The headless `SectionDocument` API is the parity guarantee: every GUI
action is a document mutation; anything the GUI can do, a script can
do via the document (or via the underlying builders directly — the
document is convenience + persistence, never the only route).

### Script export

`doc.export_script(path)` writes a **readable** apeGmsh script — the
same `g.sections.*_face` / boolean / `SectionProperties` calls a user
would hand-write for the continuum lane; `Fiber(...)` construction
with patch/layer/point literals (templates expanded, provenance
comment naming the template + params) for the fiber lane. Golden-file
tests keep exports byte-stable. Export is one-way by design;
round-trip editing is the JSON document's job.

### The builder GUI (`sections/_builder_gui.py`)

Standalone Qt + matplotlib, inspector-mold:

- **Left — canvas**: live section view (`plot_faces`-style outlines
  pre-mesh; PG-colored mesh post-build; fiber lane draws patches,
  layers, and bar points). The freehand polygon tool: click vertices,
  close the loop, drag-edit vertices; produces a `polygon` shape in
  the document.
- **Drafting aids (scope added at user request, 2026-07-19)** — the
  AutoCAD input habits, as **GUI-layer aids only**: they decide which
  coordinates get *written into* the document and add zero document
  state, so the parity law is untouched (a script writes the same
  polygon by passing the coordinates directly).
  - **Grid snap** — configurable spacing/origin, toggleable.
  - **Object snap (osnap)** — hover-snap to existing document
    geometry: endpoint/vertex, midpoint, center (circles/pipes),
    quadrant, and segment–segment intersection, with the conventional
    marker glyphs (square/triangle/circle/×). Candidates come from
    the document's resolved shape outlines; at section-builder scale
    (tens of segments) a brute-force candidate scan is fine — no
    spatial index.
  - **Ortho mode** — constrain the rubber-band segment to 0°/90°
    (toggle + Shift-hold override), AutoCAD F8 muscle memory.
  - **Typed exact input (dynamic input)** — while a segment
    rubber-bands, a floating two-field readout tracks the cursor:
    **length** and **angle**, live-updating as you move. Start typing
    to override the length; **Tab** jumps to the angle field (and
    back); **Enter** commits the vertex, **Esc** returns to mouse
    placement. Locking one field constrains the rubber band to it
    (length-locked → the vertex slides on a circle; angle-locked →
    on a ray) while the mouse still drives the free field. The same
    entry accepts `length<angle`, `dx,dy`, and absolute `x,y` forms
    in one box for the command-line habit. Ortho and snap compose:
    a locked angle wins over ortho; snap applies to the free
    component only.
  - Status-bar toggles `GRID` / `SNAP` / `ORTHO` with F7/F9/F8
    shortcuts (registered with `Qt.ApplicationShortcut` context — the
    established Qt-canvas shortcut law). Polar tracking (arbitrary
    angle increments) is deferred; ortho + typed angles cover the
    need at this scale.
- **Right — palette + forms**: shape palette (the eight parametric
  faces + polygon + the RC templates + patches/layers/points per
  lane), per-item dimension/placement forms, materials table editor,
  boolean actions (cut / fragment), mesh prefs, `disconnected`
  policy.
- **Catalog picker** (v1 extra): when apeSteel imports, a dropdown of
  AISC/EN shapes prefills the matching `*_face` (or `W_fiber`-style)
  dimension form. apeSteel absent → the picker is hidden
  (fail-soft; never a hard dependency).
- **Live properties panel** (v1 extra): the S6 inspector's tabbed
  tables + stress preview embedded, refreshed after each build.
  **No solve on the UI thread** (S6 law): edits mark the document
  dirty; build+analyze runs in a worker `QThread` with the panel
  grayed until fresh. Continuum solves are memoized per document
  state; fiber-lane "properties" are the fiber-sum identities (cheap,
  exact) plus M–κ on demand.
- **Moment–curvature preview** (v1 extra): fiber-lane one-click M–κ
  both axes — a productized `sections/_mc.py::moment_curvature(...)`
  headless API wrapping the in-process openseespy
  `zeroLengthSection` harness proven by gate G-D. openseespy absent →
  button disabled with guidance (fail-soft).
- **Handoff button** (v1 extra): copies the bridge snippet —
  `ComputedSection(analysis=..., ...)` for continuum,
  `ops.section.Fiber(...)` + `ops.uniaxialMaterial.*` construction
  for fiber — ready to paste into a frame model script.
- **Undo/redo** = document snapshots (JSON states; cheap and exact).
- Contract mirrors the inspector: `blocking=` (notebooks must pass
  `False`), Qt absent → `ImportError` with guidance,
  `QT_QPA_PLATFORM=offscreen` on Windows → `RuntimeError` in the
  launcher while the window class stays offscreen-constructible for
  widget tests.

### Placement and naming

- Module: `sections/_document.py`, `_rc_templates.py`, `_mc.py`,
  `_script_export.py`, `_builder_gui.py` — siblings of the analyzer
  and inspector in `src/apeGmsh/sections/`.
- Exports: `SectionDocument`, `launch_builder`, `moment_curvature`
  from `apeGmsh.sections`. Nothing session-bound — the builder owns
  its private sessions, like the analyzer owns its snapshot.
- The bridge is untouched except the A2 `bars=` extension
  (`opensees/section/computed.py`), which is the only
  ruff+mypy-gated code this ADR adds.

### Gate G-E — the `bars=` overlay (blocking, before B3 ships)

Bar coordinates are sign-bearing values crossing the authoring→local
mapping, the same class gate G-D covered. Required numeric gate:
fiber-sum identities including bars (`ΣA`, signed `ΣEAyz` with an
asymmetric bar layout — a mirror flips a corner-bar pattern), and the
M–κ keystone on an RC-style section: initial slope vs the transformed
`EIxx_c` (bars as `n·A_s` steel), `ElasticPP` bar plateau
contribution vs hand-computed `A_s·fy·d` couples both signs.

### RC template oracles (deterministic, primary gate)

Template expansion is closed-form testable: bar count/positions
(cover + spacing arithmetic), `ΣA_bars = n·A_bar`, patch areas sum to
`b·h` (core+cover split exact), template re-expansion is
deterministic and parameter-editable. End-to-end: M–κ of a textbook
RC rectangle vs a hand-checked yield/ultimate estimate (coarse
tolerance — a modeling check, not an exactness oracle).

## Alternatives considered

1. **Extend the S6 inspector into an editor.** Rejected: the
   inspector is a read-only view of one analyzer instance; the
   builder edits *documents* with a different lifecycle (no analyzer
   until first build, two lanes, undo). The builder *embeds* the
   inspector panels instead.
2. **Web-based builder (`show_web` family).** Rejected for v1: pulls
   in the 0014/0042 viewer-family contracts the section tools
   deliberately avoid; matplotlib-in-Qt already proven by S6.
3. **Script-only persistence (no JSON doc).** Rejected by
   ratification: round-trip editing would constrain scripts to a
   generated dialect — worse than a declared format. Script export
   keeps the readable-code benefit one-way.
4. **Full freehand CAD (arcs, splines, trim).** Deferred: DXF import
   already covers irregular outlines; v1 freehand is straight-segment
   polygons only.
5. **Bars as meshed regions in the continuum lane.** Rejected: tiny
   circles are mesh-hostile and the analyzer's J/warping for RC is
   rarely wanted; discrete bar points through the fiber lowering is
   the correct mechanics at trivial cost.

## Consequences

**Positive:** RC finally has a first-class path; the composite
authoring traps (double-cover, duplicate-edge) become impossible from
the GUI because the document owns those laws; sections become durable,
editable artifacts with a provenance trail; the M–κ harness and RC
templates are useful headless APIs independent of the GUI.

**Negative / risks:** a new persistent format to version (bounded:
own semver + additive-minor law + golden tests); a second Qt surface
(bounded: inspector-mold, matplotlib-only, embeds rather than forks
the inspector panels); two optional soft dependencies at the GUI edge
(apeSteel, openseespy) — both fail-soft, never required headless;
the `bars=` A2 extension re-opens sign-bearing lowering territory —
gate G-E is blocking for exactly that reason.

## Slices

| # | Deliverable | Verify |
|---|---|---|
| B1 | `SectionDocument` continuum lane: schema + loader/saver + headless `build()` → `SectionProperties` (shapes, polygon, booleans, materials, mesh prefs, policy) | JSON round-trip; SRC document builds and matches the hand-authored ADR 0078 example's numbers; trap-law tests (document cannot express double-cover) |
| B2 | Fiber lane + RC templates (`rc_rect_column` / `rc_circ_column` / `rc_beam`, `core_split`) + material specs → bridge resolution | expansion oracles (bar positions/areas/patch sums, exact); deterministic re-expansion; deck golden via `Fiber` |
| B3 | `bars=` overlay on `ComputedSection(kind="fiber")` (A2 amendment) | **gate G-E** (signed identities + M–κ keystone with bars, both signs) |
| B4 | Script export, both lanes | golden-file byte-stable exports; exported script re-produces the document's numbers |
| B5 | GUI shell: palette, forms, canvas, polygon tool + drafting aids (grid/object snap, ortho, typed exact input), undo/redo, doc open/save | offscreen widget tests (S6 pattern); QTimer screenshot smoke; every GUI mutation asserted equal to the document API call; snap-resolution unit tests (pure functions: candidates → snapped point, ortho projection, `length<angle` parsing) — no Qt needed |
| B6 | Live properties panel (inspector embed) + worker-thread build | no-solve-on-UI-thread assertion; panel refresh == headless `build()` results |
| B7 | Extras: catalog picker (apeSteel fail-soft), `moment_curvature` + M–κ preview (openseespy fail-soft), handoff snippet | catalog prefill vs apeSteel values; M–κ slope vs analyzer/fiber-sum `EI`; snippet round-trip compiles |
| — | Close-out: how-to + guide + skill reference §, CHANGELOG, flip to Accepted | docs build; skill mirror sync; completeness critic pass |

## Reference

- ADR 0078 (+ Amendments A1/A2) — analyzer, lowerings, inspector
  contract, provenance sidecar; gate G-D is the model for G-E.
- ADR 0014 / 0042 / 0056 — the viewer-family contracts this GUI
  deliberately stays outside of.
- ADR 0023 (+ #836 correction) — the additive-minor versioning law
  the document schema borrows, with the corrected window direction.
- `opensees/section/fiber.py` (`Fiber`, `RectPatch`, `StraightLayer`,
  `FiberPoint`, `W_fiber`), `sections/_builder.py` (`*_face`
  builders, `plot_faces`), `sections/_inspector.py` (S6 panel).
- apeSteel (optional): AISC v16 / EN 10365 catalogs for the picker.
