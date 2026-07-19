# Plan — Interactive section builder (ADR 0080)

Implements **ADR 0080**
(`src/apeGmsh/opensees/architecture/decisions/0080-section-builder.md`).

**Goal.** A declarative `SectionDocument` (JSON, versioned) covering
both section lanes — continuum (analyzer) and fiber (patch/layer/RC
templates) — plus a standalone Qt builder GUI that edits it, script
export, the `bars=` A2 extension, and the v1 extras (catalog picker,
live properties panel, M–κ preview, handoff snippet).

**User-ratified scope (2026-07-19, recorded in the ADR status line):**
palette + straight-segment freehand polygons; BOTH RC lanes; JSON doc
+ script export; all four extras in v1.

**Verified facts this plan rests on** (probed 2026-07-19; re-verify at
slice start — verify-against-live-source law):

- Flat-face builders: `sections/_builder.py` — `W_face :780`,
  `rect_face :817`, `rect_hollow_face :835`, `pipe_face :858`,
  `pipe_hollow_face :877`, `angle_face :901`, `channel_face :923`,
  `tee_face :950`, `plot_faces :973`. All centred-on-origin,
  `translate/rotate`, auto-PG by label.
- Fiber primitives: `opensees/section/fiber.py` — `RectPatch :71`,
  `StraightLayer :108`, `FiberPoint :148`, `Fiber :181` (GJ `-GJ`
  flag), `W_fiber :278` (the builder pattern RC templates follow).
- `ComputedSection` (`opensees/section/computed.py`): `kind=` axis,
  `fibers={pg: UniaxialMaterial}` exact-cover, `dependencies()`
  surfaces materials (P11), `lower_to_fiber` in
  `sections/_lowering.py` (Gauss fibers about the elastic centroid).
- Inspector contract: `sections/_inspector.py` — offscreen guard in
  `launch_inspector`, window class offscreen-constructible, no solve
  on the UI thread, `blocking=` notebook law.
- M–κ harness precedent: `tests/sections/test_fiber_lowering.py`
  `_mc_harness` (zeroLengthSection, DOFs fixed except axial+rotations,
  `getLoadFactor` = |M| under unit reference moment).
- Provenance sidecar (A1): `_internal/_computed_sections_h5.py`; a
  future `"doc"` payload key is a natural extension — NOT in this
  ADR's scope (would be an A1 amendment).
- apeSteel: separate project; import fail-soft only. openseespy: in
  opensees_venv; CI suite lane has it; always `importorskip`/disable.

**Execution hazards (standing memory — apply to every slice):**

- Python = `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe`;
  worktree `src` first on `PYTHONPATH` (`<wt>/src;<wt>`, Windows `;`)
  + `APEGMSH_QUIET=1`; judge by targeted testpaths only
  (`tests/sections`, `tests/opensees/unit`, new `tests/sections/`
  files) — never whole-tree on Windows.
- Every PR `--base main`, merge in order, verify `headRefOid` == local
  tip before merge; CHANGELOG section directly below the anchor, never
  edit existing lines.
- `opensees/section/computed.py` (B3) is ruff-hard + mypy-0 gated.
- Qt tests: offscreen widget tests run UNMARKED in-process (S6
  precedent — file pins `QT_QPA_PLATFORM=offscreen`); the launcher
  guard raises on win32-offscreen, the window CLASS stays
  constructible for `grab()` smoke tests.
- Skill edits → `scripts/sync_skill.py` (CI-gated mirror) + after
  merge `python scripts/refresh_user_skill.py`.
- Document versioning cites ADR 0023 **with the corrected window
  direction (#836)**: current loader opens previous-minor docs; older
  loader refuses newer docs loudly.

## Model & review policy

Deterministic oracles are the primary gate (ratified gate policy).
One blocking adversarial-style gate: **G-E** on the B3 `bars=`
lowering (sign-bearing coordinates — same class as G-D, ~3 focused
checks, no large panel). A single completeness critic before flipping
the ADR to Accepted. Everything else ships on its slice tests.

Model notes: B1–B4 (document model, templates, lowering, export) are
correctness-critical — implement carefully with exact oracles; B5–B7
(Qt shell, embeds, extras) are scaffolding over proven patterns.

## Slices (one PR each, `--base main`)

### B1 — `SectionDocument` continuum lane

`sections/_document.py`: schema constant `SECTION_DOC_VERSION =
"1.0.0"`, loader/saver (versioned, additive-minor law), shapes
(8 parametric + `polygon`), boolean steps (`cut(remove_tool)`,
`fragment_pair`), materials table (dual-role entries), region→material
map, mesh prefs (`lc`, `order=2`), `disconnected`. `build()` drives a
private apeGmsh session headlessly and returns `SectionProperties`.
The document API *enforces* the partition laws (assigning a material
to an overlapping un-cut region is unrepresentable or fail-loud).

**Verify:** JSON round-trip identity; the ADR 0078 SRC example as a
document reproduces the hand-authored analyzer numbers (`EA`,
`EIxx_c`, `GJ` rel 1e-9 on the same mesh prefs); polygon shape builds
(triangle + L-shape vs hand integrals); fail-loud tests for unknown
material refs / bad version / trap-law attempts.

### B2 — Fiber lane + RC templates

`sections/_rc_templates.py` + fiber-lane document entries. Templates:
`rc_rect_column`, `rc_circ_column`, `rc_beam` (cover, bar layout,
`bar_area`, `core_split`). Material specs (`{"type", "params"}`)
resolve at handoff via the `ops.uniaxialMaterial` namespace by name.

**Verify:** expansion oracles — bar count/coordinates closed-form
(cover+spacing arithmetic, corner bars once), `ΣA_bars = n·A_bar`
exact, patch areas sum to `b·h` (core+cover exact split), circular
layout angles; re-expansion determinism (edit `cover` → only bar
coords move); resolved `Fiber` deck golden.

### B3 — `bars=` overlay (A2 amendment) — **gate G-E**

`ComputedSection(kind="fiber", bars=...)`: bar points/lines from a
continuum document appended to the Gauss fibers (no concrete
deduction — documented). ADR 0078 Amendment A2 gains the `bars=`
paragraph; ruff+mypy gates apply.

**Verify (gate G-E, blocking):** fiber-sum identities incl. bars
(`ΣA`, signed `ΣEAyz` with an asymmetric corner-bar layout — mirror
catch); M–κ keystone: initial slope vs transformed-section `EIxx_c`
(n·A_s steel), `ElasticPP` bar plateau vs `ΣA_s·fy·d` couples both
signs; exact-cover/arg-family validation extended.

### B4 — Script export

`sections/_script_export.py`: continuum → readable
builders/booleans/analyzer script; fiber → `Fiber(...)` literals with
template-provenance comments.

**Verify:** golden files byte-stable; executing an exported continuum
script reproduces the document's analyzer numbers; exported fiber
script's deck == document handoff deck.

### B5 — GUI shell

`sections/_builder_gui.py` + `launch_builder(path_or_doc=None, *,
blocking=True)`: palette, per-shape forms, canvas
(outlines/mesh/fiber view), polygon tool (snap grid, vertex drag),
undo/redo (document snapshots), open/save.

**Verify:** offscreen widget tests — every GUI mutation asserted
identical to the corresponding `SectionDocument` API mutation (the
parity law as a test); QTimer screenshot smoke; launcher guard tests.

### B6 — Live properties panel

Embed the S6 inspector panels; build+analyze in a worker `QThread`,
dirty-marking, gray-until-fresh.

**Verify:** no-solve-on-UI-thread assertion (thread id check in
test); panel values == headless `build()` results; rapid-edit
coalescing (N edits → ≤ N builds, last state wins).

### B7 — Extras

Catalog picker (apeSteel fail-soft import; AISC/EN → form prefill),
`sections/_mc.py::moment_curvature(...)` (headless first) + GUI
button (openseespy fail-soft), handoff-snippet generator.

**Verify:** prefill values vs apeSteel catalog entries; M–κ slope vs
fiber-sum `EI` (elastic materials) both axes; snippet exec-compiles
and its deck matches the document handoff; absence paths (no
apeSteel / no openseespy) degrade with guidance, never crash.

### Close-out

How-to page + guide section + skill reference §; completeness critic;
flip ADR 0080 → Accepted with PR numbers; consider the A1 payload
`"doc"` key as a recorded follow-up (not implemented here).

## Risk register

| Risk | Slice | Mitigation |
|---|---|---|
| Document schema churn after GUI learnings | B1 | additive-minor law + golden round-trip tests; GUI lands 4 slices later, schema soak time |
| RC template geometry off-by-cover errors | B2 | closed-form coordinate oracles per face; corner-bar dedup test |
| `bars=` mirror / sign regression | B3 | gate G-E blocking (G-D pattern) |
| Export scripts drift from live API | B4 | exported-script *execution* tests, not just golden text |
| Qt worker-thread races (build vs edit) | B6 | document snapshots are immutable inputs to the worker; result tagged with snapshot id, stale results dropped |
| Optional-dep surface (apeSteel/openseespy) | B7 | fail-soft import guards + absence-path tests |
| GUI/headless drift | B5+ | the parity law is itself a test (GUI mutation == document API call) |
