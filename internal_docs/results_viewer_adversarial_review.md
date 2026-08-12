# Results Viewer — Adversarial Review Sweep

> **Date:** 2026-07-31  
> **Trigger:** Assessment of the Qt results viewer (`internal_docs/results_viewer_assessment.md`),
> plus the team's report that *the viewer is the subsystem with the most issues —
> we fix something and break something else*.  
> **Method:** Four independent read-only adversarial agents (event/dispatch,
> shell/geometries, test gaps, assessment critique), cross-checked against source.  
> **Tone:** Hostile to comfort, fair to evidence. No features proposed until the
> disease is named.

---

## 0. Verdict (read this first)

The original assessment is a **competent inventory** of what exists and what is
missing. It is a **weak diagnosis** of why fix–break cycles dominate.

**The disease is not “missing isosurfaces.”**  
The disease is **non-local correctness under change**:

```
Intent state  (often dual-homed)
      │  three channels: Dispatcher  |  Director direct pump  |  Qt/ActiveObjects
      ▼
Pumps (closures in results_viewer.show)  ── except Exception: pass ──► empty/wrong scene
      │
      ├─ mutate VTK / layer handles / grid.points
      ├─ restack: detach+attach (actor identity churn)
      └─ batch exit skips RENDER-lane subscribers (except one session patch)
      ▼
RENDER  ◄── also called ad-hoc from results_viewer (not G-RENDER-guarded)
      │
      ▼
Pick / outline / settings / AO  ── re-enter ──► events again
```

**Multipliers**

| Multiplier | Effect |
|---|---|
| Concurrent geometries (ADR 0058) | State × N scenes, pins, offsets; shell still one god file |
| Mid-migration dual reconciler | Director STEP/STAGE still pumps+renders; Dispatcher also does |
| Silent pumps | Failures look like “toggle did nothing” |
| G-RENDER hole | `results_viewer.py` outside the AST allowlist that polices mesh/model/ui |
| Tests are strong on seams, weak on interaction graph | ~40–60% of fragility invisible to CI |
| Multi-agent PRs | Prose contracts lose; guards don’t cover the god shell |

**Default recommendation (revised)**

| Do | Don’t |
|---|---|
| **Cascade freeze** before catalog growth | Ship isosurface / scope / slice into the unstable shell |
| Kill dual STEP/RENDER paths | “Mature event machine” marketing without finishing 0056 |
| Make silent pumps loud + tested | Carve `ResultsViewer` without freezing pump I/O |
| Promote interaction tests into CI-reachable form | Trust 15 `@pytest.mark.qt` tests that never run in CI |
| Keep domain brain (diagrams, IR, Results) | Full product rebuild / ParaView rewrite for *this* pain |

---

## 1. Why “fix X breaks Y” is structural

### 1.1 Two reconcilers wired in series

ADR 0056 said: *every gesture funnels through `Dispatcher.fire`; that is the only
place STEP/DEFORM/GATE/RENDER may run.*

What actually happens on scrub / stage change:

```text
Director.set_step / set_stage
  → registry.update_to_step(...)     # pin-BLIND
  → director._render()               # direct plotter.render
  → fire observers
       → results_viewer: dispatcher.fire(STEP_CHANGED | STAGE_CHANGED)
            → pin-AWARE _pump_step
            → _pump_deform
            → _pump_gate (stage only)
            → plotter.render again
```

Evidence (`diagrams/_director.py`):

- stage path: `reattach_all` → `_fire_stage_changed` → `update_to_step` → `_render`
- step path: `update_to_step` → `_fire_step_changed` → `_render`

Evidence (`results_viewer.py`): `subscribe_step` / `subscribe_stage` → `dispatcher.fire(...)`.

**Matrix promise:** one STEP + one DEFORM + one RENDER per scrub.  
**Runtime delivery:** ~2× STEP, 1× DEFORM, 2× RENDER; intermediate pin-blind frame.

| If you fix… | You break… |
|---|---|
| Remove director `update_to_step` / `_render` | Web / animation / headless paths that relied on mutator side effects go blank |
| Keep both, “optimize” one side | Pins flicker; double cost; one path pin-aware, one not |
| Make `DiagramRegistry.update_to_step` pin-aware but leave director dual call | Still double STEP; subtle order bugs remain |

This single fact explains a large class of “I fixed scrub, broke pins” / “I fixed pins, broke export.”

### 1.2 Pin-blind vs pin-aware STEP (same step, two semantics)

```text
registry.update_to_step(global)     # Director — ignores Geometry.stage_id
_pump_step → _effective_step_for    # Dispatcher — pin-local when pinned
```

Pinned geometry can briefly paint the **active stage’s global index** before the
pump “corrects” it. Under FPS scrub that is flicker; under a partial refactor it
is permanent wrong values.

### 1.3 Batch exit is incomplete

`gesture_batch` / `session_batch` union primitives, run pumps, `render()` —
**they do not replay RENDER-lane subscribers** unless a call site patches one.

Subscribers that can be skipped under batch:

- `_sync_substrate_visibility` (contour occlusion of grey fill)
- stage-pin LAYER_STAGE resync
- geometry offset → pick KD-tree / labels

Session restore **manually** calls `_sync_substrate_visibility` after batch
(`results_viewer.py` ~2970). Composition eye `gesture_batch` does **not**.

| If you fix… | You break… |
|---|---|
| Always replay all RENDER subs on batch exit | Session / eye cascades re-enter tree rebuilds unless handlers are id-coalesced |
| Keep omit + patch only session | Eye hide composition → contour gone, fill stays hidden (ghost empty mesh) |
| Add a third ad-hoc sync for eye | Next RENDER-only subscriber forgotten again |

### 1.4 Dual homes that never fully died

| State | Homes | Risk |
|---|---|---|
| Layer visibility intent | `Diagram._visible`, outline eye, `Composition.saved_visibility` | Snapshot desync on restore / re-show |
| Effective visibility | GATE recomputed vs actor/handle flags | Stale paint if gate skipped |
| Active composition | `CompositionManager._active_id` vs `ActiveObjects._active_composition` | Gate scope ≠ dock “what’s editing” |
| Stage / step | Director vs ActiveObjects vs per-geom pin vs Results default stage | Combined-mode fires **real** stage id while UI shows “All stages” |
| Substrate display | `Geometry.show_*` vs raw `SetVisibility` in `_apply_geometry_display` | Display outside matrix |
| RENDER trigger | `Dispatcher._render` vs `Director._render_callback` | Dual paint |

ActiveObjects was intentionally a **focus projection** (ADR 0056 OQ3), not an
owner — but code still treats it as a parallel bus. Outline writes manager first,
then AO; programmatic `set_active` without AO update → docks wrong, gate correct.

### 1.5 Contour occlusion × composition eye (concrete product bug class)

1. Contour attaches → `occludes_substrate` → fill hidden (RENDER-lane).  
2. User composition-eye off → `gesture_batch` → GATE hides contour (good).  
3. RENDER-lane occlusion sync **does not run**.  
4. Fill stays hidden → “I hid the contour and the mesh disappeared.”

Fixing occlusion by coupling into GATE without a batch protocol reintroduces
N× occlusion walks on every eye; fixing only the eye path leaves ghost/duplicate
paths wrong.

### 1.6 Session restore is a cascade amplifier

`_apply_session` inside `session_batch`:

1. Add diagrams against **pre-restore** stage.  
2. Rebuild geometry tree; pins fire **live** reattach while pumps suppressed.  
3. `set_stage` / `set_step` **inside** batch still run Director dual path (raw reattach + render).  
4. Batch exit full pump.  
5. Manual substrate sync only.  
6. Broad `except: pass` on stage restore and batch exit.

**Deserialize index hazard (CRITICAL):**  
`deserialize_session` **omits** failed/retired specs (compacts list). Geometry
snapshots store `layer_indices` into the **old** flat list. Partial kind failure
→ layers land in wrong compositions or drop.

| If you fix… | You break… |
|---|---|
| Move `set_stage` outside batch | Attach order wrong (layers before stage) |
| Suppress director render during batch | Blank first paint if exit fails |
| Compact indices “more carefully” without padding holes | Still mis-assign compositions |

### 1.7 Concurrent geometries without a RenderGraph

ADR 0058 correctly made Geometry a scene instance. Residual split brain:

| Brain | Owns | Problem |
|---|---|---|
| State | GeometryManager / CompositionManager / registry | Mostly coherent |
| Shell | `results_viewer` closures: pumps, actors, pick map, occlusion | Every geometry feature re-touches god file |
| Selection | Outline eyes vs AO vs `director.compositions` (active-geom only alias) | Layer on geom B selected while settings edit geom A |

Pick map: substrate actors registered; GP / contour / glyphs often fall through to
**active** scene — multi-geom deform makes “pick works sometimes” the default.

Offset mutates `grid.points` (correct world=grid invariant); KD-tree only fully
invalidated on offset event — **step deform staleness** still documented as
pre-existing.

### 1.8 Silent pumps (operability CRITICAL)

Pump bodies in `results_viewer.py` use `except Exception: pass` on step / deform /
gate. Pin reattach in director uses `continue`. Restack alone logs.

**User-visible symptom:** empty viewport, registry still lists layers, no toast.  
**Test-visible symptom:** none, unless escalate-on-except mode is on.

Assessment’s ~650 `except Exception` count is in the right order of magnitude
(tree-wide ~651 hits found in verification); the **worst** are the pumps, not
dock construction.

### 1.9 Contract guards stop at the door of the god object

`tests/viewers/test_viewer_state_contract.py`:

```text
_GUARDED_DIRS  = ("ui", "overlays")
_GUARDED_FILES = ("mesh_viewer.py", "model_viewer.py")
# results_viewer.py NOT included
```

UI can stay pure while orchestration reintroduces double-render and direct
`SetVisibility`. Mesh/model still have many `plotter.render()` sites (allowlist
budget). Multi-agent development will keep “fixing” ui/ while the real dual
paths live in the unguarded shell.

---

## 2. Test reality (why green CI coexists with breakage)

### 2.1 Strengths (real)

- AST pure-H5 / pure-diagram / pure-IR  
- Dispatcher matrix unit lock (`test_dispatcher_contract.py`)  
- Gate truth tables on **fake** layers (S2b)  
- Per-diagram IR via `RecordingBackend`  
- Session **JSON** schema round-trip (v8)  
- Deform-follow structural + many unit behavior tests  
- Stage-activation mask math  

### 2.2 Critical blind spots

| Gap | Detail |
|---|---|
| **15 `@pytest.mark.qt` tests never run in default CI** | `pyproject.toml` addopts: `-m not qt`; CI does not install pytest-qt |
| **Matrix tests mock pumps** | Prove which pumps fire, not that ResultsViewer pump bodies stay correct |
| **Session serialize ≠ session apply** | Full graph restore (G2+pin+deform+contour+line_force) essentially untested |
| **contour + line_force live together** | Serialize only; never both attached + scrubbed |
| **Envelope** | Zero viewer tests (feature not implemented) |
| **False confidence** | Fake actors would have green-passed the historical empty-`_actors` gate bug |

CHANGELOG already records: first real run of qt smokes found ActiveObjects never
seeded and API bit-rot. That class is still **outside CI**.

### 2.3 Estimated fraction of interaction fragility invisible to CI

**~40–60%.** Unit seams are green; product multi-gesture paths are not.

### 2.4 Top 15 adversarial tests (ranked) — do these before features

1. Contour + line_force + deform scrub (values + points)  
2. Composition gate with **real** backend diagrams (not fakes)  
3. Two geometries × independent deform × gate  
4. Stage pin + scrub + real contour components  
5. Stage activation × stage pin masks  
6. N-layer eye cascade: gate count == 1, render == 1, occlusion correct  
7. **Full session round-trip** (save → new viewer → assert state + values)  
8. Auto-apply clim/cmap on real contour  
9. Section cut + deform + stage (cut must not warp)  
10. Pick on second geom after hide/show cycle  
11. ActiveObjects seed on single-stage open  
12. Export animation with concurrent geoms  
13. Stage change while scrubber play running  
14. Composition hide → add layer → show (snapshot invalidation)  
15. Envelope (xfail until implemented — documents hole)

Many of 1–7 can be **headless pump-level** tests without full `exec()` if pumps
are extracted or invoked through a test harness — that is the right sequencing
for B1.

---

## 3. Adversarial critique of the assessment doc

### 3.1 What it got right

- Domain catalog is ahead of ParaView for structural OpenSees work  
- Deformation-as-geometry-state (0058) is the right model  
- Do not re-open VTK engine debate / no proxy layer / no pre+post megaviewer  
- Jupyter default + thin web are real product risks  
- Rebuild brain is wrong for *this* pain  

### 3.2 What it got wrong or soft

| Assessment claim | Verdict | Correction |
|---|---|---|
| Tier A “surgical low risk” for isosurface/scope/slice | **Wrong priority** | Feature nodes on an unstable graph — **medium–high** risk until cascade freeze |
| Suggested order A1→A2→A4→A5→A7 | **Mis-ranked** | Disease first: silence ban + dual-path kill + guard god shell |
| “Event machine mature” | **Overstated** | Matrix mature; **adoption incomplete** |
| Dispatcher owns single RENDER | **Overstated** | Direct `plotter.render` remains in results_viewer |
| Motion LOD missing / full quality always | **False** | `MotionLOD` installed (results/mesh/model); results LODs node cloud |
| Depth peel as missing A4 | **Overstated** | `OpacityController` + window depth peel exist; MSAA/quality policy may still be thin |
| `results_viewer` ~3.6k LOC | **Stale** | ~3917 lines now |
| ~650 excepts | **Right order** | ~651 `except Exception` hits; split empty handlers vs real handling in future |
| B1 success = “one file per diagram kind” | **Wrong attribution** | Already `_kinds.py` / decorator; B1 is shell edit size |
| B1/B2 1–2 weeks | **Optimistic** without pump I/O freeze + interaction tests |
| “No full rebuild” tone | **Slightly too comforting** | Strategically right; must mandate cascade freeze first |

### 3.3 Assessment claim verification table

| # | Claim | Verdict |
|---|---|---|
| 1 | Session schema v8 | TRUE |
| 2 | TimeMode non-SINGLE raises | TRUE |
| 3 | blocking=True Jupyter footgun | TRUE |
| 4 | Web Slice 1 view-only | TRUE |
| 5 | Expressions in available_components / dialog | TRUE (discoverability weak) |
| 6 | Kind registry anti-drift | TRUE |
| 7 | Details panel near-empty placeholder | TRUE |
| 8 | Silent pumps → empty scene | TRUE |
| 9 | Dispatcher sole RENDER | OVERSTATED |
| 10 | G-RENDER covers event discipline fully | OVERSTATED (`results_viewer` out) |
| 11 | Motion LOD missing | FALSE |
| 12 | Depth peel missing | OVERSTATED |
| 13 | LOC 3.6k | STALE (~3.9k) |

---

## 4. “Fixing X breaks Y” scoreboard (evidence-backed)

| Sev | Fix X | Breaks Y | Root |
|---|---|---|---|
| **C** | Remove director dual STEP/RENDER | Export / web / headless blank | Dual reconciler |
| **C** | Keep dual, fix only pin-aware pump | Flicker / double work | Pin-blind registry path |
| **C** | Batch exit replay all RENDER subs naively | Tree rebuild storms | No handler-id coalesce protocol |
| **C** | Batch omit RENDER-lane | Contour eye → mesh vanishes | Incomplete batch protocol |
| **C** | Session deserialize skip bad specs | Wrong composition membership | Index compaction |
| **C** | Concurrent default all visible | Coincident Z-fight | No auto-offset |
| **H** | Hide fill on occludes_substrate | Transparent contour + substrate want | Binary occlusion |
| **H** | Geometry eye = visible only (S2b) | Users expect layers hide with geom | Gate vs intent UX |
| **H** | ActiveObjects layer without activating geom | Color map / settings edit wrong geom | `director.compositions` alias |
| **H** | Scalar bar prefix only when >1 visible | Legend keys / session legend restore | Shared prefix/key resolver |
| **H** | Pin reattach silent continue | Layer gone, outline still lists | Silent attach fail |
| **H** | Auto-apply N× DIAGRAM_MODIFIED | Render storm / eye interleave | No gesture_batch on flush |
| **H** | Registry → fire COMP_ACTIVE on any change | Double GATE with eye | Misnamed subscription |
| **M** | Restack global registry order | Per-geom z intent | One VTK order for all geoms |
| **M** | Composition duplicate shares Diagram refs | Delete one comp strips other | Shared instances |
| **M** | Widen G-RENDER to results_viewer | Allowlist churn | Needed anyway |
| **M** | Persist runtime style in session | Schema creep vs “lost my scale” | Product choice |

---

## 5. Cascade-freeze program (replaces assessment Tier A order)

Do **not** start with isosurface. Sequence:

### Phase 0 — Freeze (1–2 weeks, highest ROI for fix–break)

| ID | Work | Done when |
|---|---|---|
| **F0** | Pump silence ban: STEP/DEFORM/GATE log + re-raise in tests (`APEGMSH_VIEWER_STRICT=1` or pytest escalate) | One failing attach fails the test suite, not the viewport |
| **F1** | Collapse Director STEP/STAGE to **intent + observer only** when dispatcher bound; single pin-aware STEP implementation | Matrix row matches runtime; pin scrub golden test green |
| **F2** | Batch-exit protocol: primitives ∪ RENDER-lane (id-coalesced) ∪ UI flush ∪ one render | Composition eye + contour occlusion integration test green |
| **F3** | Extend G-RENDER / G-ARTIFACT to `results_viewer.py` with measured ratchet | New direct `plotter.render` fails CI unless allowlisted |
| **F4** | Session deserialize: **pad holes** for skipped specs so `layer_indices` stay stable | Partial restore test with retired kind mid-list |
| **F5** | Headless adversarial suite for tests §2.4 #1–7 (no full QtInteractor if possible) | Interaction graph has CI-visible net |

### Phase 1 — Product P0 (after freeze)

| ID | Work |
|---|---|
| **P0a** | Notebook-safe default / hard warning |
| **P0b** | Time envelope / range (touches time path — only after F1) |
| **P0c** | Exception diet ratchet (tree-wide, pumps first) |

### Phase 2 — Shell carve (only after Phase 0 tests exist)

| ID | Work |
|---|---|
| **B1′** | Extract `PumpSet` / `SessionController` / `ActorSceneMap` with frozen interfaces |
| **B2′** | Shared viewer runtime for docks/AO/dispatcher construction (don’t merge pump vocabularies blindly) |

### Phase 3 — Catalog (only after Phase 0 green)

Threshold → slice → isosurface → scope. Each is a **new node on a stable graph**.

### Phase 4 — Web product (parallel only if Phase 0 doesn’t slip)

Web Slice 2 scrub/layers; do not pretend web already “shares the product.”

### Rebuild policy

- **No full product rebuild** for fix–break pain.  
- **Shell-only rewrite (C1)** is on the table **if Phase 2 fails twice** under the freeze tests.  
- ParaView / Rust / CAD hosts remain wrong answers to *cascade coupling*.

---

## 6. Agent reports (summary index)

| Agent | Focus | Key contribution |
|---|---|---|
| Event cascade | `_dispatch` / Director / AO | Dual reconciler; batch RENDER-lane gap; pin-blind STEP; scoreboard |
| Shell / geometries | `results_viewer` / 0058 / session | God closure map; pick map holes; session index compaction; three selection surfaces |
| Test gaps | `tests/viewers` / CI | 15 qt tests out of CI; interaction matrix holes; top-15 tests |
| Assessment critique | Doc vs code | Wrong Tier A priority; false MotionLOD claim; incomplete 0056; fragility model missing |

Full agent transcripts retained in session; this file is the durable synthesis.

---

## 7. Immediate actions (if only three)

1. **F1 + F2** — single STEP path; complete batch exit (kills the largest fix–break families).  
2. **F0 + F3** — loud pumps + guard the god file (stops agents from re-introducing dual render).  
3. **F5 items 1, 2, 6, 7** — contour+line_force+deform; real gate; eye cascade occlusion; full session apply.

Everything else (isosurface, theme polish, property descriptors) waits.

---

## 8. Relationship to other docs

| Doc | Role after this review |
|---|---|
| `results_viewer_assessment.md` | Inventory + strengths; **priorities superseded** by §5 here for sequencing |
| `plans_viewer/*` | Hypothesis sources; re-verify before scheduling (some predate 0056 coalescing) |
| ADR 0056 / 0058 | Still the right contracts; **finish adoption**, don’t replace |
| `docs/design/results.md` | User-facing design altitude; keep |

---

*End of adversarial review. Next engineering step: Phase 0 PR plan with tests named in §2.4, not a catalog sprint.*
