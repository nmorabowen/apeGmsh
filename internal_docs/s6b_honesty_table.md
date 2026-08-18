# S6b — the honesty table (ADR 0098 retirement sweep)

Companion to `plan_results_session_adr0098.md` (S6 note) and ADR 0098
Amendments 1–2. Built 2026-08-18 off `50b8a598` (S6a merged), from four
independent surveys; every load-bearing claim re-verified by hand before
it was written down here.

**Baseline before the sweep:** `tests/viewers/` = 176 files / 2672 tests,
green. 89 test files import `apeGmsh.viewers.diagrams`, carrying 1463
tests between them.

**S6 is de-publication, not deletion.** Four hatches keep importing
diagrams privately: `viewers/render.py` (its view tokens),
`viewers/web_viewer.py` (`show_web`, which holds the live director that
Amendment 2's six kinds ride), `studio/_verbs._yield_setup` (0095
INV-11), and `viewers/results_viewer.py` itself, which
`Results.export_animation` still constructs offscreen.

---

## 1. The four buckets

| Bucket | What | Files | Tests | Disposition |
|---|---|---:|---:|---|
| **A** | Per-kind, re-covered by the seven §4 slots | 17 | 202 | **DELETE** |
| **B** | The six slotless survivors (Amendment 2) | 5 | 172 | **KEEP** |
| **C** | Director / Geometry / Composition / old-window flow | 39 | 552 | **DELETE** |
| **D** | Shared infrastructure that outlives the ontology | 27 | 537 | **KEEP** |
| **E** | `_render_screenshots.py` (0 tests, manual PNG script) | 1 | 0 | **KEEP, repoint** |

Naive reading: delete A + C = 56 files / 754 tests. **That reading is
wrong.** Eleven of those files span buckets — see §2, which is the
actual work of this slice.

---

## 2. The eleven files that need SURGERY, not deletion

Deleting any of these whole removes coverage of something that survives.

### 2a. Survivor-kind tests hiding inside delete candidates

**This is the one that would have done real damage.** Amendment 2 §A2.4
says "the five dedicated test files are NOT retired by S6b". That is
true but **incomplete** — six more survivor-kind tests live in three
files that are otherwise bucket A:

| File | Bucket | Survivor tests to KEEP |
|---|---|---|
| `test_diagram_deform_follow.py` | A (4 tests) | `test_fiber_cloud_follows_deformation:253`, `test_layer_stack_follows_deformation:320`, `test_spring_force_follows_deformation:339` |
| `test_diagram_recorder_frame.py` | A (3 tests) | `test_fiber_cloud_uses_recorder_roll:180`, `test_fiber_recorder_z_axes_empty_for_non_ladruno:224` (+ 2 D-bucket local-axes-overlay tests) |
| `test_no_data_error.py` | A (3 tests) | `test_spring_force_raises_nodata_when_no_springs:106` (+ 1 C-bucket registry test to drop) |

→ **ADR 0098 A2.4 needs correcting**: the survivor set is five dedicated
files **plus six named tests in three shared files**. The decision does
not change; its consequence enumeration was under-specified.

### 2b. Bucket-D files carrying retired-machinery tests (keep file, drop tests)

| File | Keep | Drop |
|---|---:|---|
| `test_threshold.py` | ~30 (reduction / ghost-array math) | ~15 routed through `PumpSet`/`GeometryManager`/`ResultsDirector` |
| `test_scope_gizmo.py` | ~39 (hit-test / drag math) | 4 that construct a live `ResultsViewer` |
| `test_scope_box.py` | ~34 (ScopeController reduction) | ~4 live-`GeometryManager` materialization tests |
| `test_diagram_kind_registry.py` | 9 (registry guard) | 1 — `test_every_kind_round_trips_through_session_codec` (old v13 codec) |

### 2c. Bucket-C files carrying shared tests (drop file, rescue tests)

| File | Rescue | Drop |
|---|---|---:|
| `test_results_viewer_smoke.py` | 5 `build_fem_scene` / `fem_scene` tests → move to a scene-builder file | 14 |
| `test_viewport_presentation.py` | ~5: ADR 0089 constants, `PreferencesManager` width prefs, palette cmap default, icon factory | ~14 |
| `test_ui_tree_perf.py` | 1 `test_parts_tree_refresh_perf` (model/mesh infra) | 3 |

### 2d. Blocked behind a port

| File | Why |
|---|---|
| `test_section_cut_h5_autoload.py` (13 tests) | **All 13 are the auto-load contract**, none the retired picture (that is `test_section_cut_diagram.py`). Retires only after persisted-cuts-boot-as-view-clips exists. See §4. |

### 2e. Repoint, don't delete

`tests/viewers/_render_screenshots.py` — 0 pytest tests, a manual PNG
script driving contour + line_force (A) and spring_force (B). Leave on
internal imports and allowlist it, per the plan's `render_showcase`
precedent.

**Net:** 49 files delete outright; 11 need surgery; 1 waits on a port.

---

## 3. src/ — only two modules are actually dead

`export_animation` → `ResultsViewer.show(run_loop=False)` →
`_realize_headless` (`results_viewer.py:2798`) runs the **full**
`_show_impl` and merely hides the docks (`toggle_focus_mode()`). Its own
docstring: "reuses the full Qt viewer (reuse, don't reimplement…)". So
every panel and overlay built unconditionally in `_show_impl` is still
reachable.

| Module | Verdict |
|---|---|
| `viewers/_session_apply.py` | **DEAD** — only `results_viewer.py:3513`, unreachable (export passes `restore_session=False`) |
| `viewers/ui/_time_history.py` | **DEAD** — only via `_open_time_history` ← a shift-click pick callback |
| `viewers/results_viewer.py` | **MIXED** — module lives (export hatch); dead interior: `_apply_session`, `_maybe_restore_session`/`_prompt_restore`, `_open_time_history`/`_on_shift_click_world`, the interactive pick chain |
| everything else (27 modules) | **KEEP** — hatch- or live-reachable |

**Trap avoided:** `results/session/__init__.py` and `_snapshot.py` show up
in a diagrams grep, but the hits are docstring prose and a
`"diagrams" in data` dict-key check — no imports. `test_session_pure.py`
FORBIDS them from importing diagrams. Allowlisting them would leave two
guards contradicting each other. **They stay off the allowlist.**

### The import guard

Allowlist = 25 modules. Eleven of them (`ui/*`, `overlays/*`) are alive
**solely** because the headless export reuses the fat window; slimming
`export_animation` would kill all eleven at once. The three most fragile
— `ui._isochrone_panel`, `_section_panel`, `_thickness_panel` — are
reachable only via `make_side_panel` from a `setup=` callback attaching a
*slotless* kind, which no in-tree caller does.

So the guard's honest value is a **ratchet against new imports**, run
two-way like the D-SIG registry (an entry that stops importing must be
pruned) — not "the surface is small". Say so in its docstring.

**Near-miss probes** (mutation-test against these, not against obviously
absent modules):

| Probe | Must be | Why it discriminates |
|---|---|---|
| `apeGmsh.viewers._session_apply` | REFUSED | Unescaped-dot regex `apeGmsh.viewers.session\..*` matches it — the `.` after `viewers` eats the `_`. Doubles as "the deleted module must not come back". |
| `apeGmsh.viewers.session._reconciler` | REFUSED | Real, currently clean, sits *between* allowlisted `_realize` and `_specs` — catches a package-keyed guard. |
| `apeGmsh.viewers.render_pack` | REFUSED | Kills naive `startswith`; `render_pack` is already a real exported symbol of `render.py`. |
| `viewers.ui._details_panel`, `viewers.core.scope_controller` | REFUSED | Catch a guard that allows `ui.*` / `core.*` wholesale. |
| `viewers.session._realize`, `viewers.core._legend` | **ALLOWED** | The positive half — without it the guard goes green by refusing everything. |

---

## 4. section_cut — port before retiring

Nothing under `results/session/` or `viewers/session/` reads cuts from
h5 today. The render half already exists (`MeshView.add_clip`,
`_realize._apply_clips`); the load half does not.

- **Attach point:** `Results.session()`. `_boot.py` needs no change —
  its three default-picture branches call it and inherit the behaviour.
- **No double-attach:** `restore_snapshot` builds `ResultsSession(...)`
  directly, so a restored session carries only the file's own clips.
- **Preserve precedence:** the old director prefers a bound
  `OpenSeesModel.cuts()/.sweeps()` over reading the file, falling back to
  `read_cuts_and_sweeps`. A port that always reads the file diverges.
- **Size:** ~40–70 lines across 1–2 files.
- **"kwarg-wins" is moot** — `viewer(cuts=)` retired at S6a and
  `session()` grew no replacement, so only the auto-load half ports.

**DECISION OWED — the narrowing.** `ViewClip` has no `element_ids` and no
`bounding_polygon`. A `SectionCutDef` cut only the elements it named,
optionally bounded by a polygon; a `ViewClip` cuts the **whole view**.
1. Port as-is, document the narrowing.
2. Port, but notice-and-skip cuts that name a subset or carry a polygon
   (matches plan decision 15's degrade-loudly rule). ← *recommended*
3. Do not auto-load; leave cuts manual and retire the contract by
   amendment.

---

## 5. Orphan dispositions

| Behaviour | Documented? | Disposition |
|---|---|---|
| **Legend interactor** (drag scale, right-click menu) | Was; retracted at S6a | **PORT the hide gesture only (S)**; full port needs IR widening first — see note |
| **File menu** (Open results…, Save screenshot…, Export animation…, Close) | Export animation… **yes**, in the skill | **PORT (S)** — ~30 lines on `SessionWindow` |
| **Probe overlay** (point/line/plane + value HUD) | No | **DROP** — point probe superseded by §8 pick → New plot; no session concept for line/plane |
| **Opacity controller** | No | **DROP** — never invoked by production code in *either* window; `set_opacity`/`restore_all` called only from tests |
| **Threshold** | No | **KEEP now, PORT later (M)** — only "reduce by value" tool; logic is Qt-free and heavily tested; nothing forces it into S6b |
| **Eye-cascade** | Partly, already fenced | **DROP the cascade** — geometry/composition rows are a *retired concept*, not an orphan. A `slot_hidden` widening is a separate argument. |

**Legend interactor, the catch:** `_reconciler.py:381-392` tears down and
rebuilds the `LegendController` on every signature change, and the
signature includes `effective_instant` — so any position / font_scale /
fmt a gesture set is discarded on the next scrub tick. **A naive port is
worse than none.** The hide gesture is safe because
`MeshView.set_legend_hidden` is real IR — and today *nothing in the GUI
writes it*.

**Do not delete** `viewers/ui/_eye_icon_delegate.py` — shared with the
Model and Mesh outline trees. Only the results-specific
`ui/_outline_tree.py` is orphaned.

---

## 6. Three S6a defects to fix in this slice

| # | What | Evidence |
|---|---|---|
| 1 | **"Display" dock opens blank.** The shell creates it unconditionally (`_results_window.py:477`), hidden, summoned from View. `SessionWindow` never calls `set_session_widget`. Violates ADR 0087 **INV-2** ("a panel with nothing to show shows exactly one muted hint line"). | verified |
| 2 | **`docs/api/viewers.md:85` names an unreachable path** — "`Preferences → Labels → …`". The session window has no Preferences door; only `apeGmsh.viewers.settings()` reaches it. | verified |
| 3 | **The skill still says Export animation… is a GUI button.** The session window has no File menu and `_scrubber.py` has no export path. | verified |

Fix 1 is either a trimmed Display panel or not registering the dock for
this window; 2 and 3 fall out of the File-menu port in §5.

---

## 7. Open decisions

1. **section_cut narrowing** — §4, options 1/2/3. *Recommend 2.*
2. **How fat does the headless export stay?** Leave it (S6b stays lean,
   11 UI entries stay allowlisted), or slim `export_animation` onto a
   lean window / the session stills — unlocks 11+ modules but is its own
   slice. The ADR keeps the *hatch*, says nothing about the window.
   *Recommend leave, flag the eleven as fragile.*
3. **Threshold** — keep-for-now (recommended) or drop on
   never-advertised grounds.
4. **`slot_hidden`** — argue separately, or not at all.

---

## 8. What S6b actually did — and where it departed from §1–§7

Recorded because the departures are the interesting part: three of them
are cases where executing the plan literally would have removed coverage
of something that survives.

### Deletions — 48 test files (not 56)

14 bucket-A + 35 bucket-C files removed; `tests/viewers/` goes
176 → 128 files. The full suite was green immediately after
(1920 passed, 0 failed), so nothing outside those files depended on them.

### Departure 1 — mixed files are kept WHOLE, not split

§2 planned surgery on eleven files. Executed as **keep entire**, for
2a (survivor-kind tests inside per-kind files), 2b (retired-plumbing
tests inside shared-controller files) and 2c (shared tests inside
old-window files) alike. Reasons, in order of weight:

* splitting risks silently dropping coverage of a survivor, and the
  gain is maintenance-only — every test in these files still passes;
* the controllers the 2b tests drive (**threshold**, **scope**, clip,
  legend) all SURVIVE, so a test reaching them through a `ResultsViewer`
  is an implementation detail of the test, not a retired assertion;
* `ResultsViewer` itself survives behind the export hatch, so the 2c
  construction/lifecycle tests still guard live code.

Spot-checked before committing to it: `test_contour_deform_pose_parity`
(`test_session_realize.py:371`) genuinely re-covers contour
deform-follow, and more strongly than the old shift/reset test — so
bucket A's premise holds where it was checkable. The three
contour-from-gauss averaging variants have no named counterpart, and
they live in a file kept anyway.

### Departure 2 — `test_animation.py` was restored after deletion

Classified C ("driven via ResultsDirector") and deleted with the batch.
**Wrong call**: `Results.export_animation` is a *public, documented* API
(0095 INV-11), not an implementation detail, and those 12 tests were its
only coverage. Restored.

The line that resolves the tension for the rest of the sweep:

> A surviving **public surface** keeps its tests. A surviving
> **implementation detail behind a hatch** does not — ADR 0098
> Consequences explicitly excludes the director from S6's gate.

By that rule the director / registry / dispatcher / panel suites stay
deleted, and `export_animation` and `show_web` do not.

### Departure 3 — a new test the plan never asked for

Deleting the director and registry suites left **the `show_web` hatch
with no coverage at all** — there is no `test_web_viewer.py`, and
Amendment 2 rests the six kinds' survival on precisely that hatch. The
amendment would have shipped as a promise with nothing behind it.

`tests/viewers/test_diagram_hatch_survival.py` (5 tests) is the
load-bearing subset those suites were standing in for: all six kinds
registered by a bare package import, one constructing through the real
`DiagramRegistry`, the §4 catalog still closed at seven, and a near-miss
snapshot refusal. Mutation-checked — dropping one kind's registration
turns 3 of the 5 red. ADR 0098 A2.4 amended to record both this and the
six scattered survivor tests §2a found.

### The import guard

`tests/viewers/test_diagrams_import_guard.py` — 27-entry allowlist,
ratcheted **both** ways, resolving relative imports (the sibling assess
guard skips them, which here would have seen almost nothing and passed
forever). Its docstring states plainly that eleven entries are alive
only because the headless export reuses the fat window, so the guard is
a ratchet against new importers rather than evidence of a small surface.

Five mutations, all red: prefix matching instead of exact, package-keyed
matching, skipping relative imports, a removed allowlist entry, and a
new unlisted importer.

### Deferred out of the slice

* **The two dead src modules** (`viewers/_session_apply.py`,
  `viewers/ui/_time_history.py`). Removing them needs surgery inside the
  4700-line, hatch-live `results_viewer.py` for no behaviour change —
  the worst risk/reward left. Both are allowlisted with a comment
  naming them dead, and the guard's second direction will complain the
  moment they stop importing.
* **The File-menu and legend-hide ports.** Real, but they are
  *additions*; a retirement PR is the wrong carrier.

### S6a defects fixed here

1. The **"Display" dock no longer opens blank** — one muted hint line
   per ADR 0087 INV-2, with a qt-marked regression test that fails for
   the right reason when the fix is reverted.
2. `docs/api/viewers.md` no longer names `Preferences → Labels` as a
   path from this window; it points at `apeGmsh.viewers.settings()`.
3. The skill no longer claims Export animation is a GUI button.

### The section-cut port (§4 decision: notice-and-skip) — shipped

`src/apeGmsh/results/session/_cuts.py` + one call in `Results.session()`;
`_boot.py`, `_views.py`, `_realize.py` untouched as predicted.
`test_section_cut_h5_autoload.py` retired, its contract re-covered by
`tests/results/session/test_cut_autoload.py` (17 tests, mutation-verified
in four directions: skip deleted → 5 red, skip inverted → 10 red, port
reads nothing → 14 red, precedence ignored → 4 red).

**The brief for this port was wrong and the port was right to say so.**
It offered a fallback rule — "skip on `bounding_polygon` plus any
non-empty `element_ids`" — that would have skipped **100 %** of cuts,
because `SectionCutDef.__post_init__` *requires* `element_ids` to be
non-empty. Recorded because a rule that silently refuses everything is
indistinguishable from a rule that works, in a test suite that only
asserts "no clip was attached".

The real rule: compare in **OpenSees-tag space** against the model's
element ids (`results.model.elements()` tags first, else the file's
`/opensees/element_meta/*/ids`), because the bridge does not guarantee
tag == fem_eid — measured tags 2…N against eids 1…N-1, so comparing the
two spaces would be silently wrong. `set(cut.element_ids) >= universe`
⇒ whole-model ⇒ honest translation. **Universe unestablished ⇒ skip with
a notice**, never attach on an unverified guess.
