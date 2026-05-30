# Blueprint — Diátaxis docs restructure + fresh example curriculum

**Status:** proposed (awaiting curriculum sign-off) · 2026-05-30 · from design workflow `w53cojltx`
Companion to [plan_docs_learnability.md](plan_docs_learnability.md) (the diagnosis). This is the concrete authoring spec.

## Target nav (mkdocs.yml)

```
Home
Tutorials        → Overview · T1 First model in 10 min · T2 · T3 · T4   (hand-held, guaranteed success)
How-to (Recipes) → Overview · ~20 verb-titled task pages (grouped: Geometry/CAD · Meshing · Build · Physics · Solve · Results)
Concepts         → Overview · Core mental model · (de-blended guides) · FEM broker · Parts vs Session
Examples         → the E1–E11 ladder (recognizable structural problems, high-level workflow)
API Reference    → unchanged (autodoc)
Internals & Design → Architecture · Gmsh background · Planning   (demoted, last)
Migration · Changelog
```

> Reconciliation note: the nav-ia agent conservatively kept the old `01_hello_plate…` notebooks in the Tutorials list; per the curriculum decision those are **retired** and replaced by T1–T4 (tutorials) + E1–E11 (examples). Final nav uses the new slots.

## Example curriculum (fresh, from zero — teaches the HIGH-LEVEL workflow)

**Invariant:** every rung drives OpenSees through the **typed `apeSees(fem)` bridge** (never raw `ops.*` strings), uses the loads/masses/sections composites, and post-processes via `Results` + `show_web`. Each rung validates against a closed-form/known answer where one exists.

### Tutorials (hand-held, one path, zero forks)
| Slot | Title | Model | Teaches | Check |
|---|---|---|---|---|
| **T1** | Your first model in 10 minutes | end-loaded cantilever | the whole spine in <40 lines: session → geometry+PG → generate → `get_fem_data` → typed `apeSees` → `Results.from_native(model=)` → `show_web` | tip δ = PL³/3EI |
| **T2** | A plate in tension | unit square, plane stress | same bridge on a 2D **solid**: `nDMaterial`, `element(pg=, pressure=)`, read a field by PG, contour | u = σL/E |
| **T3** | SS beam, the apeGmsh way | simply-supported beam, UDL | the **composites**: `g.loads.line`/`pattern`, `g.masses`, `ops.section` — declare-then-resolve, no manual tributary loops | δ=5wL⁴/384EI, M=wL²/8 |
| **T4** | Save, reload, view | reuse T3 | persistence (`save_to`/`from_h5`) + the notebook-safe results loop (`model=` required; `show_web` vs blocking) | reload hash-integrity reproduces δ |

### Examples (recognizable structural problems, building difficulty)
| Slot | Title | Model | Headline surface | Prereq |
|---|---|---|---|---|
| **E1** | 2D portal frame | moment frame, lateral load | multiple element groups, gravity+lateral patterns, drift | T4 |
| **E2** | Modal analysis | cantilever eigen | `g.masses` + `ops.eigen` + `Results.eigen_modes` + mode playback | E1 |
| **E3** | Fiber sections / M-φ | RC column section | `uniaxialMaterial` + `section.Fiber`, moment-curvature | E2 |
| **E4** | Multi-part assembly | one column placed ×3 | `Part` template + `g.parts.add` + `fragment_all`, address by label | E1 |
| **E5** | Tie non-matching meshes | two members at an interface | `g.constraints.equal_dof/tie` **auto-emit** (v2.0 headline) | E4 |
| **E6** | STEP import + real load | bracket/plate with a hole | `load_step` + heal + name-by-query + face pressure | T2 |
| **E7** | Shell-on-solid | shell wall on solid footing | `g.node_ndf` per-node DOF, mixed-dim tie | E5 |
| **E8** | Choosing a results strategy | reuse E1 | `from_native` vs `from_recorders` vs `from_mpco` — the deferred fork, one home | T4 |
| **E9** | Compose modules | frame from saved `.h5` parts | `g.compose`/`from_h5` chain-phase (+ optional `Assembly`) | E5 |
| **E10** | Nonlinear pushover | fiber moment frame from CAD | fiber sections + DisplacementControl, capacity curve | E3 |
| **E11** | Staged SSI | soil box + structure, stages | `ops.stage` PUSH/CLAIM, `initial_stress`, staged emit | E7 |

**Retired:** all 8 current notebooks (good bones re-homed): they teach raw `ops.*` + force the capture/recorders fork early — the root cause of the diagnosis. The `10b`-with-no-`10` numbering and all dangling prereq pointers go.

## Tutorials track
4 tutorials above. Flagship T1 has a concrete cell-by-cell outline (PRELUDE with ASCII free-body + closed-form stated up front → CELL 1 install `pip install apeGmsh[all]` → … → one printed verified number + browser contour). Zero Parts, zero naming theory, zero strategy fork.

## How-to / Recipes (20, ~80% harvested from existing guides)
Import+heal STEP · tag a face as PG · set local mesh size · apply gravity · apply face pressure · add point load · **fix supports** (currently homeless in `guide_loads §11`) · prescribe SP displacement · tie non-matching meshes · rigid diaphragm/link · run static · run modal · run pushover · read a node's displacement+reactions · plot deformed/contour · choose results strategy · **save & reload** (NEW home) · **compose modules** (NEW home) · export to Tcl/py · multi-part assembly.

### Run × Read — the two strategy axes (the tutorials pick one cell on purpose)
The tutorials use **in-process run + domain capture** (single path, zero forks — the diagnosis condemned the old gallery's early capture-vs-recorders fork). The alternatives live in How-to + the **E8** example, opt-in *after* first success. Two independent axes:

| | `from_native` (capture) | `from_recorders` (classic) | `from_mpco` (STKO) |
|---|---|---|---|
| **Run in-process** (`ops.analyze`) | ← tutorials | how-to | how-to |
| **Export a deck** (`ops.tcl`/`ops.py`) | how-to | how-to | how-to |

Dedicated pages: **How-to "Export to a Tcl / openseespy script"** (`ops.tcl`/`ops.py` — the declaration/export side); **How-to "Get results via MPCO (STKO)"** (`from_mpco`, `model_h5=`); a short **"Running & reading: choose your path"** decision page rendering the grid above; **E8 = the same model solved three ways, agreeing to round-off**. The read-side API is identical across cells. T1 already carries a one-paragraph signpost to these so the single-path choice reads as "one of several."

## Concepts (Explanation)
- **NEW `concepts/mental-model.md`** — short human page (~1.5 screens) adapted from `skills/apegmsh/SKILL.md:86-129`: 6 numbered concepts (session owns kernel · composites by concern · labels/tags/PGs · `.select()` · FEMData snapshot · declare-then-resolve · typed bridge). No taxonomy tables/pitfalls. Becomes the "orient me" target.
- **Split `first_steps.md` (2,986 lines)** — distill Lessons 1–4 into the mental-model page + `guide_basics`; disperse Lessons 5–15 into the existing per-subsystem `guide_*.md`; retire the Epilogue's TODO. (Per-lesson destination map captured in workflow `w53cojltx` output.)
- **De-blend each `guide_*.md`** — lead with a "Tasks on this page" jump-list, verb-titled recipe headings, strip the "Grounded in the current source: <module paths>" manifests (move to a maintainer note).
- **Fact-fixes** — propagate PR #443 into `first_steps`/`architecture`: MP emission is automatic (not deferred); reconcile "15 vs 12 constraint kinds".

## Locked curriculum decisions (2026-05-30)

1. **T1 first win** → line-element cantilever (cleanest δ=PL³/3EI); solids arrive at T2.
2. **E6 CAD part** → **plate with a hole** (clean Kt≈3 vs Peterson check).
3. **E10 pushover** → self-consistent hand-calc check (elastic slope + plastic plateau).
4. **Fiber sections (E3/E10)** → **wire in apeSteel** for section properties/fibers (ecosystem showcase; adds apeSteel dep on those rungs).
5. **Transient rung** → **YES**, add it: **E11 = transient response-history** (SDOF/MDOF vs Newmark closed form). Slots after E2/E10, before the SSI terminal (transient integration is a prereq for SSI-with-ground-motion).
6. **SSI terminal (now E12)** → **full depth, all three stages**: (a) in-situ geostatic → (b) footing-on-contact-springs → (c) transient ground motion with Rayleigh damping. The capstone.
7. **Numbering** → keep the **T1–T4 / E1–E12 split** (makes the tutorial-vs-example Diátaxis boundary visible).

**Final ladder:** T1–T4 (tutorials) + E1–E10 as specified + **E11 transient response-history** + **E12 staged SSI (geostatic→footing-springs→ground-motion)** = **16 rungs**. E3/E10 depend on **apeSteel**; E12 depends on E7 + E11.

## Execution waves (each = one verified PR; examples run green via opensees_venv [+apeSteel])

- **Wave 1 — Skeleton + biggest cold-start win:** new `mkdocs.yml` nav; Tutorials/How-to/Concepts landing pages; `concepts/mental-model.md` (from the skill); **flagship T1** tutorial (verified runnable); hoist install into the hero + reorder cards; propagate PR #443 fact-fixes into `first_steps`/`architecture`.
- **Wave 2 — Tutorials + first examples:** T2–T4, E1 (portal), E2 (modal). Verified.
- **Wave 3 — How-to layer:** the 20 recipe pages (≈80% harvested from guides) — heavily parallelizable.
- **Wave 4 — Concepts de-blend:** split `first_steps.md` into the guides; de-blend each `guide_*.md` (task jump-lists, verb headings, strip module manifests).
- **Wave 5 — Example ladder (core):** E3 (apeSteel fibers), E4, E5, E6 (plate-with-hole STEP), E7, E8, E9.
- **Wave 6 — Analysis capstones:** E10 (pushover), E11 (transient), E12 (staged SSI). Heaviest verification.

Notebooks are committed **pre-executed** (rendered outputs) or with `execute:true`, so the published docs show the success the diagnosis found missing.
