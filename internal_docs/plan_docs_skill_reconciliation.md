# Plan — Docs + Skill reconciliation (post-v2.0 churn)

**Status:** in progress · started 2026-05-29
**Driver:** the library churned heavily after the docs/skill froze (skills ~2026-05-24, API docs ~2026-05-26). The repo skill still claims *"current version v1.0"* and *"no separate Assembly class"* — both now false (shipped: v2.0.0 three-broker, plus a large `Unreleased` block; `Assembly` construct shipped in #433).

## Locked decisions

1. **One canonical skill, derive the rest.** Canonical = `skills/apegmsh/` (checked-in, has the `references/` progressive-disclosure structure). Port the more-current body + the `# verified: tests/...` citation convention from `.claude/skills/apegmsh-helper/` into it. Derive `.claude/skills/apegmsh-helper/` via a sync step (rewrites only `name:`/`description:` front-matter). The `anthropic-skills:apegmsh-helper` copy is **runtime/account-provisioned** (injected by the harness at session start — *not* a local plugin, marketplace install, or repo on this machine; traced 2026-05-31). It has no local source to edit, `git push`, or CI-gate; refreshing it is a manual action in the Claude **web/account** skill settings (or Cowork workspace) where it was registered — out of scope for this repo.
2. **Full reconciliation** — cover the entire backlog (see table).
3. **Green-test citations** — every code snippet carries `# verified: tests/<file>::<test>` and is run green against the **worktree** `src/`.

## Verification harness (non-negotiable)

- Python: `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe` (system Python lacks pyvista/openseespy).
- **Editable-install hazard:** `import apeGmsh` resolves to the *main* repo `src/`, NOT this worktree. All snippet verification must force the worktree `src/` (PYTHONPATH-prepend the worktree `src/`, or `pip install -e .` inside the worktree). Confirm `apeGmsh.__file__` points at the worktree before trusting any green result.
- Docs gate: `mkdocs build --strict` (catches broken nav / snippets / refs).

## Backlog (Phase-0 inventory targets)

| Subsystem | Surface | Skill status |
|---|---|---|
| Results system (v1.1→v2.0 three-broker) | `Results.from_native/from_mpco/from_recorders`, `results.model.fem` chain, `results.plot`, `results.viewer()` | partial |
| Native persistence | `FEMData.to_h5/from_h5`, `save_to=`, `g.save()`, `apeGmsh.from_h5` | missing in repo skill |
| Compose v1 (Ph.3) | `g.compose()`, schema 2.10, `ColorMode.MODULE`, chain-phase routing | absent |
| Assembly + couple (#433) | `from apeGmsh.assembly import …` (pin path) | **WRONG** ("no Assembly class") |
| Orphan sweep (#378) | `g.model.geometry.find_orphans/remove_orphans/validate_pre_mesh/find_stale_metadata` | absent |
| Render seam R-B/R-C (ADR 0042) | `results.show_web()`, `serve_web()`, Jupyter controls, `Results.demo()` | absent |
| Stage-bound BCs/constraints (SSI-2.D/E) | `ops.stage()`, `s.fix/s.mass/s.region/s.recorder/s.embedded/s.initial_stress` | absent |
| Higher-order lines (ADR 0037) | `g.mesh.editing.split_higher_order_lines` | absent |
| Shell-on-solid (S1/S2/S5) | `g.node_ndf`, `ndf_for` | absent |
| CAD-import diagnostics | `g.model.io.diagnose() -> ImportHealth`, scale-aware `heal=` | absent |
| Version + mental model | v1.0 → **v2.0.0**; correct contradicted claims | **WRONG** |

## Phase-0 findings that corrected scope (2026-05-29)

- **`Assembly` + `couple` (#433) is NOT on `main`** — lives only on branch `guppi/assembly-couple-api` (`src/apeGmsh/assembly.py` absent here). The skill's *"session IS the assembly"* is therefore **correct for main**. Decision: **omit from current docs/skill, add a brief "coming soon" forward-pointer** (in `references/compose.md` only). Memory ledger (`project_model_chaining_initiative`) wrongly says "1.4 … Shipped" → must be corrected.
- **Version was inconsistent**: `pyproject.toml`=1.6.0, CHANGELOG tagged section=v2.0.0. Decision: **bumped `pyproject.toml` → 2.0.0** (done); docs/skill cite **v2.0.0**.
- **MP constraint emission is NOW SHIPPED** (ADR 0022) — the `.claude` skill's "emission deferred in apeSees / Emitter protocol has no MPC verbs" (§3.3/§5/§10.8) is **stale FALSE**; `equalDOF/rigidLink/rigidDiaphragm/ASDEmbeddedNodeElement` auto-emit. Fix everywhere.
- **Stale schema constants** in `.claude` skill: NEUTRAL `2.4.0`→**2.10.0**, bridge `2.5.0`→**2.12.0** (lines ~731/733, §5.3).
- **Results constructors require `model=`/`model_h5=`** (TypeError otherwise) — every existing constructor snippet in both skills is broken.
- **Editable-install hazard confirmed but moot**: default `import apeGmsh` → main `src/`; worktree is clean off main (identical trees) and we touch no `src/`. Citation runs still force `PYTHONPATH=<worktree>/src` + `LADRUNO_OPENSEES_QUIET=1`.

## Canonical skill taxonomy (locked: progressive disclosure at `skills/apegmsh/`)

`SKILL.md` (router + mental model + v2.0.0 + core workflow) → `references/`: `api-cheatsheet.md`, `fem-broker.md` (+persistence), `opensees-bridge.md` (+emission-shipped/staged/ndf/recorders/cuts), `results.md` (NEW), `compose.md` (NEW), `workflows.md`. Derived copy `.claude/skills/apegmsh-helper/` = structural mirror, front-matter (`name`/`description`) rewritten by `scripts/sync_skill.*`.

## Phases

- [x] **Phase 0 — Inventory** (workflow `weqzmadca`, 11 opus agents, read-only). Output: `tasks/weqzmadca.output` per-subsystem spec.
- [x] **Phase 0.5 — pyproject 1.6.0 → 2.0.0.**
- [x] **Phase 1 — Canonical skill** rebuilt at `skills/apegmsh/` (SKILL.md + 6 references, incl. NEW `results.md` + `compose.md`); `scripts/sync_skill.py` (+ `--check`) derives `.claude/skills/apegmsh-helper/`.
- [x] **Phase 2 — Skill content** (workflow `ws92h4ivt`, 7 opus agents): 71 snippets, 89 tests green, 0 blocked.
- [x] **Phase 3 — Docs top-up** (workflow `w0beyaaxy`, 9 opus agents): 37 surgical edits, 0 blocked. `mkdocs build` clean (strict was *already* red pre-change — pre-existing `../src/...` ADR links; not regressed).
- [x] **Phase 3.5 — Worktree sync incident + Assembly.** See below.

## Worktree-stale incident (2026-05-29, resolved)

Mid-reconciliation the inventory said `Assembly`/#433 was "branch only, not on main" → we chose to omit it. **That was a stale-worktree misread.** PR #433 (+ #439/#440/#441) had merged to `origin/main` *after* this worktree was cut, so the worktree tree lagged `origin/main` by 10 commits and genuinely lacked `assembly.py`. `gh pr view 433` (MERGED, base `main`) + `git show origin/main:...` caught it. Fixed: `git merge origin/main` (conflict-free — disjoint files), verified `tests/test_assembly_compose_pipeline.py` green (14), and flipped the Assembly content in `SKILL.md` + `references/compose.md` + `docs/api/session.md` from "coming soon" → **shipped, sub-path `from apeGmsh.assembly import Assembly`**. The `project_model_chaining_initiative` memory was **correct** — no change. Lesson captured in [[feedback_worktree_lags_origin_main]].

- [x] **Phase 3.6 — `mkdocs build --strict` now passes** (exit 0). Converted 19 pre-existing `src/…`/`../src/…` ADR links (in `CHANGELOG.md` + `guide_{constraints,opensees,partitioning}.md` + `apeGmsh_navigation.md`) to full GitHub blob URLs (the `docs/api/opensees.md` convention). Remaining `../src/…` links live only in `exclude_docs` / non-built pages, so they don't affect the gate.
- [x] **Phase 3.7 — Ultrareview fix:** restored `*_label` kwarg names on the `g.constraints` cheatsheet (+ `tie` tolerance 1e-6→1.0); re-synced derived copy.
- [ ] **Phase 4 — Release hygiene (optional, deferred to user):** the `Unreleased` CHANGELOG block is large; consider cutting a tagged release so the version fact stops drifting.
- [ ] **Phase 1 — Canonicalize skill structure.** Port richer body into `skills/apegmsh/`; add `scripts/sync_skill.*` + a consistency check (derived copy ≡ canonical modulo front-matter).
- [ ] **Phase 2 — Skill content (verified).** Fix wrong facts first, then add each missing subsystem; every snippet `# verified:` + green.
- [ ] **Phase 3 — Docs top-up (verified).** Web viewers + `Results.demo` → `guide_results`/`guide_obtaining_results` + `docs/api/{viewers,results}.md`; compose; assembly+couple; orphan-sweep API. Gate: `mkdocs build --strict`.
- [ ] **Phase 4 — Release hygiene (recommended).** Cut the `Unreleased` block into a tagged release so the version fact stops going stale.
