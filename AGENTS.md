# AGENTS.md — working in apeGmsh

Every agent reads this file: Claude Code imports it from `CLAUDE.md`, and Codex, Cursor
and the rest read it directly. It holds the working rules and routes to where everything
else lives; it does not restate the architecture docs or the ADRs.

## What this repo is

apeGmsh is a Gmsh wrapper for structural FEM with a typed OpenSees bridge (`apeSees`), an
HDF5 model/results format, and Qt/web viewers ([README.md](README.md)). It changes fast, so
docs, ADR "shipped" notes and the skill routinely lag the source. When they disagree, trust
these in order: a live probe of the code, the `src/` definition, the ADR's *Decision*, then
guides and skill. The code wins, and the lagging doc gets fixed.

Two skills, two audiences: `skills/apegmsh/` (mirrored to `.claude/skills/apegmsh-helper/`)
covers how to *use* apeGmsh; the task guides below cover how to *change* apeGmsh itself.

## Where things live

| Path | Holds |
|---|---|
| `architecture/` | the bridge's charter (14 principles), layout, API design, emitter, H5 schema, [testing.md](architecture/testing.md) (test layers), and [agent-onboarding.md](architecture/agent-onboarding.md) (the slice prompt template) |
| `architecture/decisions/` | the ADRs, append-only; the index is its README. **Before numbering a new ADR, list the directory on `origin/main`**, not your worktree: numbers have collided with work that merged after the cut |
| `internal_docs/` | plans (`plan_*.md`), handoffs (`handoff_*.md`), user-facing guide drafts, [changelog_workflow.md](internal_docs/changelog_workflow.md), [docs_style.md](internal_docs/docs_style.md) |
| `internal_docs/program/` | the remediation program: its charter [PROGRAM.md](internal_docs/program/PROGRAM.md), the slice-card template [slice_card.md](internal_docs/program/slice_card.md), and the panel prototypes. Live program state is on GitHub under the `program` label (board #1203). `/apegmsh-program <link>` runs a link as orchestrator; the `prog-*` workers are defined in `.claude/agents/` |
| `docs/` + `mkdocs.yml` | the published site. It follows the docs_style contract, and `mkdocs build --strict` gates it |
| `skills/apegmsh/` | the canonical user skill. `.claude/skills/apegmsh-helper/` is **derived**; never edit the mirror |
| `CHANGELOG.md` | the release ledger, assembled from `changelog.d/` fragments by [changelog_workflow.md](internal_docs/changelog_workflow.md) |
| `tests/` | the suite. Markers and lanes are in `pyproject.toml` and `.github/workflows/tests.yml` |

Do not copy the ADR list, schema versions, element rosters or marker lists into this file or
a guide: the decisions README, `h5-schema.md` and `pyproject.toml` are kept current; a copy drifts.

## Build and test

Use a Python that has the extras installed (`pip install -e ".[plot,viewer,dxf]" pytest`;
openseespy for `live`): on the maintainer's machine, `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe`.
System Python lacks pyvista, so a `ModuleNotFoundError` there means the wrong interpreter.
CI installs against `.github/ci-constraints.txt` (`PIP_CONSTRAINT`), so upstream drift shows up in the nightly `deps-canary` issue, not on `main`; bump the pins with `python scripts/update_constraints.py --run <canary run id>`.
**The editable install points at the main checkout, not your worktree.** `pytest` is safe,
because `pythonpath = ["src"]` in `pyproject.toml`. Any other in-process probe (`python -c`,
a viewer, a script) imports **main's** apeGmsh unless it runs with `PYTHONPATH=<worktree>/src`.

| CI lane (`tests.yml`) | Run locally | Trap |
|---|---|---|
| `lock-tests` | `python scripts/sync_skill.py --check`, then the lock files the job names | the locks encode contracts that were silently broken before; fix the code, never the lock |
| `static-gates` | `ruff check src/apeGmsh/opensees` and `mypy src/apeGmsh/opensees` | ruff is a hard gate; mypy is at baseline **0**, and the tool versions are pinned in the job |
| `suite` | `pytest tests -q -m "not live and not subprocess and not bench and not qt"` | a CLI `-m` **replaces** the `addopts` one, so repeat `and not qt` or real windows hang the run |
| `qt-window-tests` | `pytest -m qt <one file>`, one process per file | never run two qt files in one process (VTK/Qt pollution) |
| `live-stock` | `pytest tests -m "live and not qt and not subprocess and not bench"` with stock openseespy | a `live` test that skips everywhere is not a passing test; `-rs` shows why it skipped |
| `emit-cost-gate` | `pytest -m bench tests/benchmarks/test_emit_regression_gate.py -s` | it compares emit cost normalised by parse cost against a committed baseline |
| `docs-check` | `mkdocs build --strict` | runs on changes to `docs/`, `src/`, `README`, `CHANGELOG` and `mkdocs.yml` |
| quirk lint (last step of `static-gates`) | `git fetch origin main && python scripts/check_quirks.py` (diffs against `origin/main` by default; `--no-base` skips the diff rules; self-test: `tests/test_check_quirks.py`) | lessons that recurred after being written down, held as rules. Each finding names its lesson. Fix the code, or waive one site with `# apegmsh-lint: <rule>-ok <reason>`. Never waive a real bug to go green |

- **Judge a local run against a baseline, not a raw count.** Beside a Ladruno fork build,
  `import opensees` un-skips the `ladruno_fork` tests, which CI never runs and a stale build
  fails. Use CI's selection (a bare `-m "not qt"` also runs the bench cases) or diff against
  the same command on `origin/main`; the bridge guide names the fork paths to exclude.
- **Warn-as-contract code** ("warns iff X") is verified with `pytest <files> -W error::<Category>`.
  A bare `pytest` passes while the warning fires where it must stay silent (#317 → #321).
- **A test restores the process state it touches.** Snapshot and restore `sys.modules` (or use
  `monkeypatch`) when stubbing gmsh/pyvista or purging `apeGmsh.*`, never call a raw
  `gmsh.finalize()` in a fixture, and keep `tests/__init__.py` (without it `tests/opensees`
  shadowed the `opensees` module: #1055; guarded by `tests/test_import_namespacing.py`).
  Earlier leaks: 624a9c2c (gmsh refcount), #742 (a live-ops cache).
- **Never run a process-killing native call in the shared pytest process**: a real Qt viewer
  window (the viewer guide says where it now raises), a 3-D global `recombine()`, or
  gmsh/openseespy from a worker thread. Use a subprocess or skip (3165568c, 06f82f9a: Linux
  CI segfaults, 2026-08-17).

## How work lands

- **Land with `python scripts/land_pr.py <N>`** (`--dry-run` runs the checks and stops). It is
  this checklist as code: it refuses on the first failed check, squash-merges (never `--auto`,
  never `--delete-branch`), then proves the squash commit reached `main`. Its checks, in order:
  - **The base is `main`**, `--base main` on every PR, sequenced ones included: hand-stacking with
    `--base <prev-branch>` orphaned three PRs (#295–#297 → #298) and lost #858 for two months (→ #1169).
    `lock-tests` flags a wrong base but cannot block it; after retargeting, push a commit (a re-run reuses the old merge ref).
  - **Open, not a draft, not already merged**: a push after the merge lands on an orphaned
    branch (#335 → #336), so check `gh pr view <N> --json state` before pushing to a PR branch.
  - **Local `HEAD` equals `headRefOid`, and the tree is clean**; wait until they match before
    merging a branch you just pushed (#555 → #556).
  - **No changed file is inside an open `freeze:<file>` label** (a split window, announced on
    the board with its expiry).
  - **Not conflicting, and the five required checks** (`lock-tests`, `emit-cost-gate`,
    `static-gates`, `suite`, `live-stock`) **are green on the head SHA.** `main` takes squash
    merges only and does not require an up-to-date branch. "Zero checks visible" means conflicting
    or a stalled Actions, not green; pending is a refusal. Never `--auto`: it ignores the non-required
    lanes, and #757 merged under it before any check was required. Run the suite locally when CI
    has not visibly run (#630 merged during an Actions stall).
  - **After the merge, `compare/main...<merge-sha>` reads `behind` or `identical`**; `diverged`
    means it did not reach `main` (#858), and `git merge-base --is-ancestor` cannot tell once the
    head branch is deleted. A nightly Action (`orphans.yml`, `scripts/find_orphans.py`) repeats
    this over recent PRs and opens one `[Orphans]` issue when work is missing.
  - **The branch tip is still the merged head.** The merge pins the head the checks saw
    (`--match-head-commit`); `headRefOid` freezes at the merge, so the script reads `git ls-remote`
    (the replay uses #1097's SHAs). A push after the landing is the orphan detector's job (#1230).
- **Never pass `--delete-branch` from a worktree**: it fails on the local step and hides
  whether the merge landed.
- **Two green PRs can merge into a red `main`** when both edit the same set/dict/list literal:
  git sees no textual conflict (#605 + #606 → #608). After merging a PR that shares a literal
  with an open one, re-run the other's tests on the merge.
- **CHANGELOG: add one fragment `changelog.d/<slug>.md` (one `### ` section) and never edit
  `CHANGELOG.md`**: CI fails a PR that does (`changelog.py --check --base HEAD^1`), because
  GitHub flags every pair of such PRs CONFLICTING (#1267, #1279). A maintainer runs
  `--assemble` at release time; never in CI.
- **PR bodies:** `gh pr create --body-file -` with a heredoc. `--body -` sets a literal "-".
- **Skill changes** follow the adr-docs guide, "The skill": edit `skills/apegmsh/` only and
  regenerate the mirror with `python scripts/sync_skill.py`.
- **A worktree does not follow `origin/main`.** Before concluding a feature is unmerged, check
  `git log HEAD..origin/main` or `git show origin/main:<path>`.

**Navigating the code.** Before reading a file over 2,000 lines or grepping for a Python
symbol, ask `python scripts/nav.py <cmd>`. It parses the AST and never imports apeGmsh, so
it reads your worktree. `map FILE` outlines a file with line ranges, `at FILE:LINE` names the
enclosing symbol, `where NAME` finds definitions, `refs NAME` lists code references without
comments or docstrings, `h5 PATH` sorts HDF5 writes from reads, and `family BASE` lists every
table and dispatch a new member must join. Every answer fits in 60 lines, and a cut one names
the flag that narrows it. Keep Grep for prose, comments and non-Python files, and Read for
the range nav points you to.

## Task guides

Read the guide before starting that kind of work. Each is a checklist that points at the lesson.

| Doing this | Read first |
|---|---|
| Adding or changing an OpenSees primitive, a fork (Ladruno) feature, the emit path, a FEMData stream or the H5 schema | [`.claude/skills/apegmsh-bridge-feature/SKILL.md`](.claude/skills/apegmsh-bridge-feature/SKILL.md) |
| Changing `results/` or `viewers/`: readers, derived results, the results↔viewer boundary. It routes viewer internals to `apegmsh-viewers-change` and visual proof to `apegmsh-viewers-visual-check` (#1170) | [`.claude/skills/apegmsh-viewer-results/SKILL.md`](.claude/skills/apegmsh-viewer-results/SKILL.md) |
| Writing an ADR, a plan, a CHANGELOG section, a docs page, an example or the skill | [`.claude/skills/apegmsh-adr-docs/SKILL.md`](.claude/skills/apegmsh-adr-docs/SKILL.md) |
| Running a remediation-program link as orchestrator (`/apegmsh-program <link>`), or working a program slice issue | [`.claude/skills/apegmsh-program/SKILL.md`](.claude/skills/apegmsh-program/SKILL.md) (orchestrator); [PROGRAM.md](internal_docs/program/PROGRAM.md) §7 (worker protocol) |

A lesson that bites again and names a pattern a machine can see becomes a rule in
`scripts/check_quirks.py`, proven against the commit that had the bug; every other lesson is a
line in one of these guides, which points at the lesson rather than copying it.

## Behavioural guidelines

Condensed from the previous `CLAUDE.md`; they bias toward caution over speed, so use judgment on trivial tasks.

1. **Think before coding.** State your assumptions; present the interpretations instead of picking
   one silently; say so when a simpler approach exists; if something is unclear, stop, name it, ask.
2. **Simplicity first.** The minimum code that solves the problem: no unasked features, single-use
   abstractions, configurability or impossible-case handling. If it reads overcomplicated, rewrite it.
3. **Surgical changes.** Touch only what the request needs, match the existing style, leave adjacent
   code and pre-existing dead code alone (mention it), and remove only the orphans your change made.
4. **Goal-driven execution.** Turn the task into verifiable goals ("fix the bug" → a test that
   reproduces it, then passes), state a step → check plan for multi-step work, and loop until green.
