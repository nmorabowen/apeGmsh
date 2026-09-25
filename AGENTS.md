# AGENTS.md — working in apeGmsh

Every agent reads this file: Claude Code imports it from `CLAUDE.md`, and
Codex, Cursor and the rest read it directly. It holds the working rules
and routes to where everything else lives; it does not restate the
architecture docs or the ADRs.

## What this repo is

apeGmsh is a Gmsh wrapper for structural FEM with a typed OpenSees bridge
(`apeSees`), an HDF5 model/results format, and Qt/web viewers
([README.md](README.md)). It changes fast, so docs, ADR "shipped" notes and
the skill routinely lag the source. When they disagree, trust these in
order: a live probe of the code, the `src/` definition, the ADR's
*Decision*, then guides and skill. The code wins, and the lagging doc gets
fixed.

**Two skills, two audiences.**
- `skills/apegmsh/` (mirrored to `.claude/skills/apegmsh-helper/`) covers
  how to *use* apeGmsh: writing models, meshes and bridge calls.
- The task guides below cover how to *change* apeGmsh itself.

Neither replaces the other.

## Where things live

| Path | Holds |
|---|---|
| `src/apeGmsh/opensees/architecture/` | the bridge's charter (14 principles), layout, API design, emitter, H5 schema, [testing.md](src/apeGmsh/opensees/architecture/testing.md) (test layers), and [agent-onboarding.md](src/apeGmsh/opensees/architecture/agent-onboarding.md) (the slice prompt template) |
| `src/apeGmsh/opensees/architecture/decisions/` | the ADRs, append-only; the index is its README. **Before numbering a new ADR, list the directory on `origin/main`**, not your worktree: numbers have collided with work that merged after the cut |
| `internal_docs/` | plans (`plan_*.md`), handoffs (`handoff_*.md`), user-facing guide drafts, [changelog_workflow.md](internal_docs/changelog_workflow.md), [docs_style.md](internal_docs/docs_style.md) |
| `docs/` + `mkdocs.yml` | the published site. It follows the docs_style contract, and `mkdocs build --strict` gates it |
| `skills/apegmsh/` | the canonical user skill. `.claude/skills/apegmsh-helper/` is **derived**; never edit the mirror |
| `CHANGELOG.md` | one section per PR, by [changelog_workflow.md](internal_docs/changelog_workflow.md) |
| `tests/` | the suite. Markers and lanes are in `pyproject.toml` and `.github/workflows/tests.yml` |

Do not copy the ADR list, schema versions, element rosters or marker lists
into this file or a guide. Consult the decisions README, `h5-schema.md`
and `pyproject.toml` instead: they are gated or kept current, and a copy
here would drift.

## Build and test

Use a Python that has the extras installed (`pip install -e ".[plot,viewer,dxf]" pytest`;
openseespy for `live`). On the maintainer's machine that is
`C:\Users\nmora\venv\opensees_venv\Scripts\python.exe`. System Python
lacks pyvista, and a `ModuleNotFoundError` there means the wrong
interpreter, not a missing dependency.

**The editable install points at the main checkout, not your worktree.**
`pytest` is safe, because `pythonpath = ["src"]` in `pyproject.toml`. Any
other in-process probe (`python -c`, a viewer, a script) imports **main's**
apeGmsh unless it runs with `PYTHONPATH=<worktree>/src`.

| CI lane (`tests.yml`) | Run locally | Trap |
|---|---|---|
| `lock-tests` | `python scripts/sync_skill.py --check`, then the five lock files the job names | the locks encode contracts that were silently broken before; fix the code, never the lock |
| `static-gates` | `ruff check src/apeGmsh/opensees` and `mypy src/apeGmsh/opensees` | ruff is a hard gate; mypy is at baseline **0**, and the tool versions are pinned in the job |
| `suite` | `pytest tests -q -m "not live and not subprocess and not bench and not qt"` | a CLI `-m` **replaces** the `addopts` one, so repeat `and not qt` or real windows hang the run |
| `qt-window-tests` | `pytest -m qt <one file>`, one process per file | never run two qt files in one process (VTK/Qt pollution) |
| `live-stock` | `pytest tests -m "live and not qt and not subprocess and not bench"` with stock openseespy | a `live` test that skips everywhere is not a passing test; `-rs` shows why it skipped |
| `emit-cost-gate` | `pytest -m bench tests/benchmarks/test_emit_regression_gate.py -s` | it compares emit cost normalised by parse cost against a committed baseline |
| `docs-check` | `mkdocs build --strict` | runs on changes to `docs/`, `src/`, `README`, `CHANGELOG` and `mkdocs.yml` |
| quirk lint (last step of `static-gates`) | `python scripts/check_quirks.py` (self-test: `tests/test_check_quirks.py`) | lessons that recurred after being written down, held as rules. Each finding names its lesson. Fix the code, or waive one site with `# apegmsh-lint: <rule>-ok <reason>`. Never waive a real bug to go green |

- **Never judge a change by the raw full-tree count on Windows.** One
  process over all of `tests/` cascades into hundreds of errors from
  ordering pollution and cp1252 capture, even on a clean tree. Run the
  touched areas, or diff against a baseline run of the same command on
  `origin/main`.
- **Warn-as-contract code** ("warns iff X") is verified with
  `pytest <files> -W error::<Category>`. A bare `pytest` passes while the
  warning fires where the contract says it must stay silent (#317 →
  #321).

## How work lands

- **`--base main` on every PR**, including sequenced ones. Hand-stacking
  with `--base <prev-branch>` merged three PRs into orphaned branches
  (#295–#297, recovered by #298).
- **`main` has no required status checks.** "Zero checks visible" is not
  "green", and `gh pr merge --auto` merges immediately. Run the suite
  locally before merging when CI has not visibly run (#630 during an
  Actions stall, #757 under `--auto`).
- **Before pushing to a PR branch, check `gh pr view <N> --json state`.**
  A push after the merge lands on an orphaned branch (#335 → #336).
  Before merging a branch you just pushed, wait until `headRefOid` equals
  your local tip (#555 → #556). Never pass `--delete-branch` from a
  worktree: it fails on the local step and hides whether the merge landed.
- **Two green PRs can merge into a red `main`** when both edit the same
  set/dict/list literal, because git sees no textual conflict (#605 + #606
  → #608). After merging a PR that shares a literal with an open one,
  re-run the other's tests on the merge.
- **CHANGELOG: insert one section at the anchor and never edit existing
  lines** ([changelog_workflow.md](internal_docs/changelog_workflow.md),
  guarded by `tests/test_changelog_structure.py`).
- **PR bodies:** `gh pr create --body-file -` with a heredoc. `--body -`
  sets the body to a literal "-".
- **Skill changes:** edit `skills/apegmsh/`, run
  `python scripts/sync_skill.py` to regenerate the mirror, and after the
  merge run `python scripts/refresh_user_skill.py` (the user-level copy is
  outside CI's reach).
- **A worktree does not follow `origin/main`.** Before concluding a
  feature is unmerged, check `git log HEAD..origin/main` or
  `git show origin/main:<path>`.

## Task guides

Read the guide before starting that kind of work. Each is a checklist
that points at the lesson; it never copies it.

| Doing this | Read first |
|---|---|
| Adding or changing an OpenSees primitive, a fork (Ladruno) feature, the emit path, a FEMData stream or the H5 schema | [`.claude/skills/apegmsh-bridge-feature/SKILL.md`](.claude/skills/apegmsh-bridge-feature/SKILL.md) |
| Changing `viewers/` or `results/`: Qt/VTK windows and docks, readers, derived results | [`.claude/skills/apegmsh-viewer-results/SKILL.md`](.claude/skills/apegmsh-viewer-results/SKILL.md) |
| Writing an ADR, a plan, a CHANGELOG section, a docs page, an example or the skill | [`.claude/skills/apegmsh-adr-docs/SKILL.md`](.claude/skills/apegmsh-adr-docs/SKILL.md) |

A lesson that bites again and names a pattern a machine can see becomes
a rule in `scripts/check_quirks.py`, proven against the commit that had
the bug. Every other lesson is a line in one of these guides, which
points at the lesson rather than copying it.

## Behavioural guidelines

Moved verbatim from the previous `CLAUDE.md`.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
