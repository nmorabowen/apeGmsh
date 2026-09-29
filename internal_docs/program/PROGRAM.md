# apeGmsh remediation program: charter

**Status:** Active since 2026-09-28.

**Board:** #1203 (pinned). **Panel papers:** #1192.

**Sources:**
- `internal_docs/plan_architectural_strains_2026-09.md`, the assessment, with errata E1–E7.
- `internal_docs/plan_expert_panel_2026-09.md`, the panel's decisions.

This file is the **static** charter: roles, rules, the chain map and the templates. It changes only at weekly triage (#1202). **Live state is on GitHub**:
- the chain issues;
- the slice issues;
- handoff comments;
- the board.

Never keep program state in private memory.

## 1. How the program runs

One session is one **link** of a **chain**. A session is started from a task chip (or by hand) with `/apegmsh-program <link>`.

The orchestrator:
1. boots from the chain issue and its last handoff;
2. checks the link's gates;
3. splits the link into slices (GitHub issues, with slice cards as their bodies);
4. dispatches pinned-model workers;
5. verifies and lands what its merge rights allow;
6. posts a handoff;
7. queues the next link as a chip.

The orchestrator **never implements**, and its context stays small. It reads issue text and worker reports of at most 300 words.

## 2. Roster

Model and effort are pinned in `.claude/agents/prog-*.md`, so routing depends on the slice type, not on per-call judgment.

| Agent | Model | Effort | Tools | Use for |
|---|---|---|---|---|
| `prog-mechanic` | Sonnet | medium | edit, own worktree | Pure moves (codemods), deletions with proofs, lint rules, CI plumbing, fragments, table derivations |
| `prog-builder-opus` | Opus | high | edit, own worktree | Semantic slices with oracles |
| `prog-builder-fable` | Fable | high | edit, own worktree | Semantic slices with oracles (alternate family) |
| `prog-architect-opus` | Opus | max | read + write brief | Independent design for irreversible decisions |
| `prog-architect-fable` | Fable | xhigh | read + write brief | The second, independent design |
| `prog-reviewer-opus` | Opus | high | read-only | Pre-merge review of Fable- or Sonnet-authored `semantic` PRs |
| `prog-reviewer-fable` | Fable | high | read-only | Pre-merge review of Opus-authored `semantic` PRs |
| `prog-auditor` | Haiku | low | read-only + Bash | KPIs, kill criteria, claim probes |

**Rules:**
- The reviewer's family must differ from the author's. Every PR body carries an `Author-model:` line.
- Mechanical PRs proven by `verify_move` need no reviewer.
- Designs for irreversible decisions come from both architects, working independently. The orchestrator reconciles them, and the maintainer ratifies.

**The orchestrator session's own model** is set in the model picker. Workers are pinned regardless, so it only affects dispatch.

| Chains | Orchestrator |
|---|---|
| P0, K, T | Opus @ high |
| C, F, S | Opus @ medium |
| A, B, N, X | Fable @ medium |

## 3. The ledger

**Issues**

| Issue | Chain |
|---|---|
| #1203 | Board (pinned) |
| #1193 | P0 Bootstrap |
| #1194 | A Landing infra |
| #1195 | B Silent defects |
| #1196 | C Safety nets (W0) |
| #1197 | N Navigation & context |
| #1198 | F Fork lane |
| #1199 | X Cuts |
| #1200 | S Hub splits |
| #1201 | K Archive is the program |
| #1202 | T Weekly triage |

**Slice issues** carry the labels `program`, `slice` and `chain:<X>`, plus either `mechanical` or `semantic`. The body is `internal_docs/program/slice_card.md`, filled in. Close the slice when its PR lands.

**Labels**

| Label | Meaning |
|---|---|
| `in-flight` | A worker is on it now. |
| `blocked` | The slice is waiting. |
| `human-gate` | Needs the maintainer. |
| `lesson` | A failure class has occurred twice. It must become a rule within 7 days. |
| `lock:<file>` | One in-flight semantic PR per hub file. Create on demand with `gh label create --force`. |
| `freeze:<file>` | A split window is open, announced on the board with an expiry date. Create on demand. |

**WIP:** at most 8 open program PRs at any time.

## 4. Merge rights and human gates

**The orchestrator may squash-merge** (`gh pr merge --squash`, never `--auto`) only a PR that meets all of these:
- it is labelled `mechanical`;
- the required checks are green on the head SHA;
- the card's done-when holds;
- it passes `scripts/land_pr.py`. Until A2 ships that script, use the "How work lands" checklist in AGENTS.md;
- it carries no `human-gate` label.

After merging, confirm the PR reached `main`:

```
gh api repos/{owner}/{repo}/compare/main...<sha> --jq .status
```

The result must be `behind` or `identical`.

**The maintainer's decision or click is always needed for:**
- semantic PRs;
- ADR acceptance;
- charter changes;
- deletions of user-facing features (the legacy viewer, trame, `ModelData`);
- GitHub settings;
- anything in the fork repository;
- raising any ratchet baseline;
- releases;
- MEMORY or global-config changes.

## 5. Chain map

The full link specs are in the chain issues.

| Link | Title | Gate |
|---|---|---|
| P0.1 | Program scaffolding (this PR) | the maintainer merges |
| P0.2 | Charter v2 ADR ("B with amendments") and the ADR 0102 amendment | ratification |
| P0.3 | Maintainer decisions: K24, K19, E1/E2, `strict`, housekeeping | none |
| A1 | CI no-cancel on main; deterministic API index; generated ADR index | none |
| A2 | Changelog fragments; `land_pr.py`; orphan detector | A1 |
| B1 / B2 | Shepherd the queued D-fixes / D8–D10 | none |
| C1 | Family-completeness gate + `getattr` lint + Unknown-fails | none |
| C2 | Golden corpus + canonical H5 dump + `verify_move.py` | none |
| C3 | Re-key the guards, `inherited_members`, F821, the K2 ratchet, the provenance lint | C1's quirk rule first |
| N1 | `nav.py`, the navigation paragraph, the doc-path rule; delete the atlas and `layout.md` | none |
| N2 / N3 | AGENTS.md ≤150 lines + ADR summaries / move `architecture/` out of `src/` | N1 |
| F0 → F3 | Clean fork workspace → artifact, fail-closed parsers, hook → `FORK_PIN` + `live-fork` + `BackendInfo` → two-sided roster | the maintainer (fork) |
| X1 → X4 | Dead code and dead ends → port to the session → delete the legacy viewer → retirement ledger | B1; C2 for X3 |
| S1 → S3 | `_StageBuilder` + procedures → `build/` layers → `nd`/compose/h5io | C2 + C3; S1 must need ≤1 fix in 7 days |
| K0 → K4 | `VERBS` design → archive completeness → round-trip oracle → fork loader (KC1–6) → dated flip | C1 + C2; F2 for K3 |

**Waves:**

| Week | Links |
|---|---|
| 1 | P0 → A1 · B1 · C1 · N1 |
| 2 | A2 · C2 · C3 · F0/F1 · X1 |
| 3 | F2 · N2 · K0/K1 · S1 · X2 |
| 4 and later | K2 · S2 · X3 · F3 · X4 · N3 · S3 |

Run at most 2–3 orchestrator sessions at once. Put the parallelism inside each session.

## 6. Handoff template

Post this as a comment on the chain issue at the end of every session:

```markdown
### Handoff — <link> — <YYYY-MM-DD> — orchestrator <model> @ <effort>
- **Landed:** #<pr> (<sha>) — <one line> …
- **In flight:** #<pr> — <state> — <agent> …
- **Blocked / human gates:** …
- **Decisions needed from the maintainer:** …
- **Lessons** (second occurrence → open a `lesson` issue): …
- **KPI deltas** (if measured): …
- **Next:** <link> — gates <green|red: which> — chip queued: <yes|no>
```

Then refresh that chain's row on the board (#1203).

## 7. Worker protocol

All `prog-*` agents follow this protocol.

1. **Read first.** Read `AGENTS.md`, then your slice issue (`gh issue view <n>`), then the task guide AGENTS.md routes your change to. Read the reports only where the card cites a section.
2. **Stay in scope.** Edit only the files the card says you own. Never read a file over 2,000 lines whole: use `python scripts/nav.py map|at|where|refs|family` once it exists, otherwise grep an outline and read ranges.
3. **Python.** Use `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe`. Probes need `PYTHONPATH=<your worktree>/src`, because the editable install points at another checkout. Running `pytest` from the worktree root is fine. Never import openseespy or opensees inside the shared pytest process unless the card calls for a live test.
4. **Verify.** Run the card's commands exactly. A fix must prove its regression test fails with the fix reverted, and the PR body must say so. Warn-as-contract changes are verified with `pytest -W error::<Category>`.
5. **Land-ready PR.**
   - Commit with the attribution trailer your harness specifies.
   - Open the PR with `gh pr create --base main --body-file -`.
   - The body lists `Slice: #<n>`, `Author-model: <model>`, `Class: mechanical|semantic`, and a verification summary.
   - Add the CHANGELOG entry as `internal_docs/changelog_workflow.md` currently specifies.
6. **Never:**
   - merge;
   - push to `main`;
   - use `--auto`;
   - touch `C:\Users\nmora\Github\OpenSees_Compile\OpenSees`;
   - edit outside your owned files;
   - raise a ratchet baseline.
7. **Report.** Send the orchestrator at most 300 words: the PR URL, what changed, the verification result, risks and follow-ups. Long material goes in the PR body.
8. **If blocked,** stop and report rather than improvise. That covers a red gate, an ambiguous card, or scope that must grow.

## 8. Kill criteria and KPIs

The kill criteria are measured weekly by the triage (#1202). The fallback in every case is to keep the gates and fixes and stop the structural work.

| Criterion | Action |
|---|---|
| Strict fix-prefixed PR share above 25% for 4 weeks (baseline 22%) | Stop splits, keep the gates. |
| A split PR needs more than 2 fixes within 7 days | Revert the split. |
| Non-program PRs below 50% of the trailing monthly mean for 2 months | The program is starving the product. |
| A phase runs past 2× its estimate | Cancel the next phase. |
| A shadow registry or path reaches 30 days without a flip | Delete it. S3's families list is exempt: it is gate configuration. |
| The fork lane is not green within 30 days of the pin | Make an explicit C-or-D decision. |
| The index-touch share is not below 20% 30 days after fragments ship | Fragments have failed. |

**K3-only criteria (KC1–KC6):**
- KC1: 14 consecutive green `live-fork` nights before any loader PR.
- KC2: the digest must be sensitive to every archived parameter.
- KC3: the tag law is settled first.
- KC4: the xfail ledger may only shrink, and must reach zero within 60 days.
- KC5: the command-registration hook (R10) lands first.
- KC6: a cluster consumer exists within 60 days, or the loader is deleted.

**KPI baselines (2026-09-28):**

| KPI | Baseline |
|---|---|
| Strict fix-prefixed share | 22% |
| Index-touch share | 80–85% |
| Lost-work cases | 12 |
| Red `main` | 2 in 3 weeks |
| Files over 2,000 lines | 15 |
| Lazy cross-package imports | 71% |
| Private cross-package imports | 48% |
| `live-fork` green streak | 0 |
| Legacy viewer LOC | 23.9k |

## 9. Next-link chip prompt

At the end of a session, queue the next link with a task chip whose prompt follows this template:

```text
apeGmsh program link <ID> — <title>. You are the orchestrator.
Recommended session model/effort: <M> @ <E> (switch in the model picker if needed; workers are pinned regardless).
1. Confirm internal_docs/program/PROGRAM.md exists on origin/main; if not, stop and tell the user.
2. Invoke the `apegmsh-program` skill with argument `<ID>` and follow it exactly.
Ledger: chain issue #<n>; board #1203; charter internal_docs/program/PROGRAM.md.
```

## 10. Baseline decisions and open decisions

**Adopted from the panel** (see `plan_expert_panel_2026-09.md` §0):
- **Fork strategy: "B with amendments."** Auto default target, a neutral-layer ratchet, fork-only as a review trigger, and a generated `ladrunoSchema`.
- **The archive is the program**, test-first.
- **Structure.** No DeckProgram, ColumnSpec or picture IR. The `BuiltModel` split is dropped. Pure-move splits happen one hub at a time.
- **Scope.** Cuts of about 37k LOC. Studio is frozen, interop kept.
- **Tooling.** `land_pr.py` of about 80 lines; `preflight` is cut; `nav.py` plus the completeness gate.

**Open decisions** (P0.3 in #1193):
- K24 (staged live);
- K19 (`ModelData`);
- E1/E2 timing;
- `strict: true`;
- the fork-repo asks;
- housekeeping.

## 11. Program end and expiries

The program closes when chains A, B, C, N, X, S1–S3 and K0–K2 are done and the KPIs are on target. F3 and later, K3 and K4 then continue as normal gated work.

`internal_docs/program/prototypes/` **expires on 2026-11-30**. Delete each prototype when the link that productionizes it lands; X4's retirement ledger tracks this.
