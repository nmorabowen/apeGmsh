---
name: apegmsh-program
description: >-
  Orchestrate one link of the apeGmsh remediation program. The charter is
  internal_docs/program/PROGRAM.md; the board is issue #1203. Use when the user runs
  "/apegmsh-program <link>" (e.g. A1, B1, C2, K0) or asks to run or continue a
  program link or chain. It boots from the GitHub ledger, splits the link into
  slice issues, dispatches pinned-model prog-* workers, gets cross-family
  review, lands mechanical PRs, posts the handoff, and queues the next link as a
  task chip.
---

# /apegmsh-program <link>: run one program link

You are the **orchestrator**. You plan, dispatch, verify, land and record. You
never implement.

## Hard rules

- **Don't implement.**
  - Never edit `src/` or `tests/` yourself.
  - Never read a file over 2,000 lines.
  - Keep your context small: issue text and worker reports of 300 words or fewer.
- **State lives on GitHub and in `PROGRAM.md`**, never in memory. Read before
  you act, and write before you stop.
- **Merge only what PROGRAM.md §4 allows:** `mechanical`, green, card
  done-when met, landing checklist passed, no `human-gate` label. Label
  everything else `human-gate` and list it in the handoff.
- **Stop and ask the maintainer at every human gate** (PROGRAM.md §4).
- **Never touch the maintainer's fork checkout**
  `C:\Users\nmora\Github\OpenSees_Compile\OpenSees`.
- **Respect the capacity limits:**
  - at most 8 open program PRs;
  - at most one in-flight semantic PR per hub file (`lock:<file>`);
  - hub splits only under `freeze:<file>`.

## 1. Boot (at most 8 tool calls)

1. **Sync the worktree with main.** Use ccd_host `sync_with_base_branch`, or
   confirm `git rev-list --count HEAD..origin/main` is `0`.
2. **Read the chain issue.** Run `gh issue view <chain issue> --comments` to get
   the link spec and the last handoff. The chain issue numbers are in
   PROGRAM.md §3.
3. **Check the link's gates** in the chain issue and PROGRAM.md §5. If a gate
   is red, post a handoff saying `blocked: <gate>` and stop.
4. **Survey in-flight work.** Run
   `gh pr list --label program --state open --json number,title,labels,headRefName`
   and `gh issue list --label in-flight`. Note locks, freezes, and the current
   WIP count.

## 2. Plan

- **Split the link into slices.** Use at most 6 slices, with disjoint owned
  files. Create one slice issue per slice:
  - body: `internal_docs/program/slice_card.md`, filled in;
  - labels: `program slice chain:<X>`, plus `mechanical` or `semantic`, plus
    `lock:<file>` when the slice touches a hub.
- **Choose an agent per slice** (PROGRAM.md §2):
  - mechanical → `prog-mechanic`;
  - semantic → `prog-builder-opus` or `prog-builder-fable`, alternating the
    family across slices;
  - an irreversible design → `prog-architect-opus` **and**
    `prog-architect-fable` in parallel on the same brief. Give each its own
    scratch path, reconcile the two briefs yourself, and have the maintainer
    ratify.

## 3. Dispatch

- **Send all independent slices in one message** of parallel Agent calls:
  - `subagent_type`: `prog-<type>`;
  - prompt: `Program slice #<n> (link <ID>). Read it with gh issue view <n>, then follow internal_docs/program/PROGRAM.md §7. Report in 300 words or fewer.`
  - For architects, add their scratch path.
- **Label each dispatched slice** `in-flight`.

## 4. Verify

- **CI.** For each PR, run ccd_pr `get_status`; if the PR isn't listed there,
  run `bind_pr`. Read CI from that tool. Never poll or schedule CI checks
  yourself.
- **`semantic` PRs get a reviewer from the other family**, based on the PR
  body's `Author-model:` line. Sonnet- and Fable-authored PRs go to
  `prog-reviewer-opus`; Opus-authored PRs go to `prog-reviewer-fable`.
  - Post the verdict as a PR comment:
    `Review-verdict: <model> <approve|changes> <sha>`.
  - If the verdict is `changes`, send the findings back to the same worker
    (SendMessage), or re-dispatch.
- **Mechanical move PRs** must carry `verify_move` evidence in the body, once
  C2 has landed.

## 5. Land

- **Merge only what PROGRAM.md §4 allows.** Run `gh pr merge <n> --squash`:
  - never with `--auto`;
  - never with `--delete-branch` from a worktree.

  Then confirm `compare/main...<sha>` reports `behind` or `identical`. Once A2
  ships, use `scripts/land_pr.py` instead.
- **Clean up:** close the slice issue and remove `in-flight`.

## 6. Record and chain

1. **Glance at the kill criteria** (PROGRAM.md §8). If a split needed more than
   2 fixes within 7 days, or a gate fired, say so in the handoff.
2. **Post the handoff comment** (PROGRAM.md §6) on the chain issue. Tick the
   link only if its done-when is observable on `main`.
3. **Refresh the chain's row on board #1203.**
4. **Open a `lesson` issue** on the second occurrence of any failure class. It
   is due as a rule within 7 days.
5. **Queue the next link:**
   - Spawn a task chip with the PROGRAM.md §9 prompt, filling in the ID, title,
     model/effort and chain issue.
   - Use the chip's `cwd` for fork-repo links (chain F).
   - If the next link waits on the maintainer, queue nothing and state exactly
     what is needed.

## Stop conditions

Record first, then stop, when any of these holds:
- the link is done;
- a human gate is reached;
- a kill criterion fires;
- WIP is full;
- your own context is getting large. In that case, post the handoff and queue a
  continuation chip for the same link.
