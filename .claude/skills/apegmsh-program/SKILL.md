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
- **Derive owned files, never write them.** For a move, rename or deletion, run
  the card's own verification grep against `origin/main` untruncated (no
  `head`) and paste its file list into the card; if the grep and the Owned list
  disagree, the card is not ready. For every path or symbol a card cites, run
  `git ls-tree origin/main <path>` or `python scripts/nav.py where <symbol>`
  first.
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
  ships, use its `land_pr` script instead.
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

## Coordinator mode (`/apegmsh-program T`, the live coordinator)

One session coordinates the whole program: it holds no slice, arbitrates the
locks, relays the maintainer, and runs the weekly triage. These rules come from
the 2026-10-07 → 10 batch (#1202).

**Boot.** Read the last coordinator handoff on #1202 and the latest comment on
board #1203. They hold the lock holders, the queue, the open gates and the
link prompts. Then run `ListAgents`: if another session titled "T coordinator"
is live, stop (lesson #1507). Announce yourself to every live program session
and ask each to report to you at every landing, blocker, maintainer question,
lock handover and handoff.

**Locks.**
- Only the holder carries a `lock:<file>` label (lesson #1544). The queue order
  lives in board text, never on labels.
- Only a slice that a **live session** is working may hold a lock. A slice with
  no session waits in the queue text.
- A chain does not hold a hub across a series of slices: each slice releases on
  landing and re-queues behind any waiting slice that has a live session.
- Prefer routes that avoid the hub. A slice that finds one skips the queue.
- On landing, the holder removes its labels and gives them to **no one**; the
  coordinator hands the lock over and says so on the next holder's issue.

**Stalls (lesson #1567).** Notifications get lost: app restarts, dead
reviewers, the 600 s watchdog. Never end a turn waiting for CI or a review. At
every check, look for (a) approved and green PRs not landed, (b) open PRs with
no `Review-verdict` after about 6 h, (c) lock holders with no PR activity. Read
the PR, then nudge the session with the exact next command. Reviewers run long
checks in the background.

**Red `main`.**
1. Confirm it from `gh run list --branch main` first.
2. Freeze landings program-wide; the fix PR and the revert are the only
   exceptions.
3. Fix forward under a deadline, with a revert PR ready as a draft.
4. Prove a flaky fix with 2–3 consecutive green reruns.
5. Lift the freeze by a condition every session can check itself: "`main`'s
   run on `<sha>` is green".
6. No masking: no skips, no `os._exit`, no xfail. An environment pin is allowed
   as a labelled TEMP stopgap.
7. Check the environment as well as the code: #1571 was a PySide6 release, not
   the PR that looked guilty.

**Approvals across merges.** A verdict binds to a SHA. An approval carries over
only when the new delta is a merge of `main` that touches no PR file: compare
the PR's own diff before and after, and post the carry-over with its reason. A
hand-resolved conflict needs a delta review.

**Maintainer questions.**
- Ask one question at a time, explaining what it means for their modelling,
  with a recommendation first. Record every ruling on the issue it governs
  before relaying it.
- Deletions (for example AS5) start with a parity table of v1 versus v2. Every
  v1 feature must map to a v2 equivalent; any gap stops the deletion and goes
  to the maintainer.
- Standing rules:
  - Qt viewers are in sunset: they get only changes that unblock CI or stop a
    user-facing crash.
  - Silent wrong answers go to chain B.
  - Opus reviews.
  - Sessions land their own green, approved PRs.

**Wrap-up (batch end or large context).**
1. Each session stops its workers, pushes any WIP, posts its §6 handoff with
   the exact PR heads, review state and lock holders, refreshes its board row,
   and reports.
2. The coordinator archives each session that has handed off.
3. The coordinator posts its own handoff on #1202. That handoff includes the
   **ready-to-paste prompts** for the next batch's sessions, because the next
   batch may run on another machine or account where chips do not exist.

**Portable environment.** Never hard-code one machine.
- Find the Python with the extras (`pyvista` and `gmsh` importable).
- Probes need `PYTHONPATH=<worktree>/src`.
- Treat the OpenSees fork checkout as optional and read-only.
- Cross-session messages reach only sessions on the same machine.
- On Windows, a script can die with 0xC000070A or exit 127 (#1376): retry once.
