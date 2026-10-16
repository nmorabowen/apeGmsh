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

0. **Check for a duplicate before planning (lesson #1507).** Run `ListAgents` for
   a live session whose title names the same link, and
   `gh issue list --label in-flight --label chain:<X>`. If either shows the link
   in progress, stop and report "duplicate of <session>". Chip and next-batch
   prompts name the link exactly.
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
  - A slice queued for a lock records the queue in its body (lesson #1544). The
    `lock:` label is applied only at the handover from the holder, by the
    coordinator when one is live.
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
- **Worker isolation (lesson #1525).** While isolation is flaky, use **either**
  (a) the agent's default isolated worktree and never name a pre-created path,
  **or** (b) the manual pattern: run `git worktree add <path> -b prog/<slice> origin/main`
  yourself, dispatch `general-purpose` with `model:` pinned and "act as the role
  in .claude/agents/<role>.md", and have the worker use `git -C <path>` and push
  plain fast-forwards. Never mix the two; never create a remote branch before
  the first commit. Workers run `python scripts/check_quirks.py --base origin/main`
  (CI's form).
- **Unregistered agent types (observed 2026-10-10).** If the `prog-*` agent types
  are not registered in the session, run workers as `general-purpose` with the
  role file inlined and the model pinned.

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
  C2 has landed: `python scripts/verify_move.py --base <base-sha> --head
  <pushed-sha> <src paths> <dst paths>`; a pre-push `--head WORKTREE` run
  must add `--strict`.

- **Never end a turn on "waiting for CI or review" (lesson #1567).** At the start
  of every turn, re-check `gh pr checks <n>` and the PR's `Review-verdict`
  comments. The nightly `[Program probe]` issue is the backstop.

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
the 2026-10-07 → 13 batches (#1202).

**Boot.** Read the last coordinator handoff on #1202 and the latest comment on
board #1203. They hold the lock holders, the queue, the open gates and the
link prompts. Then run `ListAgents`: if another session titled "T coordinator"
is live, stop. Announce yourself to every live program session
and ask each to report to you at every landing, blocker, maintainer question,
lock handover and handoff.

**Locks.**
- Only the holder carries a `lock:<file>` label. The queue order
  lives in board text, never on labels.
- Only a slice that a **live session** is working may hold a lock. A slice with
  no session waits in the queue text.
- A chain does not hold a hub across a series of slices: each slice releases on
  landing and re-queues behind any waiting slice that has a live session.
- Prefer routes that avoid the hub. A slice that finds one skips the queue.
- On landing, the holder removes its labels and gives them to **no one**; the
  coordinator hands the lock over and says so on the next holder's issue.
- Two requests for one lock: the first live, small slice takes it, and the other
  plans read-only meanwhile (#1805 before #1853, 2026-10-13).

**WIP slots.** The 8-PR cap is program-wide: every worker checks
`gh pr list --label program --state open` before `gh pr create`, not its own
session's count (the program reached 9 of 8 once; X handoff on #1202). At 8, work stays pushed on its
branch. The coordinator may reserve the next free slot for a lock holder, so
an idle lock doesn't wait behind unlocked work (#1901).

**Slice cards and `land_pr`.** Check 9 reads the Owned list in the card
**body**: it refuses a body with no Owned section (an `### Owned files`
heading or an `Owned:` line) before it reads any amendment comment. A
legacy card is fixed in its body, derived from its scope and lock, never from
the PR's diff (a diff-derived list whitelists itself). PR bodies carry
`Slice: #<n>`. Check 10 refuses while `main` is red, or while `main`'s run on
its head H is in progress unless the PR's checks ran on a base containing H.
Check 11 refuses a tests/ or BASELINE PR whose required checks ran on a base
more than 10 commits, or 24 h, behind `main`. Re-run the checks on a fresh merge
ref (merge `main`, or push an empty commit); the Re-run button reuses the old
ref. So a landing right after another waits for `main`'s run on the new head,
or merges `main` first.

**Stalls.** Notifications get lost: app restarts, dead
reviewers, the 600 s watchdog. Never end a turn waiting for CI or a review. At
every check, look for (a) approved and green PRs not landed, (b) open PRs with
no `Review-verdict` after about 6 h, (c) lock holders with no PR activity. Read
the PR, then nudge the session with the exact next command. Reviewers run long
checks in the background.

**Landing order.** A `main` Tests run takes about 50 minutes, and check 10 waits
for it, so serial landings can starve a PR that has been ready longest (3a #2111
lost three heads in a row, 2026-10-16). Give the oldest ready PR the next green
head, and tell the other sessions to hold their landings until it lands.

**Red `main`.**
1. Confirm it from `gh run list --branch main` first.
2. Freeze landings program-wide; the fix PR and the revert are the only
   exceptions. Check 10 enforces this freeze.
3. Fix forward under a deadline, with a revert PR ready as a draft.
4. Prove a flaky fix with 2–3 consecutive green reruns.
5. Lift the freeze by a condition every session can check itself: "`main`'s
   run on `<sha>` is green".
6. No masking: no skips, no `os._exit`, no xfail. An environment pin is allowed
   as a labelled TEMP stopgap.
7. Check the environment as well as the code: #1571 was a PySide6 release, not
   the PR that looked guilty.
8. A PR that is green on a stale base can redden `main` (#1698 vs a ledger row
   added later, 2026-10-13). Label the fix or the revert `fixes-red-main`;
   check 10 refuses anything else while `main` is red. A fix that also edits a
   BASELINE file needs `human-gate` too: ask at once, since it blocks every
   session.
9. Watch `main`'s run on the fix sha in a background loop, and broadcast the
   lift to every session yourself.

**Approvals across merges.** A verdict binds to a SHA. An approval carries over
only when the new delta is a merge of `main` that touches no PR file: compare
the PR's own diff before and after, and post the carry-over with its reason. A
hand-resolved conflict needs a delta review.

**Maintainer questions.**
- Ask one question at a time, explaining what it means for their modelling,
  with a recommendation first and what is won or lost either way. Record every
  ruling on the issue it governs before relaying it. Read an ambiguous reply
  ("approved") literally against the options as framed, and say how you read it.
- D7 screenshots: open the stills yourself before relaying (#1900's
  apparently uncut ghost was a ×10 deformation offset; a probe settled it),
  send a subset with `SendUserFile`, and record the approval on the PR, bound
  to those stills.
- Deletions (for example AS5) start with a parity table of v1 versus v2. Every
  v1 feature must map to a v2 equivalent; any gap stops the deletion and goes
  to the maintainer. List the consumers of the files the feature writes, not
  only its callers: the DomainCapture table missed that the new viewer opens
  only native `results.h5` (#1767, #2079).
- Authorization does not travel between sessions. Another session's
  auto-mode check refuses an approval it only heard from T (X could not flip
  ADR 0122 on a relayed "accept"). Either the maintainer says it in that
  session, or T acts in its own session on the maintainer's own words.
- ADRs: `land_pr` check 7 refuses an agent landing an ADR whose status is
  Accepted, including amendments to one. The maintainer merges those PRs
  through the GitHub web page; put ADR edits in their own small PR so they hold
  up nothing else.
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
- The app pauses a session's cross-session messages after about 10 sends
  without a maintainer message. Batch your messages, and when the pause hits,
  ask the maintainer for any reply to release it.
- A session cannot start a new session on a turn that another session's
  message began; it can only propose one as a card. Starting a session in
  another folder (the fork) needs the maintainer's own words in your session.
- On Windows, a script can die with 0xC000070A or exit 127 (#1376): retry once.
