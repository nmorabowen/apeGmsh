---
title: apeGmshViewer — Agent Stills and Presenting Views
aliases: [viewer-agent-stills, plan_viewer_agent_stills]
tags: [apeGmsh, apeGmshViewer, agents, stills, plan]
---

# apeGmshViewer — Agent Stills and Presenting Views

> **Status: proposed** (2026-11-01). Nothing built. Slice 0 is
> [ADR 0128](../architecture/decisions/0128-agent-views-in-apegmshviewer.md)
> (Proposed); where this plan and the ADR differ, the ADR wins.

> [!summary] One-line
> Agents get images from apeGmshViewer through a small, versioned view
> spec: a headless batch mode writes PNGs and a numeric manifest, and a
> live mode offers a view to the human (who clicks to apply it) or
> snapshots what the human is looking at. This is the "agent stills after
> P4" open question of ADR 0112, and the move of ADR 0095's `render`,
> `highlight` and `get_selection` that ADR 0112 D8 says is due before P4.

## What exists

- **The agent path today is VTK.** `fem.render` / `results.render` /
  `render_pack` (ADR 0094, `src/apeGmsh/viewers/render.py`) take a closed
  view set: `mesh`, `contour`, `deformed`, `reactions`. ADR 0112 D8 keeps
  them until the app produces the same stills headlessly; ADR 0116 retires
  the renderer at that point.
- **The app already captures headlessly, for proof stills only.**
  `npm run capture` starts `--mode=capture`: a hidden window,
  `webContents.capturePage()` (`capture:still`) and `capture:resize` in
  `apeGmshViewer/src/main/main.ts`. The shots are a fixed slice-acceptance
  sequence hardcoded in `apeGmshViewer/src/renderer/app.ts`
  (`captureResults`, `captureContours`, `captureElements`, ...). An agent
  cannot ask for a view.
- **The store already describes a view.** ADR 0112 D6 (one store, one-way
  flow) means a view is the open file set plus a sequence of reducer
  events. The `Event` union (`apeGmshViewer/src/state/types.ts`) covers
  phase and step, contour field, range and colormap, deform and scale, the
  ghost, section planes, isolate and hide, colour-by, selection and line
  diagrams.
- **The camera is not in the store.** It lives in
  `apeGmshViewer/src/renderer/navigation.ts`, so a still cannot be
  reproduced from state.
- **The second-open path exists.** A second open goes to the running
  window (README, "Open"), which is the transport for live mode.

## Decisions

| | Decision | Why |
|---|---|---|
| D1 | The external contract is a **curated, versioned view spec** (JSON) that compiles to reducer events; never the raw `Event` union | an internal rename must not break agents; a closed set like ADR 0094 INV-10 |
| D2 | The spec **fails closed**: unknown key, missing field or component, a stage or step out of range, or a contour that never reaches the screen is an error naming the reason; never a fallback | the house rule; the capture code's `until(contourShown)` checks already do this |
| D3 | **Every PNG ships with a manifest entry**: the spec as applied, the camera, field and component, colour range and its mode, step and time, every banner and notice on screen, no-data and missing-displacement counts, extrema with node or element id and position, file hashes, app version | an agent assesses and reports from numbers, not pixels |
| D4 | **Batch mode** `apeGmshViewer --render spec.json --out dir/`: many shots in one launch, writes only under `--out` | Electron startup is seconds; ADR 0112 D4 (read-only) holds |
| D5 | **Live present is an offer, never a takeover.** `--present spec.json` reaches the running window as a proposed view (named, with who sent it); the view changes only when the human clicks it. Snapshots never change the view | owner ruling 2026-11-01: the viewer is the human's review tool |
| D6 | **Live snapshot** `--snapshot --out dir/` returns the current view as a spec plus a PNG and its manifest | replaces ADR 0095's selection envelope as the human-to-agent handle |
| D7 | The app never runs Python. A **thin Python shim** finds the installed exe and spawns it; the CLI comes first so any agent can use it | ADR 0112 D4 wall, in the allowed direction |
| D8 | **Style.** Every text the app or an agent writes around a still (offer titles, captions, legend and banner strings in the manifest, the summary an agent hands on) is in English, in the ape-tono voice: impersonal, declarative, a value with its reference and one sentence of consequence. Colours and fonts come only from the app's theme tokens (`apeGmshViewer/src/theme/tokens.ts`, already the apeGraphStyle office palette and Archivo Narrow); batch stills default to its `LIGHT` white-paper theme; contours keep **jet** as the default colormap | stills that leave the app read as office output; one palette, not one per tool |

## Slices

| # | Slice | Done when |
|---|---|---|
| 0 | ADR: D1–D7, the spec v1 schema, the manifest schema; amends 0094 and 0095 as ADR 0112 D8 asks. List `origin/main`'s `architecture/decisions/` for the number first | Accepted by the owner |
| 1 | Camera into state: named views (iso, ±X/Y/Z, fit, fit selection) and explicit eye/target/up, through a `setCamera` event | the same spec gives the same frame twice (pixel diff) |
| 2 | `--mode=render` spec runner on the capture plumbing, with the D2 validator | specs for the four ADR 0094 views render on the fixtures; each bad-spec case refuses with its reason |
| 3 | The D3 manifest | manifest values match the Python oracle on a fixture |
| 4 | Python shim and a skill paragraph (`skills/apegmsh/`, then `scripts/sync_skill.py`) | an agent goes from model to PNG plus manifest in one call |
| 5 | Live `--present` (the D5 offer UI) and `--snapshot`, over the second-open path | snapshot → present → click gives the same view |
| 6 | Parity with `render_pack`, then retire the VTK stills (ADR 0112 P4 gate) | parity checklist and owner-approved screenshots |

Slice 1 blocks 2; 3 and 5 need 2; 4 needs 3; 6 needs all.

## Open questions

- **Linux and CI.** Does `capturePage` on a hidden window work under xvfb?
  Slice 2 checks first; until it does, headless stills are Windows-only.
- **Annotations in spec v1?** Markers or labels at node ids, a pin on the
  extremum. In v1, or a later spec version.
- **Offer lifetime.** Do unclicked offers stack as a list, expire, or does a
  new one replace the last? Decided in the slice 0 ADR.

## Orchestration

Run as one link of chain V ([#1283](https://github.com/nmorabowen/apeGmsh/issues/1283)) under
`/apegmsh-program`: link **VA** (agent views), slices VA0p and VA1 to VA6.
The orchestrator runs Opus @ medium (PROGRAM.md §2, chain V from V2). It
plans, dispatches, verifies and lands, and never implements. State lives in
the slice issues, the handoffs on #1283 and the V row of board #1203.

### Gates

| Gate | Holds | Cleared by |
|---|---|---|
| G0 | ADR 0128 Accepted | the maintainer merges #2644 in the web UI (`land_pr` check 7 refuses an agent) |
| G1 | #2643 has landed; it edits `types.ts`, `reduce.ts`, `effects.ts`, `app.ts` and `viewport.ts`, which VA1 needs | #2643 merged, or its author hands over the files |
| D7 | owner approval of screenshots for VA1 (the view is unchanged), VA5 (the offer list) and VA6 (parity) | stills sent with `SendUserFile`; approval recorded on the PR, bound to those stills |
| DEL | deleting the ADR 0094 renderer is a user-facing deletion (PROGRAM.md §4) | the maintainer, after the parity table |

Only VA0p dispatches before G0.

### Locks

None of these files is a hub by size (all are under 2,000 lines), but every
V slice crosses them. Each holds a `lock:` label, so at most one in-flight
semantic PR edits each: `apeGmshViewer/src/state/types.ts`,
`apeGmshViewer/src/state/reduce.ts`, `apeGmshViewer/src/renderer/app.ts` and
`apeGmshViewer/src/main/main.ts`. The coordinator (T) hands the locks over
between slices, and the queue lives in board text.

### Waves

| Wave | Slices | Starts when |
|---|---|---|
| W0 | VA0p | now |
| W1 | VA1 | G0 + G1 |
| W2 | VA2 | VA1 landed |
| W3 | VA3 ∥ VA5 | VA2 landed (their owned files are disjoint, see below) |
| W4 | VA4 (∥ VA5 if VA5 is still open) | VA3 landed |
| W5 | VA6: parity table, then the deletion PR | VA3 + VA4 landed, VA0p answered, D7 + DEL |

At most two VA PRs are open at once (program WIP: 0 of 8 on 2026-11-01).

### Slices

Owned files below are the planned set. Each card's list is derived at filing
time (`git ls-tree origin/main <path>`, `nav.py where`), never copied from here.

| Slice | Class / agent | Review | Owned (planned) | Verification and done-when |
|---|---|---|---|---|
| VA0p — Linux capture probe | mechanical / `prog-mechanic` | none | a `workflow_dispatch`-only job in `.github/workflows/`, never required | `npm run capture` on the fixture under `xvfb-run` on ubuntu: PNG written or the failure logged. The answer goes on #1283 and into the ADR's open question |
| VA1 — camera and theme into the store | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `types.ts`, `reduce.ts` (`setCamera`, `setTheme`), `renderer/navigation.ts`, `renderer/viewport.ts`, `renderer/viewcube.ts`, `renderer/app.ts`, `test/navigation.test.ts` | the same events give the same frame twice (pixel diff on the maintainer's machine); camera survives a re-read; the `npm run measure` orbit row is within noise of the row before; D7 stills show no visual change |
| VA2 — spec compiler, `--render`, `--describe` | semantic / `prog-builder-opus` | `prog-reviewer-opus` | new `src/spec/` (pure, Node-tested), new `src/renderer/agentRender.ts`, `main/main.ts`, `main/preload.ts`, `scripts/launch.mjs`, `test/spec/` | one Node test per D2 refusal case (each refusal writes no PNG); specs for the four ADR 0094 views render on the fixtures; the human's window is untouched during a `--render` |
| VA3 — the manifest | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `src/spec/manifest.ts`, `src/renderer/agentRender.ts`, `state/selectors.ts`, `test/spec/`, one pytest oracle under `tests/` that writes golden JSON from `Results` | manifest values (range, extrema ids and positions, no-data counts) equal the Python oracle on a fixture; INV-6 |
| VA4 — Python shim and skill | semantic / `prog-builder-opus` | `prog-reviewer-opus` | one new module under `src/apeGmsh/viewers/`, its tests under `tests/`, `skills/apegmsh/`, mirror `.claude/skills/apegmsh-helper/` (Allowed), `_api_index.json` if the name is public | tests run the shim against a stub executable (the spawn contract); one real run on the maintainer's machine is pasted in the PR; `sync_skill.py --check`; the skill states the D8 voice rule for captions |
| VA5 — offers and snapshot | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `types.ts`, `reduce.ts` (offer events), `main/main.ts` (`second-instance`), new `panels/offers.ts`, `renderer/style.css`, `test/offers.test.ts` | INV-3 (spec → apply → snapshot gives the same spec) and INV-5 (no view change without a click) as Node tests; `--present` with no window exits non-zero; D7 approval of the offer list |
| VA6 — parity and retirement | `prog-auditor` (table), then `prog-mechanic` (deletion with proof) | none (mechanical) + DEL | the table: `internal_docs/`. The deletion: `src/apeGmsh/viewers/render.py`, `_stills.py` consumers, their tests, skill lines | every `render` / `render_pack` view maps to a spec, and every gap stops the deletion; if VA0p says no Linux capture, the VTK path stays for Linux and the deletion narrows to match |

VA3 and VA5 are disjoint: VA3 keeps out of `types.ts`, `reduce.ts` and
`main.ts` (the manifest reads state through `selectors.ts`), and VA5 keeps
out of `src/spec/` and `selectors.ts`.

Builders default to Opus, and every semantic PR is reviewed by Opus (the
maintainer rulings of 2026-10-10). A Fable builder needs a reason stated in
its PR body.

Every viewer slice runs `npm run typecheck`, `npm test` and `npm run lint` in
`apeGmshViewer/` (the CI viewer job), plus
`python scripts/check_quirks.py --base origin/main`.

### Per-slice loop

1. File the card from `internal_docs/program/slice_card.md` (labels
   `program slice chain:V semantic|mechanical`, plus `lock:` when it holds
   one). At G0, file VA2 to VA6 as `blocked`.
2. Dispatch the worker in its own worktree, with the PROGRAM.md §7 prompt.
3. Read CI through ccd_pr `get_status`. Then get the Opus review, posted as
   `Review-verdict: <model> <approve|changes> <sha>`.
4. For VA1, VA5 and VA6, open the stills yourself first, then send a subset
   to the maintainer for D7.
5. Land with main's `land_pr.py`. Close the card, release the locks, post the
   §6 handoff on #1283 and refresh the V row of #1203.

### Kill criteria

- VA1 needs more than 2 fixes within 7 days of landing: stop. Moving the
  camera into the store is the riskiest refactor; redesign it before VA2.
- Two shots of the same spec differ on one machine: stop VA2. The pixel and
  manifest determinism in the ADR's acceptance is not met.
- Electron start-up makes a single shot cost more than about 10 s on the
  maintainer's machine: reopen the daemon alternative in the ADR before VA4
  ships the shim.

### Next-link chip

```text
apeGmsh program link VA — agent views in apeGmshViewer (ADR 0128). You are the orchestrator.
Recommended session model/effort: Opus @ medium (switch in the model picker if needed; workers are pinned regardless).
1. Confirm internal_docs/program/PROGRAM.md exists on origin/main; if not, stop and tell the user.
2. Confirm ADR 0128 is Accepted on origin/main; if not, dispatch only VA0p and stop.
3. Invoke the `apegmsh-program` skill with argument `VA` and follow it exactly; the slice plan is internal_docs/plan_viewer_agent_stills.md, "Orchestration".
Ledger: chain issue #1283; board #1203; charter internal_docs/program/PROGRAM.md.
```
