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
