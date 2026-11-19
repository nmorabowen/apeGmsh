# Plan — states, quantities and `<stem>.user.h5` (ADR 0130)

The orchestration of [ADR 0130](../architecture/decisions/0130-states-quantities-and-the-user-results-file.md):
how the slices of its plan are split, gated, dispatched and landed. The
decisions are in the ADR; this file says only who does what, in which order,
and what stops the work.

## Orchestration

Run as one link of chain V ([#1283](https://github.com/nmorabowen/apeGmsh/issues/1283))
under `/apegmsh-program`: link **VU** (user results), slices VU0 to VU11,
numbered as the ADR's slice plan. The orchestrator runs Opus @ medium
(PROGRAM.md §2, chain V from V2). It plans, dispatches, verifies and lands,
and never implements. State lives in the slice issues, the handoffs on #1283
and the V row of board #1203.

### Gates

| Gate | Holds | Cleared by |
|---|---|---|
| G0 | ADR 0130 Accepted, with VU0's zone layouts and the combination descriptor written into it as an amendment | the maintainer merges the acceptance PR (`land_pr` check 7 refuses an agent) |
| D7 | owner approval of stills for VU3, VU7, VU8 (viewer half) and VU10 | stills sent with `SendUserFile`; approval recorded on the PR, bound to those stills |
| HG9 | VU9 makes the app write a file for the first time (ADR 0112 D3 amended) | `human-gate` on the PR; the maintainer clicks |
| YOO | VU11 changes the YOO repository, which is not in the program | the YOO session or the owner; the program only hands over the API |

Only VU0 dispatches before G0. ADR 0124's spectrum driver has landed (#2467,
#2538), so VU6 has no external gate.

### Locks

Every viewer slice crosses the store and the app shell, which already carry
`lock:` labels: `apeGmshViewer/src/state/types.ts`,
`apeGmshViewer/src/state/reduce.ts`, `apeGmshViewer/src/renderer/app.ts`,
`apeGmshViewer/src/main/main.ts`. On the Python side,
`src/apeGmsh/results/Results.py` (`lock:src/apeGmsh/results/Results.py`)
is held by VU2 only, and `src/apeGmsh/opensees/emitter/h5.py`
(`lock:src/apeGmsh/opensees/emitter/h5.py`) by VU5 only. New code goes in a
new package, `src/apeGmsh/results/user/`, so VU1, VU6 and VU8 do not need
`Results.py`. The coordinator (T) hands the locks over between slices; the
queue lives in board text.

### Waves

| Wave | Slices | Starts when |
|---|---|---|
| W0 | VU0 | now |
| W1 | VU1 ∥ VU5 | G0 (disjoint: `results/user/` against `opensees/`) |
| W2 | VU2 ∥ VU3 | VU1 landed (disjoint: `src/` against `apeGmshViewer/`) |
| W3 | VU4 ∥ VU6 ∥ VU8 (Python half) | VU4: VU2 + VU3; VU6 and VU8: VU1 |
| W4 | VU7 ∥ VU8 (viewer half) ∥ VU10 | VU7: VU4 + VU6; VU8 viewer: VU8 Python + VU3; VU10: VU3 |
| W5 | VU9, then VU11 | VU9: VU4 + HG9; VU11: VU2 + VU8, handed to YOO |

At most three VU PRs are open at once, and at most one viewer PR, because
the viewer slices share the four locked files. Program WIP was 0 of 8 on
2026-11-01.

### Slices

Owned files below are the planned set. Each card's list is derived at filing
time (`git ls-tree origin/main <path>`, `nav.py where`), never copied from here.

| Slice | Class / agent | Review | Owned (planned) | Verification and done-when |
|---|---|---|---|---|
| VU0 — schema design | design / `prog-architect-opus` ∥ `prog-architect-fable` | the orchestrator reconciles; the maintainer ratifies | one brief each, outside the repo; then an ADR 0130 amendment and an `architecture/h5-schema.md` section | the two briefs cover every zone of D3, the descriptor of D4 (terms, rules, sign, nesting, names) and the staleness rule; the reconciled amendment answers or defers each open question of the ADR |
| VU1 — `.user.h5` writer and reader | semantic / `prog-builder-opus` | `prog-reviewer-opus` | new `src/apeGmsh/results/user/`, its tests under `tests/results/user/` | atomic write (a killed writer leaves the previous file whole: a subprocess test); `/meta` hashes byte-equal the model's; a mismatched `results_hash` reads as stale; zone versions checked |
| VU2 — combination evaluator | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `results/user/`, `Results.py` (the read entry point), the conformance fixture under `tests/fixtures/user_results/` | every D4 rule on the fixture matches hand-computed goldens; a combined force is computed from the per-term forces (the ADR 0124 D6 test); cross-variant references are refused |
| VU3 — app families and picker groups | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `apeGmshViewer/src/reader/` (a `.user.h5` reader), `state/types.ts`, `state/reduce.ts`, `panels/phase.ts`, `panels/legend.ts`, `renderer/app.ts`, `main/main.ts` (the set gains a file), tests under `apeGmshViewer/test/` | the phase bar lists Stages, Modes, Spectra and User Combinations; the pickers list Results and User Results; a stale `.user.h5` shows as stale; D7 stills on the reference models; the `npm run measure` first-frame row is within noise of the row before |
| VU4 — TypeScript evaluator | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `apeGmshViewer/src/` (a combination module), `test/` | the TypeScript evaluator equals VU2's goldens on the same fixture, to `1e-12` relative; unsigned results are refused as a deformed shape |
| VU5 — `apeSees` combination declarations | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `src/apeGmsh/opensees/` (the declaration and its record), `emitter/h5.py`, `architecture/h5-schema.md` (the `/opensees/combinations` section), tests | a declared combination round-trips through `model.h5`; `_api_index.json` rebuilt if the name is public; a name clash with `.user.h5` is refused at read |
| VU6 — modal table and spectra | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `results/user/`, the writer hooks in `opensees/analysis/modal.py`, `gravity_modal.py` and `spectrum.py`, tests | the `/modes` table equals `modalProperties` on a live fork frame; `/spectra` per-mode values equal the driver's in-memory ones; CQC over them equals the driver's own combination |
| VU7 — app Spectra and modal table | semantic / `prog-builder-opus` | `prog-reviewer-opus` | `apeGmshViewer/src/` panels and reader, the four locked files, `test/` | San Ramón eigen reference set shows its table; a spectrum case's CQC draws as a contour and never as a deformed shape; D7 stills |
| VU8 — entities | semantic / `prog-builder-opus` (two PRs: Python, then viewer) | `prog-reviewer-opus` | Python: `results/user/`, `src/apeGmsh/cuts/` (export of cut definitions); viewer: reader, a renderer layer, the locked files | a cut's resultant equals the `cuts` module's on a fixture; storeys and point sets draw where they are; D7 stills |
| VU9 — authoring in the app | semantic / `prog-builder-opus` | `prog-reviewer-opus` + HG9 | `apeGmshViewer/src/` (an authoring panel, the writer in `main/`), `test/` | the write is atomic and keeps one backup; a crash test leaves the previous file whole; nothing else is ever written |
| VU10 — variants | semantic / `prog-builder-opus` | `prog-reviewer-opus` | the manifest reader, a variant switch and a compare view, the locked files, `test/` | the YOO slab cracked and uncracked sets switch and difference on a quantity both define; a combination across variants is refused; D7 stills |
| VU11 — YOO migration | outside the program | — | YOO repository | the slab's custom stages and combinations move from `NativeWriter` to `.user.h5` before ADR 0122 slice 10 |

Builders default to Opus, and every semantic PR is reviewed by Opus (the
maintainer rulings of 2026-10-10). A Fable builder needs a reason stated in
its PR body. VU0 uses both architects, as PROGRAM.md §2 requires for an
irreversible design.

Every viewer slice runs `npm run typecheck`, `npm test` and `npm run lint` in
`apeGmshViewer/`; every slice runs `python scripts/check_quirks.py --base origin/main`.

### Acceptance corpus

The reference models in `C:\nmb\My Libraries\nmb\gitAPE Companion\apeGmsh Reference Models\`
(nine sets, see its `README.md`) are the D7 corpus: the YOO slab for
combinations and variants, San Ramón for modes, the frame and shear
connections for stages and element results. They live on the maintainer's
machine, so CI uses the small fixtures, and the D7 stills come from the
corpus.

### Per-slice loop

1. File the card from `internal_docs/program/slice_card.md` (labels
   `program slice chain:V semantic|mechanical`, plus `lock:` when it holds
   one). At G0, file VU1 to VU10 as `blocked`.
2. Dispatch the worker in its own worktree (`isolation: "worktree"`), with
   the PROGRAM.md §7 prompt.
3. Read CI through ccd_pr `get_status`. Then get the Opus review, posted as
   `Review-verdict: <model> <approve|changes> <sha>`.
4. For VU3, VU7, VU8 (viewer) and VU10, open the stills yourself first, then
   send a subset to the maintainer for D7.
5. Land with main's `land_pr.py`. Close the card, release the locks, post the
   §6 handoff on #1283 and refresh the V row of #1203.

### Follow-ups outside the link

- ADR 0130 D10: the SLE analysis model is not what its `model.h5` describes
  (raw commands after `apeSees`). File as its own issue at G0.
- #2668 (LadrunoBrick `GP_PARAM`) blocks element user results on bricks in
  VU8 only if a card targets them.

### Kill criteria

- The two evaluators disagree on the fixture beyond `1e-12` relative after
  VU4's second fix: stop, and reopen D4's "evaluated on read in two
  evaluators" against a single evaluator with materialised output.
- A killed writer corrupts a `.user.h5` in VU1's test: stop; the atomic-write
  premise of D2 does not hold on this platform.
- VU3 moves the `npm run measure` first-frame row by more than 10 % on the
  YOO slab reference set: stop and profile before VU7.
