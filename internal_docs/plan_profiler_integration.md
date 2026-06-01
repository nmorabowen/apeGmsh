# Plan — Ladruno profiler integration (`ops.profiler.*` emit + skip the reader)

**Status:** proposed (2026-06-01) · **Owner:** nmora · **Scope:** apeGmsh-side
emit of the OpenSees *Ladruno fork*'s stack profiler.

This is the apeGmsh half of item **#5 (Profiler / monitor pipeline)** from the
fork's apeGmsh-facing contract
(`nmorabowen/OpenSees@origin/docs/apegmsh-feature-recs:Ladruno_implementation/06_profiler.md`).
The fork-side work is shipped (P1–P8: engine timing core, phase seams, memory
counters, `profiler(...)` command, **and the out-of-tree Python backend + React
frontend viewer** at `Ladruno_tools/profiler_viewer/`). This plan is the **thin**
apeGmsh surface that emits the control command and gets out of the way.

> **Why this is the smallest of the five features.** The profiler is a *control*
> command (`profiler('start')` … `profiler('report', 'profile.h5')`) that brackets
> the analyze loop — **not** a model-definition primitive (no node/element/material
> to register) and **not** a recorder (no per-step `OPS_Stream` fan-out, no
> `_response_catalog` entry, no class tag). It writes one HDF5 file that an
> **already-shipped, well-factored viewer reads end-to-end**. apeGmsh's job is to
> emit ~3 lines of deck text and gate the fork requirement at run.

---

## Canonicity invariant (and why it mostly doesn't bite here)

`profile.h5` is canonical — if apeGmsh ever reads it, it READS DIRECT, NEVER
TRANSCODES. The fork already ships the canonical reader: `ProfilerResults`
(`Ladruno_tools/profiler_viewer/profiler_results.py`) opens the HDF5 read-only and
returns `manifest()` / `rollup()` / `series()` / `memory()` / `diff()` dicts that
**are** the wire format — no derived/cached on-disk representation, no re-encode.
That loader is **Jupyter-usable as-is** (the module docstring says so explicitly).

So the canonicity question collapses into a build-vs-skip call on whether apeGmsh
*adds its own reader at all* — see the next section. (If it ever does, the rule is
the same as the recorder plan: read the HDF5 groups lazily, surface read-time
interpretation only, never write a sidecar.)

## Governing constraints (non-negotiable)

1. **Fork is opt-in; vanilla never breaks.** apeGmsh must keep running on stock
   `openseespy`. `profiler(...)` does not exist there — gate **at the point of
   use**, never force the fork, never fail at import.
   - *Emit:* `ops.profiler.start()` / `.report('profile.h5')` produce deck text on
     **any** build — emission is just a `profiler ...` line. The fork requirement
     bites only when the deck actually runs: `ops.profiler.start(run=...)` via the
     live emitter, or the subprocess in `ops.tcl(run=True)` / `ops.py(run=True)`,
     fails with a clear *"requires the Ladruno fork build"* message. Plain
     `openseespy` raises its own "unknown command profiler" — apeGmsh catches that
     at the live-emit seam and re-raises the friendly error.
2. **No class tags, no `_response_catalog`.** The profiler has no element/recorder
   class tag — it is a process-global singleton driven by a command. The fork's
   private ≥33000 band (and the dead sub-300 values) is **irrelevant to this
   feature**. Nothing here reads `classTags.h`.
3. **Output is HDF5, not JSON at rest.** `profile.h5` = one group per run under
   `/runs/<id>`, node tree as nested groups (metrics = attrs), `series` as chunked
   datasets, plus a `/memory` group. JSON is only the wire format the fork's
   FastAPI backend serves the React frontend. apeGmsh emits the **filename**; it
   does not touch the schema.

---

## What's different from a recorder (read this before mirroring the recorder plan)

The recorder plan (`plan_ladruno_recorder_integration.md`) is the wrong template
to copy *structurally*, for three reasons — the profiler is a different kind of
thing:

- **It is a driver-bracket command, not a registered primitive.** A recorder is a
  frozen dataclass that `_register`s on the bridge and emits during the model
  fan-out. The profiler is imperative state toggled **around** `analyze` — its
  closest apeGmsh cousin is **`eigen`** (`apesees.py:4691`) and the SSI **step-hook
  wrap** of the analyze loop (`emitter/base.py:275` note), not `MPCO`
  (`recorder.py:339`). `profiler('start')` must emit *before* `analyze`,
  `profiler('stop')`/`'report'` *after*.
- **There is no reader to build.** The recorder plan's center of gravity is
  `LadrunoReader` + the per-family basis lib. The profiler ships its own complete
  reader out of tree. apeGmsh adds **zero** reader code in the recommended scope
  (the build-vs-skip call below argues this explicitly).
- **It needs no `Emitter` Protocol widening for the line itself** — `profiler ...`
  is a single generic command. But the Protocol *is* frozen (ADR 0008), and the
  bridge has no generic "emit an arbitrary control line" method, so we add **one**
  narrow method `profiler(self, *args)` mirroring `recorder(self, kind, *args)`
  (`emitter/base.py:235`). That is the whole Protocol cost.

---

## Recommended thin surface (`ops.profiler.*`)

A single small namespace `_ProfilerNS` (backing `ops.profiler.<verb>(...)`),
mirroring the no-primitive control methods. **Five** verbs, 1:1 with the
**shipped** fork command (`start|stop|reset|report|memory` — verified in
`ladruno:SRC/interpreter/OpenSeesCommands.cpp` `OPS_profiler()`):

```python
ops.profiler.start(deep=False, memory=False, per_step=False)  # profiler start [-deep] [-memory] [-perStep]
ops.profiler.stop()                                           # profiler stop
ops.profiler.reset()                                          # profiler reset
ops.profiler.report('profile.h5', run='caseA')                # profiler report <file> -run <id>
ops.profiler.memory()                                         # profiler memory  -> peak bytes (live only)
```

> [!warning] **No `config` verb / no `-warmupSteps`.** The fork *design doc*
> (`06_profiler.md`) showed a `profiler config -perStep -warmupSteps` verb, but the
> **shipped** command never wired it — `OPS_profiler()` accepts only the five
> subcommands above and would reject `profiler config ...` with
> *"unknown subcommand 'config'"*. So `-perStep` is a **flag on `start`** (not a
> `config` option), and `-warmupSteps` is **not exposed in v1**. Add a `config` verb
> only if/when a future fork build actually implements it — and confirm it on a live
> build first (fork-side ask #1).

- **Deck emit (tcl / py):** each verb appends one `profiler ...` line to the
  emitter stream. Because the profiler brackets the analyze loop, the verbs are
  *ordered relative to* the `analyze_steps` append in `ops.tcl` / `ops.py`
  (`apesees.py:4800` / `4854`). v1 keeps it explicit and simple: the user calls
  `ops.profiler.start(...)` before `ops.tcl(path, analyze_steps=N)` and
  `ops.profiler.report(...)` after — apeGmsh records the verb order and replays it
  bracketing the `analyze` line. (No auto-injection, no "smart" placement — that is
  speculative; the user controls the order.)
- **Live (`ops.analyze`):** `ops.profiler.start(...)` toggles a flag on the bridge;
  `ops.analyze(...)` emits the live `profiler start` → `analyze` → (on the next
  `ops.profiler.report`) `profiler report`. The live emitter wraps the openseespy
  call in `try/except` and re-raises the friendly fork-required error if openseespy
  rejects `profiler`.
- **`memory()` returns a value** (peak bytes) from the live emitter — same
  one-shot-returns-from-live shape as `eigen` (`emitter/base.py:393`); tcl/py emit
  the line and return `None`.

That is the entire feature: one namespace, one Protocol method, the live re-raise.
No new dataclass family, no schema zone, no reader.

---

## Build-vs-skip: the `profile.h5` reader (the explicit call)

**Call: SKIP the apeGmsh-side reader in v1. Point users at the shipped out-of-tree
viewer.** Reasoning, weighed against simplicity-first:

| Factor | Verdict |
|---|---|
| Does a reader already exist? | **Yes** — `ProfilerResults` is complete (manifest/rollup/series/memory/diff), read-direct, and the docstring advertises Jupyter use. An apeGmsh reader would be near-duplication. |
| Does the profiler output belong in `Results`? | **No, weakly.** `Results` is a *model-response* surface (displacements, forces, stresses keyed to nodes/elements). A profiler rollup tree + wall-time series is **solver telemetry**, not model response — it has no nodes, no steps-as-pseudo-time-of-the-structure, nothing the viewers (`results.viewer` / `show_web`) render. Bolting `r.profile()` onto `Results` mixes two ontologies. |
| Is there a lineage story? | **No.** `.mpco`/`.ladruno` readers require `model_h5=` for the fem→model→results chain. `profile.h5` has no model lineage — it keys on solver config, not the structure. `from_*` factories (`Results.py:229/270/367`) would gain a member that breaks their shared contract. |
| What would an apeGmsh reader add over `ProfilerResults`? | **A dependency and a docs pointer.** At most a one-line convenience re-export. Re-implementing diff/normalizers would duplicate verified fork code; re-exporting `ProfilerResults` would couple apeGmsh to a path inside the fork tree (which isn't pip-installed and lives on a git branch). |
| Cost of skipping | **One sentence in the docs.** "Read `profile.h5` with the fork's `Ladruno_tools/profiler_viewer` (headless `ProfilerResults` API or the React viewer)." |

**Do not build a second viewer** (the brief is explicit) and do not build a second
*reader* either — the second reader earns nothing and costs a fork-path coupling.
The escape hatch if a user genuinely wants it in-notebook: a **thin, optional**
`apeGmsh.profiler.open(path)` helper (P3, deferred) that *imports the fork's
`ProfilerResults` if it is on `sys.path`* and otherwise raises a clear "install the
fork's `profiler_viewer` tools" error — a re-export, **never a re-implementation**,
and gated so it never imports at apeGmsh import time. Ship P1–P2 first; only add
P3 if a real user asks.

---

## Seam map (apeGmsh files to touch)

Grounded against current `src/`. Anchors verified by Read/Grep.

| Seam | File(s) | Mirror of | Change |
|---|---|---|---|
| Protocol method | `src/apeGmsh/opensees/emitter/base.py` (`Emitter`, `recorder` at :235, `eigen` at :393) | `recorder(self, kind, *args)` | Add **one** `profiler(self, *args: int \| float \| str) -> int \| None` (returns peak bytes from live `memory`, `None` else) |
| tcl emit | `src/apeGmsh/opensees/emitter/tcl.py` | `recorder` impl | Emit `profiler <args...>` line |
| py emit | `src/apeGmsh/opensees/emitter/py.py` | `recorder` impl | Emit `ops.profiler(<args...>)` |
| live emit + fork gate | `src/apeGmsh/opensees/emitter/live.py` | `eigen` live impl | Call `ops.profiler(...)`; wrap in `try/except` → re-raise friendly *"requires the Ladruno fork build"* on unknown-command; return peak bytes for `memory` |
| h5 / recording emit | `src/apeGmsh/opensees/emitter/h5.py`, `emitter/recording.py` | `eigen` no-op / capture | h5 **no-op** (profiler is runtime telemetry, nothing to archive — same rationale as `eigen`); recording **captures** for tests |
| Namespace | new `src/apeGmsh/opensees/_internal/ns/profiler.py` (`_ProfilerNS`) | `_RecorderNS` (`ns/recorder.py`) + `_AnalysisNS` no-param verbs (`ns/analysis.py:467`) | Five verbs (`start/stop/reset/report/memory`) → build the `profiler` arg tuple → emit via bridge |
| Bridge wiring | `src/apeGmsh/opensees/apesees.py` (`eigen` :4691, `analyze` :4653, `tcl` :4765, `py` :4826) | `eigen` driver + `_register` (:5020) | Mount `ops.profiler` namespace; record verb order; bracket `analyze` in deck emit; live `try/except` re-raise |
| Export wiring | `src/apeGmsh/opensees/_internal/ns/__init__.py`, `opensees/__init__.py` | existing NS exports | Export `_ProfilerNS` |
| Docs/skill | **canonical** `skills/apegmsh/references/` (new `profiler.md` or a section in `ladruno.md`) — **NOT** the `.claude/skills/apegmsh-helper/` mirror (derived via `sync_skill.py` + CI `--check`) | recorder ref | Document `ops.profiler.*`; **point at the out-of-tree viewer for reading `profile.h5`** |

**No** new entries in `_response_catalog.py`, `_element_capabilities.py`,
`results/`, or `results/schema/`. That absence is the point.

---

## Testing strategy (no fork at test time)

The whole feature is emit + a live re-raise, so testing is **fork-free and
fixture-free** — even lighter than the recorder plan:

- **Deck-emit tests (tcl + py):** assert the emitted deck contains the literal
  `profiler start -deep`, `profiler report profile.h5 -run caseA`, etc., **and**
  that the lines bracket the `analyze` line in the right order. This is the bulk of
  the suite. No fork, no openseespy.
- **Recording-emitter tests:** assert the `RecordingEmitter` captures each
  `profiler(...)` call with the right args (mirrors existing `eigen` recording
  tests).
- **Fork-gate test:** drive the live emitter against a **stub** that raises on the
  `profiler` call (stock-openseespy simulation); assert apeGmsh re-raises the
  friendly *"requires the Ladruno fork build"* `RuntimeError`, **not** the raw
  openseespy error. (Same shape as any "vanilla rejects a fork command" test.)
- **No `profile.h5` fixtures** — apeGmsh reads nothing. (If P3 ships, one tiny
  fixture + a skip-if-`ProfilerResults`-absent test.)

---

## Phased delivery

Two phases ship the whole recommended scope; P3 is deferred/optional.

### P1 — Emit surface `ops.profiler.*`  *(no fork, no fixture)*
- `_ProfilerNS` (**five** verbs) + the one `Emitter.profiler(...)` Protocol method +
  tcl/py/live/h5/recording impls. Grammar exactly (shipped fork command):
  `profiler start [-deep] [-memory] [-perStep]` / `stop` / `reset` /
  `report <file> [-run <id>]` / `memory`. **No `config` / `-warmupSteps`** (not in
  the shipped command).
- Mount `ops.profiler` on the bridge; record verb order so deck emit brackets the
  `analyze` line.
- **Verify:** deck-emission tests (tcl + py) assert the literal `profiler ...`
  lines AND their order around `analyze`; recording-emitter captures; full apeGmsh
  suite green.

### P2 — Live fork-gate + docs  *(no fork at test time; stub the rejection)*
- Live emitter wraps `ops.profiler(...)` in `try/except`; re-raises the friendly
  fork-required error. `memory()` returns peak bytes from live, `None` from tcl/py.
- Docs: canonical `skills/apegmsh/references/profiler.md` documents `ops.profiler.*`
  and **points at `Ladruno_tools/profiler_viewer`** (headless `ProfilerResults` +
  React viewer) for reading `profile.h5` (let `sync_skill.py` regenerate the mirror).
  CHANGELOG; contract row.
- **Verify:** fork-gate stub test (friendly re-raise, not raw error); `memory()`
  return-value test; docs cross-link the viewer, do **not** promise an apeGmsh
  reader.

### P3 — *(deferred, optional)* thin `apeGmsh.profiler.open(path)` re-export
- Only if a real user wants `profile.h5` in-notebook without leaving apeGmsh.
  Imports the fork's `ProfilerResults` if on `sys.path`; else raises a clear
  install-the-tools error. **Re-export, never re-implement.** Gated so it never
  imports at apeGmsh import time.
- **Verify:** skip-if-absent test; one tiny fixture; assert it forwards to
  `ProfilerResults` (no duplicated diff/normalizer logic).

---

## Open questions

> [!question] **Deck-emit ordering of the profiler bracket.** `ops.tcl` /
> `ops.py` append the `analyze` line *after* the model fan-out (`apesees.py:4800` /
> `4854`). v1 has the user call `ops.profiler.start()` before and
> `ops.profiler.report()` after, and apeGmsh replays that order around `analyze`.
> Is an explicit `ops.profiler.report(...)` call after `ops.tcl(...,
> analyze_steps=N)` ergonomic enough, or should `ops.tcl` grow a
> `profile='profile.h5'` convenience kwarg that auto-brackets? **Leaning explicit
> (no kwarg) for v1** — the kwarg is speculative configurability; add it only if
> the explicit form proves clumsy in real use.

> [!question] **Does `ops.analyze` (live, single-call) need a paired
> `ops.profiler.report` driver?** `ops.analyze` builds+emits+runs in one call
> (`apesees.py:4653`); there is no natural "after analyze" user seam in live mode.
> Options: (a) `ops.analyze(profile='profile.h5')` kwarg that wraps the live run in
> start/…/report; (b) require deck emit (`ops.tcl`/`ops.py`) for profiled runs.
> **Leaning (b) for v1** — profiled runs are deliberate and usually subprocess'd;
> avoid threading profiler state through the live one-shot. Revisit if users want
> live profiling.

> [!question] **`memory()` return type.** The fork's `profiler('memory')` returns
> peak bytes (a scalar). Live returns it; tcl/py emit the line and return `None`.
> Confirm the scalar shape against a current-fork build before locking the return
> annotation (the fork doc says "→ live/peak bytes snapshot"). Low risk — it is one
> int.

---

## Fork-side asks (request from the Ladruno team, not work around)

The profiler is the *thinnest* surface, so the asks are minimal:

1. **Stable `profiler(...)` command grammar** — confirm the final subcommand /
   flag spelling on a current-fork build so the emitted lines match exactly. **The
   shipped command is `start|stop|reset|report|memory` only** (`OPS_profiler()`),
   with `-perStep` a flag on `start`; the design doc's `config` verb +
   `-warmupSteps` were **never wired**. Confirm whether `config`/`-warmupSteps` will
   be implemented (then apeGmsh adds the verb) or dropped from the design doc — this
   ask explicitly covers that gap, since the doc and the shipped command disagree.
2. **`profiler_viewer` as the canonical reader** — confirm apeGmsh should point
   users at `Ladruno_tools/profiler_viewer` (`ProfilerResults` + React) rather than
   ship its own reader. (This plan assumes yes; a contract row makes it explicit.)
3. **Packaging hint for `ProfilerResults`** — if P3 ever ships, a stable import
   path / pip-installable shim for the headless loader would let
   `apeGmsh.profiler.open` re-export cleanly instead of `sys.path`-probing a git
   branch.

## Out of scope (this plan)

- The profiler **viewer** (Python backend + React) — ships out of tree
  (`Ladruno_tools/profiler_viewer/`); explicitly NOT apeGmsh's job.
- An apeGmsh `profile.h5` **reader** — skipped (see build-vs-skip); P3 is a deferred
  *re-export*, not a reader.
- **Explicit-integrator / auto-dt surface (the implied 6th workstream).** Separate
  plan. **Cross-reference:** the profiler is *most useful on explicit runs*
  (`CentralDifferenceLadruno` / `ExplicitBathe(LNVD)` + `criticalTimeStep()`
  auto-dt — contract "Recommended apeGmsh approach"), because explicit runs are
  where "where is time going / is `dt` oversampling" matters most. Keep them
  **separate**: the explicit-integrator plan owns `ops.integrator.ExplicitBathe`,
  the auto-dt helper, and lumped-mass consistency; this plan owns only the
  `profiler(...)` bracket. They compose at the user level (profile an explicit run),
  not in code.
- Ladruno recorder #1 / energy #2 / bezier #3-#4 — separate plans.

## References
- Fork profiler design: `…@origin/docs/apegmsh-feature-recs:Ladruno_implementation/06_profiler.md`
- Fork contract: `…:Ladruno_implementation/ladruno_apegmsh_contract.md`
- Out-of-tree viewer (reader + backend + frontend): `…@ladruno:Ladruno_tools/profiler_viewer/`
  (headless loader = `profiler_results.py` `ProfilerResults`; API = `profiler_api.py`; UI = `frontend/`)
- Mirror seams: `eigen` (`opensees/apesees.py:4691`, `emitter/base.py:393`),
  `_RecorderNS` (`opensees/_internal/ns/recorder.py`), `recorder`
  (`emitter/base.py:235`), `_AnalysisNS` no-param verbs
  (`opensees/_internal/ns/analysis.py:467`)
- Sibling plan: `internal_docs/plan_ladruno_recorder_integration.md` (#1/#2)
