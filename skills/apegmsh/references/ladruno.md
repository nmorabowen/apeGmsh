# OpenSees fork (Ladruno) integration

apeGmsh can target the **Ladruno fork** of OpenSees (`nmorabowen/OpenSees`,
branch `ladruno`) in addition to stock `openseespy`. The fork adds features
apeGmsh emits/reads that stock OpenSees does **not** have. Stock `openseespy`
stays first-class — the fork is **opt-in**; gate fork-only features at the
point of use, never force the fork.

## Fork-only features apeGmsh touches

| Feature | Kind | Notes |
|---|---|---|
| **BezierTri6** | element | fork-only element |
| **ExplicitBathe / ExplicitBatheLNVD / CentralDifferenceLadruno** | explicit integrators | not in stock OpenSees |
| **EnergyBalance** | recorder | fork-only |
| **`.ladruno` recorder** | recorder | `recorder ladruno` — note `.ladruno`, a sibling of the vanilla `.mpco` |
| **stack profiler** | control command | `ops.profiler.*` — brackets the analyze loop; writes `profile.h5` |

The `.ladruno` recorder **does** write `MODEL/LOCAL_AXES` (per-class quaternion
`FRAME`) for beams — unlike vanilla `.mpco`, which omits beam local axes. Don't
carry the stale "MPCO carries no beam LOCAL_AXES" assumption into `.ladruno`
readers.

`Results.from_ladruno(...)` (model_h5 **optional** — a `.ladruno` is self-sufficient)
surfaces this as **`results.elements.local_axes(...)`** → a `LocalAxes` with per-element
scalar-first quaternions plus `.matrices` / `.x_axis` / `.y_axis` / `.z_axis`. The local
axes are the **rows** of each matrix (OpenSees `quatFromMat` stores the transpose), in
global coords — verified: a beam's `.x_axis` points along node1→node2. So beam
orientation for line/section-force diagrams comes straight from `.ladruno` (wired
classes; ElasticBeam3d today), **not** the native `vecxz` path. Energy lands via
**`results.energy(region=)`** → a DataFrame `KE/IE/DW/ULW/RES/ERR` (recorder `-G energy`).

## Contract lives in the fork repo

The exact emit/read contracts — command grammar, apeGmsh touch-points
(`_ELEM_REGISTRY` / `_response_catalog` / `Results.from_ladruno`), the
class-tag band, and the `.ladruno` schema notes — live in the fork's own
reference doc:

> `Ladruno_implementation/ladruno_apegmsh_contract.md` in
> `nmorabowen/OpenSees@ladruno`
> raw: `raw.githubusercontent.com/nmorabowen/OpenSees/ladruno/Ladruno_implementation/ladruno_apegmsh_contract.md`

**Read it before wiring any fork-only emitter or reader.**

## Profiler (`ops.profiler.*`)

The fork's stack profiler is a **control command** that brackets the analyze
loop — not a model primitive, not a recorder (no class tag, no
`_response_catalog` entry). It writes one `profile.h5`; apeGmsh ships **no
reader** — read it with the fork's out-of-tree
`Ladruno_tools/profiler_viewer/` (the headless `ProfilerResults` API, which is
Jupyter-usable, or the React viewer).

The five verbs map 1:1 to the shipped fork command
(`start|stop|reset|report|memory`):

```python
ops.profiler.start(deep=False, memory=False, per_step=False)  # profiler start [-deep] [-memory] [-perStep]
ops.profiler.stop()                                           # profiler stop
ops.profiler.reset()                                          # profiler reset
ops.profiler.report("profile.h5", run="caseA")               # profiler report profile.h5 -run caseA
ops.profiler.memory()                                         # profiler memory
```

There is **no** `config` verb and **no** `-warmupSteps` (the design doc showed
them but the shipped `OPS_profiler()` never wired them; `-perStep` is a flag on
`start`).

**Deck emit (Tcl / Py) — explicit verbs.** Record the verbs *before* the
`ops.tcl(...)` / `ops.py(...)` call; the bridge brackets the appended `analyze`
line. Bracket side is by **verb**, not call order: `start` / `reset` emit before
`analyze`; `stop` / `report` / `memory` after.

```python
ops.profiler.start(deep=True)
ops.profiler.report("profile.h5", run="caseA")
ops.tcl("deck.tcl", run=True, analyze_steps=200)   # → profiler start -deep / analyze 200 / profiler report ...
```

**Live (`ops.analyze`) — the `profile=` kwarg.** The live single-call has no
"after analyze" seam, so it takes the bracket as kwargs:

```python
ops.analyze(steps=200, profile="profile.h5", profile_run="caseA", profile_deep=True)
```

**Fork gate.** Emitting the deck text works on **any** build. Running needs the
fork: `ops.tcl(run=True)` is the recommended profiled path (the `profiler`
command is registered in the Tcl interpreter). The live / py-deck paths call the
openseespy binding `ops.profiler(...)`; on stock openseespy the live emitter
re-raises a clear *"requires the Ladruno fork build"* error. (Whether the fork
exposes `profiler` in the openseespy **Python** module, not only Tcl, is a
fork-side confirmation — prefer the Tcl-deck path until confirmed.)

## Class-tag band

Fork-only class tags live in the **private `≥33000` band**. Don't hardcode
the dead sub-300 values — read them live from the fork's `classTags.h` /
ledger. (See also `~/.claude/CLAUDE.md`: the OpenSees C++ source is at
`C:\Users\nmora\Github\OpenSees_Compile\OpenSees`.)
