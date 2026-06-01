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

The three **explicit integrators** are emittable via typed primitives:
`ops.integrator.ExplicitBathe(p=0.54, cfl=True, ...)`,
`ops.integrator.ExplicitBatheLNVD(p=0.54, alpha=0.8, ...)`, and
`ops.integrator.CentralDifferenceLadruno(cfl=True, ...)`. They share an order-free
option grammar (`cfl` / `cfl_abort` / `tangent` / `recompute=N` /
`lump="rowsum"|"diagonal"` / `verbose` / `divergence=f`). Emission works on **any**
build (it's just an `integrator <Type> ...` line); the fork is required only to
*run* the deck — stock OpenSees raises "unknown integrator" at `ops.analyze(...)`.
Defaults: Bathe `p∈(0,1)`=0.54, LNVD `alpha∈[0,1)`=0.80; `lump` defaults to RowSum
on the Bathe schemes and Diagonal on CentralDifferenceLadruno (omit to inherit).
Pair with `ops.system.Diagonal()` (lumped diagonal mass) for explicit runs.

The runtime critical-time-step (`dt_cr`) is exposed on the bridge:
`ops.critical_time_step() -> float` (builds, primes one tiny step, queries — needs
an explicit integrator with `cfl=True`, a `Transient` analysis, and **element**
mass density via `-rho`/`-mass`; the eigensolve ignores `ops.mass` nodal mass).
`ops.analyze_explicit(duration=, safety=0.9, dt_max=None)` drives the whole run:
it queries `dt_cr` and sub-steps `analyze(n, duration/n)` with `n=ceil(duration/
(safety·dt_cr))` (ADR D5), returning an `ExplicitRunResult(n, dt, dt_cr)`. Both
raise `ValueError` on a non-usable `dt_cr` (no `cfl`, non-explicit integrator, or
pure nodal-mass model — the eigensolve uses element mass, not `ops.mass`).

**Stiffening caveat:** `dt_cr` is queried once on the initial stiffness. If the
tangent stiffens mid-run (contact, geometric/material) the true step shrinks and a
fixed `dt` can diverge. `analyze_explicit` warns (`OpenSeesExplicitSolverWarning`)
unless the integrator is built with `cfl_abort=True` (and `recompute=N`), and
re-raises a non-zero `analyze` as `RuntimeError` instead of returning it silently.

**System guards (apeGmsh, build/analyze-time):**
- `system Diagonal`/`MPIDiagonal` + an element with `c_mass=True` → **`BridgeError`**:
  the solver keeps only the diagonal, so off-diagonal *consistent* mass is silently
  dropped. Use lumped mass (drop `c_mass`) with a diagonal solver, or a non-diagonal
  system.
- an explicit integrator + a non-diagonal system → **`OpenSeesExplicitSolverWarning`**:
  correct but factors the full mass each step (loses the O(N) point of explicit).
  `Diagonal` (lumped) is the right pairing.

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
classes; ElasticBeam3d today), **not** the native `vecxz` path: `results.plot.line_force(...)`
prefers the recorder frame (true cross-section roll) over the geometric guess. Energy
lands via **`results.energy(region=)`** → a DataFrame `KE/IE/DW/ULW/RES/ERR` (recorder
`-G energy`).

**Element value channels** read through the same `results.elements.*` API as any
backend, with one Ladruno-specific split (the file is self-describing, so component
names come from the file):

- `results.elements.gauss.get(component="stress_xx")` — continuum stress/strain,
  **neutral** vocabulary (handles both `sigma11` and `sigma_xx`/`eps_xx`/`gamma_xy`
  token forms; cross-backend).
- `results.elements.line_stations.get(component="axial_force")` — beam internal-force
  diagrams, **neutral** (`axial_force`/`shear_y`/…; `localForce` end forces get the
  sign-continuity flip, `basicForce` is one station at ξ=0).
- `results.elements.get(component="localForce")` — **token-driven**: the component is
  the file's `ON_ELEMENTS/<token>` key (`basicForce`/`localForce`/`force`/`globalForce`)
  and the slab is the raw `(T, E, NUM_COLUMNS)` block in the file's column order. (This
  is the one place Ladruno's element API differs from MPCO's neutral
  `nodal_resisting_force_*` — Ladruno is file-driven; the neutral beam view is
  `line_stations`.)

Multi-partition runs (`<stem>.part-N.ladruno`) auto-discover siblings and merge
(node-union + element-concat), like `from_mpco`. Higher-order / Bézier elements are
self-describing: GP world coords are reconstructed from the file's `BASIS` +
`GP_PARAM` via the neutral `apeGmsh._basis` evaluator (shared with the Bézier read
path), since a `.ladruno` from a Bézier element carries no `GLOBAL_GP_COORDS`.

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

**Reading `profile.h5`.** apeGmsh ships no profiler reader, but
`apeGmsh.profiler` is a thin bridge to the fork's out-of-tree viewer:

```python
import apeGmsh
with apeGmsh.profiler.open("profile.h5") as pr:   # → fork's ProfilerResults
    pr.manifest()                                 # run picker rows
    pr.rollup("caseA")                            # flame graph
    pr.series("caseA")                            # per-step time history (the "monitor")
    pr.diff("caseA", "caseB")                     # prove a fix
apeGmsh.profiler.show_web("profile.h5")           # launch the React UI at :8000
```

It **re-exports** `Ladruno_tools/profiler_viewer` (never re-implements). The dir
must be importable — pass `viewer_dir=` , set `LADRUNO_PROFILER_VIEWER`, or have
it on `sys.path`; otherwise a clear install-hint error fires. The one-click
`Profiler_Viewer.bat` / `profiler_viewer.sh` opens a browser with no setup.

## Class-tag band

Fork-only class tags live in the **private `≥33000` band**. Don't hardcode
the dead sub-300 values — read them live from the fork's `classTags.h` /
ledger. (See also `~/.claude/CLAUDE.md`: the OpenSees C++ source is at
`C:\Users\nmora\Github\OpenSees_Compile\OpenSees`.)
