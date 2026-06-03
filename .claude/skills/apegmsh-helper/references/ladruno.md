# OpenSees fork (Ladruno) integration

apeGmsh can target the **Ladruno fork** of OpenSees (`nmorabowen/OpenSees`,
branch `ladruno`) in addition to stock `openseespy`. The fork adds features
apeGmsh emits/reads that stock OpenSees does **not** have. Stock `openseespy`
stays first-class — the fork is **opt-in**; gate fork-only features at the
point of use, never force the fork.

## Targeting a build — `OpenSeesTarget` (where) vs capabilities (what)

Keep two ideas separate: *which* OpenSees runs, and *what* that build
can do. They are wired through different mechanisms.

**Where** — pin the runtimes the subprocess paths bind, once on the
bridge (see `opensees-bridge.md` → "Which OpenSees runs"):

```python
from apeGmsh.opensees import apeSees, OpenSeesTarget

ops = apeSees(fem, opensees=OpenSeesTarget(
    binary="C:/Program Files/Ladruno/OpenSees/bin/OpenSees.exe",   # ops.tcl(run=True)
    python="C:/Users/nmora/venv/opensees_venv/Scripts/python.exe", # ops.py(run=True)
    require_fork=True,                                             # LIVE-path assertion
))
```

**What** — never inferred from the path. Pointing `binary=` at the fork
does *not* tell apeGmsh the build has `BezierTet10`; fork-only features
stay gated at the point of use (the sections below). To branch yourself,
probe the **live** build:

```python
if ops.capabilities().has_fork:      # OpenSeesCapabilities(has_fork=, has_profiler=, version=)
    ops.element.BezierTet10(pg="Body", material=m)
else:
    ops.element.FourNodeTetrahedron(pg="Body", material=m)
```

Two facts that shape the API:

- **Live (`run`/`analyze`/`eigen`) can't be re-pointed** — `import
  openseespy` binds to the active venv. So `binary`/`python` are inert
  for live; to run fork features in-process, launch the script under the
  fork's venv. `require_fork=True` makes that contract loud (fails at the
  live boundary, not three primitives deep).
- **Subprocess (`tcl`/`py` with `run=True`) *can* be re-pointed** —
  `binary`/`python` (or the `bin=`/`python=` per-call args, or
  `$OPENSEES_BIN`/`$OPENSEES_VENV`) select any build.

`has_fork` tracks the fork-only `profiler` command (confirmed present in
the fork's openseespy **Python** module, not only Tcl) — the same gate
the live emitter uses for `ops.profiler`.

## Fork-only features apeGmsh touches

| Feature | Kind | Notes |
|---|---|---|
| **BezierTri6 / BezierTet10** | elements | fork-only Bézier (Bernstein) continuum elements — typed primitives `ops.element.BezierTri6/BezierTet10` |
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

## Bézier elements (`BezierTri6` / `BezierTet10`)

Two fork-only Bézier (Bernstein) continuum elements (Kadapa 2018), exposed as
typed primitives:

```python
ops.element.BezierTri6(pg=…, thickness=…, material=m, plane_type="PlaneStrain",
                       bbar=False, consistent_mass=False,
                       pressure=None, rho=None, body_force=None)   # 2D, 6 nodes
ops.element.BezierTet10(pg=…, material=m, bbar=False, consistent_mass=False,
                        rho=None, body_force=None, pressure=None)  # 3D, 10 nodes
```

Emit grammar is **flag-prefixed** (each option independently optional), unlike
`SixNodeTri`'s positional tail:

```
element BezierTri6  $tag $n1..$n6  $thick $type $matTag [-bbar] [-cMass] [-pressure $p] [-rho $r] [-bodyForce $b1 $b2]
element BezierTet10 $tag $n1..$n10 $matTag [-bbar] [-cMass] [-rho $r] [-bodyForce $b1 $b2 $b3] [-pressure $p]
```

Key points:

- **`plane_type` (Tri6 only) accepts ONLY `PlaneStrain` / `PlaneStress`** — not the
  `*2D` spellings `SixNodeTri` tolerates (the fork factory rejects them).
- **B-bar guard (Tri6):** `bbar=True` under `PlaneStress` warns
  (`BezierBBarPlaneStressWarning`) and drops the `-bbar` flag (mirrors the fork's
  D5 warn-and-disable). Tet10 has no plane-stress degeneracy, so B-bar is always
  valid (no guard).
- **Node order is verbatim Gmsh.** On a straight-sided mesh the Gmsh `tri6`
  (etype 9) / `tet10` (etype 11) nodes coincide with the element control points, so
  connectivity passes through unpermuted. The tet10 mid-edge order is
  `(1-2, 2-3, 1-3, 1-4, 3-4, 2-4)` — machine-precision-locked (the O11 test).
- **Fork required only to RUN.** Emission (`ops.tcl` / `ops.py`) works on any build;
  running in-process (`ops.run()` / `ops.analyze()`) on a stock build raises a clear
  *"element BezierTri6 requires the Ladruno fork build … use the direct-drive
  fallback"* error rather than a cryptic openseespy failure.
- **Direct-drive fallback (no apeGmsh change needed).** The elements also run via
  *direct-drive*: mesh a straight-sided domain to T6/T10 on stock py3.11, dump
  `nodes` + `fem.elements.<group>.connectivity` to JSON, and feed those verbatim to
  `ops.element('BezierTri6'|'BezierTet10', …)` on the fork build — Gmsh order is
  byte-identical to the control-point order. See the fork's
  `bezier_apegmsh_integration.md`.

**Result reads** go through the usual `results.elements.gauss.get(...)`. The
`.ladruno` reader is self-describing (`FAMILY="bernstein"` + `QUADRATURE/GP_PARAM`),
so GP stress/strain (axis-form `sigma_xx`/`eps_xx`/`gamma_xy` tokens) and the GP
**world** coordinates both come straight from the file — `slab.global_coords(fem)`
reconstructs `x = B(ξ)·X` via the neutral `apeGmsh._basis` Bernstein evaluator
(never a catalog GP order). The committed pipeline is straight-sided only (no curved
high-order geometry).

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
  sign-continuity flip, `basicForce` is one station at ξ=0). For **force-based** beams
  this also serves `section.force`/`section.deformation` (`P`→`axial_force`,
  `kappaZ`→`curvature_z`, …) — one station per integration point, its ξ read from the
  element's `GP_PARAM` (not synthesized).
- `results.elements.fibers.get(component="fiber_stress")` — fiber-section stress/strain
  (`fiber_stress`/`fiber_strain`), one row per (element, GP, fiber), with `y`/`z`/`area`/
  `material_tag` from `MODEL/SECTION_ASSIGNMENTS`. (A `.ladruno` has no distinct *layer*
  or *spring* level — layered shells serialise as fiber sections; zeroLength force/material
  state is reachable via the element/gauss reads.)
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
