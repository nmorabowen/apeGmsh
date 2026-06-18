# OpenSees fork (Ladruno) integration

apeGmsh can target the **Ladruno fork** of OpenSees (`nmorabowen/OpenSees`,
branch `ladruno`) in addition to stock `openseespy`. The fork adds features
apeGmsh emits/reads that stock OpenSees does **not** have. Stock `openseespy`
stays first-class — the fork is **opt-in**; gate fork-only features at the
point of use, never force the fork.

## Targeting a build + gating fork features

*Which* OpenSees runs (`OpenSeesTarget`) and the precedence/inert-live
rules are in `opensees-bridge.md` → "Which OpenSees runs" — not repeated
here. The fork-specific idiom: **never infer capability from the path**;
branch on the **live** probe, and use `require_fork=True` to fail loud at
the live boundary instead of three primitives deep.

```python
ops = apeSees(fem, opensees=OpenSeesTarget(require_fork=True))   # live-path assertion
if ops.capabilities().has_fork:      # OpenSeesCapabilities(has_fork=, has_profiler=, version=)
    ops.element.BezierTet10(pg="Body", material=m)
else:
    ops.element.FourNodeTetrahedron(pg="Body", material=m)
```

`has_fork` tracks the fork-only `profiler` command (present in the fork's
openseespy module) — the same gate the live emitter uses.

## Fork-only features apeGmsh touches

| Feature | Kind | Notes |
|---|---|---|
| **BezierTri6 / BezierTet10** | elements | fork-only Bézier (Bernstein) continuum elements — typed primitives `ops.element.BezierTri6/BezierTet10` (Tet10 also takes `-geom linear/corot/finite` + `-fbar`) |
| **LadrunoBrick** | element | fork-only unified 8-node hex (tag 33002) — typed `ops.element.LadrunoBrick` with `-formulation`/`-geom`/`-hourglass`/`-damp` |
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

## LadrunoBrick (unified 8-node hex)

Fork-only 8-node hexahedron (class tag **33002**, Gmsh hex8 / etype 5) that folds
the anti-locking treatment into one `-formulation` selector and an orthogonal
`-geom` kinematics selector — one class reproduces upstream `stdBrick`/`bbarBrick`/
`SSPbrick` where they overlap and adds the cheap explicit hex:

```python
ops.element.LadrunoBrick(pg=…, material=m,
                         formulation="std",      # std|bbar|uri|ssp|eas
                         geom="linear",          # linear|corot|finite
                         hourglass=None,         # uri only: viscous|stiffness|physical
                         hourglass_coeff=None,
                         lumped=False, body_force=None,
                         damp=None)              # element-flag -damp (ADR 0053)
```

```
element LadrunoBrick $tag $n1..$n8 $matTag [-formulation std|bbar|uri|ssp|eas]
    [-geom linear|corot|finite] [-hourglass viscous|stiffness|physical [$coeff]]
    [-lumped] [-b $bx $by $bz] [-damp $dampTag]
```

Key points (apeGmsh fails loud at construction, mirroring the fork's parse guards):

- **`formulation`:** `std` (full integration, default), `bbar` (mean-dilatation),
  `uri` (1-pt reduced + hourglass control), `ssp` (stabilized single-point), `eas`
  (true Simo–Rifai enhanced assumed strain).
- **`geom` `corot`/`finite` ship `std`/`bbar` only** — `uri`/`ssp`/`eas` under
  corot/finite raise (deferred in the fork). `finite` needs a finite-strain
  material; `finite` + `bbar` = F-bar (unsymmetric tangent → `FullGeneral`/`UmfPack`).
- **`hourglass` is `uri`-only** (raises with any other formulation); the optional
  `hourglass_coeff` requires `hourglass` to be set. `viscous` is explicit-only.
- **`lumped`** emits `-lumped` (diagonal mass) — required for explicit integrators.
- **`damp`** attaches a `Damping` object via the element's own `-damp` flag
  (ADR 0053 element-flag attach) — honoured **only** with `std`/`bbar`; apeGmsh
  raises for the other formulations rather than letting the fork silently drop it.
  Defaults (`std`/`linear`) are elided so decks stay byte-clean.
- **Result reads** go through the usual `results.elements.gauss.get(...)`; the
  recorder always returns 8-GP `Vector(48)` (single-point forms mirror slot 0).

## Bézier elements (`BezierTri6` / `BezierTet10`)

Two fork-only Bézier (Bernstein) continuum elements (Kadapa 2018), exposed as
typed primitives:

```python
ops.element.BezierTri6(pg=…, thickness=…, material=m, plane_type="PlaneStrain",
                       bbar=False, consistent_mass=False,
                       pressure=None, rho=None, body_force=None)   # 2D, 6 nodes
ops.element.BezierTet10(pg=…, material=m, bbar=False, consistent_mass=False,
                        rho=None, body_force=None, pressure=None,
                        geom="linear", fbar="centroid")            # 3D, 10 nodes
```

Emit grammar is **flag-prefixed** (each option independently optional), unlike
`SixNodeTri`'s positional tail:

```
element BezierTri6  $tag $n1..$n6  $thick $type $matTag [-bbar] [-cMass] [-pressure $p] [-rho $r] [-bodyForce $b1 $b2]
element BezierTet10 $tag $n1..$n10 $matTag [-bbar] [-cMass] [-rho $r] [-bodyForce $b1 $b2 $b3] [-pressure $p] [-geom linear|corot|finite] [-fbar centroid|mean_dilatation]
```

Key points:

- **`geom` (Tet10): `linear` (default) / `corot` / `finite`.** `corot` = large
  rotation / small strain (EICR); `finite` = large strain (updated-Lagrangian),
  which needs a finite-strain material (`setTrialF(F)`, e.g. `nDMaterial LogStrain`)
  — the fork rejects a small-strain material there at run time. `finite` + `bbar` =
  **F-bar** (volumetric-locking cure; unsymmetric tangent → use `FullGeneral`/
  `UmfPack`). `pressure` is **rejected** under `corot`/`finite`. Defaults elide the
  flag so existing decks stay byte-identical.
- **`fbar` (Tet10): `centroid` (default) / `mean_dilatation`** — only meaningful with
  `bbar=True` + `geom="finite"`; apeGmsh raises if set otherwise.

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
- `results.elements.get(component="localForce")` — **the fork-only escape hatch**,
  token-driven: the component is the file's `ON_ELEMENTS/<token>` key
  (`basicForce`/`localForce`/`force`/`globalForce`) and the slab is the raw
  `(T, E, NUM_COLUMNS)` block in the file's column order. Prefer the neutral
  sub-composites above (`gauss`/`line_stations`/`fibers`, cross-backend); drop to
  this only for raw fork tokens the neutral views don't expose.

Multi-partition runs (`<stem>.part-N.ladruno`) auto-discover siblings and merge
(node-union + element-concat), like `from_mpco`. Higher-order / Bézier elements are
self-describing: GP world coords are reconstructed from the file's `BASIS` +
`GP_PARAM` via the neutral `apeGmsh._basis` evaluator (shared with the Bézier read
path), since a `.ladruno` from a Bézier element carries no `GLOBAL_GP_COORDS`.

## Live monitor (`ops.recorder.Monitor` + `read_monitor` / `tail_monitor`)

The **Monitor** is a *lightweight live-telemetry sidecar* — distinct from the
canonical `.ladruno` recorder. It streams a few selected nodal scalars to a small
SWMR-HDF5 file (`FORMAT="ladruno-monitor"`: `COLUMNS`/`STEP`/`TIME`/`FRAMES`) that a
viewer process can **tail while the analysis is still running**; the same file is a
valid at-rest result once the run ends. Fork-only — emit on any build, the fork is
needed only to *run*.

Emit:

```python
ops.recorder.Monitor(sink="live.h5", nodes=(roof,), dofs=(1, 2), resp="disp",
                     every=5)         # or pg="roof_nodes"; resp ∈ disp|vel|accel|reaction
```

Channels are nodes × dofs, labelled `node<N>.<resp>.dof<D>` in **node-major** order;
`every=K` (step decimation) and `hz=H` (wall-clock throttle) bound the stream.

Read — **not** a `Results` object (it carries no FEM), a thin time-history instead:

```python
from apeGmsh.results import read_monitor, tail_monitor
m = read_monitor("live.h5")          # at-rest snapshot
m.to_dataframe(index="time")         # DataFrame, one column per channel label
m.channel("node5.disp.dof1")         # one [T] history

for step, t, row in tail_monitor("live.h5", timeout=2.0):   # follow a growing sink
    ...                              # row is [nCols] in m.columns order
```

For a reader in a *separate process* from the solver, set
`HDF5_USE_FILE_LOCKING=FALSE` before `h5py` is imported (the SWMR/libhdf5 quirk).

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
