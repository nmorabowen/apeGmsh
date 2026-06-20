# DRM load-pattern study — scope & plan

Goal: bring jaabell's latest DRM work into the ladruno fork **with confidence**,
by reproducing/validating the `H5DRMLoadPattern` against a controlled ShakerMaker
dataset, then deciding the open judgment-calls from evidence (not by guessing).

## 1. The C++ delta (what "the latest DRM load" actually is)

Branch: **`jaabell/merged-Explicit-ASDP-master`** (there is no `explicit-merge`).
The whole DRM directory is identical to ours **except one file**,
`SRC/domain/pattern/drm/H5DRMLoadPattern.cpp`. Our fork already partially
integrated this stream and even carries an extra fix jaabell lacks.

Commits jaabell has, we don't:
- `9be15c3d0` "Includes CFACTOR" — 4 things: (1) cFactor on DRM_F **[we already
  took this, commit 8a4393995]**; (2) end-of-window guard restructure + `i2`
  clamp; (3) **Z-flip removal** (`d/a[2] = -[2]` commented out); (4) **>6-DOF
  node-skip removal**.
- `99b7c2c11` "hold final displacement past dataset end, zero acceleration" —
  **the real new feature**; depends structurally on (2).

We have, jaabell doesn't:
- `3c7941371` "solving memory leak in h5drm" (amnp95) — `cleanup()` closes for
  dis/acc datasets+dataspaces, removes per-timestep `H5Dget_space` leaks, adds
  `H5Pclose(ih5_xfer_plist)` in the read helpers.

⚠️ A wholesale file copy from jaabell **regresses the memory-leak fix** (same
functions). Integration must be selective.

Open judgment-calls (to settle empirically with the test below):
- **Z-flip** — ShakerMaker is z=Down(+down); our FEM model frame choice decides
  whether the `d[2]=-d[2]` flip is needed. Study isolates this.
- **>6-DOF skip** — protects PML split-field nodes (3D PML ndf=9). Keep unless
  evidence says otherwise.

## 2. The fixture (DONE)

`drm_small.h5drm` — ShakerMaker DRMBox, SCEC_LOH_1 crust, 1 Gaussian point
source. Regular **5×5×4 grid** (98 stations: 25 internal / 73 boundary) at **50 m**
spacing, `X∈[5.9,6.1] Y∈[7.9,8.1] Z∈[0,0.15]` km (Z=depth, z-down). dt=0.025,
nt=1200 (30 s). `drm_qa_direct.npz` = free-field oracle at box center.
Generator: `gen_drm_dataset.py` (see memory `project_shakermaker_venv_drm`).
Two silent-zero bugs found+fixed (venv DLL PATH; `add_to_response` negative-t0).

## 3. apeGmsh side — what exists vs what we must create

Exists: `parts/drm_box.py` (`DRMBox` Part — SSI inner/transition/outer absorbing
layout), `parts/plane_wave_box.py`, ADR 0054 ASD absorbing boundary, full
structured/transfinite meshing (`g.mesh.structured.set_transfinite_box`).

Gaps (the "needed apeGmsh methods"):
1. **A DRM-grid mesh matched to a `.h5drm`** — a structured hex mesh whose nodes
   coincide with the dataset's station `xyz` (so H5DRM node-matching is exact),
   with interior/boundary tagging from the `internal` flag. The existing
   `DRMBox` is the wrong shape (absorbing layout, not the FK station grid).
   Proposal: a small helper that reads a `.h5drm`, derives the grid
   (origin/spacing/counts), and emits the matching transfinite box + PGs.
2. **An H5DRM load-pattern emitter** — `ops.pattern.H5DRM(filename, factor=,
   crd_scale=, distance_tolerance=, transform=...)` on the apeSees bridge
   (currently only Plain/UniformExcitation). For the **baseline study** we can
   defer the typed emitter and hand-append the `pattern H5DRM ...` line to the
   apeGmsh-emitted deck.

## 4. Plan (baseline-first, NO C++ rebuild yet — user decision)

P0 (now): apeGmsh script → structured hex soil box on the `.h5drm` grid (km→m via
  crd_scale, model built in ShakerMaker's z-down frame so transform=identity →
  isolates the Z-flip question). Elastic soil = SCEC_LOH_1 surface layer
  (Vp=4000, Vs=2000, rho=2600 → E≈2.77e10, nu≈0.333). stdBrick/SSPbrick (3-DOF,
  8-node — H5DRM requires 3-DOF ≤8-node). Fix base.
P1: driver adds the H5DRM pattern + transient (Newmark), runs the **current**
  binary, records interior-node motion.

  **H5DRM invocation (KEY — Tcl vs openseespy differ):**
  - Tcl `pattern H5DRM $tag $file $factor` exposes ONLY 3 args (crd_scale=1,
    identity, tol=1e-3) — won't rescale km→m. Use **openseespy** instead.
  - openseespy `pattern('H5DRM', tag, file, factor, crd_scale, dist_tol,
    do_transform, T00..T22, x00..x02)` (parser `OPS_H5DRMLoadPattern`).
  - C++ transform: `xyz_model = T·((xyz_station − drmbox_x0)·crd_scale) + x0`,
    `drmbox_x0` = box CENTER from file ([6,8,0] km). Model built centered at
    lateral origin ⇒ `crd_scale=1000, do_transform=1, T=identity, x0=0` matches
    exactly. distance_tolerance compared in model space (m).
  - Needs 3-DOF ≤8-node elements (stdBrick ✓). Run under the python312 env that
    loads `dist/bin/opensees.pyd` (memory `project_opensees_test_env`).
P2: compare interior motion vs `drm_qa_direct.npz` free-field oracle →
  characterize baseline: does it match? is Z flipped? what happens past tend?
P3: from evidence, decide Z-flip / DOF-skip / hold-past-end; THEN do the selective
  C++ integration + rebuild loop (preserving the memory-leak fix).
P4: promote the helpers into apeGmsh (DRM-grid-from-h5drm + typed H5DRM pattern).

## P1/P2 baseline ATTEMPT — findings (2026-06-19)

Ran the apeGmsh box + H5DRM against the **shipped** binary (`dist/bin`, commit
8c8edb21, 98 commits behind worktree HEAD). Two blocking findings:

1. **openseespy does NOT expose H5DRM.** `SRC/interpreter/OpenSeesPatternCommands.cpp`
   registers only Plain + UniformExcitation → `ops.pattern('H5DRM',...)` =
   "unknown pattern type". H5DRM is reachable ONLY from the Tcl interpreter
   (`runtime/commands/.../pattern.cpp`, 3-arg: tag file factor). ⇒ a fork
   follow-up: wire H5DRM (with crd_scale/transform) into the py interpreter.
2. **Tcl H5DRM path mis-initializes the transform matrix `T`.** Run log:
   `drmbox_x0 = (6000 8000 0)` read OK, `crd_scale = 1` OK, but
   `T = (2.46e-313 ...)` GARBAGE (and user `x0` garbage too — i.e. the trailing
   defaulted ctor args). Result: transformed station coords are garbage →
   **1 of 98 nodes matched, 0 DRM elements** → DRM applies nothing. Reproducible.
   Root cause likely a default-argument / build-staleness mismatch on the 3-arg
   `new H5DRM(tag,file,factor)` call (args 7–18 not filled). The shipped binary
   is 98 commits stale.

**Conclusion:** the shipped binary cannot run our H5DRM dataset. A correct
baseline REQUIRES a fresh build of the current source (so "baseline-first, no
rebuild" is invalidated by evidence — the rebuild is on the critical path). The
build naturally folds into the integration phase: build current source first to
get the true baseline, then layer in the selective jaabell changes + the
openseespy-exposure fix, rebuilding as we go.

### ROOT CAUSE of the garbage-T (resolved 2026-06-19)

The H5DRM source is **byte-identical** between the shipped binary's commit
(8c8edb21) and worktree HEAD, so it's NOT staleness — it's a real parser bug.
There are TWO Tcl pattern dispatchers:
- `SRC/runtime/commands/.../pattern.cpp` (the one the exe uses) had a stunted
  **3-arg** `new H5DRM(tag,filename,factor)` relying on ctor defaults that
  arrived as uninitialized garbage for T/x0.
- `SRC/domain/pattern/TclPatternCommand.cpp` (legacy) has the full parser but
  ALSO leaves `double stuff[12]` uninitialized when <12 transform args given
  (same latent bug).
- `OPS_H5DRMLoadPattern()` (openseespy parser, H5DRMLoadPattern.cpp:86) exists
  but is NOT registered in `SRC/interpreter/OpenSeesPatternCommands.cpp`.
jaabell's latest (`merged-Explicit-ASDP-master`) does NOT fix any of this.

**FIX applied (loop iter 1):** rewrote the runtime `pattern.cpp` H5DRM branch as
a full-arg parser with **T defaulting to identity, x0 to zero**, calling the real
`H5DRMLoadPattern` ctor explicitly. Now `pattern H5DRM $tag $file $factor
$crd_scale` works: with crd_scale=1000 + identity transform, the km dataset maps
`(xyz_km − drmbox_x0)·1000` = our centered-meters model → exact node match.
Driver `drm_run.tcl` updated to `pattern H5DRM 1 drm_small.h5drm 1.0 1000.0 1.0 1`.
Ledgered in LEDGER_vanilla_files.md. Building OpenSees target → re-test baseline.
(Deferred: wiring `OPS_H5DRMLoadPattern` into openseespy — P4.)

### Loop iter 2 — REAL garbage-T root cause + DRM now works (2026-06-19)

The runtime-pattern.cpp fix above was for the WRONG dispatcher. The exe's classic
Tcl shell dispatches `pattern H5DRM` through the **legacy
`SRC/domain/pattern/TclPatternCommand.cpp`**, whose `double stuff[12];` (the 3×3
T + x0 transform buffer) was **uninitialized** and only filled when all 12
trailing args are passed. With `argc<19` it kept stack garbage → garbage T →
1/98 nodes. FIX (agent, marked `// Ladruno`): `double stuff[12] = {1,0,0,
0,1,0, 0,0,1, 0,0,0};` (identity). Both pattern parsers now hardened + ledgered.
After rebuild: **T=identity, 98/98 nodes connected, 40 DRM boundary elements,
transient runs to t=30** — DRM forces are applied. The garbage was in the
*arguments*, never the constructor/getCopy/recvSelf.

### Loop iter 2 — test-MODEL instability (next sub-problem)

The free box DIVERGES: interior disp blows up to 28/227/447 m (N/E/Dn) by t=30,
amplitude ratios 38×/714×/94× vs prescribed, **divergence onset t=1.575s — BEFORE
wave arrival (~3-5s)**. So it's unconstrained rigid-body modes, NOT field
tracking. The H5DRM CODE is validated; the minimal box needs an exterior +
**absorbing boundary (Lysmer)** so the effective forces don't excite rigid modes.
Z-flip unreadable through the drift (corr uz/Dn = -0.27, inconclusive). NEXT:
stabilize the model (absorbing boundaries) → re-measure Z-flip + past-tend, then
jaabell's hold-past-end. apeGmsh has ADR 0054 ASD absorbing boundary + Lysmer
support; memory `project_absorbing_pml_guide` / `lysmer_asd_absorbing_boundaries`.

### Loop iter 3 — DRM VALIDATED + Z-flip answered + hold-past-end integrated (2026-06-19) ✅

**Stabilized model (agent):** inner DRM region (the .h5drm grid) + 2 exterior
buffer layers (50 m) on 4 sides + bottom (NOT top free surface) + **FIXED far
boundary** on the outermost faces (non-dataset nodes, so H5DRM's all-nodes-DRM
rule at L580 excludes them). 486 nodes, 320 stdBricks. Bare LysmerTriangle did
NOT stabilize (pure dashpots can't restrain the rigid null-space the DRM residual
excites; needs a pin, and even then tracking is poor) — **fixed far boundary is
the clean winner**. Scripts: `build_drm_buffered.py`, `drm_run_buffered.tcl`,
`bc_fixed.tcl`, `post_drm_buffered.py`.

**Interior reproduces the free-field** (post-arrival 2-25 s, fixed BC):
N ratio 1.001 corr +0.946 · E ratio 0.987 corr +0.911 · Dn ratio 0.998 corr
+0.980. Overall peak ratio 1.03, NO divergence. (Robust over buffer thickness.)

**Z-flip ANSWERED:** with the `d[2]/a[2]` flip REMOVED, uz tracks prescribed Dn
(corr +0.98, ratio ~1.0). ⇒ **removing the flip is CORRECT** for a model built in
ShakerMaker's native z-down frame with identity transform. jaabell's removal is
right; integrated.

**Hold-past-end INTEGRATED + verified:** ran the stabilized model to t=35 (5 s
past tend=30) — bounded (~0.05 m near tend, small ring-down after), `nfail=0`,
NO "Failed to read", NO `exit(-1)`. Memory-leak fix preserved (reuses member
`ih5_dis_ds`); `>6 DOF` skip kept.

**C++ changes (ledgered):** TclPatternCommand.cpp (stuff[12]→identity),
H5DRMLoadPattern.cpp (Z-flip removed + hold-final-displacement + i2 clamp),
runtime pattern.cpp (full-arg parser, defensive). All `// Ladruno`-marked.

### Loop iter 4 — P4 openseespy H5DRM exposure DONE (2026-06-19) ✅

Registered H5DRM in the openseespy `OPS_Pattern` dispatch
(`SRC/interpreter/OpenSeesPatternCommands.cpp`: fwd-decl + `else if H5DRM/h5drm`
branch). **GOTCHA (CMake):** `add_compile_definitions(_H5DRM)` at CMakeLists
line ~1209 is directory-scoped and only reaches targets created AFTER it, but
`OPS_INTERPRETER` is created at line ~594 → the interpreter compiled WITHOUT
`_H5DRM` → the new branch was `#ifdef`'d out (still "unknown pattern type" after
the first OpenSeesPy rebuild). FIX: `target_compile_definitions(OPS_INTERPRETER
PRIVATE _H5DRM _HDF5)` in the HDF5≥1.12 block. Also a ninja quirk: after the
CMake edit, `cmake --build` regenerated build.ninja but did NOT recompile the
interpreter until the source mtime was bumped (touch) — and a stray python.exe
held the pyd, blocking the copy (killed the pythoncore-3.12 procs).
VERIFIED: `ops.pattern('H5DRM', tag, file, factor, 1000.0, 1.0, 1, T..., x0...)`
→ banner prints, crd_scale=1000 parsed, **98/98 nodes**, 40 DRM elements.
NB: run_drm_baseline.py's `DIST` must point at the WORKTREE dist/bin (the fresh
pyd), not the main-repo dist/bin (stale shipped binary).

**ALL LOOP GOALS COMPLETE.** Remaining (future, not in this loop): promote the
apeGmsh helpers (DRM-grid-from-h5drm builder + buffered-box + typed
H5DRM/absorbing emit). These are vanilla bug-fixes/feature-integrations, not a
new class → no banner row.

### Loop iter 5 — hardening: regression test (2026-06-19) ✅

Adversarial edge-case re-read of the hold-past-end + i2 clamp: sound (the t>=tend
branch and the last-interval clamp are continuous; i1/i2 valid for any realistic
N_timesteps). Added `tests/test_h5drm_drm_loadpattern.py` — SELF-CONTAINED
(synthesizes a tiny 3x3x3 .h5drm with h5py, no ShakerMaker), guards the 3 core
fixes: (1) openseespy H5DRM registered + crd_scale (no "unknown pattern type"),
(2) node-matching drives the model (27/27 matched, a free node moves — guards the
garbage-T fix), (3) past-tend analyze() returns 0 + bounded (hold-past-end). PASSES
(`pytest ... 1 passed`). Run under the python312 worktree-pyd env.
