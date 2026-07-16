# ADR 0077 — Parallel modal analysis (distributed FEAST + serial-gather stopgap)

**Status:** Proposed (2026-07-15; **reworked after adversarial review** —
the original plain-`eigen`-over-`MumpsParallelSOE` "v1" is REFUTED and
struck; see the review banner and the appendix). Brings modal analysis
to apeGmsh's partitioned / HPC path. The only *correct* distributed
modal path in this ecosystem is the fork's FEAST line (fork ADR 43,
COMPLETE); this ADR consumes it runtime-agnostically and adds a correct
serial-gather stopgap for node-sized models. Rides the apeGmsh HPC path
(ADRs 0060 / 0061). Modal-response family (ADR 0075) stays
single-process.

> ⚠ **Adversarial review outcome (2026-07-15).** An originally-proposed
> **v1 — plain `eigen` over `MumpsParallelSOE` in a classic Tcl deck
> under `OpenSeesMP.exe` — was REFUTED** (two adversarial agents + direct
> source check). Under `_PARALLEL_INTERPRETERS` the `eigen` command
> builds an `ArpackSOE` with **no** `setProcessID`/`setChannels` (the
> parallel-eigen wiring is `#ifdef _PARALLEL_PROCESSING`/OpenSeesSP-only,
> `commands.cpp:6030` / `:6046-6054`); ARPACK's generalized-problem
> `M*v` reduction is SP-gated (`ArpackSolver.cpp:249-274`), so the
> shift-invert `(K−σM)` solve is distributed (global K) while `M*v` stays
> **local** — an incoherent operator → per-rank-local / garbage modes.
> This is exactly the SP/MP non-composition the fork built FEAST to
> resolve, and **apeGmsh already fails loud on the same construct**:
> `apesees.py:4285-4293` refuses `ops.damping.modal` under MPI because
> *"a bare eigen solves each rank's LOCAL subdomain under OpenSeesMP, so
> the modes … would be wrong."* Full detail + the harvest defects
> (F2–F4) are in the appendix. The Decision below is the reworked one.

## Context

apeGmsh has a complete **single-process** modal surface (ADR 0025
`eigen`; ADR 0075 `modal_properties` / `eigen_feast` / the
modal-response family) — every driver builds one fresh live domain
(`build()` → `LiveOpsEmitter(wipe=True)` → `bm.emit`) and runs in one
process. None of it touches the partitioned / MPI path. The HPC path
(ADR 0060 remote SLURM, ADR 0061 per-rank deck emission) emits
**partitioned classic-Tcl decks run under `OpenSeesMP.exe`** and submits
them to a cluster — but carries no modal analysis. A user with a model
too large to eigen-solve on one node has no route.

### The central fact: upstream SP and MP parallel-eigen do not compose

- **No distributed Krylov eigensolver exists upstream.** The only driver
  is *serial* ARPACK Lanczos (`ArpackSolver`, `dsaupd_`/`dseupd_`).
- Two *incompatible* parallel wirings, neither a true distributed
  eigensolver: **OpenSeesSP** (`_PARALLEL_PROCESSING`) partitions the
  domain (`PartitionedDomain::eigenAnalysis` → per-subdomain
  `ArpackSOE`) but each subdomain solves *serially*; **OpenSeesMP**
  (`_PARALLEL_INTERPRETERS`) can do a distributed `MumpsParallelSOE`
  linear solve but every rank redundantly re-runs the *whole serial
  Lanczos*, and — the fatal part — the ARPACK `M*v` reduction that would
  globalize the generalized problem is **SP-gated**, so under MP it never
  fires and the operator is global-K / local-M (the refuted v1). Under
  classic `OpenSeesMP.exe` there is **no** `PartitionedDomain` at all:
  each rank holds a plain `Domain` with the element subset apeGmsh's
  `if {[getPID]==K}` blocks give it.
- **Modal post-processing is MPI-blind.** `DomainModalProperties` and
  `ResponseSpectrumAnalysis` do zero MPI reduction — under any
  partitioned run they see one domain only, so participation factors and
  effective modal mass are **wrong**. This is upstream, and FEAST does
  **not** fix it.

### What the fork landed — FEAST is the parallel-modal answer

Fork ADR 43 (COMPLETE, PRs #515→#532). `eigen -feast fmin fmax
[-certify] [-rci]` — band-targeted contour-integration eigensolver
(`FeastEigenSOE` 33022 / `FeastEigenSolver` 33023). The blessed
distributed path is **L3-only**: under OpenSeesMP/PyMP the `dfeast_srci`
outer loop runs replicated on every rank and each contour solve `(zM−K)`
is a distributed `dmumps` SYM=2 block-real factor+solve across all ranks
(solution broadcast for lockstep). It reduces distributed eigen to
independent distributed *linear* solves — the composition upstream
lacks. Band-targeted + self-certifying (`-certify` = Sturm/inertia
negative-pivot count). **L2 (quadrature-parallel) was evaluated and
deliberately NOT built** (Amdahl φ-correction → ~1× real speedup).

### The load-bearing wiring caveat 🔑

`-feast` is parsed **only** by the modern `SRC/interpreter/
OpenSeesCommands.cpp` (`OPS_eigenFeast`, `:2207`) — compiled into
openseespy (`PythonModule.cpp`), the interpreter Tcl main, and **PyMP
(OpenSeesMP-Python, `PythonMPIModule.cpp`)**. It is **NOT** in the legacy
`SRC/tcl/commands.cpp` (`OpenSeesMP.exe` classic Tcl main, `:5993-6025`,
no `-feast` branch) and not in xara. apeGmsh's production HPC path emits
**classic Tcl decks run under `OpenSeesMP.exe`**, which physically cannot
parse `eigen -feast`. Reaching FEAST therefore requires an *unlock* — the
core of the Decision below.

### What already exists to build on

- Deck-line emission for `eigen`, `modalProperties`, `eigen_feast` on all
  five emitters (`emitter/tcl.py:943/957/966`; ADR 0025 / 0075).
- Parallel analysis-chain primitives: `system Mumps` (= `MumpsParallelSOE`
  under MP), `numberer ParallelPlain` (existing partitioned default,
  `apesees.py:5292`) / `ParallelRCM` (`analysis/{system,numberer}.py`).
- Partitioned deck emission (`_emit_partitioned`, monolithic or
  `per_rank=True`) + HPC submit/harvest (ADR 0060 / 0061). Note the
  per-rank **`py()`** path currently *raises* (ADR 0061 §3) — relevant to
  the PyMP unlock.
- The cross-partition result merge (`results/readers/_mpco_multi.py`,
  `_ladruno_multi.py`) — keys on node ids **embedded in the MPCO /
  `.ladruno` HDF5 model group**; it does **not** cover plain headerless
  `.out` recorder files (adversarial finding F3).

## Decision

Two tiers. **Tier 0 ships correct today; Tier 1 is the real distributed
answer, designed runtime-agnostic so it ships on whichever FEAST unlock
lands first.** Plain `eigen` over `MumpsParallelSOE` is **rejected**
(F1) — never emitted.

### Tier 0 — serial-gather stopgap (correct, node-sized, ships now)

`apeSees.eigen(...)` / `modal_properties(...)` already run serial on the
**unpartitioned** build. Tier 0 makes that reachable for a model the user
*authored* with partitions but whose eigensolve fits one node: build the
single-domain (non-partitioned) model and run the existing serial
`eigen` / `modalProperties` — fully correct, including participation
factors and effective mass. **Honest caveat, stated in the API and
docs:** it does **not** scale the eigensolve (whole model assembled on
one rank); it is a convenience for node-sized models, not distributed
modal analysis. No new solver, no MPI.

### Tier 1 — distributed FEAST (`eigen -feast … -rci`), runtime-agnostic

`apeSees.eigen_parallel(band=(f_min, f_max), *, certify=False,
target=...)` emits a **partitioned modal deck** (not a live run),
submittable through the existing HPC path, plus a harvest. The driver is
**runtime-agnostic**: it assembles one logical modal deck and renders it
for whichever unlock is available —

- **(2a) PyMP `.py` deck** — an OpenSeesMP-Python deck run under
  `mpiexec … python` (PyMP parses `-feast`). Requires a parallel `py()`
  emission path (per-rank `py()` raises today, ADR 0061 §3) + a PyMP
  launcher shim in the HPC layer. Zero fork dependency.
- **(2b) classic-Tcl `-feast` parity** — a fork PR wiring `eigen -feast
  … -rci` into `SRC/tcl/commands.cpp` (mirroring the interpreter
  parser); then apeGmsh's *existing* partitioned Tcl deck +
  `OpenSeesMP.exe` HPC path carries FEAST by emitting the one line.
  Smallest apeGmsh change; gated on fork rebuild + cluster redeploy.

Ship whichever lands first; support both. The band form has no a-priori
mode count (the contour *is* the band) — the result surface handles a
dynamic mode count, and `-certify` adds a completeness flag.

### Preamble (Tier 1) — forced, not auto-emitted (finding N2)

`eigen_parallel` emits its own eigen preamble; it does **not** lean on
the general auto-emit:

- `constraints Transformation` — **forced unconditionally** (the
  auto-emit only fires when MP constraints exist, `apesees.py:5070`;
  `Penalty` pollutes M with penalty mass and `Lagrange` injects
  zero-mass DOFs → spurious modes, so eigen always needs
  `Transformation`).
- a parallel numberer (`ParallelPlain`, matching the existing partitioned
  default; `ParallelRCM` optional).
- `system Mumps` — **load-bearing**: with any *serial* `system`, FEAST's
  distributed inner solve degrades to a per-rank local solve → silent
  per-partition garbage. Pinned and asserted.

No `test`/`algorithm`/`integrator`/`analysis` line (the eigensolve needs
none; `eigen` self-fires `domainChanged()`, `DirectIntegrationAnalysis.
cpp:311`, so no prior `analyze`/`domainChange` is required).

### Harvest (Tier 1) — corrected per findings F2 / F3 / F4

- **Eigenvalues** — capture the **single** solve's return once, on every
  rank, then write from rank 0: `set _lam [eigen -feast …]; if {[getPID]
  == 0} { … puts $_lam … }`. Never a second `[eigen …]` (F2: the original
  double call is a redundant distributed solve *and* a rank-0-only
  collective → deadlock).
- **Mode shapes** — routed through an **MPCO / `.ladruno` HDF5 recorder**
  (eigenvector results), so the *existing* node-id cross-partition merge
  (`_mpco_multi.py` / `_ladruno_multi.py`) applies. Plain `recorder Node
  -file …out "eigen" $k` is **rejected** (F3: headerless `.out` carries
  no node ids and no existing merge covers it; F4: it needs an explicit
  `record` trigger to fire at all). P3 verifies the chosen recorder
  actually records eigenvectors per rank and merges by global node id; if
  no HDF5 recorder supports eigenvectors, the fallback is a new
  labeled-`.out` reader (and INV-3 is dropped, not asserted).
- **Modal properties in parallel — DEFERRED, fail-loud.** Upstream
  `modalProperties` is MPI-blind (wrong effective mass under
  partitioning; C7) and FEAST does not change that. Tier 1 does **not**
  emit it; the result surface raises a clear `NotImplementedError` on
  `.participation_factors(...)` / `.mass_ratios`, directing seismic
  mass-participation users to Tier 0 (node-sized) or a future MPI-aware
  reduction (Deferred).

### Result surface

`ParallelModalResult` (frozen dataclass, `analysis/modal.py`) — eager
(no `_live`; the run is remote / already complete). Carries
`eigenvalues` (+ derived ω / f / T), a `certified: bool | None` flag
(from `-certify`), and a `mode_shape(mode) -> np.ndarray` reader over the
harvested, node-id-merged eigenvectors. Bindable to `Results` / `FEMData`
for viewing (same shape the live `DomainCapture.modal` path feeds the
viewer). Loud property-accessor guard per INV-2.

## Rejected alternatives

- **Plain `eigen` over `MumpsParallelSOE` under `OpenSeesMP.exe`
  ("ship-now v1").** REFUTED (F1) — global-K / local-M incoherent
  operator → per-rank-local garbage modes; apeGmsh's own
  `_emit_global_damping_partitioned` already refuses the identical
  construct. This is the path FEAST exists to replace; it is never
  emitted.
- **L2 quadrature-parallel FEAST in apeGmsh.** Not ours to build; the
  fork ruled it out (Amdahl φ → ~1× speedup; L3 owns the 10⁵–10⁶-DOF
  regime). apeGmsh consumes L3-only.
- **OpenSeesSP (`_PARALLEL_PROCESSING`) partitioned eigen.** The fork
  de-scoped SP for FEAST (MP is the single blessed config), and upstream
  SP modal post-processing is broken. Not a target.
- **Emit `modalProperties` in the parallel deck.** MPI-blind upstream —
  would silently produce wrong effective mass. Deferred + raised-on.
- **A live in-process-MPI modal driver.** apeGmsh's parallel story is
  deck-emit + remote submit (ADR 0060/0061); no demand for in-process
  MPI.

## Invariants

- **INV-1** — Tier 1 emits a **deck**, never runs live; reuses
  `_emit_partitioned` and the deck is the HPC entry point (ADR 0060/0061
  submit/transfer unchanged).
- **INV-2** — no `modalProperties` in the parallel deck (Tier 1); the
  parallel result surface raises loudly on properties accessors.
- **INV-3** — Tier-1 mode-shape harvest uses a recorder format the
  **existing** node-id cross-partition merge covers (MPCO / `.ladruno`
  HDF5); plain `.out` is not used unless a new labeled reader is written
  (then this invariant is restated for that reader). *Verified at P3, not
  assumed.*
- **INV-4** — the Tier-1 eigen preamble is **forced** `constraints
  Transformation` → parallel numberer → `system Mumps`, with no
  test/algorithm/integrator/analysis line; `system Mumps` is asserted
  present (silent-garbage guard).
- **INV-5** — eigenvalues are captured from a **single** `eigen -feast`
  return and written once from rank 0 (no double solve, no rank-0-only
  collective).
- **INV-6** — Tier 0 is bit-for-bit the existing serial `eigen` /
  `modalProperties` on the unpartitioned build (correctness by
  reduction to an already-tested path).
- **INV-7** — the Tier-1 driver is runtime-agnostic: the same logical
  modal deck renders to a PyMP `.py` deck (2a) or a classic-Tcl deck
  (2b); only the emitted solver-invocation surface differs.

## Phased plan

**Tier 0 (ships first, no dependency):**
- **P0 — serial-gather stopgap. ✅ DONE (2026-07-15).** No new solver:
  the live emitter's `supports_partitions = False` (`emitter/live.py:313`)
  already makes `eigen` / `modal_properties` build the full gathered
  model in one process on a partition-authored model. Added the "does
  not scale the eigensolve" caveat to both docstrings
  (`apesees.py`) and a live regression test pinning
  partitioned-serial == unpartitioned (bit-identical eigenvalues) +
  `modal_properties` available with participation
  (`tests/opensees/live/test_eigen_partitioned_serial_gather.py`, 2
  tests green under the worktree src). Satisfies INV-6.

**Tier 1 (distributed FEAST):**
- **P1 — modal deck skeleton (tcl target). ✅ DONE (2026-07-16).**
  `apeSees.modal_deck(path, *, band, certify=False, target="tcl",
  per_rank=False, out=)` builds the partitioned model (the partitioned
  emit already lays down the `numberer ParallelPlain` / `system Mumps`
  preamble, INV-4) and appends a single captured
  `set _lam [eigen -feast fmin fmax -rci [-certify]]` + rank-0 eigenvalue
  write-out (INV-5) via a new `TclEmitter.eigen_feast_parallel`. `target=
  "pymp"` (unlock 2a) + non-partitioned + staged all fail loud; `per_rank`
  honored. Verified by deck-text tests
  (`tests/opensees/integration/test_modal_deck_parallel_feast.py`, 6
  green): captured solve is emitted exactly once (no double/deadlock),
  preamble present, `modalProperties` absent (INV-2). The `target` seam
  (INV-7) is stubbed for the PyMP rendering; the live run needs the fork
  `-feast` classic-Tcl parity build (unlock 2b).
- **P2 — the two unlock backends.** (2a) parallel `py()` emission +
  PyMP launcher shim; (2b) consume fork classic-Tcl `-feast` parity once
  it lands. Verify per backend: `-feast` reaches the solver at
  `mpiexec -n 2/4`.
- **P3 — harvest.** Choose + verify the eigenvector recorder format
  (MPCO / `.ladruno`) merges by global node id (INV-3); wire the rank-0
  eigenvalue file. Verify: `mpiexec -n 2/4` mode shapes vs a
  single-process FEAST oracle (MAC ≥ 0.999); merged Φ has every global
  node once (no boundary double-count).
- **P4 — `ParallelModalResult` + surface.** Eager dataclass, `certified`
  flag, `mode_shape` reader, loud property-accessor guard (INV-2), viewer
  binding. Verify: viewer renders a harvested parallel mode; property
  accessor raises with the documented redirect.
- **P5 — HPC e2e + docs.** Full emit → `run_remote` → harvest on the
  cluster (mid-size model); skill/CHANGELOG. Verify: distributed spectrum
  == single-process FEAST oracle; `-certify` completeness reported.

## Cross-references

- ADR 0025 — `Emitter.eigen` widening; ADR 0075 — modal-response family +
  `eigen_feast` (single-process siblings; the classic-Tcl `-feast`
  caveat).
- ADR 0060 / 0061 — remote HPC submission + per-rank deck emission (the
  substrate; note per-rank `py()` raises, relevant to unlock 2a).
- ADR 0027 — cross-partition result merge (the node-id eigenvector
  merge; finding F3 bounds its applicability).
- Fork ADR 43 (`43_ladruno_feast_eigensolver_adr.md`) +
  `modal_gap_study/00_SYNTHESIS.md` §3 (the SP/MP non-composition FEAST
  resolves) + `feast_l2_profile/README.md` (L2 "don't build").

## Deferred

- **Parallel modal properties** (participation, effective mass) — needs
  an MPI-aware `modalProperties` (upstream/fork fix) or client-side
  computation from harvested Φ + a mass export. Not in Tier 0/1.
- **Parallel modal-response family** (ADR 0075) — stays single-process.
- **Per-stage parallel modal** — inherits the ADR 0075 / SSI-2.A staged
  deferral.

## Appendix — adversarial review findings (2026-07-15)

Two adversarial agents (fork-source verification + design refutation)
plus a direct source check. The fork-facts pass confirmed C1–C9 (parse
gap, FEAST L3-only, L2 not built, MP-only, `modalProperties` MPI-blind,
Node `eigen` recorder exists, eigenvalues replicated per rank). The
design pass found the fatal flaws that reshaped this ADR:

- **F1 (fatal, decisive).** Plain `eigen` over `MumpsParallelSOE` under
  `OpenSeesMP.exe` → per-rank-local garbage modes (global-K / local-M;
  `M*v` reduction SP-gated, `ArpackSolver.cpp:249-274`; `ArpackSOE` built
  without `setProcessID`/`setChannels`, `commands.cpp:6030`). Corroborated
  by apeGmsh's own guard `apesees.py:4285-4293`. → Plain-eigen v1
  rejected; FEAST elevated to the first real slice.
- **F2.** Eigenvalue write-out re-invoked `eigen` (redundant distributed
  solve + rank-0-only collective → deadlock). → INV-5 (single capture).
- **F3.** The ADR 0027 merge covers only MPCO / `.ladruno` HDF5 node-id
  groups, not plain `.out` (`_mpco_multi.py:20`; `_recorder.py:389`;
  `Results.py:390`). → INV-3 (HDF5 recorder or a new labeled reader,
  verified at P3).
- **F4.** Node recorders never fire without a `record`/`analyze` trigger
  (`NodeRecorder::record` ← `Domain::record`). → folded into the P3
  harvest design (HDF5-recorder route sidesteps the bare-`.out` trigger
  gap).
- **N2.** Existing auto-emit is `numberer ParallelPlain` (not
  `ParallelRCM`) and `constraints Transformation` is conditional. →
  INV-4 (forced preamble).
- **C7 (validated).** `modalProperties` / RSA MPI-blind → the fail-loud
  deferral (INV-2) was already right, and stays under FEAST.
