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

> **P2 live correction (2026-07-16), supersedes the earlier partitioned
> framing.** The fork's L3 FEAST requires a **REPLICATED** model: every
> rank assembles the FULL (K, M) CSR and the RCI kernel slices the 2n
> block system's triplets across ranks for the distributed dmumps —
> distribution lives *inside the kernel*, not in domain decomposition. A
> partitioned `if {[getPID]==K}` deck fails
> `FeastEigenSOE::setSize — vertex not in graph` (observed live at
> `mpiexec -n 2`). Two consequences: the modal deck is emitted **flat**
> (partitions on the fem are ignored — the same `supports_partitions =
> False` seam the live emitter uses), and the deck's `system` line is
> **not part of the FEAST solve path** (a serial `UmfPack` deck produced
> genuinely distributed solves — kernel debug proof on both ranks). RAM
> trade-off: full model per rank — the documented L3 regime (~1e5–1e6
> DOF).

`apeSees.modal_deck(path, *, band=(f_min, f_max), certify=False,
target="tcl", out=)` emits a **replicated modal deck** (not a live run),
submittable through the existing HPC path. The driver is
**runtime-agnostic** (`target` seam):

- **(2a) PyMP `.py` deck** — an OpenSeesMP-Python deck run under
  `mpiexec … python` (PyMP parses `-feast`). Needs a PyMP launcher shim;
  the replicated finding removes the per-rank-`py()` blocker the earlier
  draft assumed. Zero fork dependency. Not implemented (raises).
- **(2b) classic-Tcl `-feast` parity** — **SHIPPED + VERIFIED**: fork PR
  #578 wires `eigen -feast … -rci` into `SRC/tcl/commands.cpp`
  (+ adversarial hardening: per-call SOE mutate-in-place, FEAST-failure
  `TCL_ERROR` gate, empty-band NUL fix); built `OpenSees.exe` /
  `OpenSeesMP.exe` and validated end-to-end (see P2). Deployment to the
  cluster pending.

The band form has no a-priori mode count (the contour *is* the band) —
the result surface handles a dynamic mode count, and `-certify` adds a
completeness flag.

### Preamble (Tier 1) — deterministic, every rank identical

`modal_deck` emits its own eigen preamble after the flat model:

- `constraints Transformation` — **forced unconditionally** (the
  auto-emit only fires when MP constraints exist, `apesees.py:5070`;
  `Penalty` pollutes M with penalty mass and `Lagrange` injects
  zero-mass DOFs → spurious modes).
- `numberer RCM` — every rank must number identically (replicated model;
  the parallel numberers are for partitioned domains and are not used).
- `system UmfPack` — **corrected by P2**: the earlier "`system Mumps` is
  load-bearing" claim was carried over from the refuted plain-eigen
  design and is **wrong for FEAST** — the RCI kernel owns its own
  distributed dmumps; the deck's `system` never enters the FEAST solve.
  A serial system keeps the deck runnable under plain `OpenSees` too.

No `test`/`algorithm`/`integrator`/`analysis` line (the eigensolve needs
none; `eigen` self-fires `domainChanged()`, `DirectIntegrationAnalysis.
cpp:311`, so no prior `analyze`/`domainChange` is required).

### Harvest (Tier 1)

- **Eigenvalues** — capture the **single** solve's return once, on every
  rank, then write from rank 0: `set _lam [eigen -feast …]; if {[getPID]
  == 0} { … puts $_lam … }`. Never a second `[eigen …]` (F2: the original
  double call is a redundant distributed solve *and* a rank-0-only
  collective → deadlock). The `getPID` shim is emitted with the solve so
  the same deck runs single-process.
- **Mode shapes — SIMPLIFIED by the replicated finding; implemented at
  P3.** Every rank holds ALL nodes, so mode shapes need only an
  **ordinary rank-0 recorder** — the whole cross-partition-merge concern
  (old findings F3/F4: MPCO node-id merge vs headerless `.out`, `record`
  trigger) applies to a partitioned field that no longer exists. The
  P3 emit (rank-0-guarded block AFTER the captured solve): write a
  `mode_shapes.json` sidecar (sorted mesh node tags in recorder column
  order + the envelope dof count — the headerless `.out` gains its
  node→column map without the deck), then create one `recorder Node
  -file mode_shape_<k>.out -node <sorted tags> -dof 1..ndf "eigen k"`
  per **found** mode (`llength $_lam` — the band count is dynamic, and
  recording an unfound mode corrupts the row: `NodeRecorder::record`
  skips a node whose eigenvector matrix lacks the column WITHOUT
  advancing its write cursor), fire them with a single `record`, close
  via `remove recorders`. Post-solve creation is sound (source-checked:
  the eigen dataFlag reads `Node::getEigenvectors()` only at record
  time; `Domain::addRecorder` does not auto-fire) and required by the
  dynamic count. DOFs a node does not carry are recorded as `0.0`
  (cursor-safe padding, verified in `NodeRecorder.cpp:782-789`).
- **Modal properties in parallel — DEFERRED, fail-loud.** Upstream
  `modalProperties` is MPI-blind (C7) and FEAST does not change that.
  Tier 1 does **not** emit it; the result surface raises a clear
  `NotImplementedError` on `.participation_factors(...)` /
  `.mass_ratios`, directing seismic mass-participation users to Tier 0
  or a future MPI-aware reduction (Deferred).

### Result surface

`ParallelModalResult` (frozen dataclass, `analysis/modal.py`) — eager
(no `_live`; the run is remote / already complete). Carries
`eigenvalues` (+ derived ω / f / T + `n_modes`), a `certified: bool |
None` flag (from `-certify`), and `from_job(job_dir, out=)` harvesting
the rank-0 write-out plus — when the run dir carries the P3 sidecar —
the full-field mode shapes: `mode_shape(node, mode)` (length-`ndf`,
matching the `EigenResult` convention), `mode_shape_field(mode)`
(`(n_nodes, ndf)`), and `shape_nodes` (tags in row order). A pre-P3 run
dir still harvests eigenvalues; the shape accessors fail loud. Loud
property-accessor guard per INV-2.

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

- **INV-1** — Tier 1 emits a **deck**, never runs live; the deck is the
  HPC entry point (ADR 0060 submit/transfer unchanged). *(P2 correction:
  the deck uses the flat emit, not `_emit_partitioned`.)*
- **INV-2** — no `modalProperties` in the parallel deck (Tier 1); the
  parallel result surface raises loudly on properties accessors.
- **INV-3** *(rewritten by the P2 replicated finding)* — the Tier-1
  modal deck is emitted **flat/replicated**: no partition blocks, every
  rank builds the full model. Mode-shape harvest therefore needs **no**
  cross-partition merge — a rank-0-guarded ordinary recorder carries the
  full field (P3). The earlier MPCO-merge requirement applied to a
  partitioned field that does not exist in this deck.
- **INV-4** *(corrected by P2)* — the Tier-1 eigen preamble is **forced**
  `constraints Transformation` → `numberer RCM` → `system UmfPack`, with
  no test/algorithm/integrator/analysis line. The deck's `system` is NOT
  part of the FEAST solve path (the RCI kernel owns its own distributed
  dmumps) — the earlier "`system Mumps` is load-bearing" claim was a
  carry-over from the refuted plain-eigen design and was disproven live
  (serial-`UmfPack` deck produced kernel-verified distributed solves).
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
- **P1 — modal deck skeleton (tcl target). ✅ DONE (2026-07-16; revised
  same day by the P2 live findings).** `apeSees.modal_deck(path, *,
  band, certify=False, target="tcl", out=)` emits the **flat/replicated**
  model (partitions ignored via the `supports_partitions = False` seam)
  + the deterministic preamble (`Transformation`/`RCM`/`UmfPack`,
  INV-4) + the `getPID` shim + a single captured `set _lam [eigen -feast
  fmin fmax -rci [-certify]]` + rank-0 eigenvalue write-out (INV-5) via
  `TclEmitter.eigen_feast_parallel`. `target="pymp"` (unlock 2a) + staged
  fail loud. Deck-text tests
  (`tests/opensees/integration/test_modal_deck_parallel_feast.py`):
  captured solve exactly once, flat emit pinned (no partition blocks even
  for a partition-authored fem), `modalProperties` absent (INV-2). *(The
  first P1 iteration emitted a partitioned deck with a
  `ParallelPlain`/`Mumps` preamble — refuted live at P2 and rewritten.)*
- **P2 — unlock backend live validation (tcl). ✅ DONE (2026-07-16).**
  Fork PR #578 (`guppi/feast-classic-tcl-parity`, 2 commits: parity +
  adversarial hardening — per-call SOE mutate-in-place, FEAST-failure
  `TCL_ERROR` gate, empty-band NUL fix — each verified by the checked-in
  `Ladruno_scripts/verify_feast_classic_tcl.tcl` smoke against analytic
  eigenvalues). Built `OpenSees.exe` + `OpenSeesMP.exe`; end-to-end at
  `mpiexec -n 2`: apeGmsh-emitted `modal_deck` → distributed FEAST
  (kernel debug proof on both ranks, disjoint triplet slices) →
  `ParallelModalResult.from_job` → 4 modes @ 123.280887 Hz, max rel err
  vs the analytic oracle 9.7e-16. **The partitioned-deck attempt failed
  `FeastEigenSOE::setSize — vertex not in graph`, establishing the
  replicated-model requirement** (and the failure gate from the fork
  hardening stopped the deck correctly). (2a PyMP backend remains
  unimplemented — on demand.)
- **P3 — mode-shape harvest (simplified by P2). ✅ DONE (2026-07-17).**
  Rank-0-guarded eigenvector recorders created AFTER the captured solve
  (dynamic found-mode count — see the Harvest section for why post-solve
  creation is both sound and required) + `record` trigger + `remove
  recorders` + `mode_shapes.json` sidecar, all in
  `TclEmitter.eigen_feast_parallel(shape_nodes=, shape_ndf=)`;
  `modal_deck` pins the column order to sorted mesh node tags. Reader:
  `from_job` loads sidecar + per-mode rows → `mode_shape(node, mode)` /
  `mode_shape_field(mode)` / `shape_nodes`. **LIVE-VERIFIED** on the
  two-column frame (fork classic-Tcl `-feast` build, serial `OpenSees` +
  `mpiexec -n 2 OpenSeesMP`, `LADRUNO_FEAST_MPI` rank 0/1 proof): on a
  degeneracy-broken variant (distinct tip masses 100/120/140/160 → 4
  distinct modes 97.46–123.28 Hz, λ rel err 2.3e-8 vs analytic 6e7/m)
  per-mode **MAC = 1.0** (9 decimals) distributed-vs-serial AND
  distributed-vs-live-openseespy plain-`eigen` oracle; on the stock
  frame (exactly 4-fold degenerate — 1-to-1 MAC is basis-dependent
  there) subspace principal-angle cosines all 1.0. Deck-text + reader
  tests extended (16 green).
- **P4 — `ParallelModalResult` + surface. ◑ PARTIAL (2026-07-16).** Eager
  frozen dataclass (`analysis/modal.py`, re-exported): `eigenvalues` +
  derived ω/f/T + `n_modes` + `certified` flag + `from_job(job_dir,
  out="eigenvalues.out")` eigenvalue harvest (the write-out format is
  pinned by P1, so this is verifiable now — 6 unit cases in
  `tests/opensees/unit/test_parallel_modal_result.py`). Loud
  property-accessor guard (`participation_factors`/`mass_ratios` →
  MPI-blind `NotImplementedError`, INV-2). The `mode_shape` reader
  landed with P3 (2026-07-17). **Viewer binding ✅ DONE (2026-07-17):**
  `ParallelModalResult.to_native(path, fem)` writes the harvested modes
  as mode-kind stages in a native results H5 — the exact
  `DomainCapture.capture_modes` layout (`mode_<k>` / `kind="mode"` /
  eigenvalue-frequency-period-index attrs / `displacement_*` +
  `rotation_*` at one `time=[0.0]` station) — so `Results.from_native`
  → `r.modes` / `r.viewer()` consume the distributed run with **zero
  new viewer code**. The sidecar gained an `"ndm"` key (missing key ⇒
  3-D, the only pre-rev decks) so column→component mapping follows the
  `capture_modes` convention (`displacement` = first `min(3, ndm, ndf)`
  columns; `rotation_x/y/z` when `ndf >= 6`); non-positive eigenvalues
  warn + write `frequency = period = 0` (same contract). Verified live:
  real serial-FEAST harvest → `to_native` → `Results.modes` round-trips
  every component exactly. **P4 COMPLETE.**
- **P5 — HPC e2e + docs.** Full emit → `run_remote` → harvest on the
  cluster (mid-size model); skill/CHANGELOG. Verify: distributed spectrum
  == single-process FEAST oracle; `-certify` completeness reported.

## Cross-references

- ADR 0025 — `Emitter.eigen` widening; ADR 0075 — modal-response family +
  `eigen_feast` (single-process siblings; the classic-Tcl `-feast`
  caveat).
- ADR 0060 / 0061 — remote HPC submission + per-rank deck emission (the
  substrate; note per-rank `py()` raises, relevant to unlock 2a).
- ADR 0027 — cross-partition result merge (referenced by the pre-P2
  harvest design; **not used** — the replicated deck needs no merge).
- Fork PR #578 — classic-Tcl `-feast` parity + hardening (unlock 2b);
  `Ladruno_scripts/verify_feast_classic_tcl.tcl` is its smoke.
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
