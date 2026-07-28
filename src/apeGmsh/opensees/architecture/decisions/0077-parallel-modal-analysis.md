# ADR 0077 — Parallel modal analysis (distributed FEAST + distributed ARPACK + serial-gather stopgap)

**Status:** Accepted (2026-07-27 — Tier 0 + both Tier-1 backends
implemented and live-verified. FEAST backend: P0–P4, PRs #800 / #806 /
#807. ARPACK backend: P6, this slice. The remaining P5 cluster e2e is a
deployment task, not a design gate: the whole chain rides the unchanged
ADR 0060 submit path and is blocked only on the fork build reaching the
cluster. Unlock 2a — the PyMP `.py` backend — is **parked on demand**:
it still requires deploying a fork artifact, so it buys nothing wherever
the classic-Tcl exes can be placed. Proposed 2026-07-15; **reworked
after adversarial review** — the original plain-`eigen`-over-
`MumpsParallelSOE` "v1" was REFUTED against vanilla and struck, then
**un-refuted 2026-07-27** when fork PR #668 wired it; see the review
banner and the appendix). Brings modal analysis to apeGmsh's
partitioned / HPC path. Rides the apeGmsh HPC path (ADRs 0060 / 0061).
Modal-response family (ADR 0075) stays single-process.

> ⚠ **Adversarial review outcome (2026-07-15) — SUPERSEDED IN PART on
> 2026-07-27 by fork PR #668; read the amendment below before acting on
> this banner.** An originally-proposed **v1 — plain `eigen` over
> `MumpsParallelSOE` in a classic Tcl deck under `OpenSeesMP.exe` — was
> REFUTED** (two adversarial agents + direct source check). Under
> `_PARALLEL_INTERPRETERS` the `eigen` command builds an `ArpackSOE`
> with **no** `setProcessID`/`setChannels`, so ARPACK's
> generalized-problem `M*v` reduction never fires and the shift-invert
> `(K−σM)` solve is distributed (global K) while `M*v` stays **local** —
> an incoherent operator. The observed pre-fix behaviour was worse than
> "garbage modes": rank 0 returned an **empty spectrum** and rank 1
> **deadlocked** (fork negative control, `mpiexec -n 2`). Full detail +
> the harvest defects (F2–F4) are in the appendix.
>
> ✅ **Amendment (2026-07-27) — F1 is CLOSED against a #668 build.**
> Fork PR #668 (`5a522b03b`) gave `ArpackSOE` public
> `setProcessID`/`setChannels` and made the MP `eigen` command apply
> them — on the creation path **and** the reuse-by-classTag path —
> **conditioned on the analysis `LinearSOE` being `MumpsParallelSOE`**.
> With that wiring the three collectives the class always carried
> (the P0-star `M*v` merge `ArpackSolver::myMv`, the `ido` lockstep
> guard `checkSameInt`, the global-size collective `setSize`) wake up,
> and the algorithm takes the same shape as FEAST L3: replicated ARPACK
> outer loop, distributed `dmumps` factor+solve of `(K−σM)`, merged
> `M*v`. Verified serial / np2 / np4 at ≤1.96e-15 against an analytic
> oracle (`Ladruno_scripts/verify_arpack_mp_mumps.tcl`, 4 checks).
> **The refutation was correct against vanilla and is false against a
> #668 build**, so this route is now a shipped Tier-1 backend (Tier 1B
> below) — not a rejected alternative. What did *not* change: the
> **runtime** `processID` was always the operative gate (see the
> appendix correction), `modalProperties` stays MPI-blind, and
> `ops.damping.modal` under MPI stays refused (`apesees.py`) — with the
> `eigen` now succeeding, that guard is the only thing left standing
> between a user and a silently rank-count-dependent damped response.

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
  Lanczos*, and — the fatal part *in vanilla* — nobody ever called
  `setProcessID` on the `ArpackSOE`, so it stayed at `processID = -1`
  and the `M*v` reduction that would globalize the generalized problem
  never fired: operator global-K / local-M (the refuted v1). Under
  classic `OpenSeesMP.exe` there is **no** `PartitionedDomain` at all:
  each rank holds a plain `Domain` with the element subset apeGmsh's
  `if {[getPID]==K}` blocks give it. **Fork PR #668 closed the MP half
  of this**: with `setProcessID`/`setChannels` applied whenever the
  analysis `LinearSOE` is `MumpsParallelSOE`, the ARPACK route becomes a
  genuine distributed eigensolve over a partitioned domain — the
  ecosystem's second correct one (Tier 1B). The SP half is unchanged and
  still not a target.
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
answer.** Tier 1 now carries **two solver backends** — the FEAST one it
was designed around (Tier 1A, replicated) and the ARPACK one fork PR
#668 unlocked (Tier 1B, partitioned). They are not variants of one
deck: the two invert each other on the two facts that matter (flat vs
partitioned emit; `system` inert vs load-bearing), which is why each
carries its own preamble, its own harvest, and its own invariants.
Choosing between them:

| | **Tier 1A — FEAST** | **Tier 1B — ARPACK** |
|---|---|---|
| model on each rank | **full** (replicated) | **its partition only** |
| what is distributed | the contour solves, inside the RCI kernel | the `(K−σM)` factor+solve, via the deck's SOE |
| deck `system` line | inert (INV-4) | **load-bearing** — must be `Mumps` (INV-8) |
| mode selection | frequency band + `-certify` | lowest `num_modes` |
| RAM per rank | O(full model) — the L3 regime, ~1e5–1e6 DOF | O(model / ranks) |
| fork build needed | classic-Tcl `-feast` parity (PR #578) | ARPACK MP wiring (PR #668, `5a522b03b`) |

**Pick 1B when the model does not fit on one node** — that is the case
it strictly dominates, since both the storage and the factorization are
distributed. Pick 1A when you want a frequency window rather than the
lowest N, or want `-certify` completeness. Pick **Tier 0 for everything
that fits on one node**: serial ARPACK is still the fastest path at
every size measured, and it stays the **only** route to correct
participation factors and effective modal mass.

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

### Tier 1A — distributed FEAST (`eigen -feast … -rci`), runtime-agnostic

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

`apeSees.modal_deck(path, *, solver="feast", band=(f_min, f_max),
certify=False, target="tcl", out=)` emits a **replicated modal deck**
(not a live run), submittable through the existing HPC path. The driver
is **runtime-agnostic** (`target` seam):

- **(2a) PyMP `.py` deck** — an OpenSeesMP-Python deck run under
  `mpiexec … python` (PyMP parses `-feast`). Needs a PyMP launcher shim;
  the replicated finding removes the per-rank-`py()` blocker the earlier
  draft assumed. **PARKED on demand** (raises). The original "zero fork
  dependency" selling point does not hold in practice: PyMP is the
  fork's `PythonMPIModule` build and the deployed pyd predates FEAST,
  so this route still deploys a fork artifact — it only becomes
  interesting in an environment that allows Python packages but not
  executables.
- **(2b) classic-Tcl `-feast` parity** — **SHIPPED + VERIFIED**: fork PR
  #578 wires `eigen -feast … -rci` into `SRC/tcl/commands.cpp`
  (+ adversarial hardening: per-call SOE mutate-in-place, FEAST-failure
  `TCL_ERROR` gate, empty-band NUL fix); built `OpenSees.exe` /
  `OpenSeesMP.exe` and validated end-to-end (see P2). Deployment to the
  cluster pending.

The band form has no a-priori mode count (the contour *is* the band) —
the result surface handles a dynamic mode count, and `-certify` adds a
completeness flag.

### Preamble (Tier 1A) — deterministic, every rank identical

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

### Harvest (Tier 1A)

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
  `modalProperties` is MPI-blind (C7) and neither FEAST nor #668 changes
  that. Tier 1 does **not** emit it; the result surface raises a clear
  `NotImplementedError` on `.participation_factors(...)` /
  `.mass_ratios`, directing seismic mass-participation users to Tier 0
  or a future MPI-aware reduction (Deferred).

### Tier 1B — distributed ARPACK (plain `eigen` over `MumpsParallelSOE`)

`apeSees.modal_deck(path, *, solver="arpack", num_modes=N, target="tcl",
out=)` emits the **ordinary partitioned deck** — `if {[getPID]==K}`
blocks and all, straight off `_emit_partitioned` — plus a preamble and a
single captured plain `eigen`. Everything the FEAST section says about
replication is *inverted* here: partitions are the point, not an
obstacle, and each rank holds only its own slice of `(K, M)`.

```tcl
# ... partitioned model blocks, as emitted today ...
constraints Transformation                       ;# forced (INV-10)
if {[catch {numberer ParallelPlain} _err]} { numberer RCM }
if {[catch {system Mumps} _err]} { system UmfPack }   ;# LOAD-BEARING (INV-8)
set _lam [eigen 4]                               ;# ONE capture (INV-11)
if {[getPID] == 0} { … puts $_lam … }
```

The runtime-conditional numberer/system pair is the existing ADR 0027
INV-5 form, forced here rather than auto-emitted: under `OpenSeesMP` it
resolves to `ParallelPlain` / `Mumps` (which is what engages the #668
wiring), and under single-process `OpenSees` it degrades to `RCM` /
`UmfPack` so the deck still **parses**.

> ⚠ **It does not, however, run as its own serial oracle** — and this
> is the one place Tier 1B's intuition from Tier 1A actively misleads.
> The FEAST deck is replicated, so it gives the same answer at any rank
> count. This deck is partitioned: single-process, the `getPID` shim
> returns 0 and **only rank 0's block executes**, so `OpenSees` builds
> rank 0's *submodel* and solves that. Observed on the 8-mass chain
> split in two — the 4-DOF submodel fails ARPACK's `NCV > NEV` guard,
> `eigen` returns an **empty list**, and the deck writes it out without
> complaint; `from_job` reporting `n_modes == 0` is the only signal.
> The oracle for this backend is a separate **unpartitioned** Tier-0
> run (or the analytic spectrum), never the same deck at `np = 1`.

**Runtime precondition, stated in the docstring:** this deck needs an
`OpenSeesMP.exe` built from `ladruno` at or after `5a522b03b` (PR #668).
Against an older binary it does not degrade — it **hangs or returns an
empty spectrum**. No runtime feature probe exists; the fork's
`Ladruno_scripts/verify_arpack_mp_mumps.tcl` is the deployment gate
(run under a timeout, require `cases=4` on every rank — never the exit
code, see INV-13).

### Harvest (Tier 1B) — the FEAST harvest does NOT transfer

The Tier-1A mode-shape harvest works *because* its deck is replicated:
rank 0 holds every node, so one rank-0 recorder captures the whole
field. Here each rank holds only its partition, so the identical code
would capture **only rank 0's slice** and `mode_shape_field` would
return a partial mode **with no error**. Instead:

- **Per-rank sidecars + client-side merge.** Inside each rank's guarded
  block, after the captured solve: write `mode_shapes_rank<P>.json`
  (the tags *that rank owns*, in its own recorder column order, plus
  `ndf` / `ndm` — same key names as the replicated sidecar), then one
  recorder per found mode over that rank's nodes
  (`mode_shape_<k>_rank<P>.out`), a single `record`, `remove recorders`.
  Same post-solve creation mechanics and the same source-verified
  reasons as Tier 1A. `from_job` globs the per-rank sidecars,
  concatenates, and sorts by node tag into the same `(n_nodes, ndf)`
  layout the replicated path produces — one reader surface, two deck
  shapes.
- **The boundary-node agreement check (INV-12) — a correctness assertion,
  not a tidiness one.** A node on a partition boundary is defined on
  **both** touching ranks and therefore appears in two sidecars. Do not
  take the first: **compare them.** Both ranks ran the same replicated
  ARPACK outer loop over the same global operator, so the components
  must agree to solver tolerance. Disagreement means each rank ran a
  *private* Lanczos — exactly the failure this whole backend exists to
  prevent, and exactly what a pre-#668 binary produces. `from_job`
  raises with the offending node tag and both values. This is the
  cheapest available proof that the run was really distributed, and it
  costs one comparison per shared node.

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
  ("ship-now v1"), *against a vanilla / pre-#668 build*.** REFUTED (F1)
  — the `ArpackSOE` was never given a `processID`, so the operator was
  global-K / local-M and the observed behaviour was an empty spectrum on
  rank 0 with rank 1 deadlocked. **No longer a rejected alternative:**
  fork PR #668 (`5a522b03b`) wired it, and it is now Tier 1B above. The
  rejection stands only for a build without that commit — and since
  there is no runtime feature probe, "wrong binary" is a deployment
  precondition, not a guard the deck can carry.
- **Emitting the Tier-1B deck with any `system` other than `Mumps`.**
  The fork wires the `ArpackSOE` collectives *only* when the analysis
  `LinearSOE` is `MumpsParallelSOE`. `ParallelProfileSPD` and
  `MPIDiagonal` are the trap: they are genuinely distributed and their
  own collectives are live, so the deck runs — while the `ArpackSOE`
  stays at `processID -1`. That is the identical incoherent composition
  as the negative control, with no error (ADR-1000 §34.4). `modal_deck`
  raises rather than emit it (INV-8).
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
  the **Tier-1A** deck uses the flat emit, not `_emit_partitioned`;
  Tier 1B uses `_emit_partitioned` — see INV-9.)*
- **INV-2** — no `modalProperties` in the parallel deck (Tier 1); the
  parallel result surface raises loudly on properties accessors.
- **INV-3** *(rewritten by the P2 replicated finding; scoped to 1A)* —
  the **Tier-1A** modal deck is emitted **flat/replicated**: no
  partition blocks, every rank builds the full model. Mode-shape harvest
  therefore needs **no** cross-partition merge — a rank-0-guarded
  ordinary recorder carries the full field (P3). The earlier MPCO-merge
  requirement applied to a partitioned field that does not exist in
  *that* deck. It very much exists in Tier 1B (INV-9 / INV-12).
- **INV-4** *(corrected by P2; scoped to 1A)* — the **Tier-1A** eigen
  preamble is **forced** `constraints Transformation` → `numberer RCM` →
  `system UmfPack`, with no test/algorithm/integrator/analysis line. The
  deck's `system` is NOT part of the FEAST solve path (the RCI kernel
  owns its own distributed dmumps) — the earlier "`system Mumps` is
  load-bearing" claim was a carry-over from the refuted plain-eigen
  design and was disproven live (serial-`UmfPack` deck produced
  kernel-verified distributed solves). **This is the invariant Tier 1B
  inverts** (INV-8); the two must never be conflated.
- **INV-5** — eigenvalues are captured from a **single** `eigen -feast`
  return and written once from rank 0 (no double solve, no rank-0-only
  collective).
- **INV-6** — Tier 0 is bit-for-bit the existing serial `eigen` /
  `modalProperties` on the unpartitioned build (correctness by
  reduction to an already-tested path).
- **INV-7** — the Tier-1 driver is runtime-agnostic: the same logical
  modal deck renders to a PyMP `.py` deck (2a) or a classic-Tcl deck
  (2b); only the emitted solver-invocation surface differs. `target` is
  the *runtime* axis and `solver` the *eigensolver* axis — they are
  independent seams. Tier 1B is classic-Tcl only (`target="pymp"`
  raises): the modern interpreter still builds its `ArpackSOE` bare, so
  openseespy / PyMP carry the same latent F1 defect (ADR-1000 §34.4).

Tier 1B (ARPACK) — each of these is load-bearing; the guide's INV-A*
names are given for cross-reading:

- **INV-8** *(guide INV-A1)* — **`system Mumps` is load-bearing** in the
  Tier-1B deck, the exact opposite of INV-4. The fork wires the
  `ArpackSOE` collectives **only** when the analysis `LinearSOE` is
  `MumpsParallelSOE`; any other system and the wiring silently does not
  engage. A user-declared non-`Mumps` system on a Tier-1B deck **raises**
  — it is not overridden silently and not warned past.
- **INV-9** *(guide INV-A2)* — the Tier-1B deck is **partitioned**, not
  flat: the ordinary `_emit_partitioned` output. It must **not** reuse
  the Tier-1A `supports_partitions = False` seam, and `solver="arpack"`
  on an unpartitioned model raises (Tier 0 is that model's answer).
- **INV-10** *(guide INV-A3)* — `constraints Transformation` is
  **forced**, same reasoning as INV-4: the auto-emit only fires when MP
  constraints exist, while `Penalty` pollutes M with penalty mass and
  `Lagrange` injects zero-mass DOFs — either fabricates spurious modes.
- **INV-11** *(guide INV-A4)* — a **single** captured `eigen`, identical
  to INV-5. A second `[eigen …]` inside the rank-0 write-out is a
  redundant distributed solve *and* a rank-0-only collective ⇒ deadlock.
- **INV-12** *(guide INV-A6 + the §3 check)* — **owner discipline for
  additive nodal quantities, and cross-rank agreement on shared nodes.**
  The `M*v` merge sums per-rank contributions, so a boundary node given
  mass on two ranks contributes twice; because the Tcl `eigen` path
  always has `shift = 0`, K stays exactly right while M goes wrong, and
  **nothing surfaces as an error** — a plausible spectrum, biased low.
  apeGmsh owns this: `_emit_partitioned` already routes `mass` (and
  pattern `load`) lines through `primary_owner_map`, one owning rank per
  node, so the Tier-1B deck inherits the discipline rather than trusting
  the user. On the harvest side the *idempotent* replication is kept
  deliberately — shared nodes are recorded on **every** owning rank —
  because comparing the two copies is the cheapest proof the run was
  really distributed. `from_job` raises on disagreement.
- **INV-13** — **never key success off an MPI exit code.** Under
  `_PARALLEL_INTERPRETERS` a Tcl error exits **0** (`g3TclMain` discards
  its exit code; `mpiParameterMain` returns 0 unconditionally), and a
  rank that `exit`s while a peer is deadlocked blocks forever in
  `MPI_Finalize`. Orchestration and tests key off **harvested
  artifacts, under a timeout**.

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
- **P5 — HPC e2e + docs (deployment-gated — no new machinery).** The
  chain rides the unchanged ADR 0060 path (`modal_deck` → deck-agnostic
  `Cluster.submit(binary=…)` → `Job.fetch` → `from_job`); blocked only
  on deploying the fork build to the cluster.
  Full emit → `run_remote` → harvest on the
  cluster (mid-size model); skill/CHANGELOG. Verify: distributed spectrum
  == single-process FEAST oracle; `-certify` completeness reported.
  Applies to both backends.

**Tier 1B (distributed ARPACK) — the second backend:**
- **P6 — ARPACK-MP backend. ✅ DONE (2026-07-27).** Fork dependency
  merged (PR #668, `5a522b03b`); the emitter guide
  (`Ladruno_implementation/ladruno_arpack_mp_apegmsh_emitter_guide.md`,
  fork PR #673) is its spec and ADR-1000 §34 the normative background.
  Landed: `TclEmitter.eigen_parallel(...)` beside `eigen_feast_parallel`
  (single captured `eigen $num_modes`, rank-0 eigenvalue write-out,
  per-rank sidecars + per-rank per-mode recorders);
  `apeSees.modal_deck(..., solver="feast"|"arpack", num_modes=)` with
  the partitioned emit and the forced
  `Transformation`/`ParallelPlain`/`Mumps` preamble (INV-8/9/10) and
  fail-loud guardrails; `ParallelModalResult.from_job` partitioned
  branch (per-rank sidecar glob → merge → boundary-agreement assertion,
  INV-12).
- **P6 verification. ✅ LIVE.** Fixture: the fixed-free 8-mass axial
  chain the fork smoke uses (`make_axial_chain_partitioned` — the one
  partitioned fixture whose ranks actually *touch*; the two-column frame
  splits into disjoint halves and shares nothing), emitted by
  `modal_deck(solver="arpack", num_modes=4)` and run against
  `build/build/Release/OpenSeesMP.exe` (post-`5a522b03b`). Compared
  against the analytic oracle `λ_j = 4(k/m)·sin²((2j−1)π/(2(2n+1)))`:

  | run | max rel err in λ |
  |---|---:|
  | Tier 0 (unpartitioned, in-process `ops.eigen`) | 5.36e-16 |
  | distributed `mpiexec -n 2` | **1.30e-15** |
  | distributed `mpiexec -n 4` | **8.30e-16** |
  | Tier 0 vs distributed | 1.57e-15 |

  Digit-for-digit the fork smoke's own np2 / np4 figures. The merge
  returned the **full** 9-node field from both (and all four) ranks with
  the shared boundary node agreeing — mode-1 shape error 4.0e-6 against
  the analytic sine profile, which is recorder *text* precision (~6
  significant digits), not solver error. `np = 1` is **not** in the
  table on purpose: a partitioned deck at one rank builds only rank 0's
  submodel (see the Tier-1B warning above), so the single-process oracle
  is Tier 0, not this deck at `-n 1`.
- **P6 mutation testing.** Every load-bearing assertion is paired with
  the defect it claims to catch, in
  `test_modal_deck_parallel_arpack_mutations.py` — the mutation is
  applied to the **emitter**, not to the deck text, so it proves the
  test catches a code defect rather than that a regex is sensitive.
  Caught: `system UmfPack` instead of `Mumps`; flat emit instead of
  partitioned; rank-0-only harvest (the FEAST copy-paste); mass on every
  owner instead of the primary one; the boundary check removed; the
  boundary tolerance widened to 10 %. Plus one **live** mutation — the
  emitted deck with `system UmfPack` at `mpiexec -n 2` returned an
  **empty spectrum** (`n_modes == 0`), the fork's documented negative
  control, confirming INV-8 is load-bearing in fact and not just on
  paper. This is the discipline #668 itself needed — its original smoke
  *passed* a gate-deleted binary at 1.3e-15 — and it is now the standard
  for this ADR: **a green test that cannot distinguish is not a test.**
- **P6 adversarial pass (2026-07-27).** Nine probes that *executed* the
  new code rather than re-read it. Two findings, seven non-findings.

  | # | finding | verdict |
  |---|---|---|
  | **A1** | A modal deck **silently drops an `enforce="equation"` tie.** The forced `constraints Transformation` (INV-4 / INV-10) is emitted last and wins over the `Lagrange` upgrade the model needs (ADR 0068 INV-4), so the deck runs to completion and returns the spectrum of a *different structure*. Measured: six `equationConstraint` rows under a bare `constraints Transformation`. **Hit BOTH backends — Tier 1A shipped with it at P1 (2026-07-16); Tier 1B inherited it.** | **FIXED** — `_guard_modal_deck_constraint_handler` refuses equation ties (and contact, same shape: it needs `LadrunoContact`) on both solvers. Only one handler can be active, so no emit satisfies both. `ops.tcl` is untouched and still auto-upgrades. Regression-tested with a control asserting the ordinary path still emits `Lagrange` + 6 rows. |
  | **A2** | A **cross-rank MP constraint makes the distributed eigensolve return an EMPTY spectrum.** Two 4-element chains, tips `equalDOF`-tied across the partition: Tier 0 gives 4 modes, `mpiexec -n 2` gives 0 with `MumpsParallelSolver … Matrix is Singular Numerically`. Root cause is **not** in this slice: the ADR 0027 foreign-node declaration emits the ghost node **bare — without replicating the owner's `fix`** — so the non-owning rank holds a free, massless, stiffness-less DOF and the two ranks disagree on whether it is constrained. | **FIXED in ADR 0027, not here** (2026-07-27, same session). The foreign-node declaration now carries the owner's SP constraints; see ADR 0027 INV-2 for the amendment, the before/after table and the four mutation-tested regressions. Follow-up measurement corrected the scope this row originally recorded: **an ordinary partitioned STATIC `analyze` hits the identical singularity** (`returned: -3`, empty recorder), so this was never modal-specific — the modal path merely surfaced it first, and worse, by writing an *empty* result where static at least printed a failure. After the fix: modal matches the Tier-0 oracle at **6.07e-16**, static returns the analytic `u₅ = u₁₀ = 0.02`. **Tier 1A was unaffected throughout** (flat deck, no ghost nodes). |

  **Non-findings** — hypotheses discarded on evidence, recorded so they
  are not re-probed: the merge handles a node shared by **three** ranks,
  catches a *third* rank disagreeing (not just the second), tolerates
  unsorted sidecar node lists, and is order-independent at 12 ranks
  (lexicographic glob puts `rank10` before `rank2`); `to_native`
  round-trips a **merged** partitioned harvest through
  `Results.from_native` with the shared node appearing once (now a
  regression test); the `mass_from_model` streaming path (ADR 0065
  Tier 2 — what a production-size model actually uses) honours the
  same one-owning-rank discipline as explicit `ops.mass` records;
  a sidecar whose name does not parse is ignored rather than crashing.
  Two accepted-as-is: a user-declared `system Mumps` yields two `system
  Mumps` lines (top-of-deck + forced preamble — same token, last wins,
  harmless), and `num_modes=4.9` truncates to 4 rather than raising
  (`int()` coercion, consistent with the rest of the emitter).
- **P6 note — the deck's rank count is baked in at emission.** A
  2-partition deck run at `mpiexec -n 4` leaves two ranks holding empty
  domains; measured, it does not error but diverges in the ARPACK
  lockstep guard (`ido values not the same`) after `Graph::recvSelf`
  failures. Emit for the rank count you will run. The ADR 0060 submit
  path already derives ranks from `len(fem.partitions)`, so this only
  bites a hand-run deck.

## Cross-references

- ADR 0025 — `Emitter.eigen` widening; ADR 0075 — modal-response family +
  `eigen_feast` (single-process siblings; the classic-Tcl `-feast`
  caveat).
- ADR 0060 / 0061 — remote HPC submission + per-rank deck emission (the
  substrate; note per-rank `py()` raises, relevant to unlock 2a).
- ADR 0027 — per-rank fan-out + cross-partition MP-constraint policy.
  Tier 1B rides it directly: its `primary_owner_map` mass routing is
  what makes INV-12's owner discipline free, and its INV-5 runtime-
  conditional numberer/system pair is the form the Tier-1B preamble
  forces. (Its *result* merge is still unused — that is MPCO/`.ladruno`
  HDF5 node-id matching; the Tier-1B shape merge is the sidecar one.)
- Fork PR #578 — classic-Tcl `-feast` parity + hardening (unlock 2b);
  `Ladruno_scripts/verify_feast_classic_tcl.tcl` is its smoke.
- **Fork PR #668 (`5a522b03b`) — the ARPACK MP wiring Tier 1B depends
  on**; `Ladruno_scripts/verify_arpack_mp_mumps.tcl` is its smoke (4
  checks: gate TRUE side, gate FALSE side, reuse re-gate both
  directions) and doubles as the deployment feature-probe for a build.
  Fork PR #673 — the emitter guide this backend implements.
- **Fork ADR-1000 §34** — normative background for Tier 1B: the exact
  defect mechanism, the wiring, the evidence table, and §34.4's list of
  what it does NOT give (MPI-blind `modalProperties`; silently wrong
  `modalDamping`; latent F1 in openseespy/PyMP; `ParallelProfileSPD` /
  `MPIDiagonal` left unwired; shared-node mass double-count).
- Fork ADR 43 (`43_ladruno_feast_eigensolver_adr.md`) +
  `modal_gap_study/00_SYNTHESIS.md` §3 (the SP/MP non-composition FEAST
  resolves) + `feast_l2_profile/README.md` (L2 "don't build").

## Deferred

- **Parallel modal properties** (participation, effective mass) — needs
  an MPI-aware `modalProperties` (upstream/fork fix) or client-side
  computation from harvested Φ + a mass export. Not in Tier 0/1, and
  #668 does not change it: participation factors and effective modal
  mass stay **Tier 0 only**.
- **Parallel modal-response family** (ADR 0075) — stays single-process.
- **Per-stage parallel modal** — inherits the ADR 0075 / SSI-2.A staged
  deferral. Applies to both Tier-1 backends (`modal_deck` raises on a
  staged model).
- **`ops.damping.modal` under MPI stays refused** — deliberately *not*
  unblocked by Tier 1B. #668 makes the preceding `eigen` correct, but
  `modalDamping`'s projection is still a rank-local partial sum
  (`Σ_r β_r M_r φ`, not `β M φ`), so damped response silently changes
  with rank count. The eigen no longer failing means the guard in
  `_emit_global_damping_partitioned` is now the *only* thing stopping
  this — it must not be relaxed on the grounds that "the eigen works
  now."
- **openseespy / PyMP** carry the same latent F1 defect
  (`OpenSeesCommands.cpp:376` builds its `ArpackSOE` equally bare) —
  out of scope here, and the reason `solver="arpack"` refuses
  `target="pymp"`.

## Appendix — adversarial review findings (2026-07-15)

Two adversarial agents (fork-source verification + design refutation)
plus a direct source check. The fork-facts pass confirmed C1–C9 (parse
gap, FEAST L3-only, L2 not built, MP-only, `modalProperties` MPI-blind,
Node `eigen` recorder exists, eigenvalues replicated per rank). The
design pass found the fatal flaws that reshaped this ADR:

- **F1 (fatal, decisive — CLOSED 2026-07-27 by fork PR #668).** Plain
  `eigen` over `MumpsParallelSOE` under `OpenSeesMP.exe` → incoherent
  global-K / local-M operator (`ArpackSOE` built without
  `setProcessID`/`setChannels`, `commands.cpp:6030`). Corroborated by
  apeGmsh's own guard `apesees.py`. → Plain-eigen v1 rejected; FEAST
  elevated to the first real slice.

  **Two corrections to how this finding was written up, both from the
  #668 verification:**

  1. **The gate was never `#ifdef _PARALLEL_PROCESSING`.** This appendix
     originally described the `M*v` reduction as SP-gated at compile
     time. It is not: `ArpackSolver`'s three collectives (the P0-star
     `M*v` merge, the `ido` lockstep `checkSameInt`, the global-size
     `setSize`) are compiled in unconditionally and gated at **runtime**
     on `processID != -1`. That is *why* the class's collectives were
     present but dormant under MP — the only wiring path was
     `sendSelf`/`recvSelf`, which happens in SP only. Same consequence,
     different key — and the difference is exactly what made the fix two
     small edits rather than a port.
  2. **The observed failure was worse than "garbage modes".** The
     pre-fix negative control (`mpiexec -n 2`, partitioned deck) gave an
     **empty spectrum on rank 0 and a deadlocked rank 1** — unusable,
     not a degraded answer. The "per-rank-local garbage modes" phrasing
     was a prediction from the source read; the measurement is louder.

  The finding was correct and decisive for its build. Against a
  `5a522b03b`-or-later build it no longer holds, and the route is
  Tier 1B.
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
