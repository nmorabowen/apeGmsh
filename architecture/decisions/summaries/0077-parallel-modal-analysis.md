# ADR 0077 — Parallel modal analysis (distributed FEAST + distributed ARPACK + serial-gather stopgap) (summary)

**Status:** Accepted (2026-07-27 — Tier 0 + both Tier-1 backends implemented and live-verified.
FEAST backend: P0–P4, PRs #800 / #806 / #807. ARPACK backend: P6, this slice. […]

## Decision

Two tiers; Tier 1 carries two backends that invert each other and must not be conflated.

- **Tier 0: serial-gather stopgap.** For a partitioned model whose eigensolve fits one node,
  build the unpartitioned model and run the existing serial `eigen` / `modalProperties`. It
  does not scale the eigensolve. It is the fastest path at every size measured and the only
  route to participation factors and effective modal mass.
- **Tier 1A: distributed FEAST, replicated.** `apeSees.modal_deck(path, *, solver="feast",
  band=(f_min, f_max), certify=False, target="tcl", out=)` emits a deck (not a live run) for
  the ADR 0060 HPC path. The model is **flat/replicated** on every rank (a partitioned deck
  fails `FeastEigenSOE::setSize`); the RCI kernel owns the distributed dmumps. Preamble:
  `constraints Transformation`, `numberer RCM`, `system UmfPack` (the `system` is inert for
  FEAST). Mode shapes: a rank-0 `mode_shapes.json` sidecar plus one `recorder Node` per found
  mode, one `record`, `remove recorders`. Target 2b (classic-Tcl `-feast`, fork PR #578) is
  shipped; 2a (PyMP `.py`) is parked and raises.
- **Tier 1B: distributed ARPACK, partitioned.** `modal_deck(..., solver="arpack",
  num_modes=N, target="tcl")` emits the ordinary `_emit_partitioned` deck plus a forced
  preamble (`constraints Transformation`; `numberer ParallelPlain` else `RCM`; `system Mumps`
  else `UmfPack`) and one captured plain `eigen`. Needs an `OpenSeesMP.exe` at or after
  `5a522b03b` (fork PR #668); an older binary hangs or returns an empty spectrum. The deck is
  not its own serial oracle (at `np = 1` only rank 0's submodel runs). Harvest: per-rank
  `mode_shapes_rank<P>.json` + `mode_shape_<k>_rank<P>.out`, merged by `from_job`.
- **Result surface:** frozen, eager `ParallelModalResult` (`analysis/modal.py`): `eigenvalues`
  (+ ω / f / T, `n_modes`), `certified`, `from_job(job_dir, out=)`, `mode_shape(node, mode)`,
  `mode_shape_field(mode)`, `shape_nodes`; property accessors raise.

Invariants:
- **INV-1** Tier 1 emits a deck, never runs live. **INV-2** no `modalProperties` in a
  parallel deck; properties accessors raise. **INV-6** Tier 0 is bit-for-bit the serial path.
- **INV-3/4** (1A) flat/replicated, so no shape merge; forced `Transformation` → `RCM` →
  `UmfPack`, and the `system` is not in the solve. **INV-5** one `eigen -feast` capture.
- **INV-7** `target` (runtime) and `solver` (eigensolver) are independent seams; 1B is
  classic-Tcl only (`target="pymp"` raises).
- **INV-8** (1B) `system Mumps` is load-bearing; a user-declared non-`Mumps` system raises.
  **INV-9** the deck is partitioned; `solver="arpack"` on an unpartitioned model raises.
  **INV-10** `constraints Transformation` is forced. **INV-11** a single captured `eigen`.
- **INV-12** (1B) owner discipline for additive nodal quantities (`mass`, pattern `load` via
  `primary_owner_map`); `from_job` compares shared-node copies and raises on disagreement.
- **INV-13** never key success off an MPI exit code; use harvested artifacts under a timeout.

Deferred: parallel modal properties; the ADR 0075 modal-response family (single-process);
per-stage parallel modal (`modal_deck` raises on a staged model); `ops.damping.modal` under
MPI stays refused; openseespy / PyMP carry the same latent F1 defect.

## Amendments

- 2026-07-15: adversarial review; the plain-`eigen`-over-`MumpsParallelSOE` v1 was REFUTED
  against vanilla (appendix, findings F1–F4).
- 2026-07-16: P2 live correction; FEAST needs a replicated model (supersedes the partitioned
  framing and the "`system Mumps` is load-bearing" claim for 1A).
- 2026-07-27: F1 CLOSED against a #668 build; the ARPACK route ships as Tier 1B (P6).

Full text: [../0077-parallel-modal-analysis.md](../0077-parallel-modal-analysis.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
