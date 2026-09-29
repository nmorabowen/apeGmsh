# ADR 0061 — Per-rank Tcl deck emission: driver + rank-local sourced fragments

**Status:** Accepted (2026-06-12) — implemented in PR #641; run-verified
on Esmeralda. **The bench (see the dated note at the end) refuted this
ADR's headline parse-time premise at 66k hexes and revised the
mechanism: the binding constraints at production scale are per-rank
*resident deck text* and *NFS read volume* (plus Tcl's 2 GB string
ceiling), each still O(model) → O(model/np) — not CPU parse time, which
measured ~130 ms, not ~5 s. The decision stands on those grounds; the
default stays monolithic and `per_rank=True` is the large-model
escape hatch.**

## Context

Partitioned decks are monolithic: one `.tcl` file holding every rank's
submodel inside `if {[getPID] == K} { ... }` guards
(`TclEmitter.partition_open` / `partition_close`,
`opensees/emitter/tcl.py`). Every MPI rank parses the **entire** file and
executes only its own blocks — per-rank parse cost and resident deck text
are O(model), independent of what the rank owns.

Measured (Esmeralda strong scaling, 2026-06): ~5 s of deck parse at 66k
hexes, with the deck growing at ~250 B/hex. Extrapolated to production
fidelity (P=8 / f=8 ≈ 5.6M hexes): a ~1.4 GB deck, minutes of parse per
rank before the solver does any work, and 16 ranks/node each holding the
full deck text — node-RAM pressure on 64 GB nodes. Deck parse is the
Amdahl serial term of the emit → push → solve pipeline; it caps usable
model size at ~0.5–1M hexes regardless of how many ranks the solve
itself scales to.

What already exists (all verified on main, 2026-06-12):

- **Per-rank structure in the deck.** The partitioned emit path already
  brackets each rank's nodes / elements / fixes / masses / MP constraints
  / patterns between `partition_open(rank)` and `partition_close()` —
  base loop at `apesees.py` (`_emit_partitioned`), plus the nested
  per-rank blocks inside each stage (Phase SSI-2.C). The content split is
  done; only the on-disk layout is monolithic.
- **Multi-file writer precedent.** ADR 0043 split-emit
  (`_write_split_tcl`) already writes a driver that `source`s fragments
  via `[file join [file dirname [info script]] ...]` so the deck runs
  from any cwd. It splits along the compose-*module* axis and explicitly
  rejects partitioned models; this ADR is its rank-axis sibling.
- **Transfer layer is multi-file-free.** `hpc/_ssh.push_dir` ships the
  whole job dir as one gzipped tarball — file count is irrelevant to the
  push, and deck text already compresses ~10× on the wire. The push is
  *not* the bottleneck; runtime parse and resident text are.

The Tcl evaluation rule that makes the fix cheap: a brace-quoted `if`
body is not evaluated (or byte-compiled) unless the condition is true,
and `source` reads its file only when executed. A guard whose body is a
single `source` line therefore costs non-matching ranks one line of
parse — not the fragment.

## Decision

1. **Observation-only span recording in `TclEmitter`.**
   `partition_open(rank)` / `partition_close()` additionally record
   `(rank, body_start, body_end)` line-index spans into a
   `_partition_spans` list (mirroring split-emit's `_SplitLayout`
   bookkeeping). Emitted lines are unchanged — the default monolithic
   output stays byte-identical.

2. **`apeSees.tcl(path, per_rank=True)` → `_write_per_rank_tcl`.** For
   each recorded span: dedent the 4-space body and write it to
   `ranks/rank<K>_<seq>.tcl`; replace the span in the driver with
   `if {[getPID] == K} { source [file join [file dirname [info script]]
   ranks rank<K>_<seq>.tcl] }`. One fragment per span — a rank gets its
   base-model fragment plus one per stage it appears in. The driver
   retains everything global and everything sequential: materials,
   sections, transforms, timeSeries, damping, the `getPID` shim, analysis
   chains, recorders, and the stage skeleton (`domainChange`, per-stage
   analysis chains, `analyze` loops, `loadConst`, `wipeAnalysis`) —
   preserving stage ordering exactly because the driver, not the
   fragments, owns sequence.

3. **Orthogonal flag, fail-loud guards.** `per_rank=` is a separate
   keyword from `split=` (rank axis ≠ module axis; different invariants).
   `per_rank=True` requires a partitioned model
   (`len(fem.partitions) > 1`) and raises otherwise; `split=True` with
   `per_rank=True` raises (compose × partition splitting deferred until a
   real model needs both). `py()` per-rank is out of scope for v1 and
   raises — the HPC path is Tcl, mirroring how ADR 0043 shipped
   Tcl-first.

4. **HPC layer untouched.** The driver is the deck entry point, so
   `Cluster.submit(job_dir, deck="main.tcl")`, `push_dir`, and
   `ops.run_remote` work unchanged — the job dir is pushed wholesale
   (ADR 0060 §2).

5. **Single-process compatibility preserved.** The existing shim makes
   `getPID` return 0 under plain OpenSees, so a single process sources
   only the rank-0 fragments — the same rank-0-submodel semantics the
   monolithic deck has today (ADR 0027 INV-5).

## Alternatives considered

- **Post-process the monolithic text** (regex over `if {[getPID] == K}`
  blocks outside the emitter): rejected — fragile against nested braces
  in patterns and stage blocks; the emitter knows the spans exactly and
  recording them is a few lines.
- **One file per rank, sourced once:** rejected — staged decks interleave
  per-rank topology with the global sequential skeleton (`domainChange`,
  per-stage analysis chains), so a single per-rank file cannot preserve
  ordering. One fragment per span keeps the driver authoritative for
  sequence.
- **Separate `TclEmitter` per rank inside the build loop:** rejected for
  v1 — invasive to the orchestrator (the base partitioned loop plus four
  staged per-rank blocks), with no benefit over span slicing at the
  writer.
- **Attack the push instead** (compression, rsync): orthogonal and
  already done — `push_dir` tarball-gzips. The bottleneck is runtime-side
  parse and resident text, which only a layout change fixes.

## Consequences

- Per-rank parse drops from O(model) to **O(global + model/np)** — the
  measured serial term collapses ~np×. Node-wide resident deck text drops
  the same way: 16 ranks/node × full deck becomes ≈ one deck total per
  node spread across its ranks.
- Total emitted bytes are unchanged, so push time is unchanged (it was
  already compressed). Emission wall-time on the authoring machine is
  also unchanged — and becomes the next ceiling: at 5–6M hexes the
  Python-side line buffer is multi-GB. Streaming emit is a separate
  follow-up item, deliberately not coupled to this ADR.
- File count is `np × (1 + stages-the-rank-appears-in)` plus the driver —
  a few hundred files at np=64 with stages; trivial for the tarball push
  and for NFS.
- Verification surface:
  (a) byte-identity — `per_rank=False` output unchanged (span recording
  is observation-only);
  (b) reassembly parity — re-inlining + re-indenting fragments into the
  driver reproduces the monolithic deck line-for-line (pure-text test,
  exhaustive and cheap);
  (c) live parity — `mpiexec -n 2/4` per-rank deck vs monolithic deck
  produce identical results (existing `test_emit_partitioned_*` e2e
  template);
  (d) bench — re-emit the 66k-hex strong-scaling deck with
  `per_rank=True` and confirm the ~5 s parse term collapses before
  trusting the extrapolation to multi-M-hex models.
- Recorders stay global in the driver (one declaration, every rank
  executes it; per-rank output routing is the recorder's own concern —
  unchanged from today).

> **Run-verified bench (2026-06-12, Esmeralda jobs 143849–143852) —
> verification item (d), and an honest revision of the Context.** The
> 66,564-hex notebook-5 model was re-emitted {monolithic, per_rank} ×
> np{16, 64} with `clock milliseconds` timers injected at deck top and
> just before the stage banner (interpreter start → end of base-model
> parse+build, before any MPI collective). Measured slowest-rank
> parse+build: **monolithic 130 ms (np16) / 51 ms (np64); per-rank
> 162 ms / 58 ms** — per-rank is marginally *slower* at this scale (two
> extra NFS file opens per rank). The "~5 s deck parse" this ADR's
> Context inherited from the strong-scaling campaign was a
> **misattribution**: Tcl brace-scans non-matching `if` bodies at
> GB/s and only evaluates the rank's own ~14k commands; the real
> ~2.6 s fixed cost sits between end-of-model and end-of-deck
> (stage/analysis initialization + MPI collectives — identical across
> layouts and untouchable by emission), plus srun startup outside the
> deck. What survives, and what the decision now rests on, all still
> O(model) → O(model/np): **(a) resident deck text** — `source` slurps
> the whole file per rank, so a 1.4 GB monolithic deck × 16 ranks/node
> ≈ 22 GB/node of deck strings before Tcl object overhead (the hard
> OOM blocker for P=8/f=8 production fidelity); **(b) NFS read
> volume** — every rank reads the full deck vs. its 1/np slice;
> **(c) Tcl's 2 GB string-length ceiling** (`Tcl_Obj` lengths are
> `int`) — monolithic decks become *impossible to source at all* past
> ~8M hexes, layout aside. CPU scan extrapolates to a bounded ≤~5 s at
> 1.4 GB, not minutes. Practical guidance: keep the monolithic default
> below ~100 MB decks; reach for `per_rank=True` when deck size ×
> ranks-per-node threatens node RAM, NFS bandwidth, or the 2 GB
> ceiling. Bench artifacts:
> `Downloads/results/ladruno_wave_propagation/bench_adr0061/`.
