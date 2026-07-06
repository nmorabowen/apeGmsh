# ADR 0065 — Streaming deck emission: write-through sink to remove the author-side line buffer

**Status:** ACCEPTED — fully shipped (2026-07-06). Tier 1 (write-through
`write_to`) shipped 2026-06-18. The Tier-2 demand-gate profile ran as
"M0" of the v2 plan (`internal_docs/plan_emit_memory_columnar.md`, #772):
it confirmed the line buffer was ONE of several dominant terms — the
build-time object graph (element plan, PG fan-out memo, ownership
dicts/sets, MassRecords) was collectively larger, and was removed first as
plan Routes B+C (#773–#776, columnar containers). **Tier 2 (Decision §1–§5:
dual-mode sink + live per-rank fragment routing + atomic writes) shipped as
`ops.tcl(path, stream=True)` in #777** — byte-identical to list mode
(including per-rank fragments vs `_write_per_rank_tcl`), tracemalloc-proven
O(1) line-buffer memory. Cumulative measured effect of the v2 plan at the
103k-hex / 64-rank / staged reference: emit phase-peak 2,291 → 1,039 B/hex.

## Context

Tcl and Python deck emission accumulate the **entire deck as a Python
`list[str]` in RAM**, then write it once. The emit loop itself is already
streaming-friendly — `BuiltModel.emit` iterates *live* over the FEMData
numpy arrays (`for nid, xyz in zip(self.fem.nodes.ids,
self.fem.nodes.coords)`, `apesees.py`) and never builds an intermediate
node/element list. The memory cost is entirely in the **sink and the final
write**:

1. **The line buffer.** Every Protocol method calls
   `self._lines.append(...)` into `_LineBuf` (a `list[str]` subclass,
   `emitter/tcl.py`; the Python emitter has its own in `emitter/py.py`).
   At 5–6M hexes that is ~12M `str` objects — multiple GB in object
   overhead alone, before character data.

2. **The write-time double-copy.** `apesees.py`:
   ```python
   f.write("\n".join(emitter.lines()) + "\n")
   ```
   `emitter.lines()` returns `list(self._lines)` (a full list copy), and
   `"\n".join(...)` then materializes **one contiguous string holding the
   entire deck** — a second full pass over all character data. At peak the
   process holds: FEMData arrays **+** the line list **+** a copy of the
   list **+** the joined mega-string.

Reported field symptom: ~30 GB RSS on the largest models — far past a
workstation's free RAM. (Profiling caveat: 30 GB exceeds a pure ~6M-hex
*text* buffer estimate of ~2–3 GB; confirm with `tracemalloc` that the line
buffer is the dominant term and not FEMData arrays plus a still-resident
Gmsh kernel during emit — streaming fixes only the first.)

This is an artifact of the accumulate-then-join design, not anything
intrinsic to generating a correct deck. ADR 0061 (per-rank emission)
explicitly named this as the separate follow-up: *"at 5–6M hexes the
Python-side line buffer is multi-GB. Streaming emit is a separate follow-up
item, deliberately not coupled to this ADR."*

ADR 0061 and this ADR fix **different ceilings** and compose:

- **Per-rank (0061)** fixes the *runtime* RAM — `source` slurps the file
  per rank, and Tcl's 2 GB string ceiling makes a monolithic deck
  impossible to `source` past ~8M hexes.
- **Streaming (this ADR)** fixes the *author-side* RAM at emit time.

Two facts make the fix unusually tractable for its impact:

- **The sink is already centralized.** All 125 `self._lines.append(...)`
  call sites in `TclEmitter` funnel through `_LineBuf.append` (which already
  intercepts every append to apply the partition indent). Redirecting the
  sink is a *one-class* change, not a 125-site edit.
- **No mid-stream backpatch.** Element/node tags are allocated *before*
  emit (`allocate_element_tags()`); the loops only append. The sole
  front-insert is `preamble()` → `self._lines.insert(0, ...)`, a header
  concern bounded to the first few comment lines.

## Decision

Introduce a **write-through sink** as an *opt-in* emit mode; the default
stays list-accumulation, byte-identical to today.

1. **Sink abstraction in `_LineBuf`.** `_LineBuf` gains an optional file
   handle. In list mode (default) `append` stores as today. In stream mode
   `append` writes `indent + line + "\n"` straight to the handle and stores
   nothing. The partition-indent logic is unchanged (it already lives in
   `append`). All 125 call sites are captured for free.

2. **Header ordering.** In stream mode the banner + any `preamble()` lines
   are buffered until the first body line, then flushed once — preserving
   the leading-comment contract without a retroactive `insert(0, ...)`.

3. **Per-rank streaming via live sink routing (the production path).**
   `partition_open(K)` opens / switches the active sink to
   `ranks/rank<K>_<seq>.tcl`; `partition_close()` closes it and returns the
   active sink to the driver. Global / sequential lines (materials,
   sections, transforms, timeSeries, damping, stage skeleton, recorders)
   stream to the driver sink; per-rank bodies stream to the fragment file.
   This replaces `_write_per_rank_tcl`'s post-hoc buffer slicing with live
   routing — the spans (`_partition_spans`) become unnecessary in stream
   mode. **This is the only path that solves the 30 GB case on partitioned
   production models** (monolithic streaming alone still buffers nothing,
   but a partitioned model that needs per-rank would otherwise rebuild the
   full buffer to slice it).

4. **Atomic write.** Stream to `path.tmp` (and `ranks/*.tcl.tmp`) and
   `os.replace` on clean completion, so a mid-emit exception never leaves a
   half-written deck where list mode was all-or-nothing.

5. **Scope and guards (v1).** Tcl-first, mirroring ADR 0061 (`py()`
   per-rank already raises). `stream=True` is orthogonal to the existing
   `per_rank=`/`split=` flags; `stream=True` automatically implies live
   fragment routing when the model is partitioned. List mode remains the
   default so the entire test suite and every `emitter.lines()` consumer
   (introspection, reassembly-parity tests, `LiveOpsEmitter`) are
   untouched. H5 emission is a separate axis (see Alternatives) and is the
   recommended path for the very largest models regardless.

## Difficulty assessment (the question this ADR exists to answer)

Two tiers, very different risk profiles.

### Tier 1 — Quick win (hours, ~zero risk)

Replace the write-time `f.write("\n".join(emitter.lines()) + "\n")` (two
sites: tcl + py) with `f.writelines(l + "\n" for l in emitter._lines)`.
This kills the `list()` copy **and** the joined mega-string — eliminating
roughly the larger half of peak emit memory — while still holding the line
list. No architectural change, no behavior change, byte-identical output.
Likely takes 30 GB toward ~10 GB. Add one `tracemalloc` regression test.

### Tier 2 — Full streaming solution (moderate, bounded)

Not *hard* — the centralized sink and the already-live emit loop remove
what would normally be the expensive parts — but "doing it right" carries
five concrete obligations, in rough order of effort:

- **(a) Dual-mode `_LineBuf` + header buffering.** Small, well-contained;
  the indent logic already proves the interception point works.
- **(b) Per-rank live sink routing.** The crux and the only genuine new
  design. Rework `_write_per_rank_tcl` from "slice the buffer" to "the emit
  already wrote the fragments; write the driver." Two sinks alternate under
  `partition_open`/`partition_close`. ADR 0061 already separated
  driver-global from per-rank-body content, so this is *re-timing* an
  existing split (post-hoc → live), not inventing one.
- **(c) Verification strategy.** Stream output must be byte-identical to
  list output for the same model (a cheap, exhaustive text diff). Per-rank's
  existing "reassembly parity" test must gain a stream-vs-list equivalence
  check.
- **(d) Atomic temp+rename** across the driver and every fragment file.
- **(e) py emitter + `split=` axis.** Mirror or explicitly scope out for
  v1 (HPC path is Tcl).

**Blast radius:** concentrated in two files — `emitter/tcl.py` (the sink)
and `apesees.py` (the two writers + per-rank routing). Estimate: a focused
~2–4 day change, with (b) carrying essentially all the risk.

### The scoping fork that decides quick-vs-full

**Is the 30 GB model partitioned or single-domain?**

- **Single-domain huge model:** monolithic streaming alone fully solves it.
  Tier 1 helps immediately; Tier 2 without per-rank routing finishes it.
- **Partitioned production model** (the P=8/f=8 ≈ 5.6M-hex case ADR 0061
  targets): the quick fix and monolithic streaming **do not fully solve
  it** — a partitioned deck still materializes the buffer to slice into
  ranks. Only Decision §3 (live fragment routing) gets to constant emit
  memory. This case needs Tier 2 with §3.

## Alternatives considered

- **Quick fix only (Tier 1), stop there.** Viable if the failing models are
  single-domain and ~10 GB peak is acceptable on the authoring machine.
  Rejected as a *complete* answer because it does not reach constant memory
  and does nothing for the partitioned production path. Strong as a *first
  step*.
- **Use H5 emission for big models instead of streaming text.** Already the
  right answer for the largest models: binary, compact, no Tcl 2 GB ceiling.
  Its emitter buffers `list[int]` node/element records that could write the
  FEMData numpy arrays straight into chunked/resizable datasets — an
  independent, smaller memory win. Complementary, not a substitute: the HPC
  Tcl path still exists and still needs a bounded-memory author.
- **Stream the monolithic deck only; leave per-rank slicing as-is.**
  Rejected as the production fix — the partitioned model that motivates the
  whole concern is exactly the one that would rebuild the full buffer to
  slice. Acceptable only if production is never partitioned.
- **mmap / spill the line list to a temp file.** Rejected — reinvents a
  buffered file write with extra machinery; the sink redirection is simpler
  and the OS write buffer already does the batching.

## Consequences

- **Memory:** stream mode drops peak emit memory to ~the FEMData arrays +
  an OS write buffer (tens of KB), independent of model size. Tier 1 alone
  removes the join + list-copy term.
- **Performance:** total bytes written are unchanged; the join allocation
  and the list copy (both O(total-text) memcpy passes) are *eliminated*, so
  streaming is the same or marginally **faster**. Python file objects
  buffer writes — per-line `f.write` is not a syscall per line. No expected
  loss.
- **What stream mode gives up:** in-memory introspection
  (`emitter.lines()`) for the streamed path, and list mode's implicit
  all-or-nothing atomicity (restored by §4). Both mitigated by keeping list
  mode the default.
- **Per-rank simplification:** in stream mode `_partition_spans` slicing is
  retired in favor of live routing — net code reduction on that path.
- **HPC layer untouched** — the driver remains the deck entry point;
  `push_dir` / `Cluster.submit` / `run_remote` are unaffected (same as ADR
  0061 §4).
- **Verification surface:** (a) stream-vs-list byte-identity on a fixture
  model; (b) per-rank stream-vs-list fragment equivalence; (c) a
  `tracemalloc` ceiling test asserting stream-mode peak is O(1) in element
  count; (d) live `mpiexec -n 2/4` parity for streamed per-rank decks
  (reuse the `test_emit_partitioned_*` template).

## Implementation note (2026-06-18) — profiling reframed the problem; `mass_from_model` shipped

The reported ~30 GB was **in-RAM emit footprint** (the local run was killed
mid-emit; no 30 GB file exists), measured on the real LOH.1 P10 model:
**6.71M hex / 7.0M nodes, partitioned (np=256) + staged.** A scaled
loh1-mirror profile (`add_plane_wave_box` → `masses.volume` per layer →
`stdBrick` per PG → per-node `ops.mass` loop → `ops.tcl`) at 0.1M/0.5M
elements gave stable linear coefficients; extrapolated to 6.7M/7M and
cross-checked against the kill point: the dominant terms are the **build-time
Python object graph**, not the deck text. Roughly: emit transient
(`element_plan` tuples + ~54M boxed connectivity ints + line buffer) ~7.5 GB;
FEMData incl. **7M per-node kernel `MassRecord`s** ~5.4 GB; **bridge mass
double-store** (the `for m in fem.nodes.masses: ops.mass(...)` loop building a
*second* 7M-object list) ~1.3 GB; ~1.9× heap-fragmentation/RSS multiplier on
top. **Tier 1 (streaming write) addresses only the ~1.8 GB join string — a
real but minor slice of this total.** The object graph is the event.

**Shipped this slice: `ops.mass_from_model()`** (bridge entry + `BuiltModel`
flag). It streams per-node masses straight from `fem.nodes.masses` at emit —
replacing the user's 7M-call `ops.mass` loop — so the **bridge double-store is
never built** (~1.3 GB + 7M objects). Wired through all three mass emit paths
(flat `_emit_masses`, partitioned per-rank `_emit_masses_partitioned`, and the
**bucketed partitioned path** that the real LOH.1 run uses — primary-owner
bucketed, additive-under-MP-safe). Byte-identical to the explicit per-node
loop (same `fit_dof_vector` + `emitter.mass` in `fem.nodes.masses` order),
proven by flat **and** partitioned equivalence tests. Two fail-loud guards:
overlap with explicit `ops.mass` (MP double-count) and H5-emitter rejection
(masses already persist in `model.h5`; re-streaming would rebuild the 7M
materialization). Verified: 4 new tests + full `tests/opensees` (4636 passed),
ruff/mypy clean.

**Deferred, with rationale:**
- **Lazy `element_plan` + unboxed connectivity** (the larger ~7.5 GB term):
  the lazy generator design is **flat-path-only** — staged / split /
  partitioned all still materialize via `allocate_element_tags`. Because
  LOH.1 is partitioned+staged, that slice would **not** help it; a
  partitioned/per-rank-aware streaming design is a separate, larger item.
- **numpy-backed `from_h5` mass read** (kills the 7M `MassRecord`
  rehydration, ~2 GB): independent; helps the `emit_repro.py` load-and-profile
  path, **not** the in-session meshing run (where `g.masses.volume` builds the
  7M records). Sequenced after the in-session resolve is confirmed a hotspot.
