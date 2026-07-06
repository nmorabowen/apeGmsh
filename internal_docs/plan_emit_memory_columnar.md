# Plan — emit memory for extremely large models (ADR 0065 v2: Routes B / C / A)

**Status:** Proposed (2026-07-06). Companion to ADR 0065 (streaming deck emission),
whose Implementation note reframed the problem: **the deck text is not the event —
the build-time Python object graph is.** This plan turns the remaining ledger into
shippable slices. Route E (fork-side binary bulk loader) is deliberately out of
scope here — it is a fork ADR of its own; trigger condition at the end.

## Goal and budget

Author an ~11M-node / ~10M-hex **partitioned + staged** deck on a 27.6 GB desktop
with peak emit memory **< 6 GB**, with headroom to ~50M nodes. Reference failing
case (ADR 0065): LOH.1 P10 — 6.71M hex / 7.0M nodes, np=256, staged — killed at
~28 GB RSS mid-emit.

### Ledger (LOH.1 scale, post-shipped-work)

| Term | Est. | Route |
|---|---|---|
| `element_plan` per-element tuples + ~54M boxed connectivity ints | ~4–6 GB | **B** |
| `fem_eid_to_ops_tag` dict (6.7M entries) + `element_owner` dict | ~0.7–1.3 GB | **B** |
| 7M non-slotted `MassRecord` objects in FEMData | ~3–5 GB | **C** |
| Line buffer (~13–20M `str`), pinned by per-rank span slicing | ~2–4 GB | **A** |
| Heap-fragmentation multiplier ×~1.9 | (falls out) | B+C+A |
| Bridge mass double-store | — | ✅ shipped (`ops.mass_from_model`) |
| Write-time join mega-string + list copy | — | ✅ shipped (Tier-1 `write_to`) |

## M0 — re-baseline first (half a day, do before spending)

The shipped work (Tier 1, `mass_from_model`, per-class ndf inference, PG memo,
partitioned pre-bucketing) changed the ledger since the June profile. Extend
`tests/benchmarks/emit_throughput_profile.py` with `tracemalloc` snapshots at
0.1M / 0.5M hexes (flat, 64-rank, staged-partitioned), fit the linear
coefficients, extrapolate to 6.7M. **Gate:** confirm the element plan is the top
term. This is exactly the demand-gate ADR 0065 asked for, and it becomes the
regression harness every slice below must move.

## Verified facts this plan rests on

1. **Per-spec element tags are contiguous.** `TagAllocator`
   (`opensees/_internal/tag_allocator.py:22`) is per-kind sequential 1-based;
   `allocate_element_tags` (`opensees/_internal/build.py:4420`) allocates all
   element tags in one early pass, spec by spec; MP-constraint elements allocate
   later. ⇒ a spec's tags are `arange(tag_start, tag_start + n)`.
2. `expand_pg_to_elements` (`build.py:1491`) materializes `[(eid, conn-tuple)]`
   Python lists from numpy group arrays, memoised per snapshot
   (WeakKeyDictionary); `expand_spec_to_elements` (`build.py:1622`) adds the
   node-pair synthetic `(MISSING_FEM_ELEMENT_ID, (i, j))`.
3. `bucket_pre_allocated_by_rank` (`build.py:4718`) builds per-rank
   dict-of-lists via `element_owner` dict lookups.
4. `fem_eid_to_ops_tag` is built as a dict comprehension over the plan
   (`apesees.py:1247/1309/1574/2261/2332`); consumers: `_emit_rayleigh` /
   `_emit_damping_attach` region `-ele` lists (`apesees.py:1369–1370`), point
   host lookups (`apesees.py:1936`, MP couplings), staged `remove_element`
   translation (`apesees.py:2519–2667`).
5. Group connectivity in FEMData is already numpy (`mesh/_group_set.py:252`,
   `ndarray(E, npe)`; mixed-type groups are object-dtype padded with −1 —
   an emit-side per-row slice concern).
6. Mass storage in `model.h5` is **already columnar** (`/masses` compound,
   `mesh/_femdata_h5_io.py:2084`); `_read_masses` (`:3608`) eagerly boxes 7M
   `MassRecord`s at `from_h5`. `MassRecord` (`_kernel/records/_masses.py:18`)
   is a non-slotted dataclass (~400–700 B/node). `MassSet` extends
   `_RecordSetBase` (`_kernel/record_sets.py:65`) = plain `list[MassRecord]`;
   `_with_record` is an O(N) copy. Producer sites: `MassesComposite.py:799`,
   `_kernel/resolvers/_mass_resolver.py:95`, `_chain_phase_router.py:174`,
   `_femdata_h5_io.py:3623`, `h5_reader.py:862`, `apesees.py:5815/8242`,
   `opensees_model.py:942`.
7. `_write_per_rank_tcl` (`apesees.py:592`) slices a **full copy** of the line
   buffer (`emitter.lines()` at `apesees.py:6672`) by `PartitionSpan`s and
   strips the 4-space indent per body line. Split emit also copies
   (`:6684/6748`). **Waste:** `len(emitter.lines())` at
   `apesees.py:1619/1623/1647/1648` makes a full list copy just for a length.
8. `_LineBuf` mutation surfaces beyond `append`: `insert(0)` (preamble),
   `indent` toggling (analyze/strategy/partition), `len` reads (span/module
   recording), slicing (per-rank/split writers).

---

## Route B — columnar element plan (P1, the dominant term)

**Idea:** a plan entry becomes `(spec, eids: int64[], conn: int64[N,k] | rows-ref,
tag_start: int)`. Connectivity is read as views of the FEMData group arrays at
emit time; tags are `tag_start + i` by position. Rank bucketing becomes an
`owner_by_pos: int32[]` array + boolean masks/argsort instead of dict-of-lists.
`fem_eid_to_ops_tag` becomes a small `FemToOpsTagMap` object: per-spec
`(sorted eids, permutation, tag_start)`, `searchsorted` point lookups, a
vectorized `translate(array)` for region lists. Works identically for flat,
partitioned, and staged+partitioned — no flat-only shortcut.

**Slices (each byte-identity-gated on the 100 MB deck diff fixtures):**

- **B1 — plan representation + compatibility iterator.** New columnar
  `ElementPlan` carrying the arrays, plus a lazy `rows()` iterator yielding
  `(eid, conn_tuple, tag)` so *unconverted* consumers keep working (transient
  tuples per row, not resident). `allocate_element_tags` returns the columnar
  form; peak drops immediately because the resident tuple graph is gone.
  ~2–3 days.
- **B2 — convert the emit loops.** Flat `emit_element_spec`, partitioned
  `emit_element_spec_partitioned` (`build.py:4740`), staged per-rank blocks:
  iterate the arrays directly (per-row f-string emit off array scalars —
  `int()` at the line boundary keeps output byte-identical). Rank bucketing
  via `bucket_pre_allocated_by_rank` → mask/argsort form; `element_owner`
  and `compute_stage_ownership` outputs move to positional arrays.
  ~3–4 days.
- **B3 — retire the tag dict.** `FemToOpsTagMap` replaces the dict
  comprehensions at all five build sites; consumers: vectorized translate for
  rayleigh/damping region `-ele` lists and staged `remove_element`; point
  `.get()` for coupling hosts. Fail-loud on unknown eid preserved. ~2 days.
- **B4 — array-native PG fan-out.** `expand_pg_to_elements` memo caches
  `(eids, conn)` arrays instead of tuple lists (node fan-outs too, where the
  consumers allow); `expand_spec_to_elements` wraps node-pair specs as 1-row
  arrays with the `MISSING_FEM_ELEMENT_ID` sentinel. Validators/ndf-inference
  already went numpy — align their inputs. ~2 days.

**Edge cases:** node-pair synthetic elements (1-row arrays, sentinel eid never
enters the tag map); mixed-type groups (object-dtype padded conn — emit slices
each row to its valid width; keep the boxed path for these rare groups if
needed, they are never the 6.7M-hex band); ghost/canonical-rank couplings
(host translation through the map API only); `OpenSeesModel.build()` deck-replay
(shares the same build path — converted for free, verify with replay tests);
same-snapshot re-emit memoisation must key the *array* cache identically.

**Expected win:** ~150 B → ~16–24 B per element resident ⇒ ~4–6 GB → ~0.2 GB
at LOH.1 scale, plus the ~1 GB of dicts.

## Route C — columnar mass records (P2)

**Idea:** back `MassSet` with `(node_ids: int64[N], mass: float64[N,6],
names: dict[int,str] sparse)`; iteration yields lightweight views (or constructs
transient `MassRecord`s on the fly — records are treated read-only today;
verify no in-place mutation in the inventory above). `from_h5` adopts the
compound-dataset columns zero-copy — the exact values written, so float `repr`
in decks is unchanged. The in-session resolver (`_mass_resolver.py:95`)
accumulates per-node contributions via index staging + `np.add.at` (or
sort+`reduceat`) instead of building 7M objects. Compose/tag-rewrite
(`tag_rewrite_spec`) becomes a vectorized `node_id` remap on the array.
`_with_record` (small-N convenience) stays object-path — it converts to
columnar at set construction.

**Slices:** C1 columnar store + view iteration (+ inventory-driven API
compatibility), C2 producers (resolver + `from_h5`), C3 emit paths read columns
directly (`mass_from_model` order is already positional). ~4–6 days total.

**Skip-condition / interaction:** explicit runs using fork `LadrunoBrick
-lumped` never carry the nodal-mass band — Route C matters for consistent-mass,
implicit, and added-mass models. **Follow-on (scope only):** `NodalLoadSet` /
sp records share the `_RecordSetBase` pattern — a DRM/plane-wave box with
millions of load records hits the same wall; treat as an optional C4 after
measuring.

## Route A — Tier-2 streaming sink (P3, per ADR 0065 Decision §1–§5)

- **A0 (immediate micro-PR, hours):** `emitter.line_count()`; replace the four
  `len(emitter.lines())` full-copy call sites. Also pass `emitter._lines`
  (documented read-only) to `_write_per_rank_tcl`/`_write_split_*` instead of
  the `lines()` copy.
- **A1 — dual-mode `_LineBuf`** (write-through file sink; header buffered until
  first body line; `insert(0)` only legal pre-body), `stream=True` on
  `ops.tcl`. List mode stays default; `lines()` in stream mode fails loud.
- **A2 — live per-rank routing.** `partition_open(K)` switches the active sink
  to `ranks/rank<K>_<seq>.tcl`; `partition_close()` returns to the driver sink
  and writes the source-guard line. Must reproduce `_write_per_rank_tcl`'s
  naming and content **byte-identically** — note fragments are written
  *unindented* today (indent stripped post-hoc), so the live route suppresses
  the partition indent instead of writing-then-stripping. Spans retire in
  stream mode.
- **A3 — atomic writes** (`.tmp` + `os.replace` for driver + fragments;
  mid-emit exception leaves no half-deck) **+ verification**: stream-vs-list
  byte-identity on flat / per-rank / staged-partitioned fixtures; a
  `tracemalloc` ceiling test asserting stream-mode peak is O(1) in element
  count; `mpiexec -n 2/4` parity reuse.

~3–5 days total. `split=` + `stream` mutually exclusive v1; `py()` emitter out
of scope v1 (HPC path is Tcl) — document.

## Sequencing

```
M0 baseline ─→ B1 ─→ B2 ─→ B3 ─→ B4        (dominant term)
A0 (anytime, hours)
C1 ─→ C2 ─→ C3                              (parallel-safe with B: disjoint files)
A1 ─→ A2 ─→ A3                              (after B2 — the loops churn once, then the sink)
```

B before A because B rewrites the emit loops A's sink sits under — landing A
first means rebasing its routing tests across B2's churn. C is independent
(`_kernel/` + h5 io vs `opensees/_internal/`).

**Acceptance:** every slice byte-identical on the deck-diff fixtures + full
`tests/opensees` + ruff/mypy; M0 harness re-run per slice showing the intended
term drop; final gate = staged-partitioned 0.5M-hex profile extrapolating to
**< 6 GB at 11M nodes**.

## Route E pointer (not this plan)

Fork-side binary bulk loader (`model.h5` → C++ node/element/mass creation, deck
shrinks to a driver): kills desktop formatting, the Tcl 2 GB `source` ceiling,
and cluster parse time in one move. Write as a **fork ADR** when (a) models pass
~30–50M nodes, or (b) cluster-side parse time dominates job wall-clock, or
(c) desktop emit *time* (not memory) becomes the bottleneck after B/C/A.

## What this does and does not do to RAM

After B+C+A, peak emit ≈ FEMData numpy arrays + O(1) buffers: ~100 B/node ⇒
~1.1 GB at 11M nodes, ~5 GB at 50M. Emit stops being the RAM bottleneck. Still
linear in N elsewhere: **Gmsh meshing itself** (session-side, upstream of
emit), the FEMData arrays, and **cluster-side solve RAM** (per-rank decks
already bound parse RAM; Route E removes deck text entirely).
