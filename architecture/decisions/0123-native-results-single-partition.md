# ADR 0123 — Native results stay single-partition; no native PARTITION_REDUCTION

**Status:** Proposed (2026-10-22; for ratification on #2450)

**Owner:** nmora

**Evidence:** two architect briefs written independently, one by Opus and
one by Fable, from one card; neither read the other. They are kept in T's
notes folder as `native_partition_reduction/brief_opus.md` and
`native_partition_reduction/brief_fable.md`, beside the card
(`architect_card.md`). Both recommended option A below. This ADR reconciles
them; where they differed, it says so under *Open points for the
maintainer*.

**Applies** [ADR 0122](0122-ladruno-canonical-results-format.md) D2 (native
`results.h5` is frozen, then deleted) to the partition axis. It does not
amend ADR 0122. It keeps the #2360 ruling (agreeing cross-partition copies
read, disagreeing ones refuse) as written. It lapses with native results at
ADR 0122 slice 10.

## Context

Native `results.h5` has a partition axis,
`/stages/{sid}/partitions/{pid}` (`results/schema/_native.py::partition_path`),
but no rule for how a node's copies in several partitions combine. MPCO and
`.ladruno` have one, `PARTITION_REDUCTION` in {`NONE`, `SUM`,
`UNSUPPORTED`} (`readers/_mpco_translation.py::PARTITION_REDUCTIONS`):

- `.ladruno` stores it as an attribute on each `ON_NODES` result group
  (`_ladruno.py::LadrunoReader.node_partition_reduction`).
- MPCO has no attribute, so the reader derives it from the group name
  (`partition_reduction_from_name`): `REACTION*`, `UNBALANCED*` and
  `RAYLEIGH*` sum.
- Both multi-part readers reduce at read time
  (`_mpco_multi.py::_merge_node_slabs`, `_ladruno_multi.py`), and refuse
  an element that two parts hold
  (`_mpco_multi.py::_refuse_element_in_two_partitions`).
- The viewer's part-set reader mirrors this, and refuses a component
  whose attribute any part lacks (`apeGmshViewer/src/reader/ladrunoParts.ts::reductionOf`).

#2450 called the native reduction "unreachable today". Both architects
surveyed `main` at 94b40bcf5, and found that the axis has no producer:

- **Every native writer in `src/` writes the literal `partition_0`.** The
  callers are `transcoders/_recorder.py::RecorderTranscoder`,
  `opensees/analysis/modal.py` (eigen, parallel modal, and
  `capture_live_modes`), `footfall_result.py::FootfallResult.to_results`,
  `results/demo.py` and `readers/_out_fiber_io.py`. DomainCapture wrote one
  partition too, and was deleted at ADR 0122 slice 7.
- **Multi-rank inputs are reduced or refused before a native write.**
  `ParallelModalResult._merge_partitioned_shapes` gathers the rank
  harvests into one partition. `opensees/_results_route.py::_classify`
  refuses text recorders from a partitioned deck ("Record a partitioned run
  with ops.recorder.MPCO").
- **Only a third-party caller of the public `NativeWriter` API** can write
  a second partition. In the repo, only the #2360 tests do.

What the readers do with such a file today:

- **Python** (`readers/_native.py`, after #2360): a node id repeated
  within one partition refuses at open (`NativeRepeatedNodeId`). A node
  several partitions hold reads only when its copies are bit-for-bit equal
  over the stage; otherwise every read of that component raises
  `NativePartitionCopiesDisagree`. Element rows are concatenated, with no
  check for an element in two partitions (#2363).
- **Viewer** (`apeGmshViewer/src/reader/results.ts`): a stage with other
  than one partition is refused as `unsupported`.
- **The schema doc** (`architecture/h5-schema.md`, "Partitions") still said
  "first occurrence wins", stale since #2360. This PR corrects it.

ADR 0122 D2 governs the format: native "gets no new writer", stays
readable until slice 10, and is then deleted behind a human gate. On `main`,
slice 7 (DomainCapture deletion) and slice 9a (eigen and parallel-modal
`.ladruno` writers) are done; 9b, 9c and 10 have not started.

## Decision

### D1 — Native results hold at most one partition per stage

Until ADR 0122 slice 10 deletes the format, a native `results.h5` stage holds
exactly one partition. The `partitions/` level stays in the layout only
because files on disk carry it. It is not a feature, and it is not removed
before slice 10: that would be a major bump of a frozen zone, and slice 10
removes the reader whole.

### D2 — No `PARTITION_REDUCTION` in native

No native writer writes a `PARTITION_REDUCTION` attribute, no native reader
(Python or TypeScript) reduces, and the results zone is not bumped for one.
It stays at 1.2.0 with floor 1.0.0 (ADR 0113). The native reader never sums
by component name: that would be guessing. Its only move on a shared node is
to read agreeing copies once or refuse.

### D3 — Partitioned runs reduce where they land

A partitioned run reaches a reader as a `.ladruno` part set (fork) or an MPCO
part set (stock, converted to `.ladruno` under ADR 0122 slice 6). Their
`PARTITION_REDUCTION` vocabulary is the one partition rule in apeGmsh.

### D4 — Legacy multi-partition files keep the #2360 rule

A native file that already holds several partitions per stage still opens.
Agreeing node copies read once; one disagreeing node makes every read of
that component raise `NativePartitionCopiesDisagree`. This ADR does not
reopen that ruling.

### D5 — The writer refuses a second partition

`NativeWriter._require_partition`, which every `write_*` passes through,
refuses a partition id that differs from the one a stage already has. It
raises a `ValueError` subclass before creating any group, and names the
file, the stage, the existing id and the refused id. The rule is about
count, not name: a stage whose only partition is `partition_3` is legal,
which matches the viewer's per-stage gate. This is the one change to a
frozen writer; see open point 1.

### D6 — The viewer is unchanged

`results.ts` keeps refusing a native stage that does not hold exactly one
partition, as `unsupported`. It never merges. Its header comment calling
merging "a follow-up" is withdrawn in slice 2.

## Options considered

| | Option | What it wins | What it costs |
|---|---|---|---|
| **A** | **Single partition by contract** (this ADR) | Matches D2 and every producer. No schema bump, no viewer change, about 60 lines, all in files slice 10 deletes. Reversible: lifting a refusal is additive | Narrows a public writer API. Legacy third-party files are read by agreement, never summed. The #2360 tests move their second partition to raw h5py |
| A0 | Correct the docs only | Cheapest | "Single partition" stays an unchecked convention, so a writer can still hand the viewer a file it refuses |
| B | A `PARTITION_REDUCTION` attribute on native node components (results 1.3.0); Python and TS readers reduce; an absent attribute means agree-or-refuse | One vocabulary across three formats | A feature with no producer on a format that has a deletion date. Amends ADR 0122 D2. Every older apeGmsh refuses a 1.3.0 file. Two reducers and their oracles, with nothing to exercise them. **Irreversible:** 1.3.0 files need those readers until slice 10 |
| C | Reduce at write time (rank 0 or an offline merge writes one reduced partition) | Readers never reduce | Nothing to merge: parallel modal already gathers, and the part sets already reduce on read. A multi-rank native writer contradicts D2 |
| D | Per-rank native part files (`*.part-N.results.h5`) | Mirrors `.ladruno` part sets | A new writer, set reader and viewer stitcher for a dying format. Duplicates ADR 0122 slice 8 (`ladrunoParts.ts`, `_ladruno_multi.py`) |

## Consequences

**Won.** The partition rule has one home, the part-set formats, which
already carry and test it. Native stays frozen, as ADR 0122 D2 rules: no
schema bump, no new reader code in Python or TypeScript, and a writer
guard that keeps any new file inside what the viewer opens. Every change
lands in files slice 10 deletes.

**Lost.** A third-party script that wrote several partitions through
`NativeWriter` stops working (no such caller is known). A legacy
multi-partition file with per-rank partial reactions is refused rather than
summed. Partial copies that happen to be equal, such as a support on an
interface where both ranks hold half of a symmetric reaction, still read as
one rank's share (open point 2).

**Schema impact.** None. The results zone stays at 1.2.0, floor 1.0.0.
`architecture/h5-schema.md` "Partitions" is rewritten to state the contract
and the #2360 rule (this PR).

## Slice plan

| Slice | Chain | Work | Owned files | Done when |
|---|---|---|---|---|
| 1 | T | This ADR, its index row, and the `h5-schema.md` "Partitions" correction | `architecture/decisions/0123-native-results-single-partition.md`, `architecture/decisions/README.md`, `architecture/h5-schema.md` | The maintainer accepts |
| 2 | B | D5's writer refusal; a raw-h5py legacy builder that the #2360 tests and `tests/test_results_native_partitions.py` move onto; a new single-partition test; the `results.ts` header comment; a changelog fragment | `src/apeGmsh/results/writers/_native.py`, `tests/results/_native_legacy.py` (new), `tests/results/test_native_repeated_node_ids.py`, `tests/test_results_native_partitions.py`, `tests/results/test_native_single_partition.py` (new), `apeGmshViewer/src/reader/results.ts` (comment only), `changelog.d/2450-native-single-partition.md` | Main refuses a second partition by name, and the legacy-read tests stay green. It shares `writers/_native.py` with #2361: fold it into that card or queue it after |
| 3 | B | A test for `_classify`'s partitioned-deck refusal, which nothing tests today | `tests/opensees/unit/test_results_route.py` | The test lands and fails when the refusal is reverted |

The element-in-two-partitions refusal (an element id held by two partitions
of one stage refuses, as MPCO does) belongs to #2363, which already owns the
native element-side repeat check.

### Oracles

- **Revert-fail.** With D5's guard removed, the second-partition test fails.
  After a refused write, the file still holds one partition and opens.
- **Count, not name.** A guard that refuses any id other than `partition_0`
  must fail a test whose only partition is `partition_3`.
- **Legacy reads on raw-h5py fixtures** (`NativeWriter` now refuses them):
  `reaction_force_x` copies of 3.0 and 4.0 raise, and never read 7.0;
  equal displacement copies read once; a three-partition stitch returns every
  column.
- **Single-partition round trip.** The `make_demo_results` file reads
  `np.array_equal` before and after slice 2, and carries no
  `PARTITION_REDUCTION` attribute. Versions come from
  `tests/fixtures/schema.py`, never a literal (the `schema-literal` rule).
- **Viewer.** The existing `results.test.ts` case ("several partitions are
  refused") stays as it is.

## What not to build

- No `PARTITION_REDUCTION` in native, no results 1.3.0, and no native
  reducer in Python or TypeScript.
- No multi-rank native writer, no rank-0 merge, no `.part-N.results.h5`, and
  no conversion from `.ladruno` or MPCO to native (ADR 0122 D1).
- No removal of the `partitions/` level before slice 10.
- No `merge_partitions=` option on `Results.from_native`.

## Open points for the maintainer

1. **Does D2's "no new writer" admit D5's guard?** Both architects added the
   guard; Fable asked whether a frozen writer may change at all. The
   recommendation is yes: the guard narrows, it does not grow the format. If
   "frozen" means untouched, drop D5 and slice 2's writer change, and the
   contract is held by the doc and the readers only.
2. **Equal partial copies.** Should the native reader refuse `reaction_*`
   and the unbalanced `force_*` / `moment_*` that several partitions hold
   even when the copies agree? Opus leans yes, inside #2363. It amends the
   #2360 ruling for those components, and no file in the repo is affected.
   Fable keeps #2360 as written. The recommendation is to keep #2360: no
   producer writes such a file.
3. **Refuse now, or warn for one release?** No third-party caller is known.
   The recommendation is to refuse now.
4. **If slice 10 is carded in the next batch**, slices 2 and 3 can be skipped
   and this ADR's slice 1 alone closes #2450.

## Adjacent findings (separate cards, not this ADR's scope)

- **Converted stock part sets lack `PARTITION_REDUCTION`.** `convert_mpco`
  converts `.part-N.mpco` sets to `.part-N.ladruno` sets
  (`readers/_ladruno_from_mpco.py`), and neither it nor
  `readers/_ladruno_writer.py` writes the attribute. The Python reader falls
  back to the group name and sums `REACTION*`; the viewer's
  `ladrunoParts.ts::reductionOf` refuses every nodal component that lacks
  it. A stock MPI run converted at `results()` (ADR 0122 slice 5) is therefore
  not viewable as a part set, and the two readers disagree on the same file.
  This is where a partition reduction is actually missing today. A chain-X card
  can stamp the attribute in `convert_mpco` (from
  `partition_reduction_from_name`), and in the 9c `.out` writer.
- **`capture_live_modes` writes native.** It arrived with #2342 (closed 2026-10-20), after
  ADR 0122 D2 was accepted. Whether it moves to `.ladruno` belongs to
  slice 9b.
