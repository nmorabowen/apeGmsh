# Golden emit corpus

This corpus is the oracle for pure moves of the OpenSees bridge's emit path
(panel plan `internal_docs/plan_expert_panel_2026-09.md`, "The proof
regime"). A refactor that claims "no behaviour change" must leave every
golden here unchanged: every byte except float literals, which must agree
within a last-ulp tolerance (see "Floats" below). It replaces the single deck
`tests/opensees/parity/partitioned_mixed_npe.golden.tcl` as that oracle;
that deck and its test stay where they are.

## The grid

A cell is one fixture x one mode x one output: 7 x 5 x 3 = 105 cells.
`MANIFEST.json` lists every cell. A cell is either `golden` or
`n/a: <reason>`. Today there are 86 golden cells and 19 n/a cells.
`builder.applicability()` is the single source of the n/a reasons, and
`test_manifest_is_the_grid` fails when the MANIFEST drifts from it.

**Fixtures.** The fixtures are the 7 `make_*` factories in
`tests/opensees/fixtures/fem_stub.py`. `builder.FIXTURES` drives each one
with a small model:

| Fixture | Model |
|---|---|
| `two_node_beam`, `two_column_frame`, `two_module_frame`, `two_column_frame_partitioned` | 3-D `elasticBeamColumn` on `Cols` with `geomTransf Linear` (vecxz 1 0 0); `Base` fixed; load on `Top` |
| `two_column_frame_with_labels_and_selection` | as above; with recorders, it also declares `ops.recorder.declare(label="east_column")` and `(selection="upper_band")`, which pins the label and selection resolvers |
| `axial_chain_partitioned` | 2-D `Truss` over `ElasticMaterial` on `Chain`; `Base` fixed; the transverse DOF of every mass fixed; load on `Masses` |
| `arch_with_orientation_fan_out` | 3-D `elasticBeamColumn` on `Arch` under `Spherical` orientation, so one `geomTransf` is emitted per distinct vecxz; `Springing` fixed; load on `Crown` |

**Modes.** The mode chooses the model shape and the emit call:

| Mode | Model | Tcl call |
|---|---|---|
| `flat` | global load pattern and global analysis chain | `ops.tcl(path, flat=True)` |
| `partitioned` | as `flat`, on a partitioned fixture | `ops.tcl(path)` |
| `staged` | two stages (`gravity`, `push`), each with its own stage-scoped pattern, analysis chain and `run` | `ops.tcl(path, flat=True)` |
| `staged_partitioned` | as `staged`, on a partitioned fixture | `ops.tcl(path)` |
| `per_rank` | as `staged_partitioned`, so the fragments cover the base block and both stage blocks | `ops.tcl(path, per_rank=True)` |

A fixture that its factory returns unpartitioned enters the partitioned
modes through `FEMStub.set_partitions` with an honest 2-rank split: one
column per rank for the frames (one compose module per rank for
`two_module_frame`), and elements {1, 2} | {3} for the arch, which share
node 3. The partitioned modes declare `ParallelRCM` and `Mumps`, the pair
OpenSeesMP can run. The serial modes declare `RCM` and `UmfPack`. The
`flat` and `staged` cells of an already-partitioned fixture keep the serial
pair, so the bridge's advisory `OpenSeesAutoEmitWarning` fires there. It is
silenced by default in `pyproject.toml`.

**Outputs.**

- `tcl` is the Tcl deck from the call in the mode table.
- `py` is `ops.py(path)`.
- `recording` is the same Tcl deck as `tcl`, with recorders declared.

The `recording` cells use the smallest recorder set that pins node,
element and MPCO-region emission:

- `recorder Node` on the loaded group (`disp`, DOFs 1 2);
- `recorder Element` on the element group (`globalForce`);
- `recorder mpco` with `nodes_pg` and `elements_pg`, which makes the bridge
  emit a `region`. In the partitioned modes that region is the
  partition-aware MPCO plan (ADR 0027 INV-4): one `-R` recorder line
  globally, and one rank-owned `region` per rank.

**Why cells are n/a.** `MANIFEST.json` holds the exact reason for each n/a
cell:

- `two_node_beam` has one element, so any 2-rank split leaves a rank that
  owns no element. All 9 of its cells in the partitioned, staged-partitioned
  and per-rank modes are n/a.
- `per_rank` x `py` for the other 6 fixtures: the per-rank fragment split
  is Tcl-only (6 cells).
- `flat` and `staged` x `py` for the two already-partitioned fixtures:
  `ops.py` has no `flat=` switch, so these would repeat the partitioned
  py deck (4 cells).

## Rebar cells

The grid has no reinforcement, so `g.rebar` gets its own cell family in
`rebar.py`: one fixture, `rebar_column_cage`, x two placements x two
outputs (`tcl`, `py`), four cells in all. They are listed under `rebar` in
`MANIFEST.json` and live in `rebar_cells/<fixture>/<placement>/<output>.golden`.

**Why it is not a grid row.** The grid is closed over the `make_*` factories
of `fem_stub.py` (`test_fixture_roster_is_every_make_factory`), and its five
modes vary the analysis shape of a stub. A rebar cage needs a real gmsh
session, and what it pins is the bar geometry and its coupling. A grid row
would be 13 of 15 cells n/a, plus a factory that is not a stub. The family
reuses the grid's normalise, mask and float-tolerance helpers, and regen
writes it.

**Model.** The X4-e harness (#1848):

- a 0.4 x 0.4 x 1.5 box (`Col`), with a `g.rebar.column` cage under
  `ACI318_seismic`: 3x3 #6 bars, #3 ties @ 0.25, cover 0.04;
- material `rebar` (`Steel02`), global mesh size 0.15, `ndm 3 ndf 3`, and
  the target pinned to stock, as in the grid.

| Placement | Coupling | Host mesh | The Tcl deck holds |
|---|---|---|---|
| `conformal` | `coupling="conformal", emit_elements=True`; no cross-ties | Delaunay tets, with the bars embedded so the host shares their nodes | 262 `CorotTruss` (the golden keeps the 270 bar nodes only) |
| `embedded` | `coupling="embedded", perfect=1e8, emit_elements=True`; with cross-ties | transfinite tets (`set_transfinite_box(size=0.2, recombine=False)`) | 366 `CorotTruss`, 400 `LadrunoEmbeddedRebar` |

`test_rebar_golden_exercises_both_emit_paths` pins those counts, so a regen
that silently dropped `emit_elements` or the ties cannot pass as a match.

**Determinism.**

- **The embedded host is structured.** Its nodes are algebraic, not a
  Delaunay outcome. Many bar nodes sit on a face that two tets share.
  `locate_point` keeps the first host, in centroid-distance order, that
  holds the node. In this model the chosen host is at least ~1% nearer
  than any other host that also holds the node, so a last-ulp change in the
  distances cannot flip the choice.
- **The conformal host is unstructured.** gmsh cannot embed curves in a
  transfinite volume, so its host-interior nodes are a Delaunay outcome.
  They differ between hosts: gmsh 4.15.0 on Windows against 4.15.2 on
  Linux CI agreed on nodes 1 to 434 and diverged after. The deck declares
  no host element, so nothing in it references those nodes. The conformal
  goldens therefore keep only the `node` lines of nodes on a bar physical
  group (ids 9 to 330), plus every other line. This is
  `rebar._bar_nodes_only`. It fails closed: an element line that is not a
  `CorotTruss` on bar nodes raises, so the filter can never hide a node an
  element uses.

**The coupling is pinned by count.** Neither deck emits a host element, so
neither can show whether `place` embedded the bars. Each rebar golden
therefore opens with a `=== <coupling> ===` section that is not an
emitted file:

| Placement | `bar_nodes` | `bar_nodes_in_host_connectivity` |
|---|---|---|
| `conformal` | 270 | 270 |
| `embedded` | 400 | 0 |

These are counts of shared nodes in the topology, not coordinates, so they
hold on every host. A conformal placement whose embed is skipped drops to
0 of 270 and fails the cell. `test_rebar_golden_pins_the_coupling` pins
the counts in the committed goldens too, so a regen over a broken embed
cannot pass as a match (review F1).

**Decks only.** The issue asks for the decks. There is no `.h5dump`: a
real `FEMData` archive carries a neutral zone whose `/provenance@base_dir`
and session ids change from run to run, and the dump script does not mask
them.

**Artifacts.** Both the session's `end()` and the bridge's first emit write
`<model_name>.h5` (ADR 0112 D1). Each render points `APEGMSH_ARTIFACT_DIR`
at its own scratch directory and restores the variable afterwards, so regen
never writes an `.h5` into this package.

## What a golden holds

`cells/<fixture>/<mode>/<output>.golden` holds every file the emit call
wrote, sorted by relative path. Each file starts with a `=== <path> ===`
header line. A `per_rank` golden therefore holds the driver and each
`ranks/rank<K>_<seq>.tcl` fragment.

`cells/<fixture>/<mode>/<output>.h5dump` is the canonical dump of that
cell's model written through `ops.h5()`, produced by
`scripts/h5_canonical_dump.py`. Each group, dataset and attribute gets one
line with its path, dtype, shape and a sha1 of its value. The sha1 does not
depend on byte order or chunking, and floats are hashed at 12 significant
digits (see "Floats" below). `FEMStub` is not a real `FEMData`, so the
archive has only the bridge zone (`/meta` and `/opensees`) and no neutral
zone. The `tcl` and `py` cells of one mode share a model, so their dumps are
identical; the dump sits beside each cell so that each cell is
self-contained.

## Bridge-only by design

The corpus stays on `FEMStub`, so its archives carry no neutral zone and
`OpenSeesModel.from_h5` refuses them with `BridgeOnlyArchiveError`. That is
deliberate (T ruling on #1797): the goldens pin what the bridge emits, and
giving every fixture a real `FEMData` would regenerate all 86 cells for no
gain in what they check. The R1b round-trip oracle
(`test_roundtrip_oracle.py`) therefore replays the real-`FEMData` builders
in `tests/opensees/h5`, not these cells. It compares `ops.tcl(flat=True)`
with `build("tcl", flat=True)`, which leaves out the partitioned
numberer/system auto-chain a partitioned archive records (#1832). The
replay writes its declarations, commands and analysis chain in the order
the archive's `/opensees/program` records (#2193), so no row is class B;
a partitioned archive, whose program records the partitioned emit, sorts
by the flat deck's section first and emit index second (#2277). Each
replayed slot keeps the int/float type its declaration emits, recovered
from `/opensees/decl_params` (#2194), so no row is class A either.

One cell has a replayable twin instead of a row. The oracle's last row was
class S: `test_h5_decl_params::_broad_partitioned` called `build_model`
for `two_column_frame_partitioned / staged_partitioned / recording`, so its
archive was bridge-only. That builder now rebuilds the same model (the
geometry, the two-rank split, both stages and the recorders) on a real
`FEMData`, so the oracle replays it and the ledger is empty (#2195).
`test_roundtrip_oracle.py` holds the twin to this cell: its non-flat deck
equals `staged_partitioned/recording.golden` byte for byte, except for the
`# apeGmsh <VERSION>; backend stock` line. The cell pins
`OpenSeesTarget(mode="stock")` and the twin pins no target, because the
archive does not carry the target mode (#2278). The cell itself stays on
`FEMStub`, so the goldens are unchanged.

## Determinism: normalised and masked fields

- **Line endings.** Goldens are written with LF. `.gitattributes` marks them
  `-text`, and the test also reads them with CRLF folded to LF. A Windows
  checkout therefore compares equal.
- **Paths.** Each cell emits into a fresh scratch directory. Any occurrence
  of that directory in the emitted text would be replaced by `<OUT>`; no
  deck embeds it today. Recorder files are relative (`out/...`), and the
  archive's `model_name` is always `model`.
- **Masked in the H5 dump.** Three attributes are masked, each only at
  this exact path:
  - `/meta@created_iso` is the wall-clock write time.
  - `/meta@apeGmsh_version` is the release identity of the installed
    distribution, not an emit output; masking it keeps a version bump
    from rewriting every golden.
  - `/meta/lineage@model_hash` is a derived digest: blake2b over the raw
    float bytes under `/opensees` (`compute_model_hash` in
    `opensees/_internal/lineage.py`). A last-ulp libm difference in the
    vecxz therefore changes it on another platform (#1258 review). Every
    dataset it summarises is already pinned line by line, so masking it
    loses no coverage.

  An attribute with the same name anywhere else is hashed. A new
  wall-clock field is a determinism bug, and the dump must expose it.
- **Masked in the decks.** The provenance stamp on the second line of
  every deck, `# apeGmsh <version>; backend <kind>[; build <sha>]`
  (F2-d, #1511), has its version masked as `<VERSION>` and its build as
  `<BUILD>` (`STAMP_LINE` in `builder.py`). The kind is not masked: it is
  what the deck was emitted for. `build_model` pins
  `OpenSeesTarget(mode="stock")`, so every cell reads `backend stock`
  whatever the process's live resolver has answered.
- **Not masked.** `schema_version`, `opensees_schema_version` and
  `snapshot_id` (empty for a stub). These are emit outputs, so a change to
  any of them is a real change.
- **Floats.** Values computed through libm, or through numpy reductions
  (`np.dot` and `np.linalg.norm` round by CPU dispatch, BLAS build and
  memory alignment), can differ in the last ulp between platforms. #1258
  and #1279 hit this on the `Spherical` orientation vecxz: one host emitted
  `geomTransf Linear 3 0.25881904510252085 0.0 0.9659258262890684` and
  another `0.2588190451025208 0.0 0.9659258262890682`. The cause was numpy
  reductions, not libm. The orientation math now runs on Python floats in a
  fixed order, so the vecxz is bit-identical everywhere, and
  `tests/opensees/unit/test_vecxz_determinism.py` pins it exactly. The
  tolerance below still absorbs any libm-derived value. Two rules apply:
  - **Decks** are compared token by token (`builder.first_deck_mismatch`).
    Float literals, meaning digits with a `.` or an exponent that are not
    glued to an identifier, must agree within a relative tolerance of 1e-12
    (`FLOAT_REL_TOL`), with an absolute floor of 1e-15 (`FLOAT_ABS_TOL`).
    Integers, signs, keywords, whitespace and line counts stay exact. The
    committed text is not rewritten: regen treats a deck within tolerance
    as unchanged. When `src` deliberately changes the bits of an emitted
    float, run `python -m tests.opensees.golden.regen --exact`, which
    rewrites every deck whose bytes differ.
  - **H5 dumps** hash every float dataset and attribute through its text at
    12 significant digits, and any `|x| < 1e-15` hashes as 0
    (`FLOAT_SIG_DIGITS` and `FLOAT_ZERO_FLOOR` in the dump script). This
    matters for `/opensees/transforms/Linear_<n>/per_element_vecxz`, which
    stores the same orientation vecxz as the deck. A relative change of
    about 1e-11 or more still changes the sha1. A value that sits exactly
    on a 12-digit rounding boundary can still flip on an ulp change. The
    odds are about 1e-4 per libm-derived value; if that happens, the fix
    belongs in this rounding, not in a regen.
- **Integer width.** Integer dtypes are pinned as written, for example
  `<i8`. numpy 2 writes `int64` for Python ints on every platform. On
  numpy 1.x under Windows, the platform default is `int32`, and an emitter
  that writes `np.asarray(list_of_ints)` would dump as `<i4`. That would be
  a real platform dependence in `src`, and it would show up here.

## Regenerating

```
python -m tests.opensees.golden.regen
```

Run it from the repository root. It emits from this checkout's `src/`,
even when the editable install points elsewhere. It rewrites
`MANIFEST.json` and the goldens, deletes goldens that no cell owns, and
prints one `changed` / `new` / `unchanged` / `removed` line per cell,
followed by a count. On a clean tree it reports 0 changed.

A regen is a maintainer-visible act:

- The test never regenerates. On a mismatch it fails with a unified diff of
  the differing cell and prints this command.
- `test_no_workflow_invokes_regen` fails if any `.github/workflows/*.yml`
  calls this command.
- **A PR that regenerates goldens must list the changed cells in its body**
  (paste the regen output), so review can accept each deck change on its
  own.
