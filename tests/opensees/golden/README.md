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
