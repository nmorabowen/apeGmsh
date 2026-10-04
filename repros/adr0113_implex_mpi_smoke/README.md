# ADR 0113: 2-rank OpenSeesMP smoke of the IMPL-EX time driver

This smoke run is the MPI evidence ADR 0113 lacks. The bridge's static
per-rank form has not run under OpenSeesMP yet: the port's G20 gate ran
serially, and its G21 smoke tested only the runtime `getEleTags` form.
ADR 0113 stays **Proposed** until this run passes.

## Files

| File | What it is |
|---|---|
| `make_deck.py` | Builds the model on the bridge (a partitioned stub, 2 ranks) and writes the two files below |
| `implex_smoke.tcl` | The emitted deck, plus one probe per stage (marked `ADR 0113 smoke probe`) |
| `expected.json` | The targets per rank and each stage's increment |
| `check_output.py` | Checks the run's stdout |

## The model

The model has six fiber columns, three per rank. Each column is two
`forceBeamColumn` elements with a fixed base, a top mass, and a gravity load.

- Rank 0 holds elements 1–4: two IMPL-EX `ASDConcrete1D` columns, the driver's
  targets. It also holds 5–6, an elastic column that is not a target.
- Rank 1 holds the same layout: targets 7–10 and the elastic pair 11–12.

The three stages, with the chain `ParallelPlain` + `Mumps` in each:

| Stage | Analysis | Increment |
|---|---|---|
| gravity | `LoadControl 0.1` × 10 | 0.1 |
| hold | `LoadControl 0.02` × 5 | 0.02 |
| transient | `Newmark` | dt 0.01 × 5 |

After each stage's analyze loop, every rank prints, for each target it holds,
`eleResponse $e section 1 fiber 0 time`. That is `ASDConcrete1D`'s
`dTime dTimeCommit dTimeInitial` (response 4000).

## Run it (esmeralda, 2 ranks)

```bash
cd repros/adr0113_implex_mpi_smoke
mpirun -np 2 OpenSeesMP implex_smoke.tcl > run.log 2>&1
python check_output.py run.log
```

Use the cluster's Ladruno or stock `OpenSeesMP`; the deck uses no fork-only
command. Never schedule on node5.

## What passes

`check_output.py` prints `PASS` and exits 0 when all of these hold:

1. No "no objects were able to identify parameter" line.
2. Each rank probed exactly its own targets once per stage:
   - rank 0: 1 2 3 4
   - rank 1: 7 8 9 10
3. Every `dTime`, `dTimeCommit` and `dTimeInitial` equals the stage's
   increment, to 1e-12 relative.
4. All three stages were probed.

## Local checks already done (serial, fork build `fd87e396d`)

The run was serial: `OpenSees.exe` with `ParallelPlain` changed to `RCM` and
`Mumps` changed to `UmfPack` in a scratch copy.

- Rank 0's 12 probes all equal their stage's increment, and there are no "no
  objects" lines.
- `check_output.py` fails only on rank 1's missing probes, as it should for a
  serial run.
- **Negative control:** the same deck with the `_apesees_implex_dt` calls
  removed reports `dTime` = 0.0999… in the hold and transient stages, and
  `dTimeInitial` = 0.1. So the check tells driver-on from driver-off.

## Regenerate

From the worktree root:

```bash
PYTHONPATH="src:." python repros/adr0113_implex_mpi_smoke/make_deck.py
```

On Windows, use `src;.`.
