# ssi_frame_wall — model source

Two scripts, which are the **driver / verify pair** the habitat case
runner expects, plus `open_viewer.py`:

| Script | Role | Writes / reads |
|---|---|---|
| `build.py` | driver — builds, solves, records | writes `model.h5`, the recorder set (`out/`), `modes.h5`, `bench.json` and the transcode cache (`cache/`) into **cwd** |
| `check_slots.py` | verify — the acceptance gate | opens both result sources from **cwd** through `build.open_results`, writes `slots/*.png` |
| `open_viewer.py` | a ready-made viewer session | opens both result sources of a case through `build.open_results` |

The run has two result sources: the recorder set (stages `gravity` and
`dynamic`), transcoded on open, and `modes.h5` (six mode shapes, from
the bridge's `ops.eigen(6).to_native`). `build.open_results(results_dir)` returns
`(run, modes)`; every consumer opens the run that way, so none depends on
where the transcode cache puts its file.

Both default to the current directory rather than a path beside
themselves, because `scripts/run_case.py` runs each with **no arguments**
and with cwd set to the case's `results/`. Run by hand and they behave
the same way — outputs land wherever you are, so `cd` somewhere you don't
mind first, or pass `--out`.

`check_slots.py` prints one `PASS …` / `FAIL …` line per check, at the
start of the line, because that is what the runner tallies into
`run.json`'s `verify` block. Section headers are indented so they are not
counted.

## Running a case

```bash
python scripts/run_case.py --model ssi_frame_wall --case c002_something --script models/ssi_frame_wall/src/build.py --verify models/ssi_frame_wall/src/check_slots.py
```

Roughly 5½ minutes and ~120 MB for the default `small` size. Run records
are immutable — a case that already has `run.json` is refused, so a new
question gets a new case id.

The runner passes no arguments, so a case always runs at the default
size. For a heavier mesh, add a sibling driver that calls
`build.main()`-equivalent with `--size large` (a three-line script) and
point `--script` at it; that keeps "which size" a property of the case,
which is where it belongs, instead of a flag nobody can see in `run.json`.

## What the model is

The canonical ADR 0098 Results-viewer fixture: a soil block with a raft
cast into its surface, carrying a three-storey frame, a shear wall and
floor slabs — three element families, nine physical groups, two result
sources (gravity + dynamic, and the modes), and every one of the seven
result slots fed with real data.

The full rationale (why `forceBeamColumn` and not `elasticBeamColumn`,
why the structure is tied into the raft rather than sharing nodes, why
the soil carries no self-weight, and what the bench has already found)
lives in the habitat root [`README.md`](../../../README.md).
