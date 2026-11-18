# Killed recorder run fixtures (M2-S8 #3003)

The fork's recorder (`recorder ladruno`) writes under HDF5 SWMR since OpenSees#1084, in the
release `ladruno-v20261116` (`FORK_PIN`, sha `6ee9680606e3d5d62c410d286547b8009d581159`). The
contract is the fork's `Ladruno_implementation/ladruno_schema_v1.md` §7b.2 at that commit.
**Never edit these files here.** Remake the run with the script, and take a reference change
from the fork.

## `killed_swmr.ladruno`

A run of the release build, killed while it recorded. `make_killed_run.py` made it on Windows
with that build's `opensees.pyd` (`ladrunoBuild()` = the pin sha, checked by the script), with
h5py 3.15.1 / HDF5 1.14.6 in the parent:

```bash
python tests/fixtures/ladruno_killed/make_killed_run.py --opensees-bin <the pinned build's dist/bin>
```

The deck (the script's `CHILD`) is a 2-D cantilever of four `elasticBeamColumn` under a sine
tip load, Newmark, `recorder ladruno -N displacement -N modesOfVibration -E force -flush 0`.
`eigen 2` runs after step 10, and `domainChange` after step 20. The child is killed with
`Popen.kill` (TerminateProcess) after it prints step 35. The kill lands wherever the child is,
so a rebuild is not byte-identical. What the committed file holds (git blob
`d2eda003791f3548dc5e7e3796333025df132bae`, 477,075 bytes):

| | |
|---|---|
| superblock flags | 0x05 (killed with SWMR armed): a plain open refuses it, and an SWMR-read open (reader rule step 2) reads it |
| `MODEL_STAGE[1]` | 20 committed steps (`STEP` 0..19); `MODES_OF_VIBRATION(U)/DATA/STEP_10/MODE_0..1` |
| `MODEL_STAGE[2]` | 15 committed steps (`STEP` 20..34); `MODES_OF_VIBRATION(U)` with `ID` and an empty `DATA` (no snapshot) |
| T' (the fork's `committed_steps`) | 20 and 15; `DATA`, `TIME` and `STEP` agree in length in both stages (not ragged) |

The tests build every other case from a copy and never commit it. A ragged stage has
`MODEL_STAGE[2]`'s `DISPLACEMENT/DATA` resized to 14 rows and `force/STEP` to 13, so T' is 13.
Flags 0x00 or 0x01 are set with the fork's `set_superblock`, or with the viewer's `lookup3`.
A torn stage has the first byte of its group's object header signature flipped. The headers are
at `MODEL_STAGE[1]` 45736 and `MODEL_STAGE[2]` 263035; the tests assert `OHDR` there.

## `fork_6ee968060/`

The stage rule's and crash rule's **reference code** at the pin, read as an oracle only:
`committed_steps`, `stage_report`, `recover_ladruno`, `open_ladruno`, `set_superblock`.

| File here | Fork path at `6ee9680606e3d5d62c410d286547b8009d581159` | git blob sha |
|---|---|---|
| `fork_6ee968060/ladruno_format.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_format.py` | `a854346a231eba3c86bf27ddc47d92d51298f31d` |
| `fork_6ee968060/ladruno_basis.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_basis.py` | `2c1f178877797e7977f66b3bf826fa8f9b1292a7` |

Check a copy: `git hash-object tests/fixtures/ladruno_killed/<file>` must print its sha above.
