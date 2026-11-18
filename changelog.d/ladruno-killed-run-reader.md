### FIXED — `Results.from_ladruno` and the viewer read a killed recorder run up to its last committed step

The fork's recorder writes under HDF5 SWMR since the release
`ladruno-v20261116` (OpenSees#1084, the `FORK_PIN`), so a run that is killed
while it records stays readable. `LadrunoReader` opened every file with a
plain `h5py.File(path, "r")`, which refuses a file killed with SWMR armed
("already open for write"). It now follows the fork's schema §7b.2:

- **The open.** `LadrunoReader` opens through the shared reader-rule open
  (`apeGmsh.results.readers._ladruno_open.open_ladruno`): read-only, else
  SWMR-read, else a repaired **copy**. The user's file is never written.
- **The stage rule.** A last stage that does not read was torn while it was
  laid out. It is skipped with `LadrunoStageSkippedWarning` after a plain
  open or a repaired copy, and refused after an SWMR-read open. An earlier
  stage that does not read refuses the file (`LadrunoStageCorrupt`).
- **The crash rule.** On a killed file (superblock flags other than 0x00),
  each stage is cut to T', the fewest rows any of its streaming results
  holds in `DATA`, `TIME` or `STEP`. The time vector stops there, and so
  does every node, element and energy read. A cleanly closed file is read
  as it is.
- **A damaged modes snapshot.** A `DATA/STEP_<k>` is damaged when it does
  not open or lacks a `MODE_<j>`, that is, its `MODE_<j>` keys have a gap
  or its modal groups disagree. What happens follows the stage rule:
  - In a run killed while a layout was written (flags 0x01), the snapshot
    was being written when the run was killed. It is skipped, together with
    every later snapshot, and `ModeSnapshotSkippedWarning` names each one.
  - In a run killed with SWMR armed (flags 0x05), every snapshot must read,
    so any damage raises `ModeSnapshotCorrupt`, which names the snapshot.
  - A cleanly closed file reads as before. A snapshot that does not open
    raises `ModeSnapshotCorrupt` instead of a raw `KeyError`.

  A snapshot with fewer modes than the one before it is whole and is listed:
  the writer stores one snapshot per eigen call, with that call's mode count.
- **A live run.** A run that is still writing raises `LadrunoFileBusyError`.
  The message names the SWMR tail read, `h5py.File(path, "r", swmr=True,
  locking=False)`. `stamp_provenance` still refuses a file the recorder
  holds open: the recorder's guard lock refuses its open.

`read_stage_stamp` opens the same way and sees the same stages.
`LadrunoReader` reports `open_step`, `killed` and `committed_steps(stage_id)`.

The viewer (`reader/ladruno.ts`) applies the same stage rule and T' cut to
a file without particles, and the same modes-snapshot rule to every file. A node history stops at the stage's last
committed step. Its busy error names the same SWMR tail read. h5wasm takes
no file lock, so the viewer reads a run that is still writing as the
contract's locking-off reader does: its committed prefix.

The empty `MODES_OF_VIBRATION(U|R)` groups that OpenSees#1084 lays out on
every stage are no mode stage. That held already, and it is now pinned on a
killed run and on a live run at the pin.
