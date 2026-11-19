### ADDED — both particle readers open a killed fork run by the reader rule, and skip or refuse its torn last stage by the step that opened it

The fork's MPM writer runs HDF5 SWMR from the end of each stage's layout to
close, so a killed run leaves a file that a plain open may refuse. The
reader rule of the fork's contract 232a §2.5 (fork `a1d4c7260`) is now
implemented in Python and in apeGmshViewer:

1. open read-only;
2. else, with superblock flags 0x05, open SWMR-read, never after `h5clear`;
3. else, with flags 0x01, repair a **copy** (flags 0x00, end-of-allocation
   max(stored, file size) zero-padded, lookup3 over bytes 0-43 at byte 44)
   and open the copy. The user's file is never written.

A lock error ("unable to lock file", or a raw read the OS refuses) raises a
distinct busy error, and nothing is copied. The viewer cannot see the fork
writer's guard lock (a byte-range lock at 2^62 on Windows, `flock` on POSIX,
with HDF5 file locking off): Node has no file-lock call, so a live run in a
layout step opens in the viewer by step 3, from a copy, and reads as killed.
The user's file is still never written. The Python reader reports it as busy. The Python open is the shared
module `apeGmsh.results.readers._ladruno_open` (`open_ladruno`), which
reports the step; `LadrunoReader` adopts it in #3003. The viewer opens every
`.ladruno` this way (`reader/h5open.ts` `openH5ByRule`). h5wasm 0.10.3 opens
a flags-0x05 file SWMR-read (mode `"Sr"`), and a repaired copy reads the
same committed outputs and byte-identical rows; both are pinned on a killed
run made with the fork's `hdf5_crash_replay.py`.

The stage rule lives in the particle readers (`ParticleFile` /
`particle_stage_report` in Python, `LadrunoParticles` in the viewer). A stage
that does not read, including one whose registry columns differ in length
(crash rule step 0), and that holds no output was torn while it was laid
out. As the last stage it is skipped by name after step 1 or 3 and refused
after step 2. Before another stage it is corruption and refused. A failing
stage that holds outputs is refused after any step. This replaces the C2-REG
refusal for the torn last stage.

A stage also fails when HDF5 cannot read it: a damaged link B-tree, an
unreadable `OFFSETS`, or a committed `ID` / `DATA` row whose chunk index does
not read (each stage's last committed row is read when the file opens). A
stage carrying `PARTICLE_EPOCH` whose group does not list is a failing
particle stage, not an FE stage. In a file with particles, a stage whose own
groups do not read (the torn last stage of a layout kill, or damage) no longer
refuses the whole file in the viewer: its FE part is left out with a warning,
and the stage rule decides its particles. Only a read failure or a missing
group or dataset does this; a schema refusal (a bad `KIND`) still refuses the
file. Both readers give the same outcome on 966 damaged copies of a two-stage
file (flips, zeroed blocks, truncations through the last stage's layout,
attribute damage), but for two named differences on attribute damage: the
viewer refuses a damaged field alone where Python refuses its stage, and
refuses the file for a `KIND` it cannot read. Python never reads a stage the
fork's reference skips or refuses. h5wasm's `slice` returns its buffer as data when HDF5
refuses a read, so the viewer's particle reader now reads through
`reader/h5open.ts` `readRows`, which checks HDF5's status.

Both readers also refuse a particle stage whose `UNIT_SYSTEM` names an
apeGmsh unit system that disagrees with `UNIT_LENGTH` / `UNIT_FORCE` /
`UNIT_TIME` (C2-UNIT), so the points layer never converts from the wrong
base.
