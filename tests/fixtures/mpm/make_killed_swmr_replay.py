"""Rebuild ``particles_killed_swmr_replay.h5`` with the fork's own replay tool.

The fork's ``Ladruno_scripts/mpm_crash_tests/hdf5_crash_replay.py`` at
``f0b9a1df6`` (release ``ladruno-v20261114``) runs the MPM particle writer's
HDF5 call pattern (``write_run``, here with ``swmr=True``: the writer since
cbe1b7149) through a logging file object, then rebuilds the file as it stands
after any write: the state a TerminateProcess leaves. This script uses the tool
unmodified: ``write_run(fobj, 6, False, 8 << 20, 0, True)``, the image after
write op ``KILL_OP``, then the tool's own ``check()``, which sets the superblock
flags to 0x05 (sec2's SWMR flags; the fileobj driver writes 0x04) with a new
lookup3, and reads the file by the reader rule.

At op 237 output 4's rows are on disk (STEP 5, ID 60 rows) but its OFFSETS entry
is not, and the stored end-of-allocation (38115) lags the file (39580 bytes):
the reader rule commits 4 outputs, after an SWMR-read open (step 2).

usage (from a directory holding the fork's files at their fork paths)::

    python tests/fixtures/mpm/make_killed_swmr_replay.py <fork_root> [out]
"""
from __future__ import annotations

import sys
from pathlib import Path

N_OUTPUTS = 6
KILL_OP = 237


def main(fork_root: Path, out: Path) -> None:
    sys.path.insert(0, str(fork_root / "Ladruno_scripts" / "mpm_crash_tests"))
    import hdf5_crash_replay as rp  # the fork's tool; it puts tests/ on sys.path itself

    fobj = rp.LogFile()
    rp.write_run(fobj, N_OUTPUTS, False, 8 << 20, 0, True)
    image = bytearray()
    for e in fobj.log[: KILL_OP + 1]:       # as the tool's main() rebuilds it
        if e[0] == "w":
            _, off, data = e
            end = off + len(data)
            if end > len(image):
                image.extend(b"\0" * (end - len(image)))
            image[off:end] = data
        elif e[0] == "t":
            size = e[1]
            if size < len(image):
                del image[size:]
            else:
                image.extend(b"\0" * (size - len(image)))
    out.write_bytes(bytes(image))
    print(rp.check(str(out)))               # ('OK', 4, ''): flags 0x05, 4 committed


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(Path(sys.argv[1]),
         Path(sys.argv[2]) if len(sys.argv) > 2 else here / "particles_killed_swmr_replay.h5")
