"""The HDF5 damage sweep both particle readers must agree on (#3008 review, F3).

From ``particles_two_stages_v1m2.ladruno`` (two whole particle stages), one damaged
copy per case and per superblock flags value (0x00, 0x05, 0x01, each with a new
lookup3 over bytes 0-43 at byte 44):

- ``flip<off>``: the first byte of every ``TREE`` / ``SNOD`` / ``HEAP`` signature
  XOR 0x20 (145 sites);
- ``zero<off>``: 256 zero bytes at every 4 KiB boundary after the superblock
  (31 sites).

528 files in all, built in a temp folder by the tests, never committed. The
expected outcome of each is ``damage_sweep_expected.json``, recorded from the Python
reader (``ParticleFile``) by running this file. ``tests/results/test_ladruno_particles.py``
checks Python still gives it; the viewer's ``test/reader/ladrunoParticles.test.ts``
builds the same files with the same rule and checks ``openLadruno`` gives it too.

An outcome is ``["step <n>", "MODEL_STAGE[<i>] read <k>" | "... skipped" |
"... refused" | "... refused (corruption)", ...]``, stages in id order, or
``["file-error"]`` when the file does not open as a particle file.

Run from the repository root to re-record: ``python tests/fixtures/mpm/damage_sweep.py``.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
BASE = HERE / "particles_two_stages_v1m2.ladruno"
EXPECTED = HERE / "damage_sweep_expected.json"
FLAGS = (0x00, 0x05, 0x01)


def cases(raw: bytes) -> Iterator[tuple[str, bytes]]:
    """``(name, damaged bytes)`` for every case and flags value, in a fixed order."""
    from apeGmsh.results.readers._ladruno_open import lookup3

    sites = [(f"flip{m.start()}", "flip", m.start())
             for m in re.finditer(rb"TREE|SNOD|HEAP", raw)]
    sites += [(f"zero{k * 4096}", "zero", k * 4096) for k in range(1, len(raw) // 4096 + 1)]
    for label, kind, off in sites:
        for flags in FLAGS:
            b = bytearray(raw)
            if kind == "flip":
                b[off] ^= 0x20
            else:
                b[off:off + 256] = bytes(len(b[off:off + 256]))
            b[11] = flags
            b[44:48] = lookup3(bytes(b[:44])).to_bytes(4, "little")
            yield f"{label}_{flags:02x}", bytes(b)


def outcome(path: Path) -> list[str]:
    """The Python reader's outcome for one file."""
    from apeGmsh.results.readers._ladruno_particles import ParticleFile

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = ParticleFile(path)
    except Exception:  # noqa: BLE001 - any refusal of the file is one outcome
        return ["file-error"]
    with pf:
        return [f"step {pf.step}"] + [
            f"{s.group} " + (
                f"read {s.committed}" if s.status == "read" else
                s.status + (" (corruption)" if s.status == "refused"
                            and "corruption" in s.reason else ""))
            for s in pf.states]


def sweep(tmp: Path) -> dict[str, list[str]]:
    """Every case's outcome, each file written to ``tmp`` and removed after."""
    out = {}
    for name, data in cases(BASE.read_bytes()):
        p = tmp / f"{name}.ladruno"
        p.write_bytes(data)
        out[name] = outcome(p)
        p.unlink()
    return out


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        table = sweep(Path(d))
    EXPECTED.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(len(table), "cases")
