"""The HDF5 damage sweep both particle readers must agree on (#3008 review).

From ``particles_two_stages_v1m2.ladruno`` (two whole particle stages), one damaged
copy per case and per superblock flags value (0x00, 0x05, 0x01; each copy gets a new
lookup3 over bytes 0-43 at byte 44, as the reader rule's step 3 expects):

- ``flip<off>``: the first byte XOR 0x20 of every HDF5 object signature in the file
  (``TREE``, ``SNOD``, ``HEAP``, ``OHDR``, ``OCHK``, ``GCOL``, fractal- and v2-B-tree,
  extensible- and fixed-array signatures, as present);
- ``zero<off>``: 256 zero bytes at every 4 KiB boundary after the superblock;
- ``zk<off>``: 1 KiB of zero bytes at 1536 + 3072 k (offsets off the 4 KiB grid);
- ``cut<len>``: the file cut to ``len`` bytes, every 2 KiB from 65536 (the second
  stage's layout starts near 66.5 KB), the stored end-of-allocation left as it was;
- ``attr<off>``: a byte XOR 0x20 in an attribute message (at each ``ATTRIBUTE_NAMES``
  name, and 8 bytes before it).

The files are built in a temp folder by the tests and never committed. The expected
outcome of each is ``damage_sweep_expected.json``, recorded from the Python reader
(``ParticleFile``) by running this file. ``tests/results/test_ladruno_particles.py``
checks Python still gives it, and that Python never reads a stage the fork's reference
``particle_stage_report`` (``fork_a1d4c7260/``) skips or refuses; the viewer's
``test/reader/ladrunoParticles.test.ts`` builds the same files by the same rule and checks
``openLadruno`` gives it too.

An outcome is ``["step <n>", "MODEL_STAGE[<i>] read <k>" | "... skipped" | "...
refused" | "... refused (corruption)", ...]``, stages in id order, or ``["file-error"]``
when the file does not open as a particle file, or ``["hang"]``: HDF5 itself does not
return on that file (it hangs in h5py and in the reference alike), so no test opens it.

Run from the repository root to re-record: ``python tests/fixtures/mpm/damage_sweep.py``.
Each case is opened in a worker process with a timeout, so a hang is recorded, not hit.
A known HDF5 hang (byte 178 XOR 0x20 at flags 0x05: h5py and the reference never
return) is not a site of this sweep; the timeout is for any other.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
BASE = HERE / "particles_two_stages_v1m2.ladruno"
EXPECTED = HERE / "damage_sweep_expected.json"
FLAGS = (0x00, 0x05, 0x01)
SIGNATURES = (b"TREE", b"SNOD", b"HEAP", b"OHDR", b"OCHK", b"GCOL", b"FRHP", b"FHDB",
              b"FHIB", b"FSHD", b"FSSE", b"BTHD", b"BTIN", b"BTLF", b"EAHD", b"EAIB",
              b"EADB", b"EASB", b"FAHD", b"FADB")
#: Attribute messages: a byte XOR 0x20 at each of these names in the file, and 8 bytes
#: before it (inside the attribute message's header).
ATTRIBUTE_NAMES = (b"KIND", b"PARTICLE_EPOCH", b"UNIT_SYSTEM", b"COMPONENTS",
                   b"PARTITION_REDUCTION")
HANG_SECONDS = 20


def sites(raw: bytes) -> list[tuple[str, str, int]]:
    """``(label, kind, offset or length)`` for every damage site, in a fixed order."""
    pattern = b"|".join(re.escape(s) for s in SIGNATURES)
    out = [(f"flip{m.start()}", "flip", m.start()) for m in re.finditer(pattern, raw)]
    out += [(f"zero{k * 4096}", "zero", k * 4096) for k in range(1, len(raw) // 4096 + 1)]
    out += [(f"zk{off}", "zk", off) for off in range(1536, len(raw), 3072)]
    out += [(f"cut{n}", "cut", n) for n in range(65536, len(raw), 2048)]
    attr = sorted({m.start() + rel for name in ATTRIBUTE_NAMES
                   for m in re.finditer(re.escape(name), raw) for rel in (0, -8)})
    out += [(f"attr{at}", "flip", at) for at in attr]
    return out


def damaged(raw: bytes, kind: str, at: int, flags: int) -> bytes:
    from apeGmsh.results.readers._ladruno_open import lookup3

    b = bytearray(raw)
    if kind == "flip":
        b[at] ^= 0x20
    elif kind == "zero":
        b[at:at + 256] = bytes(len(b[at:at + 256]))
    elif kind == "zk":
        b[at:at + 1024] = bytes(len(b[at:at + 1024]))
    elif kind == "cut":
        del b[at:]
    else:
        raise ValueError(kind)
    b[11] = flags
    b[44:48] = lookup3(bytes(b[:44])).to_bytes(4, "little")
    return bytes(b)


def cases(raw: bytes) -> Iterator[tuple[str, bytes]]:
    """``(name, damaged bytes)`` for every site and flags value."""
    for label, kind, at in sites(raw):
        for flags in FLAGS:
            yield f"{label}_{flags:02x}", damaged(raw, kind, at, flags)


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


def reference_reads(path: Path, scratch: Path, mc, lf) -> dict[str, str]:
    """What the fork's reference makes of each stage: ``{stage: "read <k>" |
    "torn" | "refused"}``; every stage ``"refused"`` when its open refuses the file.
    ``mc`` / ``lf`` are ``mpm_contracts`` / ``ladruno_format`` at a1d4c7260."""
    names =["MODEL_STAGE[1]", "MODEL_STAGE[2]"]
    try:
        try:
            with lf.open_ladruno(path) as f:
                report = mc.particle_stage_report(f)
        except lf.StageLayoutKilledError:
            report = mc.recover_particle_file(path, scratch / f"ref_{path.name}")
    except mc.TornStageError as e:
        bad = next((n for n in names if f"{n} will not open" in str(e)), None)
        return {n: "refused" for n in names} if bad is None else {bad: "refused"}
    except (OSError, KeyError, ValueError, RuntimeError):
        return {n: "refused" for n in names}
    out = {}
    for entry in report:
        out[entry["stage"]] = ("torn" if entry.get("torn_while_laid_out")
                               else f"read {entry['committed']}")
    return out


def sweep(tmp: Path, skip: "set[str] | frozenset[str]" = frozenset()) -> dict[str, list[str]]:
    """Every case's outcome but those in ``skip``; each file written to ``tmp`` and removed."""
    out = {}
    for name, data in cases(BASE.read_bytes()):
        if name in skip:
            continue
        p = tmp / f"{name}.ladruno"
        p.write_bytes(data)
        out[name] = outcome(p)
        p.unlink()
    return out


def _worker() -> None:
    """Read file paths from stdin, print each outcome as one JSON line (the recorder's worker)."""
    for line in sys.stdin:
        p = Path(line.strip())
        print(json.dumps(outcome(p)), flush=True)


def _record() -> dict[str, list[str]]:
    import queue
    import subprocess
    import tempfile
    import threading

    table: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as d:
        proc = None
        lines: "queue.Queue[str]" = queue.Queue()

        def start():
            p = subprocess.Popen([sys.executable, __file__, "--worker"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=lambda: [lines.put(x) for x in p.stdout], daemon=True).start()
            return p

        for name, data in cases(BASE.read_bytes()):
            if proc is None:
                proc = start()
            p = Path(d) / f"{name}.ladruno"
            p.write_bytes(data)
            proc.stdin.write(f"{p}\n")
            proc.stdin.flush()
            try:
                table[name] = json.loads(lines.get(timeout=HANG_SECONDS))
            except queue.Empty:
                proc.kill()
                proc.wait()
                proc = None
                table[name] = ["hang"]
                lines = queue.Queue()
        if proc is not None:
            proc.stdin.close()
            proc.wait()
    return table


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker()
    else:
        table = _record()
        EXPECTED.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(len(table), "cases;", sum(v == ["hang"] for v in table.values()), "hang")
