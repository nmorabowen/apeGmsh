"""Regenerate ``.github/ci-constraints.txt`` from ``pip freeze`` output.

CI installs every lane with ``PIP_CONSTRAINT`` pointing at that file, so a
third-party release cannot turn ``main`` red (#1571). The nightly
``deps-canary`` workflow installs the *latest* releases and uploads their
``pip freeze`` as an artifact; this script turns a green freeze into the
new constraints file, so bumping is one PR.

    python scripts/update_constraints.py --run 123456789      # download the freeze artifacts of a canary run
    python scripts/update_constraints.py --freeze suite.txt [--freeze live.txt]
    python scripts/update_constraints.py --freeze suite.txt --diff   # markdown drift vs the file

The first ``--freeze`` is the base (the Python 3.11 ``suite`` environment).
Later freezes (the 3.12 ``live-stock`` environment) only contribute packages
the base lacks (openseespy); a package in both at different versions is
reported and the base wins. Stdlib only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent.parent / ".github" / "ci-constraints.txt"
# The project itself is installed editable; never constrain it.
SELF = {"apegmsh"}
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(text: str) -> dict[str, tuple[str, str]]:
    """``norm-name -> (display name, version)`` from freeze/constraint text."""
    out: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        m = _PIN.match(line.strip())
        if m and norm(m.group(1)) not in SELF:
            out[norm(m.group(1))] = (m.group(1), m.group(2))
    return out


def merge(freezes: list[str]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    base = parse(freezes[0])
    notes: list[str] = []
    for extra in freezes[1:]:
        for key, (disp, ver) in parse(extra).items():
            if key not in base:
                base[key] = (disp, ver)
            elif base[key][1] != ver:
                notes.append(f"{disp}: base {base[key][1]} kept, extra freeze has {ver}")
    return base, notes


def render(pins: dict[str, tuple[str, str]], source: str) -> str:
    head = (
        "# CI constraints: exact versions of the last green environment (#1576).\n"
        "# Applied to every pip install in tests.yml through PIP_CONSTRAINT.\n"
        "# Do not edit by hand: python scripts/update_constraints.py --run <canary run id>\n"
        f"# Source: {source}\n"
        "# User-facing caps (e.g. PySide6<6.12) live in pyproject.toml; this file\n"
        "# only pins what the green run resolved under them.\n"
    )
    body = "".join(f"{d}=={v}\n" for _, (d, v) in sorted(pins.items()))
    return head + body


def diff_md(old: dict, new: dict) -> str:
    rows = []
    for key in sorted(set(old) | set(new)):
        o = old.get(key, (None, None))[1]
        n = new.get(key, (None, None))[1]
        if o != n:
            disp = (new.get(key) or old[key])[0]
            rows.append(f"| {disp} | {o or '(absent)'} | {n or '(absent)'} |")
    if not rows:
        return ""
    return "| package | constraints | latest |\n|---|---|---|\n" + "\n".join(rows) + "\n"


def download(run_id: str, dest: Path) -> list[str]:
    subprocess.run(
        ["gh", "run", "download", run_id, "-p", "freeze-*", "-D", str(dest)], check=True
    )
    # One subdirectory per artifact; the suite freeze is the base.
    out = []
    for name in ("freeze-suite.txt", "freeze-live-stock.txt"):
        out += [p.read_text() for p in sorted(dest.rglob(name))]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--freeze", action="append", default=[], help="pip freeze file (repeatable)")
    ap.add_argument("--run", help="canary run id; downloads its 'freeze' artifact")
    ap.add_argument("--diff", action="store_true", help="print drift vs the file, write nothing")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    a = ap.parse_args(argv)

    if a.run:
        with tempfile.TemporaryDirectory() as td:
            texts = download(a.run, Path(td))
        source = f"canary run {a.run}"
    else:
        texts = [Path(p).read_text() for p in a.freeze]
        source = ", ".join(Path(p).name for p in a.freeze)
    if not texts:
        ap.error("give --run or at least one --freeze")

    pins, notes = merge(texts)
    for n in notes:
        print("note:", n, file=sys.stderr)
    if a.diff:
        old = parse(a.file.read_text()) if a.file.exists() else {}
        sys.stdout.write(diff_md(old, pins))
        return 0
    a.file.write_text(render(pins, source), encoding="utf-8", newline="\n")
    print(f"wrote {len(pins)} pins to {a.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
