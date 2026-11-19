"""Refresh the snapshot of the fork's ``mpm`` subcommands at ``FORK_PIN``.

Run it at every fork pin bump::

    python scripts/refresh_mpm_fork_surface.py               # gh api, at FORK_PIN's sha
    python scripts/refresh_mpm_fork_surface.py --fork <dir>  # a local fork checkout
    python scripts/refresh_mpm_fork_surface.py --check       # compare, do not write

It reads ``SRC/domain/pattern/ladrunoMPM/OPS_LadrunoMPM.cpp`` at ``FORK_PIN``'s
``sha`` and takes, from the body of ``int OPS_LadrunoMPM(void)``, every
``sub == "<name>"`` the parser dispatches on. The parser's own usage line (the
``fail("<grid|body|...> ...")`` it answers a bare ``mpm`` with) is read too,
and the two sets must agree, so a refactor that moves the dispatch out of the
function fails here instead of shrinking the snapshot.

The snapshot (``tests/fixtures/mpm_fork_surface/surface.json``) is what
``tests/opensees/parity/test_mpm_command_parity.py`` checks offline: its sha
against ``FORK_PIN``, and its names against the ``mpm`` members and the
``NOT_OFFERED`` rows of ``tests/opensees/parity/mpm_fork_surface.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "tests" / "fixtures" / "mpm_fork_surface" / "surface.json"
SOURCE = "SRC/domain/pattern/ladrunoMPM/OPS_LadrunoMPM.cpp"
FUNCTION = "int OPS_LadrunoMPM(void)"


class SurfaceError(RuntimeError):
    """The parser source does not read as the dispatch this script expects."""


def read_pin(path: Path = REPO / "FORK_PIN") -> dict[str, str]:
    pin: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if sep:
            pin[key.strip()] = val.strip()
    for key in ("tag", "sha", "repo"):
        if not pin.get(key):
            raise SurfaceError(f"{path} has no {key}= line")
    return pin


def _function_body(text: str) -> str:
    """The definition's body (a declaration ``...(void);`` is skipped)."""
    text = text.replace("\r\n", "\n")
    defs = list(re.finditer(r"^" + re.escape(FUNCTION) + r"\s*\{", text, re.M))
    if len(defs) != 1:
        raise SurfaceError(f"{SOURCE}: expected one definition of `{FUNCTION}`, "
                           f"found {len(defs)}")
    start = defs[0].start()
    end = text.find("\n}\n", start)
    if end < 0:
        raise SurfaceError(f"{SOURCE}: `{FUNCTION}` has no closing brace at column 0")
    return text[start:end]


def _usage(body: str) -> set[str]:
    """The names in the one ``fail("<a|b|...> ...")`` of the function (C
    string literals split over lines are joined first)."""
    found = []
    for m in re.finditer(r'fail\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)', body):
        text = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)))
        if text.startswith("<"):
            found.append(text)
    if len(found) != 1:
        raise SurfaceError(f"{SOURCE}: expected one fail(\"<sub|sub|...> ...\") "
                           f"usage line in `{FUNCTION}`, found {len(found)}")
    u = re.fullmatch(r"<([A-Za-z_]\w*(?:\|[A-Za-z_]\w*)*)> \.\.\.", found[0])
    if u is None:
        raise SurfaceError(f"{SOURCE}: the usage line does not read as "
                           f"`<sub|sub|...> ...`: {found[0]!r}")
    return set(u.group(1).split("|"))


def extract_subcommands(text: str) -> list[str]:
    """The sorted subcommands ``OPS_LadrunoMPM`` dispatches on, checked
    against the parser's own usage line."""
    body = _function_body(text)
    dispatch = set(re.findall(r'\bsub == "([A-Za-z_]\w*)"', body))
    usage = _usage(body)
    if not dispatch:
        raise SurfaceError(f"{SOURCE}: no `sub == \"...\"` dispatch in `{FUNCTION}`")
    if dispatch != usage:
        raise SurfaceError(
            f"{SOURCE}: the dispatch and the usage line disagree; only "
            f"dispatched: {sorted(dispatch - usage)}, only in the usage line: "
            f"{sorted(usage - dispatch)}. Read the parser by hand before "
            f"refreshing")
    return sorted(dispatch)


def fetch_source(pin: dict[str, str], fork: Path | None) -> str:
    if fork is not None:
        cmd = ["git", "-C", str(fork), "show", f"{pin['sha']}:{SOURCE}"]
    else:
        cmd = ["gh", "api", "-H", "Accept: application/vnd.github.raw",
               f"repos/{pin['repo']}/contents/{SOURCE}?ref={pin['sha']}"]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        raise SurfaceError(f"`{' '.join(cmd)}` failed: {done.stderr.strip()}")
    return done.stdout


def snapshot(pin: dict[str, str], text: str) -> dict[str, object]:
    return {
        "fork_tag": pin["tag"],
        "fork_sha": pin["sha"],
        "source": SOURCE,
        "function": FUNCTION,
        "refreshed_by": "python scripts/refresh_mpm_fork_surface.py",
        "subcommands": extract_subcommands(text),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fork", type=Path, help="a local fork checkout holding FORK_PIN's sha")
    ap.add_argument("--check", action="store_true",
                    help="compare with the committed snapshot; write nothing")
    args = ap.parse_args(argv)
    pin = read_pin()
    text = fetch_source(pin, args.fork)
    count = len(extract_subcommands(text))
    body = json.dumps(snapshot(pin, text), indent=2) + "\n"
    if args.check:
        old = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
        if old != body:
            print(f"{SNAPSHOT.relative_to(REPO)} differs from the fork at "
                  f"{pin['tag']} ({pin['sha'][:9]}): re-run without --check",
                  file=sys.stderr)
            return 1
        print(f"up to date: {count} subcommands at {pin['tag']}")
        return 0
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(body, encoding="utf-8", newline="\n")
    print(f"wrote {SNAPSHOT.relative_to(REPO)}: {count} subcommands at "
          f"{pin['tag']} ({pin['sha'][:9]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
