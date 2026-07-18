#!/usr/bin/env python
"""Refresh the user-level ``~/.claude/skills/apegmsh`` copy from origin/main.

Why this exists
---------------
The user-level skill directory used to be a Windows **junction** into the
main checkout's ``skills/apegmsh`` working tree. That coupled the served
skill to whatever branch the dev checkout happened to sit on (stale for
weeks when the branch was old), silently served uncommitted edits, and
made "write to the user skill" mutate the repo working tree. Retired
2026-07-18 in favour of a real directory refreshed by this script.

What it does
------------
1. ``git fetch origin`` (so origin/main is current).
2. Reads ``skills/apegmsh`` **from the origin/main tree object** (never
   the working tree — a dirty checkout cannot leak into the served copy).
3. Replaces ``~/.claude/skills/apegmsh`` atomically-ish: extract to a
   temp sibling, swap directories. If the target is still the legacy
   junction, only the link is removed (``os.rmdir`` on a junction
   detaches it without touching the repo it pointed at).

Run it after merging any skill PR:

    python scripts/refresh_user_skill.py            # refresh
    python scripts/refresh_user_skill.py --check    # exit 1 if stale

The repo-side derived mirror (``.claude/skills/apegmsh-helper``) is
handled by ``scripts/sync_skill.py`` and CI; this script only maintains
the user-level copy, which lives outside the repo and CI's reach.
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PREFIX = "skills/apegmsh"
TARGET = Path.home() / ".claude" / "skills" / "apegmsh"


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True, capture_output=True,
    ).stdout


def _origin_main_files() -> dict[str, bytes]:
    """``{relative_path: content}`` for the canonical skill at origin/main."""
    tar_bytes = _git("archive", "origin/main", PREFIX)
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            rel = member.name[len(PREFIX) + 1:]
            fh = tf.extractfile(member)
            assert fh is not None
            out[rel] = fh.read()
    return out


def _current_files() -> dict[str, bytes]:
    if not TARGET.is_dir():
        return {}
    return {
        p.relative_to(TARGET).as_posix(): p.read_bytes()
        for p in sorted(TARGET.rglob("*")) if p.is_file()
    }


def _remove_target() -> None:
    """Remove the target dir; a legacy junction is detached, not emptied."""
    if not TARGET.exists():
        return
    try:
        TARGET.rmdir()          # junction (or empty dir): removes the link only
        return
    except OSError:
        pass
    shutil.rmtree(TARGET)       # real, non-empty directory


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if the user-level copy differs from origin/main",
    )
    args = parser.parse_args(argv)

    subprocess.run(
        ["git", "-C", str(REPO), "fetch", "origin", "--quiet"], check=True,
    )
    fresh = _origin_main_files()
    sha = _git("rev-parse", "--short", "origin/main").decode().strip()

    if args.check:
        if _current_files() == fresh:
            print(f"user-level skill is in sync with origin/main@{sha}.")
            return 0
        print(
            f"user-level skill is STALE vs origin/main@{sha} — run "
            f"python scripts/refresh_user_skill.py"
        )
        return 1

    staging = TARGET.with_name(TARGET.name + ".refresh-tmp")
    if staging.exists():
        shutil.rmtree(staging)
    for rel, content in fresh.items():
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    _remove_target()
    staging.rename(TARGET)
    print(
        f"refreshed {TARGET} from origin/main@{sha} "
        f"({len(fresh)} files)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
