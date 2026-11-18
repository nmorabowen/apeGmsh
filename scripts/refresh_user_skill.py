#!/usr/bin/env python
"""Refresh every apeGmsh skill under ``~/.claude/skills`` from origin/main.

Run this after ``git pull`` on each PC, and after merging any skill PR.

Why this exists
---------------
The user-level skill directory used to be a Windows **junction** into the
main checkout's ``skills/apegmsh`` working tree. That coupled the served
skill to whatever branch the dev checkout happened to sit on (stale for
weeks when the branch was old), silently served uncommitted edits, and
made "write to the user skill" mutate the repo working tree. Retired
2026-07-18 in favour of real directories refreshed by this script. The
repo dev skills (``.claude/skills/apegmsh-*``, including ``apegmsh-program``
and the derived ``apegmsh-helper`` mirror) have user-level copies too, and
those drifted until this script covered them as well.

What it does
------------
1. ``git fetch origin`` (so origin/main is current).
2. Reads ``skills/apegmsh`` and every ``.claude/skills/<name>`` **from the
   origin/main tree object** (never the working tree: a dirty checkout
   cannot leak into the served copies).
3. Replaces ``~/.claude/skills/<name>`` for each: extract to a temp
   sibling, swap directories. If a target is still a legacy junction, only
   the link is removed (``os.rmdir`` detaches it without touching the repo
   it pointed at).
4. A user-level skill that is not in the repo is never touched.

    python scripts/refresh_user_skill.py            # refresh all
    python scripts/refresh_user_skill.py --check    # exit 1 if any is stale

``--check`` compares content with line endings ignored (CRLF == LF).
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
CANONICAL = ("apegmsh", "skills/apegmsh")
DEV_ROOT = ".claude/skills"


def _skills_root() -> Path:
    return Path.home() / ".claude" / "skills"


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True, capture_output=True,
    ).stdout


def _archive_files(prefix: str) -> dict[str, bytes]:
    """``{relative_path: content}`` for ``prefix`` at the origin/main tree."""
    tar_bytes = _git("archive", "origin/main", prefix)
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            rel = member.name[len(prefix) + 1:]
            fh = tf.extractfile(member)
            assert fh is not None
            out[rel] = fh.read()
    return out


def _origin_main_skills() -> dict[str, dict[str, bytes]]:
    """``{skill_name: files}`` for every repo skill at origin/main."""
    skills = {CANONICAL[0]: _archive_files(CANONICAL[1])}
    listing = _git(
        "ls-tree", "-d", "--name-only", "origin/main", DEV_ROOT + "/",
    ).decode().splitlines()
    for entry in listing:
        name = entry.rstrip("/").rsplit("/", 1)[-1]
        skills[name] = _archive_files(f"{DEV_ROOT}/{name}")
    return skills


def _current_files(target: Path) -> dict[str, bytes]:
    if not target.is_dir():
        return {}
    return {
        p.relative_to(target).as_posix(): p.read_bytes()
        for p in sorted(target.rglob("*")) if p.is_file()
    }


def _same(a: dict[str, bytes], b: dict[str, bytes]) -> bool:
    """Equal file sets and contents, line endings ignored."""
    if a.keys() != b.keys():
        return False
    return all(
        a[k].replace(b"\r\n", b"\n") == b[k].replace(b"\r\n", b"\n")
        for k in a
    )


def _remove_target(target: Path) -> None:
    """Remove the target dir; a legacy junction is detached, not emptied."""
    if not target.exists():
        return
    try:
        target.rmdir()          # junction (or empty dir): removes the link only
        return
    except OSError:
        pass
    shutil.rmtree(target)       # real, non-empty directory


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if any user-level copy differs from origin/main",
    )
    args = parser.parse_args(argv)

    subprocess.run(
        ["git", "-C", str(REPO), "fetch", "origin", "--quiet"], check=True,
    )
    fresh = _origin_main_skills()
    sha = _git("rev-parse", "--short", "origin/main").decode().strip()
    root = _skills_root()

    if args.check:
        stale = [
            n for n, files in fresh.items()
            if not _same(_current_files(root / n), files)
        ]
        for n in fresh:
            print(f"  {n}: {'STALE' if n in stale else 'ok'}")
        if not stale:
            print(
                f"all {len(fresh)} user-level skills in sync with "
                f"origin/main@{sha}."
            )
            return 0
        print(
            f"{len(stale)} user-level skill(s) STALE vs origin/main@{sha}: "
            f"run python scripts/refresh_user_skill.py"
        )
        return 1

    root.mkdir(parents=True, exist_ok=True)
    for name, files in fresh.items():
        target = root / name
        staging = target.with_name(name + ".refresh-tmp")
        if staging.exists():
            shutil.rmtree(staging)
        for rel, content in files.items():
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        _remove_target(target)
        staging.rename(target)
        print(f"refreshed {target} ({len(files)} files)")
    print(f"{len(fresh)} skills refreshed from origin/main@{sha}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
