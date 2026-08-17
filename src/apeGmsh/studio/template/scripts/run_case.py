#!/usr/bin/env python3
"""Run one model case per the habitat case contract (ADR 0095 Amendment 10).

    python scripts/run_case.py --model <id> --case <case>
        --script models/<id>/src/.../driver.py [--verify .../verify.py]

Creates ``models/<id>/cases/<case>/{results,logs,deck}``, runs the model
script with cwd = ``results/`` (outputs land in place; the optional
verify script later reads the same cwd), captures stdout+stderr to
``logs/``, auto-writes the deck disclosure README when ``deck/`` stays
empty (Amendment 9 F3), and writes ``run.json`` with ``git_provenance()``
captured at launch. A failed run is a recorded run (``run_exit`` says
which). A case that already has ``run.json`` is refused, not overwritten
— run records are immutable; a new question gets a new case id.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _habitat import HABITAT, git_provenance, utc_now  # noqa: E402


def _root_rel(path: Path) -> str:
    """Root-relative posix under the habitat; absolute outside (S5i)."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(HABITAT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _valid_id(name: str) -> bool:
    """Model / case ids are plain folder names — a path-shaped id would
    land the case (and its run.json) outside the cases tree."""
    return name not in ("", ".", "..") and "/" not in name and "\\" not in name


def _resolve_input(path: Path, kind: str) -> Optional[Path]:
    resolved = (path if path.is_absolute() else HABITAT / path).resolve()
    if not resolved.is_file():
        print(f"error: {kind} not found: {resolved}", file=sys.stderr)
        return None
    return resolved


def _run_logged(cmd: list, cwd: Path, log_path: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    log_path.write_text(
        (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8"
    )
    return proc.returncode


def _listing(case: Path, sub: str) -> list:
    return sorted(
        p.relative_to(case).as_posix()
        for p in (case / sub).rglob("*")
        if p.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model id (models/<id>/)")
    parser.add_argument(
        "--case", required=True, dest="case_id", help="case id (new folder name)"
    )
    parser.add_argument(
        "--script", required=True, type=Path, help="model/driver script to run"
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="oracle script run from results/ after a clean run",
    )
    args = parser.parse_args()

    for kind, name in (("model", args.model), ("case", args.case_id)):
        if not _valid_id(name):
            print(
                f"error: {kind} id must be a plain folder name, got {name!r}",
                file=sys.stderr,
            )
            return 2

    script = _resolve_input(args.script, "script")
    if script is None:
        return 2
    verify = None
    if args.verify is not None:
        verify = _resolve_input(args.verify, "verify script")
        if verify is None:
            return 2

    case = HABITAT / "models" / args.model / "cases" / args.case_id
    if (case / "run.json").is_file():
        print(
            f"error: {args.case_id} already has run.json — run records are "
            "immutable; use a new case id",
            file=sys.stderr,
        )
        return 1
    results, logs, deck = case / "results", case / "logs", case / "deck"
    for d in (results, logs, deck):
        d.mkdir(parents=True, exist_ok=True)

    provenance = git_provenance() or {}
    started = utc_now()
    t0 = time.monotonic()
    run_exit = _run_logged([sys.executable, str(script)], results, logs / "run.log")
    duration = round(time.monotonic() - t0, 1)
    print(f"run: exit={run_exit} ({duration}s) -> logs/run.log")

    verify_block = None
    if verify is not None and run_exit == 0:
        vexit = _run_logged(
            [sys.executable, str(verify)], results, logs / "verify.log"
        )
        vlines = [
            ln
            for ln in (logs / "verify.log")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.startswith(("PASS", "FAIL"))
        ]
        verify_block = {
            "script": _root_rel(verify),
            "exit": vexit,
            "passed": sum(1 for ln in vlines if ln.startswith("PASS")),
            "failed": sum(1 for ln in vlines if ln.startswith("FAIL")),
            "lines": vlines,
        }
        print(
            f"verify: exit={vexit} ({verify_block['passed']} PASS, "
            f"{verify_block['failed']} FAIL) -> logs/verify.log"
        )

    if not any(deck.iterdir()):
        (deck / "README.md").write_text(
            "No deck: this case ran in-process; nothing was emitted. "
            "Disclosure per ADR 0095 Amendment 9.\n",
            encoding="utf-8",
        )
        deck_note = "none — in-process run (see deck/README.md)"
    else:
        deck_note = "deck/"

    manifest = {
        "case": args.case_id,
        "model": args.model,
        "script": _root_rel(script),
        "ran_at": started,
        "duration_s": duration,
        "run_exit": run_exit,
        "results": _listing(case, "results"),
        "logs": _listing(case, "logs"),
        "deck": deck_note,
    }
    if verify_block is not None:
        manifest["verify"] = verify_block
    manifest.update(provenance)

    (case / "run.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OK: run.json -> {_root_rel(case / 'run.json')}")
    if run_exit != 0:
        return run_exit
    return verify_block["exit"] if verify_block is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
