"""``python -m apeGmsh.studio SCRIPT.py`` — open the studio host.

Replays the script with the Gmsh session held open up to ``--phase``,
then opens the Qt host: MeshViewer if the script generated a mesh,
otherwise ModelViewer. Picks write ``.apegmsh/selection.json`` under
cwd. A successful stop writes ``.apegmsh/names.json`` and appends
``.apegmsh/runs.jsonl``.

``python -m apeGmsh.studio --status`` reads those files without replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.studio",
        description=(
            "Replay an apeGmsh script (session held open) up to --phase "
            "and open the Qt host (MeshViewer if meshed, else ModelViewer). "
            "Picks write .apegmsh/selection.json; a successful stop writes "
            ".apegmsh/names.json and appends .apegmsh/runs.jsonl. "
            "--status prints that state without replaying."
        ),
    )
    parser.add_argument(
        "script",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the apeGmsh Python script to replay.",
    )
    parser.add_argument(
        "--envelope",
        type=Path,
        default=None,
        help=(
            "Envelope JSON path (default: .apegmsh/selection.json under cwd)."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("model", "mesh", "results"),
        default="model",
        help=(
            "Stop at this gate: model = before generate(), "
            "mesh = before apeSees/Results, results = run to completion "
            "(default: model)."
        ),
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Replay only; do not open the Qt host (for tests / headless).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print last run + names + pick from .apegmsh/ (no replay).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="With --status, print the inspect payload as JSON.",
    )
    args = parser.parse_args(argv)

    if args.status:
        return _print_status(as_json=args.as_json)

    script = args.script
    if script is None:
        print(
            "error: script path required (or pass --status)",
            file=sys.stderr,
        )
        return 2
    if not script.is_file():
        print(f"error: script not found: {script}", file=sys.stderr)
        return 2

    from apeGmsh.studio._paths import envelope_path
    from apeGmsh.studio._replay import ReplayRunner

    dest = Path(args.envelope) if args.envelope is not None else envelope_path()
    runner = ReplayRunner()
    result = runner.run_until(script, phase=args.phase)
    if not result.ok:
        print(result.error or "replay failed", file=sys.stderr)
        return 1
    if result.session is None:
        print(
            "error: script did not open an apeGmsh session",
            file=sys.stderr,
        )
        return 2

    if args.no_viewer:
        print(
            f"replay ok  phase={result.phase}  "
            f"stopped_at={result.stopped_at!r}  "
            f"session={result.session.name!r}  hash={result.geometry_hash}"
        )
        if result.session.is_active:
            result.session.end()
        return 0

    from apeGmsh.studio._host import open_host

    try:
        open_host(
            result.session,
            envelope_path=dest,
            title=f"apeGmsh.studio — {script.name}",
        )
    finally:
        if result.session is not None and getattr(result.session, "is_active", False):
            result.session.end()
    return 0


def _print_status(*, as_json: bool) -> int:
    from apeGmsh.studio._status import collect_status, format_status, has_studio_state

    payload = collect_status()
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_status(payload))
    return 0 if has_studio_state(payload) else 2


if __name__ == "__main__":
    raise SystemExit(main())
