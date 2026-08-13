"""``python -m apeGmsh.studio SCRIPT.py`` — open the studio host.

Replays the script with the Gmsh session held open, then opens the
Qt host: MeshViewer if the script generated a mesh, otherwise
ModelViewer. Picks write ``.apegmsh/selection.json`` under cwd.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.studio",
        description=(
            "Replay an apeGmsh script (session held open) and open the "
            "Qt host (MeshViewer if meshed, else ModelViewer). Picks write "
            "a names-first SelectionEnvelope to .apegmsh/selection.json."
        ),
    )
    parser.add_argument(
        "script",
        type=Path,
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
        "--no-viewer",
        action="store_true",
        help="Replay only; do not open the Qt host (for tests / headless).",
    )
    args = parser.parse_args(argv)

    script = args.script
    if not script.is_file():
        print(f"error: script not found: {script}", file=sys.stderr)
        return 2

    from apeGmsh.studio._paths import envelope_path
    from apeGmsh.studio._replay import ReplayRunner

    dest = Path(args.envelope) if args.envelope is not None else envelope_path()
    runner = ReplayRunner()
    result = runner.run_until(script)
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
        print(f"replay ok  session={result.session.name!r}  hash={result.geometry_hash}")
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


if __name__ == "__main__":
    raise SystemExit(main())
