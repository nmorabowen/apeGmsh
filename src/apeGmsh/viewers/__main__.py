"""``python -m apeGmsh.viewers <path>`` — open a Results file in a fresh viewer.

Used by ``Results.viewer(blocking=False)`` to spawn a subprocess that
survives a notebook/kernel crash. Picks ``Results.from_native`` or
``Results.from_mpco`` based on the path's extension and runs the
viewer's Qt event loop until the window closes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def _open_results(path: Path):
    """Pick the right reader by extension."""
    from apeGmsh.results import Results
    if path.suffix.lower() == ".mpco":
        return Results.from_mpco(path)
    return Results.from_native(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.viewers",
        description="Open an apeGmsh Results file in the post-solve viewer.",
    )
    parser.add_argument(
        "path",
        help="Path to a results file (.h5 native or .mpco STKO).",
    )
    parser.add_argument(
        "--title", default=None,
        help="Window title (defaults to 'Results — <filename>').",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: results file not found: {path}", file=sys.stderr)
        return 2

    results = _open_results(path)
    results.viewer(blocking=True, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
