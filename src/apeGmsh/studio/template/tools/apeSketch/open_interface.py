#!/usr/bin/env python3
"""Open the apeSketch board for this habitat.

Session store for this run is redirected under::

  tools/apeSketch/human_sketches/     (--role human, default)
  tools/apeSketch/agentic_sketches/   (--role agentic)

Usage (from habitat root)::

  python tools/apeSketch/open_interface.py
  python tools/apeSketch/open_interface.py --role agentic
  python tools/apeSketch/open_interface.py --http-only --no-browser
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
TOOLS = TOOL_DIR.parent
sys.path.insert(0, str(TOOLS))

from _launch import HABITAT, ensure_src, role_root, sketch_dirs  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Open apeSketch for this APE habitat")
    parser.add_argument(
        "--role",
        choices=("human", "agentic"),
        default="human",
        help="session archive root for this run (human vs agentic sketches)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9966)
    parser.add_argument("--ws-port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--http-only", action="store_true")
    parser.add_argument("--load", default=None, help="session id or .apesketch.json path")
    parser.add_argument("--resume", action="store_true")
    args, passthrough = parser.parse_known_args(argv)

    ensure_src("APESKETCH_SRC", "apeSketch")
    human, agentic = sketch_dirs(TOOL_DIR)
    out = role_root(TOOL_DIR, args.role)
    sessions = out / "sessions"
    assets = out / "assets"
    sessions.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    os.environ["APE_HABITAT_ROOT"] = str(HABITAT)
    os.environ["APESKETCH_HUMAN_SKETCHES"] = str(human)
    os.environ["APESKETCH_AGENTIC_SKETCHES"] = str(agentic)
    os.environ["APESKETCH_SESSION_SKETCHES"] = str(out)

    import apeSketch.host.server as sketch_server

    # Habitat-local archives (do not write into the library clone).
    sketch_server.SESSIONS_DIR = sessions
    sketch_server.ASSET_DIR = assets

    print(f"habitat     {HABITAT}")
    print(f"role        {args.role}")
    print(f"sessions    {sessions}")
    print(f"assets      {assets}")
    print(f"human       {human}")
    print(f"agentic     {agentic}")
    print()

    sketch_argv = ["--host", args.host, "--port", str(args.port)]
    if args.ws_port is not None:
        sketch_argv.extend(["--ws-port", str(args.ws_port)])
    if args.no_browser:
        sketch_argv.append("--no-browser")
    if args.http_only:
        sketch_argv.append("--http-only")
    if args.load:
        sketch_argv.extend(["--load", args.load])
    if args.resume:
        sketch_argv.append("--resume")
    sketch_argv.extend(passthrough)
    sketch_server.main(sketch_argv)


if __name__ == "__main__":
    main()
