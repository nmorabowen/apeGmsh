#!/usr/bin/env python3
"""Open the apeCAD scratchpad for this habitat.

Saves / exports should land under::

  tools/apeCAD/human_sketches/     (human-led)
  tools/apeCAD/agentic_sketches/   (agent-led)

Usage (from habitat root)::

  python tools/apeCAD/open_interface.py
  python tools/apeCAD/open_interface.py --role agentic
  python tools/apeCAD/open_interface.py --port 8765 --no-browser
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
    parser = argparse.ArgumentParser(description="Open apeCAD for this APE habitat")
    parser.add_argument(
        "--role",
        choices=("human", "agentic"),
        default="human",
        help="default sketch archive for this session (human vs agentic)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args, passthrough = parser.parse_known_args(argv)

    ensure_src("APECAD_SRC", "apeCAD")
    human, agentic = sketch_dirs(TOOL_DIR)
    out = role_root(TOOL_DIR, args.role)

    os.environ["APE_HABITAT_ROOT"] = str(HABITAT)
    os.environ["APECAD_HUMAN_SKETCHES"] = str(human)
    os.environ["APECAD_AGENTIC_SKETCHES"] = str(agentic)
    os.environ["APECAD_SESSION_SKETCHES"] = str(out)

    print(f"habitat     {HABITAT}")
    print(f"role        {args.role}")
    print(f"save here   {out}")
    print(f"human       {human}")
    print(f"agentic     {agentic}")
    print("Tip: use Save / Save As in the scratchpad and pick the folder above.")
    print()

    from apeCAD.scratchpad.server import main as cad_main

    cad_argv = ["--host", args.host, "--port", str(args.port)]
    if args.no_browser:
        cad_argv.append("--no-browser")
    cad_argv.extend(passthrough)
    cad_main(cad_argv)


if __name__ == "__main__":
    main()
