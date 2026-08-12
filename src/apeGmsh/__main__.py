"""``python -m apeGmsh`` — command-line entry point.

Subcommands
-----------
``doctor``
    One-shot environment preflight: interpreter identity, apeGmsh
    import path, gmsh, viewer/GL stack, OpenSees backend, and
    baseUnits agreement. Prints a markdown report; exits ``1`` when
    any error-severity finding exists. See :mod:`apeGmsh.doctor`.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh",
        description="apeGmsh command-line utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    sub.add_parser(
        "doctor",
        help=(
            "one-shot environment preflight (interpreter, import path, "
            "gmsh, viewers, OpenSees, baseUnits); exits 1 on errors"
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from apeGmsh.doctor import main as _doctor_main

        return _doctor_main()
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
