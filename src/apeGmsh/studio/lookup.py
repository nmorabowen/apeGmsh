"""``python -m apeGmsh.studio.lookup SYMBOL`` — ADR 0096 S2.

Prints ~20 lines: symbol, signature, skill pointer, one-line doc.
``--build`` regenerates ``_api_index.json`` from live composites.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def _utf8_streams() -> None:
    """Make stdout/stderr UTF-8 so a cp1252 console can print signatures.

    Signatures and docstrings carry non-Latin-1 characters (``ρ`` in
    ``g.loads.gravity``, ``…`` in the ambiguous list); a Windows console
    on cp1252 raised ``UnicodeEncodeError`` on them.  Streams that cannot
    be reconfigured (a captured test stream) are left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - detached stream
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _utf8_streams()
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.studio.lookup",
        description=(
            "Look up a public apeGmsh / apeSees signature from the "
            "generated index (ADR 0096). Not a source browser."
        ),
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Symbol or suffix (add_box, g.model.geometry.add_box, ops.fix).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Regenerate src/apeGmsh/studio/_api_index.json from live classes.",
    )
    args = parser.parse_args(argv)

    if args.build:
        from apeGmsh.studio._index_build import write_index

        dest = write_index()
        print(dest)
        if not args.symbol:
            return 0

    if not args.symbol:
        print("error: symbol required (or pass --build)", file=sys.stderr)
        return 2

    from apeGmsh.studio._lookup import lookup

    text, code = lookup(args.symbol)
    stream = sys.stdout if code == 0 else sys.stderr
    stream.write(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
