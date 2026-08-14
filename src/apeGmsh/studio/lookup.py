"""``python -m apeGmsh.studio.lookup SYMBOL`` — ADR 0096 S2.

Prints ~20 lines: symbol, signature, skill pointer, one-line doc.
``--build`` regenerates ``_api_index.json`` from live composites.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
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
