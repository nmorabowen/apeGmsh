"""``python -m apeGmsh.viewers <path>`` — open a Results file in a fresh viewer.

Used by ``Results.viewer(blocking=False)`` to spawn a subprocess that
survives a notebook/kernel crash. Picks ``Results.from_native`` or
``Results.from_mpco`` based on the path's extension and runs the
viewer's Qt event loop until the window closes.

Phase 8 (ADR 0020 INV-1) — for native ``.h5`` results the viewer reads
the OpenSeesModel from ``Results.model`` (auto-resolved against the
embedded ``/opensees/`` zone of the Composed-file pattern). The
``.mpco`` path has no embedded model zone, so ``--model-h5 PATH``
identifies the sibling ``model.h5`` and is required for that case.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def _open_results(path: Path, model_h5: Optional[Path]):
    """Pick the right reader by extension.

    For ``.mpco`` the sibling ``model.h5`` is required — the file
    itself carries no ``/opensees/`` zone, so the reader needs an
    explicit pointer (Phase 8 made this mandatory on
    :meth:`Results.from_mpco`).
    """
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results import Results
    if path.suffix.lower() == ".mpco":
        if model_h5 is None:
            print(
                "error: --model-h5 PATH is required for .mpco files "
                "(sibling model archive).",
                file=sys.stderr,
            )
            sys.exit(2)
        return Results.from_mpco(path, model_h5=model_h5)
    # Native results file — the model normally lives in the same path
    # (Composed-file pattern carries ``/opensees/`` at root). When
    # ``--model-h5`` is supplied, the model is read from that sibling
    # archive instead — for results whose embedded ``/model`` zone is not
    # independently readable (e.g. ``Results.demo()``).
    model_src = model_h5 if model_h5 is not None else path
    model = OpenSeesModel.from_h5(model_src)
    return Results.from_native(path, model=model, model_path=model_h5)


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
    parser.add_argument(
        "--model-h5",
        dest="model_h5",
        default=None,
        type=Path,
        help=(
            "Path to the sibling ``model.h5`` archive. Required for "
            ".mpco files (which carry no embedded OpenSees zone). For "
            "native .h5 results the model is auto-resolved from the file "
            "itself; pass this to override with a sibling archive when "
            "the embedded ``/model`` zone is not independently readable."
        ),
    )
    parser.add_argument(
        "--defs",
        dest="defs",
        default=None,
        type=Path,
        help=(
            "Path to a JSON sidecar of user-defined scalar expressions "
            "(ADR 0076), written by the parent Results.viewer() launch. "
            "Re-registered on the composites so custom scalars appear in "
            "the picker."
        ),
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: results file not found: {path}", file=sys.stderr)
        return 2

    results = _open_results(path, args.model_h5)
    _apply_defs(results, args.defs)
    results.viewer(blocking=True, title=args.title)
    return 0


def _apply_defs(results, defs_path: Optional[Path]) -> None:
    """Re-register custom scalar definitions from the ``--defs`` sidecar.

    A malformed or invalid sidecar is a warning, not a fatal error: the
    viewer still opens with the built-in components (the definitions are
    a convenience overlay, not part of the result data).
    """
    if defs_path is None:
        return
    import json
    try:
        payload = json.loads(Path(defs_path).read_text(encoding="utf-8"))
        results._apply_definitions_payload(payload)
    except Exception as exc:  # noqa: BLE001 — best-effort overlay
        print(
            f"warning: could not apply custom scalar definitions from "
            f"{defs_path}: {exc}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
