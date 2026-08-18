"""``python -m apeGmsh.results.session render …`` — the snapshot door.

ADR 0098 §11 S5c. Draw a still of a pane of a saved session, out of
process: the agent gets the picture a human arranged, and a GL crash
stays out of the caller's kernel (the ADR 0094 S5 discipline, applied to
the new ontology).

Its own entry rather than a subcommand of ``python -m apeGmsh.viewers``
because that module flips at S6 and this door outlives the flip.

    python -m apeGmsh.results.session render out.h5.session.json shot.png
    python -m apeGmsh.results.session render snap.json shot.png --pane mesh-2

The broker comes from ``--results`` or, failing that, from the
``results_path`` the snapshot recorded when it was written — so the
common case is two arguments.

A v13 ``.viewer-session.json`` is REFUSED here and never renamed: the
rename-aside is the human flow's (ADR 0098 Consequences), and a batch
door that quietly moved a user's file aside would be doing surgery
nobody asked for.

Call-time imports keep the S0 purity guard green: this package stays
free of Qt/VTK, and the projection lives in ``apeGmsh.viewers.session``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_CAMERAS = ("iso", "xy", "xz", "yz")


def _resolve_session(
    snapshot: Path,
    results_path: "Optional[Path]" = None,
    model_h5: "Optional[Path]" = None,
) -> tuple[object, tuple[str, ...]]:
    """``(session, notices)`` — a snapshot bound to its broker.

    Shared with the tests, which use it as the IR-level oracle: the
    session this returns must realize exactly like the one that was
    saved. Keeping it a function (not inlined in ``main``) is what lets
    that oracle exist without a subprocess.
    """
    from apeGmsh.results._open import open_results
    from apeGmsh.results.session import load_snapshot

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    recorded = payload.get("results_path") if isinstance(payload, dict) else None
    target = results_path if results_path is not None else (
        Path(recorded) if recorded else None
    )
    if target is None:
        raise ValueError(
            f"{snapshot} records no results_path, so there is nothing to "
            f"realize it against — pass --results PATH."
        )
    if not target.is_file():
        raise ValueError(
            f"the results file this snapshot names is missing: {target}"
            + ("" if results_path is not None else
               " (it was recorded when the snapshot was written; pass "
               "--results PATH if it moved)")
        )
    # rename_legacy=False: refuse an old-schema file, touch nothing.
    restored = load_snapshot(
        snapshot,
        results=open_results(target, model_h5),
        rename_legacy=False,
    )
    assert restored is not None  # only the renaming flow returns None
    return restored.session, restored.notices


def _main_render(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.results.session render",
        description=(
            "Write an offscreen still of one pane of a saved session "
            "snapshot (ADR 0098 S5c)."
        ),
    )
    parser.add_argument("snapshot", help="A <results>.session.json file.")
    parser.add_argument("output", help="Output PNG.")
    parser.add_argument(
        "--pane",
        default=None,
        metavar="ID",
        help="Pane id (default: the only pane, or the first mesh pane).",
    )
    parser.add_argument(
        "--results",
        default=None,
        type=Path,
        metavar="PATH",
        help="Results file. Default: the path the snapshot recorded.",
    )
    parser.add_argument(
        "--model-h5",
        dest="model_h5",
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Sibling model.h5. Required when the snapshot's results are "
            "a .mpco file, which carries no model zone of its own."
        ),
    )
    parser.add_argument(
        "--camera",
        choices=sorted(_CAMERAS),
        default=None,
        help="Closed camera set (default: xy for a planar model, else iso).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help='Print {"ok": true, "written": [...]} instead of path lines.',
    )
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot)
    if not snapshot.is_file():
        return _fail(args, f"file not found: {snapshot}")

    try:
        session, notices = _resolve_session(
            snapshot, args.results, args.model_h5,
        )
    except Exception as exc:  # noqa: BLE001 — every refusal is the same door
        return _fail(args, str(exc))

    # Degradations are reported, never swallowed: a still drawn at a
    # different instant than the file asked for must say so (S5a).
    for notice in notices:
        print(f"[session] {notice}", file=sys.stderr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = session.render(out, args.pane, camera=args.camera)
    except Exception as exc:  # noqa: BLE001
        return _fail(args, str(exc))

    # ``render`` returns None on the ADR 0094 skip (no GL / the env
    # flag), having printed its own notice. ok=True with nothing
    # written: a skip is not a failure, and it is not a success either.
    paths = [str(written)] if written is not None else []
    if args.as_json:
        print(json.dumps({"ok": True, "written": paths}))
    else:
        for line in paths:
            print(line)
    return 0


def _fail(args: argparse.Namespace, message: str) -> int:
    if getattr(args, "as_json", False):
        print(json.dumps({"ok": False, "written": [], "error": message}))
    else:
        print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] == "render":
        return _main_render(argv_list[1:])
    print(
        "usage: python -m apeGmsh.results.session render "
        "SNAPSHOT.session.json OUTPUT.png [--pane ID] [--results PATH]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
