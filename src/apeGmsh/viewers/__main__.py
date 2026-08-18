"""``python -m apeGmsh.viewers`` — interactive Results viewer, or stills.

``python -m apeGmsh.viewers <path>`` opens a Results file in a fresh
Qt window. Used by ``Results.viewer(blocking=False)`` to spawn a
subprocess that survives a notebook/kernel crash.

ADR 0098 S6a flipped what that window is: the child calls
``results.viewer(blocking=True)``, which now opens a ``ResultsSession``
(``apeGmsh.viewers.session``), so this door flipped with it and needs no
ontology of its own. ``--restore-session`` / ``--no-save-session`` carry
the parent's open policy across the argv hop; before the flip the child
silently used the defaults.

``python -m apeGmsh.viewers render <path> …`` writes stills via
``viewers.render`` (ADR 0094 S5) so a GL crash stays out of the
kernel. In-process ``results.render`` / ``fem.render`` stay the
default. The first token must be exactly ``render``; a file named
``render.h5`` still opens the interactive viewer.

Phase 8 (ADR 0020 INV-1) — for native ``.h5`` results the viewer reads
the OpenSeesModel from ``Results.model`` (auto-resolved against the
embedded ``/opensees/`` zone of the Composed-file pattern). The
``.mpco`` path has no embedded model zone, so ``--model-h5 PATH``
identifies the sibling ``model.h5`` and is required for that case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def _open_results(path: Path, model_h5: Optional[Path]):
    """Pick the right reader by extension.

    Delegates to :func:`apeGmsh.results._open.open_results` — the
    dispatch belongs beside the readers, and the session render door
    (ADR 0098 S5c) needs it without importing this module, which flips
    at S6. Kept as a wrapper so this CLI's ``sys.exit(2)`` contract for
    a ``.mpco`` without its sibling archive is unchanged.
    """
    from apeGmsh.results._open import open_results

    try:
        return open_results(path, model_h5)
    except ValueError as exc:
        if "--model-h5" in str(exc):
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        raise


def _is_model_only(path: Path) -> bool:
    """True when ``path`` is a standalone ``model.h5`` (no results stages).

    ``.mpco`` / ``.ladruno`` are always results. A native results file
    carries ``/stages``. A FEMData ``to_h5`` archive has ``/meta`` and
    ``/nodes`` at the root and no ``/stages`` — that is ``fem.render``.
    Unreadable files return False so ``_open_results`` can raise.
    """
    if path.suffix.lower() in {".mpco", ".ladruno"}:
        return False
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "stages" in f:
                return False
            if any(str(k).startswith("MODEL_STAGE[") for k in f.keys()):
                return False
            info = f.get("INFO")
            if info is not None and "GENERATOR" in info.attrs:
                return False
            return "meta" in f and "nodes" in f
    except Exception:
        return False


def _open_fem(path: Path):
    # ADR 0014: viewers/ may not import apeGmsh.mesh. OpenSeesModel.from_h5
    # is already allowed on this CLI (Phase 8) and yields .fem.
    from apeGmsh.opensees import OpenSeesModel
    return OpenSeesModel.from_h5(path).fem


def _parse_deform_flag(text: str):
    """``SCALE`` or ``FIELD,SCALE`` — same shapes as ``results.render``."""
    raw = str(text).strip()
    if "," in raw:
        field, scale = raw.split(",", 1)
        field = field.strip()
        if not field:
            raise argparse.ArgumentTypeError(
                "deform FIELD,SCALE needs a non-empty field name."
            )
        try:
            return (field, float(scale))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"deform scale must be a number; got {scale!r}."
            ) from exc
    try:
        return float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "deform must be SCALE or FIELD,SCALE; "
            f"got {text!r}."
        ) from exc


def _parse_window_flag(text: str):
    """``WxH`` pixels, e.g. ``1280x720``."""
    raw = str(text).strip().lower()
    if raw.count("x") != 1:
        raise argparse.ArgumentTypeError(
            "window must be WxH (e.g. 1280x720)."
        )
    w_s, h_s = raw.split("x", 1)
    try:
        width, height = int(w_s), int(h_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "window must be WxH with integer pixels."
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            "window dimensions must be positive."
        )
    return (width, height)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] == "render":
        return _main_render(argv_list[1:])
    return _main_interactive(argv_list)


def _main_interactive(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.viewers",
        description=(
            "Open an apeGmsh Results file in the post-solve viewer. "
            "Use `python -m apeGmsh.viewers render …` to write stills "
            "without Qt."
        ),
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
    parser.add_argument(
        "--restore-session",
        dest="restore_session",
        choices=("yes", "no", "prompt"),
        default="prompt",
        help=(
            "What to do with <results>.viewer-session.json: yes restores "
            "silently, no ignores it, prompt (default) asks. Emitted by "
            "the parent Results.viewer(blocking=False) launch so the "
            "child honours the same policy the caller asked for."
        ),
    )
    parser.add_argument(
        "--no-save-session",
        dest="save_session",
        action="store_false",
        help=(
            "Do not write <results>.viewer-session.json when the window "
            "closes."
        ),
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: results file not found: {path}", file=sys.stderr)
        return 2

    results = _open_results(path, args.model_h5)
    _apply_defs(results, args.defs)
    results.viewer(
        blocking=True,
        title=args.title,
        restore_session={"yes": True, "no": False}.get(
            args.restore_session, "prompt",
        ),
        save_session=args.save_session,
    )
    return 0


def _main_render(argv: Sequence[str]) -> int:
    """``python -m apeGmsh.viewers render …`` — stills, no Qt event loop."""
    from apeGmsh.viewers.render import _CAMERAS, _DEFAULT_WINDOW, _VIEWS

    default_window = f"{_DEFAULT_WINDOW[0]}x{_DEFAULT_WINDOW[1]}"
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.viewers render",
        description=(
            "Write offscreen stills via viewers.render (ADR 0094 S5). "
            "In-process results.render / fem.render stay the default; "
            "this CLI is the out-of-kernel door."
        ),
    )
    parser.add_argument(
        "path",
        help=(
            "Results file (.h5 native / .mpco / .ladruno) or a "
            "standalone model.h5 (fem.render)."
        ),
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output PNG (required unless --pack DIR).",
    )
    parser.add_argument(
        "--view",
        choices=sorted(_VIEWS),
        default="contour",
        help="Closed view set (default: contour). Unused with --pack.",
    )
    parser.add_argument(
        "--component",
        default=None,
        metavar="NAME",
        help="Nodal component (default: S1 default_contour_component).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        metavar="N",
        help="Step index (default: -1 = last). Out of range is an error.",
    )
    parser.add_argument(
        "--deform",
        type=_parse_deform_flag,
        default=None,
        help="SCALE or FIELD,SCALE (same as results.render deform=).",
    )
    parser.add_argument(
        "--camera",
        choices=sorted(_CAMERAS),
        default="iso",
        help="Closed camera set (default: iso).",
    )
    parser.add_argument(
        "--window",
        type=_parse_window_flag,
        default=_DEFAULT_WINDOW,
        metavar="WxH",
        help=f"Pixel size (default: {default_window}).",
    )
    parser.add_argument(
        "--pack",
        dest="pack",
        default=None,
        type=Path,
        metavar="DIR",
        help="Write results.render_pack(DIR) instead of one still.",
    )
    parser.add_argument(
        "--model-h5",
        dest="model_h5",
        default=None,
        type=Path,
        help=(
            "Sibling model.h5. Required for .mpco files (same as the "
            "interactive command)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print {\"ok\": true, \"written\": [...]} instead of path lines.",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    if args.pack is not None and args.output is not None:
        print(
            "error: pass either OUTPUT.png or --pack DIR, not both.",
            file=sys.stderr,
        )
        return 2
    if args.pack is None and args.output is None:
        print(
            "error: OUTPUT.png is required unless --pack DIR is given.",
            file=sys.stderr,
        )
        return 2

    try:
        if _is_model_only(path):
            return _render_fem(path, args)
        return _render_results(_open_results(path, args.model_h5), args)
    except ValueError as exc:
        if getattr(args, "as_json", False):
            print(json.dumps({"ok": False, "written": [], "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        if getattr(args, "as_json", False):
            print(json.dumps({"ok": False, "written": [], "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


def _render_fem(path: Path, args: argparse.Namespace) -> int:
    if args.pack is not None:
        msg = (
            "error: --pack requires a results file; "
            "there is no fem.render_pack."
        )
        if getattr(args, "as_json", False):
            print(json.dumps({"ok": False, "written": [], "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 2
    written = _open_fem(path).render(
        Path(args.output),
        camera=args.camera,
        window_size=args.window,
    )
    return _exit_written(written, as_json=getattr(args, "as_json", False))


def _render_results(results, args: argparse.Namespace) -> int:
    if args.pack is not None:
        written = results.render_pack(
            args.pack,
            camera=args.camera,
            window_size=args.window,
        )
        return _exit_written(written, as_json=getattr(args, "as_json", False))
    written = results.render(
        Path(args.output),
        view=args.view,
        component=args.component,
        step=args.step,
        deform=args.deform,
        camera=args.camera,
        window_size=args.window,
    )
    return _exit_written(written, as_json=getattr(args, "as_json", False))


def _exit_written(written, *, as_json: bool = False) -> int:
    """Skip (None / ()) is exit 0. Print each written Path (or JSON)."""
    paths: list[str] = []
    if written is None or written == ():
        paths = []
    elif isinstance(written, tuple):
        paths = [str(item) for item in written]
    else:
        paths = [str(written)]
    if as_json:
        print(json.dumps({"ok": True, "written": paths}))
    else:
        for item in paths:
            print(item)
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
