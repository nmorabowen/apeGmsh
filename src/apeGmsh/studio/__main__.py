"""``python -m apeGmsh.studio SCRIPT.py`` — open the studio host.

Replays the script with the Gmsh session held open up to ``--phase``,
then opens the Qt host: MeshViewer if the script generated a mesh,
otherwise ModelViewer. Picks write ``.apegmsh/selection.json`` under
the resolved project root (``--root`` / ``APEGMSH_ROOT`` / nearest
``.apegmsh/`` / cwd — INV-15). A successful stop writes
``.apegmsh/names.json`` and appends ``.apegmsh/runs.jsonl``.

``python -m apeGmsh.studio --status`` reads those files without replay.
``--assess PATH`` / ``--animate PATH`` are S4b/S4d workers.
``--pin`` / ``--emit-report`` are S4c/Amendment 2 (ledger pin + docs/
Markdown archive; ``--format html|canvas`` are skins of the same bundle).
Stills go through ``python -m apeGmsh.viewers render``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.studio",
        description=(
            "Replay an apeGmsh script (session held open) up to --phase "
            "and open the Qt host (MeshViewer if meshed, else ModelViewer). "
            "Picks write .apegmsh/selection.json; a successful stop writes "
            ".apegmsh/names.json and appends .apegmsh/runs.jsonl under the "
            "resolved project root (--root / APEGMSH_ROOT / nearest "
            ".apegmsh/ / cwd). "
            "--status prints that state without replaying. "
            "--assess / --animate are headless S4b/S4d workers. "
            "--pin / --emit-report are S4c (docs/ markdown archive; "
            "html/canvas are skins of the same bundle)."
        ),
    )
    parser.add_argument(
        "script",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the apeGmsh Python script to replay.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Project root for .apegmsh/ habitat files (INV-15). "
            "Empty / whitespace is treated as unset. "
            "Default: APEGMSH_ROOT, else nearest ancestor with .apegmsh/, "
            "else cwd."
        ),
    )
    parser.add_argument(
        "--envelope",
        type=Path,
        default=None,
        help=(
            "Envelope JSON path (default: .apegmsh/selection.json under "
            "the resolved project root)."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("model", "mesh", "results"),
        default="model",
        help=(
            "Stop at this gate: model = before generate(), "
            "mesh = before apeSees/Results, results = run to completion "
            "(default: model)."
        ),
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Replay only; do not open the Qt host (for tests / headless).",
    )
    parser.add_argument(
        "--no-watch",
        dest="no_watch",
        action="store_true",
        help=(
            "Disable the auto-refresh file watch (on by default, ADR 0095 "
            "S6b). The host's 'Watch' toolbar toggle can still enable it "
            "live."
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print last run + names + pick from .apegmsh/ (no replay).",
    )
    parser.add_argument(
        "--assess",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Assess a model.h5 or results file (no replay). "
            "Prints markdown, or JSON with --json."
        ),
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="With --assess, write stills under .apegmsh/visors/.",
    )
    parser.add_argument(
        "--animate",
        type=Path,
        default=None,
        metavar="PATH",
        help="Export results.export_animation (kind=history|yield). No setup=.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output gif/mp4 for --animate, or chapter path for --emit-report.",
    )
    parser.add_argument(
        "--kind",
        choices=("history", "yield", "formation"),
        default="history",
        help="animate(kind=) catalog token (S4d ships history and yield).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="With --animate, frames per second (default: 30).",
    )
    parser.add_argument(
        "--step-stride",
        dest="step_stride",
        type=int,
        default=1,
        help="With --animate, capture every Nth step (default: 1).",
    )
    parser.add_argument(
        "--model-h5",
        dest="model_h5",
        type=Path,
        default=None,
        help=(
            "Sibling model.h5 for .mpco (assess/animate), or the model "
            "archive to pin with --pin."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="With --status, --assess, --pin, --emit-report, or --animate, print JSON.",
    )
    parser.add_argument(
        "--pin",
        action="store_true",
        help="Pin model.h5 / results path+hash into .apegmsh/runs.jsonl.",
    )
    parser.add_argument(
        "--results",
        dest="results_path",
        type=Path,
        default=None,
        help="With --pin, results file (.h5 / .mpco / .ladruno).",
    )
    parser.add_argument(
        "--emit-report",
        dest="emit_report",
        action="store_true",
        help="Write a ReportBundle skin (markdown archive; html/canvas skins).",
    )
    parser.add_argument(
        "--format",
        dest="report_format",
        choices=("markdown", "html", "canvas"),
        default="markdown",
        help=(
            "emit_report format (markdown archive; html print skin; "
            "canvas live Cursor projection)."
        ),
    )
    parser.add_argument(
        "--pin-id",
        dest="pin_id",
        default=None,
        help="With --emit-report, use this pin id (default: last pin).",
    )
    args = parser.parse_args(argv)

    from apeGmsh.studio._paths import resolve_root

    root = resolve_root(args.root)

    n_modes = sum(
        (
            bool(args.status),
            args.assess is not None,
            args.animate is not None,
            bool(args.pin),
            bool(args.emit_report),
        )
    )
    if n_modes > 1:
        print(
            "error: pass only one of --status, --assess, --animate, "
            "--pin, --emit-report",
            file=sys.stderr,
        )
        return 2

    if args.status:
        return _print_status(as_json=args.as_json, root=root)
    if args.assess is not None:
        return _run_assess(args, root=root)
    if args.animate is not None:
        return _run_animate(args, root=root)
    if args.pin:
        return _run_pin(args, root=root)
    if args.emit_report:
        return _run_emit(args, root=root)

    script = args.script
    if script is None:
        from apeGmsh.studio._project import OutsideRootError, resolve_entry_script

        try:
            script = resolve_entry_script(None, root=root)
        except OutsideRootError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if script is None:
            print(
                "error: no script= and no .apegmsh/project.json entry "
                "(or pass --status / --assess / --animate / --pin / "
                "--emit-report)",
                file=sys.stderr,
            )
            return 2
    else:
        from apeGmsh.studio._paths import resolve_under

        script = resolve_under(root, script)
    if not script.is_file():
        print(f"error: script not found: {script}", file=sys.stderr)
        return 2

    from apeGmsh.studio._paths import envelope_path
    from apeGmsh.studio._replay import ReplayRunner

    dest = Path(args.envelope) if args.envelope is not None else envelope_path(root)
    runner = ReplayRunner(root=root)
    result = runner.run_until(script, phase=args.phase)
    if not result.ok:
        err = result.error or "replay failed"
        print(err, file=sys.stderr)
        if str(err).startswith("BUSY:"):
            return 3
        return 1
    if result.session is None:
        print(
            "error: script did not open an apeGmsh session",
            file=sys.stderr,
        )
        return 2

    if args.no_viewer:
        print(
            f"replay ok  phase={result.phase}  "
            f"stopped_at={result.stopped_at!r}  "
            f"session={result.session.name!r}  hash={result.geometry_hash}"
        )
        if result.session.is_active:
            result.session.end()
        return 0

    from apeGmsh.studio._host import open_host

    try:
        open_host(
            result.session,
            envelope_path=dest,
            title=f"apeGmsh.studio — {script.name}",
            root=root,
            script=script,
            runner=runner,
            watch=not args.no_watch,
        )
    finally:
        if result.session is not None and getattr(result.session, "is_active", False):
            result.session.end()
    return 0


def _print_status(*, as_json: bool, root: Path) -> int:
    from apeGmsh.studio._status import collect_status, format_status, has_studio_state

    payload = collect_status(root)
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_status(payload))
    return 0 if has_studio_state(payload) else 2


def _run_assess(args: argparse.Namespace, *, root: Path) -> int:
    import contextlib

    from apeGmsh.studio._paths import visors_path
    from apeGmsh.studio._verbs import assess_artifact, dumps_assess

    try:
        with contextlib.redirect_stdout(sys.stderr):
            payload = assess_artifact(
                args.assess,
                figures=args.figures,
                model_h5=args.model_h5,
                out_dir=visors_path(root),
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(dumps_assess(payload))
    elif payload.get("ok"):
        print(payload.get("text") or "")
    else:
        print(payload.get("error") or "assess failed", file=sys.stderr)
    return int(payload.get("returncode") or (0 if payload.get("ok") else 1))


def _run_animate(args: argparse.Namespace, *, root: Path) -> int:
    import contextlib

    from apeGmsh.studio._paths import visors_path
    from apeGmsh.studio._verbs import animate_artifact

    dest = (
        args.output if args.output is not None else visors_path(root) / f"{args.kind}.gif"
    )
    try:
        with contextlib.redirect_stdout(sys.stderr):
            payload = animate_artifact(
                args.animate,
                dest,
                kind=args.kind,
                model_h5=args.model_h5,
                fps=args.fps,
                step_stride=args.step_stride,
            )
    except ValueError as exc:
        if args.as_json:
            print(json.dumps({"ok": False, "written": [], "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, TypeError, RuntimeError) as exc:
        if args.as_json:
            print(json.dumps({"ok": False, "written": [], "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    if not payload.get("ok"):
        if args.as_json:
            print(json.dumps({
                "ok": False,
                "written": [],
                "error": payload.get("error") or "animate failed",
            }))
        else:
            print(payload.get("error") or "animate failed", file=sys.stderr)
        return int(payload.get("returncode") or 1)
    out = payload.get("output")
    if args.as_json:
        written = [str(out)] if out else []
        print(json.dumps({"ok": True, "written": written}))
    elif out:
        print(out)
    return 0


def _run_pin(args: argparse.Namespace, *, root: Path) -> int:
    from apeGmsh.studio._bundle import results_pin

    try:
        payload = results_pin(
            args.model_h5, args.results_path, root=root,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(payload.get("id") or "")
    return 0 if payload.get("ok") else 1


def _run_emit(args: argparse.Namespace, *, root: Path) -> int:
    from apeGmsh.studio._bundle import emit_report

    try:
        payload = emit_report(
            format=args.report_format,
            output=args.output,
            pin_id=args.pin_id,
            root=root,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(payload, indent=2, default=str))
    elif payload.get("path"):
        print(payload["path"])
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
