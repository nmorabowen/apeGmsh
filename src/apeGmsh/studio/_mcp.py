"""MCP tool bodies (ADR 0095 S4a–S4d / Amendment 1).

``status``, ``get_selection``, ``results_pin``, ``emit_report``,
``highlight``, and ``promote_selection`` read/write habitat files
in-process. ``run_until`` / ``assess`` / ``animate`` shell to
``python -m apeGmsh.studio``. ``render`` shells to
``python -m apeGmsh.viewers render`` (ADR 0094 S5). Subprocess so this
process never holds the Gmsh kernel or a Qt/VTK context (INV-5).

No ``gmsh``, no Qt, no viewers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ._envelope import read_envelope
from ._paths import envelope_path, visors_path
from ._replay import PHASES
from ._status import collect_status, has_studio_state

_VIEWS = frozenset({"mesh", "contour", "deformed", "reactions"})
_CAMERAS = frozenset({"iso", "xy", "xz", "yz"})
_ANIMATE_KINDS = ("history", "yield", "formation")
_ANIMATE_SHIPPED = frozenset({"history", "yield"})


def _root(root: Path | str | None) -> Path:
    if root is None:
        return Path.cwd()
    return Path(root)


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return (base / p).resolve()
    return p.resolve()


def _shell(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — argv list, same interpreter
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def status(*, root: Path | str | None = None) -> dict[str, Any]:
    """Last run, names, and pick. No replay, no Qt."""
    base = _root(root)
    payload = collect_status(base)
    payload["empty"] = not has_studio_state(payload)
    return payload


def get_selection(*, root: Path | str | None = None) -> dict[str, Any]:
    """Names-first pick envelope, or ``present=false`` if none."""
    base = _root(root)
    path = envelope_path(base)
    if not path.is_file():
        return {
            "present": False,
            "path": str(path),
            "labels": [],
            "physical_groups": [],
            "phase": None,
            "unnamed": [],
            "envelope": None,
        }
    try:
        env = read_envelope(path)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {
            "present": False,
            "path": str(path),
            "error": str(exc),
            "labels": [],
            "physical_groups": [],
            "phase": None,
            "unnamed": [],
            "envelope": None,
        }
    data = env.to_dict()
    return {
        "present": True,
        "path": str(path),
        "labels": list(env.labels),
        "physical_groups": list(env.physical_groups),
        "phase": env.phase,
        "unnamed": data.get("unnamed") or [],
        "envelope": data,
    }


def run_until(
    script: str | Path,
    *,
    phase: str = "model",
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Replay *script* up to *phase* in a subprocess (INV-5)."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}; got {phase!r}")
    base = _root(root)
    script_path = _resolve(script, base)
    if not script_path.is_file():
        return {
            "ok": False,
            "returncode": 2,
            "phase": phase,
            "script": str(script_path),
            "stdout": "",
            "stderr": f"error: script not found: {script_path}",
            "status": status(root=base),
        }
    proc = _shell(
        [
            sys.executable,
            "-m",
            "apeGmsh.studio",
            str(script_path),
            "--phase",
            phase,
            "--no-viewer",
        ],
        cwd=base,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "phase": phase,
        "script": str(script_path),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": status(root=base),
    }


def assess(
    path: str | Path,
    *,
    figures: bool = False,
    model_h5: str | Path | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """``fem.assess()`` / ``results.assess()`` via studio CLI (INV-5)."""
    base = _root(root)
    src = _resolve(path, base)
    if not src.is_file():
        return {
            "ok": False,
            "returncode": 2,
            "error": f"file not found: {src}",
            "path": str(src),
            "stdout": "",
            "stderr": f"error: file not found: {src}",
        }
    argv = [
        sys.executable,
        "-m",
        "apeGmsh.studio",
        "--assess",
        str(src),
        "--json",
    ]
    if figures:
        argv.append("--figures")
    if model_h5 is not None:
        argv.extend(["--model-h5", str(_resolve(model_h5, base))])
    proc = _shell(argv, cwd=base)
    payload: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "path": str(src),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.stdout.strip():
        try:
            payload.update(json.loads(proc.stdout))
        except json.JSONDecodeError:
            payload["ok"] = False
    payload["ok"] = bool(payload.get("ok")) and proc.returncode == 0
    payload["returncode"] = proc.returncode
    return payload


def render(
    path: str | Path,
    output: str | Path | None = None,
    *,
    view: str = "contour",
    component: str | None = None,
    step: int = -1,
    camera: str = "iso",
    pack: bool = False,
    deform: str | None = None,
    model_h5: str | Path | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """One still or ``render_pack`` via ``python -m apeGmsh.viewers render``."""
    if view not in _VIEWS:
        raise ValueError(f"view must be one of {sorted(_VIEWS)}; got {view!r}")
    if camera not in _CAMERAS:
        raise ValueError(
            f"camera must be one of {sorted(_CAMERAS)}; got {camera!r}"
        )
    if pack and output is not None:
        raise ValueError("pass output= or pack=True, not both")
    base = _root(root)
    src = _resolve(path, base)
    if not src.is_file():
        return {
            "ok": False,
            "returncode": 2,
            "error": f"file not found: {src}",
            "path": str(src),
            "stdout": "",
            "stderr": f"error: file not found: {src}",
            "written": [],
        }
    visors = visors_path(base)
    visors.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "apeGmsh.viewers", "render", str(src)]
    if pack:
        dest = visors / "pack"
        dest.mkdir(parents=True, exist_ok=True)
        argv.extend(["--pack", str(dest)])
    else:
        out = _resolve(output, base) if output is not None else visors / f"{view}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        argv.append(str(out))
        argv.extend(["--view", view, "--step", str(step), "--camera", camera])
        if component:
            argv.extend(["--component", component])
        if deform:
            argv.extend(["--deform", deform])
    if model_h5 is not None:
        argv.extend(["--model-h5", str(_resolve(model_h5, base))])
    proc = _shell(argv, cwd=base)
    written = [
        ln.strip()
        for ln in proc.stdout.splitlines()
        if ln.strip() and not ln.startswith("[")
    ]
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "path": str(src),
        "view": view,
        "pack": pack,
        "written": written,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def animate(
    path: str | Path,
    output: str | Path | None = None,
    *,
    kind: str = "history",
    model_h5: str | Path | None = None,
    fps: int = 30,
    step_stride: int = 1,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """``animate(kind=history|yield)`` via studio CLI. No ``setup=`` (INV-11)."""
    if kind not in _ANIMATE_KINDS:
        raise ValueError(
            f"kind must be one of {_ANIMATE_KINDS}; got {kind!r}"
        )
    if kind not in _ANIMATE_SHIPPED:
        raise ValueError(
            f"animate kind={kind!r} is not shipped "
            "(formation is later; S4d ships history and yield)"
        )
    base = _root(root)
    src = _resolve(path, base)
    if not src.is_file():
        return {
            "ok": False,
            "returncode": 2,
            "kind": kind,
            "error": f"file not found: {src}",
            "path": str(src),
            "stdout": "",
            "stderr": f"error: file not found: {src}",
        }
    dest = (
        _resolve(output, base)
        if output is not None
        else visors_path(base) / f"{kind}.gif"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "apeGmsh.studio",
        "--animate",
        str(src),
        "--output",
        str(dest),
        "--kind",
        kind,
        "--fps",
        str(fps),
        "--step-stride",
        str(step_stride),
    ]
    if model_h5 is not None:
        argv.extend(["--model-h5", str(_resolve(model_h5, base))])
    proc = _shell(argv, cwd=base)
    written = [
        ln.strip()
        for ln in proc.stdout.splitlines()
        if ln.strip() and not ln.startswith("[")
    ]
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "kind": kind,
        "path": str(src),
        "output": written[-1] if written else str(dest),
        "written": written,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def results_pin(
    model_h5: str | Path | None = None,
    results: str | Path | None = None,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Stamp model.h5 / results paths + hashes into the ledger (INV-12)."""
    from ._bundle import results_pin as _pin

    base = _root(root)
    return _pin(model_h5, results, root=base)


def emit_report(
    *,
    format: str = "markdown",
    output: str | Path | None = None,
    pin_id: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Write a ReportBundle skin. Markdown is the archive. No invented prose."""
    from ._bundle import emit_report as _emit

    base = _root(root)
    return _emit(format=format, output=output, pin_id=pin_id, root=base)


def highlight(
    names: list[str] | tuple[str, ...] | str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Names-first Point tool. Writes ``.apegmsh/highlight.json`` only (INV-5)."""
    from ._highlight import highlight as _highlight

    if isinstance(names, str):
        wanted = [names]
    else:
        wanted = list(names)
    return _highlight(wanted, root=_root(root))


def promote_selection(*, root: Path | str | None = None) -> dict[str, Any]:
    """Suggested ``.to_label()`` / ``.to_physical()`` edits. Does not write .py."""
    from ._highlight import promote_selection as _promote

    return _promote(root=_root(root))
