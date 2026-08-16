"""MCP tool bodies (ADR 0095 S4a–S4e / S5a–S5c / 0096 S3).

``status``, ``get_selection``, ``lookup``, ``results_pin``, ``emit_report``,
``highlight``, and ``promote_selection`` read habitat files or the
generated API index in-process. ``run_until`` / ``assess`` / ``animate``
shell to ``python -m apeGmsh.studio``. ``render`` shells to
``python -m apeGmsh.viewers render`` (ADR 0094 S5). Subprocess so this
process never holds the Gmsh kernel or a Qt/VTK context (INV-5).

Validation failures and timeouts return
``{"ok": false, "error": {"code": str, "message": str}}`` (S5c) —
they do not raise into the MCP adapter.

``lookup`` is See-family inspect (ADR 0096): the same ~20-line payload
as ``python -m apeGmsh.studio.lookup``. It does not grep ``src/`` and
does not wrap ``g.model.*``.

No ``gmsh``, no Qt, no viewers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._envelope import read_envelope
from ._paths import envelope_path, resolve_root, visors_path, display_path
from ._replay import PHASES
from ._status import brief_status, collect_status, has_studio_state

_VIEWS = frozenset({"mesh", "contour", "deformed", "reactions"})
_CAMERAS = frozenset({"iso", "xy", "xz", "yz"})
_ANIMATE_KINDS = ("history", "yield", "formation")
_ANIMATE_SHIPPED = frozenset({"history", "yield"})
_TIMEOUT_ENV = "APEGMSH_STUDIO_TIMEOUT"
_DEFAULT_TIMEOUT_S = 600.0


@dataclass
class _ShellResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _root(root: Path | str | None) -> Path:
    """Resolved project root (INV-15)."""
    return resolve_root(root)


def _require_root_dir(root: Path | str | None) -> Path | dict[str, Any]:
    """Resolved root that exists as a directory, else a BAD_ROOT failure."""
    base = _root(root)
    if not base.is_dir():
        return _fail(
            "BAD_ROOT",
            f"project root is not a directory: {base}",
            root=str(base),
        )
    return base


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return (base / p).resolve()
    return p.resolve()


def _fail(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Uniform MCP failure envelope (S5c)."""
    payload: dict[str, Any] = {
        "ok": False,
        "error": {"code": code, "message": message},
    }
    payload.update(extra)
    return payload


def _timeout_s() -> float:
    raw = os.environ.get(_TIMEOUT_ENV)
    if raw is None or not str(raw).strip():
        return _DEFAULT_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_S


def _shell(argv: list[str], *, cwd: Path, timeout: float | None = None) -> _ShellResult:
    limit = _timeout_s() if timeout is None else timeout
    try:
        proc = subprocess.run(  # noqa: S603 — argv list, same interpreter
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=limit,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else (
            exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        )
        err = exc.stderr if isinstance(exc.stderr, str) else (
            exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        )
        return _ShellResult(
            returncode=-1,
            stdout=out or "",
            stderr=err or f"timeout after {limit}s",
            timed_out=True,
        )
    except OSError as exc:
        return _ShellResult(
            returncode=-1,
            stdout="",
            stderr=str(exc),
            timed_out=False,
        )
    return _ShellResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def status(
    *,
    mode: str = "brief",
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Last run, names summary, and pick. No replay, no Qt.

    ``mode="brief"`` (default) is the agent token-budget shape — root,
    last-run intent, label/PG lists, counts summary, pick summary, plus
    ``text`` matching CLI ``format_status``. ``mode="full"`` returns the
    complete ``collect_status`` dict (including ``names.entities``).
    """
    wanted = str(mode or "brief").strip().lower()
    if wanted not in ("brief", "full"):
        return _fail(
            "INVALID_MODE",
            f"mode must be 'brief' or 'full'; got {mode!r}",
            mode=mode,
        )
    base = _root(root)
    full = collect_status(base)
    if wanted == "full":
        payload = dict(full)
        payload["mode"] = "full"
        payload["empty"] = not has_studio_state(full)
        payload["ok"] = True
        payload["error"] = None
        return payload
    payload = brief_status(full)
    payload["ok"] = True
    payload["error"] = None
    return payload


def get_selection(*, root: Path | str | None = None) -> dict[str, Any]:
    """Names-first pick envelope, or ``present=false`` if none."""
    base = _root(root)
    path = envelope_path(base)
    if not path.is_file():
        return {
            "ok": True,
            "error": None,
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
        return _fail(
            "UNREADABLE",
            str(exc),
            present=False,
            path=str(path),
            labels=[],
            physical_groups=[],
            phase=None,
            unnamed=[],
            envelope=None,
        )
    data = env.to_dict()
    return {
        "ok": True,
        "error": None,
        "present": True,
        "path": str(path),
        "labels": list(env.labels),
        "physical_groups": list(env.physical_groups),
        "phase": env.phase,
        "unnamed": data.get("unnamed") or [],
        "envelope": data,
    }


def lookup(
    symbol: str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Generated-index signature. Same payload as the lookup CLI. Not CAD.

    ``root`` is accepted for habitat logging (mcp_calls) only — the index
    is package-global.
    """
    del root  # INV-15 adapter surface; lookup itself is not root-scoped
    from ._lookup import lookup as resolve

    text, code = resolve(symbol)
    if code == 0:
        kind = "hit"
    elif text.startswith("miss:"):
        kind = "miss"
    elif text.startswith("ambiguous:"):
        kind = "ambiguous"
    else:
        kind = "error"
    return {
        "ok": code == 0,
        "error": None if code == 0 else {"code": kind.upper(), "message": text},
        "code": code,
        "kind": kind,
        "text": text,
    }


def run_until(
    script: str | Path | None = None,
    *,
    phase: str = "model",
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Replay *script* up to *phase* in a subprocess (INV-5).

    When *script* is omitted / blank, uses ``.apegmsh/project.json``
    (S5g). Returns ``NO_SCRIPT`` if neither is available. A
    ``project.json`` entry outside the habitat root returns
    ``OUTSIDE_ROOT`` (S5j).
    """
    from ._project import OutsideRootError, resolve_entry_script

    if phase not in PHASES:
        return _fail(
            "INVALID_PHASE",
            f"phase must be one of {PHASES}; got {phase!r}",
            phase=phase,
            script=str(script) if script is not None else None,
        )
    base = _require_root_dir(root)
    if isinstance(base, dict):
        return {
            **base,
            "phase": phase,
            "script": str(script) if script is not None else None,
            "returncode": 2,
            "stdout": "",
            "stderr": base["error"]["message"],
        }
    try:
        script_path = resolve_entry_script(script, root=base)
    except OutsideRootError as exc:
        return _fail(
            "OUTSIDE_ROOT",
            str(exc),
            returncode=2,
            phase=phase,
            script=None,
            stdout="",
            stderr=f"error: {exc}",
            status=status(root=base),
        )
    if script_path is None:
        return _fail(
            "NO_SCRIPT",
            "no script= and no .apegmsh/project.json entry",
            returncode=2,
            phase=phase,
            script=None,
            stdout="",
            stderr="error: no script= and no .apegmsh/project.json entry",
            status=status(root=base),
        )
    if not script_path.is_file():
        return _fail(
            "NOT_FOUND",
            f"script not found: {script_path}",
            returncode=2,
            phase=phase,
            script=display_path(script_path, base),
            stdout="",
            stderr=f"error: script not found: {script_path}",
            status=status(root=base),
        )
    from ._busy import read_busy

    held = read_busy(base)
    if held.get("busy"):
        return _fail(
            "BUSY",
            f"habitat locked by pid={held.get('pid')} op={held.get('op')}",
            returncode=3,
            phase=phase,
            script=display_path(script_path, base),
            root=str(base),
            busy=held,
            status=status(root=base),
        )
    proc = _shell(
        [
            sys.executable,
            "-m",
            "apeGmsh.studio",
            str(script_path),
            "--phase",
            phase,
            "--no-viewer",
            "--root",
            str(base),
        ],
        cwd=base,
    )
    if proc.timed_out:
        return _fail(
            "TIMEOUT",
            f"run_until exceeded {_timeout_s()}s",
            returncode=-1,
            phase=phase,
            script=display_path(script_path, base),
            root=str(base),
            stdout=proc.stdout,
            stderr=proc.stderr,
            status=status(root=base),
        )
    if proc.returncode == 3 or (proc.stderr or "").lstrip().startswith("BUSY:"):
        return _fail(
            "BUSY",
            (proc.stderr or proc.stdout or "habitat busy").strip() or "habitat busy",
            returncode=3,
            phase=phase,
            script=display_path(script_path, base),
            root=str(base),
            stdout=proc.stdout,
            stderr=proc.stderr,
            busy=read_busy(base),
            status=status(root=base),
        )
    return {
        "ok": proc.returncode == 0,
        "error": None if proc.returncode == 0 else {
            "code": "REPLAY_FAILED",
            "message": (proc.stderr or proc.stdout or "replay failed").strip()
            or "replay failed",
        },
        "returncode": proc.returncode,
        "phase": phase,
        "script": display_path(script_path, base),
        "root": str(base),
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
    base = _require_root_dir(root)
    if isinstance(base, dict):
        return {**base, "path": str(path), "returncode": 2, "stdout": "", "stderr": base["error"]["message"]}
    src = _resolve(path, base)
    if not src.is_file():
        return _fail(
            "NOT_FOUND",
            f"file not found: {src}",
            returncode=2,
            path=display_path(src, base),
            stdout="",
            stderr=f"error: file not found: {src}",
        )
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
    if proc.timed_out:
        return _fail(
            "TIMEOUT",
            f"assess exceeded {_timeout_s()}s",
            returncode=-1,
            path=display_path(src, base),
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    payload: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "error": None,
        "returncode": proc.returncode,
        "path": display_path(src, base),
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
    if not payload["ok"] and payload.get("error") is None:
        payload["error"] = {
            "code": "ASSESS_FAILED",
            "message": (proc.stderr or "assess failed").strip() or "assess failed",
        }
    elif payload["ok"]:
        payload["error"] = None
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
        return _fail(
            "INVALID_VIEW",
            f"view must be one of {sorted(_VIEWS)}; got {view!r}",
            view=view,
            written=[],
        )
    if camera not in _CAMERAS:
        return _fail(
            "INVALID_CAMERA",
            f"camera must be one of {sorted(_CAMERAS)}; got {camera!r}",
            camera=camera,
            written=[],
        )
    if pack and output is not None:
        return _fail(
            "INVALID_ARGS",
            "pass output= or pack=True, not both",
            written=[],
        )
    base = _require_root_dir(root)
    if isinstance(base, dict):
        return {**base, "written": [], "path": str(path), "returncode": 2}
    src = _resolve(path, base)
    if not src.is_file():
        return _fail(
            "NOT_FOUND",
            f"file not found: {src}",
            returncode=2,
            path=display_path(src, base),
            stdout="",
            stderr=f"error: file not found: {src}",
            written=[],
        )
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
    argv.append("--json")
    proc = _shell(argv, cwd=base)
    if proc.timed_out:
        return _fail(
            "TIMEOUT",
            f"render exceeded {_timeout_s()}s",
            returncode=-1,
            path=display_path(src, base),
            view=view,
            pack=pack,
            written=[],
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    written = [display_path(w, base) for w in _parse_written(proc.stdout)]
    ok = proc.returncode == 0
    return {
        "ok": ok,
        "error": None if ok else {
            "code": "RENDER_FAILED",
            "message": (proc.stderr or "render failed").strip() or "render failed",
        },
        "returncode": proc.returncode,
        "path": display_path(src, base),
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
        return _fail(
            "INVALID_KIND",
            f"kind must be one of {_ANIMATE_KINDS}; got {kind!r}",
            kind=kind,
        )
    if kind not in _ANIMATE_SHIPPED:
        return _fail(
            "NOT_SHIPPED",
            f"animate kind={kind!r} is not shipped "
            "(formation is later; S4d ships history and yield)",
            kind=kind,
        )
    base = _require_root_dir(root)
    if isinstance(base, dict):
        return {
            **base,
            "kind": kind,
            "path": str(path),
            "returncode": 2,
            "stdout": "",
            "stderr": base["error"]["message"],
        }
    src = _resolve(path, base)
    if not src.is_file():
        return _fail(
            "NOT_FOUND",
            f"file not found: {src}",
            returncode=2,
            kind=kind,
            path=display_path(src, base),
            stdout="",
            stderr=f"error: file not found: {src}",
        )
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
        "--json",
    ]
    if model_h5 is not None:
        argv.extend(["--model-h5", str(_resolve(model_h5, base))])
    proc = _shell(argv, cwd=base)
    if proc.timed_out:
        return _fail(
            "TIMEOUT",
            f"animate exceeded {_timeout_s()}s",
            returncode=-1,
            kind=kind,
            path=display_path(src, base),
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    written_raw = _parse_written(proc.stdout)
    written = [display_path(w, base) for w in written_raw]
    ok = proc.returncode == 0
    return {
        "ok": ok,
        "error": None if ok else {
            "code": "ANIMATE_FAILED",
            "message": (proc.stderr or "animate failed").strip() or "animate failed",
        },
        "returncode": proc.returncode,
        "kind": kind,
        "path": display_path(src, base),
        "output": written[-1] if written else display_path(dest, base),
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
    try:
        payload = _pin(model_h5, results, root=base)
    except ValueError as exc:
        return _fail("INVALID_ARGS", str(exc))
    if isinstance(payload, dict) and "error" not in payload:
        payload = {
            **payload,
            "error": None if payload.get("ok") else {
                "code": "PIN_FAILED",
                "message": "results_pin failed",
            },
        }
    return payload


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
    try:
        payload = _emit(format=format, output=output, pin_id=pin_id, root=base)
    except ValueError as exc:
        return _fail("INVALID_ARGS", str(exc), format=format)
    if isinstance(payload, dict) and "error" not in payload:
        payload = {
            **payload,
            "error": None if payload.get("ok") else {
                "code": "EMIT_FAILED",
                "message": "emit_report failed",
            },
        }
    return payload


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
    try:
        payload = _highlight(wanted, root=_root(root))
    except ValueError as exc:
        return _fail("INVALID_ARGS", str(exc))
    if isinstance(payload, dict) and "error" not in payload:
        payload = {
            **payload,
            "ok": payload.get("ok", True),
            "error": None if payload.get("ok", True) else {
                "code": "HIGHLIGHT_FAILED",
                "message": "highlight failed",
            },
        }
    return payload


def promote_selection(*, root: Path | str | None = None) -> dict[str, Any]:
    """Suggested ``.to_label()`` / ``.to_physical()`` edits. Does not write .py."""
    from ._highlight import promote_selection as _promote

    payload = _promote(root=_root(root))
    if isinstance(payload, dict) and payload.get("ok") is False:
        if "error" not in payload or not isinstance(payload.get("error"), dict):
            msg = str(payload.get("error") or "promote_selection failed")
            payload = {**payload, "error": {"code": "UNREADABLE", "message": msg}}
        return payload
    if isinstance(payload, dict) and "error" not in payload:
        payload = {**payload, "ok": payload.get("ok", True), "error": None}
    return payload


def _parse_written(stdout: str) -> list[str]:
    """Prefer a JSON ``written`` list; fall back to path-per-line scrape."""
    text = (stdout or "").strip()
    if not text:
        return []
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "written" in data:
                raw = data["written"]
                if isinstance(raw, list):
                    return [str(x) for x in raw]
                if raw is None:
                    return []
                return [str(raw)]
    return [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.startswith("[") and not ln.startswith("{")
    ]
