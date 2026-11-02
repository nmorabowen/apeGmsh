"""Batch stills from apeGmshViewer, for an agent (ADR 0128 D4, D7).

A thin shim over the app's command line, which is the contract:
``apeGmshViewer --render <spec.json> --out <dir>`` takes every shot of a
view spec in a hidden window and writes one PNG per shot plus
``manifest.json`` into ``<dir>``. This module finds the installed
executable, writes the spec, runs the app and returns the manifest as a
dict. It imports nothing from ``apeGmshViewer/`` and does not check the
spec: the app's validator is the only one (ADR 0128 INV-2), and its
refusals come back here as :class:`AgentViewError`.

Exit codes, as ``apeGmshViewer/src/main/main.ts`` states them for
``--mode=render``: 0 every shot drawn, 1 a shot refused or the run failed,
2 the spec or ``--out`` refused (no window opens, no PNG is written).

The executable is ``$APEGMSH_VIEWER`` when that is set, else the per-user
install (``%LOCALAPPDATA%\\Programs\\apegmsh-viewer\\apeGmshViewer.exe``,
from ``npm run dist``; see ``apeGmshViewer/README.md``). A
``$APEGMSH_VIEWER`` that names no file is refused, never passed over for
the install.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["AgentViewError", "find_viewer", "render_stills"]

#: The environment variable that names the executable.
VIEWER_ENV = "APEGMSH_VIEWER"
#: What ``--render`` writes into ``--out`` last (``src/spec/manifest.ts``).
MANIFEST_FILE = "manifest.json"
#: The per-user install, relative to ``%LOCALAPPDATA%`` (electron-builder's
#: one-click NSIS installer names the folder after package.json's ``name``).
_INSTALL = ("Programs", "apegmsh-viewer", "apeGmshViewer.exe")
#: Seconds before a run that has not exited is killed. The app ends a hung
#: page on its own after 60 s (``--hang-ms``); this bounds the rest.
DEFAULT_TIMEOUT = 900.0


class AgentViewError(RuntimeError):
    """The app refused the spec, a shot or ``--out``, or the run failed.

    ``manifest`` is the manifest this run wrote (the evidence of what was
    drawn and what was refused), or ``None`` when it wrote none.
    ``returncode`` is the app's exit code (``None`` after a timeout).
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None,
        manifest: dict[str, Any] | None,
        stderr: str,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.manifest = manifest
        self.stderr = stderr


def find_viewer() -> Path:
    """Return the apeGmshViewer executable: ``$APEGMSH_VIEWER``, else the per-user install.

    Raises ``FileNotFoundError`` when ``$APEGMSH_VIEWER`` is set and names
    no file, or when it is unset and the app is not installed.
    """
    named = os.environ.get(VIEWER_ENV)
    if named:
        path = Path(named)
        if not path.is_file():
            raise FileNotFoundError(
                f"${VIEWER_ENV} is {named!r}, which is not a file; set it to the "
                "apeGmshViewer executable, or unset it to use the per-user install"
            )
        return path
    local = os.environ.get("LOCALAPPDATA") if sys.platform == "win32" else None
    if local:
        path = Path(local).joinpath(*_INSTALL)
        if path.is_file():
            return path
        where = f"it is not installed at {path}"
    else:
        where = "this platform has no per-user install (the installer is Windows-only)"
    raise FileNotFoundError(
        f"apeGmshViewer was not found: ${VIEWER_ENV} is unset and {where}. "
        "Install it from apeGmshViewer/ (npm ci, npm run dist, run the installer), "
        f"or set ${VIEWER_ENV} to the executable."
    )


def _absolute_files(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The spec with relative ``files`` paths made absolute against the working folder.

    The app reads a relative path against the spec file's folder, and a
    spec handed over as a dict is written to a temporary folder, so its
    relative paths mean the caller's working folder, as any Python path
    does. Anything that is not a string path is left for the app to refuse.
    """
    out = dict(spec)
    files = spec.get("files")
    if isinstance(files, Mapping):
        out["files"] = {
            k: (str(Path(v).resolve()) if isinstance(v, str) and not Path(v).is_absolute() else v)
            for k, v in files.items()
        }
    return out


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentViewError(
            f"apeGmshViewer wrote {path}, which is not readable JSON: {exc}",
            returncode=None,
            manifest=None,
            stderr="",
        ) from exc
    if not isinstance(data, dict):
        raise AgentViewError(
            f"apeGmshViewer wrote {path}, which is not a JSON object",
            returncode=None,
            manifest=None,
            stderr="",
        )
    return data


def _refusals(manifest: dict[str, Any]) -> list[str]:
    """Every refusal the manifest states: the run's, then each shot's."""
    out: list[str] = []
    run = manifest.get("refused")
    if run is not None:
        if isinstance(run, Mapping):
            out.append(f"the run is refused: {run.get('key')}: {run.get('reason')}")
        else:
            out.append(f"the run is refused: {run!r}")
    shots = manifest.get("shots")
    if not isinstance(shots, list):
        out.append(f"the manifest has no shot list (shots = {shots!r})")
        return out
    for s in shots:
        if not isinstance(s, Mapping):
            out.append(f"a shot entry is not an object: {s!r}")
        elif s.get("status") != "ok":
            out.append(f"shot {s.get('name')!r} is {s.get('status')}: {s.get('reason')}")
    return out


def render_stills(
    spec: Mapping[str, Any] | str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    viewer: str | os.PathLike[str] | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Take every shot of a view spec with apeGmshViewer and return its manifest.

    Parameters
    ----------
    spec
        An ``apegmsh-view/1`` spec as a dict, or the path of a spec file.
        A dict's relative ``files`` paths are read against the working
        folder; a file's, against the file's folder (the app's rule).
    out
        The folder the PNGs and ``manifest.json`` go to. It must be absent
        or empty: the app refuses any other.
    viewer
        The executable; by default :func:`find_viewer`.
    timeout
        Seconds before the run is killed and refused; ``None`` waits.

    Returns
    -------
    dict
        ``manifest.json`` as the app wrote it, when every shot was drawn.
        Each entry's ``png`` is relative to ``out``. Quote values from the
        manifest, never from the pixels (ADR 0128 D3).

    Raises
    ------
    AgentViewError
        The spec, a shot or ``out`` was refused, or the run failed. The
        message names every refusal, and ``.manifest`` holds this run's
        manifest when it wrote one (the shots that were drawn are in it).
    FileNotFoundError
        No executable was found (:func:`find_viewer`).
    """
    exe = Path(viewer) if viewer is not None else find_viewer()
    out_dir = Path(out).resolve()
    manifest_path = out_dir / MANIFEST_FILE
    # A manifest already in `out` is an earlier run's, and the app refuses a
    # folder that is not empty: never read it as this run's evidence.
    stale = manifest_path.exists()
    with tempfile.TemporaryDirectory(prefix="apegmsh-view-") as tmp:
        if isinstance(spec, Mapping):
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text(
                json.dumps(_absolute_files(spec), indent=2, allow_nan=False), encoding="utf-8"
            )
        else:
            spec_path = Path(spec).resolve()
        argv = [str(exe), "--render", str(spec_path), "--out", str(out_dir)]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
            raise AgentViewError(
                f"apeGmshViewer --render did not exit within {timeout} s and was killed; "
                f"nothing it wrote in {out_dir} is evidence",
                returncode=None, manifest=None, stderr=err,
            ) from exc
    stderr = proc.stderr or ""
    manifest = None if stale else _read_manifest(manifest_path)
    refusals = _refusals(manifest) if manifest is not None else []
    if proc.returncode == 0 and manifest is not None and not refusals:
        return manifest
    if proc.returncode == 0:
        why = (
            f"it wrote no {MANIFEST_FILE} in {out_dir}" if manifest is None
            else "its manifest states: " + "; ".join(refusals)
        )
        reason = f"apeGmshViewer --render exited 0, but {why}"
    else:
        reason = f"apeGmshViewer --render exited {proc.returncode}"
        if refusals:
            reason += ": " + "; ".join(refusals)
        elif stale:
            reason += f" ({manifest_path} was there before the run, so it is not read)"
    tail = stderr.strip()
    if tail:
        reason += f"\nstderr: {tail}"
    raise AgentViewError(reason, returncode=proc.returncode, manifest=manifest, stderr=stderr)
