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

The spec helpers at the end (``ladruno_files``, ``shoot``,
``mesh_spec``) are what apeGmsh's own still verbs build their specs with on
Windows (ADR 0128 slice 6). ``ladruno_files`` refuses apeGmsh inputs the app cannot
open, by the #2655 ruling; they still do not check the spec.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "CAMERAS",
    "MODE_FIELDS",
    "NODAL_FIELDS",
    "SPEC_VERSION",
    "AgentViewError",
    "find_viewer",
    "ladruno_files",
    "mesh_spec",
    "render_stills",
    "shoot",
]

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


def _kill_tree(proc: subprocess.Popen[str]) -> str:
    """Kill the process the shim started and every process under it; return its stderr.

    ``Popen.kill`` reaches the child alone, and its children (Electron's GPU
    and renderer processes, or the app under a ``.cmd`` wrapper) keep the
    pipes open, so reading them would wait until those exit on their own.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()
    try:
        _, err = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        return ""
    return err or ""


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
        # Its own process group (POSIX), so a timeout can kill the app's children too.
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", start_new_session=sys.platform != "win32",
        )
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            err = _kill_tree(proc)
            raise AgentViewError(
                f"apeGmshViewer --render did not exit within {timeout} s and was killed; "
                f"nothing it wrote in {out_dir} is evidence",
                returncode=None, manifest=None, stderr=err,
            ) from exc
        except BaseException:
            # Ctrl-C or any other interruption: never leave the app's tree running.
            _kill_tree(proc)
            raise
    stderr = stderr or ""
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


# =====================================================================
# Spec helpers for apeGmsh's own stills (ADR 0128 slice 6, VA6b)
# =====================================================================
#
# The parity plan (internal_docs/plan_viewer_agent_stills_parity.md)
# maps apeGmsh's still verbs onto apegmsh-view/1. These helpers are the
# app's contract: the spec version, the v1 camera and field names, the
# stem rule for the files a spec names, and a run whose PNGs land where
# the verb was asked. ladruno_files refuses a results file the app cannot
# open; nothing here checks the spec.

#: The v1 camera tokens as ``apegmsh-view/1`` cameras (parity plan
#: R4-R7): ``xy`` looks along -z, ``xz`` along +y, ``yz`` along -x.
#: ``iso`` becomes the app's own iso (R4: the look changes).
CAMERAS: dict[str, str] = {"iso": "iso", "xy": "top", "xz": "-y", "yz": "+x"}

#: The ``apegmsh-view/1`` version every spec built here names.
SPEC_VERSION = "apegmsh-view/1"

#: ``<stem>.part-<N>.ladruno``, a partitioned run's part file (the app's
#: ``main/pairing.ts`` PART_FILE rule).
_LADRUNO_PART = re.compile(r"\.part-\d+\.ladruno$", re.IGNORECASE)
_LADRUNO_SUFFIX = ".ladruno"

#: A Python nodal prefix as the ``.ladruno`` ON_NODES group the app
#: names its field after (fork schema ``ladruno_schema_v1.md`` §7.1.1,
#: §7.1.2; the reader's ``readers/_buckets._NODAL_RESULT_NAME_MAP``
#: read backwards, the static group first as ``_NODAL_PRECEDENCE``
#: orders it). A mode stage records ``MODES_OF_VIBRATION(U|R)`` instead.
NODAL_FIELDS: dict[str, str] = {
    "displacement": "DISPLACEMENT",
    "rotation": "ROTATION",
    "velocity": "VELOCITY",
    "angular_velocity": "ANGULAR_VELOCITY",
    "acceleration": "ACCELERATION",
    "angular_acceleration": "ANGULAR_ACCELERATION",
    "reaction_force": "REACTION_FORCE",
    "reaction_moment": "REACTION_MOMENT",
}
MODE_FIELDS: dict[str, str] = {
    "displacement": "MODES_OF_VIBRATION(U)",
    "rotation": "MODES_OF_VIBRATION(R)",
}


def ladruno_files(
    results_path: "str | Path", model_h5: "str | Path | None" = None,
) -> dict[str, str]:
    """The spec's ``files`` for a ``.ladruno`` results file (R22b, R31, G8).

    The viewer and v2 stills read ``.ladruno`` only (ruled 2026-11-02 on
    #2655): any other results file is refused by name. The app opens a
    ``.ladruno`` as the set of its stem, so the model is
    ``<stem>.h5`` beside it; a ``model_h5`` that is another file is
    refused rather than silently replaced.
    """
    path = Path(results_path)
    part = _LADRUNO_PART.search(path.name)
    if part is not None:
        stem = path.name[: part.start()]
    elif path.name.lower().endswith(_LADRUNO_SUFFIX) and len(path.name) > len(_LADRUNO_SUFFIX):
        stem = path.name[: -len(_LADRUNO_SUFFIX)]
    else:
        raise ValueError(
            f"apeGmshViewer stills read .ladruno results only (ADR 0122 D1; "
            f"the #2655 ruling); {path.name} is not a .ladruno. Write the "
            f"run's results as <stem>.ladruno beside its <stem>.h5 model."
        )
    model = path.parent / f"{stem}.h5"
    if model_h5 is not None and Path(model_h5).resolve() != model.resolve():
        raise ValueError(
            f"apeGmshViewer opens {path.name} with its own model {model.name} "
            f"(the stem rule); model_h5={str(model_h5)!r} is another file. "
            f"Put the model beside the results as {model.name}."
        )
    if not model.is_file():
        raise FileNotFoundError(
            f"apeGmshViewer opens {path.name} with the model {model} beside "
            f"it, which does not exist. Write the model there "
            f"(ops.h5({model.name!r}) or fem.to_h5)."
        )
    return {"model": str(model.resolve()), "results": str(path.resolve())}


def shoot(
    spec: Mapping[str, Any], targets: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Take every shot of ``spec`` with apeGmshViewer; move each PNG to its target.

    ``targets`` maps a shot name to the path its PNG goes to. The app
    writes into a fresh temporary folder (it refuses one that is not
    empty), so a target folder that already holds files is fine. Returns
    the manifest and the written paths by shot name. Raises
    ``AgentViewError`` on any refusal (G9) and ``FileNotFoundError``
    without the app; nothing is moved then.
    """
    names = [shot["name"] for shot in spec["shots"]]
    if sorted(names) != sorted(targets):
        raise ValueError(
            f"the spec's shots {names} and the still targets "
            f"{sorted(targets)} differ"
        )
    with tempfile.TemporaryDirectory(prefix="apegmsh-stills-") as tmp:
        out = Path(tmp) / "out"
        manifest = render_stills(spec, out)
        written: dict[str, Path] = {}
        for entry in manifest["shots"]:
            name = entry["name"]
            dest = Path(targets[name])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out / entry["png"]), str(dest))
            written[name] = dest
    missing = sorted(set(targets) - set(written))
    if missing:
        raise RuntimeError(
            f"apeGmshViewer reported success but its manifest has no "
            f"entry for the shots {missing}"
        )
    return manifest, written


def mesh_spec(model_path: str | os.PathLike[str], *, camera: str, size: list[int]) -> dict[str, Any]:
    """The spec of one undeformed mesh shot of a model file (R1)."""
    if camera not in CAMERAS.values():
        raise ValueError(f"camera {camera!r} is not one of {sorted(CAMERAS.values())}")
    return {
        "spec": SPEC_VERSION,
        "files": {"model": str(Path(model_path).resolve())},
        "size": list(size),
        "shots": [{"name": "mesh", "camera": camera}],
    }
