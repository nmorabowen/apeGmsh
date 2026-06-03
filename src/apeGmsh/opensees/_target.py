"""OpenSees runtime target + capability resolution.

apeGmsh hands a model to OpenSees over three independent paths, each
binding a *different* runtime:

* **live / in-process** — ``ops.run()`` / ``ops.analyze()`` drive
  ``import openseespy.opensees`` from the active interpreter;
* **Tcl subprocess** — ``ops.tcl(path, run=True)`` shells out to an
  ``OpenSees`` Tcl binary;
* **openseespy subprocess** — ``ops.py(path, run=True)`` shells out to a
  python interpreter that has openseespy installed.

:class:`OpenSeesTarget` is the single, explicit seam that says *which*
runtime each subprocess path binds, and asserts an expectation on the
live path.  Without it, resolution falls back to environment variables
(``$OPENSEES_BIN`` / ``$OPENSEES_VENV``) and ``PATH`` exactly as before
— the target only ever *overrides* that fallback, never removes it.

Fork-ness is **not** a path.  Pointing ``binary=`` at the Ladruno fork
build does not, by itself, tell apeGmsh the build has ``BezierTet10`` —
that stays a capability detected at the point of use (see
:mod:`apeGmsh.opensees.emitter.live`).  ``require_fork`` is the one
place a target carries a fork *expectation*, and it governs only the
**live** path: you cannot swap ``import openseespy`` under a running
interpreter, so ``binary`` / ``python`` are inert for live execution and
``require_fork`` simply fails loud at the ``run()`` / ``analyze()``
boundary instead of three primitives deep.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenSeesTarget:
    """Which OpenSees runtime the subprocess paths bind, set once on the bridge.

    Parameters
    ----------
    binary
        Path to the OpenSees **Tcl** binary used by ``ops.tcl(run=True)``.
        Overrides ``$OPENSEES_BIN`` and ``shutil.which("OpenSees")``.  An
        explicit ``ops.tcl(bin=...)`` argument still wins over this.
    python
        Path to a **python interpreter with openseespy** used by
        ``ops.py(run=True)``.  Overrides ``$OPENSEES_VENV`` and
        ``shutil.which("python")``.  An explicit ``ops.py(python=...)``
        argument still wins over this.
    require_fork
        When ``True``, the **live** path (``ops.run()`` / ``ops.analyze()``)
        asserts the in-process openseespy is the Ladruno fork build before
        driving any primitive, raising a clear error otherwise.  Inert for
        the subprocess paths (a stock build there fails loud on the first
        fork-only command anyway).
    """

    binary: str | None = None
    python: str | None = None
    require_fork: bool = False


@dataclass(frozen=True)
class OpenSeesCapabilities:
    """What the **live** in-process openseespy build can do.

    Probed by :meth:`apeGmsh.opensees.apeSees.capabilities`.  ``has_fork``
    is a heuristic: the Ladruno fork registers the fork-only ``profiler``
    command, which stock openseespy lacks, so its presence is the fork
    signal (the same gate the live emitter uses for ``ops.profiler``).
    """

    source: str
    """Where the build was probed — ``"live"`` (in-process openseespy)."""
    has_fork: bool
    """True if the build looks like the Ladruno fork (see class docstring)."""
    has_profiler: bool
    """True if the fork-only ``profiler`` command is present."""
    version: str | None
    """``ops.version()`` string if the build exposes one, else ``None``."""


def resolve_opensees_binary(
    explicit: str | None, target: OpenSeesTarget | None
) -> str:
    """Resolve the OpenSees Tcl binary path.

    Precedence: explicit ``bin=`` argument → ``target.binary`` →
    ``$OPENSEES_BIN`` → ``shutil.which("OpenSees")``.  Raises
    :class:`FileNotFoundError` if none resolve.
    """
    if explicit is not None:
        return explicit
    if target is not None and target.binary is not None:
        return target.binary
    env = os.environ.get("OPENSEES_BIN")
    if env:
        return env
    on_path = shutil.which("OpenSees")
    if on_path:
        return on_path
    raise FileNotFoundError(
        "OpenSees Tcl binary not found. Tried: bin= argument, "
        "OpenSeesTarget(binary=...), $OPENSEES_BIN environment variable, "
        "shutil.which('OpenSees'). Set one of these or install OpenSees "
        "on PATH."
    )


def resolve_python_binary(
    explicit: str | None, target: OpenSeesTarget | None
) -> str:
    """Resolve the python interpreter to run an openseespy script.

    Precedence: explicit ``python=`` argument → ``target.python`` →
    ``$OPENSEES_VENV``'s python → ``shutil.which("python")`` →
    ``sys.executable``.  Always resolves (falls back to the running
    interpreter).
    """
    if explicit is not None:
        return explicit
    if target is not None and target.python is not None:
        return target.python
    venv = os.environ.get("OPENSEES_VENV")
    if venv:
        if os.name == "nt":
            candidate = os.path.join(venv, "Scripts", "python.exe")
        else:
            candidate = os.path.join(venv, "bin", "python")
        if os.path.exists(candidate):
            return candidate
    on_path = shutil.which("python")
    if on_path:
        return on_path
    return sys.executable


def probe_live_capabilities() -> OpenSeesCapabilities:
    """Introspect the in-process openseespy build.

    Imports openseespy in the active interpreter and reports what it can
    do.  Raises whatever :func:`_get_ops` raises if openseespy is not
    installed.
    """
    from .emitter.live import _get_ops

    ops = _get_ops()
    has_profiler = hasattr(ops, "profiler")
    version: str | None
    try:
        version = str(ops.version())  # type: ignore[attr-defined]
    except Exception:
        version = None
    return OpenSeesCapabilities(
        source="live",
        has_fork=has_profiler,
        has_profiler=has_profiler,
        version=version,
    )
