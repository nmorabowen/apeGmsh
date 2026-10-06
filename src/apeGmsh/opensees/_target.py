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
:mod:`apeGmsh.opensees.emitter.live`).  ``require_fork`` (or
``mode="fork"``) is the one place a target carries a fork
*expectation*, and it governs only the **live** path: you cannot swap
``import openseespy`` under a running interpreter, so ``binary`` /
``python`` are inert for live execution and ``require_fork`` simply
fails loud at the ``run()`` / ``analyze()`` boundary instead of three
primitives deep.

What the imported binary *is* has one answer, :class:`BackendInfo`,
computed by :func:`backend_info_of` from a single signal: the fork-only
``ladrunoBuild()`` command returning a git sha.  The live resolver
(:func:`apeGmsh.opensees.emitter.live.get_backend_info`) owns it; the
backend name, the build stamp and :attr:`OpenSeesCapabilities.has_fork`
all derive from it, so they cannot disagree.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Literal

#: What a backend is: the Ladruno fork, or stock OpenSees.
BackendKind = Literal["fork", "stock"]

#: How a target picks its backend kind: ``"auto"`` follows the imported
#: binary's :class:`BackendInfo`; ``"fork"`` / ``"stock"`` pin it.
TargetMode = Literal["auto", "fork", "stock"]

_TARGET_MODES: frozenset[str] = frozenset({"auto", "fork", "stock"})

#: A full git commit hash, which is what ``ladrunoBuild()`` returns on a
#: fork build compiled from a git checkout (CMake stamps
#: ``git log -1 --format=%H``; a build outside git answers ``"unknown"``).
_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class BackendInfo:
    """What the imported OpenSees binary is: the resolver's one verdict.

    The **only** fork signal is ``ops.ladrunoBuild()`` returning a 40-char
    git sha (fork PR #718, 2026-08-10).  Consequences, each deliberate:

    * a fork build predating ``ladrunoBuild`` reads as ``"stock"``;
    * a ``ladrunoBuild()`` that raises, or returns a non-string, an empty
      string or anything that is not a sha (``"unknown"`` from a build
      outside git), reads as ``"stock"`` with ``build=None``;
    * the probe never raises.

    Built by :func:`backend_info_of`; the live resolver caches the one for
    the module it bound
    (:func:`apeGmsh.opensees.emitter.live.get_backend_info`).
    """

    kind: BackendKind
    """``"fork"`` iff ``build`` is a sha, else ``"stock"``."""
    build: str | None
    """The 40-char git sha from ``ladrunoBuild()``, or ``None``."""
    version: str | None
    """``ops.version()`` as a string, or ``None`` if it is absent or raises."""
    source: str
    """The module actually imported: its ``__file__``, else its ``__name__``.

    Beside a fork install, a ``.pth`` may alias ``openseespy.opensees`` to
    the fork module; this names the file that was really bound.
    """


def _build_stamp(ops: Any) -> str | None:
    """``ops.ladrunoBuild()`` when it returns a sha, else ``None``; never raises."""
    fn = getattr(ops, "ladrunoBuild", None)  # apegmsh-lint: getattr-undefined-ok fork-only OpenSees command (fork PR #718), the one fork signal
    if not callable(fn):
        return None
    try:
        raw = fn()
    except Exception:
        return None
    if not isinstance(raw, str):
        return None
    stamp = raw.strip()
    return stamp if _SHA_RE.fullmatch(stamp) else None


def backend_info_of(ops: Any) -> BackendInfo:
    """Classify the OpenSees module ``ops`` (see :class:`BackendInfo`).

    Pure: reads the module and never imports one.  The live resolver calls
    it on the module it bound; tests call it on fakes.
    """
    build = _build_stamp(ops)
    version: str | None
    version_fn = getattr(ops, "version", None)
    try:
        version = str(version_fn()) if callable(version_fn) else None
    except Exception:
        version = None
    source = getattr(ops, "__file__", None) or getattr(ops, "__name__", None)
    return BackendInfo(
        kind="fork" if build is not None else "stock",
        build=build,
        version=version,
        source=str(source) if source else type(ops).__name__,
    )


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
        fork-only command anyway).  ``require_fork=True`` is
        ``mode="fork"``; construction keeps the two in step.
    mode
        Which backend kind the target means.  ``"auto"`` (the default) is
        whatever :class:`BackendInfo` reports for the imported binary
        (:meth:`resolve_kind`).  ``"fork"`` pins the fork and sets
        ``require_fork=True``, so the live path asserts it.  ``"stock"``
        pins stock: a declaration for provenance and the subprocess paths,
        which asserts nothing about the in-process build.
        ``mode="stock"`` with ``require_fork=True`` contradicts itself and
        raises :class:`ValueError`, as does any other mode string.
    """

    binary: str | None = None
    python: str | None = None
    require_fork: bool = False
    mode: TargetMode = "auto"

    def __post_init__(self) -> None:
        if self.mode not in _TARGET_MODES:
            raise ValueError(
                f"OpenSeesTarget(mode={self.mode!r}): mode must be one of "
                f"{sorted(_TARGET_MODES)}."
            )
        if self.require_fork and self.mode == "stock":
            raise ValueError(
                "OpenSeesTarget(require_fork=True, mode='stock') contradicts "
                "itself: require_fork means mode='fork'. Drop one of them."
            )
        # require_fork keeps meaning fork, and a fork pin keeps the live
        # assertion (the bridge reads ``require_fork``): set both, so the
        # two spellings build equal targets.
        if self.require_fork or self.mode == "fork":
            object.__setattr__(self, "mode", "fork")
            object.__setattr__(self, "require_fork", True)

    def resolve_kind(self, info: BackendInfo) -> BackendKind:
        """The backend kind this target means for the binary ``info`` describes.

        ``mode="auto"`` returns ``info.kind``; a pinned mode returns itself.
        """
        if self.mode == "auto":
            return info.kind
        return "fork" if self.mode == "fork" else "stock"


@dataclass(frozen=True)
class OpenSeesCapabilities:
    """What the **live** in-process openseespy build can do.

    Probed by :meth:`apeGmsh.opensees.apeSees.capabilities`.  ``has_fork``,
    ``version`` and ``build`` come from the resolver's one
    :class:`BackendInfo` (``kind == "fork"``: ``ladrunoBuild()`` returned a
    sha), the same verdict that names the backend (``get_backend_name()``),
    so the two cannot disagree.  ``has_profiler`` reports the ``profiler``
    command itself.
    """

    source: str
    """Where the build was probed — ``"live"`` (in-process openseespy)."""
    has_fork: bool
    """True if the build looks like the Ladruno fork (see class docstring)."""
    has_profiler: bool
    """True if the fork-only ``profiler`` command is present."""
    version: str | None
    """``ops.version()`` string if the build exposes one, else ``None``."""
    has_ladruno_up: bool = False
    """True if the build is expected to know ``element LadrunoUP`` (ADR 0074).

    Heuristic — openseespy exposes no per-element registry, so this mirrors
    ``has_fork`` (fork builds since ADR-71 P4 ship LadrunoUP).  A fork build
    predating P4 would report ``True`` here yet still fail loud at the first
    ``LadrunoUP`` element line (the live emitter's fork-element verification
    stays the authoritative gate); scripts use this key only to branch.
    """
    build: str | None = None
    """The engine's build stamp — the 40-char git hash the binary was compiled
    from (``ops.ladrunoBuild()``, fork PR #718) — or ``None`` on stock
    openseespy / a fork build predating that command.

    Unlike ``version`` (a release string shared by every build of a release),
    this identifies the exact commit, so a harness can pin the engine it ran
    against.  It is the machine-readable replacement for scraping the splash
    banner, which library and test runs suppress.
    """


#: Minimum fork build (``ops.ladrunoBuild()``) for the 2026-09-07 TIMs batch
#: (fork PRs #805, #808, #810, #811, #812, #814, #820, #821 — the fork's
#: ``ladruno_apegmsh_adoption_guide_2026-09-07.md``).  Documented, not
#: enforced (same as :data:`~apeGmsh.opensees.material.nd.ASDP_MIN_FORK_BUILD`
#: — a bare hash cannot prove ancestry).  An older fork: ``zeroLength`` /
#: contact refuse an ndf-4 u-p node in 3D (ADR 96), ``LadrunoSANISAND``
#: answers no ``psi`` / ``yieldDistance`` and writes ``C1..Cn`` for its
#: IMPL-EX responses, and ``system Pardiso -stats`` prints the old
#: once-per-pattern lines instead of the per-factorisation block.
TIMS_FORK_BATCH_MIN_BUILD = "a240b9183"


def _resolve_binary_in_dir(path: str) -> str:
    """Descend into *path* if it names a directory, else return it unchanged.

    A directory is treated as a ``dist/bin``-style folder: look for
    ``OpenSees.exe`` (Windows) / ``OpenSees`` (elsewhere) inside it.
    Raises :class:`FileNotFoundError` naming the directory if the binary
    is not there — the alternative is handing the directory straight to
    ``CreateProcess``, which fails with an opaque ``WinError 5``.
    """
    if os.path.isdir(path):
        exe_name = "OpenSees.exe" if os.name == "nt" else "OpenSees"
        candidate = os.path.join(path, exe_name)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"{path!r} is a directory but does not contain {exe_name!r}. "
            "Pass the OpenSees dist/bin directory (it will be searched "
            "for the binary) or a direct path to the executable."
        )
    return path


def resolve_opensees_binary(
    explicit: str | None, target: OpenSeesTarget | None
) -> str:
    """Resolve the OpenSees Tcl binary path.

    Precedence: explicit ``bin=`` argument → ``target.binary`` →
    ``$OPENSEES_BIN`` → ``shutil.which("OpenSees")``.  At every
    precedence level a directory (e.g. a fork ``dist/bin``) is searched
    for ``OpenSees.exe`` / ``OpenSees`` inside it; a file path is used
    as-is.  Raises :class:`FileNotFoundError` if none resolve (naming the
    directory if one was given but did not contain the binary).
    """
    if explicit is not None:
        return _resolve_binary_in_dir(explicit)
    if target is not None and target.binary is not None:
        return _resolve_binary_in_dir(target.binary)
    env = os.environ.get("OPENSEES_BIN")
    if env:
        return _resolve_binary_in_dir(env)
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
    installed.  ``has_fork`` / ``version`` / ``build`` are the resolver's
    :class:`BackendInfo`, never a second probe.
    """
    from .emitter.live import _get_ops, get_backend_info

    ops = _get_ops()
    info = get_backend_info()
    has_fork = info.kind == "fork"
    return OpenSeesCapabilities(
        source="live",
        has_fork=has_fork,
        has_profiler=hasattr(ops, "profiler"),
        version=info.version,
        has_ladruno_up=has_fork,
        build=info.build,
    )
