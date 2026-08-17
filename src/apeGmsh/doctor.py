"""``python -m apeGmsh doctor`` — one-shot environment preflight.

Runs a fixed battery of checks against the *current* interpreter and
prints a short markdown report of coded findings. Exit code is ``0``
when no error-severity findings exist, ``1`` otherwise.

Finding codes
-------------
``D0``  A doctor check itself crashed (always an error — report it).
``D1``  Interpreter identity — which python is running, and whether it
        is the reference venv named by ``APEGMSH_REFERENCE_VENV`` when
        that variable is set. Catches the classic wrong-interpreter
        ``ModuleNotFoundError`` that looks like a real bug but is only a
        second Python on the machine.
``D2``  apeGmsh import path — the imported package tree vs the
        pip-registered editable install (PYTHONPATH / cwd drift).
``D3``  ``import gmsh`` works (hard dependency).
``D4``  Viewer stack / GL: Qt binding + pyvista/pyvistaqt presence and
        the ``QT_QPA_PLATFORM=offscreen`` trap — on Windows the
        offscreen platform cannot host the VTK render window, and
        ``ViewerWindow`` refuses to start (see the guard in
        ``apeGmsh/viewers/ui/viewer_window.py``).
``D5``  Ladruno OpenSees fork importable as the live backend.
``D6``  Version agreement for the distributions named by
        ``APEGMSH_SHARED_PACKAGES`` across this interpreter and the
        reference venv. Opt-in and empty by default: it guards packages
        installed *non-editably* into several interpreters, which drift
        apart silently.

Caveats
-------
``python -m apeGmsh`` imports the ``apeGmsh`` package (and its hard
dependencies, notably ``gmsh``) *before* this module runs. If the
doctor itself dies with ``ModuleNotFoundError``, that traceback IS the
diagnosis: you are almost certainly on the wrong interpreter (D1) —
re-run with the python of the environment apeGmsh is installed into.

That chicken-and-egg is why this module also runs **standalone**::

    <any-python> path\\to\\src\\apeGmsh\\doctor.py

which needs nothing importable except the standard library. Point a
suspect interpreter at it and D1/D2 tell you whether apeGmsh is even
reachable from there, instead of dying on the package import.

Some environments install the Ladruno fork through a ``.pth`` boot hook
(``ladruno_opensees.pth`` → ``_ladruno_opensees_boot``), which prints
the fork's splash banner **to stdout at interpreter startup** — before
any user code, this module included, can run. ``apeGmsh/__init__.py``
defaults ``LADRUNO_OPENSEES_QUIET=1`` for every *later* fork boot, but
nothing in Python can pre-empt the ``.pth``. So the report may still be
preceded by that banner; for clean stdout, set the variable in the
environment *of the launching shell*::

    LADRUNO_OPENSEES_QUIET=1 python -m apeGmsh doctor

Design notes
------------
* Mirrors the frozen-dataclass issue/report shape of
  :mod:`apeGmsh.cuts._preflight` (``PreflightIssue`` /
  ``PreflightReport``). :class:`DoctorFinding` is deliberately
  **field-identical** to the ``Finding`` in ADR 0094 (``code`` /
  ``severity`` / ``message`` / ``detail``, including the ``"info"``
  severity that ``_preflight`` lacks), so once the ``apeGmsh.assess``
  package lands the swap is an import change, not a migration.
  :class:`DoctorReport` deviates on purpose: ADR 0094's
  ``AssessmentReport`` stores rendered ``text`` as a field, while this
  keeps the header facts (interpreter, version, package dir) as fields
  and renders markdown on demand via :meth:`DoctorReport.to_markdown`.
  Note the codes here (``D1`` …) are the doctor's own catalog, not
  entries in that ADR's closed catalog (INV-6); folding them in would
  mean renaming to its dotted form.
* No Qt / VTK / GL imports, module level or otherwise: the viewer
  check (D4) only inspects ``importlib.util.find_spec`` and
  environment variables, so the doctor runs on GL-less CI.
* Native-backend probes (D5, and the D6 counterparts) run in
  subprocesses, so a hard-crashing DLL import (access violation from a
  stale ``opensees.pyd``) cannot take the doctor down with it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

Severity = Literal["error", "warning", "info"]

#: Environment variable naming a *reference* virtualenv root — the
#: interpreter this machine's work is expected to run under. Unset on a
#: normal install, and then D1 simply reports which interpreter is
#: running instead of comparing against anything.
#:
#: This used to be a literal path to one developer's venv, which meant
#: ``python -m apeGmsh doctor`` printed that person's home directory on
#: every machine in the world that ran it. A site convention does not
#: belong in a published wheel; it belongs in the environment of the
#: site that has the convention.
_REFERENCE_VENV_ENV = "APEGMSH_REFERENCE_VENV"

#: Comma-separated distributions that must agree in version across the
#: reference venv and this one (D6). Empty by default: this only matters
#: for packages installed *non-editably* into several interpreters, where
#: the copies can silently drift apart. Nothing in apeGmsh's own
#: dependency set works that way, so there is no sensible default to ship.
_SHARED_PACKAGES_ENV = "APEGMSH_SHARED_PACKAGES"


def _reference_venv() -> Path | None:
    """The configured reference venv root, or ``None`` when unset."""
    raw = os.environ.get(_REFERENCE_VENV_ENV, "").strip()
    return Path(raw) if raw else None


def _shared_packages() -> tuple[str, ...]:
    """Distributions whose versions must agree across interpreters."""
    raw = os.environ.get(_SHARED_PACKAGES_ENV, "")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def _venv_python(root: Path) -> Path:
    """The interpreter inside a venv root, on either layout."""
    win = root / "Scripts" / "python.exe"
    return win if win.exists() else root / "bin" / "python"


#: Subprocess probe timeout, seconds. Importing an OpenSees backend
#: cold can take a few seconds; a hung probe should not hang the doctor.
_PROBE_TIMEOUT = 60.0


@dataclass(frozen=True)
class DoctorFinding:
    """One finding from :func:`run_doctor`.

    Parameters
    ----------
    code:
        Stable check code (``"D1"`` … ``"D6"``, or ``"D0"`` for a
        crashed check). See the module docstring for the catalog.
    severity:
        ``"error"`` if the environment cannot support the checked
        feature; ``"warning"`` for degraded-but-usable configurations;
        ``"info"`` for healthy/neutral observations.
    message:
        One-line human-readable summary (rendered into the markdown).
    detail:
        Optional structured payload (paths, versions, stderr tails).
        Caller-inspectable; not formatted into :attr:`message`.
    """

    code: str
    severity: Severity
    message: str
    detail: Mapping[str, object] | None = None


@dataclass(frozen=True)
class DoctorReport:
    """Result of :func:`run_doctor` — header facts plus flat findings."""

    python: str
    executable: str
    apegmsh_version: str
    package_dir: str
    findings: tuple[DoctorFinding, ...]

    @property
    def errors(self) -> tuple[DoctorFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[DoctorFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def ok(self) -> bool:
        """``True`` when the report has no error-severity findings."""
        return not self.errors

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def to_markdown(self) -> str:
        lines = [
            "# apeGmsh doctor",
            "",
            f"- Python {self.python} — `{self.executable}`",
            f"- apeGmsh {self.apegmsh_version} — `{self.package_dir}`",
            "",
            "## Findings",
            "",
        ]
        tag = {"error": "ERROR", "warning": "warn ", "info": "ok   "}
        for f in self.findings:
            lines.append(f"- **[{tag[f.severity]} {f.code}]** {f.message}")
        n_err, n_warn = len(self.errors), len(self.warnings)
        lines += ["", "## Result", ""]
        if self.ok:
            suffix = f" ({n_warn} warning(s))" if n_warn else ""
            lines.append(
                f"No errors{suffix} — environment looks usable. Exit code 0."
            )
        else:
            lines.append(
                f"{n_err} error(s), {n_warn} warning(s). Exit code 1."
            )
        return "\n".join(lines)


# --------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------- #
def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def _is_under(child: Path, root: Path) -> bool:
    try:
        c = Path(os.path.normcase(str(child.resolve())))
        r = Path(os.path.normcase(str(root.resolve())))
        return c == r or c.is_relative_to(r)
    except (OSError, ValueError):
        return False


def _linked_worktree_gitdir(pkg_dir: Path) -> Path | None:
    """The gitdir a linked git worktree containing ``pkg_dir`` points at.

    ``None`` when no enclosing tree is a linked worktree. The signal is
    structural and needs no ``git`` subprocess: a linked worktree's
    ``.git`` is a *file* holding ``gitdir: <path>``, whereas an ordinary
    checkout's ``.git`` is a directory.

    Searches a few levels up because the package may sit at
    ``<tree>/src/apeGmsh`` (src layout) or ``<tree>/apeGmsh`` (flat).
    """
    for parent in list(pkg_dir.parents)[:4]:
        dot_git = parent / ".git"
        if dot_git.is_dir():
            return None  # ordinary checkout — stop, don't look further up
        if not dot_git.is_file():
            continue
        try:
            text = dot_git.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        if not text.startswith("gitdir:"):
            return None
        return Path(text.split(":", 1)[1].strip())
    return None


def _find_spec_ok(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------- #
# Checks — each returns a list of findings and never raises on a
# *diagnosed* condition; unexpected crashes are caught by run_doctor.
# --------------------------------------------------------------------- #
def _check_interpreter() -> list[DoctorFinding]:
    """D1 — which interpreter is running, and whether it is the expected one.

    With no reference venv configured this is a plain identity report:
    the useful "can this interpreter actually do the work" questions are
    answered by D2 (apeGmsh importable), D3 (gmsh) and D5 (a backend),
    which do not need to know about any site convention.
    """
    exe = Path(sys.executable)
    pyver = ".".join(str(v) for v in sys.version_info[:3])
    in_venv = sys.prefix != sys.base_prefix
    reference = _reference_venv()
    detail: dict[str, object] = {
        "executable": str(exe),
        "prefix": sys.prefix,
        "in_virtualenv": in_venv,
        "reference_venv": str(reference) if reference else None,
    }
    kind = "virtualenv" if in_venv else "system interpreter"

    if reference is None:
        return [DoctorFinding(
            code="D1",
            severity="info",
            message=f"Python {pyver} at {exe} ({kind}).",
            detail=detail,
        )]
    if not reference.exists():
        return [DoctorFinding(
            code="D1",
            severity="warning",
            message=(
                f"Python {pyver} at {exe} ({kind}). "
                f"{_REFERENCE_VENV_ENV} points at {reference}, which does "
                "not exist — fix or unset the variable."
            ),
            detail=detail,
        )]
    if _same_path(Path(sys.prefix), reference):
        return [DoctorFinding(
            code="D1",
            severity="info",
            message=(
                f"Reference interpreter (Python {pyver}, {exe})."
            ),
            detail=detail,
        )]
    return [DoctorFinding(
        code="D1",
        severity="warning",
        message=(
            f"Python {pyver} at {exe} ({kind}) is not the reference venv "
            f"{reference} named by {_REFERENCE_VENV_ENV}. Packages "
            "installed only there will be missing here — a "
            "ModuleNotFoundError under this interpreter usually means "
            "wrong interpreter, not a broken install."
        ),
        detail=detail,
    )]


def _import_apegmsh():
    """Import apeGmsh, or return ``None`` when it is not reachable.

    ``None`` is only possible on the standalone-script path (see the
    module docstring) — under ``python -m apeGmsh`` the package is
    necessarily imported already.
    """
    try:
        import apeGmsh

        return apeGmsh
    except Exception:
        return None


def _check_import_path() -> list[DoctorFinding]:
    """D2 — imported apeGmsh tree vs the pip-registered install."""
    import importlib.metadata

    _pkg = _import_apegmsh()
    if _pkg is None:
        return [DoctorFinding(
            code="D2",
            severity="error",
            message=(
                "apeGmsh is NOT importable from this interpreter "
                f"({sys.executable}). This is the ModuleNotFoundError "
                "that looks like a bug but is the wrong interpreter — "
                "see D1."
            ),
            detail={"executable": sys.executable},
        )]

    pkg_dir = Path(_pkg.__file__).resolve().parent
    raw_pythonpath = os.environ.get("PYTHONPATH", "")
    shadow = [
        p for p in raw_pythonpath.split(os.pathsep)
        if p and _is_under(pkg_dir, Path(p))
    ]
    detail: dict[str, object] = {
        "package_dir": str(pkg_dir),
        "pythonpath_entries_containing_import": tuple(shadow),
    }

    try:
        dist = importlib.metadata.distribution("apeGmsh")
    except importlib.metadata.PackageNotFoundError:
        return [DoctorFinding(
            code="D2",
            severity="warning",
            message=(
                f"apeGmsh imported from {pkg_dir} but no installed "
                "distribution is registered in this interpreter — "
                "running from a source tree via sys.path/PYTHONPATH. "
                "Prefer an editable install (`pip install -e .`)."
            ),
            detail=detail,
        )]

    url_root: Path | None = None
    editable = False
    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            direct = json.loads(raw)
            editable = bool(direct.get("dir_info", {}).get("editable"))
            url = direct.get("url", "")
            if url.startswith("file:"):
                from urllib.parse import urlsplit
                from urllib.request import url2pathname

                url_root = Path(url2pathname(urlsplit(url).path))
    except Exception:
        pass
    if url_root is not None:
        detail["install_root"] = str(url_root)
    detail["editable"] = editable

    if not editable or url_root is None:
        return [DoctorFinding(
            code="D2",
            severity="info",
            message=f"apeGmsh installed (non-editable); imported from {pkg_dir}.",
            detail=detail,
        )]
    # Match the package dir EXACTLY against where the editable install
    # maps it (src-layout first, then flat). A containment test would
    # call a git worktree at <root>/.claude/worktrees/<x>/src/apeGmsh
    # "no drift" purely because it nests under <root> — which is the
    # very drift worth reporting.
    expected = (url_root / "src" / "apeGmsh", url_root / "apeGmsh")
    detail["expected_package_dirs"] = tuple(str(p) for p in expected)
    if any(_same_path(pkg_dir, cand) for cand in expected if cand.exists()):
        return [DoctorFinding(
            code="D2",
            severity="info",
            message=(
                f"Editable install at {url_root}; the import resolves to "
                "that checkout."
            ),
            detail=detail,
        )]
    via = (
        f" PYTHONPATH entry {shadow[0]} shadows the install." if shadow
        else " A sys.path entry (PYTHONPATH or cwd) shadows the install."
    )
    # A linked git worktree OF THE SAME REPO is expected drift, not a
    # finding: working in one is the normal way to run a branch, and a
    # warning every run is noise. Verified structurally rather than by
    # path shape — the tree's ``.git`` must be a worktree pointer INTO
    # the install root's own ``.git``, so a worktree of some *other*
    # repo shadowing the install still warns.
    gitdir = _linked_worktree_gitdir(pkg_dir)
    if gitdir is not None:
        detail["worktree_gitdir"] = str(gitdir)
        if _is_under(gitdir, url_root / ".git"):
            return [DoctorFinding(
                code="D2",
                severity="info",
                message=(
                    f"Editable install at {url_root}, but the import "
                    f"resolves to a git worktree of that same repo "
                    f"({pkg_dir}) — expected; you are running the "
                    "worktree's code, not the install's checkout."
                ),
                detail=detail,
            )]
    return [DoctorFinding(
        code="D2",
        severity="warning",
        message=(
            f"Import-path drift: apeGmsh imported from {pkg_dir}, but "
            f"the editable install registers {url_root}.{via} Not a git "
            "worktree of that repo, so this is unexpected code."
        ),
        detail=detail,
    )]


def _check_gmsh() -> list[DoctorFinding]:
    """D3 — gmsh importable (hard dependency of apeGmsh)."""
    import importlib.metadata

    try:
        import gmsh
    except Exception as exc:
        return [DoctorFinding(
            code="D3",
            severity="error",
            message=(
                f"`import gmsh` failed: {type(exc).__name__}: {exc}. "
                "gmsh is a hard dependency of apeGmsh — reinstall with "
                "`pip install gmsh` in this interpreter."
            ),
        )]
    api = getattr(gmsh, "GMSH_API_VERSION", None)
    try:
        pkg_ver: str | None = importlib.metadata.version("gmsh")
    except Exception:
        pkg_ver = None
    return [DoctorFinding(
        code="D3",
        severity="info",
        message=f"gmsh importable (API {api}, package {pkg_ver}).",
        detail={"api_version": api, "package_version": pkg_ver,
                "module_file": getattr(gmsh, "__file__", None)},
    )]


def _check_viewer_gl() -> list[DoctorFinding]:
    """D4 — viewer stack presence + Qt platform / display sanity.

    Inspects only ``find_spec`` and environment variables — never
    imports Qt or creates a GL context, so it is safe on GL-less CI.
    """
    qt_platform = os.environ.get("QT_QPA_PLATFORM", "")
    missing = [n for n in ("qtpy", "pyvista", "pyvistaqt") if not _find_spec_ok(n)]
    bindings = [n for n in ("PySide6", "PyQt6", "PySide2", "PyQt5")
                if _find_spec_ok(n)]
    detail: dict[str, object] = {
        "missing": tuple(missing),
        "qt_bindings": tuple(bindings),
        "QT_QPA_PLATFORM": qt_platform or None,
    }

    # Exactly one finding, so the report never pairs an "installed, ok"
    # line with a warning that the viewer cannot actually open.
    if missing or not bindings:
        gaps = list(missing) + ([] if bindings else ["a Qt binding (PySide6)"])
        return [DoctorFinding(
            code="D4",
            severity="warning",
            message=(
                f"Viewer stack incomplete — missing: {', '.join(gaps)}. "
                "The Qt viewers will not open; install with "
                "`pip install apeGmsh[viewer]`."
            ),
            detail=detail,
        )]

    if sys.platform == "win32" and "offscreen" in qt_platform.lower():
        return [DoctorFinding(
            code="D4",
            severity="warning",
            message=(
                f"Viewer stack present ({bindings[0]}) but "
                f"QT_QPA_PLATFORM={qt_platform!r} is set: on Windows the "
                "offscreen platform cannot host the VTK render window, so "
                "ViewerWindow refuses to start (RuntimeError guard in "
                "apeGmsh/viewers/ui/viewer_window.py). Unset it before "
                "opening a viewer; offscreen pyvista screenshots are "
                "unaffected."
            ),
            detail=detail,
        )]

    if (
        sys.platform.startswith("linux")
        and "offscreen" not in qt_platform.lower()
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        return [DoctorFinding(
            code="D4",
            severity="warning",
            message=(
                f"Viewer stack present ({bindings[0]}) but no display "
                "detected (DISPLAY/WAYLAND_DISPLAY unset) — windowed "
                "viewers need a display or xvfb; offscreen screenshots "
                "still work."
            ),
            detail=detail,
        )]

    return [DoctorFinding(
        code="D4",
        severity="info",
        message=(
            f"Viewer stack present (Qt binding: {bindings[0]}; "
            f"QT_QPA_PLATFORM={qt_platform or 'unset'})."
        ),
        detail=detail,
    )]


@dataclass(frozen=True)
class _ProbeResult:
    """Outcome of a subprocess probe — never carries ``None`` streams."""

    ok: bool
    """False when the process could not be launched, timed out, or exited
    non-zero. Distinct from ``returncode`` so callers need one test."""
    returncode: int | None
    stdout: str
    stderr: str

    @property
    def last_stdout_line(self) -> str:
        lines = self.stdout.strip().splitlines()
        return lines[-1] if lines else ""

    @property
    def last_stderr_line(self) -> str:
        lines = self.stderr.strip().splitlines()
        return lines[-1] if lines else "<no stderr>"


def _run_probe(argv: list[str], env: dict[str, str] | None = None) -> _ProbeResult:
    """Run a short python probe and return its streams, decode-safe.

    Decoding is pinned to UTF-8 with ``errors="replace"`` rather than
    ``text=True``: the office venv's ``.pth`` boot hook prints the fork's
    box-drawing splash banner, and under a cp1252 default the reader
    thread dies with ``UnicodeDecodeError``, handing back ``stdout=None``
    — which reads to the caller as a mysterious ``AttributeError``
    instead of a probe result.
    """
    try:
        proc = subprocess.run(
            argv, capture_output=True, timeout=_PROBE_TIMEOUT, env=env,
            encoding="utf-8", errors="replace", check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return _ProbeResult(
            ok=False, returncode=None, stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )
    return _ProbeResult(
        ok=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _probe_env() -> dict[str, str]:
    """Environment for subprocess probes.

    Pins PYTHONPATH so the child imports the *same* apeGmsh tree as the
    running doctor (matters in git worktrees), and silences the import
    banners so stdout stays parseable.
    """
    env = dict(os.environ)
    env["APEGMSH_QUIET"] = "1"
    env["LADRUNO_OPENSEES_QUIET"] = "1"
    _pkg = _import_apegmsh()
    if _pkg is not None:
        src_root = Path(_pkg.__file__).resolve().parent.parent
        prior = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src_root) + (os.pathsep + prior if prior else "")
    return env


def _check_opensees() -> list[DoctorFinding]:
    """D5 — Ladruno OpenSees fork importable as the live backend.

    Probed in a subprocess: a broken native backend (stale
    ``opensees.pyd``) can hard-crash the importing process, which must
    not take the doctor down. The probe goes through
    ``probe_live_capabilities`` so it honors the full backend
    resolution order (``APEGMSH_OPENSEES_BIN`` → bare ``opensees`` →
    stock openseespy).
    """
    if _import_apegmsh() is None:
        return [DoctorFinding(
            code="D5",
            severity="warning",
            message=(
                "OpenSees backend not probed: the probe resolves the "
                "backend through apeGmsh, which is not importable from "
                "this interpreter (see D2)."
            ),
        )]
    code = (
        "import json\n"
        "from apeGmsh.opensees._target import probe_live_capabilities\n"
        "c = probe_live_capabilities()\n"
        "print(json.dumps({'has_fork': c.has_fork, 'version': c.version,"
        " 'build': c.build}))\n"
    )
    probe = _run_probe([sys.executable, "-c", code], env=_probe_env())
    if not probe.ok:
        return [DoctorFinding(
            code="D5",
            severity="warning",
            message=(
                f"No OpenSees live backend importable "
                f"({probe.last_stderr_line}). ops.run()/ops.analyze() "
                "will fail; point APEGMSH_OPENSEES_BIN at the Ladruno "
                "fork's dist\\bin or install openseespy."
            ),
            detail={"returncode": probe.returncode,
                    "stderr_tail": probe.last_stderr_line},
        )]
    try:
        caps = json.loads(probe.last_stdout_line)
    except ValueError:
        return [DoctorFinding(
            code="D5",
            severity="warning",
            message="OpenSees backend probe returned unparseable output.",
            detail={"stdout": probe.stdout[-500:]},
        )]
    if caps.get("has_fork"):
        return [DoctorFinding(
            code="D5",
            severity="info",
            message=(
                f"Ladruno fork backend importable (version "
                f"{caps.get('version')}, build {caps.get('build')})."
            ),
            detail=caps,
        )]
    return [DoctorFinding(
        code="D5",
        severity="warning",
        message=(
            f"Stock openseespy backend (version {caps.get('version')}) — "
            "fork-only features (Bezier elements, LadrunoUP, profiler) "
            "unavailable. Point APEGMSH_OPENSEES_BIN at the Ladruno "
            "fork's dist\\bin to use the fork."
        ),
        detail=caps,
    )]


def _counterpart_interpreters() -> list[tuple[str, list[str]]]:
    """Sibling interpreters whose shared packages must agree with ours.

    The configured reference venv, plus the Windows launcher's default
    interpreter when it resolves to something else. Empty when no
    reference venv is configured — with nothing to compare against, D6
    has no work to do.
    """
    counterparts: list[tuple[str, list[str]]] = []
    reference = _reference_venv()
    if reference is not None:
        ref_py = _venv_python(reference)
        if ref_py.exists() and not _same_path(ref_py, Path(sys.executable)):
            counterparts.append(("reference venv", [str(ref_py)]))
    if sys.platform == "win32" and shutil.which("py"):
        counterparts.append(("launcher default (py -3)", ["py", "-3"]))
    return counterparts


def _check_shared_packages() -> list[DoctorFinding]:
    """D6 — shared-package version agreement across sibling interpreters.

    Opt-in via ``APEGMSH_SHARED_PACKAGES``. The failure this guards is
    specific: a distribution installed *non-editably* into two or more
    interpreters can drift apart silently, so the same code computes
    different numbers depending on which python ran it. Nothing apeGmsh
    depends on is installed that way, hence no default list — a site with
    such a package names it, everyone else gets a one-line skip.
    """
    import importlib.metadata

    packages = _shared_packages()
    if not packages:
        return [DoctorFinding(
            code="D6",
            severity="info",
            message=(
                "No shared packages configured — cross-interpreter version "
                f"agreement not checked. Set {_SHARED_PACKAGES_ENV} (and "
                f"{_REFERENCE_VENV_ENV}) if a non-editable package must "
                "agree across several interpreters here."
            ),
            detail={"configured": ()},
        )]

    # Report the child's own executable alongside the versions: launchers
    # like `py -3` can resolve back to the interpreter we are already
    # running, and "1 of 1 agree" is falsely reassuring when the one
    # counterpart was ourselves.
    src = (
        "import json, sys, importlib.metadata as m\n"
        f"names = {list(packages)!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try:\n"
        "        out[n] = m.version(n)\n"
        "    except Exception:\n"
        "        out[n] = None\n"
        "print(json.dumps({'exe': sys.executable, 'versions': out}))\n"
    )
    # A counterpart may boot a fork with a splash banner — quiet it so the
    # JSON is the last thing on stdout.
    env = dict(os.environ)
    env["LADRUNO_OPENSEES_QUIET"] = "1"
    env["APEGMSH_QUIET"] = "1"

    mine: dict[str, str | None] = {}
    for name in packages:
        try:
            mine[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mine[name] = None

    statuses: dict[str, object] = {}
    mismatches: list[str] = []
    for label, argv in _counterpart_interpreters():
        probe = _run_probe(argv + ["-c", src], env=env)
        if not probe.ok:
            statuses[label] = "probe failed"
            continue
        try:
            payload = json.loads(probe.last_stdout_line)
        except ValueError:
            statuses[label] = "probe failed"
            continue
        exe = str(payload.get("exe", ""))
        if exe and _same_path(Path(exe), Path(sys.executable)):
            continue  # resolved back to us — not a counterpart
        theirs = dict(payload.get("versions") or {})
        statuses[label] = theirs
        for name in packages:
            ours, other = mine.get(name), theirs.get(name)
            if ours and other and ours != other:
                mismatches.append(f"{name}: here {ours}, {label} has {other}")

    detail: dict[str, object] = {"current": mine, "counterparts": statuses}
    if mismatches:
        return [DoctorFinding(
            code="D6",
            severity="error",
            message=(
                "Shared-package version disagreement — "
                + "; ".join(mismatches)
                + ". A non-editable install in each interpreter drifts "
                "silently; reinstall into EACH from the package checkout: "
                "`<python> -m pip install --no-deps --force-reinstall .`"
            ),
            detail=detail,
        )]
    installed = [n for n, v in mine.items() if v]
    if not installed:
        message = (
            f"None of the configured shared packages ({', '.join(packages)}) "
            "are installed in this interpreter — agreement check skipped."
        )
    elif statuses:
        message = (
            f"{', '.join(f'{n} {mine[n]}' for n in installed)}; "
            f"{len(statuses)} counterpart interpreter(s) checked, no "
            "disagreement."
        )
    else:
        message = (
            f"{', '.join(f'{n} {mine[n]}' for n in installed)}; "
            "no counterpart interpreter found to compare against."
        )
    return [DoctorFinding(
        code="D6", severity="info", message=message, detail=detail,
    )]


# --------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------- #
_CHECKS: tuple[Callable[[], list[DoctorFinding]], ...] = (
    _check_interpreter,
    _check_import_path,
    _check_gmsh,
    _check_viewer_gl,
    _check_opensees,
    _check_shared_packages,
)


def run_doctor() -> DoctorReport:
    """Run every check; a crashing check becomes a D0 error finding."""
    findings: list[DoctorFinding] = []
    for check in _CHECKS:
        try:
            findings.extend(check())
        except Exception as exc:
            findings.append(DoctorFinding(
                code="D0",
                severity="error",
                message=(
                    f"doctor check {check.__name__} crashed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ))
    _pkg = _import_apegmsh()
    return DoctorReport(
        python=".".join(str(v) for v in sys.version_info[:3]),
        executable=sys.executable,
        apegmsh_version=(
            getattr(_pkg, "__version__", "unknown") if _pkg else "not importable"
        ),
        package_dir=(
            str(Path(_pkg.__file__).resolve().parent) if _pkg else "n/a"
        ),
        findings=tuple(findings),
    )


def main() -> int:
    """CLI body for ``python -m apeGmsh doctor``."""
    report = run_doctor()
    print(report.to_markdown())
    return report.exit_code


if __name__ == "__main__":
    # Standalone lane: `<any-python> src/apeGmsh/doctor.py`, for
    # interrogating an interpreter that cannot import apeGmsh.
    raise SystemExit(main())
