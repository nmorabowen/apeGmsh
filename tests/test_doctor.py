"""Tests for ``python -m apeGmsh doctor`` (:mod:`apeGmsh.doctor`)."""
import ast
import importlib.metadata
import json
import re
import os
import subprocess
import sys
from pathlib import Path

import pytest

from apeGmsh import doctor as doctor_mod
from apeGmsh.doctor import (
    DoctorFinding,
    DoctorReport,
    _check_gmsh,
    _check_import_path,
    _check_viewer_gl,
    run_doctor,
)


# A path that looks like it lives in somebody's home directory. The
# separator class carries BOTH slashes on purpose — a Windows report
# quotes ``C:\Users\...`` and a POSIX one ``/home/...``.
HOME_PATH_RE = r"[A-Za-z]:[\\/]+Users[\\/]+|/home/"


def _report(*findings: DoctorFinding) -> DoctorReport:
    return DoctorReport(
        python="3.11.0",
        executable="python",
        apegmsh_version="0.0.0",
        package_dir=".",
        findings=tuple(findings),
    )


# The doctor must run on GL-less CI: no Qt/VTK (or anything heavy) may
# be imported at module level. Guard it structurally.
_ALLOWED_TOP_LEVEL = {
    "__future__", "dataclasses", "json", "os", "pathlib", "shutil",
    "subprocess", "sys", "typing",
}


def test_module_level_imports_are_stdlib_only():
    tree = ast.parse(Path(doctor_mod.__file__).read_text(encoding="utf-8"))
    tops: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            tops.add((node.module or "").split(".")[0])
    assert tops <= _ALLOWED_TOP_LEVEL, tops - _ALLOWED_TOP_LEVEL


def test_check_gmsh_ok():
    findings = _check_gmsh()
    assert len(findings) == 1
    f = findings[0]
    assert f.code == "D3"
    assert f.severity == "info"  # gmsh is a hard dep of the test env


def test_viewer_check_emits_exactly_one_finding():
    """One D4 line, so the report can't say "ok" and "won't open" at once."""
    assert len(_check_viewer_gl()) == 1


def test_offscreen_platform_warns_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    findings = _check_viewer_gl()
    assert len(findings) == 1
    f = findings[0]
    if f.detail["missing"] or not f.detail["qt_bindings"]:
        pytest.skip("viewer extra not installed; offscreen branch unreachable")
    assert f.code == "D4" and f.severity == "warning"
    assert "offscreen" in f.message
    assert "ViewerWindow" in f.message


def _fake_editable_dist(root: Path):
    """A stand-in for ``importlib.metadata.distribution('apeGmsh')``."""
    payload = json.dumps({
        "url": root.as_uri(),
        "dir_info": {"editable": True},
    })

    class _Dist:
        @staticmethod
        def read_text(name):
            return payload if name == "direct_url.json" else None

    return _Dist()


def _run_import_path_check(monkeypatch, root: Path, imported_from: Path):
    import importlib.metadata

    import apeGmsh

    imported_from.mkdir(parents=True, exist_ok=True)
    (root / "src" / "apeGmsh").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(apeGmsh, "__file__", str(imported_from / "__init__.py"))
    monkeypatch.setattr(
        importlib.metadata, "distribution",
        lambda name: _fake_editable_dist(root),
    )
    findings = _check_import_path()
    assert len(findings) == 1 and findings[0].code == "D2"
    return findings[0]


def test_import_path_editable_match_is_ok(monkeypatch, tmp_path):
    root = tmp_path / "apeGmsh"
    f = _run_import_path_check(monkeypatch, root, root / "src" / "apeGmsh")
    assert f.severity == "info"


def test_import_path_nested_dir_reports_drift(monkeypatch, tmp_path):
    """A nested tree that is NOT a git worktree is drift.

    Its path is textually *inside* the install root, so a containment
    test would call it healthy — the check must compare exactly.
    """
    root = tmp_path / "apeGmsh"
    nested = root / ".claude" / "worktrees" / "wt" / "src" / "apeGmsh"
    f = _run_import_path_check(monkeypatch, root, nested)
    assert f.severity == "warning"
    assert "drift" in f.message.lower()
    assert str(nested) in f.message


def _make_linked_worktree(tree_root: Path, gitdir: Path) -> None:
    """Give ``tree_root`` the ``.git`` FILE that marks a linked worktree."""
    tree_root.mkdir(parents=True, exist_ok=True)
    (tree_root / ".git").write_text(f"gitdir: {gitdir.as_posix()}\n",
                                    encoding="utf-8")


def test_worktree_of_the_same_repo_is_info(monkeypatch, tmp_path):
    """Working in a worktree is normal — it must not warn every run."""
    root = tmp_path / "apeGmsh"
    (root / ".git").mkdir(parents=True)  # install root: ordinary checkout
    tree = root / ".claude" / "worktrees" / "wt"
    _make_linked_worktree(tree, root / ".git" / "worktrees" / "wt")

    f = _run_import_path_check(monkeypatch, root, tree / "src" / "apeGmsh")
    assert f.severity == "info"
    assert "worktree of that same repo" in f.message


def test_worktree_of_a_different_repo_still_warns(monkeypatch, tmp_path):
    """Same structure, foreign gitdir — that is shadowing, not a branch."""
    root = tmp_path / "apeGmsh"
    (root / ".git").mkdir(parents=True)
    other = tmp_path / "somewhere-else"
    tree = root / ".claude" / "worktrees" / "wt"
    _make_linked_worktree(tree, other / ".git" / "worktrees" / "wt")

    f = _run_import_path_check(monkeypatch, root, tree / "src" / "apeGmsh")
    assert f.severity == "warning"
    assert "drift" in f.message.lower()


def test_ordinary_checkout_is_not_mistaken_for_a_worktree(monkeypatch, tmp_path):
    """A `.git` DIRECTORY in the drifting tree is not a worktree pointer."""
    root = tmp_path / "apeGmsh"
    (root / ".git").mkdir(parents=True)
    tree = root / "vendored" / "clone"
    (tree / ".git").mkdir(parents=True)

    f = _run_import_path_check(monkeypatch, root, tree / "src" / "apeGmsh")
    assert f.severity == "warning"


def _fake_counterpart(payload: str) -> list[str]:
    """argv for a probe that reports ``payload`` instead of its real state.

    ``_check_shared_packages`` appends ``-c <src>`` to the argv it is
    handed; python binds the FIRST ``-c`` as the program and pushes the
    rest into ``sys.argv``, so this command wins and the real probe body
    is inert.
    """
    return [sys.executable, "-c", f"print({payload!r})"]


#: D6 is opt-in, and its old hardcoded subject (``baseUnits``) is not
#: installed outside one office machine — so these lanes used to skip
#: everywhere including CI. Point the check at apeGmsh itself, which is
#: necessarily installed wherever the suite runs.
_SHARED = "apeGmsh"


def _configure_shared(monkeypatch, counterpart_payload: str | None):
    monkeypatch.setenv(doctor_mod._SHARED_PACKAGES_ENV, _SHARED)
    if counterpart_payload is None:
        monkeypatch.setattr(doctor_mod, "_counterpart_interpreters", list)
        return
    monkeypatch.setattr(
        doctor_mod, "_counterpart_interpreters",
        lambda: [("fake counterpart", _fake_counterpart(counterpart_payload))],
    )


def _my_version() -> str:
    return importlib.metadata.version(_SHARED)


def test_d6_is_skipped_when_no_shared_packages_configured(monkeypatch):
    """Default install: opt-in check, one neutral line, nothing personal."""
    monkeypatch.delenv(doctor_mod._SHARED_PACKAGES_ENV, raising=False)
    findings = doctor_mod._check_shared_packages()
    assert len(findings) == 1
    f = findings[0]
    assert f.code == "D6"
    assert f.severity == "info"
    assert "No shared packages configured" in f.message
    assert doctor_mod._SHARED_PACKAGES_ENV in f.message


@pytest.mark.slow
def test_shared_package_disagreement_is_an_error(monkeypatch):
    bogus = json.dumps({
        "exe": "C:/fake/other/python.exe",
        "versions": {_SHARED: _my_version() + ".999"},
    })
    _configure_shared(monkeypatch, bogus)
    findings = doctor_mod._check_shared_packages()
    assert len(findings) == 1
    f = findings[0]
    assert f.code == "D6"
    assert f.severity == "error"
    assert "disagreement" in f.message
    assert "force-reinstall" in f.message  # the actionable remedy
    assert _SHARED in f.message


@pytest.mark.slow
def test_shared_package_agreement_is_ok(monkeypatch):
    agreeing = json.dumps({
        "exe": "C:/fake/other/python.exe",
        "versions": {_SHARED: _my_version()},
    })
    _configure_shared(monkeypatch, agreeing)
    f = doctor_mod._check_shared_packages()[0]
    assert f.severity == "info"
    assert f.detail["counterparts"] == {
        "fake counterpart": {_SHARED: _my_version()}
    }


@pytest.mark.slow
def test_shared_package_ignores_self_resolving_counterpart(monkeypatch):
    """A launcher resolving back to us is not an independent counterpart."""
    itself = json.dumps({
        "exe": sys.executable,
        "versions": {_SHARED: _my_version()},
    })
    _configure_shared(monkeypatch, itself)
    f = doctor_mod._check_shared_packages()[0]
    assert f.severity == "info"
    assert f.detail["counterparts"] == {}
    assert "no counterpart interpreter found" in f.message


def test_exit_code_zero_with_warnings_only():
    report = _report(
        DoctorFinding(code="D1", severity="warning", message="w"),
        DoctorFinding(code="D3", severity="info", message="i"),
    )
    assert report.ok
    assert report.exit_code == 0


def test_exit_code_one_on_error():
    report = _report(DoctorFinding(code="D6", severity="error", message="e"))
    assert not report.ok
    assert report.exit_code == 1


def test_markdown_render():
    report = _report(
        DoctorFinding(code="D6", severity="error", message="units disagree"),
        DoctorFinding(code="D4", severity="warning", message="no viewer"),
        DoctorFinding(code="D3", severity="info", message="gmsh fine"),
    )
    md = report.to_markdown()
    assert "# apeGmsh doctor" in md
    assert "[ERROR D6]" in md
    assert "[warn  D4]" in md
    assert "[ok    D3]" in md
    assert "Exit code 1" in md


@pytest.mark.slow
def test_run_doctor_emits_every_check():
    report = run_doctor()
    codes = {f.code for f in report.findings}
    assert {"D1", "D2", "D3", "D4", "D5", "D6"} <= codes, codes
    # A doctor check crashing is itself a bug in the doctor.
    assert "D0" not in codes, [
        f.message for f in report.findings if f.code == "D0"
    ]


def _cli(*args: str) -> subprocess.CompletedProcess:
    """Run ``python -m apeGmsh <args>`` against the in-tree source.

    PYTHONPATH is pinned to this tree's ``src`` so the subprocess runs
    the source under test rather than whatever the editable install
    points at (they differ in a git worktree). ``LADRUNO_OPENSEES_QUIET``
    silences the fork banner that the office venv's ``.pth`` boot hook
    prints to stdout at interpreter startup; decoding stays lenient in
    case one slips through under a non-UTF-8 console codepage.
    """
    env = dict(os.environ)
    env["APEGMSH_QUIET"] = "1"
    env["LADRUNO_OPENSEES_QUIET"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    src_dir = Path(doctor_mod.__file__).resolve().parents[1]
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + prior if prior else "")
    return subprocess.run(
        [sys.executable, "-m", "apeGmsh", *args],
        capture_output=True, timeout=600, env=env,
        encoding="utf-8", errors="replace", check=False,
    )


@pytest.mark.slow
def test_cli_smoke():
    proc = _cli("doctor")
    # Exit code reflects the machine's actual health; both are valid runs.
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    assert "# apeGmsh doctor" in proc.stdout
    assert "## Result" in proc.stdout


@pytest.mark.slow
def test_cli_rejects_unknown_command():
    assert _cli("nope").returncode == 2  # argparse usage error



# --------------------------------------------------------------------------
# Leak regression — a published wheel must not carry anyone's home directory.
# --------------------------------------------------------------------------
def test_doctor_source_hardcodes_no_user_path():
    r"""``doctor.py`` shipped one developer's venv path as a literal.

    Every install in the world printed that ``C:\Users\...`` from D1, and
    D6 probed it. Site conventions belong in the environment
    (``APEGMSH_REFERENCE_VENV`` / ``APEGMSH_SHARED_PACKAGES``), not in a
    wheel on PyPI.
    """
    src = Path(doctor_mod.__file__).read_text(encoding="utf-8")
    offenders = re.findall(
        r"""["'][A-Za-z]:[\\/]+Users[\\/]+[^"'\n]+["']|["']/home/[^"'\n]+["']""",
        src,
    )
    assert not offenders, (
        f"doctor.py hardcodes user path(s): {offenders}. Drive it from an "
        "environment variable instead."
    )


def test_default_report_names_no_user_home(monkeypatch):
    """With nothing configured, no finding may quote a SHIPPED home directory.

    Paths the doctor discovers at RUNTIME are exempt: D2 echoes wherever
    apeGmsh actually resides, and D1 echoes ``sys.executable`` /
    ``sys.prefix``. Reporting those is the report's whole job, and on any
    machine whose venv lives under the user's home they match the pattern
    below for reasons that have nothing to do with apeGmsh. Asserting on
    them turned this into a check on the CI runner's directory layout --
    it passed only because a hosted runner keeps its Python outside
    /home, and failed on every developer venv under ~.

    So: erase the runtime-discovered paths from the message first, then
    assert nothing home-shaped survives. That still catches the thing
    worth catching -- a user path baked into a message TEMPLATE. Its
    source-level sibling is ``test_doctor_source_hardcodes_no_user_path``.
    """
    monkeypatch.delenv(doctor_mod._REFERENCE_VENV_ENV, raising=False)
    monkeypatch.delenv(doctor_mod._SHARED_PACKAGES_ENV, raising=False)
    # Longest first: sys.prefix is a prefix of sys.executable, and
    # replacing the short one first would leave the tail behind.
    runtime_paths = sorted(
        {str(Path(sys.executable)), str(Path(sys.prefix)),
         str(Path(doctor_mod.__file__).resolve().parents[1])},
        key=len, reverse=True,
    )
    for f in (*doctor_mod._check_interpreter(),
              *doctor_mod._check_shared_packages()):
        redacted = f.message
        for known in runtime_paths:
            redacted = redacted.replace(known, "<runtime>")
        assert not re.search(HOME_PATH_RE, redacted), (
            f"{f.code} quotes a home directory with nothing configured: "
            f"{f.message}"
        )
