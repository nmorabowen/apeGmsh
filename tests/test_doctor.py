"""Tests for ``python -m apeGmsh doctor`` (:mod:`apeGmsh.doctor`)."""
import ast
import importlib.metadata
import json
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

    ``_check_baseunits`` appends ``-c <src>`` to the argv it is handed;
    python binds the FIRST ``-c`` as the program and pushes the rest into
    ``sys.argv``, so this command wins and the real probe body is inert.
    """
    return [sys.executable, "-c", f"print({payload!r})"]


@pytest.mark.slow
def test_baseunits_disagreement_is_an_error(monkeypatch):
    try:
        current = importlib.metadata.version("baseUnits")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("baseUnits not installed in this interpreter")

    bogus = json.dumps({"exe": r"C:\fake\other\python.exe",
                        "version": current + ".999"})
    monkeypatch.setattr(
        doctor_mod, "_baseunits_counterparts",
        lambda: [("fake counterpart", _fake_counterpart(bogus))],
    )
    findings = doctor_mod._check_baseunits()
    assert len(findings) == 1
    f = findings[0]
    assert f.code == "D6"
    assert f.severity == "error"
    assert "disagreement" in f.message
    assert "force-reinstall" in f.message  # the actionable remedy


@pytest.mark.slow
def test_baseunits_agreement_is_ok(monkeypatch):
    try:
        current = importlib.metadata.version("baseUnits")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("baseUnits not installed in this interpreter")

    agreeing = json.dumps({"exe": r"C:\fake\other\python.exe",
                           "version": current})
    monkeypatch.setattr(
        doctor_mod, "_baseunits_counterparts",
        lambda: [("fake counterpart", _fake_counterpart(agreeing))],
    )
    f = doctor_mod._check_baseunits()[0]
    assert f.severity == "info"
    assert f.detail["counterparts"] == {"fake counterpart": current}


@pytest.mark.slow
def test_baseunits_ignores_self_resolving_counterpart(monkeypatch):
    """A launcher resolving back to us is not an independent counterpart."""
    try:
        current = importlib.metadata.version("baseUnits")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("baseUnits not installed in this interpreter")

    itself = json.dumps({"exe": sys.executable, "version": current})
    monkeypatch.setattr(
        doctor_mod, "_baseunits_counterparts",
        lambda: [("myself", _fake_counterpart(itself))],
    )
    f = doctor_mod._check_baseunits()[0]
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

