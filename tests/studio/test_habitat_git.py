"""ADR 0095 Amendment 8 S9a — `studio init` git-inits the habitat.

Headless, same subprocess pattern as ``test_habitat_template.py``: shell
out to ``python -m apeGmsh.studio init`` with ``PYTHONPATH`` pinned to
this worktree's ``src``. Commit identity comes from ``GIT_*`` env vars
so the tests pass on CI runners with no global git config (the init verb
itself must never fabricate an identity).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
STUDIO_DIR = SRC / "apeGmsh" / "studio"

needs_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not on PATH"
)


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["APEGMSH_QUIET"] = "1"
    env["LADRUNO_OPENSEES_QUIET"] = "1"
    env["GIT_AUTHOR_NAME"] = "studio-test"
    env["GIT_AUTHOR_EMAIL"] = "studio-test@example.invalid"
    env["GIT_COMMITTER_NAME"] = "studio-test"
    env["GIT_COMMITTER_EMAIL"] = "studio-test@example.invalid"
    return env


def _init(target: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "apeGmsh.studio",
            "init",
            "--name",
            "demo-habitat",
            "--model",
            "demo",
            "--root",
            str(target),
            *extra,
        ],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )


def _git(target: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(target), *args], capture_output=True, text=True
    )


# ---------------------------------------------------------------------------
# Default: repo on 'main', exactly one commit, clean porcelain
# ---------------------------------------------------------------------------


@needs_git
def test_init_creates_repo_one_clean_commit(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (target / ".git").is_dir()

    count = _git(target, "rev-list", "--count", "HEAD")
    assert count.returncode == 0, count.stderr
    assert count.stdout.strip() == "1"

    branch = _git(target, "branch", "--show-current")
    assert branch.stdout.strip() == "main"

    msg = _git(target, "log", "-1", "--format=%s")
    assert msg.stdout.strip() == "studio init: demo-habitat"

    # Clean porcelain proves the shipped .gitignore covers everything
    # init writes — nothing tracked is left dangling, nothing ignored
    # leaks into status.
    porcelain = _git(target, "status", "--porcelain")
    assert porcelain.stdout.strip() == "", porcelain.stdout


@needs_git
def test_gitignore_bulk_hidden_run_json_visible(tmp_path: Path) -> None:
    """The contract the checkpoint workflow depends on (INV-26): after a
    simulated run, bulk artifacts are invisible to git while the case's
    ``run.json`` — the git-visible run record — surfaces."""
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    case = target / "models" / "demo" / "cases" / "c1"
    (case / "results").mkdir(parents=True)
    (case / "results" / "big.h5").write_bytes(b"\x00" * 16)
    (case / "run.json").write_text("{}\n", encoding="utf-8")
    (target / "models" / "demo" / "model.h5").write_bytes(b"\x00" * 16)
    (target / ".apegmsh").mkdir()
    (target / ".apegmsh" / "names.json").write_text("{}\n", encoding="utf-8")

    porcelain = _git(target, "status", "--porcelain", "-uall")
    lines = sorted(porcelain.stdout.strip().splitlines())
    assert lines == ["?? models/demo/cases/c1/run.json"], porcelain.stdout


# ---------------------------------------------------------------------------
# Degrade paths: --no-git, nested work tree, git absent
# ---------------------------------------------------------------------------


@needs_git
def test_no_git_flag_skips_repo(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target, "--no-git")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (target / ".git").exists()
    assert "--no-git" in proc.stdout


@needs_git
def test_init_inside_existing_work_tree_skips(tmp_path: Path) -> None:
    outer = _git(tmp_path, "init")
    assert outer.returncode == 0, outer.stderr

    target = tmp_path / "inner"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (target / ".git").exists()
    assert "skipping git init" in proc.stdout
    assert (target / "ape.project.yaml").is_file()


def test_git_missing_warns_and_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    sys.path.insert(0, str(SRC))
    try:
        from apeGmsh.studio import _init_habitat as ih
    finally:
        sys.path.remove(str(SRC))

    monkeypatch.setattr(ih.shutil, "which", lambda cmd: None)
    target = tmp_path / "habitat"
    rc = ih.run_init("demo-habitat", "demo", str(target))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "WARN: git not found" in out
    assert not (target / ".git").exists()


# ---------------------------------------------------------------------------
# INV-24 grep gate: only _init_habitat.py invokes git write commands
# ---------------------------------------------------------------------------

_GIT_WRITE_LINE = re.compile(r"\bgit\b.*\b(add|commit|tag|push)\b")


def test_inv24_only_init_habitat_writes_git() -> None:
    offenders = []
    for path in sorted(STUDIO_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".ps1"}:
            continue
        if path.name == "_init_habitat.py":
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if _GIT_WRITE_LINE.search(line):
                offenders.append(
                    f"{path.relative_to(STUDIO_DIR).as_posix()}:{lineno}: "
                    f"{line.strip()}"
                )
    assert offenders == [], (
        "INV-24: git writes belong to _init_habitat.py alone; "
        f"found: {offenders}"
    )
