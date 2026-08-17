"""ADR 0095 Amendment 8 S9b — the checkpoint contract ships in the template.

Headless, same subprocess pattern as ``test_habitat_git.py``. The
``git_provenance`` tests load the *initialized habitat's own*
``scripts/_habitat.py`` (self-contained per Amendment 6) under a unique
module name so two habitats in one test session cannot collide in
``sys.modules``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
TEMPLATE_DIR = SRC / "apeGmsh" / "studio" / "template"

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


def _load_habitat_module(target: Path, alias: str):
    spec = importlib.util.spec_from_file_location(
        alias, target / "scripts" / "_habitat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# checkpoints.md teaches the contract; the index and pointers exist
# ---------------------------------------------------------------------------


def test_checkpoints_md_teaches_the_contract() -> None:
    text = (TEMPLATE_DIR / "APE" / "instructions" / "checkpoints.md").read_text(
        encoding="utf-8"
    )
    # main-as-checkpoints, work branches, merge-with-review, tags, resume
    assert "`main` holds checkpoints only" in text
    assert "work/<model>/<slug>" in text
    assert "--no-ff" in text
    assert "checkpoint/<model>/<slug>" in text
    assert "git tag -l 'checkpoint/*'" in text
    # the axis rule (Amendment 8 contract point 3)
    assert "question axis" in text
    assert "Case drivers" in text
    assert "Two model ids" in text
    # local <-> PR mapping and the no-forge stance
    assert "PR is the review surface" in text
    assert "merging the PR *is* the checkpoint" in text
    # Amendment 9 F2: the model-surface / process-file split
    assert "Process files commit to `main`" in text
    assert "model surface" in text


def test_checkpoints_md_indexed_and_pointed_to() -> None:
    index = (TEMPLATE_DIR / "APE" / "instructions" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "checkpoints.md" in index
    how = (TEMPLATE_DIR / "APE" / "instructions" / "how-we-work.md").read_text(
        encoding="utf-8"
    )
    assert "instructions/checkpoints.md" in how


def test_provenance_contract_documented() -> None:
    models = (TEMPLATE_DIR / "models" / "README.md").read_text(encoding="utf-8")
    assert "git_provenance" in models
    assert "model_sha" in models
    assert "git_dirty" in models
    # Amendment 9 F3: deck/ admits a disclosure README for in-process runs
    assert "naming why none exists" in models
    reporting = (TEMPLATE_DIR / "APE" / "instructions" / "reporting.md").read_text(
        encoding="utf-8"
    )
    assert "model_sha" in reporting
    assert "Disclosure, not enforcement" in reporting


@needs_git
def test_strict_check_requires_checkpoints_md(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    (target / "APE" / "instructions" / "checkpoints.md").unlink()
    check = subprocess.run(
        [sys.executable, str(target / "scripts" / "check_template.py"), "--strict"],
        cwd=str(target),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    assert check.returncode != 0
    assert "checkpoints.md" in check.stdout


# ---------------------------------------------------------------------------
# git_provenance(): the run.json fields
# ---------------------------------------------------------------------------


@needs_git
def test_git_provenance_fields(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    hab = _load_habitat_module(target, "hab_provenance_clean")
    prov = hab.git_provenance()
    assert prov is not None
    assert len(prov["model_sha"]) == 40
    assert all(c in "0123456789abcdef" for c in prov["model_sha"])
    assert prov["git_dirty"] is False

    manifest = target / "ape.project.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n# dirty\n", encoding="utf-8"
    )
    prov2 = hab.git_provenance()
    assert prov2["git_dirty"] is True
    assert prov2["model_sha"] == prov["model_sha"]


@needs_git
def test_git_provenance_none_without_repo(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target, "--no-git")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    hab = _load_habitat_module(target, "hab_provenance_norepo")
    assert hab.git_provenance() is None


# ---------------------------------------------------------------------------
# Lifecycle prints: start shows branch @ sha (dirty: N); finish shows the
# uncommitted count — and neither ever writes to git (INV-24)
# ---------------------------------------------------------------------------


@needs_git
def test_start_session_prints_git_line(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    start = subprocess.run(
        [sys.executable, str(target / "scripts" / "start_session.py"), "--slug", "t"],
        cwd=str(target),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    assert start.returncode == 0, start.stdout + start.stderr
    assert "git: main @ " in start.stdout
    assert "(dirty: 0 files)" in start.stdout


@needs_git
def test_finish_session_prints_count_and_never_commits(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    manifest = target / "ape.project.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n# dirty\n", encoding="utf-8"
    )

    finish = subprocess.run(
        [sys.executable, str(target / "scripts" / "finish_session.py"), "--slug", "t"],
        cwd=str(target),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    assert finish.returncode == 0, finish.stdout + finish.stderr
    assert "git: 1 uncommitted files" in finish.stdout

    # INV-24: the file is still uncommitted afterwards.
    status = subprocess.run(
        ["git", "-C", str(target), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    assert "ape.project.yaml" in status.stdout
