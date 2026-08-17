"""ADR 0095 Amendment 10 S11a — `scripts/run_case.py`, the case runner.

Headless, no solver: a toy driver stands in for the model script (writes
one artifact into cwd = ``results/`` and prints), a toy verify checks it
and prints PASS/FAIL lines. Same subprocess pattern as the sibling
habitat suites.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

needs_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not on PATH"
)

TOY_DRIVER = """\
from pathlib import Path
Path("out.bin").write_bytes(b"\\x00" * 8)
print("toy driver: wrote out.bin")
"""

TOY_DRIVER_FAILING = """\
import sys
print("toy driver: about to fail")
sys.exit(3)
"""

TOY_VERIFY = """\
import sys
from pathlib import Path
ok = Path("out.bin").is_file()
print("PASS  artifact_exists: out.bin" if ok else "FAIL  artifact_exists: out.bin")
sys.exit(0 if ok else 1)
"""


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


def _seed_toys(target: Path, driver: str = TOY_DRIVER, commit: bool = True) -> None:
    """Drop the toy driver + verify into src and (by default) commit them —
    commit-then-run, so the runner's `git_dirty` honesty check stays False."""
    src = target / "models" / "demo" / "src"
    (src / "driver.py").write_text(driver, encoding="utf-8")
    (src / "verify.py").write_text(TOY_VERIFY, encoding="utf-8")
    if commit:
        subprocess.run(
            ["git", "-C", str(target), "add", "-A"], capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(target), "commit", "-m", "seed toy case scripts"],
            capture_output=True,
            env=_subprocess_env(),
        )


def _run_case(target: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(target / "scripts" / "run_case.py"),
            "--model",
            "demo",
            "--case",
            "c1",
            "--script",
            "models/demo/src/driver.py",
            *extra,
        ],
        cwd=str(target),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )


@needs_git
def test_runner_happy_path(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _seed_toys(target)

    run = _run_case(target, "--verify", "models/demo/src/verify.py")
    assert run.returncode == 0, run.stdout + run.stderr

    case = target / "models" / "demo" / "cases" / "c1"
    assert (case / "results" / "out.bin").is_file()
    assert "toy driver: wrote out.bin" in (case / "logs" / "run.log").read_text(
        encoding="utf-8"
    )
    assert (case / "logs" / "verify.log").is_file()
    # deck disclosure auto-written (Amendment 9 F3)
    assert "No deck" in (case / "deck" / "README.md").read_text(encoding="utf-8")

    manifest = json.loads((case / "run.json").read_text(encoding="utf-8"))
    assert manifest["case"] == "c1"
    assert manifest["model"] == "demo"
    assert manifest["script"] == "models/demo/src/driver.py"
    assert manifest["run_exit"] == 0
    assert manifest["results"] == ["results/out.bin"]
    assert "logs/run.log" in manifest["logs"]
    assert manifest["verify"]["exit"] == 0
    assert manifest["verify"]["passed"] == 1
    assert manifest["verify"]["failed"] == 0
    assert len(manifest["model_sha"]) == 40
    assert manifest["git_dirty"] is False


@needs_git
def test_runner_refuses_rerun(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    assert _init(target).returncode == 0
    _seed_toys(target)
    assert _run_case(target).returncode == 0

    before = (target / "models" / "demo" / "cases" / "c1" / "run.json").read_bytes()
    again = _run_case(target)
    assert again.returncode != 0
    assert "immutable" in again.stderr
    after = (target / "models" / "demo" / "cases" / "c1" / "run.json").read_bytes()
    assert before == after


@needs_git
def test_runner_records_failed_run(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    assert _init(target).returncode == 0
    _seed_toys(target, driver=TOY_DRIVER_FAILING)

    run = _run_case(target, "--verify", "models/demo/src/verify.py")
    assert run.returncode == 3, run.stdout + run.stderr

    case = target / "models" / "demo" / "cases" / "c1"
    manifest = json.loads((case / "run.json").read_text(encoding="utf-8"))
    assert manifest["run_exit"] == 3
    # verify is skipped after a failed run — no verify block claimed
    assert "verify" not in manifest
    assert "about to fail" in (case / "logs" / "run.log").read_text(encoding="utf-8")


@needs_git
def test_runner_without_repo_omits_provenance(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    assert _init(target, "--no-git").returncode == 0
    _seed_toys(target, commit=False)

    run = _run_case(target)
    assert run.returncode == 0, run.stdout + run.stderr
    manifest = json.loads(
        (target / "models" / "demo" / "cases" / "c1" / "run.json").read_text(
            encoding="utf-8"
        )
    )
    assert "model_sha" not in manifest
    assert "git_dirty" not in manifest
