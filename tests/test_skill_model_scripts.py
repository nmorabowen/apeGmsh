"""The skill's executable scripts still run and still lint clean.

Two documents bank scripts behind a ``<!-- reference-script: <name> -->``
marker: ``SKILL.md`` holds the Core workflow skeleton, the first fence an
agent copies, and ``model-scripts.md`` holds the reference scripts. Each
reference script asserts its own hand check, and the skeleton asserts that
its ``Base`` and ``Tip`` groups resolve to nodes, so exit code 0 means the
example is both runnable and right. A script runs in a subprocess (its own
gmsh and openseespy process) from a temp copy, under
``-W error::UserWarning``, with ``src`` on ``PYTHONPATH`` so the worktree's
apeGmsh is the one imported. Each must also pass the skill's own
``scripts/lint_model_script.py``, which the skill tells agents to run on
what they write.

Until SK1-Q (#2036) the skeleton failed as written
(``BridgeError: physical group 'Base' not found``); the control below
replays that.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "apegmsh"
SKILL_MD = SKILL / "SKILL.md"
DOC = SKILL / "references" / "model-scripts.md"
LINT = SKILL / "scripts" / "lint_model_script.py"
FENCE = re.compile(
    r"<!-- reference-script: (?P<name>[\w.]+) -->\n```python\n(?P<body>.*?)```",
    re.DOTALL,
)


def _scripts(doc: Path) -> dict[str, str]:
    return {m["name"]: m["body"] for m in FENCE.finditer(doc.read_text(encoding="utf-8"))}


SKELETON = _scripts(SKILL_MD)
SCRIPTS = {**_scripts(DOC), **SKELETON}


def _run(name: str, body: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-W", "error::UserWarning", str(script)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900,
    )


def test_doc_banks_the_three_families() -> None:
    assert set(_scripts(DOC)) == {"pratt_truss.py", "frame_modal.py", "staged_footing.py"}


def test_skill_md_banks_exactly_one_skeleton() -> None:
    # An empty parametrize only skips, so a lost marker must fail here.
    assert list(SKELETON) == ["skeleton.py"], list(SKELETON)


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_reference_script_runs_and_passes_its_checks(name: str, tmp_path: Path) -> None:
    pytest.importorskip("openseespy")
    run = _run(name, SCRIPTS[name], tmp_path)
    assert run.returncode == 0, f"{name} failed:\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}"


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_reference_script_lints_clean(name: str, tmp_path: Path) -> None:
    script = tmp_path / name
    script.write_text(SCRIPTS[name], encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(LINT), str(script)],
        capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, f"{name} has lint findings:\n{run.stdout}{run.stderr}"


# ── Controls: the runner and the lint each catch the defect they exist for ──

_BASE_LINE = re.compile(r"^.*\.to_physical\(\"Base\"\)\n", re.MULTILINE)
_GROUP_CHECK = re.compile(r"^assert fem\.nodes\.select\(pg=\"Base\"\).*\n", re.MULTILINE)


def test_control_skeleton_without_base_fails(tmp_path: Path) -> None:
    # The skeleton as it shipped before Q: no Base group and no check of it,
    # so only the bridge can catch it, as it did then.
    pytest.importorskip("openseespy")
    body, n = _BASE_LINE.subn("", SKELETON["skeleton.py"])
    body, m = _GROUP_CHECK.subn("", body)
    assert (n, m) == (1, 1), "the skeleton no longer names and checks its Base group in one line each"
    run = _run("skeleton.py", body, tmp_path)
    assert run.returncode != 0, "the skeleton ran without its Base group"
    assert "physical group 'Base' not found" in run.stderr, run.stderr[-3000:]


def test_control_lint_flags_a_machine_path(tmp_path: Path) -> None:
    body = SKELETON["skeleton.py"].replace('save_to="my_model.h5"', 'save_to="C:/runs/my_model.h5"')
    assert body != SKELETON["skeleton.py"], "the skeleton no longer saves to my_model.h5"
    script = tmp_path / "skeleton.py"
    script.write_text(body, encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(LINT), str(script)], capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 1 and " S3 " in run.stdout, run.stdout + run.stderr
