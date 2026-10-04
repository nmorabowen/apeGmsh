"""The reference scripts banked in the skill's ``model-scripts.md`` still run.

Each script asserts its own hand check, so exit code 0 means the example is
both runnable and right. A script runs in a subprocess (its own gmsh and
openseespy process) from a temp copy, with ``src`` on ``PYTHONPATH`` so the
worktree's apeGmsh is the one imported.
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
DOC = ROOT / "skills" / "apegmsh" / "references" / "model-scripts.md"
FENCE = re.compile(
    r"<!-- reference-script: (?P<name>[\w.]+) -->\n```python\n(?P<body>.*?)```",
    re.DOTALL,
)
SCRIPTS = {m["name"]: m["body"] for m in FENCE.finditer(DOC.read_text(encoding="utf-8"))}


def test_doc_banks_the_three_families() -> None:
    assert set(SCRIPTS) == {"pratt_truss.py", "frame_modal.py", "staged_footing.py"}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_reference_script_runs_and_passes_its_checks(name: str, tmp_path: Path) -> None:
    pytest.importorskip("openseespy")
    script = tmp_path / name
    script.write_text(SCRIPTS[name], encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    run = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", str(script)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900,
    )
    assert run.returncode == 0, f"{name} failed:\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}"
