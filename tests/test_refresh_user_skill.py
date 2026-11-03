"""scripts/refresh_user_skill.py refreshes every repo skill (#2811).

Uses a throwaway git repo + bare "origin" and a temporary HOME; never
touches the real ~/.claude.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "refresh_user_skill.py"


def _run(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    work = tmp_path / "work"
    origin = tmp_path / "origin.git"
    home = tmp_path / "home"
    work.mkdir()
    home.mkdir()
    _run(tmp_path, "init", "--bare", "-b", "main", str(origin))
    _run(work, "init", "-b", "main")
    _run(work, "config", "user.email", "t@t")
    _run(work, "config", "user.name", "t")
    for rel, text in {
        "skills/apegmsh/SKILL.md": "canonical\n",
        ".claude/skills/apegmsh-program/SKILL.md": "program\n",
        ".claude/skills/apegmsh-helper/SKILL.md": "mirror\n",
    }.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _run(work, "add", "-A")
    _run(work, "commit", "-m", "init")
    _run(work, "remote", "add", "origin", str(origin))
    _run(work, "push", "origin", "main")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    spec = importlib.util.spec_from_file_location("refresh_user_skill", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO = work
    return mod, home / ".claude" / "skills"


def test_refreshes_stale_dev_skill_and_spares_unrelated(env):
    mod, root = env
    (root / "apegmsh-program").mkdir(parents=True)
    (root / "apegmsh-program" / "SKILL.md").write_text("OLD\n")
    (root / "apegmsh-program" / "extra.md").write_text("stale file\n")
    (root / "someone-elses").mkdir()
    (root / "someone-elses" / "SKILL.md").write_text("keep me\n")

    assert mod.main(["--check"]) == 1
    assert mod.main([]) == 0

    assert (root / "apegmsh" / "SKILL.md").read_text() == "canonical\n"
    assert (root / "apegmsh-helper" / "SKILL.md").read_text() == "mirror\n"
    assert (root / "apegmsh-program" / "SKILL.md").read_text() == "program\n"
    assert not (root / "apegmsh-program" / "extra.md").exists()
    assert (root / "someone-elses" / "SKILL.md").read_text() == "keep me\n"
    assert mod.main(["--check"]) == 0


def test_check_ignores_line_endings(env):
    mod, root = env
    assert mod.main([]) == 0
    p = root / "apegmsh-program" / "SKILL.md"
    p.write_bytes(b"program\r\n")
    assert mod.main(["--check"]) == 0
    p.write_bytes(b"changed\n")
    assert mod.main(["--check"]) == 1
