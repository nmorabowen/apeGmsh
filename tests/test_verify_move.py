"""scripts/verify_move.py: a pure move passes; any body change fails.

Cases are built in a tmp_path git repo: the base commit has ``a.py``; the head
state is the working tree after the "move" to ``b.py``.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_move.py"
_spec = importlib.util.spec_from_file_location("verify_move", SCRIPT)
assert _spec is not None and _spec.loader is not None
vm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vm
_spec.loader.exec_module(vm)

A = '''\
import os

X = 1


def f(x):
    """Doc f."""
    return x + 1


class C:
    """Doc C."""

    def m(self):
        return 2
'''


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t",
                    "-c", "user.name=t", *args], check=True,
                   capture_output=True)


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text(A)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _run(repo, capsys, *extra):
    code = vm.main(["--repo", str(repo), "--base", "HEAD", "--head",
                    "WORKTREE", *extra, "a.py", "b.py"])
    return code, capsys.readouterr().out


def _move(repo, mutate=lambda s: s):
    (repo / "a.py").write_text("import b\n\nX = 1\n")
    (repo / "b.py").write_text(mutate(A).replace("import os", "import os.path"))


def test_noop_move(tmp_path, capsys):
    repo = _repo(tmp_path)
    _move(repo)
    code, out = _run(repo, capsys)
    assert code == 0
    assert "verify_move: OK — 3 defs, identical multiset" in out


def test_altered_body(tmp_path, capsys):
    repo = _repo(tmp_path)
    _move(repo, lambda s: s.replace("x + 1", "x + 2"))
    code, out = _run(repo, capsys)
    assert code == 1
    assert "changed: f" in out


def test_deleted_def(tmp_path, capsys):
    repo = _repo(tmp_path)
    _move(repo, lambda s: s.replace("    def m(self):\n        return 2\n", ""))
    code, out = _run(repo, capsys)
    assert code == 1
    assert "removed: C.m" in out


def test_docstring_change(tmp_path, capsys):
    repo = _repo(tmp_path)
    _move(repo, lambda s: s.replace("Doc f.", "Doc g."))
    code, out = _run(repo, capsys)
    assert code == 1
    assert "changed: f" in out


def test_reorder_within_file(tmp_path, capsys):
    repo = _repo(tmp_path)
    head, cls = A.split("class C:")
    (repo / "a.py").write_text("import os\n\nclass C:" + cls + "\n\n"
                               + head.split("import os\n", 1)[1])
    code, out = _run(repo, capsys)
    assert code == 0
    assert "3 defs" in out


def test_json_output(tmp_path, capsys):
    repo = _repo(tmp_path)
    _move(repo, lambda s: s.replace("x + 1", "x + 2"))
    code, out = _run(repo, capsys, "--json")
    assert code == 1
    assert '"changed"' in out and '"name": "f"' in out
