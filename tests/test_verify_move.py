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


# --- relative imports resolve by module path (S1-v) -------------------------

SRC = "src/pkg"
BODY = '''\
class K:
    def go(self):
        from {dots}x import y
        return y
'''


def _pkg_repo(tmp_path, base_path, base_text):
    _git(tmp_path, "init", "-q")
    f = tmp_path / base_path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(base_text)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _run_paths(repo, capsys, *paths):
    code = vm.main(["--repo", str(repo), "--base", "HEAD", "--head",
                    "WORKTREE", *paths])
    return code, capsys.readouterr().out


def _move_to(repo, old, new, text):
    (repo / old).unlink()
    dest = repo / new
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)


def test_relative_import_one_package_deeper_passes(tmp_path, capsys):
    repo = _pkg_repo(tmp_path, f"{SRC}/mod.py", BODY.format(dots="."))
    _move_to(repo, f"{SRC}/mod.py", f"{SRC}/sub/mod.py", BODY.format(dots=".."))
    code, out = _run_paths(repo, capsys, SRC)
    assert code == 0, out


def test_relative_import_different_target_fails(tmp_path, capsys):
    repo = _pkg_repo(tmp_path, f"{SRC}/mod.py", BODY.format(dots="."))
    # same dots after moving deeper: now resolves to pkg.sub.x, not pkg.x
    _move_to(repo, f"{SRC}/mod.py", f"{SRC}/sub/mod.py", BODY.format(dots="."))
    code, out = _run_paths(repo, capsys, SRC)
    assert code == 1
    assert "changed: K.go" in out


def test_relative_import_in_init_resolves_against_package(tmp_path, capsys):
    # `.x` inside pkg/sub/__init__.py is pkg.sub.x (the package itself).
    repo = _pkg_repo(tmp_path, f"{SRC}/sub/__init__.py", BODY.format(dots="."))
    _move_to(repo, f"{SRC}/sub/__init__.py", f"{SRC}/sub/impl.py",
             BODY.format(dots="pkg.sub."))
    code, out = _run_paths(repo, capsys, SRC)
    assert code == 0, out
    (repo / f"{SRC}/sub/impl.py").write_text(BODY.format(dots="pkg."))
    code, _ = _run_paths(repo, capsys, SRC)
    assert code == 1


def test_unresolvable_path_keeps_raw_dump(tmp_path, capsys):
    repo = _pkg_repo(tmp_path, "lib/mod.py", BODY.format(dots="."))
    _move_to(repo, "lib/mod.py", "lib/sub/mod.py", BODY.format(dots=".."))
    code, out = _run_paths(repo, capsys, "lib")
    assert code == 1  # raw level differs; no resolution outside src/, no crash
    assert "changed: K.go" in out


# --- --class-map: methods moved into mixins (S1-m) ---------------------------

CM_BASE = '''\
class A:
    """Doc A."""

    def m(self):
        return 1

    def n(self):
        return 2
'''
CM_HEAD_A = '''\
from mix import M


class A(M):
    """Doc A."""

    def n(self):
        return 2
'''
CM_HEAD_M = '''\
class M:
    def m(self):
        return 1
'''


def _cm_repo(tmp_path, head_a=CM_HEAD_A, head_m=CM_HEAD_M):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text(CM_BASE)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "a.py").write_text(head_a)
    (tmp_path / "mix.py").write_text(head_m)
    return tmp_path


def _cm_run(repo, capsys, *extra):
    code = vm.main(["--repo", str(repo), "--base", "HEAD", "--head",
                    "WORKTREE", *extra, "a.py", "mix.py"])
    return code, capsys.readouterr().out


def test_class_map_method_move_passes(tmp_path, capsys):
    repo = _cm_repo(tmp_path)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 2, out  # exempt headers: never a plain OK
    assert "moved:   A.m -> M.m" in out
    assert "identical multiset" not in out


def test_method_move_without_class_map_fails(tmp_path, capsys):
    repo = _cm_repo(tmp_path)
    code, out = _cm_run(repo, capsys)
    assert code == 1
    assert "removed: A.m" in out and "added:   M.m" in out
    assert "class headers" not in out and "moved:   " not in out


def test_class_map_changed_body_fails(tmp_path, capsys):
    repo = _cm_repo(tmp_path, head_m=CM_HEAD_M.replace("return 1", "return 9"))
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 1
    assert "removed: A.m" in out and "added:   M.m" in out


def test_class_map_name_collision_fails(tmp_path, capsys):
    head_m = CM_HEAD_M + "\n\nclass N:\n    def m(self):\n        return 1\n"
    repo = _cm_repo(tmp_path, head_a=CM_HEAD_A.replace("(M)", "(M, N)"),
                    head_m=head_m)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M,N")
    assert code == 1
    assert "removed: A.m" in out


def test_class_map_unmapped_header_fails(tmp_path, capsys):
    head_m = CM_HEAD_M + "\n\nclass Other:\n    pass\n"
    repo = _cm_repo(tmp_path, head_m=head_m)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 1
    assert "added:   Other" in out


def test_class_map_header_sections_printed(tmp_path, capsys):
    repo = _cm_repo(tmp_path)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 2
    assert "class headers (review by hand):" in out
    assert "added: M" in out and "changed: A" in out
    assert "bases" in out


def test_class_map_bad_spec_errors(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        vm.main(["--repo", str(tmp_path), "--class-map", "nonsense", "a.py"])


def test_class_map_json_needs_review(tmp_path, capsys):
    repo = _cm_repo(tmp_path)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M", "--json")
    assert code == 2
    assert '"ok": false' in out and '"needs_review": true' in out


def test_class_map_requires_inheritance_link(tmp_path, capsys):
    repo = _cm_repo(tmp_path, head_a=CM_HEAD_A.replace("(M)", ""))
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 1
    assert "removed: A.m" in out and "added:   M.m" in out


def test_class_map_shadow_in_old_not_ok(tmp_path, capsys):
    head_a = CM_HEAD_A.replace('    """Doc A."""\n',
                               '    """Doc A."""\n    m = None\n')
    repo = _cm_repo(tmp_path, head_a=head_a)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 1
    assert "class-sensitive: A.m (m is also bound by a non-def in A)" in out


def test_class_map_shadow_in_new_not_ok(tmp_path, capsys):
    head_m = CM_HEAD_M + "    m = lambda self: 99\n"
    repo = _cm_repo(tmp_path, head_m=head_m)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 1
    assert "class-sensitive: A.m (m is also bound by a non-def in M)" in out


def test_class_map_self_map_rejected(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        vm.main(["--repo", str(tmp_path), "--class-map", "A=A", "a.py"])


def _sens_run(tmp_path, capsys, body):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text(CM_BASE.replace("return 1", body))
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "a.py").write_text(CM_HEAD_A)
    (tmp_path / "mix.py").write_text(CM_HEAD_M.replace("return 1", body))
    return _cm_run(tmp_path, capsys, "--class-map", "A=M")


def test_class_map_refuses_mangled_name(tmp_path, capsys):
    code, out = _sens_run(tmp_path, capsys, "return self.__x")
    assert code == 1
    assert "class-sensitive: A.m (private name-mangled __x)" in out


def test_class_map_refuses_zero_arg_super(tmp_path, capsys):
    code, out = _sens_run(tmp_path, capsys, "return super().m()")
    assert code == 1
    assert "class-sensitive: A.m (zero-argument super())" in out


def test_class_map_refuses_dunder_class(tmp_path, capsys):
    code, out = _sens_run(tmp_path, capsys, "return __class__")
    assert code == 1
    assert "class-sensitive: A.m (__class__)" in out


def test_class_map_allows_dunder_names(tmp_path, capsys):
    code, out = _sens_run(tmp_path, capsys, "return self.__dict__")
    assert code == 2 and "class-sensitive" not in out


def test_class_map_move_never_exits_zero_sibling_base_wins(tmp_path, capsys):
    # M, N and class A(N, M) already exist in the base: no header changes, only
    # m moves A -> M, and N.m shadows it in the MRO. Only review can tell.
    wired_a = ("from mix import M, N\n\n\nclass A(N, M):\n"
               "    def m(self):\n        return 1\n\n"
               "    def n(self):\n        return 2\n")
    mix = ("class M:\n    pass\n\n\nclass N:\n"
           "    def m(self):\n        return 99\n")
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text(wired_a)
    (tmp_path / "mix.py").write_text(mix)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "a.py").write_text(wired_a.replace(
        "    def m(self):\n        return 1\n\n", ""))
    (tmp_path / "mix.py").write_text(mix.replace(
        "class M:\n    pass\n", "class M:\n    def m(self):\n        return 1\n"))
    code, out = _cm_run(tmp_path, capsys, "--class-map", "A=M", "--json")
    assert code == 2
    assert '"needs_review": true' in out and '"class_headers": []' in out


def test_class_map_same_named_old_in_other_file(tmp_path, capsys):
    # a.py's head A has no bases; other.py defines another A(M)
    repo = _cm_repo(tmp_path, head_a=CM_HEAD_A.replace("(M)", ""))
    (repo / "other.py").write_text("from mix import M\n\n\nclass A(M):\n    pass\n")
    code = vm.main(["--repo", str(repo), "--base", "HEAD", "--head", "WORKTREE",
                    "--class-map", "A=M", "a.py", "mix.py", "other.py"])
    out = capsys.readouterr().out
    assert code == 1
    assert "removed: A.m" in out


def test_class_map_import_alias_shadow_refused(tmp_path, capsys):
    head_a = CM_HEAD_A.replace('    """Doc A."""\n',
                               '    """Doc A."""\n    from os import path as m\n')
    repo = _cm_repo(tmp_path, head_a=head_a)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 1
    assert "class-sensitive: A.m" in out


def test_class_map_annassign_shadow_refused(tmp_path, capsys):
    head_m = CM_HEAD_M + "    m: int = 3\n"
    repo = _cm_repo(tmp_path, head_m=head_m)
    code, out = _cm_run(repo, capsys, "--class-map", "A=M")
    assert code == 1
