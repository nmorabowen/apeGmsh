"""scripts/check_app_wall.py: each rule fires on a positive case and passes clean code.

Builds a temp git repo with an `apeGmshViewer/` tree. The scan over the
checkout itself is a step of CI's `static-gates` job, not a test here.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_app_wall.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_app_wall", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wall = _load()


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    return tmp_path


def _put(root: Path, rel: str, source: str, track: bool = True) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    if track:
        _git(root, "add", rel)


def _rules(root: Path) -> list[str]:
    return [f.split(": ")[1].split(" ")[0] for f in wall.scan(root)]


@pytest.mark.parametrize("src", [
    "import apeGmsh\n",
    "from apeGmsh.results import Results\n",
    "from apeGmsh import apeGmsh\n",
])
def test_w1_flags_py_import(repo, src):
    _put(repo, "apeGmshViewer/tools/x.py", src)
    assert _rules(repo) == ["W1"]


@pytest.mark.parametrize("src", [
    "import x from 'apeGmsh';\n",
    "import { a } from 'apegmsh/results';\n",
    "export * from 'APEGMSH';\n",
    "const m = await import('apeGmsh/x');\n",
    "const m = require('apegmsh');\n",
    "import {\n  a,\n  b,\n} from '../../src/apeGmsh';\n",
    "import x from '../../outside';\n",
])
def test_w2_flags_specifier(repo, src):
    _put(repo, "apeGmshViewer/src/a.ts", src)
    assert _rules(repo) == ["W2"]


@pytest.mark.parametrize("src", [
    "spawn('python', ['-m', 'x']);\n",
    "execSync('py -3 run.py');\n",
    "child_process.execFile(cmd, ['a.py']);\n",
    "spawn(\n  'apeGmsh',\n  [],\n);\n",
    "utilityProcess.fork('/x/Python.exe');\n",
])
def test_w3_flags_subprocess_js(repo, src):
    _put(repo, "apeGmshViewer/src/a.js", src)
    assert _rules(repo) == ["W3"]


@pytest.mark.parametrize("src", [
    "subprocess.run(['python', 'x'])\n",
    "os.system('python x.py')\n",
])
def test_w3_flags_subprocess_py(repo, src):
    _put(repo, "apeGmshViewer/tools/x.py", "import os, subprocess\n" + src)
    assert _rules(repo) == ["W3"]


def test_clean_cases_pass(repo):
    _put(repo, "apeGmshViewer/src/a.ts", """\
        import * as THREE from 'three';
        import x from './reader';
        import y from '../lib/y';
        import z from "../../apeGmshViewer/src/z";
        // spawn('python', ['x.py']) and import from 'apeGmsh' in a comment
        /* require('apegmsh') */
        spawn('ffmpeg', ['-i', 'a.mp4']);
        spawn(process.execPath, ['apeGmshViewer/worker.js']);
        const r = /x/.exec(s);
        """)
    _put(repo, "apeGmshViewer/tools/x.py", """\
        # import apeGmsh  (comment mentioning python)
        import apeGmshViewer
        import json
        """)
    assert wall.scan(repo) == []
    assert wall.main(["--root", str(repo)]) == 0


def test_untracked_files_ignored(repo):
    _put(repo, "apeGmshViewer/ok.ts", "export const a = 1;\n")
    _put(repo, "apeGmshViewer/node_modules/p/bad.js", "require('apegmsh');\n", track=False)
    _put(repo, "apeGmshViewer/bad.py", "import apeGmsh\n", track=False)
    assert wall.scan(repo) == []


def test_missing_directory_passes(repo, capsys):
    assert wall.main(["--root", str(repo)]) == 0
    assert "nothing to check" in capsys.readouterr().out


def test_main_exit_one_and_format(repo, capsys):
    _put(repo, "apeGmshViewer/a.py", "x = 1\nimport apeGmsh\n")
    assert wall.main(["--root", str(repo)]) == 1
    assert capsys.readouterr().out.startswith("apeGmshViewer/a.py:2: W1 py-import — ")
