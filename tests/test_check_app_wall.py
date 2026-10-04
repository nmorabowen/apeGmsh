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


@pytest.mark.parametrize("src", [
    "import { mountLegend } from './legend.ts';\n",             # another panel (flat module)
    "import { x } from './legend/index.ts';\n",                 # another panel (directory)
    "import { Viewport } from '../renderer/viewport.ts';\n",   # the viewport
    "import { draw } from '../render/draw.ts';\n",             # the viewport, design name
    "import { readModel } from '../reader/read.ts';\n",        # the HDF5 reader
    "import { BlobStore } from '../state/blobs.ts';\n",        # the BlobStore
    "const b = await import('../state/blobs.ts');\n",          # dynamic import, same target
    "import { Effects } from '../effects.ts';\n",              # the bypass through effects
    "import { loadModel } from '../state/load.ts';\n",         # the bypass through the loader
    "import { reduce } from '../state/reduce.ts';\n",          # not on the allow-list either
    "import type { ChainNode } from '../chain/resolve.ts';\n", # a type import counts
    "import * as THREE from 'three';\n",                       # a package
    "import { thing } from '../ui';\n",                        # ui itself, not a module under ui/
])
def test_w4_flags_panel_import(repo, src):
    _put(repo, "apeGmshViewer/src/panels/header.ts", src)
    assert _rules(repo) == ["W4"]


def test_w4_flags_a_multi_line_import_in_a_panel(repo):
    # `import\n * as R from '...'` spans lines; the ` * as R from` line is not a comment.
    _put(repo, "apeGmshViewer/src/panels/header.ts", "import\n * as R from '../reader/read.ts';\nexport const r = R;\n")
    found = wall.scan(repo)
    assert [f.split(": ")[1].split(" ")[0] for f in found] == ["W4"]
    assert found[0].startswith("apeGmshViewer/src/panels/header.ts:2: W4 ")


@pytest.mark.parametrize("src", [
    "export { Effects } from '../effects.ts';\n",             # the laundering route
    "export { BlobStore } from '../state/blobs.ts';\n",
    "import { Store } from '../state/store.ts';\n",            # ui/ gets the types only
    "import { modelSummary } from '../state/selectors.ts';\n",
    "import * as THREE from 'three';\n",
])
def test_w4_flags_ui_leak(repo, src):
    _put(repo, "apeGmshViewer/src/ui/leak.ts", src)
    assert _rules(repo) == ["W4"]
    assert "ui module imports" in wall.scan(repo)[0]


def test_w4_allows_ui_to_import_ui_and_the_state_types(repo):
    _put(repo, "apeGmshViewer/src/ui/dom.ts", "import type { State } from '../state/types.ts';\nimport { x } from './icons.ts';\n")
    assert wall.scan(repo) == []


def test_comments_are_stripped_but_a_block_comment_line_starting_with_star_is_not_code(repo):
    _put(repo, "apeGmshViewer/src/panels/header.ts", """\
        /* a block comment
         * import { Viewport } from '../renderer/viewport.ts'
         */
        // import { readModel } from '../reader/read.ts'
        const s = "import { a } from '../effects.ts'"; // a string, not an import
        const t = `require('apegmsh')`;
        import { el } from '../ui/dom.ts';
        """)
    assert wall.scan(repo) == []


def test_w4_flags_from_a_panel_subdirectory(repo):
    _put(repo, "apeGmshViewer/src/panels/header/index.ts", "import { legend } from '../legend.ts';\n")
    assert _rules(repo) == ["W4"]
    assert "panel 'header' imports panel 'legend'" in wall.scan(repo)[0]


def test_w4_allows_the_store_selectors_types_ui_and_own_modules(repo):
    _put(repo, "apeGmshViewer/src/panels/header.ts", """\
        import { byId, el } from '../ui/dom.ts';
        import { modelSummary } from '../state/selectors.ts';
        import type { State, Store } from '../state/store.ts';
        import type { BlobRef, ChainNode } from '../state/types.ts';
        import { rows } from './header/rows.ts';
        // import { Viewport } from '../renderer/viewport.ts' in a comment
        """)
    _put(repo, "apeGmshViewer/src/renderer/app.ts", """\
        import { mountHeader } from '../panels/header.ts';
        import { mountLegend } from '../panels/legend.ts';
        import { BlobStore } from '../state/blobs.ts';
        import { Effects } from '../effects.ts';
        """)
    assert wall.scan(repo) == []


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
