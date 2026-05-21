"""``Results._spawn_viewer_subprocess`` — argv emission contract.

The matching ``__main__`` parser side is covered by
``test_viewers_main.py``. This file pins the emitter side: that
``Results.viewer(blocking=False, ...)`` produces a well-formed argv
the parser can decode.

Phase 8 (ADR 0020 INV-1) — the legacy ``model_h5=`` kwarg on
:meth:`Results.viewer` and :meth:`Results._spawn_viewer_subprocess`
has been deleted; orientation is auto-resolved from the
Composed-file ``results.h5`` through :meth:`Results.from_native`.
The argv contract is now: ``[python, -m, apeGmsh.viewers, <path>,
?--title <str>]`` — no ``--model-h5`` token ever appears.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apeGmsh import results as results_pkg


# =====================================================================
# Helpers
# =====================================================================

class _CapturedPopen:
    """Stub for ``subprocess.Popen`` — records the argv it was called with."""

    instances: "list[_CapturedPopen]" = []

    def __init__(self, args, *_, **__) -> None:
        self.args = list(args)
        self.returncode = None
        _CapturedPopen.instances.append(self)

    def wait(self) -> int:  # pragma: no cover — tests don't wait
        return 0


@pytest.fixture
def patch_popen(monkeypatch):
    """Replace ``subprocess.Popen`` in the Results module with a stub."""
    _CapturedPopen.instances = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", _CapturedPopen)
    return _CapturedPopen


@pytest.fixture
def disk_results(tmp_path: Path):
    """Build a minimal ``Results`` with a non-None ``_path`` so the
    subprocess guard at ``Results.py:491-497`` passes."""
    from apeGmsh.results.Results import Results
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    r = Results.__new__(Results)
    r._path = fpath
    r._reader = None
    r._fem = None
    r._stage_id = None
    r._stages_cache = None
    r._model = None
    return r


# =====================================================================
# Tests
# =====================================================================

def test_spawn_omits_model_h5_token(disk_results, patch_popen):
    """Phase 8 — ``--model-h5`` is never in argv.

    The spawned viewer auto-resolves orientation from the
    Composed-file ``results.h5`` through :meth:`Results.from_native`.
    """
    disk_results._spawn_viewer_subprocess(title=None)
    assert len(patch_popen.instances) == 1
    argv = patch_popen.instances[0].args
    assert "--model-h5" not in argv


def test_spawn_includes_title(disk_results, patch_popen):
    """``--title`` is forwarded when set."""
    disk_results._spawn_viewer_subprocess(title="My Run")

    argv = patch_popen.instances[0].args
    assert "--title" in argv
    assert argv[argv.index("--title") + 1] == "My Run"


def test_spawn_argv_is_parseable_by_main_parser(disk_results, patch_popen):
    """End-to-end argv contract: what ``_spawn_viewer_subprocess`` emits
    is exactly what ``__main__``'s argparse decodes. Pins the wire
    between the two halves so future drift on either side surfaces here.
    """
    disk_results._spawn_viewer_subprocess(title="My Run")

    argv = patch_popen.instances[0].args
    # Strip the leading [python, -m, apeGmsh.viewers] — pass only the
    # arguments the parser would see.
    parser_argv = argv[3:]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--title", default=None)
    ns = parser.parse_args(parser_argv)

    assert ns.path == str(disk_results._path)
    assert ns.title == "My Run"


def test_spawn_raises_for_in_memory_results(patch_popen):
    """In-memory Results refuses to spawn."""
    from apeGmsh.results.Results import Results
    r = Results.__new__(Results)
    r._path = None
    r._reader = None
    r._fem = None
    r._stage_id = None
    r._stages_cache = None
    r._model = None

    with pytest.raises(RuntimeError, match="In-memory Results cannot launch"):
        r._spawn_viewer_subprocess(title=None)
