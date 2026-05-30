"""``python -m apeGmsh.viewers`` — argparse + dispatch coverage.

The CLI itself can't easily run inside pytest (it would open a Qt
window). Instead we monkeypatch ``_open_results`` (or its underlying
``Results.from_native`` / ``Results.from_mpco``) and ``Results.viewer``
to capture the call shape, then drive ``main`` directly.

Phase 8 (ADR 0020 INV-1): ``Results.viewer(...)`` no longer takes
``model_h5=``. The CLI's ``--model-h5 PATH`` is now exclusively a
sibling-model pointer for the ``.mpco`` code path, forwarded into
``Results.from_mpco(path, model_h5=...)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import apeGmsh.viewers.__main__ as main_mod
from apeGmsh.viewers.__main__ import main


# =====================================================================
# Helpers
# =====================================================================

class _StubResults:
    """Standin for Results — records ``viewer(...)`` invocations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.viewer_calls: list[tuple[Any, ...]] = []

    def viewer(self, *, blocking: bool = True, title=None):
        self.viewer_calls.append((blocking, title))
        return None


@pytest.fixture
def patch_open_results(monkeypatch):
    """Patch ``_open_results`` to bypass the real readers.

    Returns the list of ``(path, model_h5)`` calls plus the stub
    Results each call returned, so tests can inspect both the
    dispatch shape and any subsequent ``viewer(...)`` invocations.
    """
    captured: dict[str, Any] = {
        "calls": [],         # list[tuple[Path, Path | None]]
        "results": [],       # list[_StubResults]
    }

    def _fake_open(path: Path, model_h5):
        stub = _StubResults(Path(path))
        captured["calls"].append((Path(path), model_h5))
        captured["results"].append(stub)
        return stub

    monkeypatch.setattr(main_mod, "_open_results", _fake_open)
    return captured


# =====================================================================
# Dispatch
# =====================================================================

def test_main_dispatches_h5(tmp_path: Path, patch_open_results):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    code = main([str(fpath)])
    assert code == 0
    assert patch_open_results["calls"] == [(fpath, None)]


def test_main_dispatches_mpco_with_model_h5(tmp_path: Path, patch_open_results):
    fpath = tmp_path / "run.mpco"
    fpath.write_bytes(b"")
    model_h5 = tmp_path / "frame.model.h5"
    model_h5.write_bytes(b"")
    code = main([str(fpath), "--model-h5", str(model_h5)])
    assert code == 0
    assert patch_open_results["calls"] == [(fpath, model_h5)]


def test_main_missing_file_returns_2(tmp_path: Path, patch_open_results, capsys):
    code = main([str(tmp_path / "nope.h5")])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err
    assert patch_open_results["calls"] == []


def test_main_passes_title(tmp_path: Path, patch_open_results):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    code = main([str(fpath), "--title", "My Title"])
    assert code == 0
    assert len(patch_open_results["results"]) == 1
    stub = patch_open_results["results"][0]
    assert stub.viewer_calls == [(True, "My Title")]


def test_main_invokes_viewer_blocking(tmp_path: Path, patch_open_results):
    """`__main__` always calls viewer(blocking=True) — it IS the subprocess."""
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    code = main([str(fpath)])
    assert code == 0
    stub = patch_open_results["results"][0]
    assert stub.viewer_calls == [(True, None)]


# =====================================================================
# --model-h5: required for .mpco, optional model-source override for native
# =====================================================================

def test_main_mpco_without_model_h5_exits_2(tmp_path: Path, capsys):
    """`.mpco` path with no ``--model-h5`` exits 2 with a helpful message.

    The ``model_h5 is None`` branch in ``_open_results`` calls
    ``sys.exit(2)`` before any Results call, so no patching is needed
    for the readers themselves — ``SystemExit`` propagates out of
    ``main`` and we assert on its code.
    """
    fpath = tmp_path / "run.mpco"
    fpath.write_bytes(b"")
    with pytest.raises(SystemExit) as excinfo:
        main([str(fpath)])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--model-h5" in err
    assert ".mpco" in err


def test_main_native_forwards_model_h5_override(tmp_path: Path, patch_open_results):
    """Native ``.h5`` path with ``--model-h5`` — the CLI forwards it into
    ``_open_results``, which for native files uses it as a model-source
    override (read the model from the sibling archive instead of the
    results file). We just check the arg makes it through; the override
    semantics are exercised by ``_open_results`` / the demo test."""
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    extra = tmp_path / "extra.model.h5"
    extra.write_bytes(b"")
    code = main([str(fpath), "--model-h5", str(extra)])
    assert code == 0
    assert patch_open_results["calls"] == [(fpath, extra)]


def test_main_extension_match_is_case_insensitive(tmp_path: Path, patch_open_results):
    fpath = tmp_path / "RUN.MPCO"
    fpath.write_bytes(b"")
    model_h5 = tmp_path / "frame.model.h5"
    model_h5.write_bytes(b"")
    code = main([str(fpath), "--model-h5", str(model_h5)])
    assert code == 0
    assert patch_open_results["calls"] == [(fpath, model_h5)]
