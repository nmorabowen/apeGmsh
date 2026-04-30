"""``python -m apeGmsh.viewers`` — argparse + dispatch coverage.

The CLI itself can't easily run inside pytest (it would open a Qt
window). Instead we monkeypatch ``Results.from_native`` /
``Results.from_mpco`` and ``Results.viewer`` to capture the call shape,
then drive ``main`` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apeGmsh import results as results_pkg
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
def stub_readers(monkeypatch):
    """Patch the two Results constructors to return _StubResults."""
    captured: dict[str, Any] = {"native": [], "mpco": []}

    def _fake_native(path):
        captured["native"].append(Path(path))
        return _StubResults(Path(path))

    def _fake_mpco(path):
        captured["mpco"].append(Path(path))
        return _StubResults(Path(path))

    monkeypatch.setattr(results_pkg.Results, "from_native", staticmethod(_fake_native))
    monkeypatch.setattr(results_pkg.Results, "from_mpco", staticmethod(_fake_mpco))
    return captured


# =====================================================================
# Dispatch
# =====================================================================

def test_main_dispatches_h5_to_from_native(tmp_path: Path, stub_readers):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")     # exists, contents irrelevant — readers stubbed
    code = main([str(fpath)])
    assert code == 0
    assert stub_readers["native"] == [fpath]
    assert stub_readers["mpco"] == []


def test_main_dispatches_mpco_to_from_mpco(tmp_path: Path, stub_readers):
    fpath = tmp_path / "run.mpco"
    fpath.write_bytes(b"")
    code = main([str(fpath)])
    assert code == 0
    assert stub_readers["mpco"] == [fpath]
    assert stub_readers["native"] == []


def test_main_extension_match_is_case_insensitive(tmp_path: Path, stub_readers):
    fpath = tmp_path / "RUN.MPCO"
    fpath.write_bytes(b"")
    code = main([str(fpath)])
    assert code == 0
    assert stub_readers["mpco"] == [fpath]


def test_main_missing_file_returns_2(tmp_path: Path, stub_readers, capsys):
    code = main([str(tmp_path / "nope.h5")])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err
    assert stub_readers["native"] == []


def test_main_passes_title(tmp_path: Path, stub_readers):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    # Patch the stub class to capture viewer kwargs by chaining via the
    # captured native call — both the constructor and the viewer call
    # need to run, so we intercept the title via the stub's record.
    stash = []

    def _fake_native(path):
        r = _StubResults(Path(path))
        # Replace with a viewer that records kwargs
        def viewer(*, blocking=True, title=None):
            stash.append({"blocking": blocking, "title": title})
        r.viewer = viewer
        return r

    import apeGmsh.viewers.__main__ as mod
    original = mod._open_results
    mod._open_results = lambda p: _fake_native(p)
    try:
        code = main([str(fpath), "--title", "My Title"])
    finally:
        mod._open_results = original

    assert code == 0
    assert stash == [{"blocking": True, "title": "My Title"}]


def test_main_invokes_viewer_blocking(tmp_path: Path, stub_readers):
    """`__main__` always calls viewer(blocking=True) — it IS the subprocess."""
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")

    stash = []

    def _fake_native(path):
        r = _StubResults(Path(path))
        def viewer(*, blocking=True, title=None):
            stash.append(blocking)
        r.viewer = viewer
        return r

    import apeGmsh.viewers.__main__ as mod
    original = mod._open_results
    mod._open_results = lambda p: _fake_native(p)
    try:
        code = main([str(fpath)])
    finally:
        mod._open_results = original

    assert code == 0
    assert stash == [True]
