"""Phase 6 — cache key + path resolution."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from apeGmsh.results.writers import _cache


# =====================================================================
# Cache root resolution precedence
# =====================================================================

def test_explicit_cache_root_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APEGMSH_RESULTS_DIR", str(tmp_path / "via_env"))
    explicit = tmp_path / "via_kwarg"
    root = _cache.resolve_cache_root(explicit)
    assert root == explicit
    assert root.exists()


def test_env_var_used_when_no_explicit(tmp_path: Path, monkeypatch) -> None:
    env_dir = tmp_path / "via_env"
    monkeypatch.setenv("APEGMSH_RESULTS_DIR", str(env_dir))
    root = _cache.resolve_cache_root(None)
    assert root == env_dir


def test_default_cwd_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APEGMSH_RESULTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    root = _cache.resolve_cache_root(None)
    assert root == tmp_path / "results"
    assert root.exists()


# =====================================================================
# Cache key stability + sensitivity
# =====================================================================

def test_cache_key_stable(tmp_path: Path) -> None:
    f = tmp_path / "x.out"
    f.write_text("0.0 1.0\n", encoding="utf-8")
    k1 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="abc",
    )
    k2 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="abc",
    )
    assert k1 == k2


def test_cache_key_changes_on_mtime(tmp_path: Path) -> None:
    f = tmp_path / "x.out"
    f.write_text("0.0 1.0\n", encoding="utf-8")
    k1 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="abc",
    )
    # Modify the file (changes mtime AND size)
    f.write_text("0.0 1.0\n0.1 2.0\n", encoding="utf-8")
    k2 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="abc",
    )
    assert k1 != k2


def test_cache_key_changes_on_parser_version(tmp_path: Path) -> None:
    f = tmp_path / "x.out"
    f.write_text("0.0 1.0\n", encoding="utf-8")
    k1 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="abc",
    )
    k2 = _cache.compute_cache_key(
        [f], parser_version="2.0", fem_snapshot_id="abc",
    )
    assert k1 != k2


def test_cache_key_changes_on_snapshot_id(tmp_path: Path) -> None:
    f = tmp_path / "x.out"
    f.write_text("0.0 1.0\n", encoding="utf-8")
    k1 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="abc",
    )
    k2 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="def",
    )
    assert k1 != k2


def test_cache_key_handles_missing_files(tmp_path: Path) -> None:
    """Missing files are still part of the key — they invalidate."""
    f = tmp_path / "ghost.out"   # doesn't exist
    k1 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="abc",
    )
    f.write_text("0.0 1.0\n", encoding="utf-8")
    k2 = _cache.compute_cache_key(
        [f], parser_version="1.0", fem_snapshot_id="abc",
    )
    assert k1 != k2     # appearance changed the key


def test_cache_key_order_independent(tmp_path: Path) -> None:
    f1 = tmp_path / "a.out"
    f2 = tmp_path / "b.out"
    f1.write_text("data1", encoding="utf-8")
    f2.write_text("data2", encoding="utf-8")
    k1 = _cache.compute_cache_key(
        [f1, f2], parser_version="1.0", fem_snapshot_id="abc",
    )
    k2 = _cache.compute_cache_key(
        [f2, f1], parser_version="1.0", fem_snapshot_id="abc",
    )
    assert k1 == k2     # sort_path makes order irrelevant


# =====================================================================
# Cache paths
# =====================================================================

def test_cache_paths(tmp_path: Path) -> None:
    h5, manifest = _cache.cache_paths(tmp_path, "deadbeef")
    assert h5.parent == tmp_path
    assert h5.name == "deadbeef.h5"
    assert manifest.name == "deadbeef.manifest.h5"
