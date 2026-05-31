"""Session-level autosave (``apeGmsh(save_to=...)``) and ``g.save()``.

Phase 1 of the session-save plan
([internal_docs/plan_session_save.md](../internal_docs/plan_session_save.md)):
the session writes the neutral-zone HDF5 on ``end()`` when ``save_to``
is configured at construction, and exposes a manual ``g.save()``.
OpenSees enrichment is intentionally **not** invoked here — the session
knows nothing about downstream solvers.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import pytest

from apeGmsh import apeGmsh


def _build_small_mesh(g: apeGmsh) -> None:
    """Tiny tagged box that exercises the broker + neutral-zone writer."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(3)


# ---------------------------------------------------------------------
# save_to → autosave on context-manager exit
# ---------------------------------------------------------------------


def test_autosave_writes_on_exit(tmp_path: Path) -> None:
    out = tmp_path / "auto.h5"
    with apeGmsh(model_name="auto", save_to=out) as g:
        _build_small_mesh(g)
    assert out.exists()
    with h5py.File(out, "r") as h5:
        assert "/nodes" in h5
        assert "/elements" in h5
        assert h5["/meta"].attrs["model_name"] == "auto"


def test_autosave_accepts_str_path(tmp_path: Path) -> None:
    out = tmp_path / "auto_str.h5"
    with apeGmsh(model_name="s", save_to=str(out)) as g:
        _build_small_mesh(g)
    assert out.exists()


def test_no_save_to_means_no_file(tmp_path: Path) -> None:
    out = tmp_path / "should_not_exist.h5"
    with apeGmsh(model_name="quiet") as g:
        _build_small_mesh(g)
    assert not out.exists()


# ---------------------------------------------------------------------
# Manual g.save()
# ---------------------------------------------------------------------


def test_manual_save_with_explicit_path(tmp_path: Path) -> None:
    out = tmp_path / "manual.h5"
    with apeGmsh(model_name="m") as g:
        _build_small_mesh(g)
        returned = g.save(out)
        assert returned == out
        assert out.exists()


def test_manual_save_uses_save_to_when_no_arg(tmp_path: Path) -> None:
    out = tmp_path / "default.h5"
    with apeGmsh(model_name="d", save_to=out) as g:
        _build_small_mesh(g)
        g.save()
        assert out.exists()


def test_manual_save_without_path_or_save_to_raises(tmp_path: Path) -> None:
    with apeGmsh(model_name="x") as g:
        _build_small_mesh(g)
        with pytest.raises(RuntimeError, match="requires a path"):
            g.save()


# ---------------------------------------------------------------------
# overwrite semantics
# ---------------------------------------------------------------------


def test_overwrite_false_raises_on_existing(tmp_path: Path) -> None:
    out = tmp_path / "exists.h5"
    out.write_bytes(b"")  # pre-existing file
    with apeGmsh(model_name="o", save_to=out, overwrite=False) as g:
        _build_small_mesh(g)
        with pytest.raises(FileExistsError):
            g.save()


def test_overwrite_true_replaces_existing(tmp_path: Path) -> None:
    out = tmp_path / "replace.h5"
    out.write_bytes(b"junk")
    with apeGmsh(model_name="r", save_to=out, overwrite=True) as g:
        _build_small_mesh(g)
        g.save()
    # Real HDF5 file now — h5py can open it
    with h5py.File(out, "r") as h5:
        assert "/nodes" in h5


# ---------------------------------------------------------------------
# Failure path: gmsh must still finalize when autosave raises
# ---------------------------------------------------------------------


def test_autosave_failure_does_not_block_finalize(tmp_path: Path) -> None:
    """If save_to points at an invalid location, the session must still
    finalize gmsh cleanly (so the next session can begin)."""
    bad = tmp_path / "no_such_dir" / "x.h5"  # parent dir does not exist
    with pytest.warns(UserWarning, match="autosave"):
        with apeGmsh(model_name="bad", save_to=bad) as g:
            _build_small_mesh(g)
    # If finalize was skipped, opening another session would raise.
    with apeGmsh(model_name="after") as g:
        _build_small_mesh(g)


# ---------------------------------------------------------------------
# Directory destination → resolves to <dir>/<model_name>.h5
# ---------------------------------------------------------------------


def test_save_to_directory_resolves_to_named_file(tmp_path: Path) -> None:
    # A directory destination means "drop <model_name>.h5 in here".
    # Previously this truncate-opened the dir as a file and raised a
    # cryptic PermissionError on Windows.
    d = tmp_path / "Model2D_Base"
    d.mkdir()
    with apeGmsh(model_name="frame", save_to=d) as g:
        _build_small_mesh(g)
        out = g.save()
        assert out == d / "frame.h5"
        assert out.is_file()


def test_save_path_arg_directory_resolves(tmp_path: Path) -> None:
    # Same resolution when the directory is passed to save() directly.
    d = tmp_path / "out_dir"
    d.mkdir()
    with apeGmsh(model_name="frame") as g:
        _build_small_mesh(g)
        out = g.save(d)
        assert out == d / "frame.h5"
        assert out.is_file()


def test_autosave_directory_does_not_warn(tmp_path: Path) -> None:
    # end() autosave must resolve a directory save_to the same way as
    # save(), not swallow a PermissionError into a UserWarning.
    d = tmp_path / "auto_dir"
    d.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with apeGmsh(model_name="frame", save_to=d) as g:
            _build_small_mesh(g)
    assert (d / "frame.h5").is_file()
