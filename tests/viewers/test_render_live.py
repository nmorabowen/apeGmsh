"""ADR 0094 S4 — g.model.render / g.mesh.render (live session).

Not qt-marked. GL-less CI stays green: SKIP_VIEWER returns None
and writes nothing. from_h5 / closed sessions raise RuntimeError
even under SKIP_VIEWER (INV-5: do not fake a BRep still from FEMData).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from apeGmsh import apeGmsh


def _assert_still(result: Path | None, out: Path) -> None:
    if os.environ.get("APEGMSH_EXPECT_GL"):
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0
        return
    if result is None:
        assert not out.exists()
    else:
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0


def _box_session():
    g = apeGmsh(model_name="s4-live", verbose=False)
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    return g


def test_skip_viewer_model_render_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    out = tmp_path / "geom.png"
    g = _box_session()
    try:
        assert g.model.render(out) is None
    finally:
        g.end()
    assert not out.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_skip_viewer_mesh_render_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    out = tmp_path / "mesh.png"
    g = _box_session()
    try:
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        assert g.mesh.render(out) is None
    finally:
        g.end()
    assert not out.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_skip_viewer_does_not_create_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    out = tmp_path / "missing" / "geom.png"
    g = _box_session()
    try:
        assert g.model.render(out) is None
    finally:
        g.end()
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("camera", ["iso ", "front", "camera"])
def test_closed_camera_set(tmp_path: Path, camera: str) -> None:
    g = _box_session()
    try:
        with pytest.raises(ValueError, match="camera must be one of"):
            g.model.render(tmp_path / "x.png", camera=camera)
        with pytest.raises(ValueError, match="camera must be one of"):
            g.mesh.render(tmp_path / "x.png", camera=camera)
    finally:
        g.end()


def test_from_h5_model_render_raises(tmp_path: Path) -> None:
    h5 = tmp_path / "m.h5"
    with apeGmsh(model_name="s4-h5", save_to=h5, verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
        g.physical.add_volume("body", name="body")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.mesh.queries.get_fem_data(dim=3)
    loaded = apeGmsh.from_h5(h5)
    with pytest.raises(RuntimeError, match="live Gmsh kernel"):
        loaded.model.render(tmp_path / "geom.png")
    assert not (tmp_path / "geom.png").exists()


def test_from_h5_mesh_render_raises(tmp_path: Path) -> None:
    h5 = tmp_path / "m.h5"
    with apeGmsh(model_name="s4-h5", save_to=h5, verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
        g.physical.add_volume("body", name="body")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.mesh.queries.get_fem_data(dim=3)
    loaded = apeGmsh.from_h5(h5)
    with pytest.raises(RuntimeError, match="live Gmsh kernel"):
        loaded.mesh.render(tmp_path / "mesh.png")
    assert not (tmp_path / "mesh.png").exists()


def test_closed_session_raises(tmp_path: Path) -> None:
    g = _box_session()
    g.end()
    with pytest.raises(RuntimeError, match="live Gmsh kernel"):
        g.model.render(tmp_path / "geom.png")
    with pytest.raises(RuntimeError, match="live Gmsh kernel"):
        g.mesh.render(tmp_path / "mesh.png")


def test_mesh_render_without_generate_raises(tmp_path: Path) -> None:
    g = _box_session()
    try:
        with pytest.raises(RuntimeError, match="generated mesh"):
            g.mesh.render(tmp_path / "mesh.png")
    finally:
        g.end()
    assert not (tmp_path / "mesh.png").exists()


def test_model_render_without_skip_is_path_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    out = tmp_path / "geom.png"
    g = _box_session()
    try:
        result = g.model.render(out, camera="iso", window_size=(320, 240))
    finally:
        g.end()
    _assert_still(result, out)


def test_mesh_render_without_skip_is_path_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    out = tmp_path / "mesh.png"
    g = _box_session()
    try:
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        result = g.mesh.render(out, camera="iso", window_size=(320, 240))
    finally:
        g.end()
    _assert_still(result, out)


def test_chain_phase_live_kernel_still_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After get_fem_data() the kernel is still live — stills must work."""
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    geom = tmp_path / "geom.png"
    mesh = tmp_path / "mesh.png"
    g = _box_session()
    try:
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.mesh.queries.get_fem_data(dim=3)
        geom_r = g.model.render(geom, window_size=(320, 240))
        mesh_r = g.mesh.render(mesh, window_size=(320, 240))
    finally:
        g.end()
    _assert_still(geom_r, geom)
    _assert_still(mesh_r, mesh)
