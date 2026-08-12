"""ADR 0094 S1 — fem.render / results.render.

GL-less CI stays green: the required path is
``APEGMSH_SKIP_VIEWER=1`` (returns ``None``, writes nothing, no event
loop). A live still is attempted only when the env is unset; no GL
then also returns ``None`` and is not a failure.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from apeGmsh.results import Results
from apeGmsh.viewers import render as render_mod


_RENDER_PY = (
    Path(__file__).resolve().parents[2]
    / "src" / "apeGmsh" / "viewers" / "render.py"
)


@pytest.fixture
def demo_results() -> Results:
    return Results.demo(n_steps=2, n_elements=2)


def test_skip_viewer_results_render_writes_nothing(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    out = tmp_path / "uz.png"
    assert demo_results.render(out, view="contour") is None
    assert not out.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_skip_viewer_fem_render_writes_nothing(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    out = tmp_path / "mesh.png"
    assert demo_results.fem.render(out) is None
    assert not out.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_skip_viewer_does_not_create_parent_dir(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    out = tmp_path / "missing" / "uz.png"
    assert demo_results.render(out) is None
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("view", ["poster", "setup", "contour "])
def test_closed_view_set(demo_results: Results, tmp_path: Path, view: str) -> None:
    with pytest.raises(ValueError, match="view must be one of"):
        demo_results.render(tmp_path / "x.png", view=view)


@pytest.mark.parametrize("camera", ["iso ", "front", "camera"])
def test_closed_camera_set(
    demo_results: Results, tmp_path: Path, camera: str,
) -> None:
    with pytest.raises(ValueError, match="camera must be one of"):
        demo_results.render(tmp_path / "x.png", camera=camera)


def test_bad_deform_raises(demo_results: Results, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="deform must be"):
        demo_results.render(tmp_path / "x.png", deform="auto")


def test_render_module_never_enters_event_loop() -> None:
    tree = ast.parse(_RENDER_PY.read_text(encoding="utf-8"))
    forbidden_attrs = {"exec_", "exec"}
    forbidden_names = {"QMainWindow", "ResultsViewer"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
            raise AssertionError(f"event-loop call in render.py: {node.attr}")
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise AssertionError(f"forbidden name in render.py: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            raise AssertionError(f"forbidden attr in render.py: {node.attr}")


@pytest.mark.parametrize("view", ["mesh", "contour", "deformed", "reactions"])
def test_closed_views_accepted_under_skip(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    view: str,
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    out = tmp_path / f"{view}.png"
    assert demo_results.render(out, view=view) is None
    assert not out.exists()


@pytest.mark.parametrize("view", ["mesh", "contour", "deformed", "reactions"])
def test_render_without_skip_is_path_or_none(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    view: str,
) -> None:
    """No event loop either way. File exists only on a real Path return."""
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    out = tmp_path / f"{view}.png"
    result = demo_results.render(
        out, view=view, component="displacement_x", camera="iso",
        window_size=(320, 240),
    )
    if result is None:
        assert not out.exists()
    else:
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0


def test_fem_render_without_skip_is_path_or_none(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    out = tmp_path / "fem.png"
    result = demo_results.fem.render(out, camera="iso", window_size=(320, 240))
    if result is None:
        assert not out.exists()
    else:
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0


def test_module_helpers_match_public_tokens() -> None:
    assert render_mod._VIEWS == frozenset(
        {"mesh", "contour", "deformed", "reactions"}
    )
    assert render_mod._CAMERAS == frozenset({"iso", "xy", "xz", "yz"})
    assert render_mod._require_deform(2.0) == ("displacement", 2.0)
    assert render_mod._require_deform(("velocity", 0.5)) == ("velocity", 0.5)
    assert render_mod._require_deform(None) is None
