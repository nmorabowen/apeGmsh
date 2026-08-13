"""ADR 0094 S1 — fem.render / results.render.

GL-less CI stays green: the required path is
``APEGMSH_SKIP_VIEWER=1`` (returns ``None``, writes nothing, no event
loop). A live still is attempted only when the env is unset; no GL
then also returns ``None`` and is not a failure.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
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


def test_fem_render_without_skip_is_path_or_none(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    out = tmp_path / "fem.png"
    result = demo_results.fem.render(out, camera="iso", window_size=(320, 240))
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


def test_module_helpers_match_public_tokens() -> None:
    assert render_mod._VIEWS == frozenset(
        {"mesh", "contour", "deformed", "reactions"}
    )
    assert render_mod._CAMERAS == frozenset({"iso", "xy", "xz", "yz"})
    assert render_mod._require_deform(2.0) == ("displacement", 2.0)
    assert render_mod._require_deform(("velocity", 0.5)) == ("velocity", 0.5)
    assert render_mod._require_deform(None) is None
    assert render_mod._HISTORY_LABEL == "[matplotlib]"


def test_read_nodal_vector_field_matches_scatter(demo_results: Results) -> None:
    from apeGmsh.viewers._pump_set import read_nodal_vector_field
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    scene = build_fem_scene(demo_results.fem)
    step = 1  # last of n_steps=2
    got = read_nodal_vector_field(demo_results, scene, "displacement", step)
    assert got is not None
    ref = np.zeros_like(got)
    id_to_idx = scene.node_id_to_idx
    for axis, suf in enumerate(("x", "y", "z")):
        slab = demo_results.nodes.get(
            ids=scene.node_ids,
            component=f"displacement_{suf}",
            time=[step],
        )
        if slab.values.size == 0:
            continue
        for nid, v in zip(np.asarray(slab.node_ids), slab.values[0]):
            idx = id_to_idx.get(int(nid))
            if idx is not None:
                ref[idx, axis] = float(v)
    np.testing.assert_allclose(got, ref)


def test_screenshot_typeerror_propagates(tmp_path: Path) -> None:
    class _Boom:
        def screenshot(self, path: str) -> None:
            raise TypeError("not a GL failure")

    with pytest.raises(TypeError, match="not a GL failure"):
        render_mod._screenshot(_Boom(), tmp_path / "x.png")


def test_screenshot_gl_failure_keeps_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "x.png"
    out.write_bytes(b"previous-good-still")

    class _Boom:
        def screenshot(self, path: str) -> None:
            raise RuntimeError("cannot create render window")

    assert render_mod._screenshot(_Boom(), out) is None
    assert out.read_bytes() == b"previous-good-still"


def test_step_out_of_range_raises(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    with pytest.raises(ValueError, match="out of range"):
        demo_results.render(tmp_path / "x.png", step=99)
    with pytest.raises(ValueError, match="out of range"):
        demo_results.render(tmp_path / "x.png", step=-99)
    # Genuine negative index still works (skip path is enough to prove
    # it is accepted before a still is written).
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    assert demo_results.render(tmp_path / "neg.png", step=-1) is None


def test_deformed_without_displacement_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5
    from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    fem = build_simple_frame_fem()
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    path = tmp_path / "no_u.h5"
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0",
            node_ids=ids,
            components={"velocity_x": np.zeros((1, ids.size))},
        )
        w.end_stage()
    with Results.from_native(
        path, model=_open_model_from_h5(path), fem=fem,
    ) as r:
        with pytest.raises(ValueError, match="deformed"):
            r.render(tmp_path / "d.png", view="deformed")


def test_skip_viewer_render_pack_returns_empty(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    dest = tmp_path / "pack"
    assert demo_results.render_pack(dest) == ()
    assert not dest.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_no_fem_render_pack(demo_results: Results) -> None:
    assert not hasattr(demo_results.fem, "render_pack")


def test_pack_omits_reactions_on_demo(demo_results: Results) -> None:
    assert render_mod._has_static_reactions(demo_results) is False


def test_has_static_reactions_true_when_recorded() -> None:
    class _Stage:
        kind = "static"
        id = "g"

    class _Inspect:
        def components(self, stage=None):
            return {"nodes": ["displacement_x", "reaction_force_x"]}

    class _Fake:
        stages = [_Stage()]
        inspect = _Inspect()

    assert render_mod._has_static_reactions(_Fake()) is True


def test_has_static_reactions_false_on_transient() -> None:
    class _Stage:
        kind = "transient"
        id = "g"

    class _Inspect:
        def components(self, stage=None):
            return {"nodes": ["reaction_force_x"]}

    class _Fake:
        stages = [_Stage()]
        inspect = _Inspect()

    assert render_mod._has_static_reactions(_Fake()) is False


def test_pack_primary_is_s1_heuristic(demo_results: Results) -> None:
    # demo records displacement_z (zeros) so the S1 order picks uz.
    assert render_mod._pack_primary_component(demo_results) == "displacement_z"


def test_history_is_labeled_matplotlib(
    demo_results: Results, tmp_path: Path,
) -> None:
    out = tmp_path / "history.png"
    written = render_mod._try_history(
        demo_results, out, "displacement_x",
    )
    if written is None:
        pytest.skip("matplotlib missing or series too short")
    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_pack_without_skip_is_tuple(
    demo_results: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    dest = tmp_path / "pack"
    result = demo_results.render_pack(dest, window_size=(320, 240))
    assert isinstance(result, tuple)
    for path in result:
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.name in {
            "mesh.png", "contour.png", "deformed.png",
            "reactions.png", "history.png",
        }
    assert "reactions.png" not in {p.name for p in result}
    history = dest / "history.png"
    if history in result:
        assert render_mod._HISTORY_LABEL == "[matplotlib]"
