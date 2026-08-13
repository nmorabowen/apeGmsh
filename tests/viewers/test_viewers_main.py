"""``python -m apeGmsh.viewers`` — argparse + dispatch coverage.

The CLI itself can't easily run inside pytest (it would open a Qt
window). Instead we monkeypatch ``_open_results`` (or its underlying
``Results.from_native`` / ``Results.from_mpco``) and ``Results.viewer``
to capture the call shape, then drive ``main`` directly.

Phase 8 (ADR 0020 INV-1): ``Results.viewer(...)`` no longer takes
``model_h5=``. The CLI's ``--model-h5 PATH`` is now exclusively a
sibling-model pointer for the ``.mpco`` code path, forwarded into
``Results.from_mpco(path, model_h5=...)``.

ADR 0094 S5: ``python -m apeGmsh.viewers render …`` writes stills via
``results.render`` / ``render_pack`` / ``fem.render``. The no-subcommand
path must still open the Qt viewer. Drive ``main`` directly; skip is
exit 0.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest

import apeGmsh.viewers.__main__ as main_mod
from apeGmsh.viewers.__main__ import main

_MAIN_PY = (
    Path(__file__).resolve().parents[2]
    / "src" / "apeGmsh" / "viewers" / "__main__.py"
)


# =====================================================================
# Helpers
# =====================================================================

class _StubResults:
    """Standin for Results — records ``viewer`` / ``render`` invocations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.viewer_calls: list[tuple[Any, ...]] = []
        self.render_calls: list[tuple[Any, dict]] = []
        self.pack_calls: list[tuple[Any, dict]] = []

    def viewer(self, *, blocking: bool = True, title=None):
        self.viewer_calls.append((blocking, title))
        return None

    def render(self, path, **kwargs):
        dest = Path(path)
        self.render_calls.append((dest, kwargs))
        return dest

    def render_pack(self, out_dir, **kwargs):
        dest = Path(out_dir)
        self.pack_calls.append((dest, kwargs))
        return (dest / "mesh.png",)


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


def test_file_named_render_h5_stays_interactive(
    tmp_path: Path, patch_open_results,
):
    """A results file named ``render.h5`` is not the S5 subcommand."""
    fpath = tmp_path / "render.h5"
    fpath.write_bytes(b"")
    code = main([str(fpath)])
    assert code == 0
    stub = patch_open_results["results"][0]
    assert stub.viewer_calls == [(True, None)]
    assert stub.render_calls == []


# =====================================================================
# ADR 0094 S5 — ``render`` subcommand
# =====================================================================

def test_render_dispatches_to_results_render(
    tmp_path: Path, patch_open_results, capsys,
):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    out = tmp_path / "uz.png"
    code = main(["render", str(fpath), str(out)])
    assert code == 0
    assert patch_open_results["calls"] == [(fpath, None)]
    stub = patch_open_results["results"][0]
    assert stub.viewer_calls == []
    dest, kwargs = stub.render_calls[0]
    assert dest == out
    assert kwargs["view"] == "contour"
    assert kwargs["component"] is None
    assert kwargs["step"] == -1
    assert kwargs["deform"] is None
    assert kwargs["camera"] == "iso"
    assert kwargs["window_size"] == (1280, 720)
    assert str(out) in capsys.readouterr().out


def test_render_forwards_closed_flags(tmp_path: Path, patch_open_results):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    out = tmp_path / "mesh.png"
    code = main([
        "render", str(fpath), str(out),
        "--view", "mesh",
        "--component", "displacement_x",
        "--step", "0",
        "--deform", "2.5",
        "--camera", "xy",
        "--window", "320x240",
    ])
    assert code == 0
    _, kwargs = patch_open_results["results"][0].render_calls[0]
    assert kwargs["view"] == "mesh"
    assert kwargs["component"] == "displacement_x"
    assert kwargs["step"] == 0
    assert kwargs["deform"] == 2.5
    assert kwargs["camera"] == "xy"
    assert kwargs["window_size"] == (320, 240)


def test_render_deform_field_scale(tmp_path: Path, patch_open_results):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    out = tmp_path / "d.png"
    code = main([
        "render", str(fpath), str(out),
        "--deform", "velocity,0.5",
    ])
    assert code == 0
    _, kwargs = patch_open_results["results"][0].render_calls[0]
    assert kwargs["deform"] == ("velocity", 0.5)


def test_render_pack_calls_render_pack(tmp_path: Path, patch_open_results):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    dest = tmp_path / "pack"
    code = main(["render", str(fpath), "--pack", str(dest)])
    assert code == 0
    stub = patch_open_results["results"][0]
    assert stub.render_calls == []
    pack_dir, kwargs = stub.pack_calls[0]
    assert pack_dir == dest
    assert kwargs["camera"] == "iso"
    assert kwargs["window_size"] == (1280, 720)


def test_render_pack_and_output_returns_2(
    tmp_path: Path, patch_open_results, capsys,
):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    code = main([
        "render", str(fpath), str(tmp_path / "x.png"),
        "--pack", str(tmp_path / "pack"),
    ])
    assert code == 2
    assert "not both" in capsys.readouterr().err
    assert patch_open_results["calls"] == []


def test_render_missing_output_returns_2(
    tmp_path: Path, patch_open_results, capsys,
):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    code = main(["render", str(fpath)])
    assert code == 2
    assert "OUTPUT.png" in capsys.readouterr().err
    assert patch_open_results["calls"] == []


def test_render_missing_file_returns_2(tmp_path: Path, capsys):
    code = main(["render", str(tmp_path / "nope.h5"), str(tmp_path / "x.png")])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_render_bad_view_exits_nonzero(tmp_path: Path, capsys):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    with pytest.raises(SystemExit) as excinfo:
        main(["render", str(fpath), str(tmp_path / "x.png"), "--view", "poster"])
    assert excinfo.value.code != 0
    assert "view" in capsys.readouterr().err.lower()


def test_render_mpco_forwards_model_h5(tmp_path: Path, patch_open_results):
    fpath = tmp_path / "run.mpco"
    fpath.write_bytes(b"")
    model_h5 = tmp_path / "frame.model.h5"
    model_h5.write_bytes(b"")
    out = tmp_path / "uz.png"
    code = main([
        "render", str(fpath), str(out), "--model-h5", str(model_h5),
    ])
    assert code == 0
    assert patch_open_results["calls"] == [(fpath, model_h5)]


def test_render_skip_viewer_exits_0_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
):
    from apeGmsh.results import Results

    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    real = Results.demo(n_steps=2, n_elements=2)
    monkeypatch.setattr(main_mod, "_open_results", lambda path, model_h5: real)
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    out = tmp_path / "uz.png"
    code = main(["render", str(fpath), str(out)])
    assert code == 0
    assert not out.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_render_pack_skip_viewer_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
):
    from apeGmsh.results import Results

    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    real = Results.demo(n_steps=2, n_elements=2)
    monkeypatch.setattr(main_mod, "_open_results", lambda path, model_h5: real)
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    dest = tmp_path / "pack"
    code = main(["render", str(fpath), "--pack", str(dest)])
    assert code == 0
    assert not dest.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_render_valueerror_step_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
):
    from apeGmsh.results import Results

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    real = Results.demo(n_steps=2, n_elements=2)
    monkeypatch.setattr(main_mod, "_open_results", lambda path, model_h5: real)
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    code = main([
        "render", str(fpath), str(tmp_path / "x.png"), "--step", "99",
    ])
    assert code == 2
    assert "out of range" in capsys.readouterr().err
    assert not (tmp_path / "x.png").exists()


def test_render_model_h5_calls_fem_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
):
    from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    fem = build_simple_frame_fem()
    path = tmp_path / "model.h5"
    fem.to_h5(str(path))
    assert main_mod._is_model_only(path) is True
    out = tmp_path / "mesh.png"
    code = main(["render", str(path), str(out)])
    assert code == 0
    assert not out.exists()
    assert "[skip viewer] APEGMSH_SKIP_VIEWER set" in capsys.readouterr().out


def test_render_model_h5_rejects_pack(
    tmp_path: Path, capsys,
):
    from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

    fem = build_simple_frame_fem()
    path = tmp_path / "model.h5"
    fem.to_h5(str(path))
    code = main(["render", str(path), "--pack", str(tmp_path / "pack")])
    assert code == 2
    assert "fem.render_pack" in capsys.readouterr().err


def test_is_model_only_false_for_empty_h5(tmp_path: Path):
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    assert main_mod._is_model_only(fpath) is False


def test_render_without_skip_is_path_or_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from apeGmsh.results import Results

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    real = Results.demo(n_steps=2, n_elements=2)
    monkeypatch.setattr(main_mod, "_open_results", lambda path, model_h5: real)
    fpath = tmp_path / "run.h5"
    fpath.write_bytes(b"")
    out = tmp_path / "uz.png"
    code = main([
        "render", str(fpath), str(out),
        "--view", "contour",
        "--window", "320x240",
    ])
    assert code == 0
    if os.environ.get("APEGMSH_EXPECT_GL"):
        assert out.exists()
        assert out.stat().st_size > 0
    elif out.exists():
        assert out.stat().st_size > 0
    else:
        assert not out.exists()


def test_render_subcommand_has_no_event_loop():
    """AST: render path never names exec_ / QMainWindow / ResultsViewer."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    helpers = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_main_render", "_render_fem", "_render_results",
            "_exit_written", "_open_fem", "_is_model_only",
            "_parse_deform_flag", "_parse_window_flag",
        }:
            helpers[node.name] = node
    assert "_main_render" in helpers
    forbidden_attrs = {"exec_", "exec"}
    forbidden_names = {"QMainWindow", "ResultsViewer"}
    for node in helpers.values():
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in forbidden_attrs:
                raise AssertionError(
                    f"event-loop call in {node.name}: {child.attr}"
                )
            if isinstance(child, ast.Name) and child.id in forbidden_names:
                raise AssertionError(
                    f"forbidden name in {node.name}: {child.id}"
                )
