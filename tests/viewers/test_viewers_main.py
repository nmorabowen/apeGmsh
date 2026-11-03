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

VA6b (ADR 0128 slice 6): on Windows the render door draws through
``apeGmsh.agent_view.render_stills`` with one ``apegmsh-view/1`` spec.
The ``app_backend`` tests take that path on any platform with the app
replaced by a fake that records the spec and writes stand-in PNGs; every
other test is pinned to the VTK path, which Linux keeps (G2).
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import numpy as np
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
        self.viewer_policy: list[tuple[Any, bool]] = []
        self.render_calls: list[tuple[Any, dict]] = []
        self.pack_calls: list[tuple[Any, dict]] = []

    def viewer(
        self,
        *,
        blocking: bool = True,
        title=None,
        restore_session: "bool | str" = "prompt",
        save_session: bool = True,
    ):
        # ADR 0098 S6a: the child carries the parent's open policy
        # across the argv hop. ``viewer_calls`` keeps its old shape so
        # the dispatch assertions below are untouched; the policy has
        # its own list, asserted in tests/viewers/test_viewer_flip.py.
        self.viewer_calls.append((blocking, title))
        self.viewer_policy.append((restore_session, save_session))
        return None

    def render(self, path, **kwargs):
        dest = Path(path)
        self.render_calls.append((dest, kwargs))
        return dest

    def render_pack(self, out_dir, **kwargs):
        dest = Path(out_dir)
        self.pack_calls.append((dest, kwargs))
        return (dest / "mesh.png",)


@pytest.fixture(autouse=True)
def _vtk_backend_unless_asked(request, monkeypatch):
    """Every test here runs the VTK path (Linux's, G2) on any platform,
    except the ``app_backend`` tests, which take the Windows path."""
    from apeGmsh.viewers import render as render_mod

    if "app_backend" not in request.fixturenames:
        monkeypatch.setattr(render_mod, "_stills_backend", lambda: "vtk")


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


# =====================================================================
# VA6b: the Windows path, apegmsh-view/1 through agent_view
# =====================================================================


class _FakeApp:
    """``agent_view.render_stills`` without the app: records each spec,
    writes one stand-in PNG per shot and an all-ok manifest, as the app
    does on exit 0 (``apeGmshViewer/src/spec/manifest.ts``)."""

    def __init__(self) -> None:
        self.specs: list[dict] = []

    def __call__(self, spec, out, **kwargs):
        import json

        self.specs.append(json.loads(json.dumps(spec)))
        out = Path(out)
        assert not out.exists(), "the app refuses an --out that is not empty"
        out.mkdir(parents=True)
        shots = []
        for shot in spec["shots"]:
            (out / f"{shot['name']}.png").write_bytes(b"\x89PNG stand-in")
            shots.append({"name": shot["name"], "status": "ok", "png": f"{shot['name']}.png"})
        manifest = {"spec": spec["spec"], "shots": shots}
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


@pytest.fixture
def app_backend(monkeypatch):
    """The Windows path, on any platform, with a fake app."""
    import apeGmsh.agent_view as agent_view
    from apeGmsh.viewers import render as render_mod

    fake = _FakeApp()
    monkeypatch.setattr(render_mod, "_stills_backend", lambda: "app")
    monkeypatch.setattr(agent_view, "render_stills", fake)
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    return fake


@pytest.fixture
def ladruno_set(tmp_path: Path) -> Path:
    """The demo's run as one stem: ``run.h5`` beside ``run.ladruno``,
    the pair the app opens (``main/pairing.ts``). The demo writes
    ``demo_model.h5`` / ``demo_results.ladruno``, which do not pair."""
    import shutil

    from apeGmsh.results import Results

    src = tmp_path / "demo"
    demo = Results.demo(n_steps=3, n_elements=4, path=src)
    demo.close()
    run = tmp_path / "set"
    run.mkdir()
    shutil.copy(src / "demo_model.h5", run / "run.h5")
    shutil.copy(src / "demo_results.ladruno", run / "run.ladruno")
    return run / "run.ladruno"


def _auto_scale_oracle(ladruno: Path) -> float:
    """G3's closed form, computed here from the raw slabs: 0.12 of the
    bbox diagonal over the largest |u| component at the last step."""
    from apeGmsh.results import Results

    with Results.from_ladruno(ladruno, model_h5=ladruno.with_suffix(".h5")) as r:
        coords = np.asarray(r.fem.nodes.coords, dtype=float)
        stage = r.stage(r.stages[-1].id)
        comps = set(stage.nodes.available_components())
        peak = max(
            float(np.nanmax(np.abs(np.asarray(stage.nodes.get(component=c, time=-1).values))))
            for c in ("displacement_x", "displacement_y", "displacement_z") if c in comps
        )
    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    return 0.12 * diag / peak


def test_app_pack_is_one_v2_spec(app_backend, ladruno_set, tmp_path, capsys):
    """G5: the pack is ONE apegmsh-view/1 spec (mesh, contour, deformed),
    reactions dropped (G1), history kept as matplotlib (G6), and the
    app's manifest kept beside the PNGs."""
    import json

    dest = tmp_path / "pack"
    dest.mkdir()
    (dest / "old.txt").write_text("a pack folder may hold files")
    code = main(["render", str(ladruno_set), "--pack", str(dest), "--json"])
    assert code == 0, capsys.readouterr()
    assert len(app_backend.specs) == 1
    spec = app_backend.specs[0]
    assert spec["spec"] == "apegmsh-view/1"
    assert Path(spec["files"]["model"]) == ladruno_set.with_suffix(".h5").resolve()
    assert Path(spec["files"]["results"]) == ladruno_set.resolve()
    shots = {s["name"]: s for s in spec["shots"]}
    assert list(shots) == ["mesh", "contour", "deformed"]
    assert "stage" not in shots["mesh"]
    for name in ("contour", "deformed"):
        assert shots[name]["stage"] == 0
        assert shots[name]["step"] == 2  # the last of 3 steps
        # v1's starter heuristic (uz first): the demo records all three axes.
        assert shots[name]["contour"] == {"field": "DISPLACEMENT", "component": "z"}
    assert "deform" not in shots["contour"]
    assert shots["deformed"]["deform"]["scale"] == pytest.approx(
        _auto_scale_oracle(ladruno_set), rel=1e-12,
    )
    for name in ("mesh.png", "contour.png", "deformed.png", "manifest.json", "old.txt"):
        assert (dest / name).is_file(), name
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] is True
    written = [Path(p).name for p in out["written"]]
    assert written[:3] == ["mesh.png", "contour.png", "deformed.png"]
    assert "reactions.png" not in written


def test_app_still_maps_the_flags(app_backend, ladruno_set, tmp_path):
    """R13, R19, R20, R5-R8: component, step, deform, camera and size map onto one shot."""
    out = tmp_path / "stills" / "ux.png"
    code = main([
        "render", str(ladruno_set), str(out),
        "--view", "deformed", "--component", "displacement_x",
        "--step", "-2", "--deform", "40", "--camera", "xz",
        "--window", "640x480",
    ])
    assert code == 0
    spec = app_backend.specs[-1]
    assert spec["size"] == [640, 480]
    (shot,) = spec["shots"]
    assert shot["camera"] == "-y"
    assert shot["step"] == 1
    assert shot["contour"] == {"field": "DISPLACEMENT", "component": "x"}
    assert shot["deform"] == {"scale": 40.0}
    assert out.is_file()


@pytest.mark.parametrize("camera, expected", [
    ("iso", "iso"), ("xy", "top"), ("xz", "-y"), ("yz", "+x"),
])
def test_app_model_h5_is_a_mesh_shot(app_backend, ladruno_set, tmp_path, camera, expected):
    """R1, R3-R7: a model.h5 on disk is the spec's files.model, no copy."""
    model = ladruno_set.with_suffix(".h5")
    out = tmp_path / "mesh.png"
    assert main(["render", str(model), str(out), "--camera", camera]) == 0
    spec = app_backend.specs[-1]
    assert spec["files"] == {"model": str(model.resolve())}
    assert spec["shots"] == [{"name": "mesh", "camera": expected}]
    assert out.is_file()


@pytest.mark.parametrize("name", ["run.mpco", "run.results.h5"])
def test_app_refuses_every_source_but_ladruno(app_backend, tmp_path, capsys, name):
    """R22b, R31, G8: .ladruno only; the refusal names it; no app run."""
    src = tmp_path / name
    if name.endswith(".h5"):
        import h5py

        with h5py.File(src, "w") as f:
            f.create_group("stages")
    else:
        src.write_bytes(b"x")
    code = main(["render", str(src), str(tmp_path / "x.png")])
    assert code == 2
    assert ".ladruno" in capsys.readouterr().err
    assert app_backend.specs == []


@pytest.mark.parametrize("flags, needle", [
    (["--view", "reactions"], "G1"),
    (["--view", "contour", "--deform", "velocity,2"], "G4"),
])
def test_app_refuses_the_dropped_stills(app_backend, ladruno_set, tmp_path, capsys, flags, needle):
    code = main(["render", str(ladruno_set), str(tmp_path / "x.png"), *flags])
    assert code == 2
    assert needle in capsys.readouterr().err
    assert app_backend.specs == []


def test_app_model_h5_must_be_the_stem_model(app_backend, ladruno_set, tmp_path, capsys):
    """The app opens a .ladruno with its own <stem>.h5; another model is refused."""
    other = tmp_path / "other.h5"
    other.write_bytes(ladruno_set.with_suffix(".h5").read_bytes())
    code = main(["render", str(ladruno_set), str(tmp_path / "x.png"), "--model-h5", str(other)])
    assert code == 2
    assert "stem rule" in capsys.readouterr().err
    assert app_backend.specs == []


def test_app_without_the_app_exits_2(ladruno_set, tmp_path, monkeypatch, capsys):
    """G9: no app is an error, never a silent skip."""
    import apeGmsh.agent_view as agent_view
    from apeGmsh.viewers import render as render_mod

    def _missing(spec, out, **kwargs):
        raise FileNotFoundError("apeGmshViewer was not found")

    monkeypatch.setattr(render_mod, "_stills_backend", lambda: "app")
    monkeypatch.setattr(agent_view, "render_stills", _missing)
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    out = tmp_path / "x.png"
    assert main(["render", str(ladruno_set), str(out)]) == 2
    assert "apeGmshViewer was not found" in capsys.readouterr().err
    assert not out.exists()


def test_vtk_backend_never_reaches_agent_view(patch_open_results, tmp_path, monkeypatch):
    """G2: the Linux path is unchanged; agent_view is never called."""
    import apeGmsh.agent_view as agent_view

    def _boom(*a, **k):
        raise AssertionError("the VTK path called agent_view.render_stills")

    monkeypatch.setattr(agent_view, "render_stills", _boom)
    fpath = tmp_path / "run.ladruno"
    fpath.write_bytes(b"x")
    assert main(["render", str(fpath), str(tmp_path / "x.png")]) == 0
    assert patch_open_results["results"][0].render_calls


@pytest.mark.parametrize("platform, backend", [
    ("win32", "app"), ("linux", "vtk"), ("darwin", "vtk"),
])
def test_stills_backend_is_the_platform_rule(monkeypatch, platform, backend):
    from apeGmsh.viewers import render as render_mod

    monkeypatch.undo()  # drop the autouse pin: this test reads the real switch
    monkeypatch.setattr(render_mod.sys, "platform", platform)
    assert render_mod._stills_backend() == backend
