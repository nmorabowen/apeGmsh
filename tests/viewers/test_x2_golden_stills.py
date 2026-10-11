"""X2-a — golden stills + frame counts on the director path (the X2 oracle).

X2 moves ``Results.render`` / ``render_pack`` / ``assess(figures=True)``
/ ``export_animation`` off ``ResultsDirector`` onto ``ResultsSession``;
X3 then deletes the director path. This module pins **what the director
path produces today** as committed artifacts under
``tests/viewers/golden_stills/`` (PNGs + ``manifest.json``) so every later
X2 slice is judged against a recorded answer, not a live old-vs-new A/B.

What is pinned (all on one planar 8x2 quad plate built in-test, a
``static`` stage of 5 steps whose ``displacement_y`` *pattern* changes per
step, so a wrong step is a different picture, not a rescaled one):

1. ``results.render`` for each ``view`` in mesh|contour|deformed|reactions;
2. ``results.render_pack``: the file names it writes and the VTK stills;
3. ``export_animation``: the frame schedule/count for a fixed stride plus
   the first and last frames;
4. ``results.assess(figures=True)``: the figure file names.

Comparison (stills): both frames are block-averaged by ``downsample``
(4x4 -> 160x90) and compared per channel. A frame fails when any block
moves more than ``max_delta`` levels, or more than ``max_bad_fraction``
of blocks move more than ``bad_delta`` (manifest ``tolerance``). Block
averaging absorbs anti-aliasing / font-hinting differences between GL
stacks; a wrong step, component, deform or a moved arrow shifts whole
blocks by tens of levels.

GL policy: the curated ``suite`` lane sets ``APEGMSH_EXPECT_GL=1`` (Mesa),
so there a missing still is a FAILURE, never a skip. Elsewhere no GL
skips.

Regenerate (only when a change to the director path is intended, and
only from CI's renderer): ``APEGMSH_REGEN_GOLDENS=1 pytest <this file>``
rewrites the PNGs; a mismatch message also carries the actual
downsampled frame as base64 PNG so a Linux golden can be lifted from a
CI log.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.results import Results

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden_stills"
_MANIFEST = json.loads((_GOLDEN_DIR / "manifest.json").read_text("utf-8"))
_WINDOW = tuple(_MANIFEST["window"])
_CAMERA = _MANIFEST["camera"]
_TOL = _MANIFEST["tolerance"]
_REGEN = bool(os.environ.get("APEGMSH_REGEN_GOLDENS"))


# =====================================================================
# Fixture model
# =====================================================================


def _write_plate_results(dest: Path) -> Path:
    """8x2 quad plate (4 x 1, z=0) + a 5-step static stage.

    ``displacement_y`` carries a sine whose wave number grows with the
    step, so each step paints a different pattern even after the contour
    rescales to its own range. Reactions sit on the x=0 edge and rotate
    with the step. No ``displacement_z`` is recorded, so the pack's
    default component resolves to ``displacement_x``.
    """
    from apeGmsh import apeGmsh
    from apeGmsh.results.writers import NativeWriter

    fix = _MANIFEST["fixture"]
    g = apeGmsh(model_name="x2_golden", verbose=False, _artifacts=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(
            0.0, 0.0, 0.0, fix["lx"], fix["ly"], label="plate",
        )
        g.physical.add_surface("plate", name="Plate")
        g.mesh.structured.set_transfinite("plate", n=tuple(fix["n"]))
        g.mesh.generation.generate(dim=2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()

    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
    x, y = xyz[:, 0], xyz[:, 1]
    lx = float(fix["lx"])
    n_steps = int(fix["n_steps"])
    k = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)
    amp = k / (n_steps - 1)
    uy = 0.5 * amp * (x / lx) ** 2 + 0.1 * np.sin(np.pi * x * (k + 1) / lx)
    ux = 0.1 * amp * (0.5 - y) * (x / lx)
    edge = np.where(np.isclose(x, 0.0), 1.0, 0.0)
    rx = (amp + 0.1) * edge * (y - 0.5) * (k + 1)
    ry = -(amp + 0.1) * edge * (1.0 + y)

    path = dest / "x2_plate.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="push", kind="static",
            time=np.linspace(0.0, 1.0, n_steps),
        )
        w.write_nodes(sid, "partition_0", node_ids=ids, components={
            "displacement_x": ux,
            "displacement_y": uy,
            "reaction_force_x": rx,
            "reaction_force_y": ry,
        })
        w.end_stage()
    return path


@pytest.fixture(scope="module")
def plate_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_plate_results(tmp_path_factory.mktemp("x2_golden"))


@pytest.fixture
def plate(plate_h5: Path, monkeypatch: pytest.MonkeyPatch):
    """The plate Results, with the render environment pinned.

    The theme is a process global seeded from the user's QSettings, so a
    developer's saved theme would otherwise change every pixel. Pin the
    CI default in memory (``set_theme`` would persist it).
    """
    from tests.conftest import _open_model_from_h5
    from apeGmsh.viewers.ui.theme import PALETTES, THEME

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    monkeypatch.setattr(THEME, "_current", PALETTES[_MANIFEST["theme"]])
    with Results.from_native(
        plate_h5, model=_open_model_from_h5(plate_h5),
    ) as r:
        yield r


# =====================================================================
# Comparison
# =====================================================================


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _downsample(frame: np.ndarray) -> np.ndarray:
    f = int(_TOL["downsample"])
    rgb = np.asarray(frame)[..., :3].astype(np.float64)
    h, w = rgb.shape[0] // f * f, rgb.shape[1] // f * f
    blocks = rgb[:h, :w].reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3))
    return np.rint(blocks).astype(np.uint8)


def _png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _diff_verdict(actual: np.ndarray, golden: np.ndarray) -> str | None:
    """``None`` when ``actual`` (downsampled) matches ``golden``."""
    if actual.shape != golden.shape:
        return f"shape {actual.shape} != golden {golden.shape}"
    delta = np.abs(actual.astype(np.int16) - golden.astype(np.int16)).max(axis=2)
    worst = int(delta.max())
    bad = float((delta > int(_TOL["bad_delta"])).mean())
    if worst <= int(_TOL["max_delta"]) and bad <= float(_TOL["max_bad_fraction"]):
        return None
    return (
        f"max block delta {worst} (limit {_TOL['max_delta']}), "
        f"{bad:.2%} blocks > {_TOL['bad_delta']} "
        f"(limit {float(_TOL['max_bad_fraction']):.2%})"
    )


def _assert_golden(frame: np.ndarray, name: str) -> None:
    actual = _downsample(frame)
    golden_path = _GOLDEN_DIR / name
    if _REGEN:
        golden_path.write_bytes(_png_bytes(actual))
        return
    verdict = _diff_verdict(actual, _read_png(golden_path))
    if verdict is None:
        return
    b64 = base64.b64encode(_png_bytes(actual)).decode("ascii")
    raise AssertionError(
        f"golden still {name!r} differs: {verdict}.\n"
        f"actual (downsampled PNG, base64) for {name}:\n"
        f"X2GOLDEN-BEGIN {name} {b64} X2GOLDEN-END"
    )


def _require_still(written: Any, out: Path) -> None:
    """A missing still fails where GL is expected, else skips."""
    if written is not None:
        assert Path(written) == out and out.exists()
        return
    if os.environ.get("APEGMSH_EXPECT_GL"):
        raise AssertionError(f"no still written for {out.name} on a GL lane")
    pytest.skip("no offscreen GL context")


# =====================================================================
# 1. results.render, one still per view
# =====================================================================


@pytest.mark.parametrize("view", ["mesh", "contour", "deformed", "reactions"])
def test_render_view_matches_golden(plate, tmp_path: Path, view: str) -> None:
    spec = _MANIFEST["render"][view]
    out = tmp_path / f"{view}.png"
    written = plate.render(
        out, view=view, component=spec["component"], step=spec["step"],
        camera=_CAMERA, window_size=_WINDOW,
    )
    _require_still(written, out)
    _assert_golden(_read_png(out), spec["golden"])


# =====================================================================
# 2. results.render_pack: file set + VTK stills
# =====================================================================


def test_render_pack_files_and_stills(plate, tmp_path: Path) -> None:
    spec = _MANIFEST["render_pack"]
    dest = tmp_path / "pack"
    written = plate.render_pack(dest, camera=_CAMERA, window_size=_WINDOW)
    if not written:
        _require_still(None, dest / "mesh.png")
    names = [p.name for p in written]
    assert names == spec["files"]
    assert len(written) == spec["count"]
    for name, golden in spec["stills"].items():
        _assert_golden(_read_png(dest / name), golden)


# =====================================================================
# 3. export_animation: frame schedule, count, first/last frames
# =====================================================================


class _RecordingWriter:
    """Stand-in for an imageio writer: keeps every appended frame."""

    def __init__(self, sink: list[np.ndarray]) -> None:
        self._sink = sink

    def __enter__(self) -> _RecordingWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def append_data(self, frame: np.ndarray) -> None:
        self._sink.append(np.array(frame, copy=True))


def _install_recording_imageio(
    monkeypatch: pytest.MonkeyPatch, sink: list[np.ndarray],
) -> None:
    """Shadow ``imageio.v2`` so frames are captured, not encoded.

    The frame schedule and pixels are the director path's; encoding is
    imageio's. CI's suite lane does not install the ``animation`` extra,
    so recording the frames is what lets the oracle run there.
    """
    v2 = types.ModuleType("imageio.v2")
    v2.get_writer = lambda *_a, **_k: _RecordingWriter(sink)  # type: ignore[attr-defined]
    pkg = types.ModuleType("imageio")
    pkg.v2 = v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)


def _animation_stage(results: Results, component: str):
    """Offscreen plotter + director with one contour layer.

    Mirrors ``render_results(view='contour')`` on the director path:
    themed offscreen plotter, ``ResultsDirector.bind_plotter`` over the
    FEM scene, one registered contour diagram, fixed camera.
    """
    import pyvista as pv

    from apeGmsh.viewers.backends import PyVistaQtBackend
    from apeGmsh.viewers.diagrams import DiagramSpec, ResultsDirector
    from apeGmsh.viewers.diagrams._kinds import kind_def
    from apeGmsh.viewers.diagrams._selectors import normalize
    from apeGmsh.viewers.render import _apply_camera, _apply_theme, _ensure_stage
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    try:
        plotter = pv.Plotter(off_screen=True, window_size=list(_WINDOW))
    except (RuntimeError, OSError):
        return None, None
    _apply_theme(plotter)
    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    _ensure_stage(director)
    director.bind_plotter(
        PyVistaQtBackend(plotter), scene=scene, render_callback=plotter.render,
    )
    kdef = kind_def("contour")
    spec = DiagramSpec(
        kind="contour",
        selector=normalize(component=component),
        style=kdef.make_default_style(component),
    )
    director.registry.add(kdef.diagram_class(spec, director.results))
    director.set_step(0)
    _apply_camera(plotter, _CAMERA)
    return plotter, director


def test_export_animation_frames(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apeGmsh.viewers.animation import export_animation

    spec = _MANIFEST["animation"]
    frames: list[np.ndarray] = []
    _install_recording_imageio(monkeypatch, frames)
    plotter, director = _animation_stage(plate, spec["component"])
    if plotter is None:
        _require_still(None, tmp_path / "anim.gif")
    seen: list[int] = []
    real_set_step = director.set_step

    def _spy(i: int) -> None:
        seen.append(int(i))
        real_set_step(i)

    monkeypatch.setattr(director, "set_step", _spy)
    try:
        out = export_animation(
            plotter, director, tmp_path / "anim.gif",
            fps=spec["fps"], step_stride=spec["step_stride"],
        )
    finally:
        director.unbind_plotter()
        plotter.close()
    assert out == tmp_path / "anim.gif"
    # The schedule visits the frame steps, then restores the prior step.
    assert seen[:-1] == spec["frame_steps"]
    assert len(frames) == spec["frame_count"]
    _assert_golden(frames[0], spec["first"])
    _assert_golden(frames[-1], spec["last"])


# =====================================================================
# 4. results.assess(figures=True): figure names
# =====================================================================


def test_assess_figure_names(plate, tmp_path: Path) -> None:
    report = plate.assess(figures=True, out_dir=tmp_path / "assess")
    if not report.figures:
        _require_still(None, tmp_path / "assess" / "mesh.png")
    assert [p.name for p in report.figures] == _MANIFEST["assess_figures"]
    for p in report.figures:
        assert p.parent == tmp_path / "assess" and p.stat().st_size > 0


# =====================================================================
# The comparison itself: it must reject the regressions it guards
# =====================================================================


def test_comparison_rejects_wrong_renders(plate, tmp_path: Path) -> None:
    """Mutation proof, kept as a test: a wrong step, component or deform
    must FAIL against the committed goldens."""
    if _REGEN:
        pytest.skip("regenerating goldens")
    spec = _MANIFEST["render"]
    mutants = {
        "wrong step": ("contour", dict(view="contour", step=2)),
        "wrong component": ("contour", dict(
            view="contour", component="displacement_x")),
        "deform off": ("deformed", dict(view="contour")),
        "reactions wrong step": ("reactions", dict(view="reactions", step=1)),
    }
    for i, (label, (view, kw)) in enumerate(mutants.items()):
        kw.setdefault("component", spec[view]["component"])
        kw.setdefault("step", spec[view]["step"])
        out = tmp_path / f"mutant_{i}.png"
        written = plate.render(out, camera=_CAMERA, window_size=_WINDOW, **kw)
        _require_still(written, out)
        actual = _downsample(_read_png(out))
        golden = _read_png(_GOLDEN_DIR / spec[view]["golden"])
        assert _diff_verdict(actual, golden) is not None, (
            f"{label}: the comparison accepted a wrong render of {view!r}"
        )
