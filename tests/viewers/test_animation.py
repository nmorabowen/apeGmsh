"""Animation export — drive the director, capture frames, encode.

Exercises ``export_animation`` end-to-end without spinning up Qt:
constructs a ``ResultsDirector`` against a small synthetic Results,
points the director's render callback at an off-screen plotter, and
verifies that GIF and (when ``imageio-ffmpeg`` is available) MP4
files are produced with sensible sizes and the user's step is
restored on completion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.animation import export_animation
from apeGmsh.viewers.diagrams import (
    DiagramSpec, SlabSelector, VectorGlyphDiagram, VectorGlyphStyle,
)
from apeGmsh.viewers.diagrams._director import ResultsDirector
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5

# The optional `animation` extra (imageio / imageio-ffmpeg) is
# not installed in the curated CI suite; skip cleanly when absent.
pytest.importorskip("imageio")


def _has_ffmpeg() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def animation_results(g, tmp_path: Path):
    """Cube mesh with displacement_x/y/z across 5 steps."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_steps = 5
    base = np.broadcast_to(node_ids.astype(np.float64), (n_steps, len(node_ids)))
    t = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)
    components = {
        "displacement_x": base + t * 0.1,
        "displacement_y": base + t * 0.2,
        "displacement_z": base + t * 0.3,
    }
    path = tmp_path / "anim.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids, components=components,
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def animation_setup(animation_results):
    """Off-screen plotter + director with a single vector-glyph layer."""
    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = (320, 240)  # small frames keep tests fast
    scene = build_fem_scene(animation_results.fem)
    plotter.add_mesh(scene.grid, color="lightgray", show_edges=True)

    director = ResultsDirector(animation_results)
    director._render_callback = plotter.render  # noqa: SLF001

    spec = DiagramSpec(
        kind="vector_glyph",
        selector=SlabSelector(component="displacement"),
        style=VectorGlyphStyle(scale=1.0),
    )
    diagram = VectorGlyphDiagram(spec, animation_results)
    diagram.attach(plotter, animation_results.fem, scene)
    director.registry.add(diagram)

    yield plotter, director
    plotter.close()


# =====================================================================
# Format detection / validation
# =====================================================================

def test_unsupported_suffix_raises(animation_setup, tmp_path: Path):
    plotter, director = animation_setup
    with pytest.raises(ValueError, match="suffix"):
        export_animation(plotter, director, tmp_path / "out.avi")


def test_zero_fps_raises(animation_setup, tmp_path: Path):
    plotter, director = animation_setup
    with pytest.raises(ValueError, match="fps"):
        export_animation(plotter, director, tmp_path / "out.gif", fps=0)


def test_no_steps_raises(g, tmp_path: Path):
    """Empty Results → clear error rather than a silent zero-frame file."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    path = tmp_path / "empty.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
    results = Results.from_native(path, model=_open_model_from_h5(path))
    director = ResultsDirector(results)
    plotter = pv.Plotter(off_screen=True)
    try:
        with pytest.raises(RuntimeError, match="no steps"):
            export_animation(plotter, director, tmp_path / "out.gif")
    finally:
        plotter.close()


# =====================================================================
# GIF export (Pillow — no ffmpeg required)
# =====================================================================

def test_gif_export_writes_file(animation_setup, tmp_path: Path):
    plotter, director = animation_setup
    out = tmp_path / "anim.gif"
    result = export_animation(plotter, director, out, fps=10)
    assert result == out
    assert out.exists()
    # GIF header is "GIF87a" or "GIF89a".
    head = out.read_bytes()[:6]
    assert head.startswith(b"GIF8")
    # 5 steps × small frame should be at least a few hundred bytes.
    assert out.stat().st_size > 200


def test_gif_export_step_stride(animation_setup, tmp_path: Path):
    """Stride > 1 still emits the final frame so the end state is visible."""
    plotter, director = animation_setup
    # Push the director to a non-zero step so we can verify restoration.
    director.set_step(2)
    out = tmp_path / "stride.gif"
    export_animation(plotter, director, out, fps=10, step_stride=3)
    assert out.exists()
    # Director should be back where the user was.
    assert director.step_index == 2


def test_export_restores_step_on_error(animation_setup, tmp_path: Path):
    """If the writer fails mid-stream, the user's step is still restored."""
    plotter, director = animation_setup
    director.set_step(3)
    bad_path = tmp_path / "noway.gif"

    # Patch screenshot to blow up on the second frame.
    calls = {"n": 0}
    real_screenshot = plotter.screenshot

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated capture failure")
        return real_screenshot(*args, **kwargs)

    plotter.screenshot = boom
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            export_animation(plotter, director, bad_path, fps=10)
    finally:
        plotter.screenshot = real_screenshot
    assert director.step_index == 3


# =====================================================================
# MP4 export — only when imageio-ffmpeg is available
# =====================================================================

@pytest.mark.skipif(
    not _has_ffmpeg(),
    reason="imageio-ffmpeg not installed (apegmsh[animation] extra)",
)
def test_mp4_export_writes_file(animation_setup, tmp_path: Path):
    plotter, director = animation_setup
    out = tmp_path / "anim.mp4"
    export_animation(plotter, director, out, fps=24)
    assert out.exists()
    # MP4 starts with the ftyp box (offset 4–8 = "ftyp").
    head = out.read_bytes()[:12]
    assert b"ftyp" in head
    assert out.stat().st_size > 1000


def test_mp4_without_ffmpeg_raises(animation_setup, tmp_path: Path, monkeypatch):
    """When the optional wheel is missing, MP4 raises an actionable error."""
    import sys
    # Force the import to fail by hiding the module in sys.modules.
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    plotter, director = animation_setup
    with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
        export_animation(plotter, director, tmp_path / "out.mp4")
