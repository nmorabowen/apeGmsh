"""TrameBackend + WebViewer (ADR 0042, R-C slice 1).

Verifies the web backend shares the desktop backend's ``scene_ir`` →
pyvista translation verbatim (parity), differs only in picking support,
and that the ``WebViewer`` shell builds + binds + scrubs headlessly. The
actual trame/browser display (``WebViewer.show``) needs a notebook and is
verified by eyeball — not here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from apeGmsh.viewers.backends import (
    PyVistaQtBackend,
    TrameBackend,
    PyVistaBackend,
)
from apeGmsh.viewers.scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

from tests.conftest import _open_model_from_h5


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def cube_results(g, tmp_path: Path):
    """1x1x1 tet cube + single-stage Results (zero displacement)."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    g.model.geometry.add_box(0, 0, 0, 1.0, 1.0, 1.0, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    res_path = tmp_path / "run.h5"
    with NativeWriter(res_path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="static", kind="static",
            time=np.array([0.0], dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={
                "displacement_x": np.zeros((1, node_ids.size)),
                "displacement_y": np.zeros((1, node_ids.size)),
                "displacement_z": np.zeros((1, node_ids.size)),
            },
        )
        w.end_stage()
    return Results.from_native(res_path, model=_open_model_from_h5(res_path))


def _quad_layer() -> MeshLayer:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    return MeshLayer(
        layer_id="q",
        points=PointSet(verts),
        cells=CellBlocks({"quad": np.array([[0, 1, 2, 3]], dtype=np.int64)}),
        color=ColorSpec(mode="solid", solid_rgb="#3CB371"),
    )


# ---------------------------------------------------------------------
# Backend identity / capability
# ---------------------------------------------------------------------

def test_trame_backend_is_a_pyvista_backend():
    b = TrameBackend(pv.Plotter(off_screen=True))
    assert isinstance(b, PyVistaBackend)


def test_trame_backend_picking_off_qt_on():
    trame = TrameBackend(pv.Plotter(off_screen=True))
    qt = PyVistaQtBackend(pv.Plotter(off_screen=True))
    assert trame.supports_picking() is False
    assert qt.supports_picking() is True


def test_trame_default_plotter_constructs():
    # No plotter supplied → backend makes its own pyvista.Plotter.
    b = TrameBackend()
    assert b.plotter is not None
    try:
        b.plotter.close()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Render parity — identical grids from the shared scene_ir translation
# ---------------------------------------------------------------------

def test_mesh_layer_grid_parity_qt_vs_trame():
    """Both backends build the same pyvista grid from one MeshLayer —
    they inherit ``add_layer`` from ``PyVistaBackend`` unchanged."""
    layer = _quad_layer()
    qt = PyVistaQtBackend(pv.Plotter(off_screen=True))
    trame = TrameBackend(pv.Plotter(off_screen=True))
    hq = qt.add_layer(layer)
    ht = trame.add_layer(layer)
    np.testing.assert_array_equal(hq.dataset.points, ht.dataset.points)
    assert hq.dataset.n_cells == ht.dataset.n_cells == 1
    np.testing.assert_array_equal(hq.dataset.celltypes, ht.dataset.celltypes)


# ---------------------------------------------------------------------
# WebViewer shell — headless construction / binding / scrub
# ---------------------------------------------------------------------

def test_web_viewer_builds_and_binds(cube_results):
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    assert isinstance(wv.backend, TrameBackend)
    assert wv.director.is_bound if hasattr(wv.director, "is_bound") else True
    # Substrate actor is on the plotter.
    assert "substrate" in wv.plotter.actors
    # A stage is active → at least one step.
    assert wv.n_steps >= 1
    # Programmatic scrub doesn't raise.
    wv.set_step(0)


def test_web_viewer_step_clamps(cube_results):
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    # Out-of-range step is clamped by the director, not an error.
    wv.set_step(999)


def test_show_web_returns_viewer_without_display(cube_results, monkeypatch):
    """show_web(show=False) builds the viewer but does not display."""
    from apeGmsh.viewers import web_viewer

    wv = web_viewer.show_web(cube_results, show=False)
    assert isinstance(wv, web_viewer.WebViewer)


def test_show_web_displays_widget_when_show_true(cube_results, monkeypatch):
    """show_web(show=True) must hand the widget to IPython.display — the
    WebViewer it returns has no rich repr, so without an explicit display
    nothing reaches the notebook and nothing renders (the "didn't pop up"
    bug). Returns the viewer regardless so callers can scrub / add diagrams.
    """
    import apeGmsh.viewers.web_viewer as web_viewer
    import IPython.display as ipd

    sentinel = object()
    monkeypatch.setattr(web_viewer.WebViewer, "show", lambda self, **kw: sentinel)
    displayed: list = []
    monkeypatch.setattr(ipd, "display", lambda obj: displayed.append(obj))

    wv = web_viewer.show_web(cube_results, show=True)
    assert isinstance(wv, web_viewer.WebViewer)
    assert displayed == [sentinel]


def test_show_degrades_when_view_is_not_a_widget(cube_results):
    """If pyvista returns a static-image fallback (not an ipywidget) — e.g.
    nest_asyncio2 missing so the trame server can't launch — show() must
    return the view as-is with a warning, NOT crash composing a VBox
    (regression: TraitError on PIL.Image in VBox.children)."""
    pytest.importorskip("ipywidgets")
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    sentinel = object()                      # stand-in for the static fallback
    wv._plotter.show = lambda **kwargs: sentinel
    with pytest.warns(RuntimeWarning, match="nest_asyncio2"):
        out = wv.show(controls=True)
    assert out is sentinel


def test_show_composes_controls_when_view_is_widget(cube_results):
    """When the trame view IS an ipywidget, controls stack above it."""
    W = pytest.importorskip("ipywidgets")
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    fake_view = W.HTML("trame view")
    wv._plotter.show = lambda **kwargs: fake_view
    out = wv.show(controls=True)
    assert isinstance(out, W.VBox)
    assert fake_view in out.children


# ---------------------------------------------------------------------
# render_mode toggle (R-C) — friendly name → pyvista jupyter_backend.
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "render_mode, expected_backend",
    [("client", "client"), ("server", "server"), ("hybrid", "trame")],
)
def test_render_mode_maps_to_jupyter_backend(
    cube_results, render_mode, expected_backend
):
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    seen = {}
    wv._plotter.show = lambda **kwargs: seen.update(kwargs)
    wv.show(controls=False, render_mode=render_mode)
    assert seen["jupyter_backend"] == expected_backend


def test_render_mode_defaults_to_client(cube_results):
    """The fast path is the default (fixes the slow server-render UX)."""
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    seen = {}
    wv._plotter.show = lambda **kwargs: seen.update(kwargs)
    wv.show(controls=False)
    assert seen["jupyter_backend"] == "client"


def test_jupyter_backend_overrides_render_mode(cube_results):
    """Raw jupyter_backend escape hatch wins over render_mode."""
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    seen = {}
    wv._plotter.show = lambda **kwargs: seen.update(kwargs)
    wv.show(controls=False, render_mode="client", jupyter_backend="html")
    assert seen["jupyter_backend"] == "html"


def test_unknown_render_mode_raises(cube_results):
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    wv._plotter.show = lambda **kwargs: None
    with pytest.raises(ValueError, match="Unknown render_mode"):
        wv.show(controls=False, render_mode="turbo")


def test_show_web_threads_render_mode(cube_results, monkeypatch):
    """show_web(render_mode=) reaches WebViewer.show."""
    import apeGmsh.viewers.web_viewer as web_viewer

    seen = {}
    monkeypatch.setattr(
        web_viewer.WebViewer, "show",
        lambda self, **kw: seen.update(kw),
    )
    web_viewer.show_web(cube_results, show=True, render_mode="server")
    assert seen["render_mode"] == "server"


# ---------------------------------------------------------------------
# ipywidgets controls (R-C slice 2) — wiring verified headlessly via
# traitlets' synchronous .observe; the visual push is eyeballed.
# ---------------------------------------------------------------------

@pytest.fixture
def cube_results_2steps(g, tmp_path: Path):
    """Cube + a 2-step stage so the controls panel gets a slider."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    g.model.geometry.add_box(0, 0, 0, 1.0, 1.0, 1.0, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    res_path = tmp_path / "run2.h5"
    with NativeWriter(res_path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="static", kind="static",
            time=np.array([0.0, 1.0], dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={
                "displacement_x": np.zeros((2, node_ids.size)),
                "displacement_y": np.zeros((2, node_ids.size)),
                "displacement_z": np.zeros((2, node_ids.size)),
            },
        )
        w.end_stage()
    return Results.from_native(res_path, model=_open_model_from_h5(res_path))


class _StubDiagram:
    def __init__(self, label: str) -> None:
        self._label = label
        self.is_visible = True

    def display_label(self) -> str:
        return self._label


def _slider(box):
    W = pytest.importorskip("ipywidgets")
    return next((c for c in box.children if isinstance(c, W.IntSlider)), None)


def _checkboxes(box):
    W = pytest.importorskip("ipywidgets")
    return [c for c in box.children if isinstance(c, W.Checkbox)]


def test_controls_slider_present_and_ranged(cube_results_2steps):
    pytest.importorskip("ipywidgets")
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results_2steps, plotter=pv.Plotter(off_screen=True))
    box = wv.controls()
    slider = _slider(box)
    assert slider is not None
    assert slider.min == 0
    assert slider.max == wv.n_steps - 1 == 1


def test_controls_no_slider_for_single_step(cube_results):
    pytest.importorskip("ipywidgets")
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    assert _slider(wv.controls()) is None  # 1 step → no scrubber


def test_controls_slider_drives_set_step(cube_results_2steps):
    pytest.importorskip("ipywidgets")
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results_2steps, plotter=pv.Plotter(off_screen=True))
    calls: list[int] = []
    wv.set_step = lambda i: calls.append(int(i))  # spy (lambda is late-bound)
    box = wv.controls()
    _slider(box).value = 1  # traitlets fires .observe synchronously
    assert calls == [1]


def test_controls_checkbox_drives_set_layer_visible(cube_results, monkeypatch):
    pytest.importorskip("ipywidgets")
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    diag = _StubDiagram("contour — disp")
    monkeypatch.setattr(wv, "layer_diagrams", lambda: [diag])
    toggled: list[tuple] = []
    wv.set_layer_visible = lambda d, v: toggled.append((d, v))
    box = wv.controls()
    cbs = _checkboxes(box)
    assert len(cbs) == 1
    assert cbs[0].description == "contour — disp"
    cbs[0].value = False
    assert toggled == [(diag, False)]
