"""``add_scalar_bar``'s title-path exception handling (ADR 0081 Part 3).

A horizontal legend's title is drawn as a separate VTK actor
(``PyVistaBackend._add_bar_title``) once the scalar bar itself is
registered. That title path used to sit inside a bare ``except
Exception: pass`` that also covered the bar's own bookkeeping — on
VTK 9.7 an ``AttributeError`` from the removed ``vtkRenderer.
AddActor2D`` was swallowed there, and because the bookkeeping line sat
*after* the title path, the bar was never recorded either — an
un-removable scalar bar (#1122). This narrows the catch to what the
title path can legitimately raise and moves the bookkeeping ahead of
it, so the bar is always registered and removable even when its title
fails.
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from apeGmsh.viewers.backends import PyVistaQtBackend  # noqa: E402
from apeGmsh.viewers.backends.pyvista_qt import ScalarBarTitleWarning  # noqa: E402
from apeGmsh.viewers.scene_ir import (  # noqa: E402
    CellBlocks,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarBarSpec,
)


@pytest.fixture
def backend():
    try:
        plotter = pv.Plotter(off_screen=True)
    except Exception:  # pragma: no cover - depends on GL availability
        pytest.skip("no offscreen render context available")
    yield PyVistaQtBackend(plotter)
    plotter.close()


def _mesh_handle(backend):
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    layer = MeshLayer(
        layer_id="m",
        points=PointSet(pts),
        cells=CellBlocks({"tetra": np.array([[0, 1, 2, 3]])}),
    )
    return backend.add_layer(layer)


def _spec(key="k", *, title_anchor=(0.5, 0.9)) -> ScalarBarSpec:
    return ScalarBarSpec(
        key=key, title="stress_xx", lut=LutSpec(vmin=0.0, vmax=1.0),
        vertical=False, title_anchor=title_anchor,
    )


def test_title_failure_still_registers_a_removable_bar(backend, monkeypatch):
    handle = _mesh_handle(backend)

    def _boom(self, spec, bar):
        raise AttributeError("AddActor2D")

    monkeypatch.setattr(PyVistaQtBackend, "_add_bar_title", _boom)

    with pytest.warns(ScalarBarTitleWarning):
        backend.add_scalar_bar(handle, _spec())

    assert "k" in backend._scalar_bars
    backend.remove_scalar_bar("k")
    assert "k" not in backend._scalar_bars


def test_title_failure_warns_exactly_once(backend, monkeypatch):
    handle = _mesh_handle(backend)
    monkeypatch.setattr(
        PyVistaQtBackend, "_add_bar_title",
        lambda self, spec, bar: (_ for _ in ()).throw(AttributeError("boom")),
    )

    with pytest.warns(ScalarBarTitleWarning) as record:
        backend.add_scalar_bar(handle, _spec())

    assert len(record) == 1


def test_no_title_anchor_never_calls_title_path(backend, monkeypatch):
    handle = _mesh_handle(backend)
    calls = []
    monkeypatch.setattr(
        PyVistaQtBackend, "_add_bar_title",
        lambda self, spec, bar: calls.append(1),
    )
    backend.add_scalar_bar(handle, _spec(title_anchor=None))
    assert calls == []
    assert "k" in backend._scalar_bars


def test_successful_title_path_is_unaffected(backend, recwarn):
    handle = _mesh_handle(backend)
    backend.add_scalar_bar(handle, _spec())
    assert "k" in backend._scalar_bars
    _title, _bar, title_actor = backend._scalar_bars["k"]
    assert title_actor is not None
    assert not any(
        issubclass(w.category, ScalarBarTitleWarning) for w in recwarn.list
    )
