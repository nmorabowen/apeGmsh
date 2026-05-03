"""AddDiagramDialog — Topology row exposes ContourStyle.topology.

The Contour kind owns a per-instance ``ContourStyle.topology`` field
that selects between nodal-scalar (point data) and element-constant
Gauss (cell data) rendering. The dialog now surfaces that choice via a
sub-combo that's only visible when Contour is the selected kind.

These tests build a synthetic ``Results`` with both nodal and gauss
data so the union / per-topology component lists differ visibly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _all_element_ids(fem) -> np.ndarray:
    chunks = []
    for group in fem.elements:
        chunks.append(np.asarray(group.ids, dtype=np.int64))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)


@pytest.fixture
def director_with_nodes_and_gauss(g, tmp_path: Path):
    """Native HDF5 with one stage carrying both node + gauss data."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = _all_element_ids(fem)
    n_steps = 2
    disp_z = np.zeros((n_steps, node_ids.size), dtype=np.float64)
    sxx = np.zeros((n_steps, elem_ids.size, 1), dtype=np.float64)

    path = tmp_path / "nodes_and_gauss.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_z": disp_z},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids,
            natural_coords=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            components={"stress_xx": sxx},
        )
        w.end_stage()

    from apeGmsh.viewers.diagrams._director import ResultsDirector
    return ResultsDirector(Results.from_native(path))


# =====================================================================
# Helpers
# =====================================================================


def _set_kind(dlg: Any, kind_id: str) -> None:
    for i in range(dlg._kind_combo.count()):
        entry = dlg._kind_combo.itemData(i)
        if entry.kind_id == kind_id:
            dlg._kind_combo.setCurrentIndex(i)
            return
    raise AssertionError(f"kind {kind_id} not found in combo")


def _set_topology(dlg: Any, topology: str) -> None:
    idx = dlg._topology_combo.findData(topology)
    assert idx >= 0, f"topology {topology!r} not in combo"
    dlg._topology_combo.setCurrentIndex(idx)


def _component_items(dlg: Any) -> list[str]:
    return [
        dlg._component_combo.itemText(i)
        for i in range(dlg._component_combo.count())
    ]


# =====================================================================
# Visibility — Topology row only matters for Contour
# =====================================================================


def test_topology_row_is_visible_for_contour(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    assert not dlg._topology_combo.isHidden()
    assert not dlg._topology_label.isHidden()


def test_topology_row_is_hidden_for_other_kinds(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    for kind in (
        "deformed_shape", "line_force", "fiber_section",
        "layer_stack", "vector_glyph", "gauss_marker", "spring_force",
    ):
        _set_kind(dlg, kind)
        assert dlg._topology_combo.isHidden(), (
            f"Topology row should be hidden for {kind}"
        )
        assert dlg._topology_label.isHidden()


# =====================================================================
# Component listing per topology
# =====================================================================


def test_topology_combo_only_offers_nodes_and_gauss(
    qapp, director_with_nodes_and_gauss,
):
    """The dialog dropped the legacy ``"auto"`` option — users now pick
    explicitly between nodes and gauss."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    options = [
        dlg._topology_combo.itemData(i)
        for i in range(dlg._topology_combo.count())
    ]
    assert options == ["nodes", "gauss"]


def test_nodes_topology_lists_nodes_only(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    _set_topology(dlg, "nodes")
    items = _component_items(dlg)
    assert "displacement_z" in items
    assert "stress_xx" not in items


def test_gauss_topology_lists_gauss_only(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    _set_topology(dlg, "gauss")
    items = _component_items(dlg)
    assert "stress_xx" in items
    assert "displacement_z" not in items


def test_switching_topology_repopulates_components(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")

    _set_topology(dlg, "nodes")
    items = _component_items(dlg)
    assert "displacement_z" in items
    assert "stress_xx" not in items

    _set_topology(dlg, "gauss")
    items = _component_items(dlg)
    assert "stress_xx" in items
    assert "displacement_z" not in items


# =====================================================================
# Averaging row — visible only for contour + gauss
# =====================================================================


def test_averaging_row_visible_for_gauss_topology(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    _set_topology(dlg, "gauss")
    assert not dlg._averaging_combo.isHidden()
    assert not dlg._averaging_label.isHidden()


def test_averaging_row_hidden_for_nodes_topology(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    _set_topology(dlg, "nodes")
    assert dlg._averaging_combo.isHidden()
    assert dlg._averaging_label.isHidden()


# =====================================================================
# Default still prefers displacement_z when reachable
# =====================================================================


def test_default_component_under_nodes_prefers_displacement_z(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    _set_topology(dlg, "nodes")
    assert dlg._component_combo.currentText() == "displacement_z"


def test_default_component_under_gauss_takes_first_available(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    _set_topology(dlg, "gauss")
    # stress_xx is the only gauss component in the fixture.
    assert dlg._component_combo.currentText() == "stress_xx"


# =====================================================================
# Spec construction — chosen topology threads into ContourStyle
# =====================================================================


def _stub_accepted(dlg) -> None:
    """Make ``run()`` skip exec_() and behave as if OK was clicked."""
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    accepted = QtWidgets.QDialog.Accepted
    dlg._dlg.exec_ = lambda: accepted


def test_run_with_gauss_topology_builds_gauss_style(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.diagrams._styles import ContourStyle
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    _set_topology(dlg, "gauss")
    # stress_xx is set as the default by the populate logic.
    assert dlg._component_combo.currentText() == "stress_xx"
    _stub_accepted(dlg)

    captured: list = []
    director = director_with_nodes_and_gauss
    real_add = director.registry.add
    def _capture(diagram):
        captured.append(diagram)
        return real_add(diagram)
    director.registry.add = _capture
    try:
        ok = dlg.run()
    finally:
        director.registry.add = real_add

    assert ok is True
    assert len(captured) == 1
    diagram = captured[0]
    assert isinstance(diagram.spec.style, ContourStyle)
    assert diagram.spec.style.topology == "gauss"
    assert diagram.spec.selector.component == "stress_xx"


def test_run_with_nodes_topology_builds_nodes_style(
    qapp, director_with_nodes_and_gauss,
):
    from apeGmsh.viewers.diagrams._styles import ContourStyle
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)
    _set_kind(dlg, "contour")
    _set_topology(dlg, "nodes")
    _stub_accepted(dlg)

    captured: list = []
    director = director_with_nodes_and_gauss
    real_add = director.registry.add
    director.registry.add = lambda d: (captured.append(d), real_add(d))[1]
    try:
        dlg.run()
    finally:
        director.registry.add = real_add

    assert len(captured) == 1
    assert captured[0].spec.style.topology == "nodes"


def test_non_contour_kind_unaffected_by_topology_combo(
    qapp, director_with_nodes_and_gauss,
):
    """Switching kinds away from contour and back should leave the
    component listing for non-contour kinds unaffected by whatever
    topology the user previously selected for contour."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director_with_nodes_and_gauss, parent=None)

    _set_kind(dlg, "contour")
    _set_topology(dlg, "gauss")

    _set_kind(dlg, "deformed_shape")
    items = _component_items(dlg)
    # deformed_shape lives on `nodes` regardless of any contour state.
    assert "displacement_z" in items
    assert "stress_xx" not in items
