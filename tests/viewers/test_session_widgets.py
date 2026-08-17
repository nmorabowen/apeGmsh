"""ADR 0098 S2 — the Qt projection widgets, offscreen (default lane).

THE S2 oracle: **widget event → session diff → repaint**, with a
``RecordingBackend`` injected through the pane's backend factory
(plan decision 7) so no GL context is allocated and the whole loop
runs in the default CI lane (``QT_QPA_PLATFORM=offscreen``). The
test bodies click buttons and drive combos — the session writes and
the repaint are OBSERVED, never performed directly.

One qt-marked real-window smoke rides at the bottom (per-file, under
xvfb in CI): the default backend factory building a live
``QtInteractor``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.results import Results
from apeGmsh.results.session import Contour, Deform
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.session import (
    MeshInspectorPage,
    MeshPane,
    PanePlaceholderPage,
    SessionOutline,
)

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def session_results(g, tmp_path: Path):
    """One static stage: nodal displacement_z + 1-GP gauss stress_xx."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = np.concatenate(
        [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    )
    n_steps = 3
    disp = np.zeros((n_steps, node_ids.size))
    for t in range(n_steps):
        disp[t] = node_ids + t * 1000.0
    sxx = np.zeros((n_steps, elem_ids.size, 1))
    for t in range(n_steps):
        sxx[t, :, 0] = elem_ids * 10.0 + t

    path = tmp_path / "session_widgets.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=np.zeros((1, 3)),
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def rig(qapp, session_results):
    """(session, view, backend, pane, page): the offscreen projection
    loop with an injected RecordingBackend and an IMMEDIATE defer —
    every widget gesture flushes synchronously, so assertions read the
    repainted backend directly."""
    session = session_results.session()
    view = session.panes[0]
    backend = RecordingBackend()
    pane = MeshPane(
        session,
        backend_factory=lambda parent: (None, backend),
        defer_fn=lambda fn: fn(),
    )
    page = MeshInspectorPage(session, view)
    yield session, view, backend, pane, page
    page.dispose()
    pane.dispose()


def _stable_keys(pane) -> set:
    realized = pane.reconciler.realized
    return {layer.key for layer in realized.layers} if realized else set()


def _boot_keys(view) -> set:
    """The §3 boot picture: grey mesh, mesh + outlines on."""
    return {f"{view.id}:substrate", f"{view.id}:outlines"}


def _contour_keys(view) -> set:
    """Contour occupying the one surface, outlines style still on."""
    return {f"{view.id}:contour", f"{view.id}:outlines"}


def _row(page, category):
    return page._rows[category]  # noqa: SLF001 — test introspection


# =====================================================================
# THE S2 oracle — widget event → session diff → repaint
# =====================================================================


def test_add_contour_click_paints_the_contour(rig):
    """The §9 Add action end-to-end: one button click fills the slot
    (the same ``view.contour = Contour(...)`` a script would run) and
    the reconciled backend replaces the substrate with the contour
    (INV-MESH-2) plus one scalar bar (§5). No session write happens in
    this test body — the widget performs it."""
    session, view, backend, pane, page = rig
    assert _stable_keys(pane) == _boot_keys(view)

    row = _row(page, "contour")
    row._button.click()  # "Add"

    assert view.contour is not None  # widget event → session diff
    assert _stable_keys(pane) == _contour_keys(view)  # → repaint
    assert len(backend.scalar_bars) == 1
    assert row._button.text() == "Clear"


def test_quantity_combo_changes_the_record_and_repaints(rig):
    session, view, backend, pane, page = rig
    row = _row(page, "contour")
    row._button.click()
    combo = page.widget.findChild(
        __import__("qtpy").QtWidgets.QComboBox,
        "session_slot_contour_quantity",
    )
    combo.setCurrentText("stress_xx")
    assert view.contour == Contour("stress_xx")
    combo.setCurrentText("displacement_z")
    assert view.contour == Contour("displacement_z")
    assert _stable_keys(pane) == _contour_keys(view)


def test_averaging_combo_writes_display_state(rig):
    session, view, backend, pane, page = rig
    _row(page, "contour")._button.click()
    from qtpy import QtWidgets

    mode = page.widget.findChild(
        QtWidgets.QComboBox, "session_slot_contour_mode",
    )
    mode.setCurrentText("unaveraged")
    assert view.contour.averaging == "unaveraged"


def test_clear_click_restores_the_substrate(rig):
    session, view, backend, pane, page = rig
    row = _row(page, "contour")
    row._button.click()  # Add
    row._button.click()  # Clear
    assert view.contour is None
    assert _stable_keys(pane) == _boot_keys(view)
    assert backend.scalar_bars == {}
    assert row._button.text() == "Add"


def test_deform_checkbox_poses_without_a_legend(rig):
    """Deform is a pose, never a picture (§4): toggling it warps the
    substrate and emits ZERO scalar bars."""
    session, view, backend, pane, page = rig
    substrate_id = pane.reconciler.realized.layers[0].layer_id
    flat = np.asarray(backend.layers[substrate_id].points.coords).copy()

    page._deform_check.setChecked(True)  # noqa: SLF001

    assert view.deform == Deform("displacement")
    assert backend.scalar_bars == {}
    substrate_id = pane.reconciler.realized.layers[0].layer_id
    warped = np.asarray(backend.layers[substrate_id].points.coords)
    assert not np.allclose(flat, warped)


def test_deform_scale_spin_auto_at_zero(rig):
    session, view, backend, pane, page = rig
    page._deform_check.setChecked(True)  # noqa: SLF001
    page._deform_scale.setValue(2.5)  # noqa: SLF001
    assert view.deform == Deform("displacement", scale=2.5)
    page._deform_scale.setValue(0.0)  # noqa: SLF001
    assert view.deform == Deform("displacement", scale=None)


def test_reactions_row_needs_no_editor(rig):
    from apeGmsh.results.session import Reactions

    session, view, backend, pane, page = rig
    row = _row(page, "reactions")
    row._button.click()
    assert view.reactions == Reactions()
    row._button.click()
    assert view.reactions is None


def test_unrecorded_category_add_is_disabled(rig):
    """A category the stage records nothing for shows a DISABLED Add —
    the §4 inapplicable-is-empty/disabled law at the picker level, so
    the discrete loop cannot ask for a picture that refuses."""
    session, view, backend, pane, page = rig
    # The cube stage records no beam forces and no load patterns.
    assert not _row(page, "line")._button.isEnabled()
    assert not _row(page, "loads")._button.isEnabled()
    # Recorded categories stay actionable.
    assert _row(page, "contour")._button.isEnabled()
    assert _row(page, "gauss")._button.isEnabled()


def test_occupied_slot_that_drew_nothing_reads_nothing_to_draw(rig):
    """ADR §4: a slot may legitimately emit ZERO layers — the
    inspector shows that as informational text, never an error."""
    from apeGmsh.viewers.session import RealizedPane

    session, view, backend, pane, page = rig
    view.contour = Contour("stress_xx")
    empty = RealizedPane(
        pane_id=view.id, layers=(), scalar_bars=(),
        diagrams=(), legend_controller=None,
    )
    page.set_realized(empty)
    assert _row(page, "contour")._status.text() == "(nothing to draw)"
    page.set_realized(pane.reconciler.realized)
    assert _row(page, "contour")._status.text() == ""


def test_python_edit_reflects_into_the_widgets(rig):
    """Two authors, one session (§2): a script-side write shows up in
    the page exactly like a click would."""
    from qtpy import QtWidgets

    session, view, backend, pane, page = rig
    view.contour = Contour("stress_xx", averaging="unaveraged")
    combo = page.widget.findChild(
        QtWidgets.QComboBox, "session_slot_contour_quantity",
    )
    mode = page.widget.findChild(
        QtWidgets.QComboBox, "session_slot_contour_mode",
    )
    assert combo.currentText() == "stress_xx"
    assert mode.currentText() == "unaveraged"
    assert _row(page, "contour")._button.text() == "Clear"


def test_widget_refresh_does_not_echo_writes(rig):
    """The projection sync must not write back: refreshing the page
    from a session tick fires combo signals, and an echoed write would
    tick again — an infinite loop or a phantom replace."""
    session, view, backend, pane, page = rig
    ticks: list = []
    session.subscribe(lambda: ticks.append(1))
    view.contour = Contour("stress_xx")
    assert len(ticks) == 1  # the write itself — and nothing echoed


# =====================================================================
# Outline (§9 — panes group only)
# =====================================================================


def test_outline_lists_panes_and_selection_fires(rig):
    session, view, backend, pane, page = rig
    selected: list = []
    outline = SessionOutline(
        session, on_pane_selected=selected.append,
    )
    try:
        group = outline.widget.topLevelItem(0)
        assert group.text(0) == "Views"
        # Pane rows + the two Add rows, which live in this group and
        # nowhere else (§9 / Amendment 1 A1.6: one spine).
        assert group.childCount() == 1 + 2
        outline.select_pane(view.id)
        assert selected == [view.id]

        plot = session.add_plot()
        assert group.childCount() == 2 + 2
        outline.select_pane(plot.id)
        assert selected == [view.id, plot.id]
    finally:
        outline.dispose()


def test_outline_survives_slot_ticks_without_rebuilding(rig):
    session, view, backend, pane, page = rig
    outline = SessionOutline(session, on_pane_selected=lambda _p: None)
    try:
        outline.select_pane(view.id)
        item_before = outline.widget.currentItem()
        view.contour = Contour("stress_xx")  # ticks; membership same
        assert outline.widget.currentItem() is item_before
        assert outline.selected_pane_id() == view.id
    finally:
        outline.dispose()


def test_placeholder_page_for_plot_panes():
    page = PanePlaceholderPage("Plot pane 'plot-2': lands at S4-2.")
    assert "S4-2" in page.widget.text()
    page.set_realized(None)  # inert, never raises
    page.dispose()


# =====================================================================
# qt lane — one real-window smoke per file (default factory, real GL)
# =====================================================================


@pytest.mark.qt
def test_mesh_pane_default_factory_real_interactor(qapp, session_results):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")

    session = session_results.session()
    view = session.panes[0]
    pane = MeshPane(session, defer_fn=lambda fn: fn())
    try:
        assert pane.surface is not None  # pane-owned QtInteractor
        view.contour = Contour("stress_xx")
        realized = pane.reconciler.realized
        assert realized is not None
        assert {layer.key for layer in realized.layers} == _contour_keys(view)
        # The live plotter holds exactly the realized actors.
        assert pane.backend.plotter is pane.surface
    finally:
        pane.dispose()
        pane.surface.close()
