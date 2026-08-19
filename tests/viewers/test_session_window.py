"""ADR 0098 S2 — SessionWindow: shell composition + QSettings scope.

No-VTK half (default lane): the persistence-identity guards of plan
decision 8 — the session window keeps its OWN ``QSettings`` scope
(``'ResultsSession'``, schema numbering restarted at 1) and the old
window's scope/schema stay untouched; plus the ``show_session``
skip-env discipline. Constructing the real shell allocates a
``QtInteractor`` GL context, so the full composition smoke is
qt-marked (per-file under xvfb in CI), following the established
no-VTK strategy of ``test_results_window_architecture.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.session import SessionResultsWindow, show_session
from apeGmsh.viewers.ui._results_window import ResultsWindow

from tests.conftest import _open_model_from_h5


# =====================================================================
# Decision 8 — separate QSettings scope until S6
# =====================================================================


def test_session_window_has_its_own_settings_scope():
    s = SessionResultsWindow._layout_settings()
    assert s.organizationName() == "apeGmsh"
    assert s.applicationName() == "ResultsSession"


def test_session_window_schema_numbering_restarts_at_1():
    """A fresh scope numbers from 1 — bumps follow THIS window's dock
    changes, not the old window's v7 history."""
    assert SessionResultsWindow._LAYOUT_SCHEMA_VERSION == 1


def test_old_window_scope_and_schema_untouched():
    """The live old window keeps schema v7 in the 'ResultsViewer'
    scope — S2 must not perturb either side of the isolation."""
    assert ResultsWindow._LAYOUT_SCHEMA_VERSION == 7
    s = ResultsWindow._layout_settings()
    assert s.applicationName() == "ResultsViewer"


def test_shared_dock_objectnames_by_inheritance():
    """The subclass reuses the shell's dock objectNames verbatim —
    legal across scopes (criterion 11 constrains only the four
    RETIRED names, which the existing sweep checks over all of
    viewers/**, the new session modules included)."""
    assert SessionResultsWindow.DOCK_OUTLINE == ResultsWindow.DOCK_OUTLINE
    assert SessionResultsWindow.DOCK_INSPECTOR == ResultsWindow.DOCK_INSPECTOR
    assert SessionResultsWindow.DOCK_SCRUBBER == ResultsWindow.DOCK_SCRUBBER


# =====================================================================
# show discipline
# =====================================================================


def test_show_session_honors_skip_viewer(monkeypatch, capsys):
    from apeGmsh.results.session import ResultsSession

    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    out = show_session(ResultsSession())
    assert out is None
    assert "[skip viewer]" in capsys.readouterr().out


def test_session_show_entry_is_lazy(monkeypatch, capsys):
    """``ResultsSession.show`` reaches the Qt client lazily — the S0
    purity guard keeps the session package Qt-free, and the skip-env
    path proves the wiring without a window."""
    from apeGmsh.results.session import ResultsSession

    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    assert ResultsSession().show() is None
    assert "[skip viewer]" in capsys.readouterr().out


def test_show_refuses_an_unbound_session():
    from apeGmsh.results.session import ResultsSession
    from apeGmsh.viewers.session import SessionWindow

    with pytest.raises(RuntimeError, match="no Results bound"):
        SessionWindow(ResultsSession())


# =====================================================================
# qt lane — the composed window, real shell, discrete Add loop
# =====================================================================


@pytest.fixture
def small_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    disp = np.zeros((2, node_ids.size))
    disp[1] = node_ids

    path = tmp_path / "session_window.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", stage_id="grav",
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.mark.qt
def test_display_dock_is_not_a_blank_panel(small_results):
    """ADR 0087 INV-2 — honest empty state.

    The shell builds the "Display" dock unconditionally and the View
    menu can summon it. S6a shipped a window that never mounted
    anything into it, so opening it showed a blank panel; INV-2 says a
    panel with nothing to show carries exactly one muted hint line.
    Asserts the host has a widget AND that it is disabled (muted) —
    mounting an enabled empty container would satisfy a naive
    "is not None" and still violate the invariant.
    """
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from apeGmsh.viewers.session import SessionWindow

    window = SessionWindow(small_results.session(), title="inv2-smoke")
    try:
        qapp.processEvents()
        host = window.shell._session_host  # noqa: SLF001
        mounted = host.layout().itemAt(0)

        assert mounted is not None, (
            "Display dock is empty — View → Display opens a blank "
            "panel (ADR 0087 INV-2)."
        )
        label = mounted.widget()
        assert label.text().strip(), "the hint line is empty"
        assert not label.isEnabled(), "the hint must be muted, not live"
    finally:
        window.close()
        qapp.processEvents()


@pytest.mark.qt
def test_session_window_composes_shell_and_runs_the_add_loop(
    small_results,
):
    """The acceptance shape with no Python session writes: boot the
    composed window (shell + outline + inspector + the pane host),
    click Add on the contour row, and watch that pane reconcile. The
    human loop's automated understudy.

    Since Amendment 1 the pane owns its OWN ``QtInteractor`` and the
    shell builds none (A1.1), so the window's one GL context belongs to
    the pane rather than being adopted from the shell.
    """
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from apeGmsh.viewers.session import SessionWindow

    session = small_results.session()
    view = session.panes[0]
    window = SessionWindow(session, title="s2-smoke")
    try:
        window.show(blocking=False)
        qapp.processEvents()  # boot flush (QTimer-deferred)

        frame = window.host.frame(view.id)
        realized = frame.pane.reconciler.realized
        assert realized is not None
        assert {layer.key for layer in realized.layers} == {
            f"{view.id}:substrate", f"{view.id}:outlines",
        }
        # The pane owns its interactor; the shell has none of its own.
        assert frame.pane.surface is not None
        assert frame.pane.backend.plotter is frame.pane.surface
        assert window.shell.plotter is frame.pane.surface

        # The discrete Add loop, driven through the real widgets.
        page = window.inspector_page(view.id)
        page._rows["contour"]._button.click()  # noqa: SLF001
        qapp.processEvents()  # coalesced flush

        assert view.contour is not None
        realized = frame.pane.reconciler.realized
        assert {layer.key for layer in realized.layers} == {
            f"{view.id}:contour", f"{view.id}:outlines",
        }
        # Outline lists the pane; inspector page is mounted.
        assert window.outline.selected_pane_id() == view.id
    finally:
        window.close()
        qapp.processEvents()


@pytest.mark.qt
def test_session_window_persists_into_its_own_scope(small_results):
    """Closing the session window writes layout state under the
    'ResultsSession' scope with THIS window's schema version — never
    into the old window's 'ResultsViewer' scope (decision 8)."""
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from apeGmsh.viewers.session import SessionWindow

    session = small_results.session()
    window = SessionWindow(session, title="s2-settings-smoke")
    try:
        window.show(blocking=False)
        qapp.processEvents()
    finally:
        window.close()
        qapp.processEvents()

    s = SessionResultsWindow._layout_settings()
    stored = s.value("layout/schema_version")
    assert int(stored) == SessionResultsWindow._LAYOUT_SCHEMA_VERSION
