"""R1.4 — first-run empty-state HUD + starter actions.

Three surfaces:

* :class:`EmptyStateHUD` — dumb widget behaviour (show/hide, button
  callbacks, dismiss, label-driven button visibility), offscreen Qt.
* ``diagrams._starter`` — the programmatic default-contour chain
  against a real-fixture director (no Qt, no plotter binding).
* Outline — the dead "Probes" placeholder group is gone.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import _stub_model_h5_path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_FIXTURE = Path("tests/fixtures/results/elasticFrame.mpco")


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def director():
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    from apeGmsh.results import Results
    from apeGmsh.viewers.diagrams._director import ResultsDirector
    return ResultsDirector(
        Results.from_mpco(_FIXTURE, model_h5=_stub_model_h5_path()),
    )


def _make_hud(qapp):
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    viewport = QtWidgets.QWidget()
    viewport.resize(800, 600)
    viewport.show()
    from apeGmsh.viewers.ui._empty_state_hud import EmptyStateHUD
    fired = {"primary": 0, "secondary": 0}
    hud = EmptyStateHUD(
        viewport,
        on_primary=lambda: fired.__setitem__(
            "primary", fired["primary"] + 1,
        ),
        on_secondary=lambda: fired.__setitem__(
            "secondary", fired["secondary"] + 1,
        ),
    )
    return hud, viewport, fired


# =====================================================================
# EmptyStateHUD widget behaviour
# =====================================================================


def test_hud_starts_hidden_and_shows_on_demand(qapp):
    hud, viewport, _ = _make_hud(qapp)
    assert not hud.widget.isVisible()
    hud.show()
    assert hud.widget.isVisible()
    hud.hide()
    assert not hud.widget.isVisible()
    viewport.deleteLater()


def test_primary_and_secondary_clicks_fire_callbacks(qapp):
    hud, viewport, fired = _make_hud(qapp)
    hud.set_primary("Add displacement contour")
    hud.set_secondary("Enable deform ×1")
    hud.show()
    hud._btn_primary.click()
    hud._btn_secondary.click()
    assert fired == {"primary": 1, "secondary": 1}
    viewport.deleteLater()


def test_dismiss_hides_the_card(qapp):
    hud, viewport, fired = _make_hud(qapp)
    hud.show()
    hud._btn_dismiss.click()
    assert not hud.widget.isVisible()
    # Dismiss is local — no action callbacks fire.
    assert fired == {"primary": 0, "secondary": 0}
    viewport.deleteLater()


def test_empty_label_hides_button(qapp):
    hud, viewport, _ = _make_hud(qapp)
    hud.set_primary("Add displacement contour")
    hud.set_secondary("")
    hud.show()
    assert hud._btn_primary.isVisible()
    assert hud._btn_primary.text() == "Add displacement contour"
    assert not hud._btn_secondary.isVisible()
    viewport.deleteLater()


# =====================================================================
# Starter — default contour chain (headless, real fixture)
# =====================================================================


def test_default_contour_component_prefers_displacement(director):
    from apeGmsh.viewers.diagrams._kind_catalog import _union_across_stages
    from apeGmsh.viewers.diagrams._starter import default_contour_component
    comp = default_contour_component(director)
    nodal = _union_across_stages(director, "nodes")
    assert comp in nodal
    if any(c.startswith("displacement_") for c in nodal):
        assert comp.startswith("displacement_")


def test_add_default_contour_registers_and_tags_membership(director):
    from apeGmsh.viewers.diagrams._starter import (
        add_default_contour,
        default_contour_component,
    )
    assert len(director.registry) == 0
    d = add_default_contour(director)
    assert len(director.registry) == 1
    assert d.kind == "contour"
    assert d.spec.selector.component == default_contour_component(director)
    comp = director.geometries.active.compositions.active
    assert comp is not None
    assert d in comp.layers


# =====================================================================
# Outline — Probes placeholder group is gone
# =====================================================================


def test_outline_has_no_probes_group(qapp):
    from apeGmsh.viewers.diagrams._geometries import GeometryManager
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    geometries = GeometryManager()

    class _Registry:
        def diagrams(self):
            return []

        def subscribe(self, _cb):
            return lambda: None

    class _Director:
        def __init__(self):
            self.geometries = geometries
            self.stage_id = None
            self.registry = _Registry()

        def stages(self):
            return []

        def subscribe_stage(self, _cb):
            return lambda: None

        def subscribe_diagrams(self, _cb):
            return lambda: None

    tree = OutlineTree(_Director())
    labels = [
        tree._tree.topLevelItem(i).text(0)
        for i in range(tree._tree.topLevelItemCount())
    ]
    assert labels == ["Stages", "Geometries", "Plots"]
    assert not hasattr(tree, "_group_probes")
