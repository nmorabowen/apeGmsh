"""GeometrySettingsPanel — the per-Geometry Threshold section.

ADR 0084 D1, slice 2. The panel already edited deformation, the stage
pin and display; the threshold joins them because it is per-geometry
state too. These tests drive the REAL Qt widgets against a real
:class:`ThresholdController` and a real :class:`GeometryManager`,
faking only the shell boundary (the value reader and the dispatcher),
and assert on the two things the section is for:

* what lands on the OWNER (the controller's per-geometry spec), and
* that every mutation ends in one ``STEP_CHANGED`` — the single path
  that recomputes the mask at the current cursor and repaints. The
  panel is inside the ADR 0056 AST guard, so it may not render or
  touch an artifact itself.

Same headless lane as the other UI panels (offscreen ``QApplication``,
no GL, no window).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.viewers.core.scope_controller import ScopeController
from apeGmsh.viewers.core.threshold_controller import (
    TOPOLOGY_GAUSS,
    TOPOLOGY_NODES,
    ThresholdController,
    ThresholdSettings,
)
from apeGmsh.viewers.diagrams._dispatch import (
    GEOMETRY_SCOPE_CHANGED,
    STEP_CHANGED,
)
from apeGmsh.viewers.diagrams._geometries import GeometryManager
from apeGmsh.viewers.ui._geometry_settings_panel import (
    GeometrySettingsPanel,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# Known per-component ranges so the "seed from the data range" rule is
# checkable: u spans [-4, 6], v spans [0, 1], s_xx spans [10, 20].
_RANGES = {"u": (-4.0, 6.0), "v": (0.0, 1.0), "s_xx": (10.0, 20.0)}


class _StubDirector:
    """Everything the panel reads off a director, and nothing else."""

    def __init__(self, *, scalars=None) -> None:
        self.geometries = GeometryManager()
        self.fired: list = []
        self.reads: list = []
        self.dispatcher = SimpleNamespace(
            fire=lambda kind, **kw: self.fired.append(kind),
        )
        self.thresholds = ThresholdController(
            read_values=self._read_values,
            on_failure=lambda *a, **k: None,
        )
        # The panel's Scope section reads this the way it reads
        # ``thresholds`` — the director always owns one.
        self.scopes = ScopeController()
        self._scalars = scalars

    def _read_values(self, component, step, *, stage_id=None,
                     topology=TOPOLOGY_NODES):
        self.reads.append((component, int(step), stage_id, topology))
        span = _RANGES.get(component)
        if span is None:
            return None
        return np.array([span[0], 0.5 * (span[0] + span[1]), span[1]])

    def local_step_for_active_stage(self) -> int:
        return 3

    def stages(self):
        return []

    def materialized_scene_for(self, geometry):
        """The Scope section's "Fit to model" source (never builds one).

        A 2x1x0 slab of reference points, so the fitted box is the
        hand-checkable ``(0,0,0) .. (2,1,0)``.
        """
        return SimpleNamespace(
            reference_points=np.array([
                [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [2.0, 1.0, 0.0],
            ]),
        )


def _panel(director, *, scalars=None):
    return GeometrySettingsPanel(
        director, ["displacement"],
        {"nodes": ["u", "v"], "gauss": ["s_xx"]} if scalars is None
        else scalars,
    )


@pytest.fixture
def bound(qapp):
    """A panel already showing the bootstrap geometry."""
    director = _StubDirector()
    panel = _panel(director)
    geom = director.geometries.active
    panel.show_geometry(geom.id)
    return director, panel, geom


# =====================================================================
# Controls exist and are wired to the right lists
# =====================================================================

def test_the_component_combo_lists_scalars_not_vector_prefixes(bound):
    """The Add Diagram dialog's list, NOT the ``displacement`` /
    ``velocity`` prefixes the Deformation section offers."""
    _director, panel, _geom = bound
    items = [
        panel._combo_thr_component.itemData(i)
        for i in range(panel._combo_thr_component.count())
    ]
    assert items == ["u", "v"]
    assert "displacement" not in items


def test_the_topology_combo_offers_both_tables_when_both_have_data(bound):
    _director, panel, _geom = bound
    items = [
        panel._combo_thr_topology.itemData(i)
        for i in range(panel._combo_thr_topology.count())
    ]
    assert items == [TOPOLOGY_NODES, TOPOLOGY_GAUSS]
    assert panel._combo_thr_topology.isEnabled() is True


def test_a_single_topology_is_shown_but_not_selectable(qapp):
    """Nothing to choose, but the user still sees WHICH table is read —
    the same name can exist on both and nothing infers it."""
    panel = _panel(_StubDirector(), scalars={"nodes": ["u"], "gauss": []})
    assert [
        panel._combo_thr_topology.itemData(i)
        for i in range(panel._combo_thr_topology.count())
    ] == [TOPOLOGY_NODES]
    assert panel._combo_thr_topology.isEnabled() is False


def test_the_section_is_disabled_with_a_reason_when_no_scalars_exist(qapp):
    """Same affordance the Deformation section already models."""
    panel = _panel(_StubDirector(), scalars={"nodes": [], "gauss": []})
    assert panel._cb_threshold.isEnabled() is False
    assert panel._sb_thr_min.isEnabled() is False
    assert panel._btn_thr_reset.isEnabled() is False
    assert "No scalar" in panel._cb_threshold.toolTip()


# =====================================================================
# Enabling — seeds a sensible range and lands on the owner
# =====================================================================

def test_enabling_seeds_the_data_range_and_sets_the_threshold(bound):
    director, panel, geom = bound
    panel._cb_threshold.setChecked(True)

    assert panel._sb_thr_min.value() == pytest.approx(-4.0)
    assert panel._sb_thr_max.value() == pytest.approx(6.0)
    assert director.thresholds.settings_for(geom.id) == ThresholdSettings(
        component="u", lo=-4.0, hi=6.0, topology=TOPOLOGY_NODES,
    )
    assert director.fired == [STEP_CHANGED]


def test_the_seed_reads_the_component_at_the_current_step(bound):
    director, panel, _geom = bound
    panel._cb_threshold.setChecked(True)
    assert director.reads[0] == ("u", 3, None, TOPOLOGY_NODES)


def test_disabling_clears_the_threshold_and_repaints(bound):
    director, panel, geom = bound
    panel._cb_threshold.setChecked(True)
    director.fired.clear()

    panel._cb_threshold.setChecked(False)
    assert director.thresholds.settings_for(geom.id) is None
    assert director.fired == [STEP_CHANGED]


def test_editing_the_bounds_re_aims_the_threshold(bound):
    director, panel, geom = bound
    panel._cb_threshold.setChecked(True)
    director.fired.clear()

    panel._sb_thr_min.setValue(1.0)
    panel._sb_thr_max.setValue(2.0)

    assert director.thresholds.settings_for(geom.id) == ThresholdSettings(
        component="u", lo=1.0, hi=2.0, topology=TOPOLOGY_NODES,
    )
    assert director.fired == [STEP_CHANGED, STEP_CHANGED]


def test_bounds_edited_while_disabled_touch_nothing(bound):
    """No threshold means no owner state to write — the spin boxes are
    just a parked range until the user enables the section."""
    director, panel, geom = bound
    panel._sb_thr_min.setValue(1.0)
    assert director.thresholds.settings_for(geom.id) is None
    assert director.fired == []


def test_changing_component_re_seeds_the_range(bound):
    director, panel, geom = bound
    panel._cb_threshold.setChecked(True)
    director.fired.clear()

    panel._combo_thr_component.setCurrentIndex(1)          # -> "v"

    assert director.thresholds.settings_for(geom.id) == ThresholdSettings(
        component="v", lo=0.0, hi=1.0, topology=TOPOLOGY_NODES,
    )
    assert director.fired == [STEP_CHANGED]


def test_changing_topology_repopulates_components_and_re_aims(bound):
    director, panel, geom = bound
    panel._cb_threshold.setChecked(True)
    director.fired.clear()

    panel._combo_thr_topology.setCurrentIndex(1)           # -> gauss

    assert [
        panel._combo_thr_component.itemData(i)
        for i in range(panel._combo_thr_component.count())
    ] == ["s_xx"]
    assert director.thresholds.settings_for(geom.id) == ThresholdSettings(
        component="s_xx", lo=10.0, hi=20.0, topology=TOPOLOGY_GAUSS,
    )
    assert director.fired == [STEP_CHANGED]


def test_reset_widens_a_narrowed_band_back_to_the_data_range(bound):
    director, panel, geom = bound
    panel._cb_threshold.setChecked(True)
    panel._sb_thr_min.setValue(1.0)
    panel._sb_thr_max.setValue(2.0)
    director.fired.clear()

    panel._btn_thr_reset.click()

    assert director.thresholds.settings_for(geom.id) == ThresholdSettings(
        component="u", lo=-4.0, hi=6.0, topology=TOPOLOGY_NODES,
    )
    assert director.fired == [STEP_CHANGED]


def test_a_component_with_no_data_leaves_the_boxes_alone(qapp):
    """An unreadable component must not blank the range to zeros — the
    reader returning None is a "nothing to seed from", not a value."""
    director = _StubDirector()
    panel = _panel(director, scalars={"nodes": ["ghost"], "gauss": []})
    panel.show_geometry(director.geometries.active.id)
    panel._sb_thr_min.setValue(-1.0)
    panel._sb_thr_max.setValue(1.0)

    panel._cb_threshold.setChecked(True)

    assert panel._sb_thr_min.value() == pytest.approx(-1.0)
    assert panel._sb_thr_max.value() == pytest.approx(1.0)
    assert director.thresholds.settings_for(
        director.geometries.active.id,
    ) == ThresholdSettings("ghost", -1.0, 1.0, TOPOLOGY_NODES)


# =====================================================================
# _reflect — per-geometry state, and the re-entrancy guard
# =====================================================================

def test_switching_geometry_reflects_that_geometrys_threshold(bound):
    """Per-geometry state: the panel must show the threshold of the
    geometry it is bound to, and reflecting must not fire anything."""
    director, panel, geom_a = bound
    geom_b = director.geometries.add("B", make_active=False)
    director.thresholds.set_threshold(
        geom_b.id, component="s_xx", lo=11.0, hi=12.0,
        topology=TOPOLOGY_GAUSS,
    )
    director.fired.clear()

    panel.show_geometry(geom_b.id)
    assert panel._cb_threshold.isChecked() is True
    assert panel._combo_thr_component.currentData() == "s_xx"
    assert panel._combo_thr_topology.currentData() == TOPOLOGY_GAUSS
    assert panel._sb_thr_min.value() == pytest.approx(11.0)
    assert panel._sb_thr_max.value() == pytest.approx(12.0)

    panel.show_geometry(geom_a.id)
    assert panel._cb_threshold.isChecked() is False

    # Reflecting is a pure sync — no mutation, no repaint.
    assert director.fired == []
    assert director.thresholds.settings_for(geom_b.id) == ThresholdSettings(
        "s_xx", 11.0, 12.0, TOPOLOGY_GAUSS,
    )
    assert director.thresholds.settings_for(geom_a.id) is None


# =====================================================================
# The Scope section (the spatial scope box)
# =====================================================================
#
# Same idiom as the Threshold section above, with one deliberate
# difference asserted throughout: the scope fires
# ``GEOMETRY_SCOPE_CHANGED``, never ``STEP_CHANGED``. The mask is
# evaluated against reference geometry, so it does not follow the time
# cursor and has no business on the step path.


def _scope_values(panel) -> tuple:
    return (
        tuple(sb.value() for sb in panel._sb_scope_min),
        tuple(sb.value() for sb in panel._sb_scope_max),
    )


def test_the_scope_section_offers_six_bounds_a_fit_and_a_reset(bound):
    _director, panel, _geom = bound
    assert len(panel._sb_scope_min) == 3
    assert len(panel._sb_scope_max) == 3
    assert panel._btn_scope_fit.text() == "Fit to model"
    assert panel._btn_scope_reset.text() == "Reset"
    assert panel._cb_scope.isChecked() is False


def test_enabling_seeds_the_model_bounds_and_sets_the_scope(bound):
    """First enable must never leave the user typing into a blank box."""
    director, panel, geom = bound
    panel._cb_scope.setChecked(True)

    assert _scope_values(panel) == ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))
    box = director.scopes.box_for(geom.id)
    assert box is not None
    assert box.min.tolist() == [0.0, 0.0, 0.0]
    assert box.max.tolist() == [2.0, 1.0, 0.0]
    assert director.fired == [GEOMETRY_SCOPE_CHANGED]


def test_disabling_clears_the_scope_and_repaints(bound):
    director, panel, geom = bound
    panel._cb_scope.setChecked(True)
    director.fired.clear()

    panel._cb_scope.setChecked(False)
    assert director.scopes.box_for(geom.id) is None
    assert director.fired == [GEOMETRY_SCOPE_CHANGED]


def test_editing_a_bound_re_aims_the_scope(bound):
    director, panel, geom = bound
    panel._cb_scope.setChecked(True)
    director.fired.clear()

    panel._sb_scope_min[0].setValue(0.5)
    assert director.scopes.box_for(geom.id).min.tolist() == [0.5, 0.0, 0.0]
    assert director.fired == [GEOMETRY_SCOPE_CHANGED]


def test_scope_bounds_edited_while_disabled_touch_nothing(bound):
    director, panel, geom = bound
    panel._sb_scope_min[0].setValue(0.5)
    assert director.scopes.box_for(geom.id) is None
    assert director.fired == []


def test_an_inverted_box_does_not_crash_and_leaves_the_scope_alone(bound):
    """``BBox`` raises on ``min > max`` and a spin-box pair crosses
    mid-edit all the time, so the panel must catch it rather than let a
    ValueError escape out of a Qt slot. The last GOOD box stays."""
    director, panel, geom = bound
    panel._cb_scope.setChecked(True)
    good = director.scopes.box_for(geom.id)
    director.fired.clear()

    panel._sb_scope_min[0].setValue(5.0)               # min.x > max.x

    assert director.scopes.box_for(geom.id) is good    # untouched
    assert director.fired == []                        # no repaint
    assert "inverted" in panel._cb_scope.toolTip()

    # …and it recovers the moment the pair uncrosses.
    panel._sb_scope_max[0].setValue(6.0)
    assert director.scopes.box_for(geom.id).min.tolist() == [5.0, 0.0, 0.0]
    assert panel._cb_scope.toolTip() == ""


def test_fit_to_model_widens_a_narrowed_box_back_to_the_geometry(bound):
    director, panel, geom = bound
    panel._cb_scope.setChecked(True)
    panel._sb_scope_min[0].setValue(0.5)
    panel._sb_scope_max[0].setValue(1.5)
    director.fired.clear()

    panel._btn_scope_fit.click()

    assert _scope_values(panel) == ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))
    assert director.scopes.box_for(geom.id).max.tolist() == [2.0, 1.0, 0.0]
    assert director.fired == [GEOMETRY_SCOPE_CHANGED]


def test_fit_to_model_while_disabled_only_refills_the_boxes(bound):
    director, panel, geom = bound
    panel._btn_scope_fit.click()

    assert _scope_values(panel) == ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))
    assert director.scopes.box_for(geom.id) is None
    assert director.fired == []


def test_scope_reset_turns_it_off_and_restores_the_model_bounds(bound):
    """The checkbox alone leaves the narrowed numbers behind; reset is
    the one click that undoes both."""
    director, panel, geom = bound
    panel._cb_scope.setChecked(True)
    panel._sb_scope_min[0].setValue(0.5)
    director.fired.clear()

    panel._btn_scope_reset.click()

    assert panel._cb_scope.isChecked() is False
    assert director.scopes.box_for(geom.id) is None
    assert _scope_values(panel) == ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))
    assert director.fired == [GEOMETRY_SCOPE_CHANGED]


def test_switching_geometry_reflects_that_geometrys_scope(bound):
    """Per-geometry state, and reflecting is a pure sync."""
    from apeGmsh.viewers.scene_ir import BBox

    director, panel, geom_a = bound
    geom_b = director.geometries.add("B", make_active=False)
    director.scopes.set_scope(
        geom_b.id, BBox((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    )
    director.fired.clear()

    panel.show_geometry(geom_b.id)
    assert panel._cb_scope.isChecked() is True
    assert _scope_values(panel) == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

    panel.show_geometry(geom_a.id)
    assert panel._cb_scope.isChecked() is False

    assert director.fired == []
    assert director.scopes.box_for(geom_a.id) is None
    assert director.scopes.box_for(geom_b.id) is not None
