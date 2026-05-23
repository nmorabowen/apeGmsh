"""PR5 — cross-surface integration: outline ↔ tab UI sync.

The model deduplication tests in
``test_overlay_visibility_model.py`` prove the pure-Python model's
contract.  These tests prove the SURFACES wire up correctly: a
write on the outline tree updates the tab checkbox visual state
and vice-versa.

This is the test that closes audit debt D2: the documented
"alternating writes flip the overlay" oscillation
(``_mesh_outline_tree.py:96-104``).  Pre-PR5 each surface kept its
own snapshot off Qt widget state and they could drift.  Post-PR5
both surfaces read from and write to the shared model, and each
subscribes to model changes to refresh its own UI.

Headless Qt via ``QT_QPA_PLATFORM=offscreen``; no VTK / plotter,
so no GPU dependency.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# =====================================================================
# LoadsTabPanel ↔ OverlayVisibilityModel
# =====================================================================


def test_loads_tab_checkbox_syncs_when_outline_writes(qapp):
    """Outline writes to model → tab's checkbox state updates.
    The bug pre-PR5: writing from one surface didn't update the other."""
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    from apeGmsh.viewers.ui.loads_tab import LoadsTabPanel
    from apeGmsh._kernel.defs.loads import PointLoadDef

    model = OverlayVisibilityModel()

    # Minimal LoadsComposite stub: enough surface for the tab to
    # build rows.  Real composite would have many more methods, but
    # the tab only iterates patterns() + load_defs.
    composite = SimpleNamespace(
        patterns=lambda: ["dead", "live"],
        load_defs=[
            PointLoadDef(target="A", pattern="dead",
                         force_xyz=(0.0, 0.0, -1.0)),
            PointLoadDef(target="B", pattern="live",
                         force_xyz=(0.0, 0.0, -2.0)),
        ],
        defs_in_pattern=lambda name: [
            d for d in composite.load_defs if d.pattern == name
        ],
    )

    tab = LoadsTabPanel(composite, overlay_model=model)
    # Initial state: model is empty → all checkboxes start unchecked
    # after the first model-driven sync (note: the tab's own
    # constructor sets initial state to checked, but our model-driven
    # sync fires on the first model write).
    assert "dead" in tab._pattern_items
    assert "live" in tab._pattern_items

    # Simulate an outline write: set load_patterns = {dead}.
    model.set_load_patterns({"dead"})

    # Tab's UI mirrors the model — "dead" is checked, "live" is not.
    from qtpy.QtCore import Qt
    assert tab._pattern_items["dead"].checkState(0) == Qt.CheckState.Checked
    assert tab._pattern_items["live"].checkState(0) == Qt.CheckState.Unchecked


def test_loads_tab_checkbox_sync_does_not_feedback(qapp):
    """The sync uses ``_suppress_signal`` so the programmatic
    ``setCheckState`` does NOT re-fire the ``on_patterns_changed``
    callback (which would re-write to the model and cycle, even
    though the model's idempotent setter would short-circuit it)."""
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    from apeGmsh.viewers.ui.loads_tab import LoadsTabPanel

    model = OverlayVisibilityModel()
    composite = SimpleNamespace(
        patterns=lambda: ["dead"],
        load_defs=[],
        defs_in_pattern=lambda name: [],
    )
    callbacks: list = []
    LoadsTabPanel(
        composite,
        on_patterns_changed=callbacks.append,
        overlay_model=model,
    )

    callbacks.clear()
    model.set_load_patterns({"dead"})

    # The model-driven sync flipped the checkbox programmatically,
    # but _suppress_signal prevented on_patterns_changed from firing.
    # (If the suppression were absent, this would have at least one
    # entry from the round-trip.)
    assert callbacks == []


# =====================================================================
# MassTabPanel ↔ OverlayVisibilityModel
# =====================================================================


def test_mass_tab_checkbox_syncs_when_outline_writes(qapp):
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    from apeGmsh.viewers.ui.mass_tab import MassTabPanel
    from apeGmsh._kernel.defs.masses import PointMassDef

    model = OverlayVisibilityModel()
    composite = SimpleNamespace(
        mass_defs=[PointMassDef(target="A", mass=10.0)],
    )

    tab = MassTabPanel(composite, overlay_model=model)
    assert tab._show_cb.isChecked() is False

    model.set_mass_visible(True)

    assert tab._show_cb.isChecked() is True


def test_mass_tab_sync_does_not_feedback(qapp):
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    from apeGmsh.viewers.ui.mass_tab import MassTabPanel

    model = OverlayVisibilityModel()
    composite = SimpleNamespace(mass_defs=[])
    callbacks: list = []
    MassTabPanel(
        composite,
        on_overlay_changed=callbacks.append,
        overlay_model=model,
    )

    callbacks.clear()
    model.set_mass_visible(True)

    # blockSignals on the checkbox prevented the toggled callback.
    assert callbacks == []


# =====================================================================
# Idempotency at the integration layer
# =====================================================================


def test_model_idempotency_breaks_oscillation_loop(qapp):
    """The headline correctness test for D2.

    Simulates the exact pre-PR5 oscillation pattern:
    * Surface A (outline) writes pattern set P1
    * Surface B (tab) responds by computing its own snapshot and
      writing back

    Pre-PR5 (no shared state): the snapshots could differ and each
    write triggered a real rebuild — the overlay flipped between
    the two states on every alternation.

    Post-PR5: both surfaces share the model.  Surface B's response
    write is the SAME value the model already holds (idempotent
    no-op).  The rebuild fires exactly ONCE per real state change.
    """
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel

    model = OverlayVisibilityModel()
    rebuild_calls: list[frozenset] = []
    model.subscribe(lambda: rebuild_calls.append(model.load_patterns))

    # Surface A writes {dead}.
    model.set_load_patterns({"dead"})
    # Surface B's response — would compute its own snapshot and write
    # the same value back through the model.
    model.set_load_patterns({"dead"})    # idempotent no-op
    # Surface A's response to B's response.
    model.set_load_patterns({"dead"})    # idempotent no-op

    # Despite three surface writes, the rebuild fired exactly once.
    assert rebuild_calls == [frozenset({"dead"})]
