"""GeometryManager.set_display — per-geometry mesh / node / opacity state.

Display state moved out of the global SessionPanel and onto each
Geometry so different views (deformed, dimmed-substrate, full-opacity)
can coexist. These tests cover the manager-level setter only — actor
plumbing in ResultsViewer is exercised through the headless smoke
test set.
"""
from __future__ import annotations

import pytest

from apeGmsh.viewers.diagrams._geometries import Geometry, GeometryManager


@pytest.fixture
def mgr() -> GeometryManager:
    return GeometryManager()


def test_geometry_defaults_match_historical_globals(mgr: GeometryManager):
    """A bootstrap geometry must default to the previous global look:
    mesh + nodes visible, full opacity. Anything else would silently
    change the default substrate appearance for users without a saved
    session.
    """
    g = mgr.active
    assert g is not None
    assert g.show_mesh is True
    assert g.show_nodes is True
    assert g.display_opacity == pytest.approx(1.0)


def test_set_display_updates_named_fields(mgr: GeometryManager):
    g = mgr.active
    assert g is not None
    changed = mgr.set_display(
        g.id, show_mesh=False, show_nodes=False, display_opacity=0.25,
    )
    assert changed is True
    assert g.show_mesh is False
    assert g.show_nodes is False
    assert g.display_opacity == pytest.approx(0.25)


def test_set_display_partial_update_leaves_others(mgr: GeometryManager):
    """Pass-only-what-changes contract — None means leave alone."""
    g = mgr.active
    mgr.set_display(g.id, display_opacity=0.5)
    assert g.show_mesh is True   # untouched
    assert g.show_nodes is True
    assert g.display_opacity == pytest.approx(0.5)


def test_set_display_clamps_opacity_to_unit_interval(mgr: GeometryManager):
    g = mgr.active
    mgr.set_display(g.id, display_opacity=2.5)
    assert g.display_opacity == 1.0
    mgr.set_display(g.id, display_opacity=-0.3)
    assert g.display_opacity == 0.0


def test_set_display_returns_false_when_no_change(mgr: GeometryManager):
    g = mgr.active
    # Setting the existing value is a no-op — observers shouldn't fire.
    fired = []
    mgr.subscribe(lambda: fired.append(1))
    changed = mgr.set_display(g.id, show_mesh=True, display_opacity=1.0)
    assert changed is False
    assert fired == []


def test_set_display_fires_observer_on_change(mgr: GeometryManager):
    g = mgr.active
    fired = []
    mgr.subscribe(lambda: fired.append(1))
    mgr.set_display(g.id, display_opacity=0.7)
    assert fired == [1]


def test_set_display_unknown_geom_returns_false(mgr: GeometryManager):
    assert mgr.set_display("not-a-real-id", show_mesh=False) is False


def test_duplicate_copies_display_state(mgr: GeometryManager):
    """Cloning a geometry should preserve the display state alongside
    the deformation tuple — the user expects a duplicate to look
    identical until they edit it.
    """
    src = mgr.active
    mgr.set_display(
        src.id, show_mesh=False, show_nodes=False, display_opacity=0.6,
    )
    new = mgr.duplicate(src.id)
    assert new is not None
    assert new.show_mesh is False
    assert new.show_nodes is False
    assert new.display_opacity == pytest.approx(0.6)
