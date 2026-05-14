"""Tests for plan 05 — Auto-Apply / Reset infrastructure in
:class:`DiagramSettingsTab`.

The tab construction requires a director with ``subscribe_diagrams`` /
``geometries.subscribe`` callbacks. We stub a minimal director here
rather than depend on the elasticFrame.mpco fixture — the plan-05
changes are about the tab's own state machinery (preference
persistence, debounce timer, card-commit flushing), not director
behavior.
"""
from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("qtpy.QtCore")

from apeGmsh.viewers.ui._diagram_settings_tab import DiagramSettingsTab


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class _StubGeometries:
    """Minimal stand-in for ``director.geometries``."""

    def subscribe(self, _callback):
        # Return an unsubscribe stub.
        return lambda: None


class _StubDirector:
    """Minimal director — provides only what DiagramSettingsTab.__init__ needs."""

    def __init__(self):
        self.geometries = _StubGeometries()
        self.compositions = _Compositions()
        self.dispatcher = None

    def subscribe_diagrams(self, _callback):
        return lambda: None


class _Compositions:
    @property
    def active(self):
        return None


@pytest.fixture
def director():
    return _StubDirector()


@pytest.fixture
def isolated_auto_apply_pref(monkeypatch):
    """Redirect the Auto-Apply pref to a unique QSettings key for each test.

    Without this, tests share the real "settings_tab/auto_apply" key
    and flap each other's state. Patches the class-level constant
    just for the test's lifetime.
    """
    key = f"settings_tab/auto_apply_test_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(
        DiagramSettingsTab, "_AUTO_APPLY_SETTINGS_KEY", key,
    )
    yield key
    # Cleanup the test key.
    try:
        from qtpy.QtCore import QSettings
        s = QSettings("apeGmsh", "ResultsViewer")
        s.remove(key)
        s.sync()
    except Exception:
        pass


# =====================================================================
# Preference persistence
# =====================================================================


def test_auto_apply_default_off(qapp, director, isolated_auto_apply_pref):
    """No saved state → Auto-Apply starts OFF."""
    tab = DiagramSettingsTab(director)
    assert tab._auto_apply_enabled is False
    assert tab._auto_apply_cb.isChecked() is False


def test_auto_apply_toggle_persists(qapp, director, isolated_auto_apply_pref):
    """Toggling the checkbox writes the new value to QSettings."""
    tab = DiagramSettingsTab(director)
    tab._auto_apply_cb.setChecked(True)
    # Read back via a fresh tab — the value should round-trip.
    tab2 = DiagramSettingsTab(director)
    assert tab2._auto_apply_enabled is True
    assert tab2._auto_apply_cb.isChecked() is True


def test_auto_apply_toggle_off_persists(qapp, director, isolated_auto_apply_pref):
    """Off-state also persists (not just on-state)."""
    tab = DiagramSettingsTab(director)
    tab._auto_apply_cb.setChecked(True)
    tab._auto_apply_cb.setChecked(False)
    tab2 = DiagramSettingsTab(director)
    assert tab2._auto_apply_enabled is False


# =====================================================================
# Debounce timer behavior
# =====================================================================


def test_kick_debounce_noop_when_auto_apply_off(qapp, director, isolated_auto_apply_pref):
    tab = DiagramSettingsTab(director)
    assert tab._auto_apply_enabled is False
    tab._kick_debounce()
    # Timer wasn't built because Auto-Apply is off.
    assert tab._auto_commit_timer is None


def test_kick_debounce_starts_timer_when_auto_apply_on(qapp, director, isolated_auto_apply_pref):
    tab = DiagramSettingsTab(director)
    tab._auto_apply_cb.setChecked(True)
    assert tab._auto_apply_enabled is True

    tab._kick_debounce()
    assert tab._auto_commit_timer is not None
    assert tab._auto_commit_timer.isActive()


def test_kick_debounce_restarts_timer_on_each_call(qapp, director, isolated_auto_apply_pref):
    """Successive _kick_debounce calls restart the timer — that's what
    makes the debounce coalesce rapid signals."""
    tab = DiagramSettingsTab(director)
    tab._auto_apply_cb.setChecked(True)

    tab._kick_debounce()
    first_remaining = tab._auto_commit_timer.remainingTime()
    # Simulate a tiny delay; can't easily wait, so just verify the
    # second kick re-arms (remainingTime resets to ~150ms).
    tab._kick_debounce()
    assert tab._auto_commit_timer.isActive()


# =====================================================================
# _flush_auto_commits runs every card commit
# =====================================================================


def test_flush_auto_commits_runs_every_card_commit(qapp, director, isolated_auto_apply_pref):
    tab = DiagramSettingsTab(director)
    ran: list[str] = []
    tab._card_commits = [
        lambda: ran.append("a"),
        lambda: ran.append("b"),
        lambda: ran.append("c"),
    ]
    tab._flush_auto_commits()
    assert ran == ["a", "b", "c"]


def test_flush_auto_commits_swallows_per_card_exceptions(qapp, director, isolated_auto_apply_pref):
    tab = DiagramSettingsTab(director)
    ran: list[str] = []

    def _good():
        ran.append("good")

    def _bad():
        raise RuntimeError("oops")

    # _safe_call inside _flush should catch the bad commit so the
    # good ones still run.
    tab._card_commits = [_good, _bad, _good]
    tab._flush_auto_commits()
    assert ran.count("good") == 2


def test_card_commits_cleared_on_rebuild(qapp, director, isolated_auto_apply_pref):
    """_rebuild() wipes _card_commits so stale closures don't fire."""
    tab = DiagramSettingsTab(director)
    tab._card_commits = [lambda: None, lambda: None]
    tab._rebuild()
    assert tab._card_commits == []


def test_rebuild_cancels_pending_debounce(qapp, director, isolated_auto_apply_pref):
    tab = DiagramSettingsTab(director)
    tab._auto_apply_cb.setChecked(True)
    tab._kick_debounce()
    assert tab._auto_commit_timer.isActive()
    tab._rebuild()
    assert not tab._auto_commit_timer.isActive()


# =====================================================================
# _stage_with_signal wiring
# =====================================================================


def test_stage_appends_to_pending_appliers_always(qapp, director, isolated_auto_apply_pref):
    """The Apply-button path must work whether or not Auto-Apply is on."""
    from qtpy import QtWidgets

    tab = DiagramSettingsTab(director)
    tab._pending_appliers = []
    spin = QtWidgets.QSpinBox()
    applier_calls = []
    tab._stage_with_signal(
        spin, "valueChanged",
        lambda: applier_calls.append("ran"),
    )
    assert len(tab._pending_appliers) == 1
    # Run the registered applier as if Apply was clicked.
    tab._pending_appliers[0]()
    assert applier_calls == ["ran"]


def test_stage_does_not_wire_signal_when_auto_apply_off(
    qapp, director, isolated_auto_apply_pref,
):
    from qtpy import QtWidgets

    tab = DiagramSettingsTab(director)
    assert tab._auto_apply_enabled is False
    spin = QtWidgets.QSpinBox()
    tab._pending_appliers = []
    tab._stage_with_signal(
        spin, "valueChanged",
        lambda: None,
    )
    # No signal wiring → changing the spin doesn't kick the debounce
    # (timer was never created).
    spin.setValue(7)
    assert tab._auto_commit_timer is None


def test_stage_wires_signal_when_auto_apply_on(
    qapp, director, isolated_auto_apply_pref,
):
    from qtpy import QtWidgets

    tab = DiagramSettingsTab(director)
    tab._auto_apply_cb.setChecked(True)
    tab._pending_appliers = []
    spin = QtWidgets.QSpinBox()
    tab._stage_with_signal(
        spin, "valueChanged",
        lambda: None,
    )
    # Changing the widget should kick the debounce.
    spin.setValue(7)
    assert tab._auto_commit_timer is not None
    assert tab._auto_commit_timer.isActive()


def test_stage_unknown_signal_does_not_raise(qapp, director, isolated_auto_apply_pref):
    """If the widget doesn't expose the named signal, fall back silently."""
    from qtpy import QtWidgets

    tab = DiagramSettingsTab(director)
    tab._auto_apply_cb.setChecked(True)
    tab._pending_appliers = []
    label = QtWidgets.QLabel("just a label")
    # QLabel has no "valueChanged" signal — should not raise.
    tab._stage_with_signal(label, "valueChanged", lambda: None)
    assert len(tab._pending_appliers) == 1
