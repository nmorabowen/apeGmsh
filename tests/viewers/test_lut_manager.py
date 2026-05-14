"""Unit tests for :class:`LUTManager` and :class:`LUT` (plan 06 step 1).

Tests the shared lookup-table registry in isolation — no viewer
construction, no VTK, just QObject signals + state. Mirrors
``test_active_objects.py`` shape.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtCore")

from apeGmsh.viewers.core._lut_manager import (
    LUT,
    LUTManager,
    PRESETS,
    is_preset,
    sample_preset,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def manager(qapp):
    return LUTManager()


def _collect(signal):
    """Subscribe a list-appender to ``signal``; return the list."""
    bucket: list = []
    signal.connect(lambda: bucket.append(None))
    return bucket


# =====================================================================
# Presets module
# =====================================================================


def test_presets_includes_canonical_set():
    # 10 curated presets per the plan doc.
    assert len(PRESETS) == 10
    assert "viridis" in PRESETS
    assert "coolwarm" in PRESETS
    assert "jet" in PRESETS


def test_is_preset_recognises_known_names():
    assert is_preset("viridis")
    assert is_preset("jet")
    assert not is_preset("definitely_not_a_cmap")


def test_sample_preset_returns_rgba_array():
    samples = sample_preset("viridis", n=16)
    assert samples.shape == (16, 4)
    # All values in [0, 1].
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_sample_preset_unknown_falls_back_to_viridis():
    fallback = sample_preset("not_a_real_cmap", n=8)
    viridis = sample_preset("viridis", n=8)
    # Fallback should not raise and should return something usable.
    assert fallback.shape == (8, 4)
    # Specifically, fallback samples viridis (the documented fallback).
    assert fallback == pytest.approx(viridis)


# =====================================================================
# LUT — initial state
# =====================================================================


def test_lut_initial_state(qapp):
    lut = LUT("stress_vm")
    assert lut.array_name == "stress_vm"
    assert lut.preset == "viridis"
    assert lut.vmin == 0.0
    assert lut.vmax == 1.0
    assert lut.range == (0.0, 1.0)
    assert lut.log_scale is False
    assert lut.show_scalar_bar is True


def test_lut_collapsing_zero_width_range(qapp):
    """vmin == vmax must expand to a unit-width range so downstream
    LUT builders don't divide by zero."""
    lut = LUT("u", vmin=5.0, vmax=5.0)
    assert lut.vmin == 5.0
    assert lut.vmax == 6.0


def test_lut_unknown_initial_preset_falls_back_to_viridis(qapp):
    lut = LUT("u", preset="not_a_real_cmap")
    assert lut.preset == "viridis"


# =====================================================================
# LUT — setters fire .changed once per actual change
# =====================================================================


def test_set_preset_emits_once(qapp):
    lut = LUT("u")
    emitted = _collect(lut.changed)
    lut.set_preset("plasma")
    assert lut.preset == "plasma"
    assert emitted == [None]


def test_set_preset_idempotent_no_emit(qapp):
    lut = LUT("u", preset="plasma")
    emitted = _collect(lut.changed)
    lut.set_preset("plasma")
    assert emitted == []


def test_set_preset_unknown_clamped_to_viridis(qapp):
    lut = LUT("u", preset="plasma")
    emitted = _collect(lut.changed)
    lut.set_preset("definitely_not_a_cmap")
    # Clamped to viridis — this *is* a change from plasma, so one emit.
    assert lut.preset == "viridis"
    assert emitted == [None]


def test_set_range_emits_once(qapp):
    lut = LUT("u")
    emitted = _collect(lut.changed)
    lut.set_range(-1.0, 2.0)
    assert lut.range == (-1.0, 2.0)
    assert emitted == [None]


def test_set_range_idempotent_no_emit(qapp):
    lut = LUT("u", vmin=-1.0, vmax=2.0)
    emitted = _collect(lut.changed)
    lut.set_range(-1.0, 2.0)
    assert emitted == []


def test_set_range_collapses_zero_width(qapp):
    lut = LUT("u")
    lut.set_range(3.0, 3.0)
    assert lut.range == (3.0, 4.0)


def test_set_log_scale_emits_once(qapp):
    lut = LUT("u")
    emitted = _collect(lut.changed)
    lut.set_log_scale(True)
    assert lut.log_scale is True
    assert emitted == [None]
    # Re-setting to same value: no emit.
    lut.set_log_scale(True)
    assert emitted == [None]


def test_set_show_scalar_bar_emits_once(qapp):
    lut = LUT("u")
    emitted = _collect(lut.changed)
    lut.set_show_scalar_bar(False)
    assert lut.show_scalar_bar is False
    assert emitted == [None]


# =====================================================================
# LUT — derived state
# =====================================================================


def test_color_stops_returns_n_entries(qapp):
    lut = LUT("u", preset="coolwarm")
    stops = lut.color_stops(n=5)
    assert len(stops) == 5
    # ts are monotonically increasing from 0 to 1.
    ts = [t for t, _ in stops]
    assert ts[0] == pytest.approx(0.0)
    assert ts[-1] == pytest.approx(1.0)
    assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))
    # rgb tuples are 3-tuples in [0, 1].
    for _, rgb in stops:
        assert len(rgb) == 3
        for c in rgb:
            assert 0.0 <= c <= 1.0


def test_to_pyvista_lookup_table_matches_state(qapp):
    pv = pytest.importorskip("pyvista")
    lut = LUT("u", preset="plasma", vmin=10.0, vmax=42.0)
    table = lut.to_pyvista_lookup_table()
    assert isinstance(table, pv.LookupTable)
    # scalar_range round-trips.
    sr = table.scalar_range
    assert sr[0] == pytest.approx(10.0)
    assert sr[1] == pytest.approx(42.0)


# =====================================================================
# LUTManager — registry behaviour
# =====================================================================


def test_manager_get_or_create_returns_same_instance(manager):
    a = manager.get_or_create("stress_vm")
    b = manager.get_or_create("stress_vm")
    assert a is b


def test_manager_different_names_return_different_instances(manager):
    a = manager.get_or_create("stress_vm")
    b = manager.get_or_create("strain_xx")
    assert a is not b
    assert a.array_name == "stress_vm"
    assert b.array_name == "strain_xx"


def test_manager_defaults_only_applied_on_first_create(manager):
    """Calling get_or_create a second time with different defaults
    returns the existing LUT unchanged — callers must mutate via the
    LUT's own setters to change state."""
    first = manager.get_or_create("u", preset="viridis", vmin=0.0, vmax=1.0)
    second = manager.get_or_create("u", preset="plasma", vmin=99.0, vmax=100.0)
    assert second is first
    assert first.preset == "viridis"
    assert first.vmin == 0.0


def test_manager_get_returns_none_for_missing(manager):
    assert manager.get("never_created") is None


def test_manager_contains_and_len(manager):
    assert len(manager) == 0
    assert "u" not in manager
    manager.get_or_create("u")
    manager.get_or_create("v")
    assert len(manager) == 2
    assert "u" in manager
    assert "v" in manager


def test_manager_all_preserves_insertion_order(manager):
    manager.get_or_create("z")
    manager.get_or_create("a")
    manager.get_or_create("m")
    names = [lut.array_name for lut in manager.all()]
    assert names == ["z", "a", "m"]


def test_manager_remove_is_idempotent(manager):
    manager.get_or_create("u")
    assert "u" in manager
    manager.remove("u")
    assert "u" not in manager
    # Calling again on a missing name doesn't raise.
    manager.remove("u")
    manager.remove("never_existed")
