"""DensityManager — set / toggle / subscribe / persist."""
from __future__ import annotations

import pytest

from apeGmsh.viewers.ui.density import (
    DENSITIES,
    DENSITY_COMPACT,
    DENSITY_COMFORTABLE,
    DensityManager,
)


def test_default_is_comfortable():
    m = DensityManager()
    assert m.current is DENSITY_COMFORTABLE


def test_set_density_changes_current():
    m = DensityManager()
    m.set_density("compact")
    assert m.current is DENSITY_COMPACT


def test_set_density_unknown_raises():
    m = DensityManager()
    with pytest.raises(ValueError):
        m.set_density("super-compact")


def test_set_density_no_op_if_unchanged():
    m = DensityManager()
    seen = []
    m.subscribe(lambda d: seen.append(d))
    m.set_density(m.current.name)
    assert seen == []


def test_toggle_flips_between_compact_and_comfortable():
    m = DensityManager()
    start = m.current
    m.toggle()
    assert m.current is not start
    m.toggle()
    assert m.current is start


def test_subscribe_fires_on_change():
    m = DensityManager()
    seen = []
    m.subscribe(lambda d: seen.append(d.name))
    m.toggle()
    assert len(seen) == 1


def test_unsubscribe_stops_callbacks():
    m = DensityManager()
    seen = []
    unsub = m.subscribe(lambda d: seen.append(d))
    unsub()
    m.toggle()
    assert seen == []


def test_observer_exception_does_not_break_others():
    m = DensityManager()
    seen = []
    m.subscribe(lambda _d: (_ for _ in ()).throw(RuntimeError("boom")))
    m.subscribe(lambda d: seen.append(d.name))
    m.toggle()
    assert len(seen) == 1


def test_density_tokens_have_expected_fields():
    for d in (DENSITY_COMPACT, DENSITY_COMFORTABLE):
        assert d.name in DENSITIES
        assert d.row_h > 0
        assert d.fs_body > 0
