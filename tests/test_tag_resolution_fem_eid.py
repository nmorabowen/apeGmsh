"""Unit tests for the Phase 8.6 FEM-element-id side channel.

Covers the two new helpers in
:mod:`apeGmsh.opensees._internal.tag_resolution`:

* :func:`set_current_fem_element_id` setter.
* :func:`current_fem_element_id` getter with sentinel fallback.

These mirror the existing ``set_element_nodes`` /
``current_element_nodes`` pair, but with a sentinel (``-1``) instead
of raising when no context has been set — see scope-doc §3 Choice 2.
"""
from __future__ import annotations

from typing import Any

from apeGmsh.opensees._internal.tag_resolution import (
    ATTR_CURRENT_FEM_ELEMENT_ID,
    MISSING_FEM_ELEMENT_ID,
    current_fem_element_id,
    set_current_fem_element_id,
)


class _DummyEmitter:
    """Bare attribute-bag emitter — mirrors the runtime shape of
    `H5Emitter` / `RecordingEmitter` for the attribute-based side
    channel."""


def test_missing_sentinel_is_minus_one() -> None:
    """Documented sentinel value (matches FEM-IDs-are-always-positive)."""
    assert MISSING_FEM_ELEMENT_ID == -1


def test_get_without_set_returns_sentinel() -> None:
    """A fresh emitter has no attr → getter returns the sentinel."""
    e = _DummyEmitter()
    assert current_fem_element_id(e) == MISSING_FEM_ELEMENT_ID


def test_set_then_get_round_trips() -> None:
    e = _DummyEmitter()
    set_current_fem_element_id(e, 42)
    assert current_fem_element_id(e) == 42


def test_set_coerces_to_int() -> None:
    """Non-int inputs (numpy scalars, strings of ints) coerce via int()."""
    e = _DummyEmitter()
    set_current_fem_element_id(e, "17")
    assert current_fem_element_id(e) == 17


def test_set_overrides_previous_value() -> None:
    """Side channel is idempotent — second set wins."""
    e = _DummyEmitter()
    set_current_fem_element_id(e, 5)
    set_current_fem_element_id(e, 10)
    assert current_fem_element_id(e) == 10


def test_attribute_name_matches_constant() -> None:
    """Setter / getter both go through ``ATTR_CURRENT_FEM_ELEMENT_ID``."""
    e = _DummyEmitter()
    set_current_fem_element_id(e, 99)
    raw: Any = getattr(e, ATTR_CURRENT_FEM_ELEMENT_ID)
    assert int(raw) == 99


def test_two_emitters_have_independent_state() -> None:
    """Side channel is per-instance — no shared module-level state."""
    e1 = _DummyEmitter()
    e2 = _DummyEmitter()
    set_current_fem_element_id(e1, 10)
    set_current_fem_element_id(e2, 20)
    assert current_fem_element_id(e1) == 10
    assert current_fem_element_id(e2) == 20
