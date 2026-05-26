"""Unit tests for ``ComposeRecord`` + ``ComposeSet`` (Phase 3A.1).

ADR 0038 §"Implementation pointer" mirrors the
``PartitionRecord`` / ``PartitionSet`` pattern at
``src/apeGmsh/_kernel/records/_partitions.py`` for compose
provenance.  These tests cover the dataclass contract and the set's
read-only composite behaviour, independent of any H5 round-trip
(those live in ``test_compose_schema_2_9_0``).
"""
from __future__ import annotations

import dataclasses

import pytest

from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh._kernel.record_sets import ComposeSet


def _make_record(label: str = "module_a") -> ComposeRecord:
    return ComposeRecord(
        label=label,
        source_path=f"{label}.h5",
        source_fem_hash="deadbeefcafef00d",
        source_neutral_schema_version="2.9.0",
        translate=(1.0, 2.0, 3.0),
        rotate=(0.0, 0.0, 0.0, 1.0),
        partition_rank=2,
        composed_at="2026-05-26T12:00:00Z",
        properties={"author": "test", "build_id": 7},
    )


def test_compose_record_frozen() -> None:
    """Mutating a ``ComposeRecord`` attribute raises ``FrozenInstanceError``."""
    rec = _make_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.label = "other"  # type: ignore[misc]


def test_compose_record_defaults() -> None:
    """Optional fields default per ADR 0038."""
    rec = ComposeRecord(
        label="m",
        source_path="m.h5",
        source_fem_hash="hash",
        source_neutral_schema_version="2.9.0",
        translate=(0.0, 0.0, 0.0),
    )
    assert rec.rotate is None
    assert rec.partition_rank is None
    assert rec.composed_at == ""
    assert rec.properties == {}


def test_compose_set_empty() -> None:
    """Default ``ComposeSet`` is empty + falsy."""
    s = ComposeSet(())
    assert len(s) == 0
    assert not bool(s)
    assert list(s) == []
    assert s.ids == []
    assert repr(s) == "ComposeSet(empty)"


def test_compose_set_iter_sorted_by_label() -> None:
    """Iteration is deterministic — ascending label order."""
    rec_b = _make_record("b")
    rec_a = _make_record("a")
    rec_c = _make_record("c")
    s = ComposeSet((rec_b, rec_a, rec_c))
    assert s.ids == ["a", "b", "c"]
    assert [r.label for r in s] == ["a", "b", "c"]


def test_compose_set_getitem_and_contains() -> None:
    rec = _make_record("module_a")
    s = ComposeSet((rec,))
    assert "module_a" in s
    assert "missing" not in s
    assert s["module_a"] is rec
    with pytest.raises(KeyError):
        _ = s["missing"]


def test_compose_set_equality() -> None:
    """Two sets with the same records compare equal field-by-field."""
    r1 = _make_record("a")
    r2 = _make_record("a")
    assert ComposeSet((r1,)) == ComposeSet((r2,))
    # Different label → not equal.
    assert ComposeSet((r1,)) != ComposeSet((_make_record("b"),))


def test_compose_set_accepts_dict_input() -> None:
    rec = _make_record("a")
    s = ComposeSet({"a": rec})
    assert "a" in s
    assert s["a"] is rec
