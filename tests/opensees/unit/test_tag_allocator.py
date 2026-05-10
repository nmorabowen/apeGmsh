"""Unit tests for the per-kind sequential ``TagAllocator``."""
from __future__ import annotations

from apeGmsh.opensees._internal.tag_allocator import TagAllocator


def test_allocate_starts_at_one() -> None:
    a = TagAllocator()
    assert a.allocate("uniaxialMaterial") == 1


def test_allocate_is_sequential_within_a_kind() -> None:
    a = TagAllocator()
    assert a.allocate("uniaxialMaterial") == 1
    assert a.allocate("uniaxialMaterial") == 2
    assert a.allocate("uniaxialMaterial") == 3


def test_allocate_kinds_are_independent() -> None:
    a = TagAllocator()
    assert a.allocate("uniaxialMaterial") == 1
    assert a.allocate("section") == 1  # independent counter
    assert a.allocate("element") == 1
    assert a.allocate("uniaxialMaterial") == 2
    assert a.allocate("section") == 2


def test_reset_clears_counters() -> None:
    a = TagAllocator()
    a.allocate("uniaxialMaterial")
    a.allocate("uniaxialMaterial")
    a.reset()
    assert a.allocate("uniaxialMaterial") == 1


def test_allocate_for_is_idempotent_on_same_object() -> None:
    a = TagAllocator()
    obj = object()
    t1 = a.allocate_for(obj, "uniaxialMaterial")
    t2 = a.allocate_for(obj, "uniaxialMaterial")
    assert t1 == t2 == 1


def test_allocate_for_distinct_objects_get_distinct_tags() -> None:
    a = TagAllocator()
    o1, o2 = object(), object()
    assert a.allocate_for(o1, "uniaxialMaterial") == 1
    assert a.allocate_for(o2, "uniaxialMaterial") == 2


def test_tag_for_returns_none_before_allocation() -> None:
    a = TagAllocator()
    assert a.tag_for(object()) is None


def test_tag_for_returns_assigned_tag() -> None:
    a = TagAllocator()
    obj = object()
    a.allocate_for(obj, "uniaxialMaterial")
    assert a.tag_for(obj) == 1


def test_reset_clears_assignments() -> None:
    a = TagAllocator()
    obj = object()
    a.allocate_for(obj, "uniaxialMaterial")
    a.reset()
    assert a.tag_for(obj) is None
