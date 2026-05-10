"""
TagAllocator — sequential, per-kind, 1-based tag allocation.

The bridge owns this. Primitives never see it directly; the bridge
calls :meth:`allocate` (or :meth:`allocate_for`) at registration time
and stashes the result.

OpenSees tags are scoped per command kind: a uniaxialMaterial with
tag 1, a section with tag 1, and an element with tag 1 do not
collide. The allocator therefore keeps an independent counter per
kind string.

Idempotency: :meth:`allocate_for` returns the same tag if the same
primitive instance is registered twice (lookup keyed on ``id()``).
This protects against accidental double-registration of a primitive
the user constructed standalone (P11) and then passed through the
namespace API.
"""
from __future__ import annotations


class TagAllocator:
    """Per-kind sequential 1-based tag allocator."""

    __slots__ = ("_counters", "_assignments")

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        # id(primitive) -> assigned tag. The kind context is implicit:
        # a primitive is tagged exactly once, in exactly one kind.
        self._assignments: dict[int, int] = {}

    def allocate(self, kind: str) -> int:
        """Return the next 1-based tag for ``kind`` and bump the counter."""
        n = self._counters.get(kind, 0) + 1
        self._counters[kind] = n
        return n

    def allocate_for(self, primitive: object, kind: str) -> int:
        """Allocate a tag for a primitive, idempotent on repeat calls.

        If ``primitive`` has already been allocated, return the same
        tag and do not bump the counter. This keeps standalone-then-
        registered primitives (P11) safe against accidental
        double-registration.
        """
        prev = self._assignments.get(id(primitive))
        if prev is not None:
            return prev
        tag = self.allocate(kind)
        self._assignments[id(primitive)] = tag
        return tag

    def tag_for(self, primitive: object) -> int | None:
        """Return the previously allocated tag for ``primitive``, or
        ``None`` if it has not been allocated yet."""
        return self._assignments.get(id(primitive))

    def reset(self) -> None:
        """Clear all counters and assignments — fresh allocator state."""
        self._counters.clear()
        self._assignments.clear()
