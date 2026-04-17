"""Canonical type aliases shared across the entire apeGmsh package."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

Tag = int
DimTag = tuple[int, int]
TagsLike = Tag | list[Tag] | DimTag | list[DimTag]

# Flexible entity reference — accepted wherever the library
# expects entity tags.  Resolves through :func:`resolve_to_tags`.
EntityRef = int | str | tuple[int, int]
EntityRefs = EntityRef | list[EntityRef] | None


@runtime_checkable
class SessionProtocol(Protocol):
    """Minimal contract that composites expect from their parent session.

    Every composite stores ``self._parent`` typed as this protocol.  The
    concrete class is ``_SessionBase`` (or its subclasses ``apeGmsh``
    and ``Part``), but composites never need to import those.
    """

    name: str
    _verbose: bool
    _auto_pg_from_label: bool

    @property
    def is_active(self) -> bool: ...
