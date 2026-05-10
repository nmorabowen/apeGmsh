"""
Base class for bridge namespaces.

Each per-family namespace class (``_UniaxialMaterialNS``,
``_SectionNS``, …) lives in its own module under :mod:`._internal.ns`
so that parallel Phase 1+ slice agents can extend them without
fighting over a single shared file.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...apesees import apeSees


__all__ = ["_BridgeNamespace"]


class _BridgeNamespace:
    """Base for bridge namespaces.

    Each namespace holds a back-reference to its owning bridge so
    that namespace methods can call ``self._bridge._register(...)``
    when they construct a typed primitive.
    """

    __slots__ = ("_bridge",)

    def __init__(self, bridge: "apeSees") -> None:
        self._bridge = bridge
