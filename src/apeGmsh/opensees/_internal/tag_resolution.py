"""
Cross-primitive tag resolution at ``_emit`` time.

OpenSees commands frequently take **other primitives' tags** as
positional arguments (a Fiber section's ``patch`` references a
material's tag; an element references its section's and transform's
tags). Phase 0 stores tags externally on the bridge
(:class:`apeGmsh.opensees.apesees.apeSees`) — primitive instances do
not carry their own tag. So composite primitives' ``_emit`` methods
need a way to look up dependency tags at emit time without breaking
the frozen :class:`~apeGmsh.opensees.emitter.base.Emitter` Protocol.

The contract — opt-in, attribute-based
======================================

The bridge attaches a callable resolver to the emitter via
:func:`set_tag_resolver` before driving emit. Composite primitives
call :func:`resolve_tag` to look up dependency tags. The Protocol is
unchanged — emitters that don't drive composite primitives never see
the resolver.

This is the seam Phase 4 emitters (Tcl, py, live) and the build
pipeline plug into. Each emitter ignores the attribute; the bridge's
build flow installs the resolver before calling ``BuiltModel.emit``.

Tests that exercise composite ``_emit`` directly (without driving the
full bridge) install a manual resolver via :func:`set_tag_resolver`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .types import Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "ATTR_TAG_RESOLVER",
    "TagResolver",
    "resolve_tag",
    "set_tag_resolver",
]


#: Name of the private attribute the bridge attaches to an emitter.
ATTR_TAG_RESOLVER = "_tag_for_primitive"


#: Maps a Primitive to its bridge-allocated tag.
TagResolver = Callable[[Primitive], int]


def set_tag_resolver(emitter: object, resolver: TagResolver) -> None:
    """Attach ``resolver`` to ``emitter`` so composite primitives can
    look up dependency tags during ``_emit``.

    Idempotent: calling twice replaces the resolver.
    """
    setattr(emitter, ATTR_TAG_RESOLVER, resolver)


def resolve_tag(emitter: "Emitter", primitive: Primitive) -> int:
    """Return the allocated tag for ``primitive``, using the resolver
    attached to ``emitter``.

    Raises
    ------
    RuntimeError
        If no resolver is attached. Tests and downstream code that
        drive a composite primitive's ``_emit`` directly must call
        :func:`set_tag_resolver` first.
    """
    resolver: TagResolver | None = getattr(emitter, ATTR_TAG_RESOLVER, None)
    if resolver is None:
        raise RuntimeError(
            "Composite primitive ``_emit`` requires a tag resolver "
            "attached to the emitter. Call "
            "``apeGmsh.opensees._internal.tag_resolution.set_tag_resolver"
            "(emitter, resolver)`` before driving emission."
        )
    tag: int = resolver(primitive)
    return tag
