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
    "ATTR_ELEMENT_NODES",
    "ATTR_TAG_RESOLVER",
    "TagResolver",
    "current_element_nodes",
    "resolve_tag",
    "set_element_nodes",
    "set_tag_resolver",
]


#: Name of the private attribute the bridge attaches to an emitter.
ATTR_TAG_RESOLVER = "_tag_for_primitive"

#: Name of the private attribute the bridge sets on an emitter just
#: before an :class:`Element` primitive's ``_emit`` to pass the node
#: tags for the current element of a fan-out.
ATTR_ELEMENT_NODES = "_current_element_nodes"


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


def set_element_nodes(
    emitter: object,
    node_tags: tuple[int, ...],
) -> None:
    """Set the node tags for the current element of an element fan-out.

    The bridge sets this just before driving an :class:`Element`
    primitive's ``_emit`` so the typed class can read the node tags
    via :func:`current_element_nodes` without breaking the frozen
    Emitter Protocol.

    Idempotent.
    """
    setattr(emitter, ATTR_ELEMENT_NODES, node_tags)


def current_element_nodes(emitter: "Emitter") -> tuple[int, ...]:
    """Return the node tags for the element currently being emitted.

    Used inside :class:`Element` typed primitives' ``_emit``. The
    bridge fans out the element's physical group at build time and
    sets one set of node tags per call.

    Raises
    ------
    RuntimeError
        If no element-nodes context has been set. Tests that exercise
        an element's ``_emit`` directly install the context via
        :func:`set_element_nodes`.
    """
    nodes: tuple[int, ...] | None = getattr(emitter, ATTR_ELEMENT_NODES, None)
    if nodes is None:
        raise RuntimeError(
            "Element ``_emit`` requires the bridge to set element-"
            "nodes context first. Call "
            "``apeGmsh.opensees._internal.tag_resolution.set_element_nodes"
            "(emitter, node_tags)`` before driving emission."
        )
    return nodes
