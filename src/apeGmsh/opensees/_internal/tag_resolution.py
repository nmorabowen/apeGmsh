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
    "ATTR_CURRENT_FEM_ELEMENT_ID",
    "ATTR_ELEMENT_NODES",
    "ATTR_PHANTOM_NODE_TAGS",
    "ATTR_TAG_RESOLVER",
    "MISSING_FEM_ELEMENT_ID",
    "TagResolver",
    "current_element_nodes",
    "current_fem_element_id",
    "damp_args",
    "is_phantom_node",
    "resolve_tag",
    "set_current_fem_element_id",
    "set_element_nodes",
    "set_phantom_node_tags",
    "set_tag_resolver",
]


#: Name of the private attribute the bridge attaches to an emitter.
ATTR_TAG_RESOLVER = "_tag_for_primitive"

#: Name of the private attribute the bridge sets on an emitter just
#: before an :class:`Element` primitive's ``_emit`` to pass the node
#: tags for the current element of a fan-out.
ATTR_ELEMENT_NODES = "_current_element_nodes"

#: Name of the private attribute the bridge sets on an emitter just
#: before an :class:`Element` primitive's ``_emit`` to pass the FEM
#: element ID (``fem.elements`` ``ids`` value) the current OpenSees
#: tag was fanned out from.  Phase 8.6: lets the H5 emitter record
#: the (fem_eid, ops_tag) mapping for round-trip lookup.
ATTR_CURRENT_FEM_ELEMENT_ID = "_current_fem_element_id"

#: Sentinel returned by :func:`current_fem_element_id` when no FEM
#: element ID has been set on the emitter — e.g. tests that drive an
#: :class:`Element` primitive's ``_emit`` directly without the bridge
#: fan-out.  FEM element IDs are always positive 1-based ints, so
#: ``-1`` is unambiguous.
MISSING_FEM_ELEMENT_ID: int = -1

#: Name of the private attribute the bridge sets on an emitter holding
#: the complete set of broker-synthetic phantom-node tags.  S2
#: introduced per-node ``-ndf K`` emission for real broker nodes
#: (shell-on-solid), so the H5 emitter can no longer use
#: "``ndf is not None``" as the phantom-vs-broker discriminator.
#: The MP-constraint emit pass populates this set ONCE (before any
#: node emission), and the H5 emitter consults it per-call to decide
#: whether to record the tag in
#: ``/opensees/constraints/phantom_node_tags`` (per ADR 0022 INV-3
#: and ADR 0033).  Phantom tags are guaranteed disjoint from real
#: broker tags (the resolver allocates ``> max(broker_node_tag)``),
#: so the predicate is unambiguous and order-independent.
ATTR_PHANTOM_NODE_TAGS = "_phantom_node_tags_predicate"


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


def damp_args(
    emitter: "Emitter", damp: "Primitive | None",
) -> list[int | str]:
    """Trailing ``-damp $tag`` element flag (ADR 0053, D3b).

    Returns ``["-damp", <dampTag>]`` when an allow-listed element carries
    a :class:`~apeGmsh.opensees._internal.types.Damping` object, else ``[]``.
    The object's tag is resolved against the emitter (it is declared as a
    dependency so it already has one). Only the elements whose OpenSees
    class implements ``setDamping`` accept ``damp=``; this helper just
    renders the flag once the field is present.
    """
    if damp is None:
        return []
    return ["-damp", resolve_tag(emitter, damp)]


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


def set_phantom_node_tags(
    emitter: object, tags: "set[int] | frozenset[int]",
) -> None:
    """Attach the complete set of broker-synthetic phantom-node tags
    to ``emitter`` so :func:`is_phantom_node` can classify per-call.

    The MP-constraint emit pass calls this ONCE at entry with the
    union of all phantom tags from :class:`NodeToSurfaceRecord`
    rows.  The H5 emitter then consults :func:`is_phantom_node` in
    :meth:`H5Emitter.node` to decide whether the call should land in
    the ``phantom_node_tags`` array; other emitters ignore the
    attribute.

    Per S2 (ADR 0033) the per-node ``-ndf K`` token is now legal on
    real broker nodes (shell-on-solid mixed-ndf models), so the
    H5 emitter can't infer "phantom" from ``ndf is not None``
    alone — the bridge must surface the phantom-vs-real distinction
    explicitly.  This is stateless: the set is computed once from
    ``fem.nodes.constraints.node_to_surfaces()`` and never mutated
    during emit.  Phantom tags are disjoint from real broker tags
    (the resolver allocates ``> max(broker_node_tag)``), so a
    pre-populated set correctly classifies every subsequent
    ``emitter.node(...)`` call without ordering constraints.

    Idempotent — calling twice replaces the set.
    """
    setattr(emitter, ATTR_PHANTOM_NODE_TAGS, frozenset(int(t) for t in tags))


def is_phantom_node(emitter: object, tag: int) -> bool:
    """Return ``True`` when ``tag`` is a broker-synthetic phantom node
    on this emitter.  Defaults to ``False`` when no phantom-tag set
    has been attached (e.g. tests that drive an emitter directly
    without the bridge fan-out, or models without MP constraints).

    See :func:`set_phantom_node_tags`.
    """
    tags = getattr(emitter, ATTR_PHANTOM_NODE_TAGS, None)
    if tags is None:
        return False
    return int(tag) in tags


def set_current_fem_element_id(emitter: object, fem_eid: int) -> None:
    """Set the FEM element ID for the current element of a fan-out.

    The bridge sets this in :mod:`apeGmsh.opensees._internal.build`
    just before driving an :class:`Element` primitive's ``_emit`` so
    the H5Emitter (Phase 8.6) can record the (fem_eid, ops_tag)
    pair without breaking the frozen Emitter Protocol.

    Idempotent.
    """
    setattr(emitter, ATTR_CURRENT_FEM_ELEMENT_ID, int(fem_eid))


def current_fem_element_id(emitter: "Emitter") -> int:
    """Return the FEM element ID currently being emitted, or the
    :data:`MISSING_FEM_ELEMENT_ID` sentinel (``-1``) when no context
    is set.

    Unlike :func:`current_element_nodes`, this never raises — the
    sentinel is the documented "no FEM context" signal so tests that
    drive ``.element(...)`` directly without the bridge fan-out can
    still produce a valid record.
    """
    eid: int | None = getattr(emitter, ATTR_CURRENT_FEM_ELEMENT_ID, None)
    if eid is None:
        return MISSING_FEM_ELEMENT_ID
    return int(eid)
