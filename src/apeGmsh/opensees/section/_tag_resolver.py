"""
Material-tag resolution at section ``_emit`` time.

OPEN QUESTION (Phase 1C → coordinator before Phase 2)
=====================================================

The ``Emitter`` Protocol (frozen after Phase 0, ADR 0008) takes
material tags **positionally**: ``emitter.patch("rect", mat_tag, ...)``.
But Phase 0 chose to keep tag allocation external to primitives —
``Primitive._emit(emitter, tag)`` only receives **its own** tag, not a
mapping of dependency tags. ``BuiltModel.emit()`` (in :mod:`apesees`)
walks ``self.primitives`` and resolves each tag from ``self.tag_for``,
but does **not** pass that map down to ``_emit``.

This is fine for leaf primitives (Steel02, ElasticMembranePlateSection)
that have no dependencies. Composers that *reference other primitives*
in their emitted command — Fiber sections referencing materials,
LayeredShell sections referencing nDMaterials, elements referencing
sections + transforms — need a resolver.

This module provides the **interim Phase 1C contract**: the resolver
is attached to the emitter as a private attribute ``_tag_for_primitive``
(callable: ``Primitive -> int``). The bridge's build pipeline sets
this before emit; sections look it up here. This is a closure-capture
pattern — the Protocol does not name the attribute, so it does not
break the frozen surface (P8 still holds: a new emit target only
needs to honor the Protocol; the resolver attribute is opt-in for
emitters that drive composite primitives).

If the resolver is not attached (e.g. a unit test that constructs a
section in isolation and calls ``_emit`` directly), this module
falls back to a callable kwarg-style override — see
:func:`resolve_mat_tag`.

The Phase 4 emitters (Tcl, py, live) and the bridge's build flow
will harmonize this: either the Emitter Protocol grows a
``register_resolver(...)`` method (an architecture event), or the
bridge installs the resolver attribute on every emitter it drives.
The coordinator owns this decision before Phase 2 lands (elements
reference sections + transforms — the same problem at the next
level of composition).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .._internal.types import Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "ATTR_TAG_RESOLVER",
    "TagResolver",
    "resolve_mat_tag",
    "set_tag_resolver",
]


#: Name of the private attribute the bridge attaches to an emitter to
#: install a tag resolver. Sections look here at ``_emit`` time.
ATTR_TAG_RESOLVER = "_tag_for_primitive"


#: Type alias for a tag resolver: maps a Primitive to its allocated tag.
TagResolver = Callable[[Primitive], int]


def set_tag_resolver(emitter: object, resolver: TagResolver) -> None:
    """Attach ``resolver`` to ``emitter`` so sections can look up
    dependency tags during ``_emit``.

    Idempotent: calling twice replaces the resolver.
    """
    setattr(emitter, ATTR_TAG_RESOLVER, resolver)


def resolve_mat_tag(emitter: "Emitter", material: Primitive) -> int:
    """Return the allocated tag for ``material``, using the resolver
    attached to ``emitter`` (if any), else raising.

    Raises
    ------
    RuntimeError
        If no resolver is attached to ``emitter``. Tests and
        downstream code that need to drive a composite section's
        ``_emit`` directly must call :func:`set_tag_resolver` first.
    """
    resolver: TagResolver | None = getattr(emitter, ATTR_TAG_RESOLVER, None)
    if resolver is None:
        raise RuntimeError(
            "Composite section ``_emit`` requires a tag resolver "
            "attached to the emitter. Call "
            "``apeGmsh.opensees.section._tag_resolver.set_tag_resolver"
            "(emitter, resolver)`` before driving emission. See "
            "the module docstring for the open coordinator question."
        )
    tag: int = resolver(material)
    return tag
