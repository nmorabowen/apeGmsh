"""
Compatibility re-export of the cross-primitive tag resolver.

The interim resolver was promoted to
:mod:`apeGmsh.opensees._internal.tag_resolution` at the end of Phase 1
so that elements (Phase 2), patterns (Phase 3), and any other
composite primitives that look up dependency tags share one helper.
This module preserves the original Phase 1C surface for tests and
docstrings that referenced ``section._tag_resolver``.

New code should import from :mod:`apeGmsh.opensees._internal.tag_resolution`
directly.
"""
from __future__ import annotations

from .._internal.tag_resolution import (
    ATTR_TAG_RESOLVER,
    TagResolver,
    resolve_tag,
    set_tag_resolver,
)

#: Phase 1C alias for the generic :func:`resolve_tag`. Kept for
#: backwards compat with test files written against the original
#: section-flavored name.
resolve_mat_tag = resolve_tag


__all__ = [
    "ATTR_TAG_RESOLVER",
    "TagResolver",
    "resolve_mat_tag",
    "resolve_tag",
    "set_tag_resolver",
]
