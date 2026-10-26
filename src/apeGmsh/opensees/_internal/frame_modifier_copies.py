"""Register the modified section copies a frame spec's ``modifiers=`` needs.

A ``forceBeamColumn`` / ``dispBeamColumn`` with a non-identity
:class:`~apeGmsh.opensees.element.frame_modifiers.FrameModifiers` references
a modified copy of its ``beamIntegration`` and of each ``ElasticSection`` in
it (``element/frame_modifiers.py``). When the bridge registers such a spec,
this module binds the spec to its copies, taken from the bridge's own memo
(``apeSees._frame_modifier_copies``), and registers them as synthesised,
unnamed declarations (key ``frame_modifiers:<pg>/<kind><k>``), as
``ops.spring_bed`` registers its unit materials. Both the namespace verbs and
``ops.register(spec)`` (P11) come through here.

The memo is per bridge: specs of one session that share a section and a
modifier set reference one copy, and two sessions never share one.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..element.beam_column import dispBeamColumn, forceBeamColumn
from ..element.frame_modifiers import ModifiedCopies, bind_integration, copies_of
from .types import Primitive

if TYPE_CHECKING:
    from ..apesees import apeSees


__all__ = ["ModifiedCopies", "register_frame_modifier_copies"]


def register_frame_modifier_copies(ops: "apeSees", prim: Primitive) -> None:
    """Bind ``prim`` to its modified copies and register them on ``ops``.

    A no-op for every primitive but a ``forceBeamColumn`` /
    ``dispBeamColumn`` whose modifiers need a copy. Idempotent: a copy
    already registered on ``ops`` is skipped.
    """
    if not isinstance(prim, (forceBeamColumn, dispBeamColumn)):
        return
    bound = bind_integration(prim, ops._frame_modifier_copies)
    if bound is None:
        return
    from ..apesees import _kind_of

    for derived in copies_of(bound):
        if ops._tags.tag_for(derived) is not None:
            continue
        kind = _kind_of(derived)
        k = 1
        while True:
            key = f"frame_modifiers:{prim.pg}/{kind}{k}"
            if (f"opensees/{kind}/{key}" not in ops._decl_sites
                    and not ops._provenance.has("opensees", kind, key)):
                break
            k += 1
        ops._register(derived, synthesised=key)
