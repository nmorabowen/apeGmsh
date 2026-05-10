"""
``apeGmsh.opensees.element`` — typed primitives for OpenSees
``element <Type>`` commands.

The module is split by element family. Phase 2 ships:

* :mod:`.truss`       — :class:`Truss`, :class:`CorotTruss`,
                        :class:`InertiaTruss`
* :mod:`.zero_length` — :class:`ZeroLength`,
                        :class:`ZeroLengthSection`

Constructing these classes outside a bridge is supported (P11). The
typed methods that auto-register them with an
:class:`apeGmsh.opensees.apesees.apeSees` bridge live on
:class:`apeGmsh.opensees._internal.ns.element._ElementNS` and are
exposed via ``ops.element.<Type>(...)``.

Element fan-out contract
========================

OpenSees elements take two (or more) node tags positionally. The
typed Python class does not carry a node-tag tuple — instead the
bridge fans the element spec across its physical group at build time
and sets the per-element node tags on the emitter via
:func:`apeGmsh.opensees._internal.tag_resolution.set_element_nodes`
just before invoking each ``_emit``. The typed class reads those
tags via :func:`current_element_nodes` and resolves any composed
material / section tags via :func:`resolve_tag` from the same
module. The frozen :class:`~apeGmsh.opensees.emitter.base.Emitter`
Protocol is unchanged — the contract is opt-in attribute storage on
the emitter.
"""
from __future__ import annotations

from .truss import CorotTruss, InertiaTruss, Truss
from .zero_length import ZeroLength, ZeroLengthMatDir, ZeroLengthSection


__all__ = [
    # truss
    "Truss",
    "CorotTruss",
    "InertiaTruss",
    # zero_length
    "ZeroLength",
    "ZeroLengthMatDir",
    "ZeroLengthSection",
]
