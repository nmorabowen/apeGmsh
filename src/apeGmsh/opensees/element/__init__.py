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

from .beam_column import (
    ElasticTimoshenkoBeam,
    LadrunoDispBeamColumn,
    LadrunoIMKBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from .embedded_rebar import embedded_rebar_args
from .shell import (
    ASDShellQ4,
    ASDShellT3,
    ShellDKGQ,
    ShellMITC3,
    ShellMITC4,
)
from .solid import (
    BezierBBarPlaneStressWarning,
    BezierTet10,
    BezierTri6,
    FourNodeQuad,
    FourNodeTetrahedron,
    LadrunoBrick,
    LadrunoCST,
    LadrunoQuad,
    LadrunoUP,
    SixNodeTri,
    TenNodeTetrahedron,
    Tri31,
    stdBrick,
)
from .truss import CorotTruss, InertiaTruss, Truss
from .two_node_link import TwoNodeLink
from .zero_length import (
    CoupledZeroLength,
    ZeroLength,
    ZeroLengthMatDir,
    ZeroLengthSection,
)


__all__ = [
    # beam_column
    "elasticBeamColumn",
    "forceBeamColumn",
    "dispBeamColumn",
    "ElasticTimoshenkoBeam",
    "LadrunoDispBeamColumn",
    "LadrunoIMKBeam",
    # truss
    "Truss",
    "CorotTruss",
    "InertiaTruss",
    # zero_length
    "ZeroLength",
    "ZeroLengthMatDir",
    "ZeroLengthSection",
    "CoupledZeroLength",
    # two_node_link
    "TwoNodeLink",
    # shell
    "ShellMITC3",
    "ShellMITC4",
    "ShellDKGQ",
    "ASDShellQ4",
    "ASDShellT3",
    # solid
    "FourNodeTetrahedron",
    "TenNodeTetrahedron",
    "stdBrick",
    "FourNodeQuad",
    "Tri31",
    "SixNodeTri",
    "BezierTri6",
    "BezierTet10",
    "BezierBBarPlaneStressWarning",
    "LadrunoBrick",
    "LadrunoQuad",
    "LadrunoCST",
    "LadrunoUP",
    # embedded reinforcement (Ladruno fork — resolver-produced coupling)
    "embedded_rebar_args",
]
