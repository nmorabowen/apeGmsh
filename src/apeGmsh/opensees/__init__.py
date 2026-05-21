"""
apeGmsh.opensees — Pythonic, statically-typed wrapper around OpenSees.

Public API: :class:`apeSees`. See ``architecture/README.md`` for the
design charter, the 14 principles, and the per-phase implementation
roadmap.

Example:

.. code-block:: python

    from apeGmsh.opensees import apeSees

    fem = g.mesh.queries.get_fem_data(dim=1)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    ...
    ops.tcl("frame.tcl")
    ops.run()

This is the sole OpenSees surface. The legacy in-session
``g.opensees.*`` composite and the ``apeGmsh.solvers`` package were
removed in the Phase-8 teardown (ADR 0009 — no back-compat shim).
"""
from __future__ import annotations

from .apesees import apeSees
from .model_data import ModelData
from .node import Node, NodeSet
from .opensees_model import OpenSeesModel
from ._orientation import AlongBeam, Cartesian, Cylindrical, Spherical

__all__ = [
    "apeSees",
    "ModelData",
    "OpenSeesModel",
    "Node",
    "NodeSet",
    "AlongBeam",
    "Cartesian",
    "Cylindrical",
    "Spherical",
]
